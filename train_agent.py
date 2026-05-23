import os
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv
from utils import make_stationary, purged_walk_forward_splits, compute_rolling_metrics

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def make_env(df, config, feature_cols, mean=None, std=None, random_start=True, episode_max_steps=None):
    return ForexTradingEnv(
        df=df,
        window_size=config['data']['window_size'],
        feature_columns=feature_cols,
        spread_pips=config['environment']['spread_pips'],
        max_slippage_pips=config['environment']['max_slippage_pips'],
        risk_per_trade=config['environment']['risk_per_trade'],
        leverage=config['environment']['leverage'],
        commission_percent=config['environment']['commission_percent'],
        use_atr_sl=config['environment']['use_atr_sl'],
        atr_multiplier_sl=config['environment']['atr_multiplier_sl'],
        atr_multiplier_tp=config['environment']['atr_multiplier_tp'],
        partial_close_enabled=config['environment']['partial_close_enabled'],
        partial_close_ratio=config['environment']['partial_close_ratio'],
        tp_levels=config['environment']['tp_levels'],
        fixed_sl_percent=config['environment']['fixed_sl_percent'],
        fixed_tp_percents=config['environment']['fixed_tp_percents'],
        open_penalty_pips=config['environment']['reward']['open_penalty_pips'],
        time_penalty_pips=config['environment']['reward']['time_penalty_pips'],
        slope_div_reward_scale=config['environment']['reward']['slope_div_reward_scale'],
        reward_scale=config['environment']['reward']['reward_scale'],
        open_bonus_pips=0.0,
        random_start=random_start,
        min_episode_steps=config['training']['min_episode_steps'],
        episode_max_steps=episode_max_steps or config['training']['episode_max_steps'],
        feature_mean=mean,
        feature_std=std,
        pip_value=config['data']['pip_value']
    )

def evaluate_model(model, eval_env, deterministic=True):
    obs = eval_env.reset()
    lstm_states = None
    episode_starts = np.ones((eval_env.num_envs,), dtype=bool)
    equity_curve = []
    while True:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=deterministic)
        obs, _, dones, infos = eval_env.step(action)
        episode_starts = dones
        equity_curve.append(float(infos[0]["equity_usd"]))
        if dones[0]:
            break
    return equity_curve, equity_curve[-1]

def main():
    config = load_config()
    
    # Загрузка данных
    df_full, feature_cols = load_and_preprocess_data(config['data']['path'])
    
    # Проверка стационарности и дифференцирование признаков
    print("Проверка стационарности признаков...")
    df_full = make_stationary(df_full, feature_cols, threshold=0.05)
    
    # Purged Walk-Forward сплиты
    splits = purged_walk_forward_splits(df_full, config['training']['val_years'], config['training']['purge_days'])
    print(f"Создано {len(splits)} окон Walk-Forward")
    
    all_results = []
    
    for split in splits:
        print(f"\n===== Окно: {split['name']} =====")
        train_df = df_full.loc[split['train_idx']]
        val_df = df_full.loc[split['val_idx']]
        test_df = df_full.loc[split['test_idx']]
        
        # Нормализация на обучающем окне
        train_features = train_df[feature_cols].values.astype(np.float32)
        train_mean = np.mean(train_features, axis=0)
        train_std = np.std(train_features, axis=0)
        
        # Создание сред
        def make_train_env():
            return make_env(train_df, config, feature_cols, train_mean, train_std,
                           random_start=True, episode_max_steps=config['training']['episode_max_steps'])
        
        def make_eval_env(df, mean, std):
            return make_env(df, config, feature_cols, mean, std,
                           random_start=False, episode_max_steps=None)
        
        train_vec_env = SubprocVecEnv([lambda i=i: make_train_env() for i in range(config['training']['num_envs'])])
        train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        
        val_env = DummyVecEnv([lambda: make_eval_env(val_df, train_mean, train_std)])
        val_env = VecNormalize(val_env, norm_obs=True, norm_reward=False, training=False)
        
        # Callback для расчёта метрик на валидации
        class MetricsEvalCallback(EvalCallback):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.best_sharpe = -np.inf
                
            def _on_step(self):
                ret = super()._on_step()
                if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                    equity_curve, _ = evaluate_model(self.model, self.eval_env, deterministic=True)
                    metrics = compute_rolling_metrics(equity_curve)
                    sharpe = metrics['sharpe']
                    if sharpe > self.best_sharpe:
                        self.best_sharpe = sharpe
                        self.model.save(f"best_sharpe_{self.eval_env.get_attr('window_size')[0]}")
                    print(f"Val Metrics: Sharpe={sharpe:.3f}, MaxDD={metrics['max_dd']*100:.2f}%, Calmar={metrics['calmar']:.3f}")
                return ret
        
        policy_kwargs = dict(
            net_arch=dict(shared=config['training']['policy']['net_arch_shared'],
                          pi=config['training']['policy']['net_arch_pi'],
                          vf=config['training']['policy']['net_arch_vf']),
            lstm_hidden_size=config['training']['policy']['lstm_hidden_size'],
            n_lstm_layers=config['training']['policy']['n_lstm_layers'],
            activation_fn=nn.Tanh
        )
        
        model = RecurrentPPO(
            policy="MlpLstmPolicy", env=train_vec_env, verbose=1,
            tensorboard_log=f"./logs/{split['name']}",
            policy_kwargs=policy_kwargs,
            ent_coef=float(config['training']['ent_coef']),
            learning_rate=float(config['training']['learning_rate']),
            batch_size=config['training']['batch_size'],
            n_epochs=config['training']['n_epochs'],
            gamma=config['training']['gamma'],
            gae_lambda=config['training']['gae_lambda'],
            clip_range=config['training']['clip_range'],
            vf_coef=config['training']['vf_coef'],
            max_grad_norm=config['training']['max_grad_norm']
        )
        
        os.makedirs(f"./checkpoints/{split['name']}", exist_ok=True)
        checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path=f"./checkpoints/{split['name']}")
        eval_callback = MetricsEvalCallback(val_env, best_model_save_path=f"./best_{split['name']}",
                                            log_path=f"./eval_log_{split['name']}", eval_freq=10_000,
                                            deterministic=True, render=False)
        
        model.learn(total_timesteps=config['training']['total_timesteps'], callback=[checkpoint_callback, eval_callback])
        model.save(f"model_{split['name']}")
        train_vec_env.save(f"vec_normalize_{split['name']}.pkl")
        
        # Тест на внутреннем окне
        test_env = DummyVecEnv([lambda: make_eval_env(test_df, train_mean, train_std)])
        test_env = VecNormalize.load(f"vec_normalize_{split['name']}.pkl", test_env)
        test_env.training = False
        test_env.norm_reward = False
        
        test_curve, test_final = evaluate_model(model, test_env)
        metrics = compute_rolling_metrics(test_curve)
        print(f"Inner test {split['name']}: final equity = {test_final:.2f}, Sharpe = {metrics['sharpe']:.3f}, MaxDD = {metrics['max_dd']*100:.2f}%")
        all_results.append({'window': split['name'], **metrics})
    
    with open("walkforward_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n===== Финальное обучение на всех данных до 2024 =====")
    final_train_df = df_full[df_full.index < pd.Timestamp("2024-01-01")]
    final_train_features = final_train_df[feature_cols].values.astype(np.float32)
    final_mean = np.mean(final_train_features, axis=0)
    final_std = np.std(final_train_features, axis=0)
    
    def make_final_train_env():
        return make_env(final_train_df, config, feature_cols, final_mean, final_std,
                       random_start=True, episode_max_steps=config['training']['episode_max_steps'])
    
    final_train_vec_env = SubprocVecEnv([lambda i=i: make_final_train_env() for i in range(config['training']['num_envs'])])
    final_train_vec_env = VecNormalize(final_train_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    
    final_model = RecurrentPPO(
        policy="MlpLstmPolicy", env=final_train_vec_env, verbose=1,
        tensorboard_log="./logs/final", policy_kwargs=policy_kwargs,
        ent_coef=config['training']['ent_coef'],
        learning_rate=config['training']['learning_rate'],
        n_steps=config['training']['n_steps'],
        batch_size=config['training']['batch_size'],
        n_epochs=config['training']['n_epochs'],
        gamma=config['training']['gamma'],
        gae_lambda=config['training']['gae_lambda'],
        clip_range=config['training']['clip_range'],
        vf_coef=config['training']['vf_coef'],
        max_grad_norm=config['training']['max_grad_norm']
    )
    final_model.learn(total_timesteps=config['training']['total_timesteps'] * 2)  # финальное обучение дольше
    final_model.save("model_sber_final")
    final_train_vec_env.save("vec_normalize_final.pkl")
    
    # Сохраняем метаданные
    with open("final_metadata.json", "w") as f:
        json.dump({
            'mean': final_mean.tolist(),
            'std': final_std.tolist(),
            'feature_cols': feature_cols,
            'train_start': str(final_train_df.index.min()),
            'train_end': str(final_train_df.index.max()),
            'config': config
        }, f)
    
    print("Обучение завершено.")

if __name__ == "__main__":
    main()