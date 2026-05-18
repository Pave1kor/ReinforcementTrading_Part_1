# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv


def evaluate_model(model: RecurrentPPO, eval_env: DummyVecEnv, deterministic: bool = True):
    obs = eval_env.reset()
    equity_curve = []
    # Инициализируем скрытые состояния памяти LSTM нулями (агент ничего не помнит на старте)
    lstm_states = None
    # Маска начала эпизода (True означает, что это первый шаг)
    episode_starts = np.ones((eval_env.num_envs,), dtype=bool)

    while True:
        # Передаем в predict состояния памяти и маску старта
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states, 
            episode_start=episode_starts, 
            deterministic=True
        )

        step_out = eval_env.step(action)

        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, rewards, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])

        # После первого шага агент уже находится внутри эпизода (маска становится False)
        episode_starts = np.zeros((eval_env.num_envs,), dtype=bool)

        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        # use equity from info (state *before* DummyVecEnv reset)
        eq = info.get("equity_usd", eval_env.get_attr("equity_usd")[0])
        if isinstance(eq, (list, np.ndarray)):
           eq = eq[0]
        equity_curve.append(eq)

        if done:
            break

    final_equity = float(equity_curve[-1])
    return equity_curve, final_equity



def main():
    #file_path = "data/EURUSD_15 Mins_Ask_2020.12.06_2025.12.12.csv"
    file_path = "data/test_EURUSD_Candlestick_1_Hour_BID_20.02.2023-22.02.2025.csv"
    df, feature_cols = load_and_preprocess_data(file_path)

    # Time split 80/20
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print("Training bars:", len(train_df))
    print("Testing bars :", len(test_df))

    # ---- Env factories ----
    SL_OPTS = [30, 60, 90, 120]
    TP_OPTS = [30, 60, 90, 120]
    WIN = 30

    # Train env: random starts to reduce memorization
    def make_train_env():
        return ForexTradingEnv(
            df=train_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=False,
            min_episode_steps=1000,
            episode_max_steps=None,
            feature_columns=feature_cols,
            hold_reward_weight=0.0,#0.05
            open_penalty_pips=0.5,      # 0.5 half a pip per open
            time_penalty_pips=0.02,     # 0.02 pips per bar in trade
            unrealized_delta_weight=0.0
        )

    # Train-eval env: deterministic start, NO random starts (so curve is stable/reproducible)
    def make_train_eval_env():
        return ForexTradingEnv(
            df=train_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=False,
            episode_max_steps=None,
            feature_columns=feature_cols,
            hold_reward_weight=0.00,
            open_penalty_pips=0.5,      # half a pip per open
            time_penalty_pips=0.02,     # 0.02 pips per bar in trade
            unrealized_delta_weight=0.0
        )

    # Test-eval env: deterministic
    def make_test_eval_env():
        return ForexTradingEnv(
            df=test_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=False,
            episode_max_steps=None,
            feature_columns=feature_cols,
            hold_reward_weight=0.00,
            open_penalty_pips=0.5,      # half a pip per open
            time_penalty_pips=0.02,     # 0.02 pips per bar in trade
            unrealized_delta_weight=0.0
        )

    train_vec_env = DummyVecEnv([make_train_env])
    train_eval_env = DummyVecEnv([make_train_eval_env])
    test_eval_env = DummyVecEnv([make_test_eval_env])

    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]), # Базовые слои до LSTM
        lstm_hidden_size=128,                        # Размер памяти LSTM-ячейки
        n_lstm_layers=1                              # Количество слоев LSTM
    )


    # ---- Model ----
    model = RecurrentPPO(
        policy="MlpLstmPolicy",       # Специальная политика с памятью
        env=train_vec_env,
        verbose=1,
        tensorboard_log="./tensorboard_log/",
        policy_kwargs=policy_kwargs, # Добавляем увеличенную архитектуру сети
        ent_coef=0.02,
        learning_rate=1.5e-4,       # Стандартный шаг, можно снизить до 1e-4, если график обучения будет рваным
        n_steps= 4096,             # Сколько шагов собирает один поток перед обновлением весов
        batch_size=256,            # Размер мини-батча для градиентного спуска (64 или 128)
        n_epochs=10,              # Количество эпох оптимизации на один батч данных
        gamma=0.99,               # Коэффициент дисконтирования (0.99 означает фокус на долгосрочную прибыль)
        gae_lambda=0.95,          # Параметр для Generalized Advantage Estimation
        clip_range=0.15,           # Ограничение изменения политики (чтобы веса не ломались резкими скачками)
        vf_coef=0.7               # Вес ошибки функции ценности (Value Function)
    )

    # ---- Checkpoints ----
    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=ckpt_dir,
        name_prefix="RecurrentPPO_eurusd"
    )

    # ---- Train ----
    total_timesteps = 600000
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # ---- Select best model by OOS final equity ----
    equity_curve_test_last, final_equity_train_last = evaluate_model(model, train_eval_env)
    print(f"[IS Eval] Last model final equity: {final_equity_train_last:.2f}")

    best_train_equity = final_equity_train_last
    best_path = None
    best_model = model

    ckpts = sorted(
        [f for f in os.listdir(ckpt_dir) if f.endswith(".zip") and f.startswith("ppo_eurusd")],
        key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x))
    )

    for ck in ckpts:
        ck_path = os.path.join(ckpt_dir, ck)
        try:
            # Загружаем чекпоинт
            m = RecurrentPPO.load(ck_path, env=train_vec_env)
            # Оцениваем его строго на TRAIN-EVAL (In-Sample)
            _, final_eq_train = evaluate_model(m, train_eval_env)
            print(f"[IS Eval] {ck} -> train equity: {final_eq_train:.2f}")
            
            if final_eq_train > best_train_equity:
                best_train_equity = final_eq_train
                best_path = ck_path
                best_model = m
        except Exception as e:
            print(f"[Skip] Could not evaluate checkpoint {ck}: {e}")

    # Decide best model
    if best_path is not None:
        print(f"Using best checkpoint by Train PnL: {best_path} (Train equity: {best_train_equity:.2f})")
    else:
        print("Using last model as best (no checkpoint beat it on Train data).")

    best_model.save("model_eurusd_best")
    print("Best model saved: model_eurusd_best")

    # ---- Plot BOTH: in-sample vs out-of-sample ----
    equity_curve_train, final_equity_train = evaluate_model(best_model, train_eval_env)
    equity_curve_test, final_equity_test = evaluate_model(best_model, test_eval_env)

    print(f"[IS Eval]  Final equity (train): {final_equity_train:.2f}")
    print(f"[OOS Eval] Final equity (test) : {final_equity_test:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_train, label="Train (in-sample) equity")
    plt.plot(equity_curve_test, label="Test (out-of-sample) equity")
    plt.title("Equity Curves: In-sample vs Out-of-sample (Best Model)")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
