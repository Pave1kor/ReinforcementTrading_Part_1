import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv


def evaluate_model(model: RecurrentPPO, eval_env: DummyVecEnv, deterministic: bool = True):
    # Это автоматически перестроит размерность скрытых состояний LSTM под num_envs = 1.
    model.set_env(eval_env, force_reset=False)
    obs = eval_env.reset()
    equity_curve = []

    lstm_states = None
    episode_starts = np.ones((eval_env.num_envs,), dtype=bool)

    while True:
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states, 
            episode_start=episode_starts, 
            deterministic=deterministic
        )

        obs, rewards, dones, infos = eval_env.step(action)
        done = bool(dones[0])

        # Передаем реальный статус завершения на следующий шаг для LSTM
        episode_starts = dones

        # Извлекаем словарь info для единственной среды в векторе
        info = infos[0]

        if done:
            # Если это последний шаг, вытаскиваем финальное эквити из словаря info
            if isinstance(info, dict) and "equity_usd" in info:
                equity_curve.append(float(info["equity_usd"]))
            elif isinstance(info, dict) and "terminal_observation" in info and "equity_usd" in info["terminal_observation"]:
                equity_curve.append(float(info["terminal_observation"]["equity_usd"]))
            else:
                if equity_curve:
                    equity_curve.append(equity_curve[-1])
            break # Мгновенно выходим из цикла
        else:
            # Если эпизод продолжается, берем текущее эквити из ЖИВОЙ среды eval_env
            equity_curve.append(float(eval_env.get_attr("equity_usd")[0]))

    final_equity = float(equity_curve[-1])
    return equity_curve, final_equity



def main():
    #file_path = "data/EURUSD_15 Mins_Ask_2020.12.06_2025.12.12.csv"
    file_path = "data/test_EURUSD_Candlestick_1_Hour_BID_20.02.2023-22.02.2025.csv"
    df, feature_cols = load_and_preprocess_data(file_path)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print("Training bars:", len(train_df))
    print("Testing bars :", len(test_df))

    # Извлекаем признаки для расчета параметров нормализации (БЕЗ учета таргетов/цен, если нужно)
    # Считаем среднее и дисперсию СТРОГО на обучающей выборке (Защита от Data Leakage)
    train_features_df = train_df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    train_features = train_features_df.values.astype(np.float32)
    train_mean = np.mean(train_features, axis=0)
    train_std = np.std(train_features, axis=0)

    # ---- Env factories ----
    SL_OPTS = [1.0, 1.5]
    TP_OPTS = [2.0, 3.5]
    WIN = 60
    NUM_ENVS = 4  # Задействуем 4 параллельных потока (FPS вырастет до 160-200+)
    
    # Train env: random starts to reduce memorization
    def make_train_env():
        return ForexTradingEnv(
            df=train_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.5,
            random_start=True,
            min_episode_steps=300,
            episode_max_steps=800,
            feature_mean=train_mean, 
            feature_std=train_std,    
            feature_columns=feature_cols,
            open_penalty_pips=0.1,      # 0.5 half a pip per open
            time_penalty_pips=0.005,     # 0.02 pips per bar in trade
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
            max_slippage_pips=0.5,
            random_start=False,
            episode_max_steps=None,
            feature_mean=train_mean,
            feature_std=train_std,
            feature_columns=feature_cols,
            open_penalty_pips=0.1,      # half a pip per open
            time_penalty_pips=0.005,     # 0.02 pips per bar in trade
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
            max_slippage_pips=0.5,
            random_start=False,
            episode_max_steps=None,
            feature_std=train_std,
            feature_mean=train_mean, 
            feature_columns=feature_cols,
            open_penalty_pips=0.1,      # half a pip per open
            time_penalty_pips=0.005,     # 0.02 pips per bar in trade
        )

    train_vec_env = SubprocVecEnv([lambda i=i: make_train_env() for i in range(NUM_ENVS)])
    train_eval_env = DummyVecEnv([make_train_eval_env])
    test_eval_env = DummyVecEnv([make_test_eval_env])

    policy_kwargs = dict(
        net_arch=dict(
            shared=[64],
            pi=[32], 
            vf=[32]
        ),
        lstm_hidden_size=48,
        n_lstm_layers=1
    )


    # ---- Model ----
    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=train_vec_env,
        verbose=1,
        tensorboard_log="./tensorboard_log/",
        policy_kwargs=policy_kwargs,
        ent_coef=0.02,
        learning_rate=1e-4,
        n_steps= 1024,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        vf_coef=0.5
    )

    # ---- Checkpoints ----
    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    CKPT_PREFIX = "RecurrentPPO_eurusd"
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=ckpt_dir,
        name_prefix=CKPT_PREFIX
    )

    # ---- Train ----
    total_timesteps = 500000
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # ---- Select best model by OOS final equity ----
    model.save("model_eurusd_best")
    print("Model saved: model_eurusd_best")

    # ---- Plot BOTH: in-sample vs out-of-sample ----
    plot_model_train = RecurrentPPO.load("model_eurusd_best", env=train_eval_env)
    equity_curve_train, final_equity_train = evaluate_model(plot_model_train, train_eval_env)
    
    plot_model_test = RecurrentPPO.load("model_eurusd_best", env=test_eval_env)
    equity_curve_test, final_equity_test = evaluate_model(plot_model_test, test_eval_env)


    print(f"[IS Eval]  Final equity (train): {final_equity_train:.2f}")
    print(f"[OOS Eval] Final equity (test) : {final_equity_test:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_train, label=f"Train (in-sample) equity: {final_equity_train:.2f}$")
    plt.title("Train (In-Sample) Equity Curve - Best Model")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("model1_train_clean.png")

    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_test, label=f"Test (out-of-sample) equity: {final_equity_test:.2f}$", color="orange")
    plt.title("Test (Out-of-Sample) Equity Curve - Best Model")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("model1_test_clean.png")



if __name__ == "__main__":
    main()
