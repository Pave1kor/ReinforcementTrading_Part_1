# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt

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
    # �������������� ������� ��������� ������ LSTM ������ (����� ������ �� ������ �� ������)
    lstm_states = None
    # ����� ������ ������� (True ��������, ��� ��� ������ ���)
    episode_starts = np.ones((eval_env.num_envs,), dtype=bool)

    while True:

        # �������� � predict ��������� ������ � ����� ������
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states, 
            episode_start=episode_starts, 
            deterministic=deterministic
        )

        step_out = eval_env.step(action)

        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, rewards, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])

        # ����� ������� ���� ����� ��� ��������� ������ ������� (����� ���������� False)
        episode_starts = np.zeros((eval_env.num_envs,), dtype=bool)

        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        eq = info.get("equity_usd", eval_env.get_attr("equity_usd")[0])
        if isinstance(eq, (list, np.ndarray)):
           eq = eq[0]
        equity_curve.append(float(eq))

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
    SL_OPTS = [30, 90]
    TP_OPTS = [30,90]
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
            max_slippage_pips=0.2,
            random_start=True,
            min_episode_steps=1000,
            episode_max_steps=None,
            feature_columns=feature_cols,
            hold_reward_weight=0.0,#0.05
            open_penalty_pips=0.1,      # 0.5 half a pip per open
            time_penalty_pips=0.0,     # 0.02 pips per bar in trade
            unrealized_delta_weight=1.0
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
            open_penalty_pips=0.1,      # half a pip per open
            time_penalty_pips=0.0,     # 0.02 pips per bar in trade
            unrealized_delta_weight=1.0
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
            open_penalty_pips=0.1,      # half a pip per open
            time_penalty_pips=0.0,     # 0.02 pips per bar in trade
            unrealized_delta_weight=1.0
        )

    train_vec_env = SubprocVecEnv([lambda i=i: make_train_env() for i in range(NUM_ENVS)])
    train_eval_env = DummyVecEnv([make_train_eval_env])
    test_eval_env = DummyVecEnv([make_test_eval_env])

    policy_kwargs = dict(
        net_arch=dict(
            shared=[128, 128],  # Общие слои помогают критику точнее понимать контекст актера
            pi=[64], 
            vf=[64]
        ),  # ������� ���� �� LSTM
        lstm_hidden_size=128,                        # ������ ������ LSTM-������
        n_lstm_layers=1                              # ���������� ����� LSTM
    )


    # ---- Model ----
    model = RecurrentPPO(
        policy="MlpLstmPolicy",       # ����������� �������� � �������
        env=train_vec_env,
        verbose=1,
        tensorboard_log="./tensorboard_log/",
        policy_kwargs=policy_kwargs, # ��������� ����������� ����������� ����
        ent_coef=0.015,
        learning_rate=7e-5,       # ����������� ���, ����� ������� �� 1e-4, ���� ������ �������� ����� ������
        n_steps= 1024,             # ������� ����� �������� ���� ����� ����� ����������� �����
        batch_size=256,            # ������ ����-����� ��� ������������ ������ (64 ��� 128)
        n_epochs=10,              # ���������� ���� ����������� �� ���� ���� ������
        gamma=0.99,               # ����������� ��������������� (0.99 �������� ����� �� ������������ �������)
        gae_lambda=0.95,          # �������� ��� Generalized Advantage Estimation
        clip_range=0.1,           # ����������� ��������� �������� (����� ���� �� �������� ������� ��������)
        vf_coef=0.7               # ��� ������ ������� �������� (Value Function)
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
    total_timesteps = 1500000
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # ---- Select best model by OOS final equity ----
    _, final_equity_train_last = evaluate_model(model, train_eval_env)
    print(f"[IS Eval] Last model final equity: {final_equity_train_last:.2f}")

    best_train_equity = final_equity_train_last
    best_path = None
    best_model = model

    ckpts = sorted(
        [f for f in os.listdir(ckpt_dir) if f.endswith(".zip") and f.startswith(CKPT_PREFIX)],
        key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x))
    )

    for ck in ckpts:
        ck_path = os.path.join(ckpt_dir, ck)
        try:
            # ��������� ��������
            m = RecurrentPPO.load(ck_path, env=None)
            # ��������� ��� ������ �� TRAIN-EVAL (In-Sample)
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
    
    best_model.set_env(train_vec_env, force_reset=False)
    best_model.save("model_eurusd_best")
    print("Best model saved: model_eurusd_best")

    # ---- Plot BOTH: in-sample vs out-of-sample ----
    plot_model_train = RecurrentPPO.load("model_eurusd_best", env=None)
    equity_curve_train, final_equity_train = evaluate_model(plot_model_train, train_eval_env)
    
    plot_model_test = RecurrentPPO.load("model_eurusd_best", env=None)
    equity_curve_test, final_equity_test = evaluate_model(plot_model_test, test_eval_env)


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
    plt.savefig("model1")


if __name__ == "__main__":
    main()
