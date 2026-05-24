# train_walkforward.py
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO
import torch.nn as nn
from environment import ForexTradingEnv
from manager_env import ManagerEnv

def train_manager_sequential(env_class, df, config, total_timesteps=20000):
    """Обучение Manager на хронологических сегментах."""
    episode_length = config['training'].get('manager_episode_length', 100)
    lookback = 10
    segment_len = episode_length * lookback
    num_segments = max(1, len(df) // segment_len)

    def make_env(start):
        def _init():
            env = env_class(df, config)
            env.reset(start_idx=start)
            return env
        return _init

    num_envs = min(config['training'].get('num_envs_manager', 2), num_segments)
    envs = []
    for i in range(num_envs):
        start = i * segment_len
        if start + segment_len > len(df):
            start = len(df) - segment_len
        envs.append(make_env(start))
    vec_env = DummyVecEnv(envs)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99)

    policy_kwargs = dict(
        lstm_hidden_size=config['training']['policy'].get('lstm_hidden_size', 128),
        n_lstm_layers=config['training']['policy'].get('n_lstm_layers', 1),
        activation_fn=nn.Tanh
    )
    model = RecurrentPPO(
        "MlpLstmPolicy", vec_env, verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=float(config['training']['learning_rate']),
        n_steps=min(128, episode_length // 2),
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5
    )
    model.learn(total_timesteps=total_timesteps)
    return model, vec_env


def train_worker_sequential(env_class, df, config, manager_model, genetic_cols, total_timesteps):
    """Обучение Worker на хронологических сегментах."""
    episode_length = config['training'].get('episode_length', 1000)
    num_segments = len(df) // episode_length

    def make_env(start):
        def _init():
            env = env_class(df, config, manager_model, genetic_cols)
            env.reset(start_idx=start)
            return env
        return _init

    num_envs = config['training']['num_envs']
    envs = []
    for i in range(num_envs):
        start = i * episode_length
        if start + episode_length > len(df):
            start = len(df) - episode_length
        envs.append(make_env(start))
    vec_env = DummyVecEnv(envs)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10., gamma=0.99)

    policy_kwargs = dict(
        net_arch=dict(
            shared=config['training']['policy']['net_arch_shared'],
            pi=config['training']['policy']['net_arch_pi'],
            vf=config['training']['policy']['net_arch_vf']
        ),
        lstm_hidden_size=config['training']['policy']['lstm_hidden_size'],
        n_lstm_layers=config['training']['policy']['n_lstm_layers'],
        activation_fn=nn.Tanh
    )

    model = RecurrentPPO(
        "MlpLstmPolicy", vec_env, verbose=1,
        policy_kwargs=policy_kwargs,
        ent_coef=config['training']['ent_coef'],
        learning_rate=float(config['training']['learning_rate']),
        n_steps=config['training']['n_steps'],
        batch_size=config['training']['batch_size'],
        n_epochs=config['training']['n_epochs'],
        gamma=config['training']['gamma'],
        gae_lambda=config['training']['gae_lambda'],
        clip_range=config['training']['clip_range'],
        vf_coef=config['training']['vf_coef'],
        max_grad_norm=config['training']['max_grad_norm']
    )
    model.learn(total_timesteps=total_timesteps)
    return model, vec_env


def evaluate_full_period(model, env, df_test, vec_normalize=None):
    """
    Оценка на полном тестовом периоде с start_idx=0.
    Возвращает массив эквити.
    """
    env.reset(start_idx=0)
    vec_env = DummyVecEnv([lambda: env])
    if vec_normalize is not None:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=False)
        vec_env.obs_rms = vec_normalize.obs_rms
    obs = vec_env.reset()
    if isinstance(obs, tuple): obs = obs[0]
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    equity = [1.0]
    done = False
    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        episode_starts = done
        equity.append(env.equity)
    vec_env.close()
    return np.array(equity)


def compute_metrics(equity_curve, periods_per_year=8760):
    returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = np.sqrt(periods_per_year) * returns.mean() / (returns.std() + 1e-8)
    downside_std = returns[returns < 0].std()
    sortino = np.sqrt(periods_per_year) * returns.mean() / (downside_std + 1e-8)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_dd = drawdown.min()
    calmar = (equity_curve[-1] - 1) / abs(max_dd) if max_dd != 0 else 0
    return {
        'final_equity': equity_curve[-1],
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'calmar': calmar
    }