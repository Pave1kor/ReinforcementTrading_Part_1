# manager_env.py
import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any

class ManagerEnv(gym.Env):

    # Manager действует каждые 10 баров, предсказывает risk и direction.
    # Хронологический проход без случайных reset.

    def __init__(self, df: pd.DataFrame, config: Dict[str, Any]):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.config = config
        self.lookback = 10
        self.idx = 0
        self.obs_dim = 5
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)
        self._start_idx = 0

    def reset(self, start_idx=None, seed=None, options=None):
        super().reset(seed=seed)
        if start_idx is not None:
            self._start_idx = start_idx
        else:
            self._start_idx = 0
        self.idx = self._start_idx
        return self._get_obs(), {}

    def step(self, action):
        risk = float(np.clip(action[0], -1, 1))
        direction = float(np.clip(action[1], -1, 1))
        start = self.idx
        end = min(start + self.lookback, len(self.df) - 1)
        ret = self.df['Close'].iloc[end] / self.df['Close'].iloc[start] - 1 if end > start else 0.0
        reward = direction * risk * ret
        self.idx += self.lookback
        done = self.idx >= len(self.df) - self.lookback - 1
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        idx = min(self.idx, len(self.df)-1)
        row = self.df.iloc[idx]
        return np.array([row['adx'], row['atr_norm'], row['volume_zscore'],
                         row['rsi_norm'], row['macd_hist']], dtype=np.float32)