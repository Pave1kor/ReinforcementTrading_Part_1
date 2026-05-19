# trading_env.py
from __future__ import annotations
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces
    _GYMNASIUM = False


class ForexTradingEnv(gym.Env):
    """
    RL Forex Trading Environment (Position-Persistent) - Clean, Fixed & Optimized Version
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        window_size: int = 30,
        sl_options=None,
        tp_options=None,
        feature_columns = None,
        pip_value: float = 0.0001,
        spread_pips: float = 1.0,              
        commission_pips: float = 0.0,          
        max_slippage_pips: float = 0.5,        
        lot_size: float = 100000.0,            
        reward_scale: float = 0.01,             
        random_start: bool = True,
        min_episode_steps: int = 300,          
        episode_max_steps: int | None = None,  
        feature_mean: np.ndarray | None = None, 
        feature_std: np.ndarray | None = None,  
        allow_flip: bool = False,               
        open_penalty_pips: float = 0.1,      
        time_penalty_pips: float = 0.00,     
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)

        if feature_columns is None:
            self.feature_columns = list(self.df.columns)  
        else:
            self.feature_columns = list(feature_columns)

        if sl_options is None or tp_options is None:
            raise ValueError("sl_options and tp_options must be provided.")
        self.sl_options = list(sl_options)
        self.tp_options = list(tp_options)

        if self.n_steps <= window_size + 2:
            raise ValueError("Dataframe is too short for the given window_size.")

        self.window_size = int(window_size)
        self.pip_value = float(pip_value)
        self.spread_pips = float(spread_pips)
        self.commission_pips = float(commission_pips)
        self.max_slippage_pips = float(max_slippage_pips)

        self.lot_size = float(lot_size)
        self.usd_per_pip = self.pip_value * self.lot_size

        self.reward_scale = float(reward_scale)
        self.open_penalty_pips = float(open_penalty_pips)
        self.time_penalty_pips = float(time_penalty_pips)

        self.random_start = bool(random_start)
        self.min_episode_steps = int(min_episode_steps)
        self.episode_max_steps = episode_max_steps if episode_max_steps is None else int(episode_max_steps)

        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.allow_flip = bool(allow_flip)

        # Actions map construction
        self.action_map = [("HOLD", None, None, None), ("CLOSE", None, None, None)]
        for direction in [1, -1]:  # 1 = Long, -1 = Short для явной математики
            for sl in self.sl_options:
                for tp in self.tp_options:
                    self.action_map.append(("OPEN", direction, float(sl), float(tp)))

        self.action_space = spaces.Discrete(len(self.action_map))
        self.base_num_features = len(self.feature_columns)
        self.state_num_features = 5
        self.num_features = self.base_num_features + self.state_num_features

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, self.num_features), dtype=np.float32
        )
        self._reset_state()

    def _reset_state(self):
        self.current_step = 0
        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False
        self.position = 0              
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.entry_atr = 0.0
        self.initial_equity_usd = 10000.0
        self.equity_usd = self.initial_equity_usd
        self.equity_curve = []
        self.last_trade_info = None

    def _get_state_features(self):
        pos = float(self.position)
        t_norm = float(self.time_in_trade) / 100.0
        unreal_pips = float(self._compute_unrealized_pips()) if self.position != 0 else 0.0
        unreal_scaled = unreal_pips / 50.0  
        is_flat = 1.0 if self.position == 0 else 0.0
        entry_atr_scaled = (self.entry_atr / self.pip_value) / 20.0 if self.pip_value != 0 else 0.0
        return np.array([pos, t_norm, unreal_scaled, is_flat, entry_atr_scaled], dtype=np.float32)

    def _compute_unrealized_pips(self):
        if self.position == 0 or self.entry_price is None:
            return 0.0
        close_price = float(self.df.loc[self.current_step, "Close"])
        pnl_price = (close_price - self.entry_price) if self.position == 1 else (self.entry_price - close_price)
        return pnl_price / self.pip_value

    def _apply_optional_normalization(self, obs: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None:
            return obs
        mean = self.feature_mean.reshape(1, -1)
        std = self.feature_std.reshape(1, -1)
        std = np.where(std == 0, 1.0, std)
        return (obs - mean) / std

    def _get_observation(self):
        start = max(0, self.current_step - self.window_size + 1)
        obs_df = self.df.iloc[start:self.current_step+1][self.feature_columns]
        base = obs_df.values.astype(np.float32)
        
        if base.shape[0] < self.window_size:
            pad_rows = self.window_size - base.shape[0]
            pad = np.tile(base[0], (pad_rows, 1))
            base = np.vstack([pad, base])
    
        base = self._apply_optional_normalization(base)
        current_state_features = self._get_state_features()
        state_block = np.tile(current_state_features, (self.window_size, 1))

        return np.hstack([base, state_block]).astype(np.float32)

    def _sample_slippage_pips(self) -> float:
        if self.max_slippage_pips <= 0:
            return 0.0
        return float(np.random.uniform(0.0, self.max_slippage_pips))

    def _cost_pips_round_trip(self) -> float:
        return self.spread_pips + self.commission_pips

    def _open_position(self, direction: int, sl_pips: float, tp_pips: float):
        close_price = float(self.df.loc[self.current_step, "Close"])
        current_atr = float(self.df.loc[self.current_step, "alma_atr"])
        if current_atr <= 0:
            current_atr = 15.0 * self.pip_value 

        slip_price = self._sample_slippage_pips() * self.pip_value

        if direction == 1:  
            entry = close_price + slip_price
            sl_price = entry - (sl_pips * current_atr)
            tp_price = entry + (tp_pips * current_atr)
            self.position = 1
        else:               
            entry = close_price - slip_price
            sl_price = entry + (sl_pips * current_atr)
            tp_price = entry - (tp_pips * current_atr)
            self.position = -1

        self.entry_price = entry
        self.sl_price = sl_price
        self.tp_price = tp_price
        self.time_in_trade = 0
        self.entry_atr = current_atr  

        self.last_trade_info = {
            "event": "OPEN",
            "step": self.current_step,
            "position": self.position,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "atr_pips": float(current_atr / self.pip_value)
        }

    def _close_position(self, reason: str, exit_price: float):
        pnl_price = (exit_price - self.entry_price) if self.position == 1 else (self.entry_price - exit_price)
        realized_pips = pnl_price / self.pip_value
        cost_pips = self._cost_pips_round_trip()
        net_pips = realized_pips - cost_pips
        self.equity_usd += net_pips * self.usd_per_pip

        trade_info = {
            "event": "CLOSE",
            "reason": reason,
            "step": self.current_step,
            "position": self.position,
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "realized_pips": float(realized_pips),
            "cost_pips": float(cost_pips),
            "net_pips": float(net_pips),
            "equity_usd": float(self.equity_usd),
            "time_in_trade": int(self.time_in_trade),
        }

        self.position = 0
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.last_trade_info = trade_info
        return net_pips

    def _check_sl_tp_intrabar_and_maybe_close(self) -> float | None:
        if self.position == 0 or self.current_step >= self.n_steps:
            return None

        high = float(self.df.loc[self.current_step, "High"])
        low = float(self.df.loc[self.current_step, "Low"])

        if self.position == 1:
            sl_hit = low <= self.sl_price
            tp_hit = high >= self.tp_price
        else:
            sl_hit = high >= self.sl_price
            tp_hit = low <= self.tp_price

        if sl_hit and tp_hit:
            # Консервативный подход: при одновременном касании засчитываем худший исход (SL)
            return self._close_position("SL_AND_TP_SAME_BAR_CONSERVATIVE", self.sl_price)
        elif sl_hit:
            return self._close_position("SL_HIT", self.sl_price)
        elif tp_hit:
            return self._close_position("TP_HIT", self.tp_price)

        return None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()

        if self.random_start:
            max_start = self.n_steps - max(self.min_episode_steps, self.window_size) - 2
            self.current_step = int(np.random.randint(self.window_size, max(self.window_size + 1, max_start)))
        else:
            self.current_step = self.window_size

        self.steps_in_episode = 0
        obs = self._get_observation()
        return (obs, {}) if _GYMNASIUM else obs

    def step(self, action: int):
        if self.terminated or self.truncated:
            obs = self._get_observation()
            return (obs, 0.0, True, False, {}) if _GYMNASIUM else (obs, 0.0, True, {})

        self.steps_in_episode += 1
        reward_pips = 0.0
        info = {}
        self.last_trade_info = None  # Сбрасываем лог перед шагом

        act_type, direction, sl_pips, tp_pips = self.action_map[int(action)]
        position_changed_this_step = False

        if act_type == "CLOSE" and self.position != 0:
            close_price = float(self.df.loc[self.current_step, "Close"])
            reward_pips += self._close_position("MANUAL_CLOSE", close_price)
            position_changed_this_step = True
        elif act_type == "OPEN" and self.position == 0:
            self._open_position(direction=direction, sl_pips=sl_pips, tp_pips=tp_pips)
            reward_pips -= self.open_penalty_pips
        elif act_type == "OPEN" and self.allow_flip and self.position != 0 and self.position != direction:
            close_price = float(self.df.loc[self.current_step, "Close"])
            reward_pips += self._close_position("FLIP_CLOSE", close_price)
            self._open_position(direction=direction, sl_pips=sl_pips, tp_pips=tp_pips)
            reward_pips -= self.open_penalty_pips

        # Intrabar проверка выполняется только если позиция удержана или только что открыта
        if self.position != 0 and not position_changed_this_step:
            self.time_in_trade += 1
            reward_pips -= self.time_penalty_pips
            realized_now = self._check_sl_tp_intrabar_and_maybe_close()
            if realized_now is not None:
                reward_pips += realized_now

        self.equity_curve.append(float(self.equity_usd))
        self.current_step += 1

        if self.current_step >= self.n_steps - 1:
            self.terminated = True
            if self.position != 0:
                close_price = float(self.df.loc[self.n_steps - 1, "Close"])
                reward_pips += self._close_position("END_OF_DATA", close_price)

        if self.episode_max_steps is not None and self.steps_in_episode >= self.episode_max_steps:
            self.truncated = True

        obs = self._get_observation()
        reward = float(reward_pips) * self.reward_scale

        info.update({
            "equity_usd": float(self.equity_usd),
            "position": int(self.position),
            "time_in_trade": int(self.time_in_trade),
            "reward_pips": float(reward_pips),
            "last_trade_info": self.last_trade_info
        })

        if _GYMNASIUM:
            return obs, reward, self.terminated, self.truncated, info
        return obs, reward, bool(self.terminated or self.truncated), info
