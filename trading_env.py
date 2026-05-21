from __future__ import annotations
import numpy as np
import pandas as pd
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces
    _GYMNASIUM = False

class ForexTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        window_size: int = 60,
        feature_columns=None,
        pip_value: float = 0.01,          # 1 копейка
        spread_pips: float = 1.0,
        commission_pips: float = 0.0,
        max_slippage_pips: float = 1.0,
        lot_size: float = 1.0,
        risk_per_trade: float = 0.005,
        leverage: float = 1.0,
        reward_scale: float = 0.002,
        random_start: bool = True,
        min_episode_steps: int = 300,
        episode_max_steps: int | None = None,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        base_sl_pips: float = 40.0,
        base_tp_pips: float = 80.0,
        k_sl: float = 0.3,
        k_tp: float = 0.6,
        open_penalty_pips: float = 0.0,
        time_penalty_pips: float = 0.0005,
        trailing_atr_mult: float = 2.0,
        min_atr_pips: float = 5.0,
        slope_div_reward_scale: float = 0.002,
        open_bonus_pips: float = 5.0,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)

        self.feature_columns = list(feature_columns) if feature_columns else list(df.columns)
        self.window_size = int(window_size)
        self.pip_value = float(pip_value)
        self.spread_pips = float(spread_pips)
        self.commission_pips = float(commission_pips)
        self.max_slippage_pips = float(max_slippage_pips)
        self.lot_size = float(lot_size)
        self.risk_per_trade = float(risk_per_trade)
        self.leverage = float(leverage)
        self.reward_scale = float(reward_scale)
        self.open_penalty_pips = float(open_penalty_pips)
        self.time_penalty_pips = float(time_penalty_pips)
        self.trailing_atr_mult = float(trailing_atr_mult)
        self.min_atr_pips = float(min_atr_pips)
        self.slope_div_reward_scale = float(slope_div_reward_scale)
        self.open_bonus_pips = float(open_bonus_pips)
        self.random_start = bool(random_start)
        self.min_episode_steps = int(min_episode_steps)
        self.episode_max_steps = episode_max_steps if episode_max_steps is None else int(episode_max_steps)
        self.feature_mean = feature_mean
        self.feature_std = feature_std

        self.base_sl_pips = base_sl_pips
        self.base_tp_pips = base_tp_pips
        self.k_sl = k_sl
        self.k_tp = k_tp

        self.action_space = spaces.Discrete(4)  # 0:HOLD, 1:OPEN_LONG, 2:OPEN_SHORT, 3:CLOSE

        self.base_num_features = len(self.feature_columns)
        self.state_num_features = 5
        self.num_features = self.base_num_features + self.state_num_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, self.num_features),
            dtype=np.float32
        )
        self._reset_state()

    def _reset_state(self):
        self.current_idx = 0
        self.next_idx = 1
        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False
        self.position = 0
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.entry_slope = 0.0
        self.initial_equity_usd = 100000.0
        self.equity_usd = self.initial_equity_usd
        self.peak_equity = self.initial_equity_usd
        self.last_trade_info = None
        self.usd_per_pip = self.pip_value * self.lot_size
        self.pending_close_reason = None

    def _get_equity_at_price(self, price: float) -> float:
        if self.position == 0 or self.entry_price is None:
            return self.equity_usd
        if self.position == 1:
            unrealized_pips = (price - self.entry_price) / self.pip_value
        else:
            unrealized_pips = (self.entry_price - price) / self.pip_value
        unrealized_usd = unrealized_pips * self.usd_per_pip
        return self.equity_usd + unrealized_usd

    def _get_current_equity(self) -> float:
        current_close = self.df.loc[self.current_idx, "Close"]
        return self._get_equity_at_price(current_close)

    def _get_state_features(self):
        pos = float(self.position)
        t_norm = float(self.time_in_trade) / 100.0
        if self.position != 0:
            close = float(self.df.loc[self.current_idx, "Close"])
            if self.position == 1:
                unreal_pips = (close - self.entry_price) / self.pip_value
            else:
                unreal_pips = (self.entry_price - close) / self.pip_value
        else:
            unreal_pips = 0.0
        unreal_scaled = unreal_pips / 50.0
        is_flat = 1.0 if self.position == 0 else 0.0
        entry_slope_scaled = self.entry_slope * 10000.0
        return np.array([pos, t_norm, unreal_scaled, is_flat, entry_slope_scaled], dtype=np.float32)

    def _apply_normalization(self, obs):
        if self.feature_mean is None or self.feature_std is None:
            return obs
        std = np.where(self.feature_std == 0, 1.0, self.feature_std)
        std = np.nan_to_num(std, nan=1.0)
        obs = (obs - self.feature_mean) / std
        obs = np.nan_to_num(obs, nan=0.0)
        return obs

    def _get_observation(self):
        start = max(0, self.current_idx - self.window_size + 1)
        indices = list(range(start, self.current_idx + 1))
        if len(indices) < self.window_size:
            pad = self.window_size - len(indices)
            indices = [0] * pad + indices
        obs_df = self.df.iloc[indices]
        base = obs_df[self.feature_columns].values.astype(np.float32)
        base = self._apply_normalization(base)

        current_state = self._get_state_features()
        state_block = np.tile(current_state, (self.window_size, 1))
        obs = np.hstack([base, state_block]).astype(np.float32)
        return obs

    def _sample_slippage_pips(self):
        if self.max_slippage_pips <= 0:
            return 0.0
        return np.random.uniform(0.0, self.max_slippage_pips)

    def _cost_pips_round_trip(self):
        return self.spread_pips + self.commission_pips

    def _compute_sl_tp(self, direction, slope_val, atr_pips, reference_atr=15.0):
        base_sl = self.base_sl_pips
        base_tp = self.base_tp_pips
        vol_factor = atr_pips / reference_atr
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        abs_slope = min(abs(slope_val), 1.0)
        sl_mult = 1 + self.k_sl * abs_slope
        tp_mult = 1 + self.k_tp * abs_slope
        sl_pips = base_sl * sl_mult * vol_factor
        tp_pips = base_tp * tp_mult * vol_factor
        sl_pips = np.clip(sl_pips, 5.0, 100.0)
        tp_pips = max(tp_pips, sl_pips * 1.2)
        return sl_pips, tp_pips

    def _check_margin(self, lot_size, entry_price):
        required_margin = (lot_size * entry_price) / self.leverage
        return self._get_current_equity() > required_margin * 1.05

    def _open_position(self, direction: int, bar):
        if self.position != 0:
            return False

        entry_base_price = float(bar["Open"])
        current_atr = float(bar["alma_atr"]) / self.pip_value
        if current_atr <= 1e-6:
            current_atr = 15.0

        if current_atr < self.min_atr_pips:
            return False

        slope_val = float(bar["slope_div"])
        sl_pips, tp_pips = self._compute_sl_tp(direction, slope_val, current_atr)

        if self.risk_per_trade > 0:
            risk_amount = self._get_current_equity() * self.risk_per_trade
            lot_size = risk_amount / (sl_pips * self.pip_value)
            lot_size = np.clip(lot_size, 1.0, 1000.0)
            max_lot_by_margin = (self._get_current_equity() * self.leverage) / (entry_base_price * 1.2)
            lot_size = min(lot_size, max_lot_by_margin)
            lot_size = max(lot_size, 1.0)
        else:
            lot_size = self.lot_size

        if lot_size <= 0:
            return False
        self.usd_per_pip = self.pip_value * lot_size

        if not self._check_margin(lot_size, entry_base_price):
            return False

        slip_pips = self._sample_slippage_pips()
        slip_price = slip_pips * self.pip_value

        if direction == 1:
            entry = entry_base_price + slip_price
            high = float(bar["High"])
            low = float(bar["Low"])
            entry = np.clip(entry, low, high)
            sl_price = entry - sl_pips * self.pip_value
            tp_price = entry + tp_pips * self.pip_value
            self.position = 1
        else:
            entry = entry_base_price - slip_price
            high = float(bar["High"])
            low = float(bar["Low"])
            entry = np.clip(entry, low, high)
            sl_price = entry + sl_pips * self.pip_value
            tp_price = entry - tp_pips * self.pip_value
            self.position = -1

        self.entry_price = entry
        self.sl_price = sl_price
        self.tp_price = tp_price
        self.time_in_trade = 0
        self.entry_slope = slope_val
        self.last_trade_info = {
            "event": "OPEN",
            "step": self.current_idx,
            "position": self.position,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "slope_div": slope_val,
            "atr_pips": current_atr,
            "lot_size": lot_size
        }
        penalty_usd = self.open_penalty_pips * self.usd_per_pip
        if penalty_usd > 0:
            self.equity_usd -= penalty_usd
        if self.open_bonus_pips > 0 and slope_val < 0:
            self.equity_usd += self.open_bonus_pips * self.usd_per_pip
        return True

    def _close_position(self, reason: str, exit_price: float):
        if self.position == 1:
            pnl_price = exit_price - self.entry_price
        else:
            pnl_price = self.entry_price - exit_price
        realized_pips = pnl_price / self.pip_value
        cost_pips = self._cost_pips_round_trip()
        net_pips = realized_pips - cost_pips
        self.equity_usd += net_pips * self.usd_per_pip
        trade_info = {
            "event": "CLOSE",
            "reason": reason,
            "step": self.current_idx,
            "position": self.position,
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "realized_pips": realized_pips,
            "net_pips": net_pips,
            "equity_usd": self._get_current_equity(),
            "time_in_trade": self.time_in_trade,
            "lot_size": self.usd_per_pip / self.pip_value
        }
        self.position = 0
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.entry_slope = 0.0
        self.last_trade_info = trade_info
        return net_pips

    def _check_sl_tp_intrabar(self, bar):
        if self.position == 0:
            return None
        high = float(bar["High"])
        low = float(bar["Low"])
        current_atr = float(bar["alma_atr"]) / self.pip_value
        trailing_trigger = current_atr * self.trailing_atr_mult

        if self.position == 1:
            if high >= self.entry_price + trailing_trigger * self.pip_value:
                new_sl = self.entry_price
                if new_sl > self.sl_price:
                    self.sl_price = new_sl
            if high >= self.tp_price:
                return self._close_position("TP_HIT", self.tp_price)
            if low <= self.sl_price:
                return self._close_position("SL_HIT", self.sl_price)
        else:
            if low <= self.entry_price - trailing_trigger * self.pip_value:
                new_sl = self.entry_price
                if new_sl < self.sl_price:
                    self.sl_price = new_sl
            if low <= self.tp_price:
                return self._close_position("TP_HIT", self.tp_price)
            if high >= self.sl_price:
                return self._close_position("SL_HIT", self.sl_price)
        return None

    def _get_auto_close_signal(self, bar):
        if self.position == 0:
            return None
        bull = bar["bull_div"] == 1
        bear = bar["bear_div"] == 1
        slope_div = bar["slope_div"]
        if self.position == 1 and bear and slope_div < -0.3:
            return "BEAR_DIV"
        if self.position == -1 and bull and slope_div < -0.3:
            return "BULL_DIV"
        return None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        if self.random_start:
            max_start = self.n_steps - max(self.min_episode_steps, self.window_size) - 2
            start = np.random.randint(self.window_size, max(self.window_size + 1, max_start))
            self.current_idx = start
            self.next_idx = start + 1
        else:
            self.current_idx = self.window_size
            self.next_idx = self.window_size + 1

        if self.next_idx >= self.n_steps:
            self.next_idx = self.n_steps - 1
            self.current_idx = self.next_idx - 1

        self.steps_in_episode = 0
        obs = self._get_observation()
        if _GYMNASIUM:
            return obs, {}
        return obs

    def step(self, action: int):
        if self.terminated or self.truncated:
            obs = self._get_observation()
            if _GYMNASIUM:
                return obs, 0.0, True, False, {}
            return obs, 0.0, True, {}

        self.steps_in_episode += 1

        curr_bar = self.df.iloc[self.current_idx]
        next_bar = self.df.iloc[self.next_idx] if self.next_idx < self.n_steps else None

        if next_bar is None:
            self.terminated = True
            if self.position != 0:
                last_close = curr_bar["Close"]
                self._close_position("END_OF_DATA", last_close)
            obs = self._get_observation()
            info = {"equity_usd": self._get_current_equity()}
            if _GYMNASIUM:
                return obs, 0.0, True, False, info
            return obs, 0.0, True, info

        equity_start = self._get_equity_at_price(curr_bar["Close"])

        if self.pending_close_reason is not None:
            self._close_position(self.pending_close_reason, next_bar["Open"])
            self.pending_close_reason = None

        if action == 1:
            if self.position == 0:
                self._open_position(1, next_bar)
        elif action == 2:
            if self.position == 0:
                self._open_position(-1, next_bar)
        elif action == 3:
            if self.position != 0:
                self._close_position("MANUAL_CLOSE", next_bar["Open"])

        if self.position != 0:
            self.time_in_trade += 1
            self.equity_usd -= self.time_penalty_pips * self.usd_per_pip

        if self.position != 0:
            self._check_sl_tp_intrabar(next_bar)

        if self.position != 0:
            signal = self._get_auto_close_signal(next_bar)
            if signal is not None:
                self.pending_close_reason = signal

        equity_end = self._get_equity_at_price(next_bar["Close"])

        # -------- НАГРАДА (без шума) --------
        pnl_usd = equity_end - equity_start
        atr_pips = curr_bar["alma_atr"] / self.pip_value
        atr_usd = atr_pips * self.usd_per_pip if self.usd_per_pip > 0 else 1.0
        atr_usd = max(atr_usd, 1.0)

        risk_adjusted_pnl = pnl_usd / atr_usd

        self.peak_equity = max(self.peak_equity, equity_end)
        drawdown = (self.peak_equity - equity_end) / self.peak_equity if self.peak_equity > 0 else 0
        drawdown_penalty = -drawdown * 5.0

        action_bonus = 0.0
        if action in (1, 2) and self.position != 0:
            action_bonus = 0.02
        elif action == 3 and self.position == 0:
            action_bonus = 0.01

        shaping_usd = 0.0
        if self.position != 0:
            shaping_usd = -curr_bar["slope_div"] * self.slope_div_reward_scale * self.usd_per_pip

        raw_reward = risk_adjusted_pnl + drawdown_penalty + action_bonus + (shaping_usd / atr_usd)
        reward = raw_reward * self.reward_scale
        # ------------------------------------

        self.current_idx += 1
        self.next_idx += 1

        if self.next_idx >= self.n_steps:
            self.terminated = True
            if self.position != 0:
                last_bar = self.df.iloc[-1]
                self._close_position("END_OF_DATA", last_bar["Close"])

        if self.episode_max_steps is not None and self.steps_in_episode >= self.episode_max_steps:
            self.truncated = True

        obs = self._get_observation()
        info = {"equity_usd": self._get_current_equity()}
        info["last_trade_info"] = self.last_trade_info

        if _GYMNASIUM:
            return obs, reward, self.terminated, self.truncated, info
        else:
            done = self.terminated or self.truncated
            return obs, reward, done, info