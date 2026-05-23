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
        pip_value: float = 0.01,
        spread_pips: float = 1.0,
        max_slippage_pips: float = 1.0,
        risk_per_trade: float = 0.003,
        leverage: float = 1.0,
        reward_scale: float = 0.02,
        commission_percent: float = 0.0005,
        random_start: bool = True,
        min_episode_steps: int = 300,
        episode_max_steps: int | None = None,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        # Динамический SL на основе ATR
        use_atr_sl: bool = True,
        atr_multiplier_sl: float = 2.0,
        atr_multiplier_tp: list = [3.0, 5.0, 7.0, 9.0, 12.0],
        # Частичное закрытие
        partial_close_enabled: bool = True,
        partial_close_ratio: float = 0.5,
        tp_levels: list = [0.5, 1.0],
        # Фиксированные параметры (запасные)
        fixed_sl_percent: float = 0.02,
        fixed_tp_percents: list = [0.02, 0.04, 0.06, 0.08, 0.10],
        open_penalty_pips: float = 0.1,
        time_penalty_pips: float = 0.01,
        slope_div_reward_scale: float = 0.01,
        open_bonus_pips: float = 0.0,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)

        self.feature_columns = list(feature_columns) if feature_columns else list(df.columns)
        self.window_size = int(window_size)
        self.pip_value = float(pip_value)
        self.spread_pips = float(spread_pips)
        self.max_slippage_pips = float(max_slippage_pips)
        self.risk_per_trade = float(risk_per_trade)
        self.leverage = float(leverage)
        self.reward_scale = float(reward_scale)
        self.commission_percent = float(commission_percent)
        self.open_penalty_pips = float(open_penalty_pips)
        self.time_penalty_pips = float(time_penalty_pips)
        self.slope_div_reward_scale = float(slope_div_reward_scale)
        self.open_bonus_pips = float(open_bonus_pips)
        self.random_start = bool(random_start)
        self.min_episode_steps = int(min_episode_steps)
        self.episode_max_steps = episode_max_steps if episode_max_steps is None else int(episode_max_steps)
        self.feature_mean = feature_mean
        self.feature_std = feature_std

        # Новые параметры
        self.use_atr_sl = use_atr_sl
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp = atr_multiplier_tp
        self.partial_close_enabled = partial_close_enabled
        self.partial_close_ratio = partial_close_ratio
        self.tp_levels = tp_levels
        self.fixed_sl_percent = fixed_sl_percent
        self.fixed_tp_percents = fixed_tp_percents

        self.action_space = spaces.Discrete(12)

        self.base_num_features = len(self.feature_columns)
        self.state_num_features = 6
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
        self.tp_levels_hit = []          # для частичного закрытия
        self.remaining_position_size = 1.0  # доля от исходного размера (1.0 = полный)
        self.time_in_trade = 0
        self.entry_slope = 0.0
        self.initial_equity_usd = 100000.0
        self.equity_usd = self.initial_equity_usd
        self.peak_equity = self.initial_equity_usd
        self.last_trade_info = None
        self.usd_per_pip = self.pip_value * 1.0
        self.pending_close_reason = None
        self.last_lot_multiplier = 1
        self.entry_atr = 0.0

    def _get_equity_at_price(self, price: float) -> float:
        if self.position == 0 or self.entry_price is None:
            return self.equity_usd
        if self.position == 1:
            unrealized_pips = (price - self.entry_price) / self.pip_value
        else:
            unrealized_pips = (self.entry_price - price) / self.pip_value
        unrealized_usd = unrealized_pips * self.usd_per_pip * self.remaining_position_size
        return self.equity_usd + unrealized_usd

    def _get_current_equity(self) -> float:
        current_close = self.df.loc[self.current_idx, "Close"]
        return self._get_equity_at_price(current_close)

    def _get_state_features(self):
        pos = float(self.position) * self.remaining_position_size
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
        lot_mult_norm = float(self.last_lot_multiplier) / 5.0
        return np.array([pos, t_norm, unreal_scaled, is_flat, entry_slope_scaled, lot_mult_norm], dtype=np.float32)

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

    def _cost_commission_rub(self, price, lot_size):
        return price * lot_size * self.commission_percent

    def _open_position(self, direction: int, bar, lot_multiplier: int = 1):
        if self.position != 0:
            return False

        entry_base_price = float(bar["Open"])
        atr = float(bar["alma_atr"])
        self.entry_atr = atr

        if self.use_atr_sl:
            sl_distance = atr * self.atr_multiplier_sl
            tp_distance = atr * self.atr_multiplier_tp[lot_multiplier - 1]
        else:
            sl_distance = entry_base_price * self.fixed_sl_percent
            tp_distance = entry_base_price * self.fixed_tp_percents[lot_multiplier - 1]

        risk_amount = self._get_current_equity() * self.risk_per_trade * lot_multiplier
        lot_size = risk_amount / sl_distance
        lot_size = np.clip(lot_size, 1.0, 5000.0)
        max_lot_by_margin = (self._get_current_equity() * self.leverage) / (entry_base_price * 1.2)
        lot_size = min(lot_size, max_lot_by_margin)
        lot_size = max(lot_size, 1.0)

        if lot_size <= 0:
            return False
        self.usd_per_pip = self.pip_value * lot_size

        required_margin = (lot_size * entry_base_price) / self.leverage
        if self._get_current_equity() <= required_margin * 1.05:
            return False

        slip_pips = self._sample_slippage_pips()
        slip_price = slip_pips * self.pip_value

        if direction == 1:
            entry = entry_base_price + slip_price
            high = float(bar["High"])
            low = float(bar["Low"])
            entry = np.clip(entry, low, high)
            sl_price = entry - sl_distance - self.spread_pips * self.pip_value
            tp_price = entry + tp_distance
            self.position = 1
        else:
            entry = entry_base_price - slip_price
            high = float(bar["High"])
            low = float(bar["Low"])
            entry = np.clip(entry, low, high)
            sl_price = entry + sl_distance + self.spread_pips * self.pip_value
            tp_price = entry - tp_distance
            self.position = -1

        commission_rub = self._cost_commission_rub(entry, lot_size)
        self.equity_usd -= commission_rub

        self.entry_price = entry
        self.sl_price = sl_price
        self.tp_price = tp_price
        self.tp_levels_hit = []
        self.remaining_position_size = 1.0
        self.time_in_trade = 0
        self.entry_slope = float(bar["slope_div"])
        self.last_lot_multiplier = lot_multiplier
        self.last_trade_info = {
            "event": "OPEN",
            "step": self.current_idx,
            "position": self.position,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "atr": atr,
            "sl_distance": sl_distance,
            "tp_distance": tp_distance,
            "lot_size": lot_size,
            "lot_multiplier": lot_multiplier,
            "commission_rub": commission_rub
        }

        if self.open_penalty_pips > 0:
            self.equity_usd -= self.open_penalty_pips * self.usd_per_pip
        return True

    def _close_position(self, reason: str, exit_price: float, fraction: float = 1.0):

        # Закрывает долю позиции (fraction от оставшейся).

        if self.position == 0:
            return 0.0
        
        close_fraction = min(fraction, self.remaining_position_size)
        if close_fraction <= 0:
            return 0.0
        
        if self.position == 1:
            pnl_price = exit_price - self.entry_price
        else:
            pnl_price = self.entry_price - exit_price
        realized_pips = pnl_price / self.pip_value
        cost_pips = self.spread_pips
        net_pips = realized_pips - cost_pips
        pnl_usd = net_pips * self.usd_per_pip * close_fraction
        
        # Комиссия за закрытие этой части
        commission_close = self._cost_commission_rub(exit_price, (self.usd_per_pip / self.pip_value) * close_fraction)
        pnl_usd -= commission_close
        
        self.equity_usd += pnl_usd
        
        trade_info = {
            "event": "CLOSE",
            "reason": reason,
            "step": self.current_idx,
            "position": self.position,
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "realized_pips": realized_pips,
            "net_pips": net_pips,
            "pnl_usd": pnl_usd,
            "fraction": close_fraction,
            "remaining_position_size": self.remaining_position_size - close_fraction,
            "equity_usd": self._get_current_equity(),
            "time_in_trade": self.time_in_trade,
            "lot_size": self.usd_per_pip / self.pip_value,
            "commission_rub": commission_close
        }
        
        self.remaining_position_size -= close_fraction
        if self.remaining_position_size <= 1e-6:
            # Полностью закрыли
            self.position = 0
            self.entry_price = None
            self.sl_price = None
            self.tp_price = None
            self.time_in_trade = 0
            self.entry_slope = 0.0
            self.remaining_position_size = 0.0
        else:
            # Частично закрыли – обновляем стоп и тейк пропорционально? Не меняем, так как уровень TP/SL остаётся.
            pass
        
        self.last_trade_info = trade_info
        return pnl_usd

    def _check_sl_tp_intrabar(self, bar):
        if self.position == 0 or self.remaining_position_size <= 0:
            return None
        
        high = float(bar["High"])
        low = float(bar["Low"])
        
        # Проверка стоп-лосса (всегда полное закрытие)
        if self.position == 1 and low <= self.sl_price:
            return self._close_position("SL_HIT", self.sl_price, fraction=1.0)
        if self.position == -1 and high >= self.sl_price:
            return self._close_position("SL_HIT", self.sl_price, fraction=1.0)
        
        # Проверка тейк-профита с поддержкой частичного закрытия
        tp_hit = False
        if self.position == 1 and high >= self.tp_price:
            tp_hit = True
            exit_price = self.tp_price
        elif self.position == -1 and low <= self.tp_price:
            tp_hit = True
            exit_price = self.tp_price
        
        if tp_hit:
            if self.partial_close_enabled and len(self.tp_levels_hit) == 0:
                # Первый уровень TP – закрываем часть
                fraction = self.partial_close_ratio
                self.tp_levels_hit.append(1)
                pnl = self._close_position("TP_PARTIAL", exit_price, fraction=fraction)
                # После частичного закрытия оставшаяся позиция всё ещё открыта, стоп и тейк остаются
                return pnl
            else:
                # Полное закрытие по TP
                return self._close_position("TP_HIT", exit_price, fraction=1.0)
        return None

    def _get_auto_close_signal(self, bar):
        if self.position == 0:
            return None
        bull = bar["bull_div"] == 1
        bear = bar["bear_div"] == 1
        slope_div = bar["slope_div"]
        if self.position == 1 and bear and slope_div < -0.5:
            return "BEAR_DIV"
        if self.position == -1 and bull and slope_div < -0.5:
            return "BULL_DIV"
        return None

    def _check_stop_out(self, bar):
        if self.position == 0 or self.entry_price is None:
            return None
        current_equity = self._get_equity_at_price(bar["Close"])
        required_margin = (self.usd_per_pip / self.pip_value * self.entry_price) / self.leverage
        if current_equity < required_margin * 1.02:
            return self._close_position("STOP_OUT", bar["Close"], fraction=1.0)
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
                self._close_position("END_OF_DATA", last_close, fraction=1.0)
            obs = self._get_observation()
            info = {"equity_usd": self._get_current_equity()}
            if _GYMNASIUM:
                return obs, 0.0, True, False, info
            return obs, 0.0, True, info

        equity_start = self._get_equity_at_price(curr_bar["Close"])

        if self.pending_close_reason is not None:
            self._close_position(self.pending_close_reason, next_bar["Open"], fraction=1.0)
            self.pending_close_reason = None

        if action == 0:
            pass
        elif action == 1:
            if self.position != 0:
                self._close_position("MANUAL_CLOSE", next_bar["Open"], fraction=1.0)
        elif 2 <= action <= 6:
            if self.position == 0:
                lot_mult = action - 1
                self._open_position(1, next_bar, lot_mult)
        elif 7 <= action <= 11:
            if self.position == 0:
                lot_mult = action - 6
                self._open_position(-1, next_bar, lot_mult)

        if self.position != 0:
            self.time_in_trade += 1
            if self.time_penalty_pips != 0.0:
                self.equity_usd -= abs(self.time_penalty_pips) * self.usd_per_pip * self.remaining_position_size

        if self.position != 0:
            self._check_sl_tp_intrabar(next_bar)

        if self.position != 0:
            signal = self._get_auto_close_signal(next_bar)
            if signal is not None:
                self.pending_close_reason = signal

        if self.position != 0:
            self._check_stop_out(next_bar)

        equity_end = self._get_equity_at_price(next_bar["Close"])
        pnl_usd = equity_end - equity_start
        atr_pips = curr_bar["alma_atr"] / self.pip_value
        atr_usd = atr_pips * self.usd_per_pip if self.usd_per_pip > 0 else 1.0
        atr_usd = max(atr_usd, 1.0)
        raw_reward = pnl_usd / atr_usd

        shaping = 0.0
        if self.position != 0 and self.remaining_position_size > 0:
            if self.position == 1:
                shaping = -curr_bar["slope_div"] * self.slope_div_reward_scale
            else:
                shaping = curr_bar["slope_div"] * self.slope_div_reward_scale
            shaping = shaping * self.usd_per_pip / atr_usd * self.remaining_position_size
        raw_reward += shaping
        reward = raw_reward * self.reward_scale

        self.current_idx += 1
        self.next_idx += 1

        if self.next_idx >= self.n_steps:
            self.terminated = True
            if self.position != 0:
                last_bar = self.df.iloc[-1]
                self._close_position("END_OF_DATA", last_bar["Close"], fraction=1.0)

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