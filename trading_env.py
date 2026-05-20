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
    """
    RL среда с динамическими SL/TP, управлением капиталом и устранённым look-ahead.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        window_size: int = 30,
        feature_columns=None,
        pip_value: float = 0.0001,
        spread_pips: float = 1.0,
        commission_pips: float = 0.0,
        max_slippage_pips: float = 0.5,
        lot_size: float = 100000.0,
        risk_per_trade: float = 0.01,
        leverage: float = 50.0,
        reward_scale: float = 0.01,
        random_start: bool = True,
        min_episode_steps: int = 300,
        episode_max_steps: int | None = None,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        # Параметры SL/TP (скорректированы)
        base_sl_pips: float = 35.0,          # ИЗМЕНЕНО: увеличено с 20
        base_tp_pips: float = 70.0,          # ИЗМЕНЕНО: увеличено с 40
        k_sl: float = 0.3,                   # ИЗМЕНЕНО: уменьшено с 0.5
        k_tp: float = 0.6,                   # ИЗМЕНЕНО: уменьшено с 0.8
        # Штрафы (скорректированы)
        open_penalty_pips: float = 0.5,      # ИЗМЕНЕНО: уменьшено с 1.0
        time_penalty_pips: float = 0.001,    # ИЗМЕНЕНО: уменьшено с 0.005
        trailing_trigger_pips: float = 30.0, # ИЗМЕНЕНО: добавлен трейлинг стоп
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
        self.trailing_trigger_pips = float(trailing_trigger_pips)
        self.random_start = bool(random_start)
        self.min_episode_steps = int(min_episode_steps)
        self.episode_max_steps = episode_max_steps if episode_max_steps is None else int(episode_max_steps)
        self.feature_mean = feature_mean
        self.feature_std = feature_std

        # Параметры SL/TP
        self.base_sl_pips = base_sl_pips
        self.base_tp_pips = base_tp_pips
        self.k_sl = k_sl
        self.k_tp = k_tp

        self.action_space = spaces.Discrete(4)  # 0:HOLD,1:OPEN_LONG,2:OPEN_SHORT,3:CLOSE

        self.base_num_features = len(self.feature_columns)
        self.state_num_features = 5   # позиция, время в сделке, unrealized, флаг flat, entry_slope
        self.num_features = self.base_num_features + self.state_num_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, self.num_features),
            dtype=np.float32
        )
        self._reset_state()

    def _reset_state(self):
        self.current_step = 0
        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False
        self.position = 0          # 1 long, -1 short
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.entry_slope = 0.0
        self.initial_equity_usd = 10000.0
        self.equity_usd = self.initial_equity_usd
        self.last_trade_info = None
        self.usd_per_pip = self.pip_value * self.lot_size
        # ИЗМЕНЕНО: новый атрибут для отложенного закрытия по дивергенции
        self.pending_close_reason = None

    def _get_state_features(self):
        pos = float(self.position)
        t_norm = float(self.time_in_trade) / 100.0
        unreal_pips = self._compute_unrealized_pips(use_prev_close=True) if self.position != 0 else 0.0
        unreal_scaled = unreal_pips / 50.0
        is_flat = 1.0 if self.position == 0 else 0.0
        entry_slope_scaled = self.entry_slope * 10000.0
        return np.array([pos, t_norm, unreal_scaled, is_flat, entry_slope_scaled], dtype=np.float32)

    def _compute_unrealized_pips(self, use_prev_close=True):
        if self.position == 0 or self.entry_price is None:
            return 0.0
        idx = self.current_step - 1 if use_prev_close else self.current_step
        idx = max(0, min(idx, len(self.df)-1))
        close = float(self.df.loc[idx, "Close"])
        if self.position == 1:
            return (close - self.entry_price) / self.pip_value
        else:
            return (self.entry_price - close) / self.pip_value

    def _apply_normalization(self, obs):
        if self.feature_mean is None or self.feature_std is None:
            return obs
        mean = self.feature_mean.reshape(1, -1)
        std = self.feature_std.reshape(1, -1)
        std = np.where(std == 0, 1.0, std)
        return (obs - mean) / std

    def _get_observation(self):
        start = max(0, self.current_step - self.window_size)
        indices = list(range(start, self.current_step))
        if not indices:
            indices = [0] * self.window_size
        obs_df = self.df.iloc[indices]
        if len(obs_df) < self.window_size:
            pad = self.window_size - len(obs_df)
            pad_df = pd.DataFrame(0, index=range(pad), columns=obs_df.columns)
            obs_df = pd.concat([pad_df, obs_df])
        base = obs_df[self.feature_columns].values.astype(np.float32)
        base = self._apply_normalization(base)

        # ИЗМЕНЕНО: состояние повторяется на всех шагах окна
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
        # ИЗМЕНЕНО: новая формула с учётом reference ATR и разумными пределами
        base_sl = self.base_sl_pips
        base_tp = self.base_tp_pips
        vol_factor = atr_pips / reference_atr
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        abs_slope = min(abs(slope_val), 1.0)
        sl_mult = 1 + self.k_sl * abs_slope
        tp_mult = 1 + self.k_tp * abs_slope
        sl_pips = base_sl * sl_mult * vol_factor
        tp_pips = base_tp * tp_mult * vol_factor
        sl_pips = np.clip(sl_pips, 8.0, 80.0)
        tp_pips = max(tp_pips, sl_pips * 1.2)   # TP не менее 120% SL
        return sl_pips, tp_pips

    def _check_margin(self, lot_size, entry_price):
        required_margin = (lot_size * entry_price) / self.leverage
        return self.equity_usd > required_margin * 1.05

    def _open_position(self, direction: int, step_index: int):
        if self.position != 0:
            return False

        close_price = float(self.df.loc[step_index, "Close"])
        if step_index + 1 < len(self.df):
            entry_base_price = float(self.df.loc[step_index + 1, "Open"])
        else:
            entry_base_price = close_price

        current_atr = float(self.df.loc[step_index, "alma_atr"]) / self.pip_value
        if current_atr <= 1e-6:
            current_atr = 15.0
        slope_val = float(self.df.loc[step_index, "slope_div"])

        sl_pips, tp_pips = self._compute_sl_tp(direction, slope_val, current_atr)

        if self.risk_per_trade > 0:
            risk_amount = self.equity_usd * self.risk_per_trade
            lot_size = risk_amount / (sl_pips * self.pip_value)
            lot_size = np.clip(lot_size, 1000.0, 100000.0)
            max_lot_by_margin = (self.equity_usd * self.leverage) / (entry_base_price * 1.2)
            lot_size = min(lot_size, max_lot_by_margin)
            lot_size = max(lot_size, 1000.0)
        else:
            lot_size = self.lot_size
        self.usd_per_pip = self.pip_value * lot_size

        if not self._check_margin(lot_size, entry_base_price):
            return False

        slip_pips = self._sample_slippage_pips()
        slip_price = slip_pips * self.pip_value

        if direction == 1:  # long
            entry = entry_base_price + slip_price
            sl_price = entry - sl_pips * self.pip_value
            tp_price = entry + tp_pips * self.pip_value
            self.position = 1
        else:               # short
            entry = entry_base_price - slip_price
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
            "step": self.current_step,
            "position": self.position,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "slope_div": slope_val,
            "atr_pips": current_atr,
            "lot_size": lot_size
        }
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
            "step": self.current_step,
            "position": self.position,
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "realized_pips": realized_pips,
            "net_pips": net_pips,
            "equity_usd": self.equity_usd,
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

    def _check_sl_tp_intrabar(self):
        if self.position == 0:
            return None
        high = float(self.df.loc[self.current_step, "High"])
        low = float(self.df.loc[self.current_step, "Low"])
        # ИЗМЕНЕНО: добавлен трейлинг стоп
        if self.position == 1:
            # Трейлинг: если цена превысила entry + trailing_trigger_pips, подтягиваем SL к entry
            if high >= self.entry_price + self.trailing_trigger_pips * self.pip_value:
                new_sl = self.entry_price
                if new_sl > self.sl_price:
                    self.sl_price = new_sl
            if low <= self.sl_price:
                return self._close_position("SL_HIT", self.sl_price)
            if high >= self.tp_price:
                return self._close_position("TP_HIT", self.tp_price)
        else:
            if low <= self.entry_price - self.trailing_trigger_pips * self.pip_value:
                new_sl = self.entry_price
                if new_sl < self.sl_price:
                    self.sl_price = new_sl
            if high >= self.sl_price:
                return self._close_position("SL_HIT", self.sl_price)
            if low <= self.tp_price:
                return self._close_position("TP_HIT", self.tp_price)
        return None

    def _get_auto_close_signal(self):
        """
        Возвращает причину для закрытия на основе дивергенции на ПРЕДЫДУЩЕМ баре.
        ИЗМЕНЕНО: добавлена проверка силы slope_div.
        """
        if self.position == 0:
            return None
        prev_idx = self.current_step - 1
        if prev_idx < 0:
            return None
        prev_bull = self.df.loc[prev_idx, "bull_div"] == 1
        prev_bear = self.df.loc[prev_idx, "bear_div"] == 1
        prev_slope_div = self.df.loc[prev_idx, "slope_div"]
        if self.position == 1 and prev_bear and prev_slope_div < -0.1:
            return "BEAR_DIV"
        if self.position == -1 and prev_bull and prev_slope_div > 0.1:
            return "BULL_DIV"
        return None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        if self.random_start:
            max_start = self.n_steps - max(self.min_episode_steps, self.window_size) - 2
            self.current_step = np.random.randint(self.window_size, max(self.window_size + 1, max_start))
        else:
            self.current_step = self.window_size
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
        reward_pips = 0.0

        # ИЗМЕНЕНО: сначала обрабатываем отложенное закрытие по дивергенции
        if self.pending_close_reason is not None:
            close_price = float(self.df.loc[self.current_step, "Open"])
            reward_pips += self._close_position(self.pending_close_reason, close_price)
            self.pending_close_reason = None

        # --- Действие агента (используем данные предыдущего бара для открытия) ---
        if action == 1:   # OPEN_LONG
            if self.position == 0:
                has_signal = (self.df.loc[self.current_step-1, "bull_div"] == 1)
                if self._open_position(1, self.current_step - 1):
                    reward_pips -= self.open_penalty_pips
                    if not has_signal:
                        reward_pips -= 0.3   # меньший штраф
        elif action == 2: # OPEN_SHORT
            if self.position == 0:
                has_signal = (self.df.loc[self.current_step-1, "bear_div"] == 1)
                if self._open_position(-1, self.current_step - 1):
                    reward_pips -= self.open_penalty_pips
                    if not has_signal:
                        reward_pips -= 0.3
        elif action == 3: # CLOSE
            if self.position != 0:
                close_price = float(self.df.loc[self.current_step, "Close"])
                reward_pips += self._close_position("MANUAL_CLOSE", close_price)

        # --- Внутрибаровая проверка SL/TP (включая трейлинг) ---
        if self.position != 0:
            self.time_in_trade += 1
            reward_pips -= self.time_penalty_pips
            sl_tp_reward = self._check_sl_tp_intrabar()
            if sl_tp_reward is not None:
                reward_pips += sl_tp_reward

        # --- Бонус за удержание прибыльной позиции (нововведение) ---
        if self.position != 0:
            unreal_pips = self._compute_unrealized_pips(use_prev_close=False)
            if unreal_pips > 5:
                reward_pips += 0.01 * unreal_pips

        # --- Запоминаем сигнал для следующего шага (но не закрываем сейчас) ---
        if self.position != 0:
            signal = self._get_auto_close_signal()
            if signal is not None:
                self.pending_close_reason = signal

        # --- Штраф за нереализованную просадку ---
        if self.position != 0:
            unreal_pips = self._compute_unrealized_pips(use_prev_close=False)
            unreal_loss_usd = unreal_pips * self.usd_per_pip
            if unreal_loss_usd < -0.01 * self.equity_usd:  # просадка > 1%
                reward_pips -= 0.5

        # --- Запись эквити ---
        info = {"equity_usd": float(self.equity_usd)}

        # --- Переход к следующему бару ---
        self.current_step += 1

        # --- Проверка окончания данных ---
        if self.current_step >= self.n_steps - 1:
            self.terminated = True
            if self.position != 0:
                close_price = float(self.df.loc[self.n_steps - 1, "Close"])
                reward_pips += self._close_position("END_OF_DATA", close_price)
                info["equity_usd"] = float(self.equity_usd)

        if self.episode_max_steps is not None and self.steps_in_episode >= self.episode_max_steps:
            self.truncated = True

        obs = self._get_observation()
        reward = float(reward_pips) * self.reward_scale
        info["last_trade_info"] = self.last_trade_info

        if _GYMNASIUM:
            return obs, reward, self.terminated, self.truncated, info
        else:
            done = self.terminated or self.truncated
            return obs, reward, done, info