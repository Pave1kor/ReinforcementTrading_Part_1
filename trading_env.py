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
    """Торговая среда Forex с корректным временным порядком (без look-ahead)."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        window_size: int = 30,
        sl_options=None,
        tp_options=None,
        feature_columns=None,
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
        time_penalty_pips: float = 0.005,
        unrealized_reward_coef: float = 0.1,
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
            raise ValueError("Dataframe слишком короткий для заданного window_size.")

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
        self.unrealized_reward_coef = float(unrealized_reward_coef)

        self.random_start = bool(random_start)
        self.min_episode_steps = int(min_episode_steps)
        self.episode_max_steps = episode_max_steps if episode_max_steps is None else int(episode_max_steps)

        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.allow_flip = bool(allow_flip)

        # --- Действия ---
        self.action_map = [("HOLD", None, None, None), ("CLOSE", None, None, None)]
        for direction in [0, 1]:
            for sl in self.sl_options:
                for tp in self.tp_options:
                    self.action_map.append(("OPEN", direction, float(sl), float(tp)))
        self.action_space = spaces.Discrete(len(self.action_map))

        # Размерность наблюдения
        self.base_num_features = len(self.feature_columns)
        self.state_num_features = 5   # [позиция, время_в_сделке(норм), unrealized(норм), is_flat, entry_atr(норм)]
        self.num_features = self.base_num_features + self.state_num_features

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, self.num_features),
            dtype=np.float32
        )

        # Предварительная конвертация признаков в numpy (ускорение)
        self._features_array = self.df[self.feature_columns].values.astype(np.float32)
        self._open_prices = self.df["Open"].values.astype(np.float32)
        self._high_prices = self.df["High"].values.astype(np.float32)
        self._low_prices = self.df["Low"].values.astype(np.float32)
        self._close_prices = self.df["Close"].values.astype(np.float32)

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
        self._unrealized_pips_prev = 0.0

        self.initial_equity_usd = 10000.0
        self.equity_usd = self.initial_equity_usd
        self.equity_curve = []
        self.last_trade_info = None
        self._pending_action = None   # (act_type, direction, sl_pips, tp_pips)

    def _get_state_features(self):
        """Нормализованное состояние агента для добавления к наблюдению."""
        # позиция: -1,0,1 -> [-1,0,1] нормализации не требует
        pos = float(self.position)
        # время в сделке: делим на 200 (макс ожидаемое)
        t_norm = float(self.time_in_trade) / 200.0
        # нереализованный PnL (в пипсах) масштабируем к [-1,1] с помощью деления на 100
        unreal = self._compute_unrealized_pips(self._close_prices[self.current_step]) if self.position != 0 else 0.0
        unreal_scaled = np.clip(unreal / 100.0, -1.0, 1.0)
        is_flat = 1.0 if self.position == 0 else 0.0
        # entry_atr в пипсах / 20 (макс ожидаемый ATR в пипсах ~20)
        entry_atr_scaled = (self.entry_atr / self.pip_value) / 20.0 if self.entry_atr > 0 else 0.0
        return np.array([pos, t_norm, unreal_scaled, is_flat, entry_atr_scaled], dtype=np.float32)

    def _compute_unrealized_pips(self, price: float) -> float:
        if self.position == 0 or self.entry_price is None:
            return 0.0
        if self.position == 1:
            return (price - self.entry_price) / self.pip_value
        else:
            return (self.entry_price - price) / self.pip_value

    def _apply_optional_normalization(self, obs: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None:
            return obs
        mean = self.feature_mean.reshape(1, -1)
        std = self.feature_std.reshape(1, -1)
        std = np.where(std == 0, 1.0, std)
        return (obs - mean) / std

    def _get_observation(self):
        """Формирует окно признаков (сдвинутых) + состояние агента (только для последнего шага)."""
        start = self.current_step - self.window_size + 1
        if start < 0:
            start = 0
        # Берём срез из предварительно сконвертированных признаков
        base = self._features_array[start:self.current_step+1]
        if base.shape[0] < self.window_size:
            pad_rows = self.window_size - base.shape[0]
            pad = np.tile(base[0], (pad_rows, 1))
            base = np.vstack([pad, base])
        base = self._apply_optional_normalization(base)

        current_state = self._get_state_features()
        state_block = np.zeros((self.window_size, self.state_num_features), dtype=np.float32)
        state_block[-1, :] = current_state
        obs = np.hstack([base, state_block]).astype(np.float32)
        return obs

    def _sample_slippage_pips(self) -> float:
        if self.max_slippage_pips <= 0:
            return 0.0
        return float(np.random.uniform(0.0, self.max_slippage_pips))

    def _cost_pips_round_trip(self) -> float:
        return self.spread_pips + self.commission_pips

    def _open_position(self, direction: int, sl_pips: float, tp_pips: float, entry_price: float):
        """Открывает позицию по заданной цене (например, Open текущего бара)."""
        current_atr = self._features_array[self.current_step, self.feature_columns.index("alma_atr")]
        if current_atr <= 0:
            current_atr = 15.0 * self.pip_value

        slip_pips = self._sample_slippage_pips()
        slip_price = slip_pips * self.pip_value

        if direction == 1:  # long
            entry = entry_price + slip_price
            sl_price = entry - (sl_pips * current_atr)
            tp_price = entry + (tp_pips * current_atr)
            self.position = 1
        else:               # short
            entry = entry_price - slip_price
            sl_price = entry + (sl_pips * current_atr)
            tp_price = entry - (tp_pips * current_atr)
            self.position = -1

        self.entry_price = entry
        self.sl_price = sl_price
        self.tp_price = tp_price
        self.time_in_trade = 0
        self.entry_atr = current_atr
        self._unrealized_pips_prev = 0.0

        self.last_trade_info = {
            "event": "OPEN",
            "step": self.current_step,
            "position": self.position,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "atr_pips": float(current_atr / self.pip_value)
        }

    def _close_position(self, reason: str, exit_price: float) -> float:
        """Закрывает позицию, возвращает чистый PnL в пипсах."""
        if self.position == 1:
            pnl_price = exit_price - self.entry_price
        else:
            pnl_price = self.entry_price - exit_price
        realized_pips = pnl_price / self.pip_value
        cost_pips = self._cost_pips_round_trip()
        net_pips = realized_pips - cost_pips
        self.equity_usd += net_pips * self.usd_per_pip

        self.last_trade_info = {
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
        self._unrealized_pips_prev = 0.0
        return net_pips

    def _check_sl_tp_on_bar(self, bar_idx: int):
        """Проверяет срабатывание SL/TP на баре bar_idx (High/Low)."""
        if self.position == 0:
            return False, 0.0, ""
        high = self._high_prices[bar_idx]
        low = self._low_prices[bar_idx]
        if self.position == 1:
            if low <= self.sl_price:
                return True, self.sl_price, "SL_HIT"
            if high >= self.tp_price:
                return True, self.tp_price, "TP_HIT"
        else:
            if high >= self.sl_price:
                return True, self.sl_price, "SL_HIT"
            if low <= self.tp_price:
                return True, self.tp_price, "TP_HIT"
        return False, 0.0, ""

    def _execute_pending_action(self):
        """Исполняет отложенное действие по цене открытия текущего бара."""
        if self._pending_action is None:
            return 0.0
        act_type, direction, sl_pips, tp_pips = self._pending_action
        reward = 0.0
        current_open = self._open_prices[self.current_step]

        if act_type == "CLOSE" and self.position != 0:
            reward += self._close_position("MANUAL_CLOSE", current_open)
        elif act_type == "OPEN" and self.position == 0:
            self._open_position(direction, sl_pips, tp_pips, current_open)
            reward -= self.open_penalty_pips
        elif act_type == "OPEN" and self.allow_flip and self.position != 0:
            reward += self._close_position("FLIP_CLOSE", current_open)
            self._open_position(direction, sl_pips, tp_pips, current_open)
            reward -= self.open_penalty_pips
        self._pending_action = None
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()

        if self.random_start:
            max_start = self.n_steps - max(self.min_episode_steps, self.window_size) - 1
            low = self.window_size
            high = max(low + 1, max_start + 1)
            if high <= low:
                high = low + 1
            self.current_step = np.random.randint(low, high)
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
        info = {}

        # 1. Проверка SL/TP на текущем баре
        hit, exit_price, reason = self._check_sl_tp_on_bar(self.current_step)
        if hit:
            reward_pips += self._close_position(reason, exit_price)

        # 2. Исполнение отложенного действия (от предыдущего шага)
        reward_pips += self._execute_pending_action()

        # 3. Банкротство
        if self.equity_usd <= 0:
            self.terminated = True
            reward_pips -= 1000.0
            self.last_trade_info = {"event": "BANKRUPTCY", "equity_usd": self.equity_usd}
            obs = self._get_observation()
            reward = reward_pips * self.reward_scale
            if _GYMNASIUM:
                return obs, reward, True, False, info
            return obs, reward, True, info

        # 4. Промежуточное вознаграждение за нереализованный PnL
        if self.position != 0:
            current_close = self._close_prices[self.current_step]
            current_unreal = self._compute_unrealized_pips(current_close)
            delta = current_unreal - self._unrealized_pips_prev
            self._unrealized_pips_prev = current_unreal
            reward_pips += delta * self.unrealized_reward_coef
            # Штраф за время в позиции
            self.time_in_trade += 1
            reward_pips -= self.time_penalty_pips

        # 5. Запоминаем текущее действие как отложенное (будет исполнено на следующем баре)
        act_type, direction, sl_pips, tp_pips = self.action_map[int(action)]
        self._pending_action = (act_type, direction, sl_pips, tp_pips)

        # 6. Сохраняем эквити
        self.equity_curve.append(float(self.equity_usd))

        # 7. Переход к следующему бару
        self.current_step += 1

        # 8. Проверка конца данных
        if self.current_step >= self.n_steps:
            self.terminated = True
            if self.position != 0:
                last_close = self._close_prices[self.n_steps - 1]
                reward_pips += self._close_position("END_OF_DATA", last_close)

        if self.episode_max_steps is not None and self.steps_in_episode >= self.episode_max_steps:
            self.truncated = True

        obs = self._get_observation()
        reward = reward_pips * self.reward_scale

        info.update({
            "equity_usd": float(self.equity_usd),
            "position": int(self.position),
            "time_in_trade": int(self.time_in_trade),
            "reward_pips": float(reward_pips),
            "last_trade_info": self.last_trade_info
        })

        done = self.terminated or self.truncated
        if _GYMNASIUM:
            return obs, reward, self.terminated, self.truncated, info
        else:
            return obs, reward, done, info