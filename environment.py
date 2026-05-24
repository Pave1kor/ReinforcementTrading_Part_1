# environment.py
import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

class ForexTradingEnv(gym.Env):

    # Торговая среда для акций (SBER) с иерархическим Manager.
    # Все расчёты ведутся в долях начального капитала (equity).
    
    def __init__(self, df: pd.DataFrame, config: Dict[str, Any],
                 manager_model: Optional[Any] = None,
                 genetic_feature_cols: Optional[List[str]] = None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.config = config
        self.manager = manager_model
        self.genetic_cols = genetic_feature_cols or []

        # === Параметры данных ===
        self.pip_value = config['data']['pip_value']          # стоимость 1 пипса в рублях для 1 лота/акции

        # === Параметры среды ===
        env_cfg = config['environment']
        self.spread_pips = env_cfg['spread_pips']
        self.max_slippage_pips = env_cfg['max_slippage_pips']
        self.risk_per_trade = env_cfg['risk_per_trade']      # 0.003 (0.3% капитала на сделку)
        self.commission_pct = env_cfg['commission_percent']
        self.use_atr_sl = env_cfg.get('use_atr_sl', False)
        self.atr_mult_sl = env_cfg.get('atr_multiplier_sl', 2.0)
        self.atr_mult_tp = env_cfg.get('atr_multiplier_tp', [3.0, 5.0, 7.0, 9.0, 12.0])
        self.fixed_sl_pct = env_cfg.get('fixed_sl_percent', 0.02)
        self.fixed_tp_pcts = env_cfg.get('fixed_tp_percents', [0.02, 0.04, 0.06, 0.08, 0.10])
        self.partial_close_enabled = env_cfg.get('partial_close_enabled', False)
        self.partial_close_ratio = env_cfg.get('partial_close_ratio', 0.5)
        self.tp_levels = env_cfg.get('tp_levels', [0.5, 1.0])

        # === Reward shaping ===
        rew_cfg = env_cfg.get('reward', {})
        self.open_penalty_pips = rew_cfg.get('open_penalty_pips', 0.0)
        self.time_penalty_pips = rew_cfg.get('time_penalty_pips', 0.0)
        self.slope_div_scale = rew_cfg.get('slope_div_reward_scale', 0.0)
        self.reward_scale = rew_cfg.get('reward_scale', 1.0)

        # === Риск-менеджмент ===
        risk_cfg = config.get('risk', {})
        self.max_drawdown = risk_cfg.get('max_drawdown', 1.0)  # 0.3 = 30%

        # === Пространства ===
        obs_features = 7 + len(self.genetic_cols) + 5  # market + genetic + [pos, upnl, time_in, manager_risk, manager_dir]
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_features,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, (1,), dtype=np.float32)

        # === Внутреннее состояние ===
        self.position = 0.0          # доля капитала (положительная = лонг, отрицательная = шорт)
        self.entry_price = 0.0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.idx = 0
        self.steps_in_trade = 0

        # Manager state
        self._manager_risk = 0.0
        self._manager_dir = 0.0
        self.steps_since_manager = 10

        # Для хронологического теста
        self._start_idx = 0

    def reset(self, start_idx=None, seed=None, options=None):
        super().reset(seed=seed)
        if start_idx is not None:
            self._start_idx = start_idx
        else:
            self._start_idx = 0
        self.position = 0.0
        self.entry_price = 0.0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.idx = self._start_idx
        self.steps_in_trade = 0
        self.steps_since_manager = 10
        self._manager_risk = 0.0
        self._manager_dir = 0.0
        return self._get_obs(), {}

    def step(self, action):
        # 1. Обновление manager-сигнала каждые 10 баров
        if self.steps_since_manager >= 10 and self.manager is not None:
            man_obs = self._manager_obs()
            man_action, _ = self.manager.predict(man_obs, deterministic=True)
            self._manager_risk = float(np.clip(man_action[0], -1, 1))
            self._manager_dir = float(np.clip(man_action[1], -1, 1))
            self.steps_since_manager = 0

        # 2. Текущие цены и проскальзывание
        row = self.df.iloc[self.idx]
        price = row['Close']
        next_idx = min(self.idx + 1, len(self.df) - 1)
        next_row = self.df.iloc[next_idx]
        next_price = next_row['Close']

        # Моделируем проскальзывание (наихудший случай против нас)
        slippage = np.random.uniform(0, self.max_slippage_pips * self.pip_value / price)
        if action[0] > 0:   # покупка
            exec_price = next_price * (1 + slippage)
        else:
            exec_price = next_price * (1 - slippage)

        # 3. Расчёт текущих штрафов в долях капитала
        open_penalty = self.open_penalty_pips * self.pip_value / price
        time_penalty = self.time_penalty_pips * self.pip_value / price

        reward = 0.0
        done = False
        info = {}

        # 4. Обработка открытой позиции (стопы / тейки / частичное закрытие)
        if self.position != 0.0 and self.entry_price > 0:
            # Плавающий PnL в долях капитала
            if self.position > 0:
                upnl = self.position * (exec_price / self.entry_price - 1)
            else:
                upnl = self.position * (exec_price / self.entry_price - 1)   # для шорта формула та же

            # Расчёт стоп-лосса и тейк-профитов
            sl_price = self._calc_sl()
            tp_prices = self._calc_tp()

            sl_hit = False
            tp_hit_idx = -1
            if self.position > 0:
                if exec_price <= sl_price:
                    sl_hit = True
                for i, tp in enumerate(tp_prices):
                    if exec_price >= tp:
                        tp_hit_idx = i
                        break
            else:
                if exec_price >= sl_price:
                    sl_hit = True
                for i, tp in enumerate(tp_prices):
                    if exec_price <= tp:
                        tp_hit_idx = i
                        break

            if sl_hit:
                # Закрытие всей позиции по стопу
                close_pnl = self.position * (exec_price / self.entry_price - 1)
                cost = self._close_costs()
                reward = close_pnl - cost
                self.equity *= (1 + reward)
                self.position = 0.0
                self.entry_price = 0.0
                self.steps_in_trade = 0
            elif tp_hit_idx >= 0:
                # Частичное или полное закрытие по тейку
                if self.partial_close_enabled and tp_hit_idx < len(self.tp_levels):
                    close_frac = self.tp_levels[tp_hit_idx] * self.partial_close_ratio
                else:
                    close_frac = 1.0   # полный выход
                close_size = self.position * close_frac
                partial_pnl = close_size * (exec_price / self.entry_price - 1)
                cost = self._close_costs(close_size)
                reward = partial_pnl - cost
                self.equity *= (1 + reward)
                self.position -= close_size
                if abs(self.position) < 1e-6:
                    self.position = 0.0
                    self.entry_price = 0.0
                    self.steps_in_trade = 0
            else:
                # Удержание позиции
                reward -= time_penalty
                self.steps_in_trade += 1
                # Бонус за сильную дивергенцию в направлении позиции
                if self.slope_div_scale != 0:
                    reward += self.slope_div_scale * row['slope_div'] * np.sign(self.position)

            # Обновление пика эквити
            if self.equity > self.peak_equity:
                self.peak_equity = self.equity

        # 5. Действие агента: определение желаемой позиции с учётом риска
        raw_action = action[0]
        # Масштабируем на риск и manager
        desired_pos = raw_action * self._manager_dir * self._manager_risk
        # Ограничиваем максимальный размер позиции долей капитала risk_per_trade
        target_pos = np.clip(desired_pos * self.risk_per_trade, -self.risk_per_trade, self.risk_per_trade)

        # 6. Вход в позицию
        if self.position == 0.0 and abs(target_pos) > 1e-6:
            self.position = target_pos
            self.entry_price = exec_price
            entry_cost = self._entry_costs()
            reward -= entry_cost + open_penalty
            self.equity *= (1 - entry_cost)   # немедленный учёт издержек
            self.steps_in_trade = 0
        # 7. Выход из позиции
        elif self.position != 0.0 and abs(target_pos) < 1e-6:
            close_pnl = self.position * (exec_price / self.entry_price - 1)
            cost = self._close_costs()
            reward += close_pnl - cost
            self.equity *= (1 + close_pnl - cost)
            self.position = 0.0
            self.entry_price = 0.0
            self.steps_in_trade = 0

        # 8. Проверка максимальной просадки
        if self.equity < self.peak_equity * (1 - self.max_drawdown):
            done = True
            reward -= 1.0  # штраф за нарушение риск-менеджмента

        # 9. Глобальное масштабирование reward
        reward *= self.reward_scale

        # 10. Шаг времени и завершение эпизода
        self.idx += 1
        self.steps_since_manager += 1
        if self.idx >= len(self.df) - 1:
            done = True

        return self._get_obs(), reward, done, False, info

    def _calc_sl(self):
        if self.position == 0:
            return 0.0
        if self.use_atr_sl:
            atr = self.df.iloc[self.idx]['alma_atr']
            return self.entry_price - self.atr_mult_sl * atr if self.position > 0 else self.entry_price + self.atr_mult_sl * atr
        else:
            return self.entry_price * (1 - self.fixed_sl_pct) if self.position > 0 else self.entry_price * (1 + self.fixed_sl_pct)

    def _calc_tp(self):
        if self.position == 0:
            return []
        tps = []
        if self.use_atr_sl:
            atr = self.df.iloc[self.idx]['alma_atr']
            for mult in self.atr_mult_tp:
                tp = self.entry_price + mult * atr if self.position > 0 else self.entry_price - mult * atr
                tps.append(tp)
        else:
            for pct in self.fixed_tp_pcts:
                tp = self.entry_price * (1 + pct) if self.position > 0 else self.entry_price * (1 - pct)
                tps.append(tp)
        return tps

    def _entry_costs(self):
        # Спред + комиссия (только одна сторона) в долях капитала
        spread_cost = (self.spread_pips * self.pip_value) / self.entry_price * abs(self.position)
        comm = self.commission_pct * abs(self.position)
        return spread_cost + comm

    def _close_costs(self, size=None):
        if size is None:
            size = abs(self.position)
        price = self.df.iloc[self.idx]['Close']
        spread_cost = (self.spread_pips * self.pip_value) / price * size
        comm = self.commission_pct * size
        return spread_cost + comm

    def _get_obs(self):
        idx = min(self.idx, len(self.df)-1)
        row = self.df.iloc[idx]
        market = [
            row['bull_div'], row['bear_div'],
            row['weighted_div'], row['slope_div'],
            row['adx'], row['rsi_norm'], row['atr_norm']
        ]
        genetic = [row[col] for col in self.genetic_cols] if self.genetic_cols else []
        pos = self.position
        upnl = 0.0
        if self.position != 0.0:
            upnl = self.position * (row['Close'] / self.entry_price - 1)
        time_in = self.steps_in_trade / 100.0
        return np.array(market + genetic + [pos, upnl, time_in, self._manager_risk, self._manager_dir], dtype=np.float32)

    def _manager_obs(self):
        idx = min(self.idx, len(self.df)-1)
        row = self.df.iloc[idx]
        return np.array([row['adx'], row['atr_norm'], row['volume_zscore'],
                         row['rsi_norm'], row['macd_hist']], dtype=np.float32)