# features.py
import random
import operator
import warnings
import numpy as np
import pandas as pd
from typing import Tuple, List
from deap import base, creator, tools, gp
import pandas_ta as ta

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 1. ИНДИКАТОРЫ (основная функция, работает с сырым DataFrame)
# ----------------------------------------------------------------------
def load_and_preprocess_data(csv_path: str, return_features_only: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    # Загрузка из CSV и полный расчёт индикаторов (используется только для быстрого старта)."""
    df = pd.read_csv(csv_path, parse_dates=["begin"], dayfirst=True)
    df.columns = df.columns.str.strip().str.lower()
    required = {"begin", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В CSV отсутствуют колонки: {missing}")

    df = df.set_index("begin").sort_index()
    df.index = pd.to_datetime(df.index)
    df = df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    })
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return _compute_indicators(df, return_features_only)

def load_and_preprocess_data_from_raw(df: pd.DataFrame, return_features_only: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    # Расчёт индикаторов из уже загруженного DataFrame (с колонками Open, High, Low, Close, Volume)."""
    return _compute_indicators(df.copy(), return_features_only)

def _compute_indicators(df: pd.DataFrame, return_features_only: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    # Общая логика расчёта индикаторов."""
    df["pressure"] = (df["Close"] - df["Low"]) - (df["High"] - df["Close"])
    df["bar_range"] = df["High"] - df["Low"]
    df["norm_pressure"] = df["pressure"] / (df["bar_range"] + 1e-8)
    df["delta"] = df["norm_pressure"] * df["Volume"]

    df["cvd_avg"] = ta.alma(df["delta"], length=50, sigma=0.85, distribution_offset=4)
    df["price_avg"] = ta.alma(df["Close"], length=34, sigma=0.85, distribution_offset=4)
    df["cvd_slope"] = ta.linreg(df["cvd_avg"], length=8, slope=True)
    df["price_slope"] = ta.linreg(df["price_avg"], length=8, slope=True)

    tr = ta.true_range(df["High"], df["Low"], df["Close"])
    df["alma_atr"] = ta.alma(tr, length=300, sigma=0.85, distribution_offset=4)
    df["alma_vol"] = ta.alma(df["Volume"], length=300, sigma=0.85, distribution_offset=4)

    df["price_slope"] = df["price_slope"] / (df["alma_atr"] + 1e-8)
    df["cvd_slope"] = df["cvd_slope"] / (df["alma_vol"] + 1e-8)

    df["slope_div"] = df["price_slope"] * df["cvd_slope"] * 10000.0
    df["bull_div"] = ((df["price_slope"] < 0) & (df["cvd_slope"] > 0)).astype(float)
    df["bear_div"] = ((df["price_slope"] > 0) & (df["cvd_slope"] < 0)).astype(float)
    df["weighted_div"] = np.where(
        (df["bull_div"] == 1) | (df["bear_div"] == 1),
        df["price_slope"].abs() * df["cvd_slope"].abs(),
        0.0
    )

    df["rsi_norm"] = ta.rsi(df["Close"], length=14) / 100.0
    df["price_dist_from_avg"] = (df["Close"] - df["price_avg"]) / (df["alma_atr"] + 1e-8)

    df["return_10"] = df["Close"].pct_change(10)
    df["norm_return_10"] = df["return_10"] / (df["alma_atr"] / df["Close"] + 1e-8)

    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    df["macd_hist"] = macd["MACDh_12_26_9"] / (df["alma_atr"] / df["Close"] + 1e-8)

    adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    df["adx"] = adx_df["ADX_14"] / 100.0

    rm = df["Volume"].rolling(100)
    df["volume_zscore"] = (df["Volume"] - rm.mean()) / (rm.std() + 1e-8)
    df["volume_zscore"] = df["volume_zscore"].clip(-3, 3)

    df["atr_norm"] = df["alma_atr"] / (df["alma_atr"].rolling(100).mean() + 1e-8)
    df["div_persistence"] = df["weighted_div"].rolling(5).mean()

    df.dropna(inplace=True)

    feature_names = [
        "cvd_avg", "price_avg", "cvd_slope", "price_slope",
        "slope_div", "bull_div", "bear_div", "weighted_div",
        "rsi_norm", "price_dist_from_avg",
        "norm_return_10", "macd_hist", "adx",
        "volume_zscore", "atr_norm", "div_persistence"
    ]

    if return_features_only:
        return df[feature_names].copy(), feature_names
    else:
        return df, feature_names


# ----------------------------------------------------------------------
# 2. ГЕНЕТИЧЕСКОЕ ПРОГРАММИРОВАНИЕ
# ----------------------------------------------------------------------
def protected_div(left, right):
    return left / right if abs(right) > 1e-8 else 1.0

def if_then_else(cond, tval, fval):
    return tval if cond > 0 else fval

def threshold(x, t):
    return 1.0 if x > t else 0.0

def and_(a, b):
    return 1.0 if (a > 0 and b > 0) else 0.0

def or_(a, b):
    return 1.0 if (a > 0 or b > 0) else 0.0

def gt(a, b):
    return 1.0 if a > b else 0.0

def lt(a, b):
    return 1.0 if a < b else 0.0

def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        exp_x = np.exp(x)
        return exp_x / (1.0 + exp_x)

def abs_(x):
    return abs(x)

def clamp(x, low=-1.0, high=1.0):
    return max(low, min(high, x))

def rand_05():
    return random.uniform(-0.5, 0.5)

def neg(x):
    return -x

# Примитивные множества
pset0 = gp.PrimitiveSetTyped("LEVEL0", [float, float, float, float], float)
pset0.addPrimitive(operator.add, [float, float], float)
pset0.addPrimitive(operator.sub, [float, float], float)
pset0.addPrimitive(operator.mul, [float, float], float)
pset0.addPrimitive(protected_div, [float, float], float)
pset0.addPrimitive(neg, [float], float)
pset0.addPrimitive(and_, [float, float], float)
pset0.addPrimitive(or_, [float, float], float)
pset0.addPrimitive(if_then_else, [float, float, float], float)
pset0.addPrimitive(threshold, [float, float], float)
pset0.addPrimitive(gt, [float, float], float)
pset0.addPrimitive(lt, [float, float], float)
pset0.addPrimitive(sigmoid, [float], float)
pset0.addPrimitive(abs_, [float], float)
pset0.addEphemeralConstant("rand_05", rand_05, float)
pset0.renameArguments(ARG0='bull_div', ARG1='bear_div', ARG2='weighted_div', ARG3='slope_div')

pset1 = gp.PrimitiveSetTyped("LEVEL1", [float, float, float, float], float)
pset1.addPrimitive(operator.add, [float, float], float)
pset1.addPrimitive(operator.sub, [float, float], float)
pset1.addPrimitive(operator.mul, [float, float], float)
pset1.addPrimitive(protected_div, [float, float], float)
pset1.addPrimitive(sigmoid, [float], float)
pset1.addPrimitive(threshold, [float, float], float)
pset1.addPrimitive(if_then_else, [float, float, float], float)
pset1.addEphemeralConstant("rand_05", rand_05, float)
pset1.renameArguments(ARG0='adx', ARG1='macd_hist', ARG2='rsi_norm', ARG3='price_dist_from_avg')

pset2 = gp.PrimitiveSetTyped("LEVEL2", [float, float, float], float)
pset2.addPrimitive(operator.add, [float, float], float)
pset2.addPrimitive(operator.sub, [float, float], float)
pset2.addPrimitive(operator.mul, [float, float], float)
pset2.addPrimitive(protected_div, [float, float], float)
pset2.addPrimitive(clamp, [float], float)
pset2.addPrimitive(sigmoid, [float], float)
pset2.addEphemeralConstant("rand_05", rand_05, float)
pset2.renameArguments(ARG0='volume_zscore', ARG1='cvd_slope', ARG2='atr_norm')

pset3 = gp.PrimitiveSetTyped("LEVEL3", [float, float, float], float)
pset3.addPrimitive(operator.add, [float, float], float)
pset3.addPrimitive(operator.sub, [float, float], float)
pset3.addPrimitive(operator.mul, [float, float], float)
pset3.addPrimitive(protected_div, [float, float], float)
pset3.addPrimitive(sigmoid, [float], float)
pset3.addPrimitive(if_then_else, [float, float, float], float)
pset3.addPrimitive(clamp, [float], float)
pset3.addEphemeralConstant("rand_05", rand_05, float)
pset3.renameArguments(ARG0='out0', ARG1='out1', ARG2='out2')

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

def setup_toolbox(pset, min_depth=1, max_depth=4):
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=min_depth, max_=max_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    return toolbox

toolbox0 = setup_toolbox(pset0)
toolbox1 = setup_toolbox(pset1)
toolbox2 = setup_toolbox(pset2)
toolbox3 = setup_toolbox(pset3)

def compile_vector(tree, pset):
    func = gp.compile(tree, pset)
    def vector_func(*args):
        return np.array([func(*vals) for vals in zip(*args)])
    return vector_func

def compute_sharpe(pnl: np.ndarray, periods_per_year: int = 252 * 39) -> float:
    if len(pnl) < 2 or np.std(pnl) < 1e-12:
        return 0.0
    return np.mean(pnl) / (np.std(pnl) + 1e-12) * np.sqrt(periods_per_year)

def evaluate_individuals_vectorized(individual, df_slice):
    tree0, tree1, tree2, tree3 = individual
    f0 = compile_vector(tree0, pset0)
    f1 = compile_vector(tree1, pset1)
    f2 = compile_vector(tree2, pset2)
    f3 = compile_vector(tree3, pset3)

    o0 = f0(df_slice['bull_div'].values, df_slice['bear_div'].values,
            df_slice['weighted_div'].values, df_slice['slope_div'].values)
    o1 = f1(df_slice['adx'].values, df_slice['macd_hist'].values,
            df_slice['rsi_norm'].values, df_slice['price_dist_from_avg'].values)
    o2 = f2(df_slice['volume_zscore'].values, df_slice['cvd_slope'].values,
            df_slice['atr_norm'].values)
    o3 = f3(o0, o1, o2)

    returns = df_slice['Close'].pct_change().shift(-1).values
    mask = ~np.isnan(returns)
    returns = returns[mask]
    signal = o3[mask]

    pnl = np.sign(signal) * returns
    return compute_sharpe(pnl),

def tournament_select(population, k=1, tournsize=3):
    selected = []
    for _ in range(k):
        aspirants = random.sample(population, tournsize)
        winner = max(aspirants, key=lambda ind: ind[0].fitness.values[0])
        selected.append(winner)
    return selected

def init_population(pop_size):
    pop = []
    for _ in range(pop_size):
        ind = (toolbox0.individual(), toolbox1.individual(), toolbox2.individual(), toolbox3.individual())
        pop.append(ind)
    return pop

def train_genetic_on_split(train_df, val_df, generations=20, pop_size=100):
    # Обучение GP с явным хронологическим разделением train/val."""
    return _train_genetic(train_df, val_df, generations, pop_size)

def _train_genetic(train_df, val_df, generations, pop_size):
    # Внутренняя функция обучения GP."""
    population = init_population(pop_size)

    for gen in range(generations):
        fitness = [evaluate_individuals_vectorized(ind, val_df) for ind in population]

        for ind, fit in zip(population, fitness):
            for t in ind:
                t.fitness.values = fit

        population = sorted(population, key=lambda x: x[0].fitness.values[0], reverse=True)
        elite = population[:10]
        new_pop = elite.copy()

        toolboxes = [toolbox0, toolbox1, toolbox2, toolbox3]
        while len(new_pop) < pop_size:
            p1 = tournament_select(population, k=1, tournsize=3)[0]
            p2 = tournament_select(population, k=1, tournsize=3)[0]

            child = [tb.clone(tree) for tb, tree in zip(toolboxes, p1)]

            for i, (tb, c) in enumerate(zip(toolboxes, child)):
                if random.random() < 0.7:
                    c, _ = tb.mate(c, p2[i])
                    child[i] = c
                if random.random() < 0.2:
                    c, = tb.mutate(c)
                    child[i] = c
                del c.fitness.values

            new_pop.append(tuple(child))

        population = new_pop
        print(f"Gen {gen:2d} best Sharpe: {population[0][0].fitness.values[0]:.4f}")

    return population[0]

def load_genetic_trees(t0, t1, t2, t3):
    with open(t0, 'r') as f: tree0 = gp.PrimitiveTree.from_string(f.read().strip(), pset0)
    with open(t1, 'r') as f: tree1 = gp.PrimitiveTree.from_string(f.read().strip(), pset1)
    with open(t2, 'r') as f: tree2 = gp.PrimitiveTree.from_string(f.read().strip(), pset2)
    with open(t3, 'r') as f: tree3 = gp.PrimitiveTree.from_string(f.read().strip(), pset3)
    return tree0, tree1, tree2, tree3

def add_genetic_features_vectorized(df, tree0, tree1, tree2, tree3):
    # Векторизованное добавление генетических признаков (без .apply)."""
    f0 = compile_vector(tree0, pset0)
    f1 = compile_vector(tree1, pset1)
    f2 = compile_vector(tree2, pset2)
    f3 = compile_vector(tree3, pset3)

    o0 = f0(df['bull_div'].values, df['bear_div'].values,
            df['weighted_div'].values, df['slope_div'].values)
    o1 = f1(df['adx'].values, df['macd_hist'].values,
            df['rsi_norm'].values, df['price_dist_from_avg'].values)
    o2 = f2(df['volume_zscore'].values, df['cvd_slope'].values,
            df['atr_norm'].values)
    o3 = f3(o0, o1, o2)

    df['gen_feat0'] = o0
    df['gen_feat1'] = o1
    df['gen_feat2'] = o2
    df['gen_feat3'] = o3
    return df