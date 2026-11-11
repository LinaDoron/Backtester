# muscle/engine.py
import ast
import json
import os
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # бэктест идет в фоновом потоке, окна не нужны
import matplotlib.pyplot as plt
from muscle import sandbox
from muscle.time_theory import load_and_prep_data

TRADE_RESULTS = ('Win', 'Loss', 'Closed')

ALLOWED_MODULE = 'muscle.smc_alphabet'
FORBIDDEN_NAMES = {
    'eval', 'exec', 'compile', 'open', '__import__', 'getattr', 'setattr', 'delattr',
    'globals', 'locals', 'vars', 'dir', 'input', 'breakpoint', 'help', 'exit', 'quit',
    'memoryview', 'type', 'object', 'super', 'classmethod', 'staticmethod', 'property',
}
# Методы pandas и numpy, которые читают или пишут файлы, ходят в сеть или исполняют строки.
FORBIDDEN_ATTRS = {
    'eval', 'query', 'pipe', 'style', 'plot', 'hist', 'boxplot',
    'tofile', 'dump', 'dumps', 'save', 'load', 'format', 'format_map',
}
FORBIDDEN_ATTR_PREFIXES = ('_', 'to_', 'read_')

# ГЛОБАЛЬНЫЙ КЕШ ДАННЫХ
# Кабал загрузит котировки в оперативную память ОДИН РАЗ при запуске бота.
DATA_CACHE = {}

def load_data_to_cache():
    if sandbox.bwrap_path():
        print("Песочница: bubblewrap, стратегии без сети и без доступа к файлам проекта.")
    elif os.name != 'nt':
        print("ВНИМАНИЕ: bubblewrap не найден, стратегии видят диск и сеть. Поставь: sudo apt install bubblewrap")
    print("Бот разминает мышцы: загрузка котировок в RAM...")
    DATA_CACHE['EUR'] = load_and_prep_data("data/csv/eurusd_m5.csv")
    DATA_CACHE['GBP'] = load_and_prep_data("data/csv/gbpusd_m5.csv")
    DATA_CACHE['CHF'] = load_and_prep_data("data/csv/usdchf_m5.csv")
    print("База данных готова к бэктестам!")

def validate_strategy_shape(code_str: str):
    """
    Не даем Дипсику прогонять историю вложенными циклами.
    Такой код жжет процессор и пишет одни и те же сделки много раз.
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as exc:
        return f"Код не разбирается: {exc.msg}"

    chain_calls = 0
    loops = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, 'attr', '')
            if name == 'find_smt_chain':
                chain_calls += 1
        elif isinstance(node, (ast.For, ast.While)):
            loops.append(node)

    if any(isinstance(node, ast.While) for node in loops):
        return "Цикл while нельзя. Один проход по find_smt_chain."
    if chain_calls > 1:
        return (
            "find_smt_chain можно вызвать только один раз. "
            "Связку уровней передавай списком, например ['true_date', 'macro_q']."
        )
    for node in loops:
        body = node.body + getattr(node, 'orelse', [])
        for child in body:
            if any(isinstance(inner, (ast.For, ast.While)) for inner in ast.walk(child)):
                return "Цикл внутри цикла нельзя: на каждом шаге история считается заново."
    if len(loops) > 1:
        return "Нужен один цикл по цепочкам SMT, без второго прохода по истории."
    return None

def validate_strategy_safety(code_str: str):
    """
    Белый список для кода от LLM: один импорт из smc_alphabet, без доступа к файлам,
    сети, служебным атрибутам и строкам, которые исполняются как код.
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as exc:
        return f"Код не разбирается: {exc.msg}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            return "Импорт запрещен. Разрешен только from muscle.smc_alphabet import *."
        if isinstance(node, ast.ImportFrom) and (node.module != ALLOWED_MODULE or node.level):
            return f"Импорт запрещен: {node.module}. Разрешен только from muscle.smc_alphabet import *."
        if isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith('__')):
            return f"Имя запрещено: {node.id}"
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr in FORBIDDEN_ATTRS or attr.startswith(FORBIDDEN_ATTR_PREFIXES):
                return f"Атрибут запрещен: .{attr}"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name.startswith('__'):
            return f"Имя запрещено: {node.name}"
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await)):
            return "Global, nonlocal и async в стратегии не нужны."
    return None


def _run_in_sandbox(code_str: str, config: dict):
    """
    Запускает стратегию в отдельном процессе: без токенов, с таймаутом и лимитом памяти,
    на Linux с bubblewrap — еще и без сети и без доступа к файлам проекта.
    Возвращает (deposit, trades, None) или (None, None, текст ошибки).
    """
    payload = pickle.dumps({'code': code_str, 'config': config, 'data': DATA_CACHE})
    try:
        stdout, stderr, returncode, timed_out = sandbox.run(payload)
    except (OSError, sandbox.SandboxError) as exc:
        return None, None, f"Песочница не запустилась: {exc}"
    if timed_out:
        return None, None, f"Стратегия считалась дольше {sandbox.TIMEOUT_SEC} секунд и была остановлена."

    lines = stdout.decode('utf-8', errors='replace').strip().splitlines()
    if not lines:
        stderr = stderr.decode('utf-8', errors='replace')
        if 'bwrap' in stderr:
            return None, None, f"bubblewrap установлен, но не запускается:\n{stderr[-400:]}"
        return None, None, f"Процесс стратегии упал (код {returncode}):\n{stderr[-400:]}"
    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None, None, "Процесс стратегии вернул не JSON."
    if not isinstance(result, dict):
        return None, None, "Процесс стратегии вернул не словарь."
    if result.get('error'):
        return None, None, str(result['error'])

    deposit, trades = result.get('deposit'), result.get('trades')
    if not isinstance(deposit, (int, float)) or not isinstance(trades, list):
        return None, None, "run_strategy должна вернуть (deposit, trades)."
    for trade in trades:
        if not isinstance(trade, dict) or trade.get('result') not in TRADE_RESULTS \
                or not isinstance(trade.get('balance_change'), (int, float)):
            return None, None, "В trades должны быть только сделки Win, Loss и Closed из simulate_trade_with_rm."
    return float(deposit), trades, None


def validate_config(config):
    """
    Проверка на адекватность параметров перед запуском.
    """
    if not isinstance(config, dict):
        return "config должен быть словарем."
    for key in ('deposit', 'risk_pct', 'tp_ratio'):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{key} должен быть числом."
    if config.get('risk_pct', 0) > 0.10: # Риск более 10% - это самоубийство
        return "Риск слишком высокий (>10%)!"
    if config.get('tp_ratio', 0) <= 0:
        return "Тейк-профит не может быть меньше или равен 0!"
    if config.get('deposit', 0) <= 0:
        return "Депозит должен быть больше 0!"
    return None

def run_backtest(code_str: str, config: dict):
    """
    Проверяет код стратегии, запускает его в отдельном процессе и анализирует результат.
    """

    validation_error = validate_config(config)
    if validation_error:
        return {"error": f"Настройка не прошла стресс-тест: {validation_error}"}

    if not isinstance(code_str, str):
        return {"error": "Код остановлен до запуска:\ncode должен быть строкой."}
    safety_error = validate_strategy_safety(code_str)
    if safety_error:
        print(f"КОД ОТКЛОНЕН: {safety_error}\n{code_str}")
        return {"error": f"Код остановлен до запуска:\n{safety_error}"}

    shape_error = validate_strategy_shape(code_str)
    if shape_error:
        return {"error": f"Код остановлен до запуска:\n{shape_error}"}

    os.makedirs("temp", exist_ok=True)
    plot_path = "temp/equity.png"

    final_deposit, trades, run_error = _run_in_sandbox(code_str, config)
    if run_error:
        print(f"КРАШ ЛОГ:\n{run_error}")
        return {"error": f"Ошибка выполнения Питона:\n{run_error[-400:]}"}

    # 3. АНАЛИЗ РЕЗУЛЬТАТОВ
    if not trades:
        return {"error": "Сделок не найдено. Условия слишком жесткие."}

    # --- НОВЫЙ БЛОК: ЛОГИРОВАНИЕ ---
    trades_df = pd.DataFrame(trades)
    log_path = "temp/trades_log.csv"
    trades_df.to_csv(log_path, index=False)
    print(f"\n[АХТУНГ] Лог сделок сохранен в {log_path}")
    print(trades_df.head(10)) # Печатаем первые 10 сделок в консоль
    # -------------------------------

    wins = [t for t in trades if t['result'] == 'Win']
    losses = [t for t in trades if t['result'] == 'Loss']
    closed = [t for t in trades if t['result'] == 'Closed']

    winrate = len(wins) / len(trades) * 100
    total_pnl = final_deposit - config['deposit']

    # 4. РИСУЕМ ГРАФИК EQUITY
    equity = [config['deposit']]
    current_eq = config['deposit']
    for t in trades:
        current_eq += t['balance_change']
        equity.append(current_eq)

        # Расчет Max Drawdown
    peak = equity[0]
    mdd = 0
    for val in equity:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0
        if dd > mdd:
            mdd = dd

    # Стилистика графика "Темная тема"
    plt.figure(figsize=(10, 5))
    plt.plot(equity, color='#00ffcc', linewidth=2) # Кибер-зеленый
    plt.title('Equity Curve (HFT Backtester)', color='white', pad=20)
    plt.gca().set_facecolor('#121212')
    plt.gcf().patch.set_facecolor('#121212')
    plt.tick_params(colors='white')
    plt.grid(color='#333333', linestyle='--', linewidth=0.5)

    # Сохраняем картинку
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    # Возвращаем красивый отчет
    return {
        "error": None,
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "closed": len(closed),
        "winrate": winrate,
        "final_deposit": final_deposit,
        "total_pnl": total_pnl,
        "mdd": mdd * 100, # В процентах
        "plot_path": plot_path
    }
