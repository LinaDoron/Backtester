# muscle/sandbox_runner.py
"""
Отдельный процесс для кода стратегии от LLM.
Бот передает в stdin pickle {'code', 'config', 'data'} и ждет в stdout одну строку JSON:
{'deposit': ..., 'trades': [...]} или {'error': '...'}.
Ответ — JSON, а не pickle: разбор pickle от недоверенного кода выполнил бы его в процессе бота.
"""
import builtins
import json
import os
import pickle
import sys
import traceback

import muscle.smc_alphabet  # noqa: F401  — грузим заранее, чтобы импорт из стратегии шел из кеша

ALLOWED_MODULE = 'muscle.smc_alphabet'

SAFE_BUILTIN_NAMES = (
    'abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'filter', 'float', 'int',
    'isinstance', 'len', 'list', 'map', 'max', 'min', 'print', 'range', 'reversed',
    'round', 'set', 'sorted', 'str', 'sum', 'tuple', 'zip',
    'Exception', 'ValueError', 'KeyError', 'IndexError', 'TypeError', 'ZeroDivisionError',
    'True', 'False', 'None',
)


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name == ALLOWED_MODULE:
        return builtins.__import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Импорт запрещен: {name}")


def _jsonable(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _limit_memory():
    """Linux: лимит адресного пространства. На Windows лимит ставит бот через Job Object."""
    memory_mb = os.environ.get('SANDBOX_MEMORY_MB')
    if not memory_mb:
        return
    try:
        import resource
    except ImportError:
        return
    limit = int(memory_mb) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def main():
    _limit_memory()
    out = sys.stdout
    payload = pickle.load(sys.stdin.buffer)
    # print из стратегии уходит в stderr и не ломает ответ в stdout.
    sys.stdout = sys.stderr
    try:
        safe_builtins = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES}
        safe_builtins['__import__'] = _safe_import
        namespace = {'__builtins__': safe_builtins, '__name__': 'strategy'}
        exec(compile(payload['code'], 'strategy', 'exec'), namespace)
        run_strategy = namespace.get('run_strategy')
        if run_strategy is None:
            raise NameError("В коде нет функции run_strategy")
        deposit, trades = run_strategy(payload['data'], payload['config'])
        result = {'deposit': deposit, 'trades': list(trades)}
    except MemoryError:
        result = {'error': f"Стратегия превысила лимит памяти {os.environ.get('SANDBOX_MEMORY_MB', '?')} МБ и была остановлена."}
    except Exception:
        result = {'error': traceback.format_exc()}
    out.write(json.dumps(result, default=_jsonable, ensure_ascii=False) + '\n')
    out.flush()


if __name__ == '__main__':
    main()
