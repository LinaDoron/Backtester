# muscle/sandbox_check.py
"""
Самопроверка песочницы: python -m muscle.sandbox_check
Запускать из корня проекта, тем же Python, что и бота (из venv).
"""
import os
import sys
import time

from muscle import engine, sandbox

LEAK_SCRIPT = (
    "import os, socket\n"
    "env_path = os.path.join({root!r}, '.env')\n"
    "try:\n"
    "    open(env_path).read(); print('ENV:VISIBLE')\n"
    "except OSError:\n"
    "    print('ENV:HIDDEN')\n"
    "try:\n"
    "    socket.create_connection(('1.1.1.1', 53), timeout=3); print('NET:OPEN')\n"
    "except OSError:\n"
    "    print('NET:CLOSED')\n"
)


def _line(ok, text):
    print(f"  [{'OK' if ok else 'ПЛОХО'}] {text}")


def main():
    print(f"Система: {sys.platform}, Python: {sys.executable}")
    bwrap = sandbox.bwrap_path()
    if bwrap:
        print(f"bubblewrap: {bwrap}")
    elif os.name == 'nt':
        print("bubblewrap: нет (Windows) — файлы и сеть закрыты только проверкой кода")
    else:
        print("bubblewrap: НЕ НАЙДЕН — поставь: sudo apt install bubblewrap")

    engine.load_data_to_cache()
    config = {'deposit': 5000, 'risk_pct': 0.01, 'tp_ratio': 3.0}

    print("\n1. Нормальная стратегия")
    from brain.prompts import SYSTEM_PROMPT
    sample = SYSTEM_PROMPT.split('```python\n')[1].split('```')[0]
    t0 = time.time()
    deposit, trades, err = engine._run_in_sandbox(sample, config)
    _line(err is None, f"посчиталась за {time.time() - t0:.1f} c, сделок: {len(trades or [])}" + (f"\n{err}" if err else ""))

    print("\n2. Лимит памяти")
    hog = "def run_strategy(d, c):\n    x = [list(range(10 ** 6)) for i in range(200)]\n    return 0, []"
    _, _, err = engine._run_in_sandbox(hog, config)
    _line(bool(err) and 'памяти' in err, err or "стратегия съела память и не была остановлена")

    print("\n3. Файлы и сеть внутри песочницы (код запускается напрямую, в обход проверки кода)")
    if not bwrap:
        print("  пропущено: без bubblewrap этого слоя нет")
    else:
        script = LEAK_SCRIPT.format(root=sandbox.PROJECT_ROOT)
        out, err_bytes, code, _ = sandbox.run(b'', cmd=[sys.executable, '-c', script])
        text = out.decode('utf-8', errors='replace')
        if not text.strip():
            _line(False, "пузырь не запустился:\n" + err_bytes.decode('utf-8', errors='replace')[-400:])
        else:
            _line('ENV:HIDDEN' in text, ".env не виден" if 'ENV:HIDDEN' in text else ".env ВИДЕН")
            _line('NET:CLOSED' in text, "сети нет" if 'NET:CLOSED' in text else "сеть ЕСТЬ")


if __name__ == '__main__':
    main()
