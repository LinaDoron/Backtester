# muscle/sandbox.py
"""
Запуск процесса-исполнителя стратегии (muscle/sandbox_runner.py) с ограничениями ОС.

Windows: процесс кладется в Job Object с лимитом памяти и умирает вместе с заданием.
Linux: лимит памяти ставит сам исполнитель через setrlimit. Если установлен bubblewrap
(sudo apt install bubblewrap), исполнитель запускается в пузыре: без сети, видит только
системные библиотеки, Python, venv и код muscle/ — и все только на чтение. .env, котировок
на диске и остального проекта внутри нет: данные приходят через stdin.
"""
import os
import shutil
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUSCLE_DIR = os.path.join(PROJECT_ROOT, 'muscle')
TIMEOUT_SEC = 120
MEMORY_MB = 2048
# Переменные окружения, без которых Python не стартует. Токенов тут нет.
ENV_KEYS = ('SYSTEMROOT', 'SYSTEMDRIVE', 'WINDIR', 'TEMP', 'TMP', 'PATH', 'LANG')
RUNNER_CMD = [sys.executable, '-m', 'muscle.sandbox_runner']


class SandboxError(RuntimeError):
    pass


def sandbox_env():
    env = {key: os.environ[key] for key in ENV_KEYS if key in os.environ}
    env['PYTHONPATH'] = PROJECT_ROOT
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['SANDBOX_MEMORY_MB'] = str(MEMORY_MB)
    # numpy с многопоточным BLAS резервирует много адресного пространства под потоки.
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['OMP_NUM_THREADS'] = '1'
    return env


def bwrap_path():
    if not sys.platform.startswith('linux'):
        return None
    return shutil.which('bwrap')


def wrap_command(cmd):
    """На Linux с bubblewrap — команда внутри пузыря, иначе как есть."""
    bwrap = bwrap_path()
    if bwrap is None:
        return list(cmd)

    args = [
        bwrap, '--unshare-all', '--die-with-parent', '--new-session',
        '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp',
        '--ro-bind', '/usr', '/usr',
    ]
    for path in ('/bin', '/sbin', '/lib', '/lib32', '/lib64'):
        if os.path.islink(path):
            args += ['--symlink', os.readlink(path), path]
        elif os.path.isdir(path):
            args += ['--ro-bind', path, path]
    for path in ('/etc/ld.so.cache', '/etc/localtime', '/etc/alternatives'):
        if os.path.exists(path):
            args += ['--ro-bind', path, path]

    python_dirs = {
        sys.prefix, sys.exec_prefix, sys.base_prefix, sys.base_exec_prefix,
        os.path.dirname(os.path.realpath(sys.executable)),
    }
    for path in sorted(python_dirs):
        if path and os.path.isdir(path) and not path.startswith('/usr'):
            args += ['--ro-bind', path, path]

    args += ['--ro-bind', MUSCLE_DIR, MUSCLE_DIR, '--chdir', PROJECT_ROOT]
    return args + list(cmd)


def _assign_windows_job(proc):
    """Job Object с лимитом памяти на процесс. Закрытие задания убивает процесс."""
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            'ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
            'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount',
        )]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ('PerProcessUserTimeLimit', ctypes.c_int64),
            ('PerJobUserTimeLimit', ctypes.c_int64),
            ('LimitFlags', wintypes.DWORD),
            ('MinimumWorkingSetSize', ctypes.c_size_t),
            ('MaximumWorkingSetSize', ctypes.c_size_t),
            ('ActiveProcessLimit', wintypes.DWORD),
            ('Affinity', ctypes.c_size_t),
            ('PriorityClass', wintypes.DWORD),
            ('SchedulingClass', wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ('BasicLimitInformation', JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ('IoInfo', IO_COUNTERS),
            ('ProcessMemoryLimit', ctypes.c_size_t),
            ('JobMemoryLimit', ctypes.c_size_t),
            ('PeakProcessMemoryUsed', ctypes.c_size_t),
            ('PeakJobMemoryUsed', ctypes.c_size_t),
        ]

    job_object_extended_limit_information = 9
    limit_process_memory = 0x00000100
    limit_kill_on_job_close = 0x00002000

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    kernel32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise SandboxError(f"CreateJobObject: ошибка {ctypes.get_last_error()}")

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = limit_process_memory | limit_kill_on_job_close
    info.ProcessMemoryLimit = MEMORY_MB * 1024 * 1024
    ok = kernel32.SetInformationJobObject(
        job, job_object_extended_limit_information, ctypes.byref(info), ctypes.sizeof(info)
    )
    if ok:
        ok = kernel32.AssignProcessToJobObject(job, int(proc._handle))
    if not ok:
        error = ctypes.get_last_error()
        kernel32.CloseHandle(job)
        raise SandboxError(f"Лимит памяти не поставлен: ошибка Windows {error}")
    return lambda: kernel32.CloseHandle(job)


def run(payload: bytes, cmd=None):
    """
    Запускает исполнитель и отдает ему payload в stdin.
    Возвращает (stdout, stderr, returncode, timed_out).
    Исполнитель ничего не делает, пока не прочитает stdin, а stdin пишется уже
    после установки лимитов — поэтому код стратегии не стартует без них.
    """
    proc = subprocess.Popen(
        wrap_command(cmd or RUNNER_CMD),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=PROJECT_ROOT,
        env=sandbox_env(),
    )
    close_job = None
    try:
        if os.name == 'nt':
            close_job = _assign_windows_job(proc)
        try:
            out, err = proc.communicate(payload, timeout=TIMEOUT_SEC)
            return out, err, proc.returncode, False
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            return out, err, proc.returncode, True
    except BaseException:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
        raise
    finally:
        if close_job is not None:
            close_job()
