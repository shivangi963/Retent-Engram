"""
backend/sandbox.py
==================
Runs a student's Python snippet in a separate process with a time limit.

The first version never worked: it used `_blocked_import` before defining it and put a
list inside a set literal, so the wrapper crashed before running any student code.

HOW IT WORKS
  * the snippet runs in a fresh `python -I` process (isolated mode) with stdin closed
  * a wrapper replaces __import__ for the STUDENT'S code only (the standard library keeps
    importing what it needs, e.g. `random` imports `os` internally) and blocks risky modules
  * open/exec/eval/compile/input are removed from the student's builtins
  * hard timeout, output size cap

THIS IS A SOFT SANDBOX for a personal learning app.  A determined user can still escape it.
Never expose this page to the internet.
"""
import subprocess
import sys
import tempfile
import os
from difflib import SequenceMatcher

TIMEOUT_SECONDS = 10
MAX_OUTPUT_CHARS = 20000

BLOCKED_MODULES = [
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib", "ctypes", "multiprocessing",
    "threading", "signal", "resource", "pickle", "marshal", "builtins", "gc", "inspect",
    "requests", "http", "urllib", "ftplib", "smtplib", "telnetlib", "webbrowser", "asyncio", "tempfile",
]
REMOVED_BUILTINS = ["open", "exec", "eval", "compile", "input", "breakpoint", "exit", "quit", "help",
                    "memoryview", "__loader__", "__spec__"]

_WRAPPER = r'''
import builtins, sys, traceback

BLOCKED = set(%(blocked)r)
REMOVED = %(removed)r
real_import = builtins.__import__

def guard(name, globals=None, locals=None, fromlist=(), level=0):
    # Called by student code (globals is its namespace) or directly as __import__('os') (globals is None).
    # Imports made INSIDE library modules pass the library's own globals and are allowed.
    from_student = globals is None or globals.get("__name__") == "__student__"
    if from_student and level == 0 and name.split(".")[0] in BLOCKED:
        raise ImportError("importing '" + name.split(".")[0] + "' is not allowed here")
    return real_import(name, globals, locals, fromlist, level)

safe = {k: v for k, v in vars(builtins).items() if k not in REMOVED}
safe["__import__"] = guard
source = open(sys.argv[1], encoding="utf-8").read()
try:
    code = compile(source, "<student>", "exec")
    exec(code, {"__builtins__": safe, "__name__": "__student__"})
except SystemExit:
    pass
except BaseException as exc:
    line = None
    for fr in traceback.extract_tb(exc.__traceback__):
        if fr.filename == "<student>":
            line = fr.lineno
    if isinstance(exc, SyntaxError):
        line = exc.lineno
    msg = type(exc).__name__ + ": " + str(exc).split("(<student>")[0].strip()
    sys.stderr.write(msg + (" (line %%d)" %% line if line else "") + "\n")
    sys.exit(1)
'''


def build_wrapper() -> str:
    return _WRAPPER % {"blocked": BLOCKED_MODULES, "removed": REMOVED_BUILTINS}


def run_code(user_code: str) -> dict:
    """-> {'stdout','stderr','error','timed_out','success'}"""
    res = {"stdout": "", "stderr": "", "error": None, "timed_out": False, "success": False}
    if not user_code or not user_code.strip():
        res["error"] = "No code to run."
        return res

    files = []
    try:
        for suffix, content in ((".py", build_wrapper()), (".py", user_code)):
            fd, path = tempfile.mkstemp(suffix=suffix)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            files.append(path)
        proc = subprocess.run(
            [sys.executable, "-I", files[0], files[1]], capture_output=True, text=True,
            timeout=TIMEOUT_SECONDS, stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace")
        res["stdout"] = proc.stdout[:MAX_OUTPUT_CHARS]
        res["stderr"] = proc.stderr[:MAX_OUTPUT_CHARS]
        if proc.returncode != 0:
            res["error"] = (proc.stderr.strip().splitlines() or ["The program stopped with an error."])[-1]
        res["success"] = proc.returncode == 0
    except subprocess.TimeoutExpired:
        res.update(timed_out=True, error=f"Stopped after {TIMEOUT_SECONDS} seconds - check for an infinite loop.")
    except Exception as exc:
        res["error"] = f"Could not run the code: {exc}"
    finally:
        for path in files:
            try:
                os.unlink(path)
            except OSError:
                pass
    return res


def compare_output(actual: str, expected: str, strict: bool = False) -> dict:
    a, e = actual.strip(), expected.strip()
    if strict:
        passed = a == e
    else:
        passed = " ".join(a.lower().split()) == " ".join(e.lower().split())
    sim = SequenceMatcher(None, a, e).ratio()
    if passed:
        msg = "Output matches. Great work."
    elif sim > 0.8:
        msg = f"Very close ({sim:.0%} similar) - check spacing or rounding."
    elif sim > 0.5:
        msg = f"Partly right ({sim:.0%} similar) - review your logic."
    else:
        msg = f"Output does not match ({sim:.0%} similar)."
    return {"passed": passed, "similarity": round(sim, 3), "actual": a, "expected": e, "message": msg}
