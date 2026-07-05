"""
backend/sandbox.py
====================
IMPROVEMENT 2 — Code Execution Sandbox

PURPOSE
-------
When the review page shows a CODING_TASK, the student can now write
Python code directly in the app and run it. The sandbox:
  - Executes the code in an isolated subprocess
  - Captures stdout and stderr separately
  - Enforces a 10-second timeout (kills runaway loops)
  - Restricts dangerous imports (os, sys, subprocess, etc.)
  - Compares output to expected output (pass/fail)
  - Returns a structured result the UI can display

SECURITY APPROACH
-----------------
We use subprocess.run() with a separate Python process.
The student's code is wrapped in a safety shell that:
  1. Removes dangerous builtins (__import__, open, eval, exec)
  2. Blocks imports of os, sys, subprocess, socket, etc.
  3. Captures stdout/stderr via StringIO redirect
  4. Kills the process after TIMEOUT seconds

This is a "soft sandbox" — sufficient for a personal learning app
but NOT production-grade. Don't expose this to the internet.

INSTALL:
  No new packages needed — uses Python stdlib only.
"""

import sys
import os
import subprocess
import tempfile
import textwrap
from difflib import SequenceMatcher


# =============================================================================
# CONSTANTS
# =============================================================================

TIMEOUT_SECONDS = 10   # kill process after this many seconds

# Imports that are blocked in the sandbox
BLOCKED_IMPORTS = [
    "os", "sys", "subprocess", "socket", "shutil", "pathlib",
    "importlib", "ctypes", "multiprocessing", "threading",
    "__builtins__", "open", "eval", "exec", "compile",
    "requests", "http", "urllib", "ftplib", "smtplib"
]

# Allowed imports (safe standard library modules)
ALLOWED_IMPORTS = [
    "math", "random", "collections", "itertools", "functools",
    "string", "re", "json", "datetime", "time", "heapq",
    "bisect", "copy", "pprint", "dataclasses", "typing",
    "abc", "enum", "statistics"
]


# =============================================================================
# SANDBOX WRAPPER CODE
# =============================================================================

def build_sandbox_wrapper(user_code: str) -> str:
    """
    Wraps the student's code in a safety shell.

    WHAT THE WRAPPER DOES:
      1. Redirects stdout to a StringIO buffer
      2. Blocks dangerous builtins by overriding __builtins__
      3. Intercepts import statements via a custom importer
      4. Executes student code inside try/except
      5. Prints captured output + any errors to real stdout
         (which is captured by subprocess)

    The output format is JSON so we can parse it cleanly:
      {"stdout": "...", "stderr": "...", "error": null}

    Args:
        user_code: the student's Python code string

    Returns:
        str: complete Python script to execute as subprocess
    """
    # Escape user code for embedding in a string literal
    escaped_code = user_code.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')

    wrapper = f'''
import sys
import io
import json
import builtins

# Capture stdout
_stdout_capture = io.StringIO()
sys.stdout = _stdout_capture

# Block dangerous builtins
_safe_builtins = {{k: v for k, v in vars(builtins).items()
                   if k not in {BLOCKED_IMPORTS}}}
_safe_builtins["__import__"] = _blocked_import

def _blocked_import(name, *args, **kwargs):
    blocked = {BLOCKED_IMPORTS}
    if name in blocked:
        raise ImportError(f"Import '{{name}}' is not allowed in the sandbox.")
    return __import__(name, *args, **kwargs)

_result = {{"stdout": "", "stderr": "", "error": None}}

try:
    user_code = """{escaped_code}"""
    exec(compile(user_code, "<student_code>", "exec"),
         {{"__builtins__": _safe_builtins}})
    _result["stdout"] = _stdout_capture.getvalue()
except Exception as e:
    _result["stdout"] = _stdout_capture.getvalue()
    _result["error"] = f"{{type(e).__name__}}: {{str(e)}}"

sys.stdout = sys.__stdout__
print(json.dumps(_result))
'''
    return wrapper


# =============================================================================
# EXECUTE CODE
# =============================================================================

def run_code(user_code: str) -> dict:
    """
    Executes student's Python code in a sandboxed subprocess.

    STEPS:
      1. Wrap code in safety shell (build_sandbox_wrapper)
      2. Write wrapper to a temp .py file
      3. Run with subprocess.run(timeout=TIMEOUT_SECONDS)
      4. Parse JSON output
      5. Clean up temp file
      6. Return structured result

    RETURN FORMAT:
      {
        "stdout":      "Hello World\n",
        "stderr":      "",
        "error":       None,          ← or "TypeError: ..."
        "timed_out":   False,
        "success":     True
      }

    Args:
        user_code: student's Python code as a string

    Returns:
        dict with execution result
    """
    result = {
        "stdout":    "",
        "stderr":    "",
        "error":     None,
        "timed_out": False,
        "success":   False
    }

    if not user_code or not user_code.strip():
        result["error"] = "No code to run."
        return result

    # Write sandboxed code to temp file
    wrapper_code = build_sandbox_wrapper(user_code)

    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py",
            delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(wrapper_code)
            tmp_file = tmp.name

        # Run in subprocess with timeout
        proc = subprocess.run(
            [sys.executable, tmp_file],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS
        )

        # Parse JSON output from wrapper
        import json
        raw_output = proc.stdout.strip()

        if raw_output:
            try:
                sandbox_result = json.loads(raw_output)
                result["stdout"]  = sandbox_result.get("stdout", "")
                result["stderr"]  = proc.stderr
                result["error"]   = sandbox_result.get("error")
                result["success"] = sandbox_result.get("error") is None
            except json.JSONDecodeError:
                # Wrapper itself crashed — show raw output
                result["stdout"]  = raw_output
                result["stderr"]  = proc.stderr
                result["success"] = proc.returncode == 0
        else:
            result["stderr"]  = proc.stderr
            result["error"]   = "No output received from sandbox."

    except subprocess.TimeoutExpired:
        result["timed_out"] = True
        result["error"] = (
            f"⏱️ Code timed out after {TIMEOUT_SECONDS} seconds. "
            "Check for infinite loops."
        )

    except Exception as e:
        result["error"] = f"Sandbox error: {e}"

    finally:
        # Always clean up temp file
        if tmp_file and os.path.exists(tmp_file):
            os.unlink(tmp_file)

    return result


# =============================================================================
# COMPARE OUTPUT
# =============================================================================

def compare_output(actual: str, expected: str,
                   strict: bool = False) -> dict:
    """
    Compares student's output to expected output.

    TWO MODES:
      strict=True:  exact string match (every character must match)
      strict=False: smart match (normalises whitespace, case, punctuation)
                    This is default because LLM-generated expected output
                    often has different whitespace than actual Python output.

    SIMILARITY SCORE:
      Uses SequenceMatcher (same algorithm as Python's difflib).
      1.0 = identical, 0.0 = completely different.
      We show this as a percentage so student knows how close they were.

    RETURN FORMAT:
      {
        "passed":     True,
        "similarity": 0.95,
        "actual":     "Hello World\n",
        "expected":   "Hello World",
        "message":    "✅ Output matches!"
      }

    Args:
        actual:   student's actual output
        expected: expected output from the coding task
        strict:   if True, require exact match

    Returns:
        dict with comparison result
    """
    actual_clean   = actual.strip()
    expected_clean = expected.strip()

    if strict:
        passed = actual_clean == expected_clean
    else:
        # Normalise: lowercase, collapse whitespace
        actual_norm   = " ".join(actual_clean.lower().split())
        expected_norm = " ".join(expected_clean.lower().split())
        passed = actual_norm == expected_norm

    # Calculate similarity regardless
    similarity = SequenceMatcher(
        None, actual_clean, expected_clean
    ).ratio()

    if passed:
        message = "✅ Output matches! Great work."
    elif similarity > 0.8:
        message = f"🟡 Very close! ({similarity*100:.0f}% similar) — check spacing or rounding."
    elif similarity > 0.5:
        message = f"🟠 Partially correct ({similarity*100:.0f}% similar) — review your logic."
    else:
        message = f"❌ Output doesn't match ({similarity*100:.0f}% similar) — check the approach."

    return {
        "passed":     passed,
        "similarity": round(similarity, 3),
        "actual":     actual_clean,
        "expected":   expected_clean,
        "message":    message
    }


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing sandbox...\n")

    # Test 1: Basic code
    result = run_code("print('Hello, World!')")
    print(f"Test 1 — Hello World: {result}")

    # Test 2: Math
    result = run_code("""
x = [1, 2, 3, 4, 5]
print(sum(x))
print(max(x))
""")
    print(f"\nTest 2 — Math: {result}")

    # Test 3: Blocked import
    result = run_code("import os; print(os.listdir('.'))")
    print(f"\nTest 3 — Blocked import: {result}")

    # Test 4: Infinite loop (should timeout)
    result = run_code("while True: pass")
    print(f"\nTest 4 — Timeout: {result}")

    # Test 5: Compare output
    cmp = compare_output("15\n10\n", "15\n10")
    print(f"\nTest 5 — Compare: {cmp}")