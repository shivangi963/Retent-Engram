"""
Start the whole system in the architecture of the report:  Streamlit -> FastAPI -> MongoDB / models / FAISS / Ollama

    python scripts/start_all.py

Starts the API, waits until /health answers, then starts Streamlit with RETENT_API_URL pointing at it.
Ctrl+C stops both.  (MongoDB and `ollama serve` must already be running.)
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
PORT = int(os.getenv("RETENT_API_PORT", "8000"))
URL = f"http://127.0.0.1:{PORT}"


def main():
    env = dict(os.environ, RETENT_API_PORT=str(PORT))
    api = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "run_api.py")], cwd=ROOT, env=env)
    procs = [api]
    try:
        for _ in range(60):
            if api.poll() is not None:
                sys.exit("The API stopped at startup - run `python scripts/run_api.py` alone to see the error.")
            try:
                if requests.get(URL + "/health", timeout=1).status_code == 200:
                    break
            except Exception:
                time.sleep(1)
        else:
            sys.exit("The API did not answer on " + URL)
        print(f"\nAPI ready: {URL}   (interactive docs: {URL}/docs)\nStarting Streamlit (talking to the API) ...\n")
        ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "frontend" / "main.py")],
                              cwd=ROOT, env=dict(env, RETENT_API_URL=URL))
        procs.append(ui)
        ui.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()


if __name__ == "__main__":
    main()
