"""
backend/remote.py
=================
Talks to the FastAPI backend when RETENT_API_URL is set (e.g. RETENT_API_URL=http://127.0.0.1:8000).

With it set, Streamlit sends logging, scoring, search and LLM requests to the API, which owns MongoDB,
the models, FAISS and Ollama (the architecture in the report).  Without it - or if the API is not
reachable - everything runs in-process exactly as before, so nothing breaks.
"""
import os
import time

import requests

ERROR_MARK = "[[retent-error]]"                 # the API puts this before an error message inside a text stream
_state = {"checked": 0.0, "up": False}


class RemoteError(RuntimeError):
    pass


def api_url() -> str:
    if os.getenv("RETENT_IS_API_SERVER") == "1":          # the API process itself always works locally
        return ""
    return os.getenv("RETENT_API_URL", "").strip().rstrip("/")


def available() -> bool:
    """True if RETENT_API_URL is set AND /health answered recently (re-checked every 30 s)."""
    url = api_url()
    if not url:
        return False
    now = time.time()
    if now - _state["checked"] > 30:
        try:
            _state["up"] = requests.get(url + "/health", timeout=1.5).status_code == 200
        except Exception:
            _state["up"] = False
        _state["checked"] = now
    return _state["up"]


def reset():
    _state.update(checked=0.0, up=False)


def get(path: str, **params):
    r = requests.get(api_url() + path, params={k: v for k, v in params.items() if v not in (None, "")}, timeout=30)
    if r.status_code >= 400:
        raise RemoteError(f"{path}: {r.status_code} {r.text[:120]}")
    return r.json()


def post(path: str, payload: dict, timeout: int = 60):
    r = requests.post(api_url() + path, json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise RemoteError(f"{path}: {r.status_code} {r.text[:120]}")
    return r.json()


def stream_text(path: str, payload: dict):
    """Yield text as the API produces it.  Raises RuntimeError with the server's message on an in-stream error."""
    with requests.post(api_url() + path, json=payload, stream=True, timeout=(3, 240)) as r:
        if r.status_code >= 400:
            raise RemoteError(f"{path}: {r.status_code} {r.text[:120]}")
        r.encoding = "utf-8"
        pending = ""
        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
            data = pending + (chunk or "")
            if ERROR_MARK in data:
                before, _, message = data.partition(ERROR_MARK)
                if before:
                    yield before
                message += "".join(c or "" for c in r.iter_content(chunk_size=None, decode_unicode=True))
                raise RuntimeError(message.strip() or "The API reported an error.")
            keep = len(ERROR_MARK) - 1                       # a marker may be split across two chunks
            if len(data) > keep:
                yield data[:-keep]
                pending = data[-keep:]
            else:
                pending = data
        if pending:
            yield pending
