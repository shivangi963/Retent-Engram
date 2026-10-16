"""
backend/api.py
==============
The FastAPI backend of Retent Engram.

    python scripts/run_api.py          ->  http://127.0.0.1:8000        interactive docs: /docs
    set RETENT_API_URL=http://127.0.0.1:8000     (then start Streamlit: it now talks to this API)

  POST /events               log a study event            -> MongoDB, recall refresh, streak
  GET  /recall/{user}        recall %, priority, urgency  -> retention model
  GET  /queue/{user}         today's review queue         -> review scheduler
  GET  /search               nearest note passages        -> FAISS retrieval
  POST /llm/stream           stream text from the LLM     -> Ollama
  POST /generate             full RAG for one concept     -> retrieve + prompt + LLM + save
  POST /autogen/{user}       prepare practice now         -> scheduler -> RAG
  GET  /autogen/{user}/ready what was prepared automatically
  GET  /models, /evaluation  active model and measured comparison (Table 5.3)
  GET  /health               status of every component

This file only routes requests; the work happens in backend/service.py.  While the API runs, a
background worker prepares practice for weak concepts (scheduler -> RAG) every
RETENT_AUTOGEN_INTERVAL_MIN minutes (default 30; RETENT_AUTOGEN=0 disables it).
"""
import os

os.environ["RETENT_IS_API_SERVER"] = "1"     # this process works locally - it must never call itself
os.environ.pop("RETENT_API_URL", None)

import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend import autogen, service


class EventIn(BaseModel):
    user_id: str
    concept_id: str
    score: float = Field(..., ge=0.0, le=1.0, description="0 = forgot everything, 1 = perfect")
    event_type: str = "reading"
    minutes: float = Field(5, ge=0, le=1000)
    source: str = "api"
    doc_id: Optional[str] = None
    section_id: Optional[str] = None
    timestamp: Optional[str] = Field(None, description="ISO time; default = now")
    activity: Optional[str] = None
    note: Optional[str] = None


class LLMIn(BaseModel):
    prompt: str
    kind: str = "answer"


class GenerateIn(BaseModel):
    user_id: str
    concept_id: str
    content_type: str = "flashcard"
    force: bool = False


def _dump(model) -> dict:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()      # pydantic v2 / v1


def _autogen_worker(stop: threading.Event):
    minutes = float(os.getenv("RETENT_AUTOGEN_INTERVAL_MIN", "30"))
    while not stop.wait(minutes * 60):
        if autogen.enabled():
            try:
                autogen.run_for_all_users()
            except Exception as exc:
                print(f"  auto-generation pass failed: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    stop = threading.Event()
    threading.Thread(target=_autogen_worker, args=(stop,), daemon=True, name="autogen-worker").start()
    yield
    stop.set()


app = FastAPI(title="Retent Engram API", version="1.0", lifespan=lifespan,
              description="Logging, recall prediction, review scheduling, retrieval and generation for Retent Engram.")


@app.get("/health", tags=["system"])
def health():
    return service.health()


@app.post("/events", tags=["logging"])
def create_event(event: EventIn):
    res = service.log_event(_dump(event))
    if not res.get("ok"):
        raise HTTPException(status_code=503, detail="Could not save the event (is MongoDB running?)")
    return res


@app.get("/recall/{user_id}", tags=["retention model"])
def get_recall(user_id: str):
    return service.recall(user_id)


@app.get("/queue/{user_id}", tags=["scheduler"])
def get_queue(user_id: str):
    return service.queue(user_id)


@app.get("/search", tags=["retrieval"])
def search(q: str = Query(..., min_length=1), k: int = Query(4, ge=1, le=20),
           concept_ids: Optional[str] = None, doc_id: Optional[str] = None):
    ids = [c for c in (concept_ids or "").split(",") if c]
    return service.search(q, k, ids, doc_id)


@app.post("/llm/stream", tags=["generation"])
def llm_stream(body: LLMIn):
    return StreamingResponse(service.stream_llm(body.prompt, body.kind), media_type="text/plain; charset=utf-8")


@app.post("/generate", tags=["generation"])
def generate(body: GenerateIn):
    return service.generate(body.user_id, body.concept_id, body.content_type, body.force)


@app.post("/autogen/{user_id}", tags=["scheduler"])
def run_autogen(user_id: str):
    return service.autogen_run(user_id)


@app.get("/autogen/{user_id}/ready", tags=["scheduler"])
def autogen_ready(user_id: str):
    return service.autogen_ready(user_id)


@app.get("/models", tags=["ml"])
def models():
    return service.models()


@app.get("/evaluation", tags=["ml"])
def evaluation():
    return service.evaluation()
