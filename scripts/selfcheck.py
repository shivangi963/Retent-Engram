"""
scripts/selfcheck.py
====================
One command that checks the NEW parts on YOUR machine and tells you which work:

    python scripts/selfcheck.py

  PyTorch LSTM, survival models (own Weibull + lifelines Cox), XGBoost, the full model comparison,
  LangChain, the FastAPI app (routes AND the Streamlit-to-API path), MongoDB (local or Atlas), Ollama.

PASS = works, SKIP = optional package not installed (the app falls back and keeps working), FAIL = a bug
or broken install - copy the lines printed under it and send them over.
It never touches your study data: the API checks use a throw-away user "__selfcheck__" that is deleted afterwards.
"""
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SELF_USER = "__selfcheck__"
results = []


def check(title):
    def wrap(fn):
        t0 = time.time()
        try:
            status, detail = fn()
            extra = ""
        except Exception:
            status, detail = "FAIL", "raised an error"
            extra = "\n".join("        " + ln for ln in traceback.format_exc().strip().splitlines()[-6:])
        results.append(status)
        print(f" [{status}] {title} - {detail}  ({time.time() - t0:.1f}s)")
        if extra:
            print(extra)
        return fn
    return wrap


def _cleanup_user():
    try:
        from backend.db import get_collection
        for col in ("events", "recall_scores", "users", "generated_content"):
            get_collection(col).delete_many({"user_id": SELF_USER})
    except Exception:
        pass


print("\nRetent Engram self-check\n" + "-" * 60)


@check("Core packages")
def _core():
    import numpy, pandas, scipy, sklearn
    return "PASS", f"numpy {numpy.__version__}, pandas {pandas.__version__}, scikit-learn {sklearn.__version__}, scipy {scipy.__version__}"


@check("Survival model (own Weibull)")
def _weibull():
    import numpy as np
    from backend.ml.survival import WeibullSurvival
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 3))
    T = np.exp(4 + 0.5 * X[:, 0]) * rng.weibull(1.7, 3000)
    g = np.exp(rng.uniform(np.log(5), np.log(1500), 3000))
    m = WeibullSurvival().fit(X, g, (T <= g).astype(float))
    s = [float(m.survival(X[:1], t)[0]) for t in (1, 10, 100, 1000)]
    assert m.converged_ and all(s[i] >= s[i + 1] for i in range(3)), s
    return "PASS", "fits, converges, survival curve falls with time"


@check("PyTorch LSTM sequence model")
def _lstm():
    from backend.ml import seq_model as sm
    if not sm.torch_available():
        return "SKIP", "PyTorch not installed (pip install torch) - the LSTM row stays 'not run'"
    import numpy as np
    import torch
    from backend.ml.evaluate import GATE_HOURS
    from backend.ml.sequences import build_examples
    from backend.ml.synthetic_data import generate_synthetic_pairs
    df, seqs = build_examples(generate_synthetic_pairs(n_students=40))
    y, gap = df["label"].values, df["gap"].values.astype(float)
    cut = int(len(df) * 0.8)
    m = sm.SequenceRecallModel().fit(seqs[:cut], gap[:cut], y[:cut], val=(seqs[cut:], gap[cut:], y[cut:]), epochs=12)
    p = m.predict(seqs[cut:], gap[cut:])
    assert p.shape == (len(df) - cut,) and np.isfinite(p).all() and ((p > 0) & (p < 1)).all()
    assert min(m.history) < m.history[0], f"loss did not improve: {m.history[:3]}"
    curve = m.predict([seqs[cut]] * len(GATE_HOURS), np.array(GATE_HOURS, dtype=float))
    assert all(curve[i + 1] <= curve[i] + 1e-6 for i in range(len(curve) - 1)), curve
    with tempfile.TemporaryDirectory() as d:
        m.save(Path(d) / "m.pt")
        again = sm.SequenceRecallModel.load(Path(d) / "m.pt").predict(seqs[cut:cut + 20], gap[cut:cut + 20])
    assert np.allclose(again, p[:20], atol=1e-5)
    return "PASS", f"torch {torch.__version__}; loss {m.history[0]:.3f} -> {min(m.history):.3f}; recall falls with time; save/load identical"


@check("lifelines (Cox model)")
def _cox():
    try:
        import lifelines
    except ImportError:
        return "SKIP", "lifelines not installed (pip install lifelines) - Cox row stays 'not run'"
    import numpy as np
    from backend.ml.evaluate import prepare_data
    from backend.ml.survival import CoxSurvival
    d = prepare_data(30)
    m = CoxSurvival().fit(d.cov[d.tr_all], d.gap[d.tr_all], 1 - d.y[d.tr_all])
    s = m.survival(d.cov[d.te][:50], d.gap[d.te][:50])
    assert np.isfinite(s).all() and ((s >= 0) & (s <= 1)).all()
    return "PASS", f"lifelines {lifelines.__version__}; survival probabilities in [0,1]"


@check("XGBoost with monotonic constraints")
def _xgb():
    try:
        import xgboost as xgb
    except ImportError:
        return "SKIP", "xgboost not installed - scikit-learn gradient boosting is used instead"
    import numpy as np
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 6)); y = (X[:, 0] * -1 + rng.normal(size=600) > 0).astype(int)
    m = xgb.XGBClassifier(n_estimators=60, max_depth=3, monotone_constraints="(-1,0,1,1,0,0)", eval_metric="logloss",
                          early_stopping_rounds=10, n_jobs=1)
    m.fit(X[:450], y[:450], eval_set=[(X[450:], y[450:])], verbose=False)
    assert m.predict_proba(X[:5]).shape == (5, 2)
    return "PASS", f"xgboost {xgb.__version__}"


@check("Model comparison (all models, same learners)")
def _compare():
    from backend.ml import evaluate
    res = evaluate.evaluate_all(n_students=50)
    ok = [n for n, m in res["metrics"].items() if m.get("status") == "ok"]
    print("\n" + evaluate.format_table(res["metrics"], res["winner"]) + "\n")
    assert len(ok) >= 3 and res["winner"], f"only {ok} ran"
    return "PASS", f"{len(ok)} models trained and scored; winner would be '{res['winner']}'"


@check("LangChain RAG chain")
def _langchain():
    from rag import chain
    if not chain.available():
        return "SKIP", "langchain-core not installed (pip install langchain-core langchain-ollama) - direct Ollama path is used"
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    text = "".join(chain.stream_rag("cn", "Computer Networks", "flashcard", "beginner",
                                    llm=FakeListChatModel(responses=["hello from the chain"])))
    assert text == "hello from the chain", repr(text)
    from rag import generate as gen
    note = ""
    if gen.pick_model():
        chain._default_llm("flashcard")                  # builds ChatOllama: checks the import and parameter names
        note = "; ChatOllama created"
    return "PASS", "retriever -> prompt -> model -> parser streams correctly" + note


@check("FastAPI app (all routes)")
def _fastapi():
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        return "SKIP", "fastapi/httpx not installed (pip install fastapi uvicorn httpx)"
    from backend.api import app
    notes = []
    with TestClient(app) as client:
        h = client.get("/health"); assert h.status_code == 200, h.text
        health = h.json(); notes.append(f"mongodb={health['mongodb']}, ollama={health['ollama']}")
        assert client.get("/openapi.json").json()["paths"].get("/events"), "route /events missing"
        assert client.get("/docs").status_code == 200
        assert client.get("/evaluation").status_code == 200 and client.get("/models").status_code == 200
        assert client.get("/search", params={"q": "network"}).status_code == 200
        if health["mongodb"]:
            r = client.post("/events", json={"user_id": SELF_USER, "concept_id": "os", "score": 0.7, "source": "selfcheck"})
            assert r.status_code == 200 and r.json()["ok"], r.text
            rec = client.get(f"/recall/{SELF_USER}").json()["results"]
            assert rec and 0 <= rec[0]["recall_score"] <= 100, rec
            assert client.get(f"/queue/{SELF_USER}").status_code == 200
            assert client.post("/events", json={"user_id": SELF_USER, "concept_id": "os", "score": 7}).status_code == 422
            notes.append("event -> recall -> queue OK, bad input rejected")
        if health["ollama"] and health["llm"]:
            s = client.post("/llm/stream", json={"prompt": "Reply with the single word OK.", "kind": "answer"})
            assert s.status_code == 200 and s.text.strip(), s.text
            notes.append("LLM stream OK")
    _cleanup_user()
    return "PASS", "; ".join(notes)

def client_has_mongo() -> bool:
    try:
        from backend.db import get_collection
        get_collection("users").find_one({}, {"_id": 1})
        return True
    except Exception:
        return False
    
@check("Streamlit -> API path (real server)")
def _remote():
    try:
        import socket, threading
        import uvicorn
    except ImportError:
        return "SKIP", "uvicorn not installed (pip install uvicorn)"
    from backend.api import app
    from backend import remote
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    assert server.started, "server did not start"
    old, was_server = os.environ.get("RETENT_API_URL"), os.environ.pop("RETENT_IS_API_SERVER", None)
    os.environ["RETENT_API_URL"] = f"http://127.0.0.1:{port}"
    #remote.reset()
    remote.reset()
    try:
        assert remote.available()
        from backend.ml.pipeline import compute_scores_for_user
        from backend.study_log import log_study
        from rag import store
        from rag import generate as gen
        notes = []
        if client_has_mongo():
            res = log_study(SELF_USER, "os", 0.8, "quiz", source="selfcheck")
            assert res and res["ok"], res
            rows = compute_scores_for_user(SELF_USER)
            assert rows and rows[0]["concept_id"] == "os"
            notes.append("log + recall through the API")
        assert isinstance(store.search("network", k=2), list)
        notes.append("search")
        if gen.is_ollama_running() and gen.pick_model():
            assert "".join(gen.stream_ollama("Reply with the single word OK.", "answer")).strip()
            notes.append("LLM stream")
        return "PASS", ", ".join(notes) + f" (server on port {port})"
    finally:
        if old is None:
            os.environ.pop("RETENT_API_URL", None)
        else:
            os.environ["RETENT_API_URL"] = old
        if was_server:
            os.environ["RETENT_IS_API_SERVER"] = was_server
        remote.reset()
        server.should_exit = True
        _cleanup_user()




@check("MongoDB (local or Atlas)")
def _mongo():
    if not client_has_mongo():
        return "FAIL", "cannot reach MongoDB - start it, or check MONGO_URI in .env"
    uri = os.getenv("MONGO_URI", "")
    kind = "Atlas (cloud)" if ("mongodb+srv" in uri or "mongodb.net" in uri) else "local"
    return "PASS", f"connected - {kind}"


@check("Ollama (local LLM)")
def _ollama():
    from rag import generate as gen
    st = gen.ollama_status()
    if not st["running"]:
        return "SKIP", "Ollama is not running (ollama serve) - generation is unavailable, everything else works"
    gpu = f", {st['loaded'][0]['gpu_percent']}% on GPU" if st["loaded"] else ""
    return "PASS", f"model '{st['model']}'{gpu}" + (f"; {st['hint']}" if st["hint"] else "")


print("-" * 60)
p, f, s = results.count("PASS"), results.count("FAIL"), results.count("SKIP")
print(f"{p} passed, {f} failed, {s} skipped" + ("" if not f else "   <- send me the lines under each FAIL"))
sys.exit(1 if f else 0)