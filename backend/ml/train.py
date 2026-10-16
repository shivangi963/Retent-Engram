"""
backend/ml/train.py
===================
Trains and compares ALL candidate models (see evaluate.py), keeps only a model that passes the
sanity gate, and writes the reports used in the project report.

    python scripts/run_training.py

Outputs
    models/recall_model.pkl | recall_survival.json | recall_lstm.pt   the ACTIVE model (one of them)
    models/model_metadata.json    which model is active, metrics, gate results
    models/evaluation.json        every model's metrics
    docs/model_comparison.md      measured Table 5.3 (target vs measured)
    docs/model_comparison.png     measured Fig 5.3
If no model passes the gate no model file is written and the app uses the decay-formula
"memory model" (backend/ml/scorer.py), which also handles brand-new students.
"""
import contextlib
import json
import pickle
from datetime import datetime, timezone

from backend.config import MODELS_DIR, MODEL_PATH, MODEL_META_PATH, EVALUATION_PATH, DOCS_DIR, ROOT
from backend.ml import evaluate
from backend.ml.data_prep import FEATURE_COLUMNS


def artifact_path(kind: str):
    base = MODEL_PATH.parent
    return {"sklearn": MODEL_PATH, "weibull": base / "recall_survival.json", "lstm": base / "recall_lstm.pt"}[kind]


@contextlib.contextmanager
def _mlflow_run(run_name):
    """Yields the mlflow module with a run open, or None if MLflow is missing/misconfigured."""
    mlf = None
    try:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///" + (ROOT / "mlflow.db").as_posix())
        mlflow.set_experiment("retent-engram-recall-prediction")
        mlflow.start_run(run_name=run_name)
        mlf = mlflow
    except Exception as exc:
        print(f"  (MLflow logging skipped: {str(exc)[:80]})")
    try:
        yield mlf
    finally:
        if mlf is not None:
            try:
                mlf.end_run()
            except Exception:
                pass


def persist_winner(fitted) -> dict:
    """Write the winning model's file; returns {'artifact', 'artifact_type'}."""
    for kind in ("sklearn", "weibull", "lstm"):                 # a stale file from an older run must never be loaded
        artifact_path(kind).unlink(missing_ok=True)
    path = artifact_path(fitted.kind)
    if fitted.kind == "sklearn":
        with open(path, "wb") as f:
            pickle.dump(fitted.model, f)
    else:                                                       # weibull -> json, lstm -> torch checkpoint
        fitted.model.save(path)
    return {"artifact": path.name, "artifact_type": fitted.kind}


def run_training_pipeline(n_students: int = 150, skip=()) -> dict:
    print("\n" + "=" * 70 + "\n  RETENT ENGRAM - model training and comparison\n" + "=" * 70)
    res = evaluate.evaluate_all(n_students, skip)
    metrics, winner, fitted, data = res["metrics"], res["winner"], res["fitted"], res["data"]
    print("\n" + evaluate.format_table(metrics, winner))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    candidates = {}
    for name, m in metrics.items():
        if m.get("status") == "ok":
            candidates[name] = {"auc": round(m["auc"], 4), "brier": round(m["brier"], 4), "log_loss": round(m["log_loss"], 4),
                                "accuracy": round(m["accuracy"], 4), "gate_passed": m["gate"]["passed"],
                                "gate_problems": m["gate"]["problems"]}
        else:
            candidates[name] = {"status": "not run", "reason": m.get("reason", "")}
    meta = {"trained_at": datetime.now(timezone.utc).isoformat(), "feature_columns": FEATURE_COLUMNS,
            "n_rows": int(len(data.df)), "n_real_rows": int(data.n_real),
            "train_max_hours": float(data.df["gap"].max()), "monotone_constraints": evaluate.MONOTONE,
            "candidates": candidates,
            "note": "Metrics are on held-out learners, most of them SIMULATED; they show how well each model learned "
                    "the simulated forgetting pattern, not real-world accuracy."}
    if winner:
        meta.update(persist_winner(fitted[winner]))
        meta.update(active_model=winner, gate_passed=True, **{winner: candidates[winner]})
        m = metrics[winner]
        print(f"\n  ACTIVE MODEL: {winner}  (AUC {m['auc']:.3f}, Brier {m['brier']:.3f})")
    else:
        for kind in ("sklearn", "weibull", "lstm"):
            artifact_path(kind).unlink(missing_ok=True)
        meta.update(active_model="memory_model", gate_passed=False)
        print("\n  No model passed the gate -> the app will use the decay-formula memory model (scorer.py).")
    MODEL_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    summary = evaluate.write_reports(res, DOCS_DIR, EVALUATION_PATH)
    print(f"  Reports: {EVALUATION_PATH.name}, docs/model_comparison.md" + (", docs/model_comparison.png" if summary.get("figure") else ""))

    with _mlflow_run("train_" + datetime.now().strftime("%Y%m%d_%H%M%S")) as mlf:
        if mlf:
            try:
                for name, m in metrics.items():
                    if m.get("status") == "ok":
                        mlf.log_metrics({f"{name}_auc": m["auc"], f"{name}_brier": m["brier"]})
                mlf.log_params({"rows": len(data.df), "real_rows": data.n_real, "active_model": meta["active_model"]})
            except Exception as exc:
                print(f"  (MLflow logging skipped: {str(exc)[:80]})")
    print()
    return meta


if __name__ == "__main__":
    run_training_pipeline()
