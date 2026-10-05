"""
backend/ml/train.py
===================
Trains Logistic Regression (baseline) and a boosted-tree model (XGBoost) and only
PROMOTES a model to models/recall_model.pkl if it passes a sanity gate.

What changed and why
--------------------
1. Features are timed correctly (see features.py).
2. Train/test split is by LEARNER (GroupShuffleSplit), so a student's rows never sit
   in both sets.  The report (Sec 4.8) promises per-learner splits.
3. XGBoost gets monotonic constraints: recall can never RISE with more hours since the
   last review, and never FALL with a better last/average score.  This is domain
   knowledge written into the model.
4. SANITY GATE - a model is used by the app only if
      - recall falls as the gap grows (checked on several typical learners), and
      - it clearly separates recalled/forgotten on held-out learners (AUC >= 0.70).
   If nothing passes, no model file is written and the app uses the memory model
   (backend/ml/scorer.py).  The dashboard can no longer show 65% for a topic you
   last touched 114 days ago.
5. MLflow is optional - training never fails because of it.
"""
import contextlib
import json
import pickle
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.config import MODELS_DIR, MODEL_PATH, MODEL_META_PATH, ROOT
from backend.ml.data_prep import (
    build_training_dataset_from_mongodb, FEATURE_COLUMNS, LABEL_COLUMN, GROUP_COLUMN,
)
from backend.ml.synthetic_data import get_combined_training_data

MIN_AUC = 0.70
# +1: higher feature -> higher recall.  -1: higher feature -> lower recall.
MONOTONE = {"hours_since_last": -1, "avg_score": +1, "last_score": +1}

GATE_HOURS = [1, 6, 24, 72, 168, 336, 720, 1440]
GATE_LEARNERS = [   # typical learners (everything except hours_since_last)
    dict(total_reviews=1, avg_score=0.60, last_score=0.60, success_streak=1, avg_response_time=20),
    dict(total_reviews=3, avg_score=0.70, last_score=0.75, success_streak=2, avg_response_time=20),
    dict(total_reviews=6, avg_score=0.85, last_score=0.90, success_streak=3, avg_response_time=15),
    dict(total_reviews=2, avg_score=0.40, last_score=0.35, success_streak=0, avg_response_time=25),
]


# --------------------------------------------------------------------------- gate
def run_gate(predict_proba) -> dict:
    """predict_proba(X) -> P(recalled).  Returns {"passed": bool, "problems": [...]}"""
    problems = set()
    for learner in GATE_LEARNERS:
        X = np.array([[h] + [learner[c] for c in FEATURE_COLUMNS[1:]] for h in GATE_HOURS], float)
        p = np.asarray(predict_proba(X))
        if any(p[i + 1] > p[i] + 0.02 for i in range(len(p) - 1)):
            problems.add("recall goes UP as time since last review grows")
        if p[0] - p[-1] < 0.12:
            problems.add("recall barely changes between 1 hour and 60 days")
    return {"passed": not problems, "problems": sorted(problems)}


# ------------------------------------------------------------------------- models
def _constraints():
    return [MONOTONE.get(c, 0) for c in FEATURE_COLUMNS]


def _fallback_boosted(X_tr, y_tr, cst):
    model = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.06, max_iter=300, monotonic_cst=cst,
        early_stopping=False, random_state=42)
    model.fit(X_tr, y_tr)
    return "gradient_boosting", model


def make_boosted(X_tr, y_tr, X_val, y_val):
    """XGBoost (the project's main model) with monotonic constraints.  If xgboost is
    missing or rejects a setting, scikit-learn's gradient boosting with the same
    constraints is used instead, so training always finishes."""
    cst = _constraints()
    pos = float(np.mean(y_tr))
    spw = 1.0 if 0.2 <= pos <= 0.8 else float(np.sqrt((1 - pos) / max(pos, 1e-6)))
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
            monotone_constraints="(" + ",".join(str(c) for c in cst) + ")",
            scale_pos_weight=spw, eval_metric="logloss", early_stopping_rounds=30,
            random_state=42, n_jobs=1)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return "xgboost", model
    except Exception as exc:
        print(f"  (XGBoost not used: {str(exc)[:90]} -> scikit-learn gradient boosting instead)")
        return _fallback_boosted(X_tr, y_tr, cst)


def make_logistic(X_tr, y_tr):
    pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(X_tr, y_tr)
    return "logistic_regression", pipe


def _score(name, model, X_te, y_te, y_tr):
    p = model.predict_proba(X_te)[:, 1]
    brier = brier_score_loss(y_te, p)
    ref = brier_score_loss(y_te, np.full(len(y_te), float(np.mean(y_tr))))
    gate = run_gate(lambda X: model.predict_proba(X)[:, 1])
    return {
        "name": name, "model": model,
        "auc": float(roc_auc_score(y_te, p)), "brier": float(brier),
        "brier_skill": float(1 - brier / ref) if ref > 0 else 0.0,
        "accuracy": float(accuracy_score(y_te, p >= 0.5)),
        "gate": gate,
    }


# ------------------------------------------------------------------------- mlflow
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


# -------------------------------------------------------------------------- main
def run_training_pipeline(n_students: int = 150) -> dict:
    print("\n" + "=" * 62 + "\n  RETENT ENGRAM - model training\n" + "=" * 62)
    real_df = build_training_dataset_from_mongodb()
    df = get_combined_training_data(real_df, n_students=n_students)
    X, y, groups = df[FEATURE_COLUMNS].values, df[LABEL_COLUMN].values, df[GROUP_COLUMN].values

    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(X, y, groups))
    tr2, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=7).split(X[tr], y[tr], groups[tr]))
    X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]
    X_fit, y_fit, X_val, y_val = X_tr[tr2], y_tr[tr2], X_tr[va], y_tr[va]
    print(f"\n  rows: train {len(tr)} | test {len(te)} (different learners) | positive rate {y.mean():.0%}")

    results = [
        _score(*make_logistic(X_tr, y_tr), X_te, y_te, y_tr),
        _score(*make_boosted(X_fit, y_fit, X_val, y_val), X_te, y_te, y_tr),
    ]

    print(f"\n  {'model':<22}{'AUC':>7}{'Brier':>8}{'skill':>8}  gate")
    for r in results:
        verdict = "PASS" if r["gate"]["passed"] else "FAIL: " + "; ".join(r["gate"]["problems"])
        print(f"  {r['name']:<22}{r['auc']:>7.3f}{r['brier']:>8.3f}{r['brier_skill']:>8.2f}  {verdict}")

    eligible = [r for r in results if r["gate"]["passed"] and r["auc"] >= MIN_AUC]
    winner = max(eligible, key=lambda r: (round(r["auc"], 3), -r["brier"])) if eligible else None

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "n_rows": int(len(df)), "n_real_rows": int(len(real_df)),
        "train_max_hours": float(df["hours_since_last"].max()),
        "monotone_constraints": MONOTONE,
        "candidates": {r["name"]: {"auc": round(r["auc"], 4), "brier": round(r["brier"], 4),
                                   "accuracy": round(r["accuracy"], 4),
                                   "gate_passed": r["gate"]["passed"],
                                   "gate_problems": r["gate"]["problems"]} for r in results},
        "note": "AUC is measured on held-out SYNTHETIC learners plus any real rows; "
                "it shows how well the model learned the simulator, not real-world accuracy.",
    }
    if winner:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(winner["model"], f)
        meta.update(active_model=winner["name"], gate_passed=True,
                    **{winner["name"]: meta["candidates"][winner["name"]]})
        print(f"\n  ACTIVE MODEL: {winner['name']}  (AUC {winner['auc']:.3f}, Brier {winner['brier']:.3f})")
    else:
        MODEL_PATH.unlink(missing_ok=True)          # never leave a model that failed the gate
        meta.update(active_model="memory_model", gate_passed=False)
        print("\n  No model passed the gate -> the app will use the memory model (scorer.py).")
    MODEL_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with _mlflow_run("train_" + datetime.now().strftime("%Y%m%d_%H%M%S")) as mlf:
        if mlf:
            try:
                for r in results:
                    mlf.log_metrics({f"{r['name']}_auc": r["auc"], f"{r['name']}_brier": r["brier"],
                                     f"{r['name']}_gate": float(r["gate"]["passed"])})
                mlf.log_params({"rows": len(df), "real_rows": len(real_df),
                                "active_model": meta["active_model"]})
            except Exception as exc:
                print(f"  (MLflow logging skipped: {str(exc)[:80]})")
    print()
    return meta


if __name__ == "__main__":
    run_training_pipeline()
