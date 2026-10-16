"""
backend/ml/evaluate.py
======================
Trains EVERY candidate model on the same examples and compares them on the same held-out
learners.  This produces the measured version of Table 5.3 in the report.

    logistic_regression   baseline (scikit-learn)
    xgboost               boosted trees with monotonic constraints (falls back to scikit-learn's
                          gradient boosting if xgboost is not installed)
    survival_weibull      survival analysis, time-to-forget (backend/ml/survival.py)
    survival_cox          Cox proportional hazards via lifelines - comparison only
    lstm                  PyTorch sequence model (backend/ml/seq_model.py)

A model that cannot run (PyTorch / lifelines missing, or an error) is reported as "not run"
with the reason - one failure never stops the others.

A model may become the ACTIVE one only if it passes the SANITY GATE:
    1. recall must never rise as the time since the last review grows (checked on typical learners)
    2. recall must clearly fall between 1 hour and 60 days
    3. AUC on held-out learners >= MIN_AUC
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.ml.data_prep import build_pairs_from_mongodb, FEATURE_COLUMNS
from backend.ml.sequences import build_examples, STEP_DIM
from backend.ml.survival import WeibullSurvival, CoxSurvival, survival_covariates
from backend.ml.synthetic_data import generate_synthetic_pairs

MIN_AUC = 0.70
MONOTONE = {"hours_since_last": -1, "avg_score": +1, "last_score": +1}   # +1: higher -> higher recall
GATE_HOURS = [1, 6, 24, 72, 168, 336, 720, 1440]
GATE_LEARNERS = [
    dict(total_reviews=1, avg_score=0.60, last_score=0.60, success_streak=1, avg_response_time=20),
    dict(total_reviews=3, avg_score=0.70, last_score=0.75, success_streak=2, avg_response_time=20),
    dict(total_reviews=6, avg_score=0.85, last_score=0.90, success_streak=3, avg_response_time=15),
    dict(total_reviews=2, avg_score=0.40, last_score=0.35, success_streak=0, avg_response_time=25),
]
# Table 5.3 of the report (target values): AUC, Brier
REPORT_TARGETS = {"logistic_regression": ("Logistic Regression", 0.72, 0.21),
                  "xgboost": ("XGBoost", 0.81, 0.17), "gradient_boosting": ("XGBoost", 0.81, 0.17),
                  "survival_weibull": ("Survival Model", 0.79, 0.18), "survival_cox": ("Survival Model", 0.79, 0.18),
                  "lstm": ("Sequence Model", 0.83, 0.16)}
DEPLOYABLE = {"logistic_regression", "xgboost", "gradient_boosting", "survival_weibull", "lstm"}


class Data:
    """All examples plus the learner-level split (nobody appears in two parts)."""

    def __init__(self, df: pd.DataFrame, seqs: list, n_real: int):
        self.df, self.seqs, self.n_real = df.reset_index(drop=True), seqs, n_real
        groups = self.df["group"].values
        rows = np.arange(len(self.df))
        tr_all, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(rows, groups=groups))
        fit_i, va_i = next(GroupShuffleSplit(1, test_size=0.15, random_state=7).split(tr_all, groups=groups[tr_all]))
        self.tr_all, self.te = tr_all, te
        self.fit, self.va = tr_all[fit_i], tr_all[va_i]
        self.X = self.df[FEATURE_COLUMNS].values
        self.y = self.df["label"].values
        self.gap = self.df["gap"].values.astype(float)
        self.cov = survival_covariates(self.df)

    def seq_of(self, idx):
        return [self.seqs[i] for i in idx]


class Fitted:
    def __init__(self, name, model, kind, proba_fn, curves_fn):
        self.name, self.model, self.kind = name, model, kind
        self.proba_fn, self.curves_fn = proba_fn, curves_fn        # (Data, idx) -> p ;  () -> [curve per typical learner]
        self.deployable = name in DEPLOYABLE


# ------------------------------------------------------------------------------- gate
def run_gate(curves) -> dict:
    """curves: one array per typical learner = recall at GATE_HOURS."""
    problems = set()
    for p in curves:
        p = np.asarray(p, dtype=float)
        if any(p[i + 1] > p[i] + 0.02 for i in range(len(p) - 1)):
            problems.add("recall goes UP as time since last review grows")
        if p[0] - p[-1] < 0.12:
            problems.add("recall barely changes between 1 hour and 60 days")
    return {"passed": not problems, "problems": sorted(problems)}


def _tab_row(learner, hours):
    return [hours] + [learner[c] for c in FEATURE_COLUMNS[1:]]


def gate_sequence(learner) -> np.ndarray:
    """A plausible review history matching a typical learner (for the LSTM's gate)."""
    n = int(learner["total_reviews"])
    scores = [learner["avg_score"]] * (n - 1) + [learner["last_score"]]
    mins = np.log1p(learner["avg_response_time"]) / 5.0
    return np.array([[s, 1.0 if s >= 0.6 else 0.0, np.log1p(48.0) / 8.0 if i else 0.0, mins]
                     for i, s in enumerate(scores)], dtype="float32").reshape(-1, STEP_DIM)


# ---------------------------------------------------------------------------- candidates
def fit_logistic(d: Data) -> Fitted:
    m = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]).fit(d.X[d.tr_all], d.y[d.tr_all])
    return Fitted("logistic_regression", m, "sklearn",
                  lambda dd, idx: m.predict_proba(dd.X[idx])[:, 1],
                  lambda: [m.predict_proba(np.array([_tab_row(l, h) for h in GATE_HOURS], float))[:, 1] for l in GATE_LEARNERS])


def fit_boosted(d: Data) -> Fitted:
    cst = [MONOTONE.get(c, 0) for c in FEATURE_COLUMNS]
    pos = float(np.mean(d.y[d.fit]))
    spw = 1.0 if 0.2 <= pos <= 0.8 else float(np.sqrt((1 - pos) / max(pos, 1e-6)))
    try:
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
                              monotone_constraints="(" + ",".join(str(c) for c in cst) + ")",
                              scale_pos_weight=spw, eval_metric="logloss", early_stopping_rounds=30,
                              random_state=42, n_jobs=1)
        m.fit(d.X[d.fit], d.y[d.fit], eval_set=[(d.X[d.va], d.y[d.va])], verbose=False)
        name = "xgboost"
    except Exception as exc:
        print(f"  (XGBoost not used: {str(exc)[:80]} -> scikit-learn gradient boosting)")
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=300, monotonic_cst=cst,
                                           early_stopping=False, random_state=42).fit(d.X[d.fit], d.y[d.fit])
        name = "gradient_boosting"
    return Fitted(name, m, "sklearn", lambda dd, idx: m.predict_proba(dd.X[idx])[:, 1],
                  lambda: [m.predict_proba(np.array([_tab_row(l, h) for h in GATE_HOURS], float))[:, 1] for l in GATE_LEARNERS])


def fit_weibull(d: Data) -> Fitted:
    m = WeibullSurvival().fit(d.cov[d.tr_all], d.gap[d.tr_all], 1 - d.y[d.tr_all])     # event = FORGOT
    return Fitted("survival_weibull", m, "weibull", lambda dd, idx: m.survival(dd.cov[idx], dd.gap[idx]),
                  lambda: [m.survival(np.repeat(survival_covariates(l), len(GATE_HOURS), axis=0), np.array(GATE_HOURS, float))
                           for l in GATE_LEARNERS])


def fit_cox(d: Data) -> Fitted:
    m = CoxSurvival().fit(d.cov[d.tr_all], d.gap[d.tr_all], 1 - d.y[d.tr_all])
    return Fitted("survival_cox", m, "cox", lambda dd, idx: m.survival(dd.cov[idx], dd.gap[idx]), lambda: [])


def fit_lstm(d: Data) -> Fitted:
    from backend.ml.seq_model import SequenceRecallModel, torch_available
    if not torch_available():
        raise ImportError("PyTorch is not installed (pip install torch)")
    m = SequenceRecallModel().fit(d.seq_of(d.fit), d.gap[d.fit], d.y[d.fit],
                                  val=(d.seq_of(d.va), d.gap[d.va], d.y[d.va]))
    def curves():
        return [m.predict([gate_sequence(l)] * len(GATE_HOURS), np.array(GATE_HOURS, float)) for l in GATE_LEARNERS]
    return Fitted("lstm", m, "lstm", lambda dd, idx: m.predict(dd.seq_of(idx), dd.gap[idx]), curves)


FITTERS = [("logistic_regression", fit_logistic), ("xgboost", fit_boosted), ("survival_weibull", fit_weibull),
           ("survival_cox", fit_cox), ("lstm", fit_lstm)]


# ---------------------------------------------------------------------------- evaluation
def score_model(f: Fitted, d: Data) -> dict:
    p = np.clip(np.asarray(f.proba_fn(d, d.te), dtype=float), 1e-6, 1 - 1e-6)
    y = d.y[d.te]
    ref = brier_score_loss(y, np.full(len(y), float(np.mean(d.y[d.tr_all]))))
    brier = brier_score_loss(y, p)
    curves = f.curves_fn()
    gate = run_gate(curves) if curves else {"passed": None, "problems": ["not checked (comparison only)"]}
    return {"status": "ok", "auc": float(roc_auc_score(y, p)), "brier": float(brier),
            "brier_skill": float(1 - brier / ref) if ref > 0 else 0.0, "log_loss": float(log_loss(y, p)),
            "accuracy": float(accuracy_score(y, p >= 0.5)), "gate": gate, "deployable": f.deployable}


def prepare_data(n_students: int = 150, seed: int = 42) -> Data:
    real_df, real_seqs = build_examples(build_pairs_from_mongodb(), group_prefix="real_")
    syn_df, syn_seqs = build_examples(generate_synthetic_pairs(n_students=n_students, seed=seed), group_prefix="syn_")
    df = pd.concat([real_df, syn_df], ignore_index=True) if len(real_df) else syn_df
    print(f"  examples: {len(df)} ({len(real_df)} real, {len(syn_df)} simulated), "
          f"{df['group'].nunique()} learners, {df['label'].mean():.0%} positive")
    return Data(df, real_seqs + syn_seqs, len(real_df))


def evaluate_all(n_students: int = 150, skip=()) -> dict:
    """Train + score every candidate.  Returns {'data','fitted','metrics','winner'}."""
    d = prepare_data(n_students)
    fitted, metrics = {}, {}
    for key, fitter in FITTERS:
        if key in skip:
            continue
        try:
            f = fitter(d)
            fitted[f.name] = f
            metrics[f.name] = score_model(f, d)
        except Exception as exc:                                           # one failure must not stop the rest
            metrics[key] = {"status": "not run", "reason": str(exc)[:120]}
    eligible = [(n, m) for n, m in metrics.items()
                if m.get("status") == "ok" and m["deployable"] and m["gate"]["passed"] and m["auc"] >= MIN_AUC]
    winner = max(eligible, key=lambda x: (round(x[1]["auc"], 3), -x[1]["brier"]))[0] if eligible else None
    return {"data": d, "fitted": fitted, "metrics": metrics, "winner": winner}


# -------------------------------------------------------------------------------- reports
def format_table(metrics: dict, winner=None) -> str:
    rows = [f"  {'model':<22}{'AUC':>7}{'Brier':>8}{'LogLoss':>9}  gate"]
    for name, m in metrics.items():
        if m.get("status") != "ok":
            rows.append(f"  {name:<22}{'-':>7}{'-':>8}{'-':>9}  not run ({m.get('reason', '')[:60]})")
            continue
        g = m["gate"]["passed"]
        verdict = "n/a (comparison only)" if g is None else "PASS" if g else "FAIL: " + "; ".join(m["gate"]["problems"])
        rows.append(f"  {name:<22}{m['auc']:>7.3f}{m['brier']:>8.3f}{m['log_loss']:>9.3f}  {verdict}" + ("   <- active" if name == winner else ""))
    return "\n".join(rows)


def write_reports(result: dict, docs_dir, evaluation_path) -> dict:
    """models/evaluation.json, docs/model_comparison.md (measured Table 5.3) and docs/model_comparison.png."""
    d, metrics, winner = result["data"], result["metrics"], result["winner"]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    summary = {"generated_at": stamp, "examples": int(len(d.df)), "real_examples": int(d.n_real),
               "learners": int(d.df["group"].nunique()), "test_examples": int(len(d.te)),
               "test_learners": int(d.df["group"].iloc[d.te].nunique()), "positive_rate": float(d.y.mean()),
               "winner": winner, "models": metrics}
    with open(evaluation_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    docs_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Model comparison (measured)", "",
             f"Generated {stamp}. {summary['examples']} labelled review gaps from {summary['learners']} learners "
             f"({summary['real_examples']} real, the rest simulated); metrics are on {summary['test_examples']} gaps from "
             f"{summary['test_learners']} learners NOT used for training.", "",
             "| Model | Role | AUC | Brier | Log loss | Sanity gate | Report target AUC / Brier |",
             "|---|---|---|---|---|---|---|"]
    for name, m in metrics.items():
        role, t_auc, t_brier = REPORT_TARGETS.get(name, (name, None, None))
        target = f"{t_auc:.2f} / {t_brier:.2f}" if t_auc else "-"
        if m.get("status") != "ok":
            lines.append(f"| {name} | {role} | not run | | | {m.get('reason', '')} | {target} |")
            continue
        g = m["gate"]["passed"]
        gate = "comparison only" if g is None else "pass" if g else "fail"
        mark = " (active)" if name == winner else ""
        lines.append(f"| {name}{mark} | {role} | {m['auc']:.3f} | {m['brier']:.3f} | {m['log_loss']:.3f} | {gate} | {target} |")
    lines += ["", "Higher AUC and lower Brier score are better.",
              "**Important:** most examples come from a simulator of learners, so these numbers show how well each model "
              "learned the simulated forgetting pattern, not accuracy on real students. Real accuracy needs the classmate pilot.", ""]
    (docs_dir / "model_comparison.md").write_text("\n".join(lines), encoding="utf-8")

    try:                                                   # figure in the style of Fig 5.3 (needs matplotlib)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ok = [(n, m) for n, m in metrics.items() if m.get("status") == "ok"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for ax, key, title, better in ((axes[0], "auc", "AUC (higher is better)", "high"), (axes[1], "brier", "Brier score (lower is better)", "low")):
            ax.bar([n.replace("_", "\n") for n, _ in ok], [m[key] for _, m in ok], color="#2b6cb0" if key == "auc" else "#dd6b20")
            ax.set_title(title, fontsize=10)
            if key == "auc":
                ax.set_ylim(0.5, 1.0)
            for i, (_, m) in enumerate(ok):
                ax.text(i, m[key], f"{m[key]:.2f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(docs_dir / "model_comparison.png", dpi=150)
        plt.close(fig)
        summary["figure"] = "docs/model_comparison.png"
    except Exception as exc:
        print(f"  (figure skipped: {str(exc)[:70]} - pip install matplotlib)")
    return summary
