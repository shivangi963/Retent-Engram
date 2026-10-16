"""
backend/ml/survival.py
======================
SURVIVAL view of forgetting: "how long until recall FAILS?"

Every review is an INSPECTION of the memory at time t = the gap (hours) since the previous review:
    next review went well  ->  not forgotten yet:  failure time T > t   (right-censored, likelihood S(t))
    next review failed     ->  already forgotten:  failure time T <= t  (left-censored,  likelihood 1 - S(t))
We never see the moment of forgetting itself - only whether it had happened by the review.  This is
"current-status" (interval-censored) survival data, and its likelihood is what is maximised below.
(Treating a failed review as an exact forgetting time at t - the textbook right-censored setup - makes the
curve too optimistic: the probabilities rank well but are badly calibrated.)

Two models
    WeibullSurvival  (deployable)  accelerated-failure-time model, S(t|x) = exp(-(t / scale(x)) ** shape)
        * scale(x) = exp(b . x): how long this learner's memory of this concept lasts
        * S(t|x) = chance the concept is still remembered after t hours = the recall estimate
        * decreases with t by construction and keeps decaying beyond the gaps seen in training
        * fitted here by maximum likelihood (scipy), no extra package needed
    CoxSurvival      (comparison only)  Cox proportional hazards via `lifelines` (Cox 1972, ref [4]
        in the report).  Its baseline hazard is flat beyond the longest observed gap, so it is
        reported in the comparison table but not deployed.
"""
import json

import numpy as np
from scipy.optimize import minimize

SURV_FEATURES = ["log_reviews", "avg_score", "last_score", "success_streak", "log_minutes"]
MIN_T = 0.05                                          # hours; keeps log(t) finite


def survival_covariates(df_or_features) -> np.ndarray:
    """(n, 5) covariates from the tabular features (a DataFrame, or one feature dict)."""
    f = df_or_features
    if isinstance(f, dict):
        f = {k: np.asarray([v], dtype=float) for k, v in f.items()}
    return np.column_stack([
        np.log1p(np.asarray(f["total_reviews"], dtype=float)),
        np.asarray(f["avg_score"], dtype=float),
        np.asarray(f["last_score"], dtype=float),
        np.asarray(f["success_streak"], dtype=float),
        np.log1p(np.asarray(f["avg_response_time"], dtype=float)),
    ])


class WeibullSurvival:
    def __init__(self, penalty: float = 1e-2):
        self.penalty = penalty
        self.beta = None            # [intercept, b1..bp] for log(scale)
        self.log_rho = 0.0          # log(shape)
        self.mean = None
        self.std = None

    # ----------------------------------------------------------------------- fitting
    def _design(self, X):
        Z = (np.asarray(X, dtype=float) - self.mean) / self.std
        return np.column_stack([np.ones(len(Z)), Z])

    def _nll(self, theta, D, logt, t, d):
        """d = 1 if forgotten by time t (left-censored), 0 if still remembered (right-censored)."""
        beta, r = theta[:-1], theta[-1]
        rho = np.exp(r)
        eta = D @ beta
        z = logt - eta
        u = np.clip(np.exp(np.clip(rho * z, -50, 50)), 1e-10, 700.0)         # u = (t / scale) ** rho,  S = exp(-u)
        ll = d * np.log(-np.expm1(-u)) - (1.0 - d) * u                       # log(1 - S)  /  log S
        n = len(t)
        nll = -ll.sum() / n + self.penalty * float(beta[1:] @ beta[1:])
        a = d / np.expm1(u) - (1.0 - d)                                      # d ll / d u
        g_beta = (D.T @ (a * rho * u)) / n                                   # du/d eta = -rho * u
        g_beta[1:] += 2.0 * self.penalty * beta[1:]
        g_r = -np.sum(a * rho * u * z) / n                                   # du/d log(rho) = rho * u * z
        return nll, np.append(g_beta, g_r)

    def fit(self, X, durations, events):
        """durations: gap in hours before the next review; events: 1 = that review FAILED (forgotten by then)."""
        X = np.asarray(X, dtype=float)
        t = np.maximum(np.asarray(durations, dtype=float), MIN_T)
        d = np.asarray(events, dtype=float)
        self.mean, self.std = X.mean(axis=0), X.std(axis=0) + 1e-9
        D = self._design(X)
        logt = np.log(t)
        theta0 = np.zeros(D.shape[1] + 1)
        theta0[0] = float(np.log(np.median(t) + 1.0))             # start near the typical gap
        self.n_forgot_ = int(d.sum())
        res = minimize(self._nll, theta0, args=(D, logt, t, d), jac=True, method="L-BFGS-B")
        self.beta, self.log_rho = res.x[:-1], float(res.x[-1])
        self.converged_ = bool(res.success)
        return self

    # --------------------------------------------------------------------- prediction
    def scale(self, X) -> np.ndarray:
        return np.exp(self._design(X) @ self.beta)

    def survival(self, X, hours) -> np.ndarray:
        """S(t|x): probability of NOT having forgotten after `hours` (scalar or one per row)."""
        t = np.maximum(np.asarray(hours, dtype=float), 0.0)
        return np.exp(-((t / self.scale(X)) ** np.exp(self.log_rho)))

    def median_hours(self, X) -> np.ndarray:
        """Hours until the chance of still remembering drops to 50%."""
        return self.scale(X) * np.log(2.0) ** (1.0 / np.exp(self.log_rho))

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {"type": "weibull_survival", "beta": self.beta.tolist(), "log_rho": self.log_rho,
                "mean": self.mean.tolist(), "std": self.std.tolist(), "features": SURV_FEATURES}

    @classmethod
    def from_dict(cls, d: dict):
        m = cls()
        m.beta, m.log_rho = np.asarray(d["beta"], dtype=float), float(d["log_rho"])
        m.mean, m.std = np.asarray(d["mean"], dtype=float), np.asarray(d["std"], dtype=float)
        return m

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class CoxSurvival:
    """Cox proportional hazards through lifelines (comparison only).  Needs: pip install lifelines
    lifelines' Cox cannot take current-status data, so a failed review is entered as an exact failure
    time at the gap (the standard right-censored setup) - one reason it is reported, not deployed."""

    def __init__(self, penalizer: float = 0.05):
        self.penalizer = penalizer
        self.cph = None

    def fit(self, X, durations, events):
        import pandas as pd
        from lifelines import CoxPHFitter
        df = pd.DataFrame(np.asarray(X, dtype=float), columns=SURV_FEATURES)
        df["T"] = np.maximum(np.asarray(durations, dtype=float), MIN_T)
        df["E"] = np.asarray(events, dtype=int)
        self.cph = CoxPHFitter(penalizer=self.penalizer).fit(df, duration_col="T", event_col="E")
        return self

    def survival(self, X, hours) -> np.ndarray:
        import pandas as pd
        df = pd.DataFrame(np.asarray(X, dtype=float), columns=SURV_FEATURES)
        base = self.cph.baseline_survival_.iloc[:, 0]                    # baseline S0(t), a step function
        s0 = np.interp(np.asarray(hours, dtype=float), base.index.values.astype(float), base.values.astype(float))
        ph = np.asarray(self.cph.predict_partial_hazard(df), dtype=float).ravel()
        return s0 ** ph                                                  # S(t|x) = S0(t) ** exp(b.x)
