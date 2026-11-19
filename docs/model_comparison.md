# Model comparison (measured)

Generated 2026-11-18 16:38 UTC. 4928 labelled review gaps from 151 learners (6 real, the rest simulated); metrics are on 1014 gaps from 31 learners NOT used for training.

| Model | Role | AUC | Brier | Log loss | Sanity gate | Report target AUC / Brier |
|---|---|---|---|---|---|---|
| logistic_regression | Logistic Regression | 0.911 | 0.123 | 0.376 | pass | 0.72 / 0.21 |
| xgboost | XGBoost | 0.953 | 0.088 | 0.277 | pass | 0.81 / 0.17 |
| survival_weibull | Survival Model | 0.943 | 0.098 | 0.311 | pass | 0.79 / 0.18 |
| survival_cox | Survival Model | 0.913 | 0.218 | 0.674 | comparison only | 0.79 / 0.18 |
| lstm (active) | Sequence Model | 0.965 | 0.076 | 0.239 | pass | 0.83 / 0.16 |

Higher AUC and lower Brier score are better.
**Important:** most examples come from a simulator of learners, so these numbers show how well each model learned the simulated forgetting pattern, not accuracy on real students. Real accuracy needs the classmate pilot.
