import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    accuracy_score,
    classification_report
)
from sklearn.pipeline import Pipeline as SklearnPipeline
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from backend.ml.data_prep import build_training_dataset_from_mongodb, get_X_y, FEATURE_COLUMNS
from backend.ml.synthetic_data import get_combined_training_data


PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)   # create models/ folder if it doesn't exist

# File paths
LOGISTIC_MODEL_PATH = os.path.join(MODELS_DIR, "recall_model_logistic.pkl")
XGBOOST_MODEL_PATH  = os.path.join(MODELS_DIR, "recall_model_xgboost.pkl")
BEST_MODEL_PATH     = os.path.join(MODELS_DIR, "recall_model.pkl")   # used by predict.py
METADATA_PATH       = os.path.join(MODELS_DIR, "model_metadata.json")

MLFLOW_EXPERIMENT = "retent-engram-recall-prediction"


def setup_mlflow():
    
    #Uses SQLite backend — required for MLflow . Creates mlflow.db in your project root automatically.
    
    mlflow_db_path = os.path.abspath(os.path.join(PROJECT_ROOT, "mlflow.db"))
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)



def train_logistic_regression(X_train, y_train, X_test, y_test) -> dict:
   
    print("\n" + "─" * 50)
    print("Training Model 1: Logistic Regression (Baseline)")
    print("─" * 50)

    with mlflow.start_run(run_name="logistic_regression", nested=True):
        #   z = (x - mean) / std
        #   After scaling: each feature has mean=0 and std=1
        scaler = StandardScaler()

        # LogisticRegression: outputs P(recalled=1 | features)
        lr = LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",  # compensate if labels are imbalanced
            random_state=42
        )

        # fit() on pipeline calls scaler.fit_transform(X) then lr.fit(X_scaled, y)
        pipeline = SklearnPipeline([
            ("scaler", scaler),
            ("classifier", lr)
        ])

        mlflow.log_param("model_type",    "logistic_regression")
        mlflow.log_param("C",             1.0)
        mlflow.log_param("max_iter",      1000)
        mlflow.log_param("class_weight",  "balanced")
        mlflow.log_param("n_features",    X_train.shape[1])
        mlflow.log_param("n_train",       len(X_train))
        mlflow.log_param("n_test",        len(X_test))

        pipeline.fit(X_train, y_train)

        # predict_proba returns [[P(0), P(1)], .....] for each sample
        # P(1) = probability of being recalled = column index 1
        y_prob  = pipeline.predict_proba(X_test)[:, 1]
        y_pred  = pipeline.predict(X_test)

        auc       = roc_auc_score(y_test, y_prob)
        brier     = brier_score_loss(y_test, y_prob)
        accuracy  = accuracy_score(y_test, y_pred)

        mlflow.log_metric("auc",       auc)
        mlflow.log_metric("brier",     brier)
        mlflow.log_metric("accuracy",  accuracy)

        # ── Cross-validation (5-fold) ─────────────────────────────────────────
        # Cross-validation = train/evaluate 5 times on different splits
        # Gives a more reliable estimate of real performance
        cv_scores = cross_val_score(pipeline, X_train, y_train,
                                    cv=5, scoring="roc_auc")
        cv_mean = cv_scores.mean()
        cv_std  = cv_scores.std()
        mlflow.log_metric("cv_auc_mean", cv_mean)
        mlflow.log_metric("cv_auc_std",  cv_std)

        # ── Log model to MLflow ───────────────────────────────────────────────
        mlflow.sklearn.log_model(pipeline, "logistic_model")

        # ── Save locally ─────────────────────────────────────────────────────
        with open(LOGISTIC_MODEL_PATH, "wb") as f:
            pickle.dump(pipeline, f)

        # ── Print results ─────────────────────────────────────────────────────
        print(f"  AUC:          {auc:.4f}  (target > 0.70)")
        print(f"  Brier Score:  {brier:.4f} (target < 0.20)")
        print(f"  Accuracy:     {accuracy:.4f}")
        print(f"  CV AUC:       {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"\n  Feature coefficients (importance):")
        # Extract the actual LR model from inside the pipeline
        lr_model = pipeline.named_steps["classifier"]
        for feat, coef in zip(FEATURE_COLUMNS, lr_model.coef_[0]):
            direction = "↑ helps recall" if coef > 0 else "↓ hurts recall"
            print(f"    {feat:<25} = {coef:+.4f}  ({direction})")

        print(f"\n  📁 Saved to: {LOGISTIC_MODEL_PATH}")

    return {
        "model":      pipeline,
        "auc":        auc,
        "brier":      brier,
        "accuracy":   accuracy,
        "cv_auc":     cv_mean,
        "model_path": LOGISTIC_MODEL_PATH
    }


# =============================================================================
# MODEL 2 — XGBoost (Main Model)
# =============================================================================

def train_xgboost(X_train, y_train, X_test, y_test) -> dict:
    """
    Trains an XGBoost classifier and evaluates it.

    WHY XGBOOST?
      XGBoost = eXtreme Gradient Boosting
      - Builds an ensemble of decision trees
      - Each tree corrects the mistakes of the previous one
      - Handles non-linear relationships (e.g. forgetting is exponential, not linear)
      - Works well with tabular data (our 6 features are tabular)
      - Usually outperforms Logistic Regression on structured data
      - Recommended in the project plan as the "main model"

    HOW XGBOOST WORKS (simplified):
      1. Start with a simple prediction (e.g. predict the average recall for everyone)
      2. Compute "errors" (how wrong was the prediction?)
      3. Train a decision tree to predict the errors
      4. Update predictions by adding the tree's outputs (scaled by learning_rate)
      5. Repeat steps 2–4 n_estimators times
      Each iteration reduces errors a little more.

    KEY PARAMETERS EXPLAINED:
      n_estimators=100:    number of trees to build
                           More trees = better (up to a point), slower
      max_depth=4:         how deep each tree can grow
                           Deeper = more complex patterns, risk of overfitting
      learning_rate=0.1:   how much each tree updates predictions
                           Smaller = more trees needed but smoother
      subsample=0.8:       use 80% of training data per tree (prevents overfitting)
      colsample_bytree=0.8: use 80% of features per tree (prevents overfitting)
      use_label_encoder=False: suppress deprecation warning
      eval_metric="logloss": training loss to minimize

    EARLY STOPPING:
      We use eval_set to monitor validation loss during training.
      If validation loss doesn't improve for 20 rounds, training stops.
      This prevents overfitting without having to manually tune n_estimators.

    Args:
        X_train, y_train: training features and labels
        X_test, y_test:   test features and labels

    Returns:
        dict with keys: model, auc, brier, accuracy, model_path
    """
    print("\n" + "─" * 50)
    print("Training Model 2: XGBoost (Main Model)")
    print("─" * 50)

    with mlflow.start_run(run_name="xgboost", nested=True):
        # ── Define XGBoost model ──────────────────────────────────────────────
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,      # minimum samples per leaf (reduces overfitting)
            gamma=0.1,               # minimum loss reduction to make a split
            scale_pos_weight=1,      # balance classes (adjust if very imbalanced)
            eval_metric="logloss",
            early_stopping_rounds=20,
            random_state=42,
            verbosity=0              # suppress training output
        )

        # ── Log parameters ────────────────────────────────────────────────────
        mlflow.log_param("model_type",       "xgboost")
        mlflow.log_param("n_estimators",     100)
        mlflow.log_param("max_depth",        4)
        mlflow.log_param("learning_rate",    0.1)
        mlflow.log_param("subsample",        0.8)
        mlflow.log_param("colsample_bytree", 0.8)
        mlflow.log_param("n_train",          len(X_train))
        mlflow.log_param("n_test",           len(X_test))

        # ── Train with early stopping ──────────────────────────────────────────
        # eval_set lets XGBoost monitor validation loss during training
        # If val loss doesn't improve for early_stopping_rounds → stop early
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False    # don't print each round
        )

        actual_trees = model.best_iteration + 1 if hasattr(model, 'best_iteration') else 100
        print(f"  Trees actually used: {actual_trees} (of 100 max)")

        # ── Evaluate ──────────────────────────────────────────────────────────
        y_prob  = model.predict_proba(X_test)[:, 1]
        y_pred  = model.predict(X_test)

        auc      = roc_auc_score(y_test, y_prob)
        brier    = brier_score_loss(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_pred)

        # ── Log metrics ───────────────────────────────────────────────────────
        mlflow.log_metric("auc",      auc)
        mlflow.log_metric("brier",    brier)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("actual_trees_used", actual_trees)

        # ── Feature importance ────────────────────────────────────────────────
        # XGBoost computes "gain" = how much each feature reduces prediction error
        importances = model.feature_importances_   # shape: (n_features,)
        for feat, imp in zip(FEATURE_COLUMNS, importances):
            mlflow.log_metric(f"importance_{feat}", float(imp))

        # ── Log model to MLflow ───────────────────────────────────────────────
        mlflow.xgboost.log_model(model, "xgboost_model")

        # ── Save locally ──────────────────────────────────────────────────────
        with open(XGBOOST_MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

        # ── Print results ─────────────────────────────────────────────────────
        print(f"  AUC:          {auc:.4f}  (target > 0.75)")
        print(f"  Brier Score:  {brier:.4f} (target < 0.18)")
        print(f"  Accuracy:     {accuracy:.4f}")

        print(f"\n  Feature importance (XGBoost gain):")
        sorted_features = sorted(zip(FEATURE_COLUMNS, importances),
                                 key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_features:
            bar = "█" * int(imp * 40)   # visual bar (proportional)
            print(f"    {feat:<25} {bar} ({imp:.4f})")

        # Print classification report (precision, recall, f1 per class)
        print(f"\n  Classification Report:")
        report = classification_report(y_test, y_pred,
                                       target_names=["Forgot (0)", "Recalled (1)"])
        for line in report.split("\n"):
            print(f"    {line}")

        print(f"\n  📁 Saved to: {XGBOOST_MODEL_PATH}")

    return {
        "model":      model,
        "auc":        auc,
        "brier":      brier,
        "accuracy":   accuracy,
        "model_path": XGBOOST_MODEL_PATH
    }


# =============================================================================
# MODEL COMPARISON AND SELECTION
# =============================================================================

def compare_and_select(lr_results: dict, xgb_results: dict) -> str:
    """
    Compares Logistic Regression vs XGBoost and selects the better one.

    SELECTION CRITERIA:
      Primary:   AUC (higher is better)
      Tiebreaker: Brier Score (lower is better)
      We weight AUC more heavily because ranking accuracy matters most
      for our use case (we need to rank concepts by urgency, not just
      classify them as remembered/forgotten).

    THE WINNER:
      Saved to models/recall_model.pkl — this is what predict.py loads.

    Args:
        lr_results:  results dict from train_logistic_regression()
        xgb_results: results dict from train_xgboost()

    Returns:
        str: "xgboost" or "logistic_regression" — winner name
    """
    print("\n" + "=" * 50)
    print("Model Comparison")
    print("=" * 50)
    print(f"\n  {'Model':<30} {'AUC':>8} {'Brier':>8} {'Accuracy':>10}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*10}")
    print(f"  {'Logistic Regression':<30} "
          f"{lr_results['auc']:>8.4f} "
          f"{lr_results['brier']:>8.4f} "
          f"{lr_results['accuracy']:>10.4f}")
    print(f"  {'XGBoost':<30} "
          f"{xgb_results['auc']:>8.4f} "
          f"{xgb_results['brier']:>8.4f} "
          f"{xgb_results['accuracy']:>10.4f}")

    # Select winner: higher AUC wins. If tie (within 0.01), lower Brier wins.
    auc_diff = xgb_results["auc"] - lr_results["auc"]

    if auc_diff > 0.01:
        winner = "xgboost"
        winning_model = xgb_results["model"]
        print(f"\n  🏆  Winner: XGBoost (AUC better by {auc_diff:.4f})")
    elif auc_diff < -0.01:
        winner = "logistic_regression"
        winning_model = lr_results["model"]
        print(f"\n  🏆  Winner: Logistic Regression (AUC better by {-auc_diff:.4f})")
    else:
        # AUC tie → use Brier Score as tiebreaker
        if xgb_results["brier"] < lr_results["brier"]:
            winner = "xgboost"
            winning_model = xgb_results["model"]
        else:
            winner = "logistic_regression"
            winning_model = lr_results["model"]
        print(f"\n  🏆  Winner: {winner} (AUC tie, Brier decides)")

    # Save winning model as the active model
    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump(winning_model, f)
    print(f"  💾  Best model saved to: {BEST_MODEL_PATH}")

    return winner


# =============================================================================
# SAVE METADATA
# =============================================================================

def save_metadata(winner: str, lr_results: dict, xgb_results: dict):
    """
    Saves a JSON file recording which model is active and its performance.

    USED BY:
      - predict.py: knows which model_type was used
      - pipeline.py: records "xgboost" in the recall_score MongoDB doc
      - 2_dashboard.py: shows "Model: XGBoost" in the footer

    EXAMPLE metadata.json:
      {
        "active_model": "xgboost",
        "trained_at": "2026-06-10T10:30:00",
        "xgboost":  { "auc": 0.82, "brier": 0.16, "accuracy": 0.78 },
        "logistic": { "auc": 0.74, "brier": 0.21, "accuracy": 0.70 },
        "feature_columns": ["hours_since_last", "total_reviews", ...]
      }
    """
    metadata = {
        "active_model": winner,
        "trained_at":   datetime.now(timezone.utc).isoformat(),
        "xgboost": {
            "auc":      round(xgb_results["auc"], 4),
            "brier":    round(xgb_results["brier"], 4),
            "accuracy": round(xgb_results["accuracy"], 4)
        },
        "logistic_regression": {
            "auc":      round(lr_results["auc"], 4),
            "brier":    round(lr_results["brier"], 4),
            "accuracy": round(lr_results["accuracy"], 4)
        },
        "feature_columns":     FEATURE_COLUMNS,
        "recall_threshold":    0.6
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  📋  Metadata saved to: {METADATA_PATH}")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def run_training_pipeline():
    """
    Full Phase 4 training pipeline. Call this from scripts/run_training.py.

    STEPS:
      1. Setup MLflow
      2. Load real training data from MongoDB
      3. Combine with synthetic data
      4. Split into train/test
      5. Train Logistic Regression
      6. Train XGBoost
      7. Compare models, select winner
      8. Save winner to models/recall_model.pkl
      9. Save metadata to models/model_metadata.json
    """
    print("\n" + "=" * 60)
    print("  RETENT ENGRAM — Phase 4 Model Training")
    print("=" * 60)

    # ── Step 1: Setup MLflow ─────────────────────────────────────────────────
    setup_mlflow()
    print(f"\n📊  MLflow experiment: '{MLFLOW_EXPERIMENT}'")
    print(f"     Run 'mlflow ui' to see the dashboard at http://localhost:5000")

    # ── Step 2: Load real data ────────────────────────────────────────────────
    print("\n📥  Loading training data from MongoDB...")
    real_df = build_training_dataset_from_mongodb()

    # ── Step 3: Combine with synthetic data ──────────────────────────────────
    print("\n🤖  Generating synthetic training data...")
    combined_df = get_combined_training_data(real_df, synthetic_multiplier=3)

    if combined_df.empty:
        print("❌  No training data available. Cannot train model.")
        return

    # ── Step 4: Train/test split ──────────────────────────────────────────────
    from backend.ml.data_prep import get_X_y
    X, y = get_X_y(combined_df)

    # 80% train, 20% test
    # stratify=y ensures both train and test have similar label ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y      # maintain label balance in both splits
    )

    print(f"\n📊  Dataset split:")
    print(f"    Train: {len(X_train)} rows  |  "
          f"Recalled: {int(y_train.sum())}  |  Forgot: {int((y_train==0).sum())}")
    print(f"    Test:  {len(X_test)} rows  |  "
          f"Recalled: {int(y_test.sum())}  |  Forgot: {int((y_test==0).sum())}")

    # ── Steps 5 & 6: Train both models inside one MLflow parent run ──────────
    print(f"\n🚀  Starting training runs...")

    with mlflow.start_run(run_name="phase4_training_comparison"):
        # Log overall training info
        mlflow.log_param("total_training_rows",    len(X_train))
        mlflow.log_param("test_rows",              len(X_test))
        mlflow.log_param("n_features",             len(FEATURE_COLUMNS))

        # Train both models (nested=True means these are sub-runs inside the parent)
        lr_results  = train_logistic_regression(X_train, y_train, X_test, y_test)
        xgb_results = train_xgboost(X_train, y_train, X_test, y_test)

        # ── Step 7: Compare and select ────────────────────────────────────────
        winner = compare_and_select(lr_results, xgb_results)

        # Log the winner to the parent run
        mlflow.log_param("winning_model", winner)
        mlflow.log_metric("best_auc",
                          xgb_results["auc"] if winner == "xgboost"
                          else lr_results["auc"])

    # ── Steps 8 & 9: Save metadata ────────────────────────────────────────────
    save_metadata(winner, lr_results, xgb_results)

    print("\n" + "=" * 60)
    print("  ✅  Training Complete!")
    print(f"  Winner: {winner}")
    print(f"  AUC:    {xgb_results['auc'] if winner == 'xgboost' else lr_results['auc']:.4f}")
    print("=" * 60)
    print("\n  Next steps:")
    print("  1. Run 'mlflow ui' to see the experiment dashboard")
    print("  2. Restart the Streamlit app — it will now use the ML model")
    print("  3. Check the dashboard footer to confirm model = 'xgboost'")
    print()


# Allow running directly: python backend/ml/train.py
if __name__ == "__main__":
    run_training_pipeline()