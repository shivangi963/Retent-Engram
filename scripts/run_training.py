"""
scripts/run_training.py
========================
THE ONLY SCRIPT YOU NEED TO RUN FOR PHASE 4

WHAT IT DOES
------------
  Runs the complete Phase 4 training pipeline:
    1. Connects to MongoDB, fetches real training data
    2. Generates synthetic data to supplement real data
    3. Trains Logistic Regression (baseline)
    4. Trains XGBoost (main model)
    5. Compares both on AUC + Brier Score
    6. Saves the winner to models/recall_model.pkl
    7. Logs everything to MLflow

HOW TO RUN
----------
  Make sure you are in the project root folder:
    cd Retent-Engram

  Make sure MongoDB is running:
    mongod   (or it should already be running)

  Activate your virtual environment:
    source venv/bin/activate    (Mac/Linux)
    venv\\Scripts\\activate      (Windows)

  Run:
    python scripts/run_training.py

EXPECTED OUTPUT
---------------
  ============================================================
    RETENT ENGRAM — Phase 4 Model Training
  ============================================================

  📊  MLflow experiment: 'retent-engram-recall-prediction'
       Run 'mlflow ui' to see the dashboard at http://localhost:5000

  📥  Loading training data from MongoDB...
       Fetched 45 total events from MongoDB
       Found 12 unique user-concept pairs
       Extracted 33 training rows
       ...

  🤖  Generating synthetic training data...
       Simulated 30 students × 8 concepts × 6 reviews
       Total simulated events: 1440
       Labeled training rows: 1200

  ─────────────────────────────────────────────────────
  Training Model 1: Logistic Regression (Baseline)
  ─────────────────────────────────────────────────────
    AUC:          0.7421
    Brier Score:  0.2012
    ...

  ─────────────────────────────────────────────────────
  Training Model 2: XGBoost (Main Model)
  ─────────────────────────────────────────────────────
    AUC:          0.8183
    Brier Score:  0.1621
    ...

  ==================================================
  Model Comparison
  ==================================================
    Model                          AUC    Brier   Accuracy
    Logistic Regression           0.7421  0.2012    0.6900
    XGBoost                       0.8183  0.1621    0.7600
    🏆  Winner: XGBoost
    💾  Best model saved to: models/recall_model.pkl

  ============================================================
    ✅  Training Complete!
    Winner: xgboost
    AUC:    0.8183
  ============================================================

  Next steps:
  1. Run 'mlflow ui' to see the experiment dashboard
  2. Restart the Streamlit app — it will now use the ML model
  3. Check the dashboard footer to confirm model = 'xgboost'

AFTER RUNNING
-------------
  - models/recall_model.pkl      ← winning model (used by predict.py)
  - models/recall_model_xgboost.pkl  ← XGBoost model
  - models/recall_model_logistic.pkl ← Logistic Regression model
  - models/model_metadata.json   ← metrics + which model is active
  - mlruns/                      ← MLflow experiment data

  To view MLflow dashboard:
    mlflow ui
    → open http://localhost:5000 in browser

  To retrain after logging more events:
    python scripts/run_training.py
    (each run creates a new MLflow experiment run for comparison)
"""

import sys
import os

# Add project root to Python path so we can import backend/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ml.train import run_training_pipeline

if __name__ == "__main__":
    run_training_pipeline()