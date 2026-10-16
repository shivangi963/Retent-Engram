# Retent Engram

**Personalised cognitive recall assistant** - estimates how much of each concept a student still
remembers, tells them what to review today, and writes practice material from their own notes.
Runs entirely on a laptop (no cloud LLM).

## How it works

```
Student -> Streamlit (7 pages) -> FastAPI backend -> MongoDB          events, scores, content
                                                  -> recall models    decay formula | XGBoost | survival | LSTM
                                                  -> review scheduler urgency queue, auto-prepared practice
                                                  -> RAG              FAISS retrieval + LangChain chain
                                                  -> Ollama           local LLM
```

1. Every study action is an **event** (concept, score 0-1, time). Practice done in the app is logged automatically;
   study elsewhere is logged on the *Log study* page.
2. Six **features** per concept (hours since last review, number of reviews, average and last score,
   success streak, average time) feed the **recall model**.
3. The **scheduler** ranks concepts by urgency and builds *Today*. When a concept falls below 65% recall,
   practice for it is **prepared automatically** in the background (`backend/autogen.py`).
4. **RAG**: uploaded PDFs -> sections -> chunks -> MiniLM embeddings -> FAISS. The LLM is told to use only the
   retrieved passages, at a difficulty chosen from the recall level.
5. Finishing a quiz / flashcards / coding task saves the result and the loop repeats.

## Models (all trained on the same learners, compared in `docs/model_comparison.md`)

| Model | Library | Note |
|---|---|---|
| Decay formula ("memory model") | NumPy | default; works from the first event; safety net |
| Logistic regression | scikit-learn | baseline |
| XGBoost | xgboost | monotonic constraint: waiting longer never raises recall |
| Survival (Weibull) | SciPy (own code) | time-to-forget with current-status censoring; extrapolates |
| Survival (Cox) | lifelines | comparison only |
| Sequence model | **PyTorch** LSTM | reads the review history; Weibull-shaped recall curve |

A trained model is used only if it passes a sanity gate (recall must fall as time passes, AUC >= 0.70 on held-out
learners). Most training examples come from a simulator of learners, so the reported AUC shows how well a model
learned that pattern - **not** real-world accuracy. Real accuracy needs real logged sessions / the classmate pilot.

## Run it

```
pip install -r requirements.txt
pip install -r requirements_new.txt          # optional extras
copy .env.example .env                       # then edit
ollama pull llama3.2:3b                      # small, fast; mistral also works
python scripts/run_training.py               # trains + compares all models, writes docs/model_comparison.md
python scripts/selfcheck.py                  # checks every component on your machine

streamlit run frontend/main.py                         # simple mode: one process
python scripts/start_all.py                            # full architecture: Streamlit -> FastAPI (docs at :8000/docs)
```

Atlas instead of local MongoDB: `docs/ATLAS_SETUP.md`.

## Layout

```
frontend/   Streamlit pages + components        backend/    service layer, FastAPI, ML, scheduler, auto-generation
rag/        PDF library, FAISS store, LLM       scripts/    training, API, self-check, MongoDB copy
data/       concepts.json, notes, PDF library   models/     trained model files, FAISS index (git-ignored)
```
