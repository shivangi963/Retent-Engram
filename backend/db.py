from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

uri = os.getenv("MONGO_URI")

if uri is None:
    raise EnvironmentError(
        "MONGO_URI not found in .env file! "
        "Check that .env exists at D:\\retent\\.env"
    )

client = MongoClient(uri)
db     = client["pkdp_db"]

events_col   = db["events"]
users_col    = db["users"]
concepts_col = db["concepts"]


# ── Core utility ──────────────────────────────────────────────
def get_collection(name: str):
    """Returns any MongoDB collection by name."""
    return db[name]


# ── Events ────────────────────────────────────────────────────
def insert_event(event: dict) -> str:
    result = events_col.insert_one(event)
    return str(result.inserted_id)


def get_events_by_user(user_id: str) -> list:
    return list(events_col.find({"user_id": user_id}, {"_id": 0}))


def get_recent_events(user_id: str, limit: int = 20) -> list:
    return list(
        events_col.find(
            {"user_id": user_id.strip().lower()},
            {"_id": 0}
        ).sort("timestamp", -1).limit(limit)
    )


def get_all_events() -> list:
    return list(events_col.find({}, {"_id": 0}))


# ── Users ─────────────────────────────────────────────────────
def get_or_create_user(user_id: str, name: str = "Learner") -> dict:
    uid = user_id.strip().lower()
    existing = users_col.find_one({"user_id": uid}, {"_id": 0})
    if existing:
        return existing
    new_user = {
        "user_id":        uid,
        "name":           name.strip(),
        "created_at":     datetime.utcnow().isoformat(),
        "base_retention": 0.75
    }
    users_col.insert_one(new_user)
    return new_user


# ── Recall scores ─────────────────────────────────────────────
def get_recall_scores(user_id: str) -> list:
    col = get_collection("recall_scores")
    return list(
        col.find(
            {"user_id": user_id.strip().lower()},
            {"_id": 0}
        ).sort("recall_score", 1)
    )


# ── Concepts ──────────────────────────────────────────────────
def get_all_concepts() -> list:
    return list(concepts_col.find({}, {"_id": 0}))


def insert_concept(concept: dict) -> str:
    result = concepts_col.insert_one(concept)
    return str(result.inserted_id)


# ── Test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Connected to:", db.name)
    print("Collections:", db.list_collection_names())
    print("get_collection works:", get_collection("events").name)