from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv(
    dotenv_path=os.path.join(
        os.path.dirname(__file__),
        "..",
        ".env"
    )
)

uri = os.getenv("MONGO_URI")

if not uri:
    raise EnvironmentError(
        "MONGO_URI not found in .env file! "
        "Check that .env exists at D:\\retent\\.env"
    )

client = MongoClient(uri)
db = client["pkdp_db"]

events_col = db["events"]
users_col = db["users"]
concepts_col = db["concepts"]


def normalize_user_id(user_id: str) -> str:
    return user_id.strip().lower()


def get_collection(name: str):
    
    return db[name]

def insert_event(event: dict) -> str:
    result = events_col.insert_one(event)
    return str(result.inserted_id)


def get_events_by_user(user_id: str) -> list:
    return list(
        events_col.find(
            {"user_id": normalize_user_id(user_id)},
            {"_id": 0}
        )
    )


def get_recent_events(user_id: str, limit: int = 20) -> list:
    return list(
        events_col.find(
            {"user_id": normalize_user_id(user_id)},
            {"_id": 0}
        )
        .sort("timestamp", -1)
        .limit(limit)
    )


def get_all_events() -> list:
    return list(events_col.find({}, {"_id": 0}))


def get_total_events_count(user_id: str) -> int:
    return events_col.count_documents(
        {"user_id": normalize_user_id(user_id)}
    )


def get_or_create_user(user_id: str, name: str = "Learner") -> dict:
    uid = normalize_user_id(user_id)

    existing = users_col.find_one(
        {"user_id": uid},
        {"_id": 0}
    )

    if existing:
        return existing

    new_user = {
        "user_id": uid,
        "name": name.strip(),
        "created_at": datetime.utcnow().isoformat(),
        "base_retention": 0.75
    }

    users_col.insert_one(new_user)
    return new_user


def get_all_concepts() -> list:
    return list(concepts_col.find({}, {"_id": 0}))


def insert_concept(concept: dict) -> str:
    result = concepts_col.insert_one(concept)
    return str(result.inserted_id)


def get_recall_scores(user_id: str) -> list:
    
    col = get_collection("recall_scores")

    return list(
        col.find(
            {"user_id": normalize_user_id(user_id)},
            {"_id": 0}
        )
        .sort("recall_score", 1)
    )

def get_event_counts_by_concept(user_id: str) -> dict:
    
    pipeline = [
        {
            "$match": {
                "user_id": normalize_user_id(user_id)
            }
        },
        {
            "$group": {
                "_id": "$concept_id",
                "count": {"$sum": 1}
            }
        }
    ]

    result = list(events_col.aggregate(pipeline))

    return {
        doc["_id"]: doc["count"]
        for doc in result
    }


def get_last_event_per_concept(user_id: str) -> dict:
   
    pipeline = [
        {
            "$match": {
                "user_id": normalize_user_id(user_id)
            }
        },
        {
            "$sort": {
                "timestamp": -1
            }
        },
        {
            "$group": {
                "_id": "$concept_id",
                "last_ts": {
                    "$first": "$timestamp"
                }
            }
        }
    ]

    result = list(events_col.aggregate(pipeline))

    return {
        doc["_id"]: doc["last_ts"]
        for doc in result
    }

"""
backend/db.py — Phase 5 additions
====================================
PASTE THESE FUNCTIONS AT THE BOTTOM OF YOUR EXISTING backend/db.py
Do NOT replace anything already there — just add these below.

These new functions support the queue page and settings.
"""

# ── Add these to the BOTTOM of backend/db.py ─────────────────────────────────


def get_user(user_id: str) -> dict:
    """
    Fetches the full user document from MongoDB.

    Returns the user dict (without _id field) or empty dict if not found.

    Used by: 4_queue.py to show user info, daily_goal, streak

    Args:
        user_id: student's ID

    Returns:
        dict: user document or {}
    """
    col = get_collection("users")
    result = col.find_one(
        {"user_id": user_id.strip().lower()},
        {"_id": 0}
    )
    return result or {}


def update_user_field(user_id: str, field: str, value) -> bool:
    """
    Updates a single field on a user document.

    Generic helper used for daily_goal, streak_days, last_active_date etc.
    Avoids writing a separate function for each user field.

    EXAMPLE USAGE:
      update_user_field("shivangi_01", "daily_goal", 5)
      update_user_field("shivangi_01", "streak_days", 7)

    Args:
        user_id: student's ID
        field:   MongoDB field name to update
        value:   new value

    Returns:
        bool: True if successful
    """
    try:
        col = get_collection("users")
        col.update_one(
            {"user_id": user_id.strip().lower()},
            {"$set": {field: value}},
            upsert=True
        )
        return True
    except Exception as e:
        print(f"⚠️  update_user_field error: {e}")
        return False


def get_concepts_reviewed_today(user_id: str) -> list:
    """
    Returns list of concept_ids that have been reviewed today.

    HOW: Reads recall_scores where last_reviewed_today = True.

    Used by: 4_queue.py to show which concepts are already done today.

    Args:
        user_id: student's ID

    Returns:
        list of concept_id strings
    """
    col = get_collection("recall_scores")
    docs = list(col.find(
        {"user_id": user_id, "last_reviewed_today": True},
        {"concept_id": 1, "_id": 0}
    ))
    return [d["concept_id"] for d in docs]


def get_recent_events_for_concept(user_id: str, concept_id: str, limit: int = 10) -> list:
    """
    Returns the most recent events for a specific user-concept pair.

    Used by: Queue page → "Recent history" expander per queue item
    Shows the student their recent performance on this concept.

    Args:
        user_id:    student's ID
        concept_id: which concept
        limit:      max events to return (default 10)

    Returns:
        list of event dicts sorted newest first
    """
    col = get_collection("events")
    return list(
        col.find(
            {"user_id": user_id, "concept_id": concept_id},
            {"_id": 0}
        ).sort("timestamp", -1).limit(limit)
    )


def get_generated_content_history(user_id: str, concept_id: str,
                                   limit: int = 10) -> list:
    """
    Returns recent generated content for a user-concept pair.
    Used by the review page to show content history.

    Returns:
        list of generated_content dicts, newest first
    """
    col = get_collection("generated_content")
    return list(
        col.find(
            {"user_id": user_id, "concept_id": concept_id},
            {"_id": 0}
        ).sort("generated_at", -1).limit(limit)
    )


def get_all_generated_for_user(user_id: str) -> list:
    """
    Returns all generated content for a user across all concepts.
    Used by history page in Phase 7.
    """
    col = get_collection("generated_content")
    return list(
        col.find({"user_id": user_id}, {"_id": 0})
           .sort("generated_at", -1)
    )


def get_content_ratings_summary(user_id: str) -> dict:
    """
    Returns a summary of ratings per concept for the review page.

    Returns:
        dict: { concept_id: { "good": 3, "bad": 1, "unrated": 2 } }
    """
    col = get_collection("generated_content")
    docs = list(col.find({"user_id": user_id}, {"concept_id": 1, "rating": 1, "_id": 0}))

    summary = {}
    for doc in docs:
        cid = doc.get("concept_id", "")
        rating = doc.get("rating", 0)
        if cid not in summary:
            summary[cid] = {"good": 0, "bad": 0, "unrated": 0}
        if rating == 1:
            summary[cid]["good"] += 1
        elif rating == -1:
            summary[cid]["bad"] += 1
        else:
            summary[cid]["unrated"] += 1

    return summary


if __name__ == "__main__":
    print("Connected to:", db.name)
    print("Collections:", db.list_collection_names())
    print(
        "get_collection works:",
        get_collection("events").name
    )
    