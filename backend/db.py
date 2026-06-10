from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

uri = os.getenv("MONGO_URI")

if uri is None:
    print(" ERROR: MONGO_URI not found in .env file!")
    exit()

client = MongoClient(uri)
db = client["pkdp_db"]

events_col = db["events"]
users_col = db["users"]
concepts_col = db["concepts"]


def insert_event(event: dict):
    result = events_col.insert_one(event)
    return str(result.inserted_id)

def get_events_by_user(user_id: str):
    return list(events_col.find({"user_id": user_id}, {"_id": 0}))

def test_connection():
    print("Connected to:", db.name)
    print("Collections:", db.list_collection_names())


def get_or_create_user(user_id: str, name: str) -> dict:
    """
    Looks up user by user_id. Creates them if they don't exist yet.
    Returns the user document.
    """
    existing = users_col.find_one({"user_id": user_id.strip().lower()}, {"_id": 0})
    if existing:
        return existing
    new_user = {
        "user_id": user_id.strip().lower(),
        "name": name.strip(),
        "created_at": datetime.utcnow(),
        "base_retention": 0.75
    }
    users_col.insert_one(new_user)
    return new_user

def get_recent_events(user_id: str, limit: int = 20) -> list:
    """
    Returns the most recent N events for a user, newest first.
    """
    return list(
        events_col.find(
            {"user_id": user_id.strip().lower()},
            {"_id": 0}
        ).sort("timestamp", -1).limit(limit)
    )

if __name__ == "__main__":
    test_connection()