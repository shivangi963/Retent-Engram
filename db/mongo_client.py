from pymongo import MongoClient
from dotenv import load_dotenv
import os

# This explicitly tells dotenv WHERE to find .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

uri = os.getenv("MONGO_URI")

# Debug line — shows what URI is being used
print("Using URI:", uri)

if uri is None:
    print("❌ ERROR: MONGO_URI not found in .env file!")
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

if __name__ == "__main__":
    test_connection()