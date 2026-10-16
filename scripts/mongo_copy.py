"""
scripts/mongo_copy.py
=====================
Copy your data between two MongoDB servers - for example from the local MongoDB on this laptop to a
free MongoDB Atlas cluster (so the same data is available from any device).

    python scripts/mongo_copy.py --from mongodb://localhost:27017 --to "mongodb+srv://USER:PASSWORD@CLUSTER.mongodb.net/"

Safe to run more than once: documents are matched by _id and replaced, never duplicated.
The source is only read.  Default database name: pkdp_db (or MONGO_DB_NAME).
"""
import argparse
import os
import sys
from pathlib import Path

DEFAULT_COLLECTIONS = ["users", "events", "recall_scores", "generated_content", "concepts"]


def copy_database(src_db, dst_db, collections=None, batch: int = 500, log=print) -> dict:
    """Copy `collections` (default: the app's five) from src_db to dst_db.  Returns {name: documents_copied}."""
    wanted = collections or DEFAULT_COLLECTIONS
    existing = set(src_db.list_collection_names())
    counts = {}
    for name in wanted:
        if name not in existing:
            log(f"  {name:<18} not in the source database - skipped")
            continue
        n = 0
        for doc in src_db[name].find({}):
            dst_db[name].replace_one({"_id": doc["_id"]}, doc, upsert=True)
            n += 1
        counts[name] = n
        log(f"  {name:<18} {n} documents copied")
    return counts


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    ap = argparse.ArgumentParser(description="Copy Retent Engram data between MongoDB servers")
    ap.add_argument("--from", dest="src", required=True, help="source URI, e.g. mongodb://localhost:27017")
    ap.add_argument("--to", dest="dst", required=True, help="target URI, e.g. your Atlas mongodb+srv:// string")
    ap.add_argument("--db", default=os.getenv("MONGO_DB_NAME", "pkdp_db"))
    ap.add_argument("--collections", nargs="*", help="default: " + ", ".join(DEFAULT_COLLECTIONS))
    a = ap.parse_args()
    from pymongo import MongoClient
    src, dst = MongoClient(a.src, serverSelectionTimeoutMS=8000), MongoClient(a.dst, serverSelectionTimeoutMS=8000)
    src.admin.command("ping")
    dst.admin.command("ping")
    print(f"Copying database '{a.db}' ...")
    counts = copy_database(src[a.db], dst[a.db], a.collections)
    print(f"Done: {sum(counts.values())} documents. Now set MONGO_URI in .env to the target URI and restart the app.")


if __name__ == "__main__":
    main()
