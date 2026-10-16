"""Start the FastAPI backend:   python scripts/run_api.py     (docs: http://127.0.0.1:8000/docs)"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app", host=os.getenv("RETENT_API_HOST", "127.0.0.1"),
                port=int(os.getenv("RETENT_API_PORT", "8000")), log_level="info")
