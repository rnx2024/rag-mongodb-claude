import os, uuid, datetime
from typing import Optional, List, Dict
import streamlit as st
from pymongo import MongoClient, ASCENDING
from anthropic import Anthropic

# optional: load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- config ---
MONGO_URI = os.environ["MONGO_URI"]
DB_NAME = "rag"
DOCS_COLL = "docs"
CHAT_COLL = "chat"
MODEL = "claude-3-5-sonnet-latest"
K_DEFAULT = 5
MAX_BODY_CHARS = 1200

# --- init clients ---
mc = MongoClient(MONGO_URI)
db = mc[DB_NAME]
docs = db[DOCS_COLL]
chat = db[CHAT_COLL]
anth = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# --- ensure collections/indexes (idempotent) ---
try:
    docs.create_index([("title", "text"), ("section", "text"), ("body", "text")], name="kb_text_idx")
except Exception:
    pass
try:
    chat.create_index([("session_id", ASCENDING), ("ts", ASCENDING)], name="chat_session_ts")
except Exception:
    pass

# --- helpers ---
def get_history(session_id: str, limit: int = 20) -> List[Dict]:
    cur = chat.find({"session_id": session_id}).sort("ts", 1).limit(limit)
    return list(cur)

def save_msg(session_id: str, role: str, content: str) -> None:
    chat.insert_one({
        "session_id": session_id,
        "ts": datetime.datetime.utcnow(),
        "role": role,
        "content": content
    })

def search_docs(query: str, k: int = 5, topic: Optional[str] = None) -> List[Dict]:
    match: Dict = {"$text": {"$search": query}}
    if topic:
        match["topic"] = topic
    pipeline = [
        {"$match": match},
        {"$addFields": {"score": {"$meta": "textScore"}}},
        {"$sort": {"score": -1}},
        {"$limit": int(k)},
        {"$project": {"_id": 0, "source": 1, "title": 1, "section": 1, "body": 1, "score": 1}},
    ]
    return list(docs.aggregate(pipeline))

def build_context(rows: List[Dict]) -> str:
    def trunc(s: str) -> str:
        s = (s or "").strip()
        return s if len(s) <= MAX_BODY_CHARS else s[:MAX_BODY_CHARS] + "…"
    parts = []
    for i, r in enumerate(rows, 1):
        hdr = f"[Doc {i}] {r.get('title') or r.get('source')}"
        if r.get("section"):
            hdr += f" — {r['section']}"
