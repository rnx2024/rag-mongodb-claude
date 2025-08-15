import os, uuid, datetime, re
from typing import List, Dict, Optional
import streamlit as st
from urllib.parse import quote_plus
import certifi
from pymongo import MongoClient, ASCENDING
from pymongo.server_api import ServerApi
from anthropic import Anthropic, APIStatusError

st.set_page_config(page_title="SEO Coach", page_icon="🔍")
st.title("SEO Coach")

# -------- Secrets / Config (Mongo via driver only) --------
MONGO_URI = st.secrets.get("MONGO_URI")  # optional full SRV URI
MONGO_USER = st.secrets.get("MONGO_USER")
MONGO_PASSWORD = st.secrets.get("MONGO_PASSWORD")
MONGO_HOST = st.secrets.get("MONGO_HOST") or "cluster0.bo4mikx.mongodb.net"
MONGO_DB = st.secrets.get("MONGO_DB") or "rag"
MONGO_APPNAME = st.secrets.get("MONGO_APPNAME") or "Cluster0"
MONGO_AUTH_SOURCE = st.secrets.get("MONGO_AUTH_SOURCE")  # optional, e.g. "admin"

# Claude (kept; remove if you don't want LLM responses)
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY")

DB_NAME, DOCS_COLL, CHAT_COLL = MONGO_DB, "docs", "chat"
MODEL = "claude-3-5-sonnet-latest"
K_DEFAULT, MAX_BODY_CHARS = 5, 1200
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def _build_mongo_uri() -> Optional[str]:
    if MONGO_URI:
        return MONGO_URI
    if MONGO_USER and MONGO_PASSWORD:
        user = quote_plus(MONGO_USER)
        pwd = quote_plus(MONGO_PASSWORD)
        base = f"mongodb+srv://{user}:{pwd}@{MONGO_HOST}"
        qs = f"retryWrites=true&w=majority&appName={MONGO_APPNAME}"
        # Default to admin authSource unless explicitly set
        auth_src = quote_plus(MONGO_AUTH_SOURCE or "admin")
        if MONGO_DB:
            return f"{base}/{MONGO_DB}?{qs}&authSource={auth_src}"
        return f"{base}/?{qs}&authSource={auth_src}"
    return None

# -------- Clients --------
@st.cache_resource(show_spinner=False)
def get_clients():
    anth = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

    uri = _build_mongo_uri()
    if not uri:
        st.warning("Mongo URI not configured. Set MONGO_URI or MONGO_USER/MONGO_PASSWORD in secrets.")
        return anth, None

    try:
        mc = MongoClient(
            uri,
            server_api=ServerApi("1"),
            tls=True,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=15000,
            connectTimeoutMS=15000,
            socketTimeoutMS=15000,
            appname="seo-coach",
        )
        mc.admin.command("ping")              # network
        db = mc[DB_NAME]
        db.command("listCollections")         # force auth on target DB

        # Indexes (idempotent)
        db[DOCS_COLL].create_index(
            [("title","text"),("section","text"),("body","text")],
            name="kb_text_idx"
        )
        db[CHAT_COLL].create_index([("session_id", ASCENDING), ("ts", ASCENDING)], name="chat_session_ts")
        db[CHAT_COLL].create_index([("email", ASCENDING)], name="chat_email_idx")

        # Optional seed so search works on first run
        if db[DOCS_COLL].count_documents({}) == 0:
            db[DOCS_COLL].insert_one({
                "source":"seed.md",
                "title":"Seed",
                "section":"Init",
                "body":"Seed doc",
                "topic":"SEO",
                "updated_at": datetime.datetime.utcnow()
            })

        return anth, db
    except Exception as e:
        st.error(f"Mongo connection failed: {e}")
        return anth, None

anth, db = get_clients()
docs = db[DOCS_COLL] if db is not None else None
chat = db[CHAT_COLL] if db is not None else None

# -------- Storage helpers --------
def get_history(session_id: str, email: str, limit: int = 20) -> List[Dict]:
    if chat is None: return []
    return list(chat.find({"session_id": session_id, "email": email}).sort("ts", 1).limit(limit))

def save_msg(session_id: str, email: str, role: str, content: str) -> None:
    if chat is None: return
    chat.insert_one({
        "session_id": session_id,
        "email": email,
        "ts": datetime.datetime.utcnow(),
        "role": role,
        "content": content
    })

def search_docs(query: str, k: int = 5, topic: Optional[str] = None) -> List[Dict]:
    if docs is None: return []
    match: Dict = {"$text": {"$search": query}}
    if topic: match["topic"] = topic
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
        if r.get("section"): hdr += f" — {r['section']}"
        parts.append(hdr + "\n" + trunc(r.get("body", "")))
    return "\n\n".join(parts)

def build_messages(history_rows: List[Dict], context: str, question: str) -> List[Dict]:
    msgs: List[Dict] = []
    for h in history_rows:
        msgs.append({
            "role": "assistant" if h.get("role") == "assistant" else "user",
            "content": [{"type": "text", "text": h.get("content", "")}],
        })
    msgs.append({
        "role": "user",
        "content": [{"type": "text", "text": f"[CONTEXT]\n{context}\n[/CONTEXT]\n\nQuestion: {question}"}],
    })
    return msgs

def ask_claude(messages: List[Dict]) -> str:
    if not ANTHROPIC_API_KEY:
        return "Anthropic key not configured."
    system = (
        "You are an SEO coach. Use only the provided CONTEXT for facts. "
        "If context is weak, state what is missing. Return numbered, actionable steps. "
        "Cite like (source: file, section)."
    )
    try:
        resp = anth.messages.create(model=MODEL, max_tokens=800, system=system, messages=messages)
        return resp.content[0].text
    except APIStatusError as e:
        return f"Claude API error: {getattr(e, 'message', e)}"
    except Exception as e:
        return f"Claude call failed: {e}"

# -------- UI --------
with st.sidebar:
    st.markdown("**Session**")
    email = st.text_input("Email address", value=st.session_state.get("email",""))
    st.session_state["email"] = email
    sid = st.text_input("session_id", value=st.session_state.get("sid") or str(uuid.uuid4()))
    st.session_state["sid"] = sid
    k = st.number_input("Top-K docs", 1, 10, K_DEFAULT, 1)
    topic = st.text_input("Filter topic (optional)", value="SEO")
    st.markdown("**Diagnostics**")
    st.write(f"Mongo: {'OK' if db is not None else 'unreachable'}")
    st.write(f"Anthropic: {'OK' if ANTHROPIC_API_KEY else 'missing'}")

# require email
if not (email and EMAIL_RE.match(email)):
    st.info("Enter a valid email address to start."); st.stop()
if db is None:
    st.error("MongoDB not reachable. Check Atlas allowlist, URI, and password encoding."); st.stop()

# history
for h in get_history(sid, email, limit=50):
    with st.chat_message("assistant" if h.get("role") == "assistant" else "user"):
        st.write(h.get("content", ""))

# chat
user_msg = st.chat_input("Ask an SEO question…")
if user_msg:
    with st.chat_message("user"): st.write(user_msg)
    save_msg(sid, email, "user", user_msg)
    rows = search_docs(user_msg, k=k, topic=(topic or None))
    context = build_context(rows)
    reply = ask_claude(build_messages(get_history(sid, email, limit=20), context, user_msg))
    with st.chat_message("assistant"):
        st.write(reply)
        if rows:
            st.caption("Sources: " + ", ".join(
                f"{r.get('source')}{' • '+r.get('section') if r.get('section') else ''}" for r in rows
            ))
    save_msg(sid, email, "assistant", reply)
