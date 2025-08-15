import os, uuid, datetime, re
from typing import List, Dict, Optional
import streamlit as st
from urllib.parse import quote_plus
from pymongo import MongoClient, ASCENDING
from pymongo.server_api import ServerApi
from anthropic import Anthropic, APIStatusError
import certifi

st.set_page_config(page_title="SEO Coach", page_icon="🔍")
st.title("SEO Coach")

# ---- secrets ----
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

# Preferred: provide unencoded pieces; we build a safe URI.
MONGO_USER = st.secrets.get("MONGO_USER") or os.environ.get("MONGO_USER")
MONGO_PASSWORD = st.secrets.get("MONGO_PASSWORD") or os.environ.get("MONGO_PASSWORD")
MONGO_HOST = st.secrets.get("MONGO_HOST") or os.environ.get("MONGO_HOST") or "cluster0.bo4mikx.mongodb.net"
MONGO_DB = st.secrets.get("MONGO_DB") or os.environ.get("MONGO_DB") or "rag"
MONGO_APPNAME = st.secrets.get("MONGO_APPNAME") or os.environ.get("MONGO_APPNAME") or "Cluster0"

# Fallback: allow a prebuilt URI, but pieces take precedence.
PREBUILT_URI = st.secrets.get("MONGO_URI") or os.environ.get("MONGO_URI")

def build_mongo_uri() -> Optional[str]:
    if MONGO_USER and MONGO_PASSWORD:
        user = quote_plus(MONGO_USER)
        pwd = quote_plus(MONGO_PASSWORD)
        return f"mongodb+srv://{user}:{pwd}@{MONGO_HOST}/{MONGO_DB}?retryWrites=true&w=majority&appName={MONGO_APPNAME}"
    return PREBUILT_URI

MONGO_URI = build_mongo_uri()

DB_NAME, DOCS_COLL, CHAT_COLL = MONGO_DB, "docs", "chat"
MODEL = "claude-3-5-sonnet-latest"
K_DEFAULT, MAX_BODY_CHARS = 5, 1200

# 1) Raw TLS handshake to one shard host
import socket, ssl, certifi
host = "ac-ocnld0l-shard-00-00.bo4mikx.mongodb.net"
ctx = ssl.create_default_context(cafile=certifi.where())
s = socket.create_connection((host, 27017), timeout=10)
ss = ctx.wrap_socket(s, server_hostname=host)
print("TLS OK:", ss.version()); ss.close()

# 2) Force real auth on DB after fixing allowlist/URI
from pymongo import MongoClient
from pymongo.server_api import ServerApi
mc = MongoClient(os.environ["MONGO_URI"], server_api=ServerApi("1"),
                 tls=True, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=15000)
print(mc.admin.command("ping"))
print(mc["rag"].list_collection_names())


# ---- cached clients (deferred init; TLS CA; early auth) ----
@st.cache_resource(show_spinner=False)
def get_clients():
    anth = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
    db = None
    if MONGO_URI:
        insecure = (st.secrets.get("MONGO_INSECURE_TLS") or os.environ.get("MONGO_INSECURE_TLS")) == "1"
        mc = MongoClient(
            MONGO_URI,
            server_api=ServerApi("1"),
            tls=True,
            tlsCAFile=None if insecure else certifi.where(),
            tlsAllowInvalidCertificates=insecure,
            serverSelectionTimeoutMS=15000,
            connectTimeoutMS=15000,
            socketTimeoutMS=15000,
            appname="seo-coach",
        )
        try:
            mc.admin.command("ping")          # network + handshake
            db = mc[DB_NAME]
            db.command("listCollections")     # force auth on target DB
            # Indexes (idempotent)
            db[DOCS_COLL].create_index([("title", "text"), ("section", "text"), ("body", "text")], name="kb_text_idx")
            db[CHAT_COLL].create_index([("session_id", ASCENDING), ("ts", ASCENDING)], name="chat_session_ts")
            db[CHAT_COLL].create_index([("email", ASCENDING)], name="chat_email_idx")
        except Exception as e:
            st.warning(f"MongoDB not reachable/auth failed: {e}")
            db = None
    return anth, db

anth, db = get_clients()
docs = db[DOCS_COLL] if db is not None else None
chat = db[CHAT_COLL] if db is not None else None

# ---- helpers ----
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

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
    if anth is None: return "Anthropic key not configured."
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

# ---- sidebar / session + email gate + diagnostics ----
with st.sidebar:
    st.markdown("**Session**")
    email = st.text_input("Email address", value=st.session_state.get("email", ""))
    st.session_state["email"] = email
    sid = st.text_input("session_id", value=st.session_state.get("sid") or str(uuid.uuid4()))
    st.session_state["sid"] = sid
    k = st.number_input("Top-K docs", min_value=1, max_value=10, value=K_DEFAULT, step=1)
    topic = st.text_input("Filter topic (optional)", value="SEO")
    st.markdown("**Diagnostics**")
    st.write(f"Anthropic: {'OK' if anth else 'missing'}")
    st.write(f"MongoDB: {'OK' if db is not None else 'unreachable'}")
    if not MONGO_URI:
        st.warning("Mongo URI is not built. Set MONGO_USER and MONGO_PASSWORD in secrets.")
    if not ANTHROPIC_API_KEY:
        st.warning("ANTHROPIC_API_KEY is not set.")

# require email before continuing
if not (email and EMAIL_RE.match(email)):
    st.info("Enter a valid email address to start.")
    st.stop()

# ---- history ----
for h in get_history(sid, email, limit=50):
    with st.chat_message("assistant" if h.get("role") == "assistant" else "user"):
        st.write(h.get("content", ""))

# ---- chat input ----
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
