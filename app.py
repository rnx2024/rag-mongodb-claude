import os, uuid, datetime, textwrap
import streamlit as st
from pymongo import MongoClient, ASCENDING, TEXT
from anthropic import Anthropic

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

# --- ensure collections/indexes (safe if already exist) ---
try:
    docs.create_index([("title", TEXT), ("section", TEXT), ("body", TEXT)], name="kb_text_idx")
except Exception:
    pass
chat.create_index([("session_id", ASCENDING), ("ts", ASCENDING)], name="chat_session_ts", unique=False)

# --- helpers ---
def get_history(session_id: str, limit: int = 20):
    cur = chat.find({"session_id": session_id}).sort("ts", 1).limit(limit)
    return list(cur)

def save_msg(session_id: str, role: str, content: str):
    chat.insert_one({"session_id": session_id, "ts": datetime.datetime.utcnow(), "role": role, "content": content})

def search_docs(query: str, k: int = 5, topic: str | None = None):
    match = {"$text": {"$search": query}}
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

def build_context(rows: list[dict]):
    def trunc(s: str) -> str:
        s = s.strip()
        return s if len(s) <= MAX_BODY_CHARS else s[:MAX_BODY_CHARS] + "…"
    parts = []
    for i, r in enumerate(rows, 1):
        hdr = f"[Doc {i}] {r.get('title') or r.get('source')}"
        if r.get("section"):
            hdr += f" — {r['section']}"
        parts.append(hdr + "\n" + trunc(r.get("body", "")))
    return "\n\n".join(parts)

def build_messages(history_rows, context: str, question: str):
    msgs = []
    for h in history_rows:
        msgs.append({
            "role": "assistant" if h["role"] == "assistant" else "user",
            "content": [{"type": "text", "text": h["content"]}],
        })
    msgs.append({
        "role": "user",
        "content": [{"type": "text", "text": f"[CONTEXT]\n{context}\n[/CONTEXT]\n\nQuestion: {question}"}],
    })
    return msgs

def ask_claude(messages):
    system = (
        "You are an SEO coach. Use only the provided CONTEXT for facts. "
        "If context is weak, state what is missing. Return numbered, actionable steps. "
        "Cite like (source: file, section)."
    )
    resp = anth.messages.create(model=MODEL, max_tokens=800, system=system, messages=messages)
    return resp.content[0].text

# --- UI ---
st.set_page_config(page_title="SEO Coach (Claude + MongoDB)", page_icon="🔍")
st.title("SEO Coach")

with st.sidebar:
    st.markdown("**Session**")
    sid = st.text_input("session_id", value=st.session_state.get("sid") or str(uuid.uuid4()))
    st.session_state["sid"] = sid
    k = st.number_input("Top-K docs", min_value=1, max_value=10, value=K_DEFAULT, step=1)
    topic = st.text_input("Filter topic (optional)", value="SEO")

# show history
history = get_history(sid, limit=50)
for h in history:
    with st.chat_message("assistant" if h["role"] == "assistant" else "user"):
        st.write(h["content"])

# input
user_msg = st.chat_input("Ask an SEO question…")
if user_msg:
    # show and store user message
    with st.chat_message("user"):
        st.write(user_msg)
    save_msg(sid, "user", user_msg)

    # retrieve KB
    rows = search_docs(user_msg, k=k, topic=(topic or None))
    context = build_context(rows)

    # call Claude
    messages = build_messages(get_history(sid, limit=20), context, user_msg)
    reply = ask_claude(messages)

    # show and store reply
    with st.chat_message("assistant"):
        st.write(reply)
        if rows:
            st.caption("Sources: " + ", ".join(
                f"{r.get('source')}{' • '+r.get('section') if r.get('section') else ''}" for r in rows
            ))
    save_msg(sid, "assistant", reply)
