import os, uuid, datetime, re, json
from typing import List, Dict, Optional
import streamlit as st
from urllib.parse import quote_plus
import certifi, requests
from anthropic import Anthropic, APIStatusError

st.set_page_config(page_title="SEO Coach", page_icon="🔍")
st.title("SEO Coach")

# -------- Secrets / Config (secrets only) --------
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY")

# Data API (HTTPS)
DATA_API_URL = st.secrets.get("DATA_API_URL") 
DATA_API_KEY = st.secrets.get("DATA_API_KEY")
DATA_API_SOURCE = st.secrets.get("DATA_API_SOURCE") or "Cluster0"

# PyMongo (driver). Only used if Data API not configured and egress:27017 is allowed.
MONGO_URI_SECRET = st.secrets.get("MONGO_URI")  # optional full URI in secrets
MONGO_USER = st.secrets.get("MONGO_USER")
MONGO_PASSWORD = st.secrets.get("MONGO_PASSWORD")
MONGO_HOST = st.secrets.get("MONGO_HOST") or "cluster0.bo4mikx.mongodb.net"
MONGO_DB = st.secrets.get("MONGO_DB") or "rag"
MONGO_APPNAME = st.secrets.get("MONGO_APPNAME") or "Cluster0"
MONGO_AUTH_SOURCE = st.secrets.get("MONGO_AUTH_SOURCE")  # optional (e.g., "admin")

USE_DATA_API = bool(DATA_API_URL and DATA_API_KEY)

DB_NAME, DOCS_COLL, CHAT_COLL = MONGO_DB, "docs", "chat"
MODEL = "claude-3-5-sonnet-latest"
K_DEFAULT, MAX_BODY_CHARS = 5, 1200

# -------- Clients --------
@st.cache_resource(show_spinner=False)
def get_clients():
    anth = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

    if USE_DATA_API:
        ok = False
        try:
            r = requests.post(
                f"{DATA_API_URL}/action/aggregate",
                headers={"Content-Type":"application/json","Accept":"application/json","api-key":DATA_API_KEY},
                json={"dataSource":DATA_API_SOURCE,"database":DB_NAME,"collection":DOCS_COLL,"pipeline":[{"$limit":1}]},
                timeout=10
            )
            ok = r.status_code in (200, 201)
        except Exception:
            ok = False
        return anth, {"mode":"data_api","ok":ok}

    # PyMongo mode (no passwords in code; all from st.secrets)
    from pymongo import MongoClient, ASCENDING
    from pymongo.server_api import ServerApi

    uri = MONGO_URI_SECRET
    if not uri and MONGO_USER and MONGO_PASSWORD:
        user = quote_plus(MONGO_USER)
        pwd = quote_plus(MONGO_PASSWORD)
        uri = f"mongodb+srv://{user}:{pwd}@{MONGO_HOST}/{MONGO_DB}?retryWrites=true&w=majority&appName={MONGO_APPNAME}"
        if MONGO_AUTH_SOURCE:
            uri += f"&authSource={quote_plus(MONGO_AUTH_SOURCE)}"
    if not uri:
        return anth, None

    try:
        mc = MongoClient(
            uri, server_api=ServerApi("1"), tls=True, tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=15000, connectTimeoutMS=15000, socketTimeoutMS=15000, appname="seo-coach"
        )
        mc.admin.command("ping")
        db = mc[DB_NAME]
        db.command("listCollections")  # force auth on target DB
        db[DOCS_COLL].create_index([("title","text"),("section","text"),("body","text")], name="kb_text_idx")
        db[CHAT_COLL].create_index([("session_id",1),("ts",1)], name="chat_session_ts")
        db[CHAT_COLL].create_index([("email",1)], name="chat_email_idx")
        return anth, {"mode":"pymongo","db":db}
    except Exception as e:
        st.warning(f"MongoDB not reachable/auth failed: {e}")
        return anth, None

anth, store = get_clients()

# -------- Storage adapters --------
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def da_headers():
    return {"Content-Type":"application/json","Accept":"application/json","api-key":DATA_API_KEY}

def da_insert_one(coll: str, doc: Dict) -> None:
    requests.post(f"{DATA_API_URL}/action/insertOne", headers=da_headers(),
                  json={"dataSource":DATA_API_SOURCE,"database":DB_NAME,"collection":coll,"document":doc}, timeout=10)

def da_find(coll: str, filt: Dict, sort: Dict=None, limit: int=20) -> List[Dict]:
    payload = {"dataSource":DATA_API_SOURCE,"database":DB_NAME,"collection":coll,"filter":filt,"limit":int(limit)}
    if sort: payload["sort"]=sort
    r = requests.post(f"{DATA_API_URL}/action/find", headers=da_headers(), json=payload, timeout=10)
    r.raise_for_status()
    return r.json().get("documents",[])

def da_aggregate(coll: str, pipeline: List[Dict]) -> List[Dict]:
    r = requests.post(f"{DATA_API_URL}/action/aggregate", headers=da_headers(),
                      json={"dataSource":DATA_API_SOURCE,"database":DB_NAME,"collection":coll,"pipeline":pipeline}, timeout=10)
    r.raise_for_status()
    return r.json().get("documents",[])

def get_history(session_id: str, email: str, limit: int = 20) -> List[Dict]:
    if not store: return []
    if store.get("mode") == "data_api":
        return da_find(CHAT_COLL, {"session_id":session_id,"email":email}, sort={"ts":1}, limit=limit)
    db = store["db"]
    return list(db[CHAT_COLL].find({"session_id":session_id,"email":email}).sort("ts",1).limit(limit))

def save_msg(session_id: str, email: str, role: str, content: str) -> None:
    if not store: return
    doc = {"session_id":session_id,"email":email,"ts":datetime.datetime.utcnow(),"role":role,"content":content}
    if store.get("mode") == "data_api":
        da_insert_one(CHAT_COLL, doc); return
    store["db"][CHAT_COLL].insert_one(doc)

def search_docs(query: str, k: int = 5, topic: Optional[str] = None) -> List[Dict]:
    if not store: return []
    match: Dict = {"$text":{"$search":query}}
    if topic: match["topic"]=topic
    pipeline = [
        {"$match": match},
        {"$addFields":{"score":{"$meta":"textScore"}}},
        {"$sort":{"score":-1}},
        {"$limit": int(k)},
        {"$project":{"_id":0,"source":1,"title":1,"section":1,"body":1,"score":1}},
    ]
    if store.get("mode") == "data_api":
        return da_aggregate(DOCS_COLL, pipeline)
    return list(store["db"][DOCS_COLL].aggregate(pipeline))

def build_context(rows: List[Dict]) -> str:
    def trunc(s: str) -> str:
        s = (s or "").strip()
        return s if len(s) <= MAX_BODY_CHARS else s[:MAX_BODY_CHARS] + "…"
    parts = []
    for i,r in enumerate(rows,1):
        hdr = f"[Doc {i}] {r.get('title') or r.get('source')}"
        if r.get("section"): hdr += f" — {r['section']}"
        parts.append(hdr+"\n"+trunc(r.get("body","")))
    return "\n\n".join(parts)

def build_messages(history_rows: List[Dict], context: str, question: str) -> List[Dict]:
    msgs=[]
    for h in history_rows:
        msgs.append({"role":"assistant" if h.get("role")=="assistant" else "user",
                     "content":[{"type":"text","text":h.get("content","")}]})
    msgs.append({"role":"user","content":[{"type":"text","text":f"[CONTEXT]\n{context}\n[/CONTEXT]\n\nQuestion: {question}"}]})
    return msgs

def ask_claude(messages: List[Dict]) -> str:
    if not ANTHROPIC_API_KEY: return "Anthropic key not configured."
    system=("You are an SEO coach. Use only the provided CONTEXT for facts. "
            "If context is weak, state what is missing. Return numbered, actionable steps. "
            "Cite like (source: file, section).")
    try:
        resp = Anthropic(api_key=ANTHROPIC_API_KEY).messages.create(model=MODEL, max_tokens=800, system=system, messages=messages)
        return resp.content[0].text
    except APIStatusError as e:
        return f"Claude API error: {getattr(e,'message',e)}"
    except Exception as e:
        return f"Claude call failed: {e}"

# -------- UI --------
with st.sidebar:
    st.markdown("**Session**")
    email = st.text_input("Email address", value=st.session_state.get("email",""))
    st.session_state["email"]=email
    sid = st.text_input("session_id", value=st.session_state.get("sid") or str(uuid.uuid4()))
    st.session_state["sid"]=sid
    k = st.number_input("Top-K docs", 1, 10, K_DEFAULT, 1)
    topic = st.text_input("Filter topic (optional)", value="SEO")
    st.markdown("**Diagnostics**")
    st.write(f"Mode: {'Data API' if USE_DATA_API else 'PyMongo'}")
    st.write(f"Anthropic: {'OK' if ANTHROPIC_API_KEY else 'missing'}")
    st.write(f"Store: {'OK' if store else 'unreachable'}")

if not (email and re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email)):
    st.info("Enter a valid email address to start."); st.stop()
if not store:
    st.error("Storage backend not reachable. Configure Data API or a reachable MongoDB."); st.stop()

for h in get_history(sid, email, limit=50):
    with st.chat_message("assistant" if h.get("role")=="assistant" else "user"):
        st.write(h.get("content",""))

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
            st.caption("Sources: " + ", ".join(f"{r.get('source')}{' • '+r.get('section') if r.get('section') else ''}" for r in rows))
    save_msg(sid, email, "assistant", reply)
