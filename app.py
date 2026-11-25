import html
import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from db import SecondBrainDB
from graph import build_second_brain_graph

load_dotenv()

st.set_page_config(
    page_title="Second Brain — Personal Agentic RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Tighten top padding for a modern app feel */
    .block-container {
        padding-top: 1.75rem;
        padding-bottom: 2rem;
        max-width: 980px;
    }

    /* Status chips in the main header */
    .status-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: -0.35rem;
        margin-bottom: 1.25rem;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
        background: rgba(128, 128, 128, 0.12);
        border: 1px solid rgba(128, 128, 128, 0.22);
    }

    /* Tag pills inside Memory Explorer */
    .tag-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.3rem;
        margin: 0.35rem 0 0.5rem 0;
    }
    .tag-pill {
        padding: 0.12rem 0.5rem;
        border-radius: 999px;
        font-size: 0.72rem;
        background: rgba(99, 102, 241, 0.14);
        border: 1px solid rgba(99, 102, 241, 0.28);
    }

    /* Compact stat cards in sidebar */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.45rem;
        margin: 0.4rem 0 0.8rem 0;
    }
    .stat-card {
        text-align: center;
        padding: 0.45rem 0.25rem;
        border-radius: 0.6rem;
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.16);
    }
    .stat-num {
        font-size: 1.05rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .stat-lbl {
        font-size: 0.7rem;
        opacity: 0.75;
    }

    /* Empty-state welcome banner */
    .welcome-box {
        padding: 1.5rem 1.75rem;
        border-radius: 0.9rem;
        background: rgba(99, 102, 241, 0.06);
        border: 1px solid rgba(99, 102, 241, 0.18);
        margin-bottom: 1rem;
    }
    .welcome-box h4 {
        margin: 0 0 0.35rem 0;
    }
    .welcome-box p {
        margin: 0;
        opacity: 0.85;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Session State & Config Defaults (Loaded from .env)
# ---------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None

if "cfg" not in st.session_state:
    st.session_state.cfg = {
        "api_key": os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
        "base_url": os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        "model_name": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "guardrail_model": os.getenv("GUARDRAIL_MODEL", "gemini-2.5-flash-lite"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "embedding_base_url": os.getenv("EMBEDDING_BASE_URL", ""),
    }


def mask_key(key: str) -> str:
    if not key:
        return "Missing (set LLM_API_KEY in .env)"
    if len(key) <= 8:
        return "Configured (••••)"
    return f"{key[:4]}••••{key[-4:]}"


def build_config_tooltip(cfg: dict) -> str:
    emb_url = cfg["embedding_base_url"] or cfg["base_url"]
    return (
        f"**Active Configuration (`.env`)**  \n"
        f"- **API Key:** `{mask_key(cfg['api_key'])}`  \n"
        f"- **Base URL:** `{cfg['base_url']}`  \n"
        f"- **Chat Model:** `{cfg['model_name']}`  \n"
        f"- **Guardrail Model:** `{cfg['guardrail_model']}`  \n"
        f"- **Embedding Model:** `{cfg['embedding_model']}`  \n"
        f"- **Embedding URL:** `{emb_url}`"
    )


@st.cache_resource
def get_resources(
    api_key: str,
    base_url: str,
    model_name: str,
    guardrail_model: str,
    embedding_model: str,
    embedding_base_url: str,
):
    db = SecondBrainDB(
        api_key=api_key,
        base_url=base_url,
        embedding_model=embedding_model,
        embedding_api_key=os.getenv("EMBEDDING_API_KEY") or api_key,
        embedding_base_url=embedding_base_url or base_url,
    )
    graph = build_second_brain_graph(
        api_key=api_key,
        db=db,
        model_name=model_name,
        base_url=base_url,
        guardrail_model=guardrail_model,
    )
    return db, graph


cfg = st.session_state.cfg


def load_active_resources():
    return get_resources(
        cfg["api_key"],
        cfg["base_url"],
        cfg["model_name"],
        cfg["guardrail_model"],
        cfg["embedding_model"],
        cfg["embedding_base_url"],
    )


# ---------------------------------------------------------
# Sidebar: Compact Header (? Info + Gear Override) & Memory Explorer
# ---------------------------------------------------------
with st.sidebar:
    title_col, gear_col = st.columns([5, 1], vertical_alignment="center")
    with title_col:
        # Hovering on the "?" icon displays all .env model/endpoint details
        st.subheader("🧠 Second Brain", help=build_config_tooltip(cfg))
    with gear_col:
        with st.popover("⚙️", help="Override .env settings"):
            st.caption("Temporarily override `.env` settings for this session:")
            new_key = st.text_input("API Key", value=cfg["api_key"], type="password")
            new_url = st.text_input("Base URL", value=cfg["base_url"])
            new_model = st.text_input("Chat Model", value=cfg["model_name"])
            new_guard = st.text_input("Guardrail Model", value=cfg["guardrail_model"])
            new_emb = st.text_input("Embedding Model", value=cfg["embedding_model"])
            new_emb_url = st.text_input(
                "Embedding Base URL (optional)",
                value=cfg["embedding_base_url"],
                placeholder="Defaults to Base URL",
            )
            if st.button("Apply Settings", type="primary", use_container_width=True):
                st.session_state.cfg = {
                    "api_key": new_key.strip(),
                    "base_url": new_url.strip(),
                    "model_name": new_model.strip(),
                    "guardrail_model": new_guard.strip(),
                    "embedding_model": new_emb.strip(),
                    "embedding_base_url": new_emb_url.strip(),
                }
                st.rerun()

    # Compact Session Controls
    hitl_enabled = st.toggle(
        "Approve before saving (HITL)",
        value=True,
        help="When off, HITL still pauses automatically before overwriting a conflicting memory.",
    )

    if st.button("✨ New Chat", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.pending_interrupt = None
        st.rerun()

    st.divider()

    # Memory Explorer
    st.markdown("#### 📚 Knowledge Base")

    if cfg["api_key"]:
        try:
            db, graph = load_active_resources()
            all_memories = db.list_memories()

            # Category Stats
            fact_cnt = sum(1 for m in all_memories if m["source_type"] == "fact")
            bmk_cnt = sum(1 for m in all_memories if m["source_type"] == "bookmark")
            note_cnt = sum(1 for m in all_memories if m["source_type"] == "note")

            st.markdown(
                f"""
                <div class="stat-grid">
                    <div class="stat-card"><div class="stat-num">{fact_cnt}</div><div class="stat-lbl">💡 Facts</div></div>
                    <div class="stat-card"><div class="stat-num">{bmk_cnt}</div><div class="stat-lbl">🔖 Links</div></div>
                    <div class="stat-card"><div class="stat-num">{note_cnt}</div><div class="stat-lbl">📝 Notes</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            filter_query = st.text_input(
                "Search memories",
                placeholder="🔍 Filter by keyword, tag, or URL...",
                label_visibility="collapsed",
            )

            selected_type = st.pills(
                "Category",
                options=["All", "Fact", "Bookmark", "Note"],
                default="All",
                label_visibility="collapsed",
            )
            selected_type = selected_type or "All"

            filtered = all_memories
            if selected_type != "All":
                filtered = [m for m in filtered if m["source_type"] == selected_type.lower()]

            if filter_query:
                q = filter_query.lower()
                filtered = [
                    m for m in filtered
                    if q in m["title"].lower()
                    or q in m["summary"].lower()
                    or q in m["tags"].lower()
                    or q in m["url"].lower()
                ]

            if not filtered:
                st.caption("No matching memories yet.")
            else:
                st.caption(f"Showing **{len(filtered)}** of {len(all_memories)} memories")
                for mem in filtered:
                    icon = {"fact": "💡", "bookmark": "🔖", "note": "📝"}.get(mem["source_type"], "📝")
                    with st.expander(f"{icon} {mem['title']}"):
                        st.markdown(mem["summary"])

                        if mem["tags"]:
                            tags_html = "".join(
                                f'<span class="tag-pill">#{html.escape(t.strip())}</span>'
                                for t in mem["tags"].split(",")
                                if t.strip()
                            )
                            st.markdown(f'<div class="tag-row">{tags_html}</div>', unsafe_allow_html=True)

                        if mem["url"]:
                            st.markdown(f"🔗 [{mem['url']}]({mem['url']})")

                        meta_col, del_col = st.columns([3, 1], vertical_alignment="center")
                        with meta_col:
                            saved_short = mem["created_at"].replace("T", " ")[:16]
                            st.caption(f"`{mem['memory_id']}` • {saved_short}")
                        with del_col:
                            if st.button("🗑️", key=f"del_{mem['memory_id']}", help="Delete memory"):
                                db.delete_memory(mem["memory_id"])
                                st.rerun()
        except Exception as e:
            st.error(f"Initialization error: {e}")
    else:
        st.warning("API key missing. Add `LLM_API_KEY` to `.env` or click **⚙️** above.")


# ---------------------------------------------------------
# Core Execution Helper
# ---------------------------------------------------------
def run_user_prompt(prompt_text: str):
    if not cfg["api_key"]:
        st.error("API key is missing. Add `LLM_API_KEY` in `.env` or click ⚙️ in the sidebar.")
        st.stop()

    _, active_graph = load_active_resources()
    st.session_state.chat_history.append({"role": "user", "content": prompt_text})

    with st.spinner("Thinking..."):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        result = active_graph.invoke(
            {
                "messages": [HumanMessage(content=prompt_text)],
                "intent": "",
                "input_safe": True,
                "guardrail_reason": "",
                "extracted_url": None,
                "enriched_data": None,
                "conflict_memory_id": None,
                "conflict_reason": "",
                "hitl_enabled": hitl_enabled,
                "human_decision": None,
                "search_query": "",
                "retrieved_docs": [],
                "is_relevant": False,
                "retry_count": 0,
                "action_log": [],
                "sources": [],
            },
            config=config,
        )

    if "__interrupt__" in result and result["__interrupt__"]:
        st.session_state.pending_interrupt = result["__interrupt__"][0].value
    else:
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result["messages"][-1].content,
                "action_log": result.get("action_log", []),
                "sources": result.get("sources", []),
            }
        )
    st.rerun()


# ---------------------------------------------------------
# Main Chat Area Header
# ---------------------------------------------------------
st.title("💬 Second Brain")
hitl_label = "HITL: Always Confirm" if hitl_enabled else "HITL: Conflicts Only"
st.markdown(
    f"""
    <div class="status-bar">
        <span class="status-pill">🛡️ Guardrails Active</span>
        <span class="status-pill">✋ {hitl_label}</span>
        <span class="status-pill">⚡ LangGraph CRAG</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Empty-State Welcome & Clickable Prompts
if not st.session_state.chat_history and not st.session_state.pending_interrupt:
    st.markdown(
        """
        <div class="welcome-box">
            <h4>Store anything. Recall instantly.</h4>
            <p>Drop a fact, reminder, or web link to index it into your personal memory — or ask a question to retrieve grounded answers with citations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button(
            "💡 Remember my passport expires on Nov 18, 2029",
            use_container_width=True,
            disabled=not bool(cfg["api_key"]),
        ):
            run_user_prompt("Remember that my passport expires on Nov 18, 2029.")
    with c2:
        if st.button(
            "🔖 Save Python 3.12 release notes link",
            use_container_width=True,
            disabled=not bool(cfg["api_key"]),
        ):
            run_user_prompt("Save this link for later: https://docs.python.org/3/whatsnew/3.12.html")
    with c3:
        if st.button(
            "❓ What reminders or deadlines have I saved?",
            use_container_width=True,
            disabled=not bool(cfg["api_key"]),
        ):
            run_user_prompt("What reminders or deadlines have I saved?")

# Render Chat History
for msg in st.session_state.chat_history:
    avatar = "🧑" if msg["role"] == "user" else "🧠"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

        if msg.get("sources"):
            with st.expander(f"📎 Sources ({len(msg['sources'])})", expanded=False):
                for src in msg["sources"]:
                    saved_short = src["created_at"].replace("T", " ")[:16]
                    line = f"- **{src['title']}** (`{src['source_type']}` • {saved_short})"
                    if src["url"]:
                        line += f" — [Open Link]({src['url']})"
                    st.markdown(line)

        if msg.get("action_log"):
            with st.expander("⚙️ Agent Trace", expanded=False):
                for step in msg["action_log"]:
                    st.markdown(f"- {step}")

# ---------------------------------------------------------
# Human-in-the-Loop (HITL) Review Card
# ---------------------------------------------------------
if st.session_state.pending_interrupt and cfg["api_key"]:
    _, active_graph = load_active_resources()
    payload = st.session_state.pending_interrupt

    with st.container(border=True):
        st.markdown("#### ✋ Review Before Saving")

        if payload.get("conflict_memory_id"):
            st.warning(
                f"Updates existing memory **`{payload['conflict_memory_id']}`** — "
                f"{payload.get('conflict_reason', '')}"
            )

        r1c1, r1c2 = st.columns([2, 1])
        with r1c1:
            edited_title = st.text_input("Title", value=payload.get("title", ""))
        with r1c2:
            edited_tags_str = st.text_input(
                "Tags (comma-separated)",
                value=", ".join(payload.get("tags", [])),
            )

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            edited_summary = st.text_area(
                "Summary / Resolved Date",
                value=payload.get("summary", ""),
                height=100,
            )
        with r2c2:
            edited_content = st.text_area(
                "Content to Index",
                value=payload.get("content", ""),
                height=100,
            )

        def _resume_graph(action_type: str):
            with st.spinner("Applying your decision..."):
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                cleaned_tags = [t.strip() for t in edited_tags_str.split(",") if t.strip()]
                decision = {
                    "action": action_type,
                    "edited_title": edited_title,
                    "edited_summary": edited_summary,
                    "edited_content": edited_content,
                    "edited_tags": cleaned_tags,
                }
                resumed = active_graph.invoke(Command(resume=decision), config=config)
                st.session_state.pending_interrupt = None
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": resumed["messages"][-1].content,
                        "action_log": resumed.get("action_log", []),
                        "sources": resumed.get("sources", []),
                    }
                )
            st.rerun()

        b1, b2, b3 = st.columns(3)
        with b1:
            approve_label = "✅ Approve & Overwrite" if payload.get("conflict_memory_id") else "✅ Approve & Save"
            if st.button(approve_label, type="primary", use_container_width=True):
                _resume_graph("approve")
        with b2:
            if payload.get("conflict_memory_id"):
                if st.button("📑 Keep Both", use_container_width=True):
                    _resume_graph("keep_both")
        with b3:
            if st.button("❌ Cancel", use_container_width=True):
                _resume_graph("reject")

# ---------------------------------------------------------
# Chat Input
# ---------------------------------------------------------
prompt = st.chat_input(
    "Save a fact, paste a link, or ask your Second Brain...",
    disabled=bool(st.session_state.pending_interrupt),
)

if prompt:
    run_user_prompt(prompt)