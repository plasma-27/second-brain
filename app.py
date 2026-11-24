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
)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None


@st.cache_resource
def get_resources(
    api_key: str,
    base_url: str,
    model_name: str,
    guardrail_model: str,
    embedding_model: str,
):
    db = SecondBrainDB(
        api_key=api_key,
        base_url=base_url,
        embedding_model=embedding_model,
        embedding_api_key=os.getenv("EMBEDDING_API_KEY") or api_key,
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL") or base_url,
    )
    graph = build_second_brain_graph(
        api_key=api_key,
        db=db,
        model_name=model_name,
        base_url=base_url,
        guardrail_model=guardrail_model,
    )
    return db, graph


# ---------------------------------------------------------
# Sidebar: Settings, Guardrails, HITL Toggle & Memory Explorer
# ---------------------------------------------------------
with st.sidebar:
    st.title("🧠 Second Brain")
    st.caption("Agentic RAG + Guardrails + Human-in-the-Loop")

    api_key = st.text_input(
        "API Key",
        value=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
        type="password",
    )
    base_url = st.text_input(
        "Base URL",
        value=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    model_name = st.text_input(
        "Chat Model",
        value=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    )
    guardrail_model = st.text_input(
        "Guardrail Model (same Base URL)",
        value=os.getenv("GUARDRAIL_MODEL", "gemini-2.5-flash-lite"),
    )
    embedding_model = st.text_input(
        "Embedding Model",
        value=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    )

    hitl_enabled = st.toggle(
        "Require approval before saving memories (HITL)",
        value=True,
        help="When off, HITL still pauses automatically if an existing memory is about to be overwritten.",
    )

    if st.button("🔄 New Chat Session", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.pending_interrupt = None
        st.rerun()

    st.divider()
    st.subheader("📚 Stored Memories")

    if api_key:
        try:
            db, graph = get_resources(api_key, base_url, model_name, guardrail_model, embedding_model)
            memories = db.list_memories()
            st.caption(f"Total memories indexed: **{len(memories)}**")

            filter_query = st.text_input("Filter memories", placeholder="e.g., laptop, python...")
            if filter_query:
                q = filter_query.lower()
                memories = [
                    m for m in memories
                    if q in m["title"].lower()
                    or q in m["summary"].lower()
                    or q in m["tags"].lower()
                    or q in m["url"].lower()
                ]

            if not memories:
                st.info("No memories saved yet.")
            else:
                for mem in memories:
                    icon = {"fact": "💡", "bookmark": "🔖", "note": "📝"}.get(mem["source_type"], "📝")
                    with st.expander(f"{icon} {mem['title']}"):
                        st.markdown(f"**Summary:** {mem['summary']}")
                        if mem["tags"]:
                            st.markdown(f"**Tags:** `{mem['tags']}`")
                        if mem["url"]:
                            st.markdown(f"**URL:** [{mem['url']}]({mem['url']})")
                        st.caption(f"ID: `{mem['memory_id']}` • Saved: {mem['created_at']}")
                        if st.button("🗑️ Delete", key=f"del_{mem['memory_id']}", use_container_width=True):
                            db.delete_memory(mem["memory_id"])
                            st.rerun()
        except Exception as e:
            st.error(f"Initialization error: {e}")
    else:
        st.warning("Enter your API key above to start.")

# ---------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------
st.header("💬 Talk to Your Second Brain")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("sources"):
            with st.expander(f"📎 Sources Referenced ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    line = f"- **{src['title']}** (`{src['source_type']}`, saved `{src['created_at']}`)"
                    if src["url"]:
                        line += f" — [Open Link]({src['url']})"
                    st.markdown(line)

        if msg.get("action_log"):
            with st.expander("⚙️ Agent Trace (Guardrails, HITL & LangGraph Steps)"):
                for step in msg["action_log"]:
                    st.markdown(f"- {step}")

# ---------------------------------------------------------
# Human-in-the-Loop (HITL) Pending Approval Panel
# ---------------------------------------------------------
if st.session_state.pending_interrupt and api_key:
    db, graph = get_resources(api_key, base_url, model_name, guardrail_model, embedding_model)
    payload = st.session_state.pending_interrupt

    with st.container(border=True):
        st.subheader("✋ Human-in-the-Loop Review Required")

        if payload.get("conflict_memory_id"):
            st.warning(
                f"This looks like an update to existing memory **`{payload['conflict_memory_id']}`** "
                f"({payload.get('conflict_reason', '')}). Confirm whether to overwrite it or keep both."
            )

        edited_title = st.text_input("Title", value=payload.get("title", ""))
        edited_summary = st.text_area("Summary / Resolved Date", value=payload.get("summary", ""), height=80)
        edited_content = st.text_area("Content to Index", value=payload.get("content", ""), height=100)
        edited_tags_str = st.text_input("Tags (comma-separated)", value=", ".join(payload.get("tags", [])))

        def _resume_graph(action_type: str):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            cleaned_tags = [t.strip() for t in edited_tags_str.split(",") if t.strip()]
            decision = {
                "action": action_type,
                "edited_title": edited_title,
                "edited_summary": edited_summary,
                "edited_content": edited_content,
                "edited_tags": cleaned_tags,
            }
            resumed = graph.invoke(Command(resume=decision), config=config)
            st.session_state.pending_interrupt = None
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": resumed["messages"][-1].content,
                "action_log": resumed.get("action_log", []),
                "sources": resumed.get("sources", []),
            })
            st.rerun()

        col1, col2, col3 = st.columns(3)
        with col1:
            label = "✅ Approve & Overwrite" if payload.get("conflict_memory_id") else "✅ Approve & Save"
            if st.button(label, type="primary", use_container_width=True):
                _resume_graph("approve")
        with col2:
            if payload.get("conflict_memory_id"):
                if st.button("📑 Keep Both Memories", use_container_width=True):
                    _resume_graph("keep_both")
        with col3:
            if st.button("❌ Reject / Cancel", use_container_width=True):
                _resume_graph("reject")

# ---------------------------------------------------------
# Chat Input (Disabled while waiting for HITL decision)
# ---------------------------------------------------------
prompt = st.chat_input(
    "Save a fact, paste a bookmark, or ask a question...",
    disabled=bool(st.session_state.pending_interrupt),
)

if prompt:
    if not api_key:
        st.error("Please enter your API key in the sidebar first.")
        st.stop()

    db, graph = get_resources(api_key, base_url, model_name, guardrail_model, embedding_model)

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Running guardrails & LangGraph..."):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=prompt)],
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

    # Check if LangGraph paused at human_review_node
    if "__interrupt__" in result and result["__interrupt__"]:
        st.session_state.pending_interrupt = result["__interrupt__"][0].value
        st.rerun()
    else:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["messages"][-1].content,
            "action_log": result.get("action_log", []),
            "sources": result.get("sources", []),
        })
        st.rerun()