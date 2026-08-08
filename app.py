"""
Bajaj Finserv AI Customer Assistant — Streamlit Application.

Main entry point for the Streamlit-based chat interface.
Provides demo login, conversational AI with RAG-grounded responses,
and source reference display.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "customer_data.db"

# ── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Bajaj Finserv AI Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #003B8E 0%, #0052CC 40%, #1A73E8 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 4px 20px rgba(0, 59, 142, 0.25);
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        font-size: 0.85rem;
        opacity: 0.85;
        font-weight: 300;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #001D47 0%, #002E6E 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #E0E8F5 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }

    /* Customer info card */
    .customer-card {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.8rem 0;
        backdrop-filter: blur(10px);
    }
    .customer-card .label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8BAADB;
        margin-bottom: 0.15rem;
    }
    .customer-card .value {
        font-size: 0.9rem;
        font-weight: 600;
        color: #FFFFFF;
    }

    /* Source reference cards */
    .source-card {
        background: linear-gradient(135deg, #F0F7FF 0%, #E8F0FE 100%);
        border: 1px solid #C2D8F5;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.82rem;
    }
    .source-badge {
        display: inline-block;
        background: #003B8E;
        color: white;
        padding: 0.15rem 0.5rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .source-fund {
        font-weight: 600;
        color: #003B8E;
    }
    .source-section {
        color: #5A6F8A;
        font-size: 0.75rem;
    }

    /* Grounded badge */
    .grounded-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border: 1px solid #A5D6A7;
        color: #2E7D32;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }

    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px !important;
    }

    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 2rem;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Database Helpers ─────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_customer_details(identifier: str) -> dict | None:
    """Fetch full customer details by MobileNo or Name."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.execute(
        "SELECT * FROM customers WHERE MobileNo = ? OR Name = ? COLLATE NOCASE",
        (identifier, identifier)
    )
    row = cursor.fetchone()
    if row is None:
        conn.close()
        return None
    columns = [desc[0] for desc in cursor.description]
    conn.close()
    return dict(zip(columns, row))





# ── Agent Initialization ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner="🔧 Initializing AI Agent...")
def get_chat_agent():
    """Create and cache the ChatAgent (loaded once per Streamlit session)."""
    from agents.chat_agent import create_chat_agent
    return create_chat_agent()


# ── Session State Initialization ─────────────────────────────────────────────

def init_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "customer" not in st.session_state:
        st.session_state.customer = None
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []


init_session_state()


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    """Render the sidebar with login and session controls."""
    with st.sidebar:
        st.markdown("### 🏦 Bajaj Finserv")
        st.markdown("##### AI Customer Assistant")
        st.markdown("---")

        # ── Demo Login ───────────────────────────────────────────────
        if st.session_state.customer is None:
            st.markdown("#### 🔐 Demo Login")
            st.caption("Enter your Phone Number or Name to start chatting")

            login_input = st.text_input("Phone Number or Name", key="login_input")

            if st.button("🚀 Log In", use_container_width=True, type="primary"):
                if not login_input.strip():
                    st.error("Please enter your Phone Number or Name.")
                else:
                    details = get_customer_details(login_input.strip())
                    if details:
                        st.session_state.customer = {
                            "identifier": login_input.strip(),
                            "name": details["Name"],
                        }
                        st.session_state.messages = []  # Reset chat on new login
                        st.rerun()
                    else:
                        st.error("Customer not found. Please try again.")

        else:
            # ── Logged-in state ──────────────────────────────────────
            customer = st.session_state.customer
            st.markdown("#### ✅ Logged In")

            st.markdown(f"**Welcome, {customer['name']}!**")
            st.caption("Your details are securely available to the AI Assistant.")



            st.markdown("---")

            # Session controls
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.last_sources = []
                    st.rerun()
            with col2:
                if st.button("🚪 Log Out", use_container_width=True):
                    st.session_state.customer = None
                    st.session_state.messages = []
                    st.session_state.last_sources = []
                    st.rerun()

        # ── Footer ───────────────────────────────────────────────────
        st.markdown("---")
        st.caption("📊 Data: Factsheet June 2026")
        st.caption("🤖 Model: Llama 3.3 70B via Groq")
        st.caption("🔍 Traced via LangSmith")


# ── Source References Display ─────────────────────────────────────────────────

def render_sources(sources: list[dict]):
    """Render source reference cards below a response."""
    if not sources:
        return

    # Grounded badge
    st.markdown(
        '<div class="grounded-badge">✅ Factually Grounded — Bajaj Finserv Factsheet (June 2026)</div>',
        unsafe_allow_html=True,
    )

    with st.expander(f"📄 **Sources Used** ({len(sources)} references)", expanded=False):
        for src in sources:
            page = src.get("page_number", "?")
            fund = src.get("fund_name", "General")
            section = src.get("section", "unknown")
            elem_type = src.get("element_type", "text")
            score = src.get("relevance_score", 0)
            preview = src.get("preview", "")[:350]

            section_display = section.replace("_", " ").title()
            type_icon = "📊" if elem_type == "Table" else "📝"

            st.markdown(f"""
            <div class="source-card">
                <span class="source-badge">Page {page}</span>
                <span class="source-fund">{fund}</span>
                <br/>
                <span class="source-section">{type_icon} {section_display} · Relevance: {score:.2f}</span>
                <br/>
                <span style="color: #5A6F8A; font-size: 0.78rem;">{preview}...</span>
            </div>
            """, unsafe_allow_html=True)


# ── Main Chat Interface ──────────────────────────────────────────────────────

def render_chat():
    """Render the main chat area."""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Bajaj Finserv AI Assistant</h1>
        <p>Ask about mutual funds, loan details, and financial products — powered by RAG & Groq</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.customer is None:
        # Not logged in — show welcome
        st.info("👈 Please log in from the sidebar to start chatting.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 📊 Mutual Funds")
            st.markdown("Ask about Bajaj Finserv AMC fund details, NAV, returns, "
                       "expense ratios, and portfolio composition.")
        with col2:
            st.markdown("#### 💰 Loan Information")
            st.markdown("Query your loan account details — EMI, tenure, "
                       "outstanding balance, and more. *(Phase 2)*")
        with col3:
            st.markdown("#### 🎯 Cross-Sell")
            st.markdown("Get personalized mutual fund recommendations "
                       "based on your financial profile. *(Phase 2)*")
        return

    # ── Chat History ─────────────────────────────────────────────────
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show sources for assistant messages that used RAG
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources(msg["sources"])

    # ── Chat Input ───────────────────────────────────────────────────
    if prompt := st.chat_input("Ask about mutual funds, your loan, or financial products..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    agent = get_chat_agent()

                    # Build chat history (exclude sources metadata for LLM)
                    chat_history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]  # exclude current message
                    ]

                    result = agent.invoke(
                        user_input=prompt,
                        chat_history=chat_history,
                        customer_identifier=st.session_state.customer["identifier"],
                    )

                    response_text = result["response"]
                    sources = result.get("sources", [])

                    st.markdown(response_text)

                    if sources:
                        render_sources(sources)

                    # Save to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "sources": sources,
                        "used_rag": result.get("used_rag", False),
                    })

                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    logger.exception("Agent invocation failed")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": [],
                    })


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
else:
    # Streamlit runs the module directly, not via __main__
    main()