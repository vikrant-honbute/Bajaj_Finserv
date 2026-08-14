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

from agents.cross_sell_engine import evaluate_cross_sell_eligibility

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

    /* ── Recommendation Cards ───────────────────────────────── */

    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .reco-section {
        animation: fadeSlideIn 0.6s ease-out;
        margin-bottom: 1.8rem;
    }

    .reco-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 1rem;
    }
    .reco-header h3 {
        margin: 0;
        font-size: 1.15rem;
        font-weight: 700;
        background: linear-gradient(135deg, #003B8E, #1A73E8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .reco-risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .reco-risk-badge.conservative {
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        color: #2E7D32;
        border: 1px solid #A5D6A7;
    }
    .reco-risk-badge.moderate {
        background: linear-gradient(135deg, #FFF3E0, #FFE0B2);
        color: #E65100;
        border: 1px solid #FFCC80;
    }
    .reco-risk-badge.aggressive {
        background: linear-gradient(135deg, #FFEBEE, #FFCDD2);
        color: #C62828;
        border: 1px solid #EF9A9A;
    }

    .reco-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1rem;
    }

    .reco-card {
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(0, 59, 142, 0.12);
        border-radius: 14px;
        padding: 1.3rem 1.4rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        animation: fadeSlideIn 0.6s ease-out backwards;
    }
    .reco-card:nth-child(1) { animation-delay: 0.1s; }
    .reco-card:nth-child(2) { animation-delay: 0.25s; }
    .reco-card:nth-child(3) { animation-delay: 0.4s; }

    .reco-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #003B8E, #1A73E8, #4FC3F7);
    }

    .reco-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(0, 59, 142, 0.15);
        border-color: rgba(0, 59, 142, 0.25);
    }

    .reco-card-category {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 6px;
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .reco-card-category.equity { background: #E3F2FD; color: #1565C0; }
    .reco-card-category.hybrid { background: #F3E5F5; color: #7B1FA2; }
    .reco-card-category.fixed-income { background: #E8F5E9; color: #2E7D32; }
    .reco-card-category.index { background: #FFF3E0; color: #E65100; }
    .reco-card-category.etf { background: #FFEBEE; color: #C62828; }

    .reco-card-name {
        font-size: 0.95rem;
        font-weight: 700;
        color: #1A1A2E;
        margin-bottom: 0.4rem;
        line-height: 1.3;
    }

    .reco-card-desc {
        font-size: 0.78rem;
        color: #5A6F8A;
        line-height: 1.45;
        margin-bottom: 0.8rem;
    }

    .reco-card-sip {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.5rem 0.7rem;
        background: linear-gradient(135deg, #F0F7FF, #E3EDFA);
        border-radius: 8px;
        margin-bottom: 0.7rem;
    }
    .reco-card-sip-label {
        font-size: 0.68rem;
        color: #5A6F8A;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .reco-card-sip-value {
        font-size: 1rem;
        font-weight: 700;
        color: #003B8E;
    }

    .reco-why {
        background: linear-gradient(135deg, #FAFBFF, #F0F4FA);
        border: 1px solid #E0E8F5;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-top: 0.8rem;
        animation: fadeSlideIn 0.5s ease-out 0.5s backwards;
    }
    .reco-why-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: #003B8E;
        margin-bottom: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .reco-why-item {
        font-size: 0.78rem;
        color: #4A5568;
        padding: 0.15rem 0;
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


def get_customer_all_rows(identifier: str) -> list[dict]:
    """Fetch ALL rows for a customer (they may have multiple loans)."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        "SELECT * FROM customers WHERE MobileNo = ? OR Name = ? COLLATE NOCASE",
        (identifier, identifier),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def run_cross_sell_for_customer(identifier: str) -> dict | None:
    """
    Run the cross-sell engine for a customer.

    Checks ALL of the customer's loans. If ANY active loan qualifies,
    the customer is eligible. Returns the best CrossSellResult or None.
    """
    rows = get_customer_all_rows(identifier)
    if not rows:
        return None

    best_result = None
    for row in rows:
        result = evaluate_cross_sell_eligibility(row)
        if result.eligible:
            # Pick the result with the highest SIP (best profile)
            if best_result is None or result.estimated_monthly_sip > best_result.estimated_monthly_sip:
                best_result = result

    if best_result:
        return best_result.model_dump()

    # If no row qualifies, return None (no recommendations shown)
    return None



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
    if "cross_sell_result" not in st.session_state:
        st.session_state.cross_sell_result = None


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
                        # Run cross-sell evaluation on login
                        st.session_state.cross_sell_result = run_cross_sell_for_customer(
                            login_input.strip()
                        )
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
                    st.session_state.cross_sell_result = None
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

# ── Recommendation Cards ─────────────────────────────────────────────────────

# Map fund names to categories for badge coloring
FUND_CATEGORY_MAP = {
    "Bajaj Finserv Flexi Cap Fund": "Equity",
    "Bajaj Finserv Large Cap Fund": "Equity",
    "Bajaj Finserv Large and Mid Cap Fund": "Equity",
    "Bajaj Finserv Multi Cap Fund": "Equity",
    "Bajaj Finserv Small Cap Fund": "Equity",
    "Bajaj Finserv ELSS Tax Saver Fund": "Equity",
    "Bajaj Finserv Healthcare Fund": "Equity",
    "Bajaj Finserv Consumption Fund": "Equity",
    "Bajaj Finserv Banking and Financial Services Fund": "Equity",
    "Bajaj Finserv Balanced Advantage Fund": "Hybrid",
    "Bajaj Finserv Multi Asset Allocation Fund": "Hybrid",
    "Bajaj Finserv Equity Savings Fund": "Hybrid",
    "Bajaj Finserv Arbitrage Fund": "Hybrid",
    "Bajaj Finserv Liquid Fund": "Fixed Income",
    "Bajaj Finserv Money Market Fund": "Fixed Income",
    "Bajaj Finserv Gilt Fund": "Fixed Income",
    "Bajaj Finserv Overnight Fund": "Fixed Income",
    "Bajaj Finserv Banking and PSU Fund": "Fixed Income",
    "Bajaj Finserv Low Duration Fund": "Fixed Income",
    "Bajaj Finserv Nifty 50 Index Fund": "Index",
    "Bajaj Finserv Nifty Next 50 Index Fund": "Index",
    "Bajaj Finserv Nifty 50 ETF": "ETF",
    "Bajaj Finserv Nifty Bank ETF": "ETF",
    "Bajaj Finserv Nifty 1D Rate Liquid ETF": "ETF",
}

# Short taglines for each fund (used in reco cards)
FUND_TAGLINES = {
    "Bajaj Finserv Flexi Cap Fund": "Flexible allocation across large, mid & small caps for optimal growth potential.",
    "Bajaj Finserv Large Cap Fund": "Stability-focused investing in India's top 100 blue-chip companies.",
    "Bajaj Finserv Large and Mid Cap Fund": "Balanced exposure to established large caps and high-growth mid caps.",
    "Bajaj Finserv Multi Cap Fund": "Diversified across market capitalizations for all-weather performance.",
    "Bajaj Finserv Small Cap Fund": "High-growth potential through carefully selected emerging companies.",
    "Bajaj Finserv ELSS Tax Saver Fund": "Tax-saving under Section 80C with equity growth — 3-year lock-in.",
    "Bajaj Finserv Healthcare Fund": "Thematic exposure to India's booming healthcare & pharma sector.",
    "Bajaj Finserv Consumption Fund": "Ride India's consumption story through domestic demand-driven sectors.",
    "Bajaj Finserv Banking and Financial Services Fund": "Concentrated bet on India's financial sector growth.",
    "Bajaj Finserv Balanced Advantage Fund": "Dynamic asset allocation between equity & debt for downside protection.",
    "Bajaj Finserv Multi Asset Allocation Fund": "Diversified across equity, debt, and commodities for reduced volatility.",
    "Bajaj Finserv Equity Savings Fund": "Conservative hybrid with equity upside and debt stability.",
    "Bajaj Finserv Arbitrage Fund": "Low-risk arbitrage opportunities with near-debt-like returns.",
    "Bajaj Finserv Liquid Fund": "Park surplus cash with high liquidity and minimal risk.",
    "Bajaj Finserv Money Market Fund": "Short-term money market instruments for stable, predictable returns.",
    "Bajaj Finserv Gilt Fund": "Sovereign-backed government securities — zero credit risk.",
    "Bajaj Finserv Overnight Fund": "Ultra-safe overnight lending for emergency fund parking.",
    "Bajaj Finserv Banking and PSU Fund": "High-quality bank & PSU bonds for steady fixed-income returns.",
    "Bajaj Finserv Low Duration Fund": "Short-duration debt strategy for better-than-savings returns.",
    "Bajaj Finserv Nifty 50 Index Fund": "Low-cost passive exposure to India's top 50 companies.",
    "Bajaj Finserv Nifty Next 50 Index Fund": "Passive index tracking of the next 50 emerging large caps.",
    "Bajaj Finserv Nifty 50 ETF": "Exchange-traded tracking of the Nifty 50 index.",
    "Bajaj Finserv Nifty Bank ETF": "Exchange-traded exposure to India's banking index.",
    "Bajaj Finserv Nifty 1D Rate Liquid ETF": "Ultra-liquid ETF tracking overnight lending rates.",
}


def render_recommendations():
    """Render the personalized recommendation cards if the customer is eligible."""
    result = st.session_state.get("cross_sell_result")
    if not result or not result.get("eligible"):
        return

    risk_tier = result.get("risk_tier", "Moderate")
    risk_class = risk_tier.lower()
    suggested_funds = result.get("suggested_funds", [])[:3]
    sip_amount = result.get("estimated_monthly_sip", 0)
    reasons = result.get("reasons", [])

    # Risk tier icon
    tier_icons = {"Conservative": "🛡️", "Moderate": "⚖️", "Aggressive": "🚀"}
    tier_icon = tier_icons.get(risk_tier, "📊")

    # Build the entire HTML string
    html = f'''
    <div class="reco-section">
        <div class="reco-header">
            <h3>🎯 Personalized Recommendations</h3>
            <span class="reco-risk-badge {risk_class}">{tier_icon} {risk_tier} Investor</span>
        </div>
        <div class="reco-grid">
    '''

    for fund_name in suggested_funds:
        category = FUND_CATEGORY_MAP.get(fund_name, "General")
        cat_class = category.lower().replace(" ", "-")
        tagline = FUND_TAGLINES.get(fund_name, "A Bajaj Finserv AMC mutual fund product.")

        html += f'<div class="reco-card"><span class="reco-card-category {cat_class}">{category}</span><div class="reco-card-name">{fund_name}</div><div class="reco-card-desc">{tagline}</div><div class="reco-card-sip"><div><div class="reco-card-sip-label">Suggested SIP</div><div class="reco-card-sip-value">₹{sip_amount:,}/mo</div></div></div></div>'

    html += '</div>'

    if reasons:
        reasons_html = ''.join(f'<div class="reco-why-item">✓ {r}</div>' for r in reasons)
        html += f'<div class="reco-why"><div class="reco-why-title">Why we recommend these for you</div>{reasons_html}</div>'

    html += '</div>'
    
    # Render the combined HTML
    st.markdown(html, unsafe_allow_html=True)

    # Interactive buttons — ask about each fund
    cols = st.columns(len(suggested_funds))
    for i, fund_name in enumerate(suggested_funds):
        with cols[i]:
            short_name = fund_name.replace("Bajaj Finserv ", "")
            if st.button(f"💬 Ask about {short_name}", key=f"reco_btn_{i}", use_container_width=True):
                # Pre-fill a question about this fund into the chat
                question = f"Tell me more about the {fund_name} — its objective, returns, and risk level."
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()

    st.markdown("---")


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
                       "outstanding balance, and more.")
        with col3:
            st.markdown("#### 🎯 Smart Recommendations")
            st.markdown("Get personalized mutual fund recommendations "
                       "based on your financial profile.")
        return

    # ── Layout ───────────────────────────────────────────────────────
    chat_col, reco_col = st.columns([7, 3])

    with reco_col:
        # ── Personalized Recommendations (if eligible) ───────────────────
        render_recommendations()

    with chat_col:
        # ── Chat History ─────────────────────────────────────────────────
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                # Show sources for assistant messages that used RAG
                if msg["role"] == "assistant" and msg.get("sources"):
                    render_sources(msg["sources"])

    # ── Chat Input ───────────────────────────────────────────────────
    # Render chat input at the root level so it's pinned to the bottom of the screen
    if prompt := st.chat_input("Ask about mutual funds, your loan, or financial products..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Render the new messages INSIDE the left column to maintain symmetry
        with chat_col:
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