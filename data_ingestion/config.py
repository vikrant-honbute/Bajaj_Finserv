"""
Configuration for factsheet data ingestion pipeline.

Defines constants, fund-to-page mappings, section classification rules,
and embedding model settings.
"""

from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "Factsheet_July-2026.pdf"
FAISS_STORE_DIR = PROJECT_ROOT / "data_ingestion" / "faiss_store"

# ── Embedding Model ─────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 1500          # characters per chunk
CHUNK_OVERLAP = 200        # overlap between chunks
LARGE_PAGE_THRESHOLD = 4000  # pages with more chars get split

# ── Data Date ────────────────────────────────────────────────────────────────
DATA_DATE = "June 2026"    # From PDF header: "Data as on 30th June 2026"

# ── Fund Name → TOC Page Number Mapping ──────────────────────────────────────
# Parsed from page 2 (Table of Contents) of the factsheet.
# Key = canonical fund name, Value = starting page number (1-indexed).
FUND_PAGE_MAP: dict[str, int] = {
    "Bajaj Finserv Large Cap Fund": 16,
    "Bajaj Finserv Flexi Cap Fund": 18,
    "Bajaj Finserv Large and Mid Cap Fund": 20,
    "Bajaj Finserv Multi Cap Fund": 22,
    "Bajaj Finserv Small Cap Fund": 23,
    "Bajaj Finserv ELSS Tax Saver Fund": 24,
    "Bajaj Finserv Healthcare Fund": 25,
    "Bajaj Finserv Consumption Fund": 26,
    "Bajaj Finserv Banking and Financial Services Fund": 28,
    "Bajaj Finserv Equity Savings Fund": 29,
    "Bajaj Finserv Balanced Advantage Fund": 30,
    "Bajaj Finserv Multi Asset Allocation Fund": 32,
    "Bajaj Finserv Arbitrage Fund": 34,
    "Bajaj Finserv Liquid Fund": 35,
    "Bajaj Finserv Money Market Fund": 36,
    "Bajaj Finserv Gilt Fund": 37,
    "Bajaj Finserv Overnight Fund": 38,
    "Bajaj Finserv Banking and PSU Fund": 39,
    "Bajaj Finserv Low Duration Fund": 40,
    "Bajaj Finserv Nifty 50 ETF": 41,
    "Bajaj Finserv Nifty Bank ETF": 42,
    "Bajaj Finserv Nifty 1D Rate Liquid ETF": 43,
    "Bajaj Finserv Nifty 50 Index Fund": 44,
    "Bajaj Finserv Nifty Next 50 Index Fund": 45,
}

# Strategy/mapping pages that are associated with the preceding fund
STRATEGY_PAGES: dict[int, str] = {
    17: "Bajaj Finserv Large Cap Fund",        # Large Cap Mapping
    19: "Bajaj Finserv Flexi Cap Fund",        # Megatrends Mapping
    21: "Bajaj Finserv Large and Mid Cap Fund", # Moat Mapping
    27: "Bajaj Finserv Consumption Fund",      # Consumption Mapping
    31: "Bajaj Finserv Balanced Advantage Fund", # Behavioural Indicators Mapping
    33: "Bajaj Finserv Multi Asset Allocation Fund", # GDP Mapping
}

# ── Fund Categories ──────────────────────────────────────────────────────────
FUND_CATEGORIES: dict[str, str] = {
    "Bajaj Finserv Large Cap Fund": "Equity",
    "Bajaj Finserv Flexi Cap Fund": "Equity",
    "Bajaj Finserv Large and Mid Cap Fund": "Equity",
    "Bajaj Finserv Multi Cap Fund": "Equity",
    "Bajaj Finserv Small Cap Fund": "Equity",
    "Bajaj Finserv ELSS Tax Saver Fund": "Equity",
    "Bajaj Finserv Healthcare Fund": "Equity",
    "Bajaj Finserv Consumption Fund": "Equity",
    "Bajaj Finserv Banking and Financial Services Fund": "Equity",
    "Bajaj Finserv Equity Savings Fund": "Hybrid",
    "Bajaj Finserv Balanced Advantage Fund": "Hybrid",
    "Bajaj Finserv Multi Asset Allocation Fund": "Hybrid",
    "Bajaj Finserv Arbitrage Fund": "Hybrid",
    "Bajaj Finserv Liquid Fund": "Fixed Income",
    "Bajaj Finserv Money Market Fund": "Fixed Income",
    "Bajaj Finserv Gilt Fund": "Fixed Income",
    "Bajaj Finserv Overnight Fund": "Fixed Income",
    "Bajaj Finserv Banking and PSU Fund": "Fixed Income",
    "Bajaj Finserv Low Duration Fund": "Fixed Income",
    "Bajaj Finserv Nifty 50 ETF": "ETF",
    "Bajaj Finserv Nifty Bank ETF": "ETF",
    "Bajaj Finserv Nifty 1D Rate Liquid ETF": "ETF",
    "Bajaj Finserv Nifty 50 Index Fund": "Index",
    "Bajaj Finserv Nifty Next 50 Index Fund": "Index",
}

# ── Section Classification Rules ─────────────────────────────────────────────
# Maps page ranges (1-indexed, inclusive) to section names.
SECTION_RULES: list[tuple[int, int, str]] = [
    (1, 1, "cover"),
    (2, 2, "toc"),
    (3, 3, "market_commentary"),      # MD Letter
    (4, 4, "market_commentary"),      # CIO Letter
    (5, 10, "macro"),                 # MacroScope
    (11, 11, "educational"),          # How to Read a Factsheet
    (12, 15, "fund_snapshot"),        # Fund Snapshot tables
    (16, 45, "fund_detail"),          # Individual fund pages (overridden by STRATEGY_PAGES)
    (46, 55, "performance"),          # Performance data
    (56, 58, "sip"),                  # SIP data
    (59, 60, "risk_class"),           # Potential Risk Class
    (61, 64, "risk_o_meter"),         # Risk-o-meter & Product Label
    (65, 65, "promo"),                # KYC promo
]

# Pages to skip entirely (no useful text for RAG)
SKIP_PAGES: set[int] = {1, 2, 65}


def get_section(page_num: int) -> str:
    """Return the section name for a given 1-indexed page number."""
    for start, end, section in SECTION_RULES:
        if start <= page_num <= end:
            return section
    return "unknown"


def get_fund_name(page_num: int) -> str | None:
    """Return the fund name associated with a 1-indexed page number, or None."""
    # Check strategy pages first
    if page_num in STRATEGY_PAGES:
        return STRATEGY_PAGES[page_num]

    # Check direct fund pages
    for fund_name, fund_page in FUND_PAGE_MAP.items():
        if page_num == fund_page:
            return fund_name

    # For performance/sip/risk pages, fund name is embedded in text (handled at extraction time)
    return None


def get_fund_category(fund_name: str | None) -> str | None:
    """Return the category for a fund, or None."""
    if fund_name is None:
        return None
    return FUND_CATEGORIES.get(fund_name)


def build_page_to_fund_map() -> dict[int, str]:
    """
    Build a complete page → fund_name map using TOC ordering.
    
    Each fund 'owns' pages from its TOC entry up to (but not including) the 
    next fund's TOC entry, within the fund_detail section (pages 16–45).
    """
    sorted_funds = sorted(FUND_PAGE_MAP.items(), key=lambda x: x[1])
    page_map: dict[int, str] = {}

    for idx, (fund_name, start_page) in enumerate(sorted_funds):
        if idx + 1 < len(sorted_funds):
            end_page = sorted_funds[idx + 1][1] - 1
        else:
            end_page = 45  # last fund detail page

        for p in range(start_page, end_page + 1):
            # Strategy pages take precedence (already handled in STRATEGY_PAGES)
            if p in STRATEGY_PAGES:
                page_map[p] = STRATEGY_PAGES[p]
            else:
                page_map[p] = fund_name

    return page_map


# Pre-compute at import time
PAGE_TO_FUND: dict[int, str] = build_page_to_fund_map()
