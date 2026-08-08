"""
RAG Agent — Factsheet retrieval tool for the Main Chat Agent.

Wraps the FAISS vector store as a LangChain tool, providing metadata-filtered
similarity search over the Bajaj Finserv Mutual Fund Factsheet.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from data_ingestion.config import EMBEDDING_MODEL_NAME, FAISS_STORE_DIR
from data_ingestion.metadata_filter import (
    format_retrieval_context,
    load_vector_store,
    retrieve_with_scores,
)

logger = logging.getLogger(__name__)


# ── Singleton Vector Store ───────────────────────────────────────────────────

_vector_store = None


def get_vector_store():
    """Load the FAISS vector store (singleton — loaded once per process)."""
    global _vector_store
    if _vector_store is None:
        logger.info("Loading FAISS vector store...")
        _vector_store = load_vector_store()
        logger.info(f"FAISS loaded: {_vector_store.index.ntotal} vectors")
    return _vector_store


# ── Tool Input Schema ────────────────────────────────────────────────────────

class FactsheetRetrievalInput(BaseModel):
    """Input schema for the factsheet retrieval tool."""

    query: str = Field(
        description="The search query to find relevant factsheet information. "
        "Should be a clear, specific question or topic about mutual funds."
    )
    fund_name: str | None = Field(
        default=None,
        description="Exact fund name to filter by (e.g. 'Bajaj Finserv Flexi Cap Fund'). "
        "Use None to search across all funds."
    )
    section: str | None = Field(
        default=None,
        description="Section to filter by. Options: fund_detail, performance, sip, "
        "risk_class, risk_o_meter, macro, market_commentary, fund_snapshot, educational. "
        "Use None for all sections."
    )
    fund_category: str | None = Field(
        default=None,
        description="Fund category to filter by. Options: Equity, Hybrid, "
        "Fixed Income, ETF, Index. Use None for all categories."
    )


# ── RAG Tool ─────────────────────────────────────────────────────────────────

@tool(args_schema=FactsheetRetrievalInput)
def retrieve_factsheet_info(
    query: str,
    fund_name: str | None = None,
    section: str | None = None,
    fund_category: str | None = None,
) -> str:
    """
    Retrieve information from the Bajaj Finserv Mutual Fund Factsheet (June 2026).

    Use this tool to answer ANY question about Bajaj Finserv mutual funds,
    including fund details, NAV, returns, expense ratios, AUM, holdings,
    risk ratings, SIP data, market commentary, and fund performance.

    The tool searches the factsheet PDF and returns relevant passages with
    page number references. You can optionally filter by fund name, section,
    or category for more precise results.
    """
    vs = get_vector_store()

    # Retrieve with scores for relevance ranking
    results = retrieve_with_scores(
        query=query,
        vector_store=vs,
        fund_name=fund_name,
        section=section,
        fund_category=fund_category,
        top_k=5,
    )

    if not results:
        return json.dumps({
            "context": "No relevant information found in the factsheet for this query.",
            "sources": [],
            "num_results": 0,
        })

    # Build context and sources
    documents = [doc for doc, _score in results]
    context = format_retrieval_context(documents)

    sources = []
    for doc, score in results:
        meta = doc.metadata
        sources.append({
            "page_number": meta.get("page_number", "?"),
            "fund_name": meta.get("fund_name", "General"),
            "section": meta.get("section", "unknown"),
            "fund_category": meta.get("fund_category", "General"),
            "element_type": meta.get("element_type", "text"),
            "relevance_score": round(float(score), 4),
            "preview": doc.page_content[:200].replace("\n", " "),
        })

    result = {
        "context": context,
        "sources": sources,
        "num_results": len(results),
    }

    logger.info(
        f"RAG retrieved {len(results)} docs for query='{query[:50]}...' "
        f"fund={fund_name} section={section} category={fund_category}"
    )

    return json.dumps(result, ensure_ascii=False)


# ── Helper to parse RAG response ─────────────────────────────────────────────

def parse_rag_response(raw_response: str) -> dict[str, Any]:
    """Parse the JSON response from the RAG tool into a dict."""
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        return {
            "context": raw_response,
            "sources": [],
            "num_results": 0,
        }
