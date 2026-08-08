"""
Validate the FAISS vector store by running sample queries.

Usage:
    python -m data_ingestion.validate_ingestion
"""

from __future__ import annotations

import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    from data_ingestion.config import FAISS_STORE_DIR
    from data_ingestion.metadata_filter import (
        format_retrieval_context,
        load_vector_store,
        retrieve_with_metadata,
        retrieve_with_scores,
    )

    print("=" * 60)
    print("FAISS Vector Store Validation")
    print("=" * 60)

    # Load the store
    print("\n1. Loading FAISS vector store...")
    vs = load_vector_store()
    total_docs = vs.index.ntotal
    print(f"   Total vectors in index: {total_docs}")

    # Load metadata summary
    meta_path = FAISS_STORE_DIR / "metadata_summary.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"   Pages covered: {len(meta['pages_covered'])}")
        print(f"   Funds indexed: {len(meta['funds'])}")
        print(f"   Sections: {list(meta['sections'].keys())}")
        print(f"   Element types: {meta['element_types']}")

    # ── Test Queries ────────────────────────────────────────────────

    test_cases = [
        {
            "desc": "Basic fund NAV query",
            "query": "What is the NAV of Bajaj Finserv Flexi Cap Fund?",
            "filters": {},
            "expected_fund": "Bajaj Finserv Flexi Cap Fund",
        },
        {
            "desc": "Fund detail with metadata filter",
            "query": "What is the expense ratio?",
            "filters": {"fund_name": "Bajaj Finserv Large Cap Fund"},
            "expected_fund": "Bajaj Finserv Large Cap Fund",
        },
        {
            "desc": "Category filter for Equity funds",
            "query": "What are the best performing equity funds?",
            "filters": {"fund_category": "Equity"},
            "expected_fund": None,
        },
        {
            "desc": "Performance section query",
            "query": "Returns and performance data for Arbitrage Fund",
            "filters": {"fund_name": "Bajaj Finserv Arbitrage Fund"},
            "expected_fund": "Bajaj Finserv Arbitrage Fund",
        },
        {
            "desc": "Market commentary (no fund filter)",
            "query": "What is the market outlook for Indian equities?",
            "filters": {},
            "expected_fund": None,
        },
        {
            "desc": "Table data query",
            "query": "Top holdings and portfolio composition of Nifty 50 Index Fund",
            "filters": {"fund_name": "Bajaj Finserv Nifty 50 Index Fund"},
            "expected_fund": "Bajaj Finserv Nifty 50 Index Fund",
        },
    ]

    print("\n2. Running test queries...")
    passed = 0
    failed = 0

    for i, tc in enumerate(test_cases, 1):
        print(f"\n{'─' * 50}")
        print(f"   Test {i}: {tc['desc']}")
        print(f"   Query: \"{tc['query']}\"")
        if tc["filters"]:
            print(f"   Filters: {tc['filters']}")

        results = retrieve_with_scores(
            query=tc["query"],
            vector_store=vs,
            **tc["filters"],
            top_k=3,
        )

        if not results:
            print("   ❌ No results returned!")
            failed += 1
            continue

        print(f"   ✅ Got {len(results)} results:")
        for doc, score in results:
            meta = doc.metadata
            print(
                f"      Page {meta.get('page_number', '?')} | "
                f"{meta.get('fund_name', 'N/A')} | "
                f"{meta.get('section', '?')} | "
                f"Type: {meta.get('element_type', '?')} | "
                f"Score: {score:.4f}"
            )
            # Print first 150 chars of content
            preview = doc.page_content[:150].replace("\n", " ")
            print(f"      Preview: {preview}...")

        # Validate expected fund appears in results
        if tc["expected_fund"]:
            fund_found = any(
                doc.metadata.get("fund_name") == tc["expected_fund"]
                for doc, _ in results
            )
            if fund_found:
                print(f"   ✅ Expected fund '{tc['expected_fund']}' found!")
                passed += 1
            else:
                print(f"   ⚠️  Expected fund '{tc['expected_fund']}' NOT in top results")
                failed += 1
        else:
            passed += 1

    # ── Summary ────────────────────────────────────────────────────

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{passed + failed} tests passed")
    print(f"{'=' * 60}")

    # ── Show formatted context example ─────────────────────────────
    print("\n3. Example formatted context for LLM:")
    print("─" * 50)
    results = retrieve_with_metadata(
        query="Tell me about Bajaj Finserv Healthcare Fund",
        vector_store=vs,
        fund_name="Bajaj Finserv Healthcare Fund",
        top_k=3,
    )
    context = format_retrieval_context(results)
    print(context[:1500])
    if len(context) > 1500:
        print("... (truncated)")


if __name__ == "__main__":
    main()
