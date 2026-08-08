"""
Factsheet PDF → FAISS Vector Database Ingestion Pipeline.

Uses PyMuPDF for PDF text extraction, then enriches each page with
structured metadata and stores in a FAISS vector index via LangChain.

Usage:
    python -m data_ingestion.ingest_factsheet
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from data_ingestion.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DATE,
    EMBEDDING_MODEL_NAME,
    FAISS_STORE_DIR,
    FUND_CATEGORIES,
    FUND_PAGE_MAP,
    LARGE_PAGE_THRESHOLD,
    PAGE_TO_FUND,
    PDF_PATH,
    SKIP_PAGES,
    STRATEGY_PAGES,
    get_fund_category,
    get_section,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Fund name detection helpers ──────────────────────────────────────────────

# All known fund names sorted longest-first for greedy matching
_FUND_NAMES_SORTED = sorted(FUND_PAGE_MAP.keys(), key=len, reverse=True)


def _detect_fund_in_text(text: str) -> str | None:
    """Try to detect a known fund name within a block of text."""
    text_lower = text.lower()
    for fund_name in _FUND_NAMES_SORTED:
        if fund_name.lower() in text_lower:
            return fund_name
    return None


def _detect_fund_for_page(page_num: int, text: str) -> str | None:
    """
    Determine the fund associated with a page.
    
    Priority:
    1. Direct page-to-fund map (from TOC)
    2. Strategy page map
    3. Text-based detection (for performance/sip/risk pages)
    """
    # Pages 16–45: use pre-computed map
    if page_num in PAGE_TO_FUND:
        return PAGE_TO_FUND[page_num]

    # Strategy pages
    if page_num in STRATEGY_PAGES:
        return STRATEGY_PAGES[page_num]

    # Performance, SIP, and risk pages: detect fund name from text content
    section = get_section(page_num)
    if section in ("performance", "sip", "risk_class", "risk_o_meter"):
        return _detect_fund_in_text(text)

    return None


# ── Table Detection Heuristics ───────────────────────────────────────────────

def _detect_tables_in_text(text: str) -> tuple[list[str], str]:
    """
    Simple heuristic to detect table-like content in extracted text.
    
    Returns (table_texts, remaining_text).
    Tables are identified by lines with multiple tab/pipe separators or
    consistent column-like formatting.
    """
    lines = text.split("\n")
    table_lines: list[str] = []
    text_lines: list[str] = []
    current_table: list[str] = []
    tables: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Heuristic: a table row has multiple columns separated by
        # 2+ spaces, tabs, or pipes
        has_columns = (
            len(re.split(r"\s{2,}|\t|\|", stripped)) >= 3
            and len(stripped) > 10
        )

        if has_columns:
            current_table.append(stripped)
        else:
            if current_table and len(current_table) >= 2:
                # We had a table, save it
                tables.append("\n".join(current_table))
                current_table = []
            elif current_table:
                # Too short to be a table, treat as text
                text_lines.extend(current_table)
                current_table = []
            text_lines.append(line)

    # Final flush
    if current_table and len(current_table) >= 2:
        tables.append("\n".join(current_table))
    elif current_table:
        text_lines.extend(current_table)

    return tables, "\n".join(text_lines)


# ── PyMuPDF PDF Extraction ───────────────────────────────────────────────────

def extract_pages_from_pdf(pdf_path: Path) -> dict[int, str]:
    """
    Extract text from each page of the PDF using PyMuPDF.
    
    Returns a dict mapping 1-indexed page numbers to their text content.
    """
    logger.info(f"Extracting text from PDF with PyMuPDF...")
    start = time.time()

    doc = pymupdf.open(str(pdf_path))
    pages: dict[int, str] = {}

    for page_num_0 in range(len(doc)):
        page_num = page_num_0 + 1  # Convert to 1-indexed
        page = doc[page_num_0]
        text = page.get_text("text")
        if text.strip():
            pages[page_num] = text

    doc.close()
    elapsed = time.time() - start
    logger.info(f"Extracted text from {len(pages)} pages in {elapsed:.1f}s")
    return pages


# ── Build LangChain Documents ────────────────────────────────────────────────

def build_documents(pages: dict[int, str]) -> list[Document]:
    """
    Convert extracted page texts into LangChain Documents with rich metadata.

    Applies fund detection, section classification, and handles tables
    and large pages with chunking.
    """
    documents: list[Document] = []
    skipped_pages = 0
    table_count = 0
    text_count = 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ", ", " "],
        length_function=len,
    )

    logger.info(f"Processing {len(pages)} pages...")

    for page_num in sorted(pages.keys()):
        # Skip non-useful pages
        if page_num in SKIP_PAGES:
            skipped_pages += 1
            continue

        full_text = pages[page_num]
        section = get_section(page_num)

        # Detect fund for this page
        fund_name = _detect_fund_for_page(page_num, full_text)
        fund_category = get_fund_category(fund_name)

        # Detect tables in the text
        tables, remaining_text = _detect_tables_in_text(full_text)

        base_metadata = {
            "source": PDF_PATH.name,
            "page_number": page_num,
            "section": section,
            "fund_name": fund_name or "General",
            "fund_category": fund_category or "General",
            "data_date": DATA_DATE,
        }

        # ── Handle detected tables as individual documents ───────
        for table_idx, table_text in enumerate(tables):
            if not table_text.strip():
                continue
            doc = Document(
                page_content=table_text,
                metadata={
                    **base_metadata,
                    "element_type": "Table",
                    "chunk_index": f"table_{table_idx}",
                },
            )
            documents.append(doc)
            table_count += 1

        # ── Handle remaining text ────────────────────────────────
        text_content = remaining_text.strip()
        if not text_content:
            continue

        text_metadata = {
            **base_metadata,
            "element_type": "NarrativeText",
        }

        # Split large pages into smaller chunks
        if len(text_content) > LARGE_PAGE_THRESHOLD:
            chunks = splitter.split_text(text_content)
            for chunk_idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        **text_metadata,
                        "chunk_index": f"text_{chunk_idx}",
                    },
                )
                documents.append(doc)
                text_count += 1
        else:
            doc = Document(
                page_content=text_content,
                metadata={
                    **text_metadata,
                    "chunk_index": "text_0",
                },
            )
            documents.append(doc)
            text_count += 1

    logger.info(
        f"Built {len(documents)} documents "
        f"({text_count} text chunks + {table_count} tables, "
        f"{skipped_pages} pages skipped)"
    )
    return documents


# ── Multi-Fund Page Handling ─────────────────────────────────────────────────

def _split_multi_fund_pages(documents: list[Document]) -> list[Document]:
    """
    For pages in performance/SIP/risk sections that mention multiple funds,
    try to split and re-tag with the correct fund name.
    
    This handles pages 46–64 where multiple fund performance data appears
    on a single page.
    """
    enhanced: list[Document] = []

    for doc in documents:
        section = doc.metadata.get("section", "")
        if section not in ("performance", "sip", "risk_class", "risk_o_meter"):
            enhanced.append(doc)
            continue

        # Check if multiple funds are mentioned in the text
        text = doc.page_content
        found_funds = []
        for fund_name in _FUND_NAMES_SORTED:
            if fund_name.lower() in text.lower():
                found_funds.append(fund_name)

        if len(found_funds) <= 1:
            # Single fund or no fund — keep as-is
            if found_funds:
                doc.metadata["fund_name"] = found_funds[0]
                doc.metadata["fund_category"] = get_fund_category(found_funds[0]) or "General"
            enhanced.append(doc)
            continue

        # Multiple funds: try to split text by fund sections
        # Create separate docs for each fund found
        for fund in found_funds:
            # Extract the portion of text most relevant to this fund
            fund_section = _extract_fund_section(text, fund)
            if fund_section and fund_section.strip():
                new_doc = Document(
                    page_content=fund_section,
                    metadata={
                        **doc.metadata,
                        "fund_name": fund,
                        "fund_category": get_fund_category(fund) or "General",
                        "chunk_index": f"fund_split_{found_funds.index(fund)}",
                    },
                )
                enhanced.append(new_doc)

        # If splitting didn't produce anything, keep the original
        if not any(
            d.metadata.get("page_number") == doc.metadata.get("page_number")
            for d in enhanced[-len(found_funds):]
            if d in enhanced
        ):
            enhanced.append(doc)

    return enhanced


def _extract_fund_section(text: str, fund_name: str) -> str | None:
    """
    Extract the section of text relevant to a specific fund.
    
    Tries to find the block of text between this fund's name and the next
    fund's name in the text.
    """
    text_lower = text.lower()
    fund_lower = fund_name.lower()

    start = text_lower.find(fund_lower)
    if start == -1:
        return None

    # Find the next fund name after this one
    next_fund_start = len(text)
    for other_fund in _FUND_NAMES_SORTED:
        other_lower = other_fund.lower()
        if other_lower == fund_lower:
            continue
        pos = text_lower.find(other_lower, start + len(fund_lower))
        if pos != -1 and pos < next_fund_start:
            next_fund_start = pos

    # Include some context before the fund name (up to 100 chars)
    context_start = max(0, start - 100)
    # Find a clean break point
    for sep in ["\n", ". ", " "]:
        idx = text.rfind(sep, context_start, start)
        if idx != -1:
            context_start = idx + len(sep)
            break

    return text[context_start:next_fund_start].strip()


# ── FAISS Index Creation ─────────────────────────────────────────────────────

def create_faiss_index(documents: list[Document]) -> None:
    """Create and persist a FAISS vector store from the documents."""
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )

    logger.info(f"Creating FAISS index from {len(documents)} documents...")
    start = time.time()
    vector_store = FAISS.from_documents(documents, embeddings)
    elapsed = time.time() - start
    logger.info(f"FAISS index created in {elapsed:.1f}s")

    # Persist to disk
    FAISS_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(FAISS_STORE_DIR))
    logger.info(f"FAISS index saved to {FAISS_STORE_DIR}")

    # Save metadata summary for debugging
    meta_summary = _build_metadata_summary(documents)
    meta_path = FAISS_STORE_DIR / "metadata_summary.json"
    meta_path.write_text(json.dumps(meta_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Metadata summary saved to {meta_path}")


def _build_metadata_summary(documents: list[Document]) -> dict:
    """Build a summary of all metadata for debugging and validation."""
    funds: dict[str, int] = {}
    sections: dict[str, int] = {}
    categories: dict[str, int] = {}
    element_types: dict[str, int] = {}
    pages_covered: set[int] = set()

    for doc in documents:
        meta = doc.metadata
        fund = meta.get("fund_name", "Unknown")
        section = meta.get("section", "unknown")
        category = meta.get("fund_category", "Unknown")
        elem_type = meta.get("element_type", "unknown")

        funds[fund] = funds.get(fund, 0) + 1
        sections[section] = sections.get(section, 0) + 1
        categories[category] = categories.get(category, 0) + 1
        element_types[elem_type] = element_types.get(elem_type, 0) + 1
        pages_covered.add(meta.get("page_number", 0))

    return {
        "total_documents": len(documents),
        "pages_covered": sorted(pages_covered),
        "funds": dict(sorted(funds.items(), key=lambda x: -x[1])),
        "sections": dict(sorted(sections.items(), key=lambda x: -x[1])),
        "categories": dict(sorted(categories.items(), key=lambda x: -x[1])),
        "element_types": element_types,
    }


# ── Main Pipeline ────────────────────────────────────────────────────────────

def run_ingestion() -> None:
    """Run the full ingestion pipeline."""
    logger.info("=" * 60)
    logger.info("Factsheet PDF → FAISS Ingestion Pipeline")
    logger.info("=" * 60)

    # Step 1: Validate PDF exists
    if not PDF_PATH.exists():
        logger.error(f"PDF not found: {PDF_PATH}")
        sys.exit(1)
    logger.info(f"PDF: {PDF_PATH} ({PDF_PATH.stat().st_size / 1024 / 1024:.1f} MB)")

    # Step 2: Extract text from each page using PyMuPDF
    pages = extract_pages_from_pdf(PDF_PATH)

    # Step 3: Build documents with metadata
    documents = build_documents(pages)

    # Step 4: Handle multi-fund pages (performance/SIP/risk)
    documents = _split_multi_fund_pages(documents)
    logger.info(f"After multi-fund split: {len(documents)} documents")

    # Step 5: Create and persist FAISS index
    create_faiss_index(documents)

    logger.info("=" * 60)
    logger.info("Ingestion complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_ingestion()
