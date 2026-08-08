"""
Factsheet PDF → FAISS Vector Database Ingestion Pipeline.

Uses the `unstructured` library for intelligent PDF parsing with native
table detection, then enriches each element with structured metadata
and stores in a FAISS vector index via LangChain.

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


# ── Table Formatting ─────────────────────────────────────────────────────────

def _format_table_element(element) -> str:
    """
    Convert an unstructured Table element into a clean markdown/text
    representation that embeds well.
    
    The `unstructured` library provides table data in element.metadata.text_as_html
    when available. We convert it to a readable markdown table or fall back 
    to the plain text representation.
    """
    html = getattr(element.metadata, "text_as_html", None)

    if html:
        return _html_table_to_markdown(html)

    # Fallback: use the element's text directly
    return str(element)


def _html_table_to_markdown(html: str) -> str:
    """Convert an HTML table string to a readable markdown table."""
    try:
        # Simple regex-based HTML table → markdown conversion
        # Remove tags and format as clean text table
        rows: list[list[str]] = []
        for row_match in re.finditer(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL):
            row_html = row_match.group(1)
            cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row_html, re.DOTALL)
            # Clean HTML tags from cell content
            cleaned = [re.sub(r"<[^>]+>", "", cell).strip() for cell in cells]
            if any(c for c in cleaned):  # skip completely empty rows
                rows.append(cleaned)

        if not rows:
            # Fallback: strip all tags
            return re.sub(r"<[^>]+>", " ", html).strip()

        # Build markdown table
        if len(rows) < 2:
            return " | ".join(rows[0]) if rows else ""

        # Normalize column count
        max_cols = max(len(row) for row in rows)
        for row in rows:
            while len(row) < max_cols:
                row.append("")

        lines: list[str] = []
        # Header
        lines.append("| " + " | ".join(rows[0]) + " |")
        lines.append("| " + " | ".join("---" for _ in rows[0]) + " |")
        # Data rows
        for row in rows[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    except Exception:
        # If parsing fails, just strip tags
        return re.sub(r"<[^>]+>", " ", html).strip()


# ── Unstructured PDF Extraction ──────────────────────────────────────────────

def extract_elements_from_pdf(pdf_path: Path) -> list:
    """
    Extract structured elements from the PDF using `unstructured`.

    Uses the 'fast' strategy with PyMuPDF under the hood for text,
    and detects tables as separate elements.
    """
    from unstructured.partition.pdf import partition_pdf

    logger.info("Partitioning PDF with unstructured (this may take a minute)...")
    start = time.time()

    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="fast",
        include_page_breaks=True,
        infer_table_structure=True,
    )

    elapsed = time.time() - start
    logger.info(f"Extracted {len(elements)} elements in {elapsed:.1f}s")
    return elements


# ── Build LangChain Documents ────────────────────────────────────────────────

def _get_element_page(element) -> int:
    """Get 1-indexed page number from an unstructured element."""
    page = getattr(element.metadata, "page_number", None)
    return page if page else 1


def build_documents(elements: list) -> list[Document]:
    """
    Convert unstructured elements into LangChain Documents with rich metadata.

    Groups elements by page, applies fund detection, section classification,
    and handles tables specially.
    """
    from unstructured.documents.elements import (
        PageBreak,
        Table,
    )

    documents: list[Document] = []
    skipped_pages = 0
    table_count = 0
    text_count = 0

    # Group elements by page
    page_elements: dict[int, list] = {}
    for element in elements:
        if isinstance(element, PageBreak):
            continue
        page_num = _get_element_page(element)
        page_elements.setdefault(page_num, []).append(element)

    logger.info(f"Processing {len(page_elements)} pages...")

    for page_num in sorted(page_elements.keys()):
        # Skip non-useful pages
        if page_num in SKIP_PAGES:
            skipped_pages += 1
            continue

        elems = page_elements[page_num]
        section = get_section(page_num)

        # Collect all text on the page for fund detection
        full_page_text = " ".join(str(e) for e in elems)
        fund_name = _detect_fund_for_page(page_num, full_page_text)
        fund_category = get_fund_category(fund_name)

        # Separate tables from other elements
        tables = [e for e in elems if isinstance(e, Table)]
        non_tables = [e for e in elems if not isinstance(e, Table)]

        # ── Handle tables as individual documents ────────────────────
        for table_idx, table_elem in enumerate(tables):
            table_text = _format_table_element(table_elem)
            if not table_text.strip():
                continue

            # Try to find a title/context for the table from surrounding elements
            table_context = _find_table_context(non_tables, table_elem)
            if table_context:
                table_text = f"{table_context}\n\n{table_text}"

            doc = Document(
                page_content=table_text,
                metadata={
                    "source": PDF_PATH.name,
                    "page_number": page_num,
                    "section": section,
                    "fund_name": fund_name or "General",
                    "fund_category": fund_category or "General",
                    "data_date": DATA_DATE,
                    "element_type": "Table",
                    "chunk_index": f"table_{table_idx}",
                },
            )
            documents.append(doc)
            table_count += 1

        # ── Handle non-table text ────────────────────────────────────
        text_content = "\n".join(
            str(e) for e in non_tables if str(e).strip()
        )
        if not text_content.strip():
            continue

        base_metadata = {
            "source": PDF_PATH.name,
            "page_number": page_num,
            "section": section,
            "fund_name": fund_name or "General",
            "fund_category": fund_category or "General",
            "data_date": DATA_DATE,
            "element_type": "NarrativeText",
        }

        # Split large pages into smaller chunks
        if len(text_content) > LARGE_PAGE_THRESHOLD:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", ", ", " "],
                length_function=len,
            )
            chunks = splitter.split_text(text_content)
            for chunk_idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        **base_metadata,
                        "chunk_index": f"text_{chunk_idx}",
                    },
                )
                documents.append(doc)
                text_count += 1
        else:
            doc = Document(
                page_content=text_content,
                metadata={
                    **base_metadata,
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


def _find_table_context(non_table_elements: list, table_element) -> str | None:
    """
    Find the most relevant title/heading near a table to provide context.
    
    Looks for Title or NarrativeText elements that appear before the table
    on the same page and could serve as a table header/description.
    """
    from unstructured.documents.elements import Title

    # Collect titles from the page
    titles = [str(e) for e in non_table_elements if isinstance(e, Title)]
    if titles:
        return titles[0]  # First title on the page is usually the section heading

    # Fallback: first short text element as context
    for e in non_table_elements:
        text = str(e).strip()
        if text and len(text) < 200:
            return text

    return None


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

    # Step 2: Extract elements using unstructured
    elements = extract_elements_from_pdf(PDF_PATH)

    # Step 3: Build documents with metadata
    documents = build_documents(elements)

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
