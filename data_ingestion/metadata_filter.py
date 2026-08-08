"""
Metadata-aware retrieval utility for the FAISS vector store.

Provides filtered similarity search using LangChain's FAISS wrapper,
supporting filtering by fund_name, section, fund_category, and more.
"""

from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from data_ingestion.config import FAISS_STORE_DIR, EMBEDDING_MODEL_NAME


def load_vector_store() -> FAISS:
    """Load the persisted FAISS vector store from disk."""
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.load_local(
        str(FAISS_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def _build_filter(
    fund_name: str | None = None,
    section: str | None = None,
    fund_category: str | None = None,
) -> dict | None:
    """
    Build a metadata filter dict for LangChain FAISS.

    LangChain's FAISS wrapper supports filtering via a dict where each
    key-value pair is matched exactly against document metadata.
    """
    filter_dict: dict[str, str] = {}
    if fund_name is not None:
        filter_dict["fund_name"] = fund_name
    if section is not None:
        filter_dict["section"] = section
    if fund_category is not None:
        filter_dict["fund_category"] = fund_category

    return filter_dict if filter_dict else None


def retrieve_with_metadata(
    query: str,
    vector_store: FAISS,
    fund_name: str | None = None,
    section: str | None = None,
    fund_category: str | None = None,
    top_k: int = 5,
) -> list[Document]:
    """
    Retrieve documents from FAISS with optional metadata filtering.

    Parameters
    ----------
    query : str
        The natural-language search query.
    vector_store : FAISS
        The loaded LangChain FAISS vector store.
    fund_name : str, optional
        Filter to a specific fund (exact match).
    section : str, optional
        Filter to a section type (e.g. 'fund_detail', 'performance', 'sip').
    fund_category : str, optional
        Filter to a category (e.g. 'Equity', 'Hybrid', 'Fixed Income').
    top_k : int
        Number of results to return (default 5).

    Returns
    -------
    list[Document]
        Matching documents with metadata containing page_number, fund_name, 
        section, fund_category, and source.
    """
    metadata_filter = _build_filter(fund_name, section, fund_category)

    if metadata_filter:
        # Over-fetch then filter — LangChain FAISS filter is applied post-search
        results = vector_store.similarity_search(
            query,
            k=top_k,
            filter=metadata_filter,
            fetch_k=top_k * 4,
        )
    else:
        results = vector_store.similarity_search(query, k=top_k)

    return results


def retrieve_with_scores(
    query: str,
    vector_store: FAISS,
    fund_name: str | None = None,
    section: str | None = None,
    fund_category: str | None = None,
    top_k: int = 5,
) -> list[tuple[Document, float]]:
    """
    Like retrieve_with_metadata but also returns similarity scores.

    Returns list of (Document, score) tuples sorted by relevance.
    Lower score = more similar (L2 distance).
    """
    metadata_filter = _build_filter(fund_name, section, fund_category)

    if metadata_filter:
        results = vector_store.similarity_search_with_score(
            query,
            k=top_k,
            filter=metadata_filter,
            fetch_k=top_k * 4,
        )
    else:
        results = vector_store.similarity_search_with_score(query, k=top_k)

    return results


def format_retrieval_context(documents: list[Document]) -> str:
    """
    Format retrieved documents into a context string for the LLM,
    including source page references.
    """
    if not documents:
        return "No relevant information found in the factsheet."

    context_parts: list[str] = []
    for i, doc in enumerate(documents, 1):
        meta = doc.metadata
        page = meta.get("page_number", "?")
        fund = meta.get("fund_name", "General")
        section = meta.get("section", "unknown")
        element_type = meta.get("element_type", "text")

        header = f"[Source {i} | Page {page} | {fund} | {section}]"
        if element_type == "Table":
            header += " [TABLE DATA]"

        context_parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(context_parts)
