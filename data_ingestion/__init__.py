"""
data_ingestion package — Factsheet PDF to FAISS vector store pipeline.

Key exports:
    - run_ingestion: Full pipeline (extract → enrich → embed → store)
    - load_vector_store: Load persisted FAISS index
    - retrieve_with_metadata: Metadata-filtered similarity search
    - format_retrieval_context: Format results for LLM context
"""

from data_ingestion.ingest_factsheet import run_ingestion
from data_ingestion.metadata_filter import (
    format_retrieval_context,
    load_vector_store,
    retrieve_with_metadata,
    retrieve_with_scores,
)

__all__ = [
    "run_ingestion",
    "load_vector_store",
    "retrieve_with_metadata",
    "retrieve_with_scores",
    "format_retrieval_context",
]
