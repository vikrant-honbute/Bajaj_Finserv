"""
agents package — AI Agent layer for the Bajaj Finserv Customer Assistant.

Key exports:
    - ChatAgent: Main orchestrator agent
    - create_chat_agent: Factory to create a ChatAgent
    - retrieve_factsheet_info: RAG tool for factsheet queries
    - parse_rag_response: Parse RAG tool JSON output
"""

from agents.chat_agent import ChatAgent, create_chat_agent
from agents.rag_agent import (
    get_vector_store,
    parse_rag_response,
    retrieve_factsheet_info,
)
from agents.sql_agent import query_customer_db

__all__ = [
    "ChatAgent",
    "create_chat_agent",
    "get_vector_store",
    "parse_rag_response",
    "retrieve_factsheet_info",
    "query_customer_db",
]
