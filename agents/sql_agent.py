"""
SQL Agent Tool for the Main Chat Agent.

Provides a secure, targeted tool to query customer data from the SQLite database
using the authenticated customer's identifier.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "customer_data.db"


class CustomerDataQueryInput(BaseModel):
    """Input schema for querying customer data."""
    customer_identifier: str = Field(
        description="The authenticated customer's phone number or name."
    )
    query_purpose: str = Field(
        description="A brief description of what information is needed (e.g., 'EMI amount', 'loan status', 'all details')."
    )


@tool(args_schema=CustomerDataQueryInput)
def query_customer_db(customer_identifier: str, query_purpose: str) -> str:
    """
    Retrieve specific loan and account details for the authenticated customer from the database.
    Use this tool whenever the customer asks about their own loan, EMI, balance, status, etc.
    Do NOT use this tool for general mutual fund questions.
    """
    logger.info(f"Querying DB for customer: {customer_identifier}, purpose: {query_purpose}")
    
    try:
        conn = sqlite3.connect(str(DB_PATH))
        # Ensure we return rows as dicts for easy JSON conversion
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # We query by either Name or MobileNo (case-insensitive for Name)
        cursor.execute(
            "SELECT * FROM customers WHERE MobileNo = ? OR Name = ? COLLATE NOCASE",
            (customer_identifier, customer_identifier)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return json.dumps({
                "status": "error",
                "message": f"Could not find customer record for '{customer_identifier}'."
            })
            
        # Convert row to dict
        customer_data = dict(row)
        
        # We can return the entire row to the LLM, and it will pick out what it needs based on query_purpose
        return json.dumps({
            "status": "success",
            "customer_data": customer_data,
            "note": "Use this data to answer the user's specific question."
        })
        
    except Exception as e:
        logger.error(f"Error querying customer DB: {e}")
        return json.dumps({
            "status": "error",
            "message": f"An error occurred while querying the database: {str(e)}"
        })
