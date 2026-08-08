"""
Main Chat Agent — the single orchestrator for the Bajaj Finserv AI Assistant.

Uses LangChain with Groq (Llama 3.3 70B) as the LLM backbone.
Has access to tools (RAG Agent for factsheet queries) and maintains
conversation memory per session.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from agents.prompts import SYSTEM_PROMPT
from agents.rag_agent import retrieve_factsheet_info, parse_rag_response
from agents.sql_agent import query_customer_db

load_dotenv()
logger = logging.getLogger(__name__)

# ── LLM Configuration ────────────────────────────────────────────────────────

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.3
MAX_TOKENS = 4096


def _create_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> ChatGroq:
    """Create a Groq LLM instance."""
    return ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        api_key=os.getenv("GROQ_API_KEY"),
    )


# ── Chat Agent ────────────────────────────────────────────────────────────────

class ChatAgent:
    """
    Main Chat Agent / Orchestrator.

    Uses LangChain tool-calling with Groq to route queries to the appropriate
    tool (RAG Agent for factsheet questions) and generate grounded responses.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.llm = _create_llm(model=model)
        self.tools = [retrieve_factsheet_info, query_customer_db]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self._tool_map = {t.name: t for t in self.tools}

        # Build the prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        logger.info(f"ChatAgent initialized with model={model}, tools={list(self._tool_map.keys())}")

    def invoke(
        self,
        user_input: str,
        chat_history: list[dict] | None = None,
        customer_identifier: str | None = None,
    ) -> dict[str, Any]:
        """
        Process a user message and return the agent's response.

        Parameters
        ----------
        user_input : str
            The user's message.
        chat_history : list[dict], optional
            Previous conversation messages as [{"role": "user"/"assistant", "content": "..."}].
        customer_identifier : str, optional
            The authenticated user's name or phone number.


        Returns
        -------
        dict with keys:
            - "response": str — the final response text
            - "sources": list[dict] — source references from RAG (if used)
            - "used_rag": bool — whether the RAG tool was invoked
        """
        # Convert chat history to LangChain message format
        messages = self._build_messages(user_input, chat_history or [], customer_identifier)

        # Step 1: Initial LLM call (may request tool calls)
        response = self.llm_with_tools.invoke(messages)

        sources: list[dict] = []
        used_rag = False

        # Step 2: Handle tool calls (agentic loop)
        if response.tool_calls:
            # Add the AI's response (with tool call request) to messages
            messages.append(response)

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                logger.info(f"Tool call: {tool_name}({json.dumps(tool_args, default=str)[:200]})")

                if tool_name in self._tool_map:
                    # Execute the tool
                    tool_result = self._tool_map[tool_name].invoke(tool_args)

                    # Parse RAG sources
                    if tool_name == "retrieve_factsheet_info":
                        used_rag = True
                        parsed = parse_rag_response(tool_result)
                        sources = parsed.get("sources", [])

                    # Add tool result to messages
                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"],
                    ))

            # Step 3: Final LLM call with tool results to generate response
            final_response = self.llm_with_tools.invoke(messages)
            response_text = final_response.content
        else:
            # No tool calls — direct response
            response_text = response.content

        return {
            "response": response_text,
            "sources": sources,
            "used_rag": used_rag,
        }

    def _build_messages(
        self,
        user_input: str,
        chat_history: list[dict],
        customer_identifier: str | None,
    ) -> list:
        """Build the message list for the LLM call."""
        # Inject customer identifier if available
        sys_prompt = SYSTEM_PROMPT
        if customer_identifier:
            sys_prompt += f"\n\nCURRENT AUTHENTICATED CUSTOMER IDENTIFIER: {customer_identifier}"
            
        messages = [SystemMessage(content=sys_prompt)]

        # Add conversation history (last 10 turns = 20 messages)
        history_window = chat_history[-20:]
        for msg in history_window:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        # Add current user input
        messages.append(HumanMessage(content=user_input))

        return messages


# ── Factory ──────────────────────────────────────────────────────────────────

def create_chat_agent(model: str = DEFAULT_MODEL) -> ChatAgent:
    """Create and return a ChatAgent instance."""
    return ChatAgent(model=model)
