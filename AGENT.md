# AGENTS.md

# AI Loan & Mutual Fund Customer Assistant

## Hackathon Prototype Engineering Instructions

---

## 1. PROJECT OBJECTIVE

Build a Streamlit-based AI customer assistant for existing loan customers.

The assistant must allow an authenticated/demo customer to:

1. Log in.
2. Ask natural-language questions about their own loan account.
3. Receive accurate answers using the provided customer/loan dataset.
4. Determine whether the customer qualifies for a prototype mutual-fund cross-sell.
5. Naturally introduce relevant Bajaj AMC mutual-fund products when the customer qualifies.
6. Answer general mutual-fund questions using the provided fund factsheet PDF as the source of truth.
7. Provide source/page references for factsheet-grounded answers.
8. Optionally support voice input/output if time permits.

The project is a hackathon prototype, not a production banking application.

Prioritize correctness, simplicity, explainability, and demo reliability over unnecessary infrastructure.

---

# 2. TECHNOLOGY STACK

## Required

- Python 3.11+
- Streamlit
- Groq API
- LangChain where useful
- LangSmith for tracing/observability
- langgraph for multiple agent
- SQLite
- Pandas
- Pydantic
- Environment variables via `.env`

## Document/RAG layer

The PDF processing implementation must remain modular.

Possible components may include:

- PyMuPDF
- Docling
- Unstructured
- Vision-capable LLM processing
- Other suitable document parsing libraries

Do not tightly couple the application to a single PDF parser.

## Vector storage

Use a lightweight local vector store appropriate for a hackathon.

Preferred options:

- FAISS
- Chroma

Do NOT introduce a hosted vector database unless there is a clear requirement.

## Optional

- Whisper/faster-whisper/Groq Whisper for STT
- TTS provider if voice is implemented
- Plotly only if visualizations are required by later tasks

---

# 3. CORE ARCHITECTURE

The application uses ONE conversational AI agent/orchestrator.

Do NOT create multiple independent conversational LLM agents unless explicitly required.

The main agent has access to specialized tools/services.

Architecture:

                    USER
                     |
                     v
              STREAMLIT UI
                     |
                     v
              CHAT AGENT
              / ORCHESTRATOR
                     |
        +------------+------------+
        |            |            |
        v            v            v

LOAN DATA CROSS-SELL FUND RAG
TOOL ENGINE TOOL
| | |
v v v
SQLite Business Factsheet
Rules PDF
| | |
+------------+--------------+
|
v
FINAL RESPONSE
|
v
USER

The Chat Agent is responsible for:

- Understanding user intent.
- Selecting the appropriate tool.
- Supplying only necessary context.
- Producing the final natural-language response.

The Chat Agent must NOT invent customer financial information.

---

# 4. SOURCE-OF-TRUTH POLICY

There are two primary sources of truth.

## Customer/Loan Information

SOURCE OF TRUTH:

Provided Excel dataset converted into SQLite.

Examples:

- EMI
- Loan amount
- Principal
- Interest
- Tenure
- Loan type
- Loan status
- DPD
- Application score
- Income
- Disbursement date

Do not answer customer-specific financial questions from the LLM's general knowledge.

Always retrieve customer data from SQLite.

## Mutual Fund Information

SOURCE OF TRUTH:

Provided fund factsheet PDF.

Examples:

- Fund name
- Fund objective
- Risk
- Investment strategy
- Performance information
- AUM
- Expense ratio
- Fund characteristics
- Other information explicitly present in the factsheet

Do not invent or supplement fund-specific facts from general LLM knowledge.

When information is unavailable in the factsheet, explicitly say that the information is not available in the provided factsheet.

---

# 5. CUSTOMER AUTHENTICATION / IDENTITY

Customer identity must be established BEFORE the Chat Agent handles account-specific requests.

For the hackathon, authentication can be a simplified demo login.

The authenticated session should contain:

```python
{
    "customer_id": "...",
    "lan": "...",
    "name": "..."
}
```

6. Cross-Sell Engine (Deterministic, not LLM-decided)
- Evaluates: `App_score`, `DPD_30`, `DPD_90`, `Imputed_Income`, `Loan_status`, `Loan_amount`, `Age`, `Employment_type`, loan history.
- Applies **rule-based thresholds** (for particular customer), e.g.:
  ```
  IF Loan_status = Active
  AND DPD_30 = 0
  AND DPD_90 = 0
  AND App_score >= threshold
  AND Imputed_Income >= threshold
  THEN CrossSellEligible = TRUE
  ```
- Output is a simple `Eligible / Not Eligible` (+ optionally a reason/signal summary) passed to the Chat Agent to phrase naturally — e.g., surfaced conversationally after a loan question, not as a hard sales interruption.
- Validated against the full 10K dataset to produce a defensible qualification rate for the demo (e.g., "X of 10,000 customers qualify").
- Framed explicitly as a **prototype business rule**, not a regulated suitability model.