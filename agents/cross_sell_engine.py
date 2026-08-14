"""
Cross-Sell Engine — Deterministic eligibility evaluator.

Evaluates whether a customer qualifies for mutual fund cross-sell
based on rule-based thresholds applied to their loan/profile data.
Outputs a risk tier, suggested fund categories, and specific fund names.

This is a prototype business rule, NOT a regulated suitability model.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Eligibility thresholds
MIN_APP_SCORE = 6.0
MIN_ANNUAL_INCOME = 500_000  # ₹5,00,000

# Risk-tier age boundaries
YOUNG_AGE_UPPER = 35
MID_AGE_UPPER = 50

# SIP estimation parameters
SIP_INCOME_FRACTION = 0.10      # 10% of monthly income
SIP_EMI_DRAG_FRACTION = 0.05   # 5% of EMI subtracted as load
MIN_SIP_AMOUNT = 500
MAX_SIP_AMOUNT = 50_000

# ── Fund Catalogue (from Factsheet) ─────────────────────────────────────────

FUND_CATALOGUE: dict[str, list[str]] = {
    "Equity": [
        "Bajaj Finserv Flexi Cap Fund",
        "Bajaj Finserv Large Cap Fund",
        "Bajaj Finserv Large and Mid Cap Fund",
        "Bajaj Finserv Multi Cap Fund",
        "Bajaj Finserv Small Cap Fund",
        "Bajaj Finserv ELSS Tax Saver Fund",
        "Bajaj Finserv Healthcare Fund",
        "Bajaj Finserv Consumption Fund",
        "Bajaj Finserv Banking and Financial Services Fund",
    ],
    "Hybrid": [
        "Bajaj Finserv Balanced Advantage Fund",
        "Bajaj Finserv Multi Asset Allocation Fund",
        "Bajaj Finserv Equity Savings Fund",
        "Bajaj Finserv Arbitrage Fund",
    ],
    "Fixed Income": [
        "Bajaj Finserv Liquid Fund",
        "Bajaj Finserv Money Market Fund",
        "Bajaj Finserv Gilt Fund",
        "Bajaj Finserv Overnight Fund",
        "Bajaj Finserv Banking and PSU Fund",
        "Bajaj Finserv Low Duration Fund",
    ],
    "Index": [
        "Bajaj Finserv Nifty 50 Index Fund",
        "Bajaj Finserv Nifty Next 50 Index Fund",
    ],
    "ETF": [
        "Bajaj Finserv Nifty 50 ETF",
        "Bajaj Finserv Nifty Bank ETF",
        "Bajaj Finserv Nifty 1D Rate Liquid ETF",
    ],
}

# Risk-tier → preferred fund categories (ordered by preference)
RISK_TIER_FUND_MAP: dict[str, list[str]] = {
    "Conservative": ["Fixed Income", "Hybrid"],
    "Moderate": ["Hybrid", "Equity", "Index"],
    "Aggressive": ["Equity", "Index", "ETF"],
}

# Risk-tier → top fund picks (handpicked for demo relevance)
RISK_TIER_TOP_FUNDS: dict[str, list[str]] = {
    "Conservative": [
        "Bajaj Finserv Balanced Advantage Fund",
        "Bajaj Finserv Liquid Fund",
        "Bajaj Finserv Banking and PSU Fund",
    ],
    "Moderate": [
        "Bajaj Finserv Flexi Cap Fund",
        "Bajaj Finserv Multi Asset Allocation Fund",
        "Bajaj Finserv Nifty 50 Index Fund",
    ],
    "Aggressive": [
        "Bajaj Finserv Flexi Cap Fund",
        "Bajaj Finserv Small Cap Fund",
        "Bajaj Finserv ELSS Tax Saver Fund",
    ],
}


# ── Result Model ─────────────────────────────────────────────────────────────

class CrossSellResult(BaseModel):
    """Output of the cross-sell eligibility evaluation."""

    eligible: bool = Field(description="Whether the customer is eligible for cross-sell")
    reasons: list[str] = Field(description="Eligibility reasons or rejection factors")
    risk_tier: str = Field(description="Conservative / Moderate / Aggressive")
    suggested_categories: list[str] = Field(description="Fund categories matching the risk tier")
    suggested_funds: list[str] = Field(description="Specific fund names from the factsheet")
    estimated_monthly_sip: int = Field(description="Suggested monthly SIP amount in ₹")
    signal_summary: str = Field(description="One-line summary for the chat agent")


# ── Core Engine ──────────────────────────────────────────────────────────────

def _parse_dpd(value: Any) -> int:
    """
    Parse DPD (Days Past Due) column value.

    Values can be:
    - None → no delinquency → 0
    - Numeric string → int
    - Date-like string (e.g. "2024-03") → treat as 0 (data artifact)
    """
    if value is None:
        return 0
    try:
        return int(float(str(value)))
    except (ValueError, TypeError):
        # Non-numeric (e.g. "2024-03") — treat as no delinquency
        return 0


def _determine_risk_tier(age: int | None, income: int | None, employment_type: str | None) -> str:
    """
    Determine the customer's investment risk tier based on demographics.

    Rules:
    - Young (< 35) + Salaried + High income → Aggressive
    - Mid-age (35-50) OR moderate income → Moderate
    - Older (> 50) OR Self-employed OR lower income → Conservative
    """
    age = age or 30
    income = income or 0
    emp = (employment_type or "").strip().lower()

    score = 0

    # Age factor
    if age < YOUNG_AGE_UPPER:
        score += 2
    elif age < MID_AGE_UPPER:
        score += 1
    else:
        score += 0

    # Income factor
    if income >= 1_500_000:
        score += 2
    elif income >= 800_000:
        score += 1
    else:
        score += 0

    # Employment stability factor
    if emp == "salaried":
        score += 1
    # Self-employed / other → no bonus

    # Map score to tier
    if score >= 4:
        return "Aggressive"
    elif score >= 2:
        return "Moderate"
    else:
        return "Conservative"


def _estimate_sip(monthly_income: int, emi_amount: int) -> int:
    """
    Estimate a reasonable monthly SIP amount.

    Formula: 10% of monthly income minus 5% of EMI load.
    Clamped to [MIN_SIP_AMOUNT, MAX_SIP_AMOUNT].
    """
    raw = int(monthly_income * SIP_INCOME_FRACTION - emi_amount * SIP_EMI_DRAG_FRACTION)
    # Round to nearest 500
    raw = max(raw, MIN_SIP_AMOUNT)
    raw = min(raw, MAX_SIP_AMOUNT)
    return (raw // 500) * 500


def evaluate_cross_sell_eligibility(customer_data: dict) -> CrossSellResult:
    """
    Evaluate whether a customer qualifies for mutual fund cross-sell.

    Parameters
    ----------
    customer_data : dict
        Full customer row from the SQLite database.

    Returns
    -------
    CrossSellResult
        Eligibility result with risk tier, fund suggestions, and SIP estimate.
    """
    reasons: list[str] = []
    eligible = True

    # ── Extract fields ───────────────────────────────────────────────────
    loan_status = (customer_data.get("Loan_status") or "").strip()
    dpd_30 = _parse_dpd(customer_data.get("DPD_30"))
    dpd_90 = _parse_dpd(customer_data.get("DPD_90"))
    app_score = customer_data.get("App_score")
    income = customer_data.get("Imputed_Income") or 0
    age = customer_data.get("Age")
    emi = customer_data.get("EMI_amount") or 0
    employment_type = customer_data.get("Employment_type")
    name = customer_data.get("Name", "Customer")

    # ── Rule 1: Loan must be Active ──────────────────────────────────────
    if loan_status.lower() != "active":
        eligible = False
        reasons.append(f"Loan status is '{loan_status}' (requires Active)")

    # ── Rule 2: No recent delinquency ────────────────────────────────────
    if dpd_30 > 0:
        eligible = False
        reasons.append(f"DPD_30 = {dpd_30} (requires 0)")
    if dpd_90 > 0:
        eligible = False
        reasons.append(f"DPD_90 = {dpd_90} (requires 0)")

    # ── Rule 3: Minimum App Score ────────────────────────────────────────
    if app_score is None or app_score < MIN_APP_SCORE:
        eligible = False
        reasons.append(f"App score = {app_score} (requires ≥ {MIN_APP_SCORE})")

    # ── Rule 4: Minimum Income ───────────────────────────────────────────
    if income < MIN_ANNUAL_INCOME:
        eligible = False
        reasons.append(f"Income = ₹{income:,} (requires ≥ ₹{MIN_ANNUAL_INCOME:,})")

    # ── Build positive reasons if eligible ───────────────────────────────
    if eligible:
        reasons = [
            f"Active loan with clean repayment history",
            f"Strong credit score ({app_score})",
            f"Annual income ₹{income:,}",
            f"No delinquency (DPD_30=0, DPD_90=0)",
        ]

    # ── Risk tier & fund suggestions ─────────────────────────────────────
    risk_tier = _determine_risk_tier(age, income, employment_type)
    suggested_categories = RISK_TIER_FUND_MAP.get(risk_tier, ["Hybrid"])
    suggested_funds = RISK_TIER_TOP_FUNDS.get(risk_tier, [])

    # ── SIP estimation ───────────────────────────────────────────────────
    monthly_income = income // 12
    estimated_sip = _estimate_sip(monthly_income, emi)

    # ── Signal summary ───────────────────────────────────────────────────
    if eligible:
        signal_summary = (
            f"{name} qualifies for mutual fund cross-sell. "
            f"Risk profile: {risk_tier}. "
            f"Suggested SIP: ₹{estimated_sip:,}/month. "
            f"Top picks: {', '.join(suggested_funds[:3])}."
        )
    else:
        signal_summary = f"{name} does not currently qualify for cross-sell."

    result = CrossSellResult(
        eligible=eligible,
        reasons=reasons,
        risk_tier=risk_tier,
        suggested_categories=suggested_categories,
        suggested_funds=suggested_funds,
        estimated_monthly_sip=estimated_sip,
        signal_summary=signal_summary,
    )

    logger.info(
        f"Cross-sell evaluation for {name}: eligible={eligible}, "
        f"risk_tier={risk_tier}, sip=₹{estimated_sip:,}"
    )

    return result
