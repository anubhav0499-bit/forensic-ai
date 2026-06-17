"""
Accrual Analysis — Earnings Quality via Cash vs. Accrual Earnings
==================================================================

References
----------
Sloan, R. G. (1996). "Do Stock Prices Fully Reflect Information in
    Accruals and Cash Flows About Future Earnings?" The Accounting
    Review, 71(3), 289–315.
    Balance sheet accrual = ΔWorking Capital − ΔCash − Depreciation.
    Firms with high accruals (relative to assets) experience lower
    future earnings — the "accrual anomaly."

Richardson, S. A., Sloan, R. G., Soliman, M. T., & Tuna, I. (2005).
    "Accrual Reliability, Earnings Persistence, and Stock Prices."
    Journal of Accounting and Economics, 39(3), 437–485.
    RSST extension: splits accruals into working capital, non-current
    operating, and financial accruals for finer manipulation detection.

Dechow, P. M., & Dichev, I. D. (2002). "The Quality of Accruals and
    Earnings: The Role of Accrual Estimation Errors." The Accounting
    Review, 77(Supplement), 35–59.

Accrual Thresholds
------------------
CF Accrual Ratio > 0.10 (10% of assets) = CRITICAL earnings quality risk.
CF Accrual Ratio 0.05–0.10 = ELEVATED.
Cash Earnings Ratio (CFO/NI) < 0.70 = CONCERN.
CFO/EBITDA < 0.85 = ELEVATED; < 0.70 = HIGH risk.

Audit Standards
---------------
ISA 315 §A42: Understanding revenue recognition as key assertion.
Ind AS 115 (IFRS 15): Revenue recognition — 5-step model.
SA 505 (ICAI): External Confirmations — receivables validation.

High accruals relative to assets indicate earnings not backed by cash.
Sloan (1996): Firms with high accruals tend to have lower future earnings.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from utils.helpers import safe_divide
from config import FORENSIC_THRESHOLDS


@dataclass
class AccrualResult:
    # Balance Sheet Accrual (Sloan 1996)
    bs_accrual: float = 0.0
    bs_accrual_ratio: float = 0.0

    # Cash Flow Accrual
    cf_accrual: float = 0.0
    cf_accrual_ratio: float = 0.0

    # Earnings Quality Ratio
    cash_earnings_ratio: float = 0.0  # CFO / Net Income
    cash_conversion_ratio: float = 0.0  # CFO / EBITDA

    # Quality of Revenue
    revenue_accrual_ratio: float = 0.0  # ΔAR / Revenue

    # Verdict
    earnings_quality: str = ""
    risk_level: str = ""
    interpretation: str = ""
    flags: list = field(default_factory=list)


class AccrualAnalyzer:
    """
    Multi-method accrual analysis for earnings quality assessment.
    Low cash earnings → high accruals → lower quality earnings.
    """

    def calculate(
        self,
        net_income: float,
        cfo: float,
        ebitda: float,
        total_assets: float,
        total_assets_prev: float,
        working_capital: float,
        working_capital_prev: float,
        cash: float,
        cash_prev: float,
        depreciation: float,
        revenue: float,
        accounts_receivable: float,
        accounts_receivable_prev: float,
        interest_expense: float = 0.0,
        tax: float = 0.0,
    ) -> AccrualResult:
        result = AccrualResult()
        avg_assets = (total_assets + total_assets_prev) / 2 if total_assets_prev else total_assets

        # ── Balance Sheet Accrual (Sloan 1996) ────────────────
        # BS Accrual = ΔWorking Capital - ΔCash - Depreciation
        delta_wc = working_capital - working_capital_prev
        delta_cash = cash - cash_prev
        result.bs_accrual = delta_wc - delta_cash - depreciation
        result.bs_accrual_ratio = safe_divide(result.bs_accrual, avg_assets)

        # ── Cash Flow Accrual ─────────────────────────────────
        # CF Accrual = Net Income - CFO
        result.cf_accrual = net_income - cfo
        result.cf_accrual_ratio = safe_divide(result.cf_accrual, avg_assets)

        # ── Cash Earnings Quality ─────────────────────────────
        # CFO / Net Income > 1 = earnings backed by cash (good quality)
        result.cash_earnings_ratio = safe_divide(cfo, net_income, default=0.0)

        # ── Cash Conversion Ratio ─────────────────────────────
        # CFO / EBITDA: should be > 0.85 for quality earnings
        result.cash_conversion_ratio = safe_divide(cfo, ebitda, default=0.0)

        # ── Revenue Accrual ───────────────────────────────────
        # ΔAR / Revenue: rising = revenue recognized before cash collected
        delta_ar = accounts_receivable - accounts_receivable_prev
        result.revenue_accrual_ratio = safe_divide(delta_ar, revenue)

        # ── Flags ─────────────────────────────────────────────
        flags = []

        if result.cf_accrual_ratio > FORENSIC_THRESHOLDS.accrual_ratio_high:
            flags.append({
                "type": "HIGH_ACCRUALS",
                "severity": "CRITICAL",
                "value": result.cf_accrual_ratio,
                "message": f"CF Accrual ratio {result.cf_accrual_ratio:.3f} > {FORENSIC_THRESHOLDS.accrual_ratio_high} threshold. Earnings significantly exceed cash generation."
            })
        elif result.cf_accrual_ratio > FORENSIC_THRESHOLDS.accrual_ratio_moderate:
            flags.append({
                "type": "ELEVATED_ACCRUALS",
                "severity": "HIGH",
                "value": result.cf_accrual_ratio,
                "message": f"CF Accrual ratio {result.cf_accrual_ratio:.3f} above moderate threshold."
            })

        if result.cash_earnings_ratio < 0.70:
            flags.append({
                "type": "LOW_CASH_CONVERSION",
                "severity": "HIGH",
                "value": result.cash_earnings_ratio,
                "message": f"Cash earnings ratio {result.cash_earnings_ratio:.2f}x. Only {result.cash_earnings_ratio*100:.0f}% of reported profits converted to cash."
            })

        if result.cash_conversion_ratio < FORENSIC_THRESHOLDS.cash_conversion_warning:
            flags.append({
                "type": "POOR_EBITDA_CONVERSION",
                "severity": "HIGH",
                "value": result.cash_conversion_ratio,
                "message": f"CFO/EBITDA = {result.cash_conversion_ratio:.2f}. EBITDA not converting to cash. Check working capital."
            })

        if result.revenue_accrual_ratio > 0.05:
            flags.append({
                "type": "RECEIVABLES_RISING",
                "severity": "MODERATE",
                "value": result.revenue_accrual_ratio,
                "message": f"ΔAR/Revenue = {result.revenue_accrual_ratio:.3f}. Receivables growing faster than cash revenue. Collection risk."
            })

        if result.cash_earnings_ratio < 0:
            flags.append({
                "type": "NEGATIVE_CFO_POSITIVE_INCOME",
                "severity": "CRITICAL",
                "value": result.cash_earnings_ratio,
                "message": "CRITICAL: Positive net income but negative CFO. Classic earnings manipulation signal."
            })

        result.flags = flags

        # ── Overall Earnings Quality Assessment ───────────────
        critical_flags = [f for f in flags if f["severity"] == "CRITICAL"]
        high_flags = [f for f in flags if f["severity"] == "HIGH"]

        if critical_flags or result.cf_accrual_ratio > 0.15:
            result.earnings_quality = "POOR"
            result.risk_level = "CRITICAL"
        elif len(high_flags) >= 2 or result.cf_accrual_ratio > 0.10:
            result.earnings_quality = "LOW"
            result.risk_level = "HIGH"
        elif len(flags) >= 1 or result.cf_accrual_ratio > 0.05:
            result.earnings_quality = "MODERATE"
            result.risk_level = "MODERATE"
        else:
            result.earnings_quality = "HIGH"
            result.risk_level = "LOW"

        result.interpretation = self._build_interpretation(result)
        return result

    def _build_interpretation(self, r: AccrualResult) -> str:
        parts = [
            f"CF Accrual Ratio: {r.cf_accrual_ratio:.3f} ({'⚠️ ELEVATED' if abs(r.cf_accrual_ratio) > 0.05 else '✅ Normal'})",
            f"Cash Earnings Ratio: {r.cash_earnings_ratio:.2f}x ({'✅ Good' if r.cash_earnings_ratio > 0.90 else '⚠️ Weak' if r.cash_earnings_ratio > 0.70 else '🚨 Poor'})",
            f"CFO/EBITDA: {r.cash_conversion_ratio:.2f} ({'✅ Healthy' if r.cash_conversion_ratio > 0.85 else '⚠️ Below target'})",
            f"ΔAR/Revenue: {r.revenue_accrual_ratio:.3f}",
        ]
        return " | ".join(parts)
