"""
Piotroski F-Score — Financial Strength Assessment
==================================================

References
----------
Piotroski, J. D. (2000). "Value Investing: The Use of Historical
    Financial Statement Information to Separate Winners from Losers."
    Journal of Accounting Research, 38(Supplement), 1–41.
    9 binary signals across 3 categories (profitability, leverage/liquidity,
    efficiency). Score 8–9 = strong; 0–2 = distressed.

Hyde, C. E. (2018). "The Piotroski F-Score: Evidence from Australia."
    Accounting & Finance, 58(2), 423–444. (cross-market validation)

F-Score Signal Categories
--------------------------
Profitability (F1–F4): ROA > 0; CFO > 0; ΔROA > 0; CFO/Assets > ROA.
Leverage/Liquidity (F5–F7): ΔLeverage < 0; ΔCurrent Ratio > 0; no dilution.
Efficiency (F8–F9): ΔGross Margin > 0; ΔAsset Turnover > 0.

Regulatory Context
------------------
SEBI LODR Reg. 33: Quarterly financial results — basis for YoY trend.
Ind AS 7 (IAS 7): Statement of Cash Flows — CFO signal accuracy.
CARO 2020 Para 3(vii): Comment on company's ability to service debt.

9 binary criteria across 3 categories. Score: 0-9.
Strong: 7-9 | Neutral: 3-6 | Weak: 0-2
"""

from __future__ import annotations
from dataclasses import dataclass, field
from utils.helpers import safe_divide


@dataclass
class PiotroskiInputs:
    # Current year
    net_income: float = 0.0
    total_assets: float = 0.0
    cfo: float = 0.0
    roa: float = 0.0              # Or calculated from net_income / avg_assets
    total_debt: float = 0.0
    current_ratio: float = 0.0
    shares_outstanding: float = 0.0
    gross_profit: float = 0.0
    revenue: float = 0.0
    asset_turnover: float = 0.0
    long_term_debt: float = 0.0

    # Prior year (for delta calculations)
    net_income_prev: float = 0.0
    total_assets_prev: float = 0.0
    roa_prev: float = 0.0
    total_debt_prev: float = 0.0
    current_ratio_prev: float = 0.0
    shares_outstanding_prev: float = 0.0
    gross_profit_prev: float = 0.0
    revenue_prev: float = 0.0
    asset_turnover_prev: float = 0.0
    long_term_debt_prev: float = 0.0


@dataclass
class PiotroskiCriteria:
    # Profitability
    f1_roa_positive: bool = False           # ROA > 0
    f2_cfo_positive: bool = False           # CFO > 0
    f3_roa_increasing: bool = False         # Delta ROA > 0
    f4_accrual_quality: bool = False        # CFO/TA > ROA (cash earnings > accrual earnings)
    # Leverage / Liquidity
    f5_leverage_decreasing: bool = False    # Long-term debt ratio decreased
    f6_liquidity_improving: bool = False    # Current ratio increased
    f7_no_dilution: bool = False            # No new shares issued
    # Operating Efficiency
    f8_gross_margin_improving: bool = False # Gross margin increased
    f9_asset_turnover_improving: bool = False # Asset turnover increased


@dataclass
class PiotroskiResult:
    f_score: int = 0
    criteria: PiotroskiCriteria = field(default_factory=PiotroskiCriteria)
    classification: str = ""
    risk_level: str = ""
    interpretation: str = ""
    detail: dict = field(default_factory=dict)


class PiotroskiFScore:
    """Full Piotroski F-Score implementation with detailed diagnostics."""

    def calculate(self, inp: PiotroskiInputs) -> PiotroskiResult:
        result = PiotroskiResult()
        c = result.criteria

        avg_assets = (inp.total_assets + inp.total_assets_prev) / 2 if inp.total_assets_prev else inp.total_assets
        roa_t = inp.roa if inp.roa else safe_divide(inp.net_income, avg_assets)
        roa_tm1 = inp.roa_prev if inp.roa_prev else safe_divide(inp.net_income_prev, inp.total_assets_prev)

        # ── PROFITABILITY GROUP ───────────────────────────────
        # F1: ROA > 0 (positive net income relative to assets)
        c.f1_roa_positive = roa_t > 0

        # F2: CFO > 0 (positive operating cash flow)
        c.f2_cfo_positive = inp.cfo > 0

        # F3: Delta ROA > 0 (improving profitability)
        c.f3_roa_increasing = roa_t > roa_tm1

        # F4: Accrual quality (CFO > ROA*Assets → cash earnings > accrual earnings)
        # CFO/TA > ROA means cash generation exceeds accrual-based earnings
        cfo_ta = safe_divide(inp.cfo, inp.total_assets)
        c.f4_accrual_quality = cfo_ta > roa_t

        # ── LEVERAGE / LIQUIDITY GROUP ───────────────────────
        # F5: Long-term debt ratio decreased (deleveraging)
        ltd_ratio_t = safe_divide(inp.long_term_debt, inp.total_assets)
        ltd_ratio_tm1 = safe_divide(inp.long_term_debt_prev, inp.total_assets_prev)
        c.f5_leverage_decreasing = ltd_ratio_t < ltd_ratio_tm1

        # F6: Current ratio improved
        c.f6_liquidity_improving = inp.current_ratio > inp.current_ratio_prev

        # F7: No new equity issuance (no dilution)
        c.f7_no_dilution = inp.shares_outstanding <= inp.shares_outstanding_prev * 1.005  # 0.5% tolerance

        # ── OPERATING EFFICIENCY GROUP ────────────────────────
        # F8: Gross margin improved
        gm_t = safe_divide(inp.gross_profit, inp.revenue)
        gm_tm1 = safe_divide(inp.gross_profit_prev, inp.revenue_prev)
        c.f8_gross_margin_improving = gm_t > gm_tm1

        # F9: Asset turnover improved
        at_t = inp.asset_turnover if inp.asset_turnover else safe_divide(inp.revenue, inp.total_assets)
        at_tm1 = inp.asset_turnover_prev if inp.asset_turnover_prev else safe_divide(inp.revenue_prev, inp.total_assets_prev)
        c.f9_asset_turnover_improving = at_t > at_tm1

        # ── Total Score ───────────────────────────────────────
        result.f_score = sum([
            c.f1_roa_positive, c.f2_cfo_positive, c.f3_roa_increasing, c.f4_accrual_quality,
            c.f5_leverage_decreasing, c.f6_liquidity_improving, c.f7_no_dilution,
            c.f8_gross_margin_improving, c.f9_asset_turnover_improving,
        ])

        # ── Classification ────────────────────────────────────
        if result.f_score >= 7:
            result.classification = "STRONG"
            result.risk_level = "LOW"
            result.interpretation = (
                f"F-Score {result.f_score}/9: FINANCIALLY STRONG. "
                f"Company shows positive signals across profitability, leverage, and efficiency."
            )
        elif result.f_score <= 2:
            result.classification = "WEAK"
            result.risk_level = "HIGH"
            result.interpretation = (
                f"F-Score {result.f_score}/9: FINANCIALLY WEAK. "
                f"Multiple deteriorating signals. Elevated risk of financial distress."
            )
        else:
            result.classification = "NEUTRAL"
            result.risk_level = "MODERATE"
            result.interpretation = (
                f"F-Score {result.f_score}/9: NEUTRAL financial strength. "
                f"Mixed signals. Focus on weakest individual criteria."
            )

        result.detail = {
            "PROFITABILITY": {
                "F1 ROA Positive": {"pass": c.f1_roa_positive, "value": f"ROA={roa_t:.3f}"},
                "F2 CFO Positive": {"pass": c.f2_cfo_positive, "value": f"CFO={inp.cfo:,.0f}"},
                "F3 ROA Improving": {"pass": c.f3_roa_increasing, "value": f"ΔROA={roa_t-roa_tm1:.3f}"},
                "F4 Accrual Quality": {"pass": c.f4_accrual_quality, "value": f"CFO/TA={cfo_ta:.3f} vs ROA={roa_t:.3f}"},
            },
            "LEVERAGE_LIQUIDITY": {
                "F5 Leverage Decreasing": {"pass": c.f5_leverage_decreasing, "value": f"LTD/TA: {ltd_ratio_tm1:.3f}→{ltd_ratio_t:.3f}"},
                "F6 Liquidity Improving": {"pass": c.f6_liquidity_improving, "value": f"CR: {inp.current_ratio_prev:.2f}→{inp.current_ratio:.2f}"},
                "F7 No Dilution": {"pass": c.f7_no_dilution, "value": f"Shares: {inp.shares_outstanding_prev:,.0f}→{inp.shares_outstanding:,.0f}"},
            },
            "EFFICIENCY": {
                "F8 Gross Margin Improving": {"pass": c.f8_gross_margin_improving, "value": f"GM: {gm_tm1:.3f}→{gm_t:.3f}"},
                "F9 Asset Turnover Improving": {"pass": c.f9_asset_turnover_improving, "value": f"AT: {at_tm1:.3f}→{at_t:.3f}"},
            },
        }
        return result
