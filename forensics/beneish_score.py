"""
Beneish M-Score - Earnings Manipulation Probability Model
=========================================================
Original: Beneish (1999) "The Detection of Earnings Manipulation"
M-Score > -1.78 suggests high probability of earnings manipulation.

8 Financial Ratios:
1. DSRI  - Days Sales Receivable Index
2. GMI   - Gross Margin Index
3. AQI   - Asset Quality Index
4. SGI   - Sales Growth Index
5. DEPI  - Depreciation Index
6. SGAI  - SG&A Index
7. LVGI  - Leverage Index
8. TATA  - Total Accruals to Total Assets
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from utils.helpers import safe_divide
from config import FORENSIC_THRESHOLDS


@dataclass
class BeneishInputs:
    """Financial inputs for Beneish M-Score calculation."""
    # Current Year (t)
    net_receivables_t: float = 0.0
    sales_t: float = 0.0
    cogs_t: float = 0.0
    current_assets_t: float = 0.0
    ppe_t: float = 0.0         # Plant, Property & Equipment (gross)
    total_assets_t: float = 0.0
    depreciation_t: float = 0.0
    sga_t: float = 0.0         # Selling, General & Administrative
    total_debt_t: float = 0.0
    current_liabilities_t: float = 0.0
    working_capital_t: float = 0.0
    cash_t: float = 0.0
    taxes_payable_t: float = 0.0

    # Prior Year (t-1)
    net_receivables_tm1: float = 0.0
    sales_tm1: float = 0.0
    cogs_tm1: float = 0.0
    current_assets_tm1: float = 0.0
    ppe_tm1: float = 0.0
    total_assets_tm1: float = 0.0
    depreciation_tm1: float = 0.0
    sga_tm1: float = 0.0
    total_debt_tm1: float = 0.0
    current_liabilities_tm1: float = 0.0
    working_capital_tm1: float = 0.0
    cash_tm1: float = 0.0


@dataclass
class BeneishResult:
    m_score: float = 0.0
    dsri: float = 0.0
    gmi: float = 0.0
    aqi: float = 0.0
    sgi: float = 0.0
    depi: float = 0.0
    sgai: float = 0.0
    lvgi: float = 0.0
    tata: float = 0.0
    manipulation_likely: bool = False
    risk_level: str = "LOW"
    interpretation: str = ""
    component_flags: dict = field(default_factory=dict)
    formula_trace: dict = field(default_factory=dict)
    # Count of Beneish indices above the Beneish (1999) paper manipulation thresholds.
    # 3+ = elevated risk regardless of aggregate M-Score (Beneish 1999, Table 3).
    flags_above_paper_threshold: int = 0


class BeneishMScore:
    """
    Institutional-grade Beneish M-Score calculator.

    Coefficients (8-variable model):
    M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
         + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI
    """

    COEFFICIENTS = {
        "intercept": -4.84,
        "dsri": 0.920,
        "gmi": 0.528,
        "aqi": 0.404,
        "sgi": 0.892,
        "depi": 0.115,
        "sgai": -0.172,
        "tata": 4.679,
        "lvgi": -0.327,
    }

    # Individual component thresholds.
    # "critical" = exact manipulation-signal threshold from Beneish (1999) Table 3.
    # "concern"  = early-warning level (half of paper threshold or academic norm).
    COMPONENT_THRESHOLDS = {
        "dsri": {"concern": 1.100, "critical": 1.465, "label": "Days Sales Receivable Index"},
        "gmi":  {"concern": 1.050, "critical": 1.193, "label": "Gross Margin Index"},
        "aqi":  {"concern": 1.050, "critical": 1.254, "label": "Asset Quality Index"},
        "sgi":  {"concern": 1.200, "critical": 1.607, "label": "Sales Growth Index"},
        "depi": {"concern": 1.020, "critical": 1.083, "label": "Depreciation Index"},
        "sgai": {"concern": 1.020, "critical": 1.041, "label": "SG&A Index"},
        "lvgi": {"concern": 1.050, "critical": 1.111, "label": "Leverage Index"},
        "tata": {"concern": 0.015, "critical": 0.031, "label": "Total Accruals / Total Assets"},
    }

    # Paper manipulation-signal thresholds (Beneish 1999, Table 3) used for counting.
    _PAPER_THRESHOLDS = {
        "dsri": 1.465, "gmi": 1.193, "aqi": 1.254, "sgi": 1.607,
        "depi": 1.083, "sgai": 1.041, "lvgi": 1.111, "tata": 0.031,
    }

    def calculate(self, inp: BeneishInputs) -> BeneishResult:
        result = BeneishResult()

        # ── 1. DSRI: Days Sales Receivable Index ──────────────────
        # (AR_t / Sales_t) / (AR_tm1 / Sales_tm1)
        # Rising DSRI → receivables growing faster than sales → revenue inflation risk
        dsr_t = safe_divide(inp.net_receivables_t, inp.sales_t)
        dsr_tm1 = safe_divide(inp.net_receivables_tm1, inp.sales_tm1)
        result.dsri = safe_divide(dsr_t, dsr_tm1, default=1.0)

        # ── 2. GMI: Gross Margin Index ────────────────────────────
        # (Gross Margin_tm1) / (Gross Margin_t)
        # GMI > 1 → deteriorating gross margin → earnings pressure
        gm_t = safe_divide(inp.sales_t - inp.cogs_t, inp.sales_t)
        gm_tm1 = safe_divide(inp.sales_tm1 - inp.cogs_tm1, inp.sales_tm1)
        result.gmi = safe_divide(gm_tm1, gm_t, default=1.0)

        # ── 3. AQI: Asset Quality Index ───────────────────────────
        # (1 - (CA_t + PPE_t) / TA_t) / (1 - (CA_tm1 + PPE_tm1) / TA_tm1)
        # AQI > 1 → more assets in intangibles/deferred costs → earnings manipulation risk
        aq_t = 1 - safe_divide(inp.current_assets_t + inp.ppe_t, inp.total_assets_t)
        aq_tm1 = 1 - safe_divide(inp.current_assets_tm1 + inp.ppe_tm1, inp.total_assets_tm1)
        result.aqi = safe_divide(aq_t, aq_tm1, default=1.0)

        # ── 4. SGI: Sales Growth Index ────────────────────────────
        # Sales_t / Sales_tm1
        # High SGI alone is not suspicious, but combined with DSRI/TATA, it amplifies risk
        result.sgi = safe_divide(inp.sales_t, inp.sales_tm1, default=1.0)

        # ── 5. DEPI: Depreciation Index ───────────────────────────
        # (Dep_tm1 / (PPE_tm1 + Dep_tm1)) / (Dep_t / (PPE_t + Dep_t))
        # DEPI > 1 → declining depreciation rate → could be extending asset lives to boost earnings
        dep_rate_tm1 = safe_divide(inp.depreciation_tm1, inp.ppe_tm1 + inp.depreciation_tm1)
        dep_rate_t = safe_divide(inp.depreciation_t, inp.ppe_t + inp.depreciation_t)
        result.depi = safe_divide(dep_rate_tm1, dep_rate_t, default=1.0)

        # ── 6. SGAI: SG&A Index ───────────────────────────────────
        # (SGA_t / Sales_t) / (SGA_tm1 / Sales_tm1)
        # Rising SGAI → disproportionate overhead growth
        sgai_t = safe_divide(inp.sga_t, inp.sales_t)
        sgai_tm1 = safe_divide(inp.sga_tm1, inp.sales_tm1)
        result.sgai = safe_divide(sgai_t, sgai_tm1, default=1.0)

        # ── 7. LVGI: Leverage Index ───────────────────────────────
        # ((LTD_t + CL_t) / TA_t) / ((LTD_tm1 + CL_tm1) / TA_tm1)
        # Rising leverage → debt covenant pressure → motivation to manipulate
        lev_t = safe_divide(inp.total_debt_t + inp.current_liabilities_t, inp.total_assets_t)
        lev_tm1 = safe_divide(inp.total_debt_tm1 + inp.current_liabilities_tm1, inp.total_assets_tm1)
        result.lvgi = safe_divide(lev_t, lev_tm1, default=1.0)

        # ── 8. TATA: Total Accruals to Total Assets ───────────────
        # (ΔWC - ΔCash - ΔCurrent LTD - ΔTax Payable - Depreciation) / TA
        # Positive TATA → accruals-based earnings not supported by cash
        working_capital_change = inp.working_capital_t - inp.working_capital_tm1
        result.tata = safe_divide(
            working_capital_change - (inp.cash_t - inp.cash_tm1) - inp.taxes_payable_t - inp.depreciation_t,
            inp.total_assets_t,
        )

        # ── M-Score Calculation ───────────────────────────────────
        c = self.COEFFICIENTS
        result.m_score = (
            c["intercept"]
            + c["dsri"] * result.dsri
            + c["gmi"] * result.gmi
            + c["aqi"] * result.aqi
            + c["sgi"] * result.sgi
            + c["depi"] * result.depi
            + c["sgai"] * result.sgai
            + c["tata"] * result.tata
            + c["lvgi"] * result.lvgi
        )

        # ── Interpretation ────────────────────────────────────────
        result.manipulation_likely = result.m_score > FORENSIC_THRESHOLDS.beneish_manipulation_threshold

        if result.m_score > -1.0:
            result.risk_level = "CRITICAL"
            result.interpretation = (
                f"M-Score of {result.m_score:.3f} is CRITICALLY HIGH. "
                f"Strong statistical evidence of earnings manipulation. "
                f"Threshold is -1.78. The company appears to be a MANIPULATOR."
            )
        elif result.m_score > -1.78:
            result.risk_level = "HIGH"
            result.interpretation = (
                f"M-Score of {result.m_score:.3f} exceeds the -1.78 manipulation threshold. "
                f"The model classifies this company as a likely MANIPULATOR."
            )
        elif result.m_score > -2.22:
            result.risk_level = "MODERATE"
            result.interpretation = (
                f"M-Score of {result.m_score:.3f} is in the grey zone. "
                f"Some earnings quality concerns warranting closer investigation."
            )
        else:
            result.risk_level = "LOW"
            result.interpretation = (
                f"M-Score of {result.m_score:.3f} is below the manipulation threshold. "
                f"No strong statistical evidence of earnings manipulation from this model alone."
            )

        # ── Count indices above Beneish (1999) paper thresholds ───
        indices = {
            "dsri": result.dsri, "gmi": result.gmi, "aqi": result.aqi,
            "sgi": result.sgi, "depi": result.depi, "sgai": result.sgai,
            "lvgi": result.lvgi, "tata": result.tata,
        }
        result.flags_above_paper_threshold = sum(
            1 for k, v in indices.items() if v > self._PAPER_THRESHOLDS.get(k, 9999)
        )

        # ── Flag concerning individual components ─────────────────
        result.component_flags = self._flag_components(result)
        result.formula_trace = {
            "m_score": result.m_score,
            "formula": "M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI",
            "components": {
                "DSRI": result.dsri, "GMI": result.gmi, "AQI": result.aqi,
                "SGI": result.sgi, "DEPI": result.depi, "SGAI": result.sgai,
                "TATA": result.tata, "LVGI": result.lvgi,
            },
        }
        return result

    def _flag_components(self, result: BeneishResult) -> dict:
        flags = {}
        components = {
            "dsri": result.dsri, "gmi": result.gmi, "aqi": result.aqi,
            "sgi": result.sgi, "depi": result.depi, "sgai": result.sgai,
            "tata": result.tata, "lvgi": result.lvgi,
        }
        for name, value in components.items():
            threshold = self.COMPONENT_THRESHOLDS.get(name, {})
            critical = threshold.get("critical", 9999)
            concern = threshold.get("concern", 9999)
            label = threshold.get("label", name.upper())
            if name == "tata":  # TATA uses different threshold
                if abs(value) > 0.10:
                    flags[name] = {"label": label, "value": value, "severity": "CRITICAL", "concern": True}
                elif abs(value) > 0.05:
                    flags[name] = {"label": label, "value": value, "severity": "HIGH", "concern": True}
            else:
                if value > critical:
                    flags[name] = {"label": label, "value": value, "severity": "CRITICAL", "concern": True}
                elif value > concern:
                    flags[name] = {"label": label, "value": value, "severity": "MODERATE", "concern": True}
        return flags

    def interpret_dsri(self, dsri: float) -> str:
        if dsri > 1.465:
            return f"DSRI={dsri:.3f}: ABOVE Beneish paper manipulation threshold (1.465). Receivables growing disproportionately faster than sales — HIGH RISK of revenue inflation or channel stuffing."
        elif dsri > 1.100:
            return f"DSRI={dsri:.3f}: Receivables growing noticeably faster than sales. Approaching manipulation threshold. Monitor for collection quality deterioration."
        else:
            return f"DSRI={dsri:.3f}: Receivables growing in line with or slower than sales. Normal."

    def get_manipulation_probability(self, m_score: float) -> float:
        """Convert M-Score to approximate manipulation probability (logistic transformation)."""
        import math
        try:
            # Logistic function: P = 1 / (1 + e^(-m_score))
            # Calibrated so M=-1.78 → ~50% probability
            adjusted = m_score + 1.78
            prob = 1 / (1 + math.exp(-2.5 * adjusted))
            return round(prob * 100, 1)
        except (OverflowError, ValueError):
            return 50.0
