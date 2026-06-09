"""
Altman Z-Score - Bankruptcy Prediction Model
=============================================
Original: Altman (1968) "Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy"
Revised: Altman (2000) for private firms and non-manufacturing firms.

Manufacturing (public):  Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
Non-Manufacturing:       Z' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
Private Firms:           Z'' = 0.717*X1 + 0.847*X2 + 3.107*X3 + 0.420*X4 + 0.998*X5

Safe Zone (Manufacturing):     Z > 2.99
Grey Zone:                1.81 < Z < 2.99
Distress Zone:                 Z < 1.81
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from utils.helpers import safe_divide
from config import FORENSIC_THRESHOLDS


@dataclass
class AltmanInputs:
    working_capital: float = 0.0
    total_assets: float = 0.0
    retained_earnings: float = 0.0
    ebit: float = 0.0
    market_cap: float = 0.0      # For public companies (X4 numerator)
    book_equity: float = 0.0     # For private companies (X4 numerator)
    total_liabilities: float = 0.0
    revenue: float = 0.0
    is_manufacturing: bool = True
    is_public: bool = True


@dataclass
class AltmanResult:
    z_score: float = 0.0
    x1: float = 0.0   # Working Capital / Total Assets
    x2: float = 0.0   # Retained Earnings / Total Assets
    x3: float = 0.0   # EBIT / Total Assets
    x4: float = 0.0   # Market Cap or Book Equity / Total Liabilities
    x5: float = 0.0   # Revenue / Total Assets
    zone: str = "UNKNOWN"
    zone_description: str = ""
    risk_level: str = "UNKNOWN"
    model_used: str = ""
    interpretation: str = ""
    formula_trace: dict = field(default_factory=dict)


class AltmanZScore:
    """
    Altman Z-Score with three model variants and credit risk interpretation.
    """

    MODELS = {
        "manufacturing_public": {
            "coefficients": {"x1": 1.2, "x2": 1.4, "x3": 3.3, "x4": 0.6, "x5": 1.0},
            "safe": 2.99,
            "grey_upper": 2.99,
            "grey_lower": 1.81,
            "distress": 1.81,
            "formula": "Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5",
        },
        "non_manufacturing": {
            "coefficients": {"x1": 6.56, "x2": 3.26, "x3": 6.72, "x4": 1.05, "x5": 0.0},
            "safe": 2.60,
            "grey_upper": 2.60,
            "grey_lower": 1.10,
            "distress": 1.10,
            "formula": "Z' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4",
        },
        "private": {
            "coefficients": {"x1": 0.717, "x2": 0.847, "x3": 3.107, "x4": 0.420, "x5": 0.998},
            "safe": 2.90,
            "grey_upper": 2.90,
            "grey_lower": 1.23,
            "distress": 1.23,
            "formula": "Z'' = 0.717*X1 + 0.847*X2 + 3.107*X3 + 0.420*X4 + 0.998*X5",
        },
    }

    def calculate(self, inp: AltmanInputs) -> AltmanResult:
        result = AltmanResult()

        # Select model variant
        if inp.is_public and inp.is_manufacturing:
            model_key = "manufacturing_public"
        elif inp.is_public and not inp.is_manufacturing:
            model_key = "non_manufacturing"
        else:
            model_key = "private"

        model = self.MODELS[model_key]
        coef = model["coefficients"]
        result.model_used = model_key

        # ── X1: Working Capital / Total Assets ────────────────
        # Measures liquidity. Negative WC = short-term stress
        result.x1 = safe_divide(inp.working_capital, inp.total_assets)

        # ── X2: Retained Earnings / Total Assets ──────────────
        # Measures cumulative profitability. Low = young firm or persistent losses
        result.x2 = safe_divide(inp.retained_earnings, inp.total_assets)

        # ── X3: EBIT / Total Assets ───────────────────────────
        # Measures operating profitability / asset productivity
        result.x3 = safe_divide(inp.ebit, inp.total_assets)

        # ── X4: Market Cap / Total Liabilities (public) ───────
        # OR Book Equity / Total Liabilities (private)
        equity_value = inp.market_cap if (inp.is_public and inp.market_cap > 0) else inp.book_equity
        result.x4 = safe_divide(equity_value, inp.total_liabilities)

        # ── X5: Revenue / Total Assets ────────────────────────
        # Asset turnover - measures efficiency
        result.x5 = safe_divide(inp.revenue, inp.total_assets)

        # ── Z-Score ───────────────────────────────────────────
        result.z_score = (
            coef["x1"] * result.x1
            + coef["x2"] * result.x2
            + coef["x3"] * result.x3
            + coef["x4"] * result.x4
            + coef["x5"] * result.x5
        )

        # ── Zone Classification ───────────────────────────────
        result.zone, result.zone_description, result.risk_level = self._classify(result.z_score, model)
        result.interpretation = self._interpret(result)

        result.formula_trace = {
            "model": model_key,
            "formula": model["formula"],
            "z_score": result.z_score,
            "components": {
                "X1 (WC/TA)": result.x1,
                "X2 (RE/TA)": result.x2,
                "X3 (EBIT/TA)": result.x3,
                "X4 (Equity/Liab)": result.x4,
                "X5 (Sales/TA)": result.x5,
            },
            "zone_boundaries": {
                "safe": model["safe"],
                "grey_lower": model["grey_lower"],
                "distress": model["distress"],
            },
        }
        return result

    def _classify(self, z: float, model: dict) -> tuple[str, str, str]:
        if z > model["grey_upper"]:
            return (
                "SAFE",
                f"Z-Score {z:.3f} > {model['grey_upper']}: Company in SAFE zone. Low short-term bankruptcy risk.",
                "LOW",
            )
        elif z > model["grey_lower"]:
            return (
                "GREY",
                f"Z-Score {z:.3f} in GREY ZONE ({model['grey_lower']:.2f}–{model['grey_upper']:.2f}). "
                f"Elevated financial stress. Monitor credit metrics closely.",
                "MODERATE",
            )
        else:
            return (
                "DISTRESS",
                f"Z-Score {z:.3f} < {model['distress']}: Company in DISTRESS zone. "
                f"High probability of financial difficulty within 2 years.",
                "CRITICAL",
            )

    def _interpret(self, result: AltmanResult) -> str:
        interpretations = []

        if result.x1 < 0:
            interpretations.append(
                f"NEGATIVE working capital (X1={result.x1:.3f}): Company may struggle to meet short-term obligations."
            )
        if result.x2 < 0:
            interpretations.append(
                f"NEGATIVE retained earnings (X2={result.x2:.3f}): History of losses or dividend payouts exceeding profits."
            )
        if result.x3 < 0.05:
            interpretations.append(
                f"LOW EBIT/Assets (X3={result.x3:.3f}): Asset base not generating adequate operating returns."
            )
        if result.x4 < 0.5:
            interpretations.append(
                f"LOW equity/liability coverage (X4={result.x4:.3f}): Liabilities significantly exceed equity buffer."
            )
        if result.x5 < 0.5:
            interpretations.append(
                f"LOW asset turnover (X5={result.x5:.3f}): Assets not being utilized efficiently to generate revenue."
            )

        if not interpretations:
            interpretations.append("All individual Z-Score components within acceptable ranges.")

        return " | ".join(interpretations)

    def get_implied_credit_rating(self, z_score: float) -> str:
        """Approximate credit rating implied by Z-Score (Altman 2000 mapping)."""
        if z_score > 8.15:
            return "AAA"
        elif z_score > 7.60:
            return "AA+"
        elif z_score > 7.30:
            return "AA"
        elif z_score > 7.00:
            return "AA-"
        elif z_score > 6.85:
            return "A+"
        elif z_score > 6.65:
            return "A"
        elif z_score > 6.40:
            return "A-"
        elif z_score > 6.25:
            return "BBB+"
        elif z_score > 5.85:
            return "BBB"
        elif z_score > 5.65:
            return "BBB-"
        elif z_score > 5.25:
            return "BB+"
        elif z_score > 4.95:
            return "BB"
        elif z_score > 4.75:
            return "BB-"
        elif z_score > 4.50:
            return "B+"
        elif z_score > 4.15:
            return "B"
        elif z_score > 3.75:
            return "B-"
        elif z_score > 3.20:
            return "CCC+"
        elif z_score > 2.50:
            return "CCC"
        elif z_score > 1.75:
            return "CCC-"
        elif z_score > 0:
            return "CC/C"
        else:
            return "D"
