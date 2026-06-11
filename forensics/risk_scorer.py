"""
Risk Scorer - Aggregate risk scoring across all forensic dimensions
Produces a 0-100 overall risk score with band classification.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from config import RISK_SCORE_WEIGHTS
from utils.helpers import classify_risk


@dataclass
class RiskComponents:
    fraud_indicators: float = 50.0        # Beneish, Dechow, accruals
    earnings_quality: float = 50.0        # Accrual ratios, cash conversion
    cash_flow_quality: float = 50.0       # CFO quality, FCF analysis
    governance: float = 50.0             # Board, promoter, RPT
    credit_risk: float = 50.0            # Altman, leverage, coverage
    auditor_risk: float = 50.0           # KAMs, changes, qualifications
    management_credibility: float = 50.0  # NLP, guidance track record


@dataclass
class RiskScoreResult:
    overall_score: float = 50.0
    band: str = "MODERATE RISK"
    color: str = "#f57c00"
    components: RiskComponents = field(default_factory=RiskComponents)
    component_scores: dict = field(default_factory=dict)
    key_drivers: list = field(default_factory=list)
    interpretation: str = ""


class RiskScorer:
    """
    Weighted aggregate risk scoring.
    All component scores are 0-100 (100 = highest risk).
    """

    def calculate(self, components: RiskComponents) -> RiskScoreResult:
        result = RiskScoreResult()
        result.components = components

        comp_dict = {
            "fraud_indicators": components.fraud_indicators,
            "earnings_quality": components.earnings_quality,
            "cash_flow_quality": components.cash_flow_quality,
            "governance": components.governance,
            "credit_risk": components.credit_risk,
            "auditor_risk": components.auditor_risk,
            "management_credibility": components.management_credibility,
        }
        result.component_scores = comp_dict

        # Weighted average
        result.overall_score = sum(
            score * RISK_SCORE_WEIGHTS.get(name, 0.1)
            for name, score in comp_dict.items()
        )
        result.overall_score = max(0.0, min(100.0, result.overall_score))

        # Classify
        classification = classify_risk(result.overall_score)
        result.band = classification["band"]
        result.color = classification["color"]

        # Key drivers (top 3 highest risk components)
        sorted_components = sorted(comp_dict.items(), key=lambda x: x[1], reverse=True)
        result.key_drivers = [
            {"component": k.replace("_", " ").title(), "score": v}
            for k, v in sorted_components[:3]
            if v > 50
        ]

        result.interpretation = self._build_interpretation(result)
        return result

    def _build_interpretation(self, r: RiskScoreResult) -> str:
        if r.overall_score <= 20:
            return f"Overall Risk Score {r.overall_score:.1f}/100 - VERY LOW RISK. All forensic dimensions within acceptable ranges."
        elif r.overall_score <= 40:
            return f"Overall Risk Score {r.overall_score:.1f}/100 - LOW RISK. Minor concerns in {len(r.key_drivers)} dimension(s). Monitor."
        elif r.overall_score <= 60:
            return f"Overall Risk Score {r.overall_score:.1f}/100 - MODERATE RISK. Elevated concerns warrant deeper investigation."
        elif r.overall_score <= 80:
            return f"Overall Risk Score {r.overall_score:.1f}/100 - HIGH RISK. Multiple significant red flags detected. Avoid new positions."
        else:
            return f"Overall Risk Score {r.overall_score:.1f}/100 - EXTREME RISK. Strong evidence of multiple serious issues. Exit/Avoid."

    # ── Component Score Converters ────────────────────────────
    @staticmethod
    def beneish_to_score(m_score: float) -> float:
        """Convert Beneish M-Score to 0-100 risk score."""
        if m_score > -1.0:
            return 90.0
        elif m_score > -1.78:
            return 70.0 + (m_score - (-1.78)) / (-1.0 - (-1.78)) * 20
        elif m_score > -2.5:
            return 40.0 + (m_score - (-2.5)) / (-1.78 - (-2.5)) * 30
        else:
            return max(0.0, 40.0 + (m_score + 2.5) * 5)

    @staticmethod
    def altman_to_score(z_score: float) -> float:
        """Convert Altman Z-Score to 0-100 credit risk score (inverse).
        Uses EM non-manufacturing thresholds (Altman 2000): Safe>2.60, Grey 1.10-2.60, Distress<1.10."""
        if z_score > 4.0:
            return 10.0
        elif z_score > 2.60:   # Safe zone
            return 20.0 + (2.60 - z_score) / (2.60 - 4.0) * (-10)
        elif z_score > 1.10:   # Grey zone
            return 50.0 + (1.10 - z_score) / (1.10 - 2.60) * (-30)
        elif z_score > 0:      # Distress zone
            return 75.0 + (0 - z_score) / (0 - 1.10) * (-25)
        else:
            return 95.0

    @staticmethod
    def piotroski_to_score(f_score: int) -> float:
        """Convert Piotroski F-Score to 0-100 risk score (inverse)."""
        return max(0.0, min(100.0, (9 - f_score) / 9 * 100))

    @staticmethod
    def accrual_to_score(cf_accrual_ratio: float) -> float:
        """Convert CF accrual ratio to earnings quality risk score."""
        abs_ratio = abs(cf_accrual_ratio)
        if abs_ratio < 0.02:
            return 10.0
        elif abs_ratio < 0.05:
            return 30.0
        elif abs_ratio < 0.10:
            return 55.0
        elif abs_ratio < 0.15:
            return 75.0
        else:
            return 90.0

    @staticmethod
    def cash_conversion_to_score(ratio: float) -> float:
        """Convert CFO/EBITDA to earnings quality score."""
        if ratio >= 0.90:
            return 10.0
        elif ratio >= 0.75:
            return 30.0
        elif ratio >= 0.60:
            return 55.0
        elif ratio >= 0.40:
            return 75.0
        elif ratio >= 0:
            return 85.0
        else:
            return 95.0  # Negative CFO with positive EBITDA

    @staticmethod
    def leverage_to_score(de_ratio: float, interest_coverage: float) -> float:
        """Leverage risk score from D/E and interest coverage."""
        de_risk = min(100.0, de_ratio / 4.0 * 80)
        if interest_coverage > 5:
            ic_risk = 10.0
        elif interest_coverage > 2.5:
            ic_risk = 40.0
        elif interest_coverage > 1.5:
            ic_risk = 70.0
        elif interest_coverage > 1.0:
            ic_risk = 85.0
        else:
            ic_risk = 95.0
        return (de_risk + ic_risk) / 2
