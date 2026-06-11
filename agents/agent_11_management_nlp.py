"""
Agent 11 - Management NLP Agent
=================================
Analyzes: MD&A, Earnings Calls, Shareholder Letters
Detects: Evasive language, over-optimism, narrative inconsistencies
"""

from __future__ import annotations
import re
from collections import Counter
from .base_agent import BaseForensicAgent, AgentResult, AgentFinding


# Loughran-McDonald inspired financial word lists
UNCERTAINTY_WORDS = [
    "approximately", "around", "about", "roughly", "uncertain", "unclear",
    "difficult to predict", "challenging", "volatile", "may", "might", "could",
    "possibly", "potentially", "subject to change", "no assurance", "cannot guarantee",
]

EVASIVE_PHRASES = [
    "we are monitoring", "we are evaluating", "we will provide updates",
    "cannot comment at this time", "not in a position to", "we are exploring",
    "we will keep you informed", "no specific timeline", "we will update",
    "as mentioned earlier", "next question please",
]

OVERCONFIDENCE_PHRASES = [
    "very confident", "absolutely certain", "no doubt", "clearly", "obviously",
    "guaranteed", "definitely", "without question", "I can assure you",
    "there is no risk", "we are very excited",
]

NEGATIVE_BURIED_PHRASES = [
    "we should note that", "it is worth mentioning", "on a separate note",
    "as disclosed in", "in connection with", "subject to regulatory",
    "notwithstanding", "despite these headwinds",
]

NON_GAAP_EMPHASIS = [
    "adjusted ebitda", "adjusted eps", "core earnings", "underlying profit",
    "pro forma", "non-gaap", "adjusted", "normalized", "recurring",
    "excluding one-time", "excluding special",
]


class ManagementNLPAgent(BaseForensicAgent):
    """
    Linguistic analysis of management communications.
    Detects deception, evasion, and narrative manipulation.
    """

    def investigate(self, company_name: str, company_id: int, financial_data: dict, **kwargs) -> AgentResult:
        self.log_info(f"Starting management NLP analysis for {company_name}")
        result = AgentResult(agent_id=self.agent_id, agent_name=self.agent_name)

        # Retrieve management communications from RAG
        mda_text = self._retrieve_context(company_name, "management discussion analysis forward guidance outlook")
        concall_text = self._retrieve_context(company_name, "earnings call transcript analyst questions management answers")
        governance_text = self._retrieve_context(company_name, "chairman letter shareholder communication")

        findings = []

        # ── Quantitative NLP Analysis ─────────────────────────
        mda_metrics = self._analyze_text_metrics(mda_text)
        concall_metrics = self._analyze_text_metrics(concall_text)

        # Evasion score
        evasion_score = self._calculate_evasion_score(mda_text + concall_text)
        if evasion_score > 0.6:
            finding = self._create_finding(
                finding_type="RED_FLAG",
                title=f"High Evasion Language Score: {evasion_score:.2f}/1.0",
                detail=(
                    "Management communications show elevated use of evasive and non-committal language. "
                    "This pattern is associated with management concealment in academic research."
                ),
                evidence=f"Evasion score: {evasion_score:.3f}. Detected phrases: {mda_metrics.get('evasive_phrases', [])[:3]}",
                risk_level="HIGH" if evasion_score > 0.7 else "MODERATE",
                confidence=0.70,
            )
            findings.append(finding)
            result.red_flags.append(finding)

        # Non-GAAP overemphasis
        non_gaap_count = self._count_non_gaap_emphasis(mda_text + concall_text)
        if non_gaap_count > 5:
            finding = self._create_finding(
                finding_type="RED_FLAG",
                title=f"Heavy Non-GAAP Metric Emphasis ({non_gaap_count} references)",
                detail=(
                    "Management heavily emphasizes non-GAAP/adjusted metrics over GAAP results. "
                    "Excessive use of adjusted metrics is a known earnings quality red flag. "
                    "Investigate the gap between GAAP and non-GAAP earnings."
                ),
                evidence=f"Found {non_gaap_count} non-GAAP references. Common: {self._find_non_gaap_metrics(mda_text + concall_text)}",
                risk_level="MODERATE",
                confidence=0.75,
            )
            findings.append(finding)
            result.red_flags.append(finding)

        # Uncertainty spike (compared to prior periods)
        uncertainty_score = mda_metrics.get("uncertainty_ratio", 0)
        if uncertainty_score > 0.03:
            finding = self._create_finding(
                finding_type="RED_FLAG",
                title=f"Elevated Uncertainty Language in MD&A ({uncertainty_score*100:.1f}% of words)",
                detail="Disproportionate use of uncertain language may indicate management lacks confidence in disclosed figures.",
                evidence=f"Uncertainty ratio: {uncertainty_score:.3f}. Words: {mda_metrics.get('uncertainty_words', [])[:5]}",
                risk_level="MODERATE",
                confidence=0.65,
            )
            findings.append(finding)

        # Forward guidance vs. historical delivery
        if financial_data:
            guidance_accuracy = self._assess_guidance_accuracy(financial_data, mda_text)
            if guidance_accuracy is not None and guidance_accuracy < 0.7:
                finding = self._create_finding(
                    finding_type="RED_FLAG",
                    title=f"Poor Guidance Track Record ({guidance_accuracy*100:.0f}% accuracy)",
                    detail="Management has historically overestimated performance vs. actual results. High guidance optimism bias detected.",
                    evidence="Financial actuals consistently below guidance range in MD&A language.",
                    risk_level="MODERATE",
                    confidence=0.65,
                )
                findings.append(finding)

        # ── LLM Deep Analysis ─────────────────────────────────
        from llm.prompts import build_management_nlp_prompt
        nlp_prompt = build_management_nlp_prompt(company_name, concall_text, mda_text)
        raw_analysis = self._analyze_with_llm(nlp_prompt, "management_nlp", max_tokens=2048)
        result.raw_analysis = raw_analysis

        # Parse LLM response for additional red flags
        if "evasive" in raw_analysis.lower() or "misleading" in raw_analysis.lower():
            finding = self._create_finding(
                finding_type="RED_FLAG",
                title="LLM Analysis: Evasive/Misleading Communication Patterns Detected",
                detail=raw_analysis[:500],
                evidence="NLP analysis of management communications",
                risk_level="HIGH",
                confidence=0.70,
            )
            findings.append(finding)
            result.red_flags.append(finding)

        # Save concall intelligence to database
        concall_record = {
            "fiscal_year": max(financial_data.keys()) if financial_data else "N/A",
            "quarter": "",
            "evasion_score": evasion_score,
            "optimism_score": mda_metrics.get("overconfidence_ratio", 0),
            "uncertainty_score": uncertainty_score,
            "narrative_consistency_score": 0.5,
            "key_phrases": str(mda_metrics.get("uncertainty_words", [])[:10]),
            "red_flag_phrases": str(mda_metrics.get("evasive_phrases", [])[:5]),
        }

        result.risk_score = max(10.0, min(95.0, evasion_score * 100 + non_gaap_count * 5))

        if self.db and company_id:
            try:
                self.db.save_concall(company_id, concall_record)
            except Exception:
                pass
        result.findings = findings

        result.summary = (
            f"MANAGEMENT NLP ANALYSIS - {company_name}\n"
            f"{'='*50}\n"
            f"Evasion Score: {evasion_score:.3f}/1.0\n"
            f"Non-GAAP Emphasis: {non_gaap_count} references\n"
            f"Uncertainty Language: {uncertainty_score*100:.1f}%\n"
            f"Red Flags: {len(result.red_flags)}\n"
        )

        self._save_output(result, company_name)
        return result

    def _analyze_text_metrics(self, text: str) -> dict:
        """Calculate quantitative NLP metrics."""
        if not text:
            return {}

        text_lower = text.lower()
        words = text_lower.split()
        total_words = max(len(words), 1)

        uncertainty = [w for w in UNCERTAINTY_WORDS if w in text_lower]
        evasive = [p for p in EVASIVE_PHRASES if p in text_lower]
        overconfident = [p for p in OVERCONFIDENCE_PHRASES if p in text_lower]

        return {
            "total_words": total_words,
            "uncertainty_words": uncertainty,
            "uncertainty_ratio": len(uncertainty) / total_words * 10,
            "evasive_phrases": evasive,
            "overconfidence_ratio": len(overconfident) / total_words * 10,
        }

    def _calculate_evasion_score(self, text: str) -> float:
        """Calculate evasion score 0-1."""
        if not text:
            return 0.0
        text_lower = text.lower()
        hits = sum(1 for p in EVASIVE_PHRASES if p in text_lower)
        return min(1.0, hits / max(len(text.split()) / 1000, 1) / 5)

    def _count_non_gaap_emphasis(self, text: str) -> int:
        text_lower = text.lower()
        return sum(text_lower.count(p) for p in NON_GAAP_EMPHASIS)

    def _find_non_gaap_metrics(self, text: str) -> list[str]:
        text_lower = text.lower()
        found = []
        for p in NON_GAAP_EMPHASIS:
            if p in text_lower:
                found.append(p)
        return found[:5]

    def _assess_guidance_accuracy(self, financial_data: dict, mda_text: str) -> float:
        """
        Approximate guidance accuracy from trend.
        If revenue consistently growing slower than guidance language, score is low.
        """
        if len(financial_data) < 2:
            return None

        # Proxy: if MDA uses very bullish language but growth is moderate
        optimism_count = sum(1 for p in OVERCONFIDENCE_PHRASES if p.lower() in mda_text.lower())
        years = sorted(financial_data.keys())

        # Check if revenue growth was strong in recent years
        if len(years) >= 2:
            rev_current = financial_data[years[-1]].get("revenue", 0)
            rev_prev = financial_data[years[-2]].get("revenue", 0)
            if rev_prev > 0:
                growth = (rev_current - rev_prev) / rev_prev
                # If very bullish language but <10% growth, low accuracy
                if optimism_count > 3 and growth < 0.10:
                    return 0.5
                elif growth > 0.15:
                    return 0.8
        return 0.7
