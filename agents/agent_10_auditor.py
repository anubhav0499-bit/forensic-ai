"""
Agent 10 — Auditor Intelligence Agent
========================================
Analyzes: auditor identity, tenure, key audit matters (KAMs),
going concern opinions, internal control weaknesses, restatements.
"""

from __future__ import annotations
import re
from utils.helpers import safe_divide
from .base_agent import BaseForensicAgent, AgentResult


_BIG_FOUR = {"deloitte", "pwc", "pricewaterhousecoopers", "ernst & young", "ey", "kpmg", "bdo"}
_GOING_CONCERN_PHRASES = [
    "going concern", "substantial doubt", "ability to continue",
    "material uncertainty", "going concern doubt",
]
_ICW_PHRASES = [
    "material weakness", "significant deficiency", "internal control over financial reporting",
    "icfr", "section 404", "deficiency in internal controls",
]
_RESTATEMENT_PHRASES = [
    "restatement", "restated", "prior period adjustment", "correction of error",
    "amended annual report",
]
_KAM_PHRASES = [
    "key audit matter", "critical audit matter", "significant risk",
    "emphasis of matter",
]


class AuditorIntelligenceAgent(BaseForensicAgent):
    """
    Forensic auditor analysis: independence, qualifications,
    key audit matters, going concern, and internal control weaknesses.
    """

    def investigate(self, company_name: str, company_id: int, financial_data: dict, **kwargs) -> AgentResult:
        self.log_info(f"Auditor intelligence analysis for {company_name}")
        result = AgentResult(agent_id=self.agent_id, agent_name=self.agent_name)

        years = sorted(financial_data.keys(), reverse=True) if financial_data else []
        latest_year = years[0] if years else "N/A"

        # Retrieve auditor-related document context
        auditor_context = self._retrieve_context(
            company_name,
            "auditor opinion going concern key audit matters internal control material weakness restatement"
        )
        audit_report_context = self._retrieve_context(
            company_name,
            "independent auditor report basis for opinion emphasis of matter"
        )
        combined_context = f"{auditor_context}\n\n{audit_report_context}"

        # ── Rule-based NLP analysis on retrieved text ─────────────
        signals = self._extract_audit_signals(combined_context)
        self._generate_findings(result, signals, latest_year, company_name)

        # ── Structural signals from financial data ────────────────
        if financial_data and len(years) >= 2:
            self._analyze_auditor_history(result, financial_data, years)

        # ── LLM deep analysis ─────────────────────────────────────
        prompt = self._build_prompt(company_name, signals, combined_context, latest_year)
        result.raw_analysis = self._analyze_with_llm(prompt, "auditor_intelligence")

        result.summary = self._build_summary(company_name, signals, result, latest_year)
        self._save_output(result, company_name)
        self.log_info(f"Auditor Intelligence complete. Risk={result.risk_score:.1f}/100")
        return result

    # ─── Signal Extraction ────────────────────────────────────────

    def _extract_audit_signals(self, text: str) -> dict:
        text_lower = text.lower()

        # Auditor name extraction
        auditor_name = "Unknown"
        big_four_found = None
        for firm in _BIG_FOUR:
            if firm in text_lower:
                auditor_name = firm.title()
                big_four_found = firm
                break

        # Pattern matching
        going_concern = any(phrase in text_lower for phrase in _GOING_CONCERN_PHRASES)
        material_weakness = any(phrase in text_lower for phrase in _ICW_PHRASES)
        restatement = any(phrase in text_lower for phrase in _RESTATEMENT_PHRASES)
        emphasis_of_matter = "emphasis of matter" in text_lower

        # KAM count
        kam_count = sum(text_lower.count(phrase) for phrase in _KAM_PHRASES[:2])
        kam_count = min(kam_count, 10)  # Cap at 10 to avoid double counting

        # Tenure estimation (look for "for X years" near auditor name)
        tenure_match = re.search(r"(\d+)\s+(?:years?|consecutive years?)\s+(?:as auditor|of audit)", text_lower)
        tenure_years = int(tenure_match.group(1)) if tenure_match else None

        # Opinion type
        qualified_opinion = any(kw in text_lower for kw in [
            "qualified opinion", "adverse opinion", "disclaimer of opinion",
            "except for", "subject to"
        ])
        clean_opinion = "unqualified opinion" in text_lower or "in our opinion" in text_lower

        return {
            "auditor_name": auditor_name,
            "is_big_four": big_four_found is not None,
            "going_concern_flag": going_concern,
            "material_weakness": material_weakness,
            "restatement_flag": restatement,
            "emphasis_of_matter": emphasis_of_matter,
            "kam_count": max(0, kam_count),
            "tenure_years": tenure_years,
            "qualified_opinion": qualified_opinion,
            "clean_opinion": clean_opinion,
            "text_length": len(text),
        }

    def _generate_findings(self, result: AgentResult, signals: dict, latest_year: str, company_name: str) -> None:
        score_contributors = []

        # ── Going Concern ─────────────────────────────────────────
        if signals["going_concern_flag"]:
            f = self._create_finding(
                "RED_FLAG",
                "CRITICAL: Going Concern Qualification Detected",
                "Auditors have raised substantial doubt about the company's ability to continue as a going concern. "
                "This is the most severe audit qualification possible.",
                "Going concern language detected in auditor's report. Suggests material uncertainty about survival.",
                fiscal_year=latest_year, risk_level="CRITICAL", confidence=0.90,
            )
            result.red_flags.append(f); result.findings.append(f)
            score_contributors.append(95)

        # ── Material Weakness ─────────────────────────────────────
        if signals["material_weakness"]:
            f = self._create_finding(
                "RED_FLAG",
                "Material Weakness in Internal Controls (ICFR)",
                "Material weakness in internal controls is a serious red flag. "
                "It means financial statements may contain material misstatements. "
                "Often precedes restatements.",
                "Material weakness or significant deficiency in ICFR detected in audit documentation.",
                fiscal_year=latest_year, risk_level="CRITICAL", confidence=0.88,
            )
            result.red_flags.append(f); result.findings.append(f)
            score_contributors.append(88)

        # ── Restatement ───────────────────────────────────────────
        if signals["restatement_flag"]:
            f = self._create_finding(
                "RED_FLAG",
                "Prior Period Restatement Detected",
                "Prior period restatements indicate previous financial statements were materially incorrect. "
                "Restatements are the single strongest predictor of future fraud in academic literature.",
                "Restatement language detected in filings. Previous financial statements were restated.",
                fiscal_year=latest_year, risk_level="CRITICAL", confidence=0.85,
            )
            result.red_flags.append(f); result.findings.append(f)
            score_contributors.append(90)

        # ── Qualified Opinion ─────────────────────────────────────
        if signals["qualified_opinion"]:
            f = self._create_finding(
                "RED_FLAG",
                "Qualified or Adverse Auditor Opinion",
                "A qualified opinion means auditors take exception to specific items in the financial statements. "
                "An adverse opinion means statements do not present fairly.",
                "Qualified/adverse/disclaimer language detected in auditor's report.",
                fiscal_year=latest_year, risk_level="HIGH", confidence=0.85,
            )
            result.red_flags.append(f); result.findings.append(f)
            score_contributors.append(80)

        # ── Non-Big 4 ─────────────────────────────────────────────
        if not signals["is_big_four"]:
            f = self._create_finding(
                "RED_FLAG",
                f"Non-Big 4 Auditor: {signals['auditor_name']}",
                "For a public company, a non-Big 4 auditor raises independence and competence concerns. "
                "Downgrading from Big 4 is a particularly strong red flag.",
                f"Auditor identified as: {signals['auditor_name']}. Not Deloitte/PwC/EY/KPMG.",
                fiscal_year=latest_year, risk_level="MEDIUM", confidence=0.70,
            )
            result.red_flags.append(f); result.findings.append(f)
            score_contributors.append(52)

        # ── Auditor Tenure ────────────────────────────────────────
        if signals["tenure_years"] is not None:
            if signals["tenure_years"] > 20:
                f = self._create_finding(
                    "RED_FLAG",
                    f"Excessive Auditor Tenure: {signals['tenure_years']} years",
                    "Auditor tenure exceeding 20 years raises serious independence questions. "
                    "Long-tenured auditors may become too close to management.",
                    f"Estimated auditor tenure: {signals['tenure_years']} years.",
                    fiscal_year=latest_year, risk_level="HIGH", confidence=0.70,
                )
                result.red_flags.append(f); result.findings.append(f)
                score_contributors.append(65)

        # ── High KAM Count ────────────────────────────────────────
        if signals["kam_count"] >= 5:
            f = self._create_finding(
                "RED_FLAG",
                f"Elevated Key Audit Matters Count: {signals['kam_count']} KAMs",
                "A high number of KAMs signals auditors found many areas of significant risk. "
                "Each KAM represents an area where auditors exercised significant judgment.",
                f"KAM count estimated at {signals['kam_count']}. Typical: 2-4 for a large company.",
                fiscal_year=latest_year, risk_level="MEDIUM", confidence=0.72,
            )
            result.red_flags.append(f); result.findings.append(f)
            score_contributors.append(55)

        # ── Green flag: clean Big 4 opinion ─────────────────────
        if signals["is_big_four"] and signals["clean_opinion"] and not signals["going_concern_flag"] and not signals["material_weakness"]:
            f = self._create_finding(
                "GREEN_FLAG",
                f"Clean Unqualified Opinion from Big 4 ({signals['auditor_name'].title()})",
                "Big 4 audit with unqualified opinion and no going concern, restatement, or material weakness flags.",
                f"Auditor: {signals['auditor_name'].title()} | Opinion: Unqualified | No material qualifications.",
                fiscal_year=latest_year, risk_level="POSITIVE", confidence=0.80,
            )
            result.green_flags.append(f); result.findings.append(f)
            score_contributors.append(18)

        result.risk_score = (sum(score_contributors) / len(score_contributors)) if score_contributors else 40.0
        result.risk_score = max(10.0, min(95.0, result.risk_score))

    def _analyze_auditor_history(self, result: AgentResult, financial_data: dict, years: list) -> None:
        """Detect auditor change signals from financial data anomalies around year boundaries."""
        # If financial data has huge jumps after a certain year, could indicate restatement
        # Check for negative retained earnings appearing suddenly
        for i in range(1, min(3, len(years))):
            curr = financial_data[years[i - 1]]
            prev = financial_data[years[i]]
            re_curr = curr.get("retained_earnings", 0) or 0
            re_prev = prev.get("retained_earnings", 0) or 0

            if re_prev > 0 and re_curr < 0:
                f = self._create_finding(
                    "RED_FLAG",
                    f"Retained Earnings Turned Negative FY{years[i-1]} — Possible Write-Down or Restatement",
                    "Retained earnings moving from positive to negative in one year indicates a massive loss, "
                    "impairment, or prior period restatement.",
                    f"Retained Earnings: FY{years[i]}={re_prev/1e6:.0f}M → FY{years[i-1]}={re_curr/1e6:.0f}M",
                    fiscal_year=years[i - 1], risk_level="CRITICAL", confidence=0.88,
                )
                result.red_flags.append(f); result.findings.append(f)

    def _build_prompt(self, company_name: str, signals: dict, context: str, latest_year: str) -> str:
        prompt = (
            f"Auditor intelligence investigation for {company_name} (FY{latest_year}).\n\n"
            f"AUDIT SIGNALS DETECTED:\n"
            f"  Auditor: {signals['auditor_name']} (Big 4: {signals['is_big_four']})\n"
            f"  Going Concern: {signals['going_concern_flag']}\n"
            f"  Material Weakness: {signals['material_weakness']}\n"
            f"  Restatement: {signals['restatement_flag']}\n"
            f"  Qualified Opinion: {signals['qualified_opinion']}\n"
            f"  Emphasis of Matter: {signals['emphasis_of_matter']}\n"
            f"  KAM Count (est.): {signals['kam_count']}\n"
            f"  Auditor Tenure (est.): {signals['tenure_years']} years\n\n"
        )
        if context and len(context) > 100:
            prompt += f"AUDITOR REPORT EXCERPTS:\n{context[:2500]}\n\n"
        prompt += (
            "Assess the quality of audit oversight, independence risks, and what the audit qualifications "
            "signal about management integrity. Identify specific red flags. "
            "Evidence → Analysis → Reasoning → Conclusion."
        )
        return prompt

    def _build_summary(self, company_name: str, signals: dict, result: AgentResult, latest_year: str) -> str:
        flags = []
        if signals["going_concern_flag"]: flags.append("GOING CONCERN")
        if signals["material_weakness"]: flags.append("MATERIAL WEAKNESS")
        if signals["restatement_flag"]: flags.append("RESTATEMENT")
        if signals["qualified_opinion"]: flags.append("QUALIFIED OPINION")
        flags_str = ", ".join(flags) if flags else "None"
        return (
            f"AUDITOR INTELLIGENCE — {company_name} FY{latest_year}\n"
            f"Auditor: {signals['auditor_name']} | Big 4: {signals['is_big_four']}\n"
            f"Critical Flags: {flags_str}\n"
            f"KAMs: {signals['kam_count']} | Risk Score: {result.risk_score:.1f}/100"
        )
