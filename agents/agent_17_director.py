"""
Agent 17 - Chief Investigation Director
=========================================
Aggregates all agent findings, resolves conflicts, generates final verdict.
This is the investment committee chairperson.
"""

from __future__ import annotations
import re
from .base_agent import BaseForensicAgent, AgentResult, AgentFinding
from forensics.risk_scorer import RiskScorer, RiskComponents
from utils.helpers import classify_risk
from config import AGENT_NAMES


class ChiefInvestigationDirector(BaseForensicAgent):
    """
    Synthesizes findings from all 16 specialist agents.
    Produces the final investment committee verdict.
    """

    VERDICT_OPTIONS = ["STRONG AVOID", "AVOID", "CAUTION", "MONITOR", "CAUTIOUS BUY", "BUY"]

    def investigate(
        self,
        company_name: str,
        company_id: int,
        financial_data: dict,
        all_agent_results: dict = None,
        **kwargs,
    ) -> AgentResult:
        self.log_info(f"Chief Director: Synthesizing investigation for {company_name}")
        result = AgentResult(agent_id=self.agent_id, agent_name=self.agent_name)

        if not all_agent_results:
            result.status = "FAILED"
            result.error = "No agent results provided to Director"
            return result

        # ── Aggregate All Findings ─────────────────────────────
        all_red_flags = []
        all_green_flags = []
        all_findings = []
        agent_risk_scores = {}

        for agent_id, agent_result in all_agent_results.items():
            if not isinstance(agent_result, AgentResult):
                continue
            all_red_flags.extend(agent_result.red_flags)
            all_green_flags.extend(agent_result.green_flags)
            all_findings.extend(agent_result.findings)
            agent_risk_scores[agent_id] = {
                "name": agent_result.agent_name,
                "score": agent_result.risk_score,
                "red_flags": len(agent_result.red_flags),
                "green_flags": len(agent_result.green_flags),
            }

        # ── Calculate Composite Risk Score ────────────────────
        fraud_agent = all_agent_results.get(6)
        earnings_agent = all_agent_results.get(8)
        cashflow_agent = all_agent_results.get(4)
        governance_agent = all_agent_results.get(9)
        credit_agent = all_agent_results.get(7)
        auditor_agent = all_agent_results.get(10)
        mgmt_agent = all_agent_results.get(11)

        components = RiskComponents(
            fraud_indicators=getattr(fraud_agent, "risk_score", 50.0) if fraud_agent else 50.0,
            earnings_quality=getattr(earnings_agent, "risk_score", 50.0) if earnings_agent else 50.0,
            cash_flow_quality=getattr(cashflow_agent, "risk_score", 50.0) if cashflow_agent else 50.0,
            governance=governance_agent.risk_score if governance_agent else self._estimate_governance_score(all_red_flags),
            credit_risk=getattr(credit_agent, "risk_score", 50.0) if credit_agent else 50.0,
            auditor_risk=getattr(auditor_agent, "risk_score", 50.0) if auditor_agent else 50.0,
            management_credibility=getattr(mgmt_agent, "risk_score", 50.0) if mgmt_agent else 50.0,
        )
        risk_result = RiskScorer().calculate(components)
        result.risk_score = risk_result.overall_score

        # ── Investment Verdict ─────────────────────────────────
        verdict, verdict_rationale = self._determine_verdict(
            risk_result.overall_score, all_red_flags, all_green_flags
        )

        # ── Investment Committee Perspectives ──────────────────
        # Agents 14/15/16 were merged into a single result[14].
        # Extract each perspective section from the combined raw_analysis.
        perspectives_result = all_agent_results.get(14)
        perspectives_text = getattr(perspectives_result, "raw_analysis", "") if perspectives_result else ""

        def _extract_perspective(text: str, header: str) -> str:
            m = re.search(
                rf"=== {re.escape(header)} ===\n(.*?)(?====|$)", text, re.DOTALL
            )
            return m.group(1).strip()[:600] if m else "Not available"

        bear_case_summary = _extract_perspective(perspectives_text, "BEAR CASE (Short Seller)")
        bull_case_summary = _extract_perspective(perspectives_text, "BULL CASE")
        devil_summary     = _extract_perspective(perspectives_text, "DEVIL'S ADVOCATE")

        # ── Final LLM Synthesis ────────────────────────────────
        synthesis_prompt = self._build_synthesis_prompt(
            company_name=company_name,
            overall_score=risk_result.overall_score,
            risk_band=risk_result.band,
            component_scores=risk_result.component_scores,
            critical_flags=[f for f in all_red_flags if f.risk_level == "CRITICAL"],
            high_flags=[f for f in all_red_flags if f.risk_level == "HIGH"],
            green_flags=all_green_flags,
            verdict=verdict,
            bear_case=bear_case_summary,
            bull_case=bull_case_summary,
            devil_advocate=devil_summary,
            financial_data=financial_data,
        )
        final_analysis = self._analyze_with_llm(synthesis_prompt, "investment_director", max_tokens=4096)
        result.raw_analysis = final_analysis

        # ── Compile Director Findings ──────────────────────────
        # Critical finding: overall verdict
        verdict_finding = self._create_finding(
            finding_type="OBSERVATION",
            title=f"INVESTMENT COMMITTEE VERDICT: {verdict}",
            detail=verdict_rationale,
            evidence=f"Overall Risk Score: {risk_result.overall_score:.1f}/100 ({risk_result.band}). Total Red Flags: {len(all_red_flags)}. Total Green Flags: {len(all_green_flags)}.",
            risk_level=self._score_to_risk_level(risk_result.overall_score),
            confidence=0.85,
        )
        result.findings.append(verdict_finding)
        result.red_flags = all_red_flags[:20]  # Top 20 red flags
        result.green_flags = all_green_flags[:10]

        # ── Management Questionnaire ───────────────────────────
        mgmt_questions = self._generate_management_questions(all_red_flags, company_name)

        # ── Monitoring Framework ───────────────────────────────
        monitoring_triggers = self._generate_monitoring_triggers(all_red_flags, risk_result)

        # Save comprehensive output
        director_output = {
            "verdict": verdict,
            "overall_risk_score": risk_result.overall_score,
            "risk_band": risk_result.band,
            "component_scores": risk_result.component_scores,
            "key_drivers": risk_result.key_drivers,
            "total_red_flags": len(all_red_flags),
            "total_green_flags": len(all_green_flags),
            "critical_red_flags": [
                {"title": f.title, "evidence": f.evidence[:200]}
                for f in all_red_flags if f.risk_level == "CRITICAL"
            ],
            "management_questions": mgmt_questions,
            "monitoring_triggers": monitoring_triggers,
            "agent_scores": agent_risk_scores,
            "final_analysis": final_analysis,
        }
        self.storage.save_json(director_output, "director_final_output.json", "Agent_Outputs")

        result.summary = self._build_executive_summary(
            company_name, verdict, risk_result, all_red_flags, all_green_flags, financial_data
        )

        self.log_info(f"Investigation complete. Verdict: {verdict}. Score: {risk_result.overall_score:.1f}/100")
        return result

    def _determine_verdict(
        self, risk_score: float, red_flags: list, green_flags: list
    ) -> tuple[str, str]:
        critical_flags = [f for f in red_flags if f.risk_level == "CRITICAL"]
        high_flags = [f for f in red_flags if f.risk_level == "HIGH"]

        if risk_score > 80 or len(critical_flags) >= 3:
            return "STRONG AVOID", f"Extreme risk score ({risk_score:.1f}/100) with {len(critical_flags)} critical red flags."
        elif risk_score > 65 or len(critical_flags) >= 1:
            return "AVOID", f"High risk score ({risk_score:.1f}/100). {len(critical_flags)} critical and {len(high_flags)} high-severity issues."
        elif risk_score > 50 or len(high_flags) >= 3:
            return "CAUTION", f"Moderate-to-high risk ({risk_score:.1f}/100). Multiple high-severity concerns require resolution."
        elif risk_score > 40:
            return "MONITOR", f"Moderate risk ({risk_score:.1f}/100). Monitor key metrics quarterly."
        elif risk_score > 25:
            return "CAUTIOUS BUY", f"Below-moderate risk ({risk_score:.1f}/100) with {len(green_flags)} positive signals."
        else:
            return "BUY", f"Low risk score ({risk_score:.1f}/100). Strong financial quality indicators."

    def _estimate_governance_score(self, red_flags: list) -> float:
        governance_flags = [
            f for f in red_flags
            if any(kw in f.title.lower() for kw in ["promoter", "pledge", "related party", "board", "governance", "insider"])
        ]
        return min(100.0, 30.0 + len(governance_flags) * 15)

    def _score_to_risk_level(self, score: float) -> str:
        if score > 80:
            return "CRITICAL"
        elif score > 60:
            return "HIGH"
        elif score > 40:
            return "MODERATE"
        return "LOW"

    def _generate_management_questions(self, red_flags: list, company_name: str) -> list[str]:
        """Generate 50-100 management questions based on red flags."""
        questions = []

        # Standard forensic questions
        standard_questions = [
            # Revenue
            "Please explain the mechanism for revenue recognition for multi-year contracts. At what point is revenue recorded?",
            "What percentage of total revenue comes from related parties? Provide names and terms.",
            "What are the collection days for your largest 10 customers? Have any accounts gone bad in the last 3 years?",
            "Why have receivables grown faster than revenue for the past 2 years?",
            "Describe any sales arrangements involving right of return, consignment, or bill-and-hold.",
            # Cash Flow
            "Why has CFO consistently underperformed reported EBITDA? Break down the reconciliation items.",
            "What is your capex guidance for the next 3 years? What returns do you expect?",
            "Have any assets been sold and leased back in the past 3 years? Provide terms.",
            # Debt
            "What are the key financial covenants in your debt agreements? How much headroom do you have?",
            "Have any debt covenants been waived or renegotiated in the past 2 years?",
            "What is the maturity profile of your debt? How will you refinance maturities in the next 2 years?",
            # Related Parties
            "List all related party transactions in the last 5 years with amounts, terms, and justification.",
            "Are there any outstanding loans or guarantees to promoters or related entities?",
            "Have any assets been transferred to or from promoter entities? At what valuation?",
            # Governance
            "What is the methodology for determining executive compensation? Who approves it?",
            "How many independent directors have resigned in the last 3 years? Why?",
            "What percentage of promoter holdings are pledged? For what purpose?",
            # Auditor
            "Why did you change auditors? Was the previous auditor engagement partner changed?",
            "Describe all issues raised by auditors in management letters in the last 3 years.",
            "What are the Key Audit Matters and how are they resolved?",
        ]
        questions.extend(standard_questions)

        # Red flag-specific questions
        for flag in red_flags[:10]:
            if "receivable" in flag.title.lower() or "dso" in flag.title.lower():
                questions.append(f"RECEIVABLES: {flag.title} - Please explain the specific drivers and provide the aged receivables schedule.")
            if "inventory" in flag.title.lower():
                questions.append(f"INVENTORY: {flag.title} - Provide an inventory aging schedule and NRV assessment.")
            if "fraud" in flag.title.lower() or "manipulation" in flag.title.lower():
                questions.append(f"ACCOUNTING: {flag.title} - Please walk through your revenue recognition policy with specific examples.")

        return questions[:80]  # Cap at 80

    def _generate_monitoring_triggers(self, red_flags: list, risk_result) -> list[dict]:
        """Generate quarterly monitoring triggers."""
        triggers = [
            {"metric": "Receivable Days (DSO)", "threshold": "+15 days QoQ", "action": "Immediate investigation"},
            {"metric": "CFO/EBITDA Ratio", "threshold": "< 0.70", "action": "Review working capital"},
            {"metric": "Inventory Days", "threshold": "+20 days QoQ", "action": "Investigate channel stuffing"},
            {"metric": "Net Debt/EBITDA", "threshold": "> 3.5x", "action": "Credit risk review"},
            {"metric": "Promoter Pledging", "threshold": "> 50% of holdings", "action": "Governance escalation"},
            {"metric": "Auditor Change", "threshold": "Any change", "action": "Deep audit review"},
            {"metric": "Management Turnover", "threshold": "CFO/CEO change", "action": "Governance review"},
            {"metric": "Revenue Growth vs Peers", "threshold": ">20% faster than peers", "action": "Revenue quality check"},
        ]
        return triggers

    def _build_synthesis_prompt(self, company_name: str, overall_score: float, risk_band: str,
                                 component_scores: dict, critical_flags: list, high_flags: list,
                                 green_flags: list, verdict: str, bear_case: str, bull_case: str,
                                 financial_data: dict, devil_advocate: str = "") -> str:
        critical_str = "\n".join([f"• {f.title}: {f.evidence[:100]}" for f in critical_flags[:5]])
        high_str = "\n".join([f"• {f.title}" for f in high_flags[:5]])
        green_str = "\n".join([f"• {f.title}" for f in green_flags[:5]])

        return f"""
INVESTMENT COMMITTEE FINAL REVIEW - {company_name}
===================================================

OVERALL RISK SCORE: {overall_score:.1f}/100 ({risk_band})
RECOMMENDED VERDICT: {verdict}

COMPONENT RISK SCORES:
{chr(10).join([f'  • {k.replace("_", " ").title()}: {v:.1f}/100' for k, v in component_scores.items()])}

CRITICAL RED FLAGS:
{critical_str if critical_str else "None"}

HIGH SEVERITY RED FLAGS:
{high_str if high_str else "None"}

GREEN FLAGS (Positive Indicators):
{green_str if green_str else "None"}

BEAR CASE (Short Seller Analysis):
{bear_case}

BULL CASE:
{bull_case}

DEVIL'S ADVOCATE:
{devil_advocate if devil_advocate else "Not available"}

As the Chief Investment Officer, provide:
1. EXECUTIVE SUMMARY (500 words): Investment thesis in plain English
2. VERDICT RATIONALE: Why {verdict} is the appropriate recommendation
3. TOP 5 RISKS that could materially affect the investment thesis
4. TOP 3 UPSIDE SCENARIOS that could improve the verdict
5. KEY MONITORING METRICS: What to watch quarterly
6. COMPARABLE SITUATION: Most similar historical case (companies/incidents)
7. FINAL RECOMMENDATION: Confirm or revise the verdict with clear reasoning
"""

    def _build_executive_summary(self, company: str, verdict: str, risk_result, red_flags: list, green_flags: list, financial_data: dict) -> str:
        latest_year = max(financial_data.keys()) if financial_data else "N/A"
        latest = financial_data.get(latest_year, {})

        return f"""
FORENSIC AI - EXECUTIVE SUMMARY
{'='*60}
Company:         {company}
Investigation:   {latest_year}
Verdict:         {verdict}
Risk Score:      {risk_result.overall_score:.1f}/100 ({risk_result.band})

RISK DECOMPOSITION:
{'─'*40}
{''.join([f'  {k.replace("_", " ").title():.<30}{v:.1f}/100{chr(10)}' for k, v in risk_result.component_scores.items()])}

KEY METRICS (Latest Year: {latest_year}):
{'─'*40}
  Revenue:         {latest.get('revenue', 0):>15,.0f}
  Net Income:      {latest.get('net_income', 0):>15,.0f}
  CFO:             {latest.get('cfo', 0):>15,.0f}
  Total Debt:      {latest.get('total_debt', 0):>15,.0f}
  Total Assets:    {latest.get('total_assets', 0):>15,.0f}

FINDINGS:
{'─'*40}
  Total Red Flags:    {len(red_flags)}
  Critical:           {len([f for f in red_flags if f.risk_level == 'CRITICAL'])}
  High:               {len([f for f in red_flags if f.risk_level == 'HIGH'])}
  Green Flags:        {len(green_flags)}

TOP RED FLAGS:
{'─'*40}
{''.join([f'  ❌ [{f.risk_level}] {f.title[:60]}{chr(10)}' for f in sorted(red_flags, key=lambda x: x.risk_level)[:5]])}
TOP GREEN FLAGS:
{'─'*40}
{''.join([f'  ✅ {f.title[:60]}{chr(10)}' for f in green_flags[:3]])}

KEY RISK DRIVERS:
{'─'*40}
{''.join([f'  • {d["component"]}: {d["score"]:.1f}/100{chr(10)}' for d in risk_result.key_drivers])}
"""
