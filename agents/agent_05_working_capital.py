"""
Agent 5 — Working Capital Forensics Agent
==========================================
Full DSO / DIO / DPO / CCC analysis with multi-year trend detection.
Flags: channel stuffing, inventory inflation, payable stretching.
"""

from __future__ import annotations
from utils.helpers import safe_divide
from forensics.working_capital_analysis import WorkingCapitalAnalyzer, WorkingCapitalResult
from .base_agent import BaseForensicAgent, AgentResult


class WorkingCapitalAgent(BaseForensicAgent):
    """
    Forensic working capital analysis using the WorkingCapitalAnalyzer engine.
    Detects DSO spikes (channel stuffing), inventory build-ups, and DPO stretching.
    """

    def investigate(self, company_name: str, company_id: int, financial_data: dict, **kwargs) -> AgentResult:
        self.log_info(f"Working capital forensics for {company_name}")
        result = AgentResult(agent_id=self.agent_id, agent_name=self.agent_name)

        if not financial_data:
            result.status = "PARTIAL"
            result.error = "No financial data"
            return result

        years = sorted(financial_data.keys(), reverse=True)
        result.data_quality = self._check_data_quality(financial_data)

        wc_by_year: dict[str, WorkingCapitalResult] = {}
        analyzer = WorkingCapitalAnalyzer()

        for i, year in enumerate(years[:5]):
            d = financial_data[year]
            prev = financial_data.get(years[i + 1]) if i + 1 < len(years) else {}
            wc = analyzer.calculate(
                accounts_receivable=d.get("accounts_receivable", 0) or 0,
                inventory=d.get("inventory", 0) or 0,
                accounts_payable=d.get("accounts_payable", 0) or 0,
                revenue=d.get("revenue", 1) or 1,
                cogs=d.get("cogs", 0) or 0,
                accounts_receivable_prev=prev.get("accounts_receivable", 0) or 0,
                inventory_prev=prev.get("inventory", 0) or 0,
                accounts_payable_prev=prev.get("accounts_payable", 0) or 0,
                revenue_prev=prev.get("revenue", 1) or 1,
                cogs_prev=prev.get("cogs", 0) or 0,
            )
            wc_by_year[year] = wc

        self._generate_findings(result, wc_by_year, years, company_name)

        rag_result = self._run_agentic_rag(
            company_name,
            "Working capital forensics: DSO trends and channel stuffing signals, "
            "inventory build-up and overstatement, DPO stretching under liquidity stress, "
            "and cash conversion cycle deterioration.",
            financial_data,
        )
        result.raw_analysis = rag_result.raw_text

        result.summary = self._build_summary(company_name, wc_by_year, years, result)
        self._save_output(result, company_name)
        self.log_info(f"Working Capital complete. Risk={result.risk_score:.1f}/100")
        return result

    def _generate_findings(
        self, result: AgentResult, wc_by_year: dict, years: list, company_name: str
    ) -> None:
        latest_year = years[0]
        wc = wc_by_year.get(latest_year)
        if not wc:
            return

        score_contributors = []

        # ── Convert engine flags to AgentFindings ─────────────────
        for flag in wc.flags:
            severity = flag.get("severity", "MODERATE")
            flag_type = flag.get("type", "")
            risk_level = "CRITICAL" if severity == "CRITICAL" else ("HIGH" if severity == "HIGH" else "MEDIUM")
            confidence = 0.90 if severity == "CRITICAL" else 0.82

            title_map = {
                "DSO_SPIKE": f"DSO Spike: +{flag.get('value', 0):.0f} days YoY — Channel Stuffing Risk",
                "DSO_RISING": f"DSO Gradually Rising: +{flag.get('value', 0):.0f} days — Collection Deterioration",
                "INVENTORY_INFLATION": f"Inventory Building Faster than Revenue — Possible Overstatement",
                "PAYABLE_STRETCHING": f"DPO Extended +{flag.get('value', 0):.0f} days — Liquidity Stress Indicator",
                "CCC_DETERIORATING": f"CCC Worsened +{flag.get('value', 0):.0f} days — Working Capital Drain",
            }
            title = title_map.get(flag_type, flag.get("type", "Working Capital Flag"))

            f = self._create_finding(
                "RED_FLAG",
                title,
                flag.get("message", ""),
                f"{flag.get('message', '')} | Full WC profile: {wc.interpretation}",
                fiscal_year=latest_year,
                risk_level=risk_level,
                confidence=confidence,
            )
            result.red_flags.append(f)
            result.findings.append(f)
            score_contributors.append(90 if severity == "CRITICAL" else 70 if severity == "HIGH" else 50)

        # ── Absolute DSO level ────────────────────────────────────
        if wc.dso > 120:
            f = self._create_finding(
                "RED_FLAG",
                f"Extremely High DSO: {wc.dso:.0f} days",
                "DSO above 120 days is a serious collection quality concern. Suggests weak credit controls or fictitious receivables.",
                f"DSO = {wc.dso:.0f} days ({wc.dso_change:+.0f} days YoY)",
                fiscal_year=latest_year, risk_level="HIGH", confidence=0.85,
            )
            result.red_flags.append(f); result.findings.append(f)
            score_contributors.append(72)

        # ── Multi-year deterioration ───────────────────────────────
        dso_trend = [(y, wc_by_year[y].dso) for y in sorted(wc_by_year.keys())]
        if len(dso_trend) >= 3:
            dso_values = [v for _, v in dso_trend]
            if all(dso_values[i] < dso_values[i + 1] for i in range(len(dso_values) - 1)):
                total_dso_rise = dso_values[-1] - dso_values[0]
                if total_dso_rise > 20:
                    f = self._create_finding(
                        "RED_FLAG",
                        f"Multi-Year DSO Deterioration: +{total_dso_rise:.0f} days over {len(dso_trend)} years",
                        "Sustained receivable build-up over multiple years is a systemic revenue quality issue.",
                        f"DSO trend: {', '.join([f'{y}:{v:.0f}d' for y, v in dso_trend])}",
                        fiscal_year=dso_trend[-1][0],
                        risk_level="HIGH", confidence=0.88,
                    )
                    result.red_flags.append(f); result.findings.append(f)
                    score_contributors.append(75)

        # ── Good working capital (green flag) ─────────────────────
        if wc.risk_level == "LOW" and not result.red_flags:
            f = self._create_finding(
                "GREEN_FLAG",
                f"Healthy Working Capital Cycle: CCC = {wc.ccc:.0f} days",
                f"DSO, DIO, and DPO all within normal ranges. No working capital red flags.",
                wc.interpretation,
                fiscal_year=latest_year, risk_level="POSITIVE", confidence=0.80,
            )
            result.green_flags.append(f); result.findings.append(f)
            score_contributors.append(20)

        if score_contributors:
            result.risk_score = max(10.0, min(95.0, sum(score_contributors) / len(score_contributors)))
        else:
            result.risk_score = 30.0

    def _build_prompt(self, company_name: str, wc_by_year: dict, years: list, context: str) -> str:
        latest = wc_by_year.get(years[0])
        if not latest:
            return f"Analyze working capital for {company_name}."

        prompt = (
            f"Forensic working capital analysis for {company_name}.\n\n"
            f"WORKING CAPITAL METRICS (FY{years[0]}):\n"
            f"  DSO: {latest.dso:.1f} days (Δ{latest.dso_change:+.1f} days YoY)\n"
            f"  DIO: {latest.dio:.1f} days (Δ{latest.dio_change:+.1f} days YoY)\n"
            f"  DPO: {latest.dpo:.1f} days (Δ{latest.dpo_change:+.1f} days YoY)\n"
            f"  CCC: {latest.ccc:.1f} days (Δ{latest.ccc_change:+.1f} days YoY)\n"
            f"  AR Turnover: {latest.receivable_turnover:.1f}x\n"
            f"  Inventory Turnover: {latest.inventory_turnover:.1f}x\n"
            f"  Risk Level: {latest.risk_level}\n\n"
        )
        if len(years) > 1:
            prompt += "MULTI-YEAR TRENDS:\n"
            for y in sorted(wc_by_year.keys(), reverse=True)[:4]:
                w = wc_by_year[y]
                prompt += f"  FY{y}: DSO={w.dso:.0f}d, DIO={w.dio:.0f}d, DPO={w.dpo:.0f}d, CCC={w.ccc:.0f}d\n"
        if context:
            prompt += f"\nDOCUMENT CONTEXT:\n{context[:1500]}\n"
        prompt += (
            "\nProvide an institutional forensic assessment. Flag any channel stuffing, "
            "inventory manipulation, or liquidity stress indicators. "
            "Evidence → Analysis → Reasoning → Conclusion."
        )
        return prompt

    def _build_summary(self, company_name: str, wc_by_year: dict, years: list, result: AgentResult) -> str:
        latest = wc_by_year.get(years[0])
        if not latest:
            return f"Working Capital Agent: Insufficient data for {company_name}"
        return (
            f"WORKING CAPITAL FORENSICS — {company_name} FY{years[0]}\n"
            f"DSO: {latest.dso:.0f}d (Δ{latest.dso_change:+.0f}d) | "
            f"DIO: {latest.dio:.0f}d (Δ{latest.dio_change:+.0f}d) | "
            f"DPO: {latest.dpo:.0f}d (Δ{latest.dpo_change:+.0f}d) | "
            f"CCC: {latest.ccc:.0f}d (Δ{latest.ccc_change:+.0f}d)\n"
            f"WC Risk Level: {latest.risk_level} | Agent Risk Score: {result.risk_score:.1f}/100 | "
            f"Red Flags: {len(result.red_flags)}"
        )
