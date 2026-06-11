"""
Agent 4 — Cash Flow Forensics Agent
=====================================
Investigates CFO quality, FCF sustainability, CapEx patterns,
and cash-to-earnings divergence using quantitative engines.
"""

from __future__ import annotations
from utils.helpers import safe_divide
from forensics.accrual_analysis import AccrualAnalyzer
from .base_agent import BaseForensicAgent, AgentResult, AgentFinding


class CashFlowForensicsAgent(BaseForensicAgent):
    """
    Deep cash flow forensics: FCF quality, CFO vs earnings divergence,
    CapEx sustainability, and hidden working capital cash drains.
    """

    def investigate(self, company_name: str, company_id: int, financial_data: dict, **kwargs) -> AgentResult:
        self.log_info(f"Cash flow forensics for {company_name}")
        result = AgentResult(agent_id=self.agent_id, agent_name=self.agent_name)

        if not financial_data:
            result.status = "PARTIAL"
            result.error = "No financial data"
            return result

        years = sorted(financial_data.keys(), reverse=True)
        result.data_quality = self._check_data_quality(financial_data)

        cf_by_year: dict[str, dict] = {}
        for i, year in enumerate(years[:5]):
            d = financial_data[year]
            prev = financial_data.get(years[i + 1]) if i + 1 < len(years) else {}
            cf_by_year[year] = self._compute_cf_metrics(d, prev)

        self._generate_findings(result, cf_by_year, years, company_name, financial_data)

        # LLM deep analysis
        context = self._retrieve_context(
            company_name,
            "cash flow from operations capital expenditure free cash flow dividend payout"
        )
        prompt = self._build_prompt(company_name, cf_by_year, years, context)
        result.raw_analysis = self._analyze_with_llm(prompt, "forensic_accountant")

        result.summary = self._build_summary(company_name, cf_by_year, years, result)
        self._save_output(result, company_name)
        self.log_info(f"Cash Flow Forensics complete. Risk={result.risk_score:.1f}/100")
        return result

    # ─── Quantitative Computations ────────────────────────────────

    def _compute_cf_metrics(self, d: dict, prev: dict) -> dict:
        cfo = d.get("cfo", 0) or 0
        capex = abs(d.get("capex", 0) or 0)
        net_income = d.get("net_income", 0) or 0
        revenue = d.get("revenue", 1) or 1
        ebitda = d.get("ebitda", 0) or (net_income * 1.5 if net_income > 0 else 1)
        ebit = d.get("ebit", 0) or net_income

        fcf = cfo - capex
        cfo_margin = safe_divide(cfo, revenue) * 100
        fcf_margin = safe_divide(fcf, revenue) * 100
        cfo_to_ni = safe_divide(cfo, net_income) if net_income else None
        cfo_to_ebitda = safe_divide(cfo, ebitda) if ebitda else None
        cfo_to_ebit = safe_divide(cfo, ebit) if ebit else None
        capex_to_revenue = safe_divide(capex, revenue) * 100
        capex_to_depreciation = safe_divide(capex, d.get("depreciation", 1) or 1)

        # CapEx intensity classification
        if capex_to_revenue < 2:
            capex_intensity = "ASSET_LIGHT"
        elif capex_to_revenue < 7:
            capex_intensity = "MODERATE"
        else:
            capex_intensity = "CAPITAL_INTENSIVE"

        return {
            "cfo": cfo,
            "capex": capex,
            "fcf": fcf,
            "net_income": net_income,
            "revenue": revenue,
            "cfo_margin_pct": round(cfo_margin, 2),
            "fcf_margin_pct": round(fcf_margin, 2),
            "cfo_to_ni": round(cfo_to_ni, 3) if cfo_to_ni is not None else None,
            "cfo_to_ebitda": round(cfo_to_ebitda, 3) if cfo_to_ebitda is not None else None,
            "cfo_to_ebit": round(cfo_to_ebit, 3) if cfo_to_ebit is not None else None,
            "capex_to_revenue_pct": round(capex_to_revenue, 2),
            "capex_to_depreciation": round(capex_to_depreciation, 2),
            "capex_intensity": capex_intensity,
        }

    def _generate_findings(
        self, result: AgentResult, cf_by_year: dict, years: list, company_name: str, financial_data: dict
    ) -> None:
        latest_year = years[0]
        m = cf_by_year.get(latest_year, {})

        score_contributors = []

        # ── CFO vs Net Income divergence ──────────────────────────
        cfo_to_ni = m.get("cfo_to_ni")
        if cfo_to_ni is not None:
            if cfo_to_ni < 0 and (m.get("net_income", 0) or 0) > 0:
                f = self._create_finding(
                    "RED_FLAG",
                    "CRITICAL: Positive Net Income with Negative CFO",
                    "Earnings are not backed by cash. Classic sign of accrual-based earnings manipulation.",
                    f"FY{latest_year}: CFO = {m['cfo']/1e6:.1f}M, Net Income = {m['net_income']/1e6:.1f}M. "
                    f"CFO/NI = {cfo_to_ni:.2f}x",
                    fiscal_year=latest_year, risk_level="CRITICAL", confidence=0.90,
                )
                result.red_flags.append(f); result.findings.append(f)
                score_contributors.append(90)
            elif cfo_to_ni < 0.70:
                f = self._create_finding(
                    "RED_FLAG",
                    f"Weak Cash Earnings Ratio: {cfo_to_ni:.2f}x (threshold: 0.70x)",
                    "Only a fraction of reported profits are converting to cash. Elevated accruals.",
                    f"CFO/Net Income = {cfo_to_ni:.2f}x. Benchmark: >1.0x (high quality), 0.70–1.0x (acceptable).",
                    fiscal_year=latest_year, risk_level="HIGH", confidence=0.85,
                )
                result.red_flags.append(f); result.findings.append(f)
                score_contributors.append(70)
            elif cfo_to_ni > 1.2:
                f = self._create_finding(
                    "GREEN_FLAG",
                    f"Strong Cash Earnings Ratio: {cfo_to_ni:.2f}x",
                    "Earnings well-backed by cash. High earnings quality signal.",
                    f"CFO/Net Income = {cfo_to_ni:.2f}x. Cash earnings exceed reported earnings.",
                    fiscal_year=latest_year, risk_level="POSITIVE", confidence=0.85,
                )
                result.green_flags.append(f); result.findings.append(f)
                score_contributors.append(20)

        # ── CFO / EBITDA conversion ───────────────────────────────
        cfo_to_ebitda = m.get("cfo_to_ebitda")
        if cfo_to_ebitda is not None and cfo_to_ebitda < 0.70:
            f = self._create_finding(
                "RED_FLAG",
                f"Poor EBITDA-to-Cash Conversion: {cfo_to_ebitda:.2f}x",
                "EBITDA is not converting to operating cash flows. Working capital absorption or off-balance-sheet adjustments.",
                f"CFO/EBITDA = {cfo_to_ebitda:.2f}x (threshold: >0.85x). Possible causes: rising receivables, "
                f"deferred revenue reversal, or aggressive depreciation assumptions.",
                fiscal_year=latest_year, risk_level="HIGH", confidence=0.80,
            )
            result.red_flags.append(f); result.findings.append(f)
            score_contributors.append(65)

        # ── FCF negativity ─────────────────────────────────────────
        fcf = m.get("fcf", 0)
        if fcf < 0:
            f = self._create_finding(
                "RED_FLAG",
                f"Negative Free Cash Flow: {fcf/1e6:.1f}M",
                "Company consuming more cash than it generates from operations after capex. Sustainability risk.",
                f"FCF = CFO ({m['cfo']/1e6:.1f}M) - CapEx ({m['capex']/1e6:.1f}M) = {fcf/1e6:.1f}M",
                fiscal_year=latest_year, risk_level="HIGH", confidence=0.88,
            )
            result.red_flags.append(f); result.findings.append(f)
            score_contributors.append(65)

        # ── Multi-year CFO trend ───────────────────────────────────
        cfo_trend = [(y, cf_by_year[y]["cfo"]) for y in sorted(cf_by_year.keys())]
        if len(cfo_trend) >= 3:
            cfo_values = [v for _, v in cfo_trend]
            # Check if CFO is consistently declining while NI is stable/rising
            ni_trend = [(y, cf_by_year[y]["net_income"]) for y in sorted(cf_by_year.keys())]
            if len(ni_trend) >= 3:
                cfo_delta = cfo_values[-1] - cfo_values[0]
                ni_values = [v for _, v in ni_trend]
                ni_delta = ni_values[-1] - ni_values[0]
                # CFO declining but NI growing = serious red flag
                if cfo_delta < 0 and ni_delta > 0:
                    f = self._create_finding(
                        "RED_FLAG",
                        "CFO Declining While Net Income Rising — Earnings-Cash Divergence",
                        "Multi-year trend: reported profits growing but cash from operations shrinking. Classic manipulation pattern.",
                        f"CFO trend: {', '.join([f'{y}:{v/1e6:.0f}M' for y, v in cfo_trend])}. "
                        f"NI trend: {', '.join([f'{y}:{v/1e6:.0f}M' for y, v in ni_trend])}",
                        fiscal_year=sorted(cf_by_year.keys())[-1],
                        risk_level="CRITICAL", confidence=0.88,
                    )
                    result.red_flags.append(f); result.findings.append(f)
                    score_contributors.append(88)

        # ── CapEx sustainability ───────────────────────────────────
        capex_to_dep = m.get("capex_to_depreciation", 1)
        if capex_to_dep < 0.5:
            f = self._create_finding(
                "RED_FLAG",
                f"CapEx Below Depreciation: {capex_to_dep:.2f}x — Asset Under-Investment",
                "Company spending far less on capex than it is depreciating. Assets ageing. Future growth at risk.",
                f"CapEx/Depreciation = {capex_to_dep:.2f}x. Ratio below 1.0x indicates under-investment.",
                fiscal_year=latest_year, risk_level="MEDIUM", confidence=0.75,
            )
            result.red_flags.append(f); result.findings.append(f)
            score_contributors.append(50)

        result.risk_score = (sum(score_contributors) / len(score_contributors)) if score_contributors else 35.0
        result.risk_score = max(10.0, min(95.0, result.risk_score))

    def _build_prompt(self, company_name: str, cf_by_year: dict, years: list, context: str) -> str:
        latest = cf_by_year.get(years[0], {})
        prompt = (
            f"You are a forensic cash flow analyst investigating {company_name}.\n\n"
            f"CASH FLOW METRICS (Latest Year: FY{years[0]}):\n"
            f"  CFO: {latest.get('cfo', 0)/1e6:.1f}M\n"
            f"  FCF: {latest.get('fcf', 0)/1e6:.1f}M\n"
            f"  CFO/Net Income: {latest.get('cfo_to_ni', 'N/A')}\n"
            f"  CFO/EBITDA: {latest.get('cfo_to_ebitda', 'N/A')}\n"
            f"  CFO Margin: {latest.get('cfo_margin_pct', 0):.1f}%\n"
            f"  CapEx/Revenue: {latest.get('capex_to_revenue_pct', 0):.1f}%\n"
            f"  CapEx/Depreciation: {latest.get('capex_to_depreciation', 0):.2f}x\n"
            f"  CapEx Intensity: {latest.get('capex_intensity', 'N/A')}\n\n"
        )
        if len(years) > 1:
            prompt += "MULTI-YEAR CFO TREND:\n"
            for y in sorted(cf_by_year.keys(), reverse=True)[:4]:
                cf = cf_by_year[y]
                prompt += f"  FY{y}: CFO={cf.get('cfo', 0)/1e6:.0f}M, FCF={cf.get('fcf', 0)/1e6:.0f}M, CFO/NI={cf.get('cfo_to_ni', 'N/A')}\n"
        if context:
            prompt += f"\nDOCUMENT CONTEXT:\n{context[:2000]}\n"
        prompt += (
            "\nINSTRUCTIONS: Provide an institutional forensic assessment of cash flow quality. "
            "Identify any red flags. Follow Evidence → Analysis → Reasoning → Conclusion."
        )
        return prompt

    def _build_summary(self, company_name: str, cf_by_year: dict, years: list, result: AgentResult) -> str:
        latest = cf_by_year.get(years[0], {})
        return (
            f"CASH FLOW FORENSICS — {company_name} FY{years[0]}\n"
            f"CFO: {latest.get('cfo', 0)/1e6:.1f}M | FCF: {latest.get('fcf', 0)/1e6:.1f}M\n"
            f"CFO/NI: {latest.get('cfo_to_ni', 'N/A')}x | CFO/EBITDA: {latest.get('cfo_to_ebitda', 'N/A')}x\n"
            f"CapEx Intensity: {latest.get('capex_intensity', 'N/A')}\n"
            f"Risk Score: {result.risk_score:.1f}/100 | Red Flags: {len(result.red_flags)}"
        )
