"""
Agent 8 — Earnings Quality Agent
==================================
Evaluates earnings sustainability: accruals, tax rate consistency,
non-GAAP gap analysis, one-time item patterns, and revenue recognition.
"""

from __future__ import annotations
from utils.helpers import safe_divide
from forensics.accrual_analysis import AccrualAnalyzer, AccrualResult
from .base_agent import BaseForensicAgent, AgentResult


class EarningsQualityAgent(BaseForensicAgent):
    """
    Multi-dimension earnings quality assessment:
    - Sloan accrual decomposition
    - Tax rate consistency (ETR vs. statutory)
    - Non-GAAP vs. GAAP gap tracking
    - Recurring vs. one-time income ratio
    - Gross margin sustainability
    """

    # Approximate statutory tax rates by region
    _STATUTORY_RATES = {"India": 0.25, "US": 0.21, "UK": 0.25, "default": 0.25}
    _GAAP_NONRECURRING_KEYWORDS = [
        "exceptional", "extraordinary", "one-time", "non-recurring",
        "restructuring", "impairment", "write-off", "gain on sale",
    ]

    def investigate(self, company_name: str, company_id: int, financial_data: dict, **kwargs) -> AgentResult:
        self.log_info(f"Earnings quality analysis for {company_name}")
        result = AgentResult(agent_id=self.agent_id, agent_name=self.agent_name)

        if not financial_data:
            result.status = "PARTIAL"
            result.error = "No financial data"
            return result

        years = sorted(financial_data.keys(), reverse=True)
        result.data_quality = self._check_data_quality(financial_data)

        eq_by_year: dict[str, dict] = {}
        accrual_by_year: dict[str, AccrualResult] = {}
        analyzer = AccrualAnalyzer()

        for i, year in enumerate(years[:5]):
            d = financial_data[year]
            prev = financial_data.get(years[i + 1]) if i + 1 < len(years) else {}
            eq_by_year[year] = self._compute_eq_metrics(d, prev)
            try:
                accrual_by_year[year] = analyzer.calculate(
                    net_income=d.get("net_income", 0) or 0,
                    cfo=d.get("cfo", 0) or 0,
                    ebitda=d.get("ebitda", 0) or (d.get("net_income", 0) or 0) * 1.5,
                    total_assets=d.get("total_assets", 1) or 1,
                    total_assets_prev=prev.get("total_assets", d.get("total_assets", 1)) or 1,
                    working_capital=(d.get("current_assets", 0) or 0) - (d.get("current_liabilities", 0) or 0),
                    working_capital_prev=(prev.get("current_assets", 0) or 0) - (prev.get("current_liabilities", 0) or 0),
                    cash=d.get("cash_equivalents", 0) or 0,
                    cash_prev=prev.get("cash_equivalents", 0) or 0,
                    depreciation=d.get("depreciation", 0) or 0,
                    revenue=d.get("revenue", 1) or 1,
                    accounts_receivable=d.get("accounts_receivable", 0) or 0,
                    accounts_receivable_prev=prev.get("accounts_receivable", 0) or 0,
                )
            except Exception as e:
                self.log_warning(f"Accrual calculation failed for {year}: {e}")

        self._generate_findings(result, eq_by_year, accrual_by_year, years, company_name)

        rag_result = self._run_agentic_rag(
            company_name,
            "Earnings quality forensics: Sloan accrual decomposition, non-GAAP vs GAAP gap, "
            "exceptional and one-time items inflating reported PAT, effective tax rate consistency, "
            "and gross margin sustainability.",
            financial_data,
        )
        result.raw_analysis = rag_result.raw_text

        result.summary = self._build_summary(company_name, eq_by_year, accrual_by_year, years, result)
        self._save_output(result, company_name)
        self.log_info(f"Earnings Quality complete. Risk={result.risk_score:.1f}/100")
        return result

    def _compute_eq_metrics(self, d: dict, prev: dict) -> dict:
        net_income = d.get("net_income", 0) or 0
        revenue = d.get("revenue", 1) or 1
        gross_profit = d.get("gross_profit", 0) or 0
        ebit = d.get("ebit", 0) or net_income
        ebitda = d.get("ebitda", 0) or ebit * 1.2
        tax = d.get("income_tax_expense", 0) or 0
        pretax_income = net_income + tax if tax else net_income * 1.3

        # Tax rate analysis
        effective_tax_rate = safe_divide(tax, pretax_income) if pretax_income > 0 else None
        prev_tax = prev.get("income_tax_expense", 0) or 0
        prev_pretax = (prev.get("net_income", 0) or 0) + prev_tax
        prev_etr = safe_divide(prev_tax, prev_pretax) if prev_pretax > 0 else None

        # Gross margin
        gross_margin = safe_divide(gross_profit, revenue) * 100
        prev_gross_profit = prev.get("gross_profit", 0) or 0
        prev_revenue = prev.get("revenue", 1) or 1
        prev_gross_margin = safe_divide(prev_gross_profit, prev_revenue) * 100
        gross_margin_change = gross_margin - prev_gross_margin

        # Net margin
        net_margin = safe_divide(net_income, revenue) * 100
        prev_net_income = prev.get("net_income", 0) or 0
        prev_net_margin = safe_divide(prev_net_income, prev_revenue) * 100

        return {
            "net_income": net_income,
            "revenue": revenue,
            "gross_margin_pct": round(gross_margin, 2),
            "gross_margin_change": round(gross_margin_change, 2),
            "net_margin_pct": round(net_margin, 2),
            "prev_net_margin_pct": round(prev_net_margin, 2),
            "effective_tax_rate": round(effective_tax_rate, 4) if effective_tax_rate is not None else None,
            "prev_effective_tax_rate": round(prev_etr, 4) if prev_etr is not None else None,
            "tax_rate_change": round(effective_tax_rate - prev_etr, 4) if (effective_tax_rate is not None and prev_etr is not None) else None,
        }

    def _generate_findings(
        self, result: AgentResult, eq_by_year: dict, accrual_by_year: dict, years: list, company_name: str
    ) -> None:
        latest_year = years[0]
        eq = eq_by_year.get(latest_year, {})
        accrual = accrual_by_year.get(latest_year)
        score_contributors = []

        # ── Accrual engine flags ──────────────────────────────────
        if accrual:
            for flag in accrual.flags:
                severity = flag.get("severity", "MODERATE")
                risk_level = "CRITICAL" if severity == "CRITICAL" else ("HIGH" if severity == "HIGH" else "MEDIUM")
                f = self._create_finding(
                    "RED_FLAG",
                    flag.get("type", "Accrual Flag").replace("_", " ").title(),
                    flag.get("message", ""),
                    f"{flag.get('message', '')} | Earnings Quality: {accrual.earnings_quality} | {accrual.interpretation}",
                    fiscal_year=latest_year,
                    risk_level=risk_level,
                    confidence=0.88 if severity == "CRITICAL" else 0.82,
                )
                result.red_flags.append(f); result.findings.append(f)
                score_contributors.append(88 if severity == "CRITICAL" else 68 if severity == "HIGH" else 50)

        # ── Tax rate consistency ──────────────────────────────────
        etr = eq.get("effective_tax_rate")
        etr_change = eq.get("tax_rate_change")
        if etr is not None and etr_change is not None:
            if abs(etr_change) > 0.10:  # 10% swing in tax rate
                direction = "drop" if etr_change < 0 else "increase"
                f = self._create_finding(
                    "RED_FLAG",
                    f"Tax Rate {direction.title()} of {abs(etr_change)*100:.1f}pp YoY — Suspicious",
                    "A large one-year change in effective tax rate often indicates aggressive tax planning, "
                    "deferred tax reversals, or one-time items inflating/deflating reported profits.",
                    f"ETR: {(eq.get('prev_effective_tax_rate') or 0)*100:.1f}% → {etr*100:.1f}% "
                    f"(Δ{etr_change*100:+.1f}pp). Statutory rate ≈ 25%.",
                    fiscal_year=latest_year, risk_level="HIGH", confidence=0.82,
                )
                result.red_flags.append(f); result.findings.append(f)
                score_contributors.append(65)
        if etr is not None and etr < 0.08:
            f = self._create_finding(
                "RED_FLAG",
                f"Suspiciously Low Effective Tax Rate: {etr*100:.1f}%",
                "ETR far below statutory rate. Suggests aggressive tax avoidance, offshore structures, "
                "or deferred tax asset recognition.",
                f"ETR = {etr*100:.1f}% vs. statutory ~25%",
                fiscal_year=latest_year, risk_level="MODERATE", confidence=0.75,
            )
            result.red_flags.append(f); result.findings.append(f)
            score_contributors.append(55)

        # ── Gross margin volatility ───────────────────────────────
        gm_change = eq.get("gross_margin_change", 0) or 0
        if abs(gm_change) > 5:
            direction = "spike" if gm_change > 0 else "collapse"
            risk = "MEDIUM" if abs(gm_change) < 10 else "HIGH"
            f = self._create_finding(
                "RED_FLAG" if gm_change < 0 else "OBSERVATION",
                f"Gross Margin {direction.title()}: {gm_change:+.1f}pp YoY",
                f"A {abs(gm_change):.1f}pp gross margin change in one year is unusual. "
                f"{'Rising margins could reflect favorable mix or pricing power — or aggressive cost capitalisation.' if gm_change > 0 else 'Falling margins indicate pricing pressure or cost creep.'}",
                f"Gross Margin: {(eq.get('gross_margin_pct') or 0) - gm_change:.1f}% → {eq.get('gross_margin_pct', 0):.1f}%",
                fiscal_year=latest_year, risk_level=risk, confidence=0.78,
            )
            result.findings.append(f)
            if gm_change < -3:
                result.red_flags.append(f)
                score_contributors.append(60)

        # ── Multi-year earnings quality trend ─────────────────────
        if len(accrual_by_year) >= 3:
            quality_scores = []
            for y in sorted(accrual_by_year.keys()):
                a = accrual_by_year[y]
                q_map = {"HIGH": 10, "MODERATE": 30, "LOW": 60, "POOR": 90}
                quality_scores.append((y, q_map.get(a.earnings_quality, 50)))
            if quality_scores[-1][1] > quality_scores[0][1] + 20:
                f = self._create_finding(
                    "RED_FLAG",
                    "Earnings Quality Deteriorating Over Multi-Year Period",
                    "Accrual ratios and cash conversion metrics have worsened consistently.",
                    f"Quality trend: {', '.join([f'{y}: {accrual_by_year[y].earnings_quality}' for y, _ in quality_scores])}",
                    risk_level="HIGH", confidence=0.85,
                )
                result.red_flags.append(f); result.findings.append(f)
                score_contributors.append(72)

        # ── High earnings quality green flag ─────────────────────
        if accrual and accrual.earnings_quality in ("HIGH",) and not result.red_flags:
            f = self._create_finding(
                "GREEN_FLAG",
                f"High Earnings Quality: {accrual.interpretation}",
                "Earnings are well backed by cash flows. Low accruals. Sustainable profits.",
                accrual.interpretation,
                fiscal_year=latest_year, risk_level="POSITIVE", confidence=0.82,
            )
            result.green_flags.append(f); result.findings.append(f)
            score_contributors.append(15)

        result.risk_score = (sum(score_contributors) / len(score_contributors)) if score_contributors else 40.0
        result.risk_score = max(10.0, min(95.0, result.risk_score))

    def _build_prompt(self, company_name: str, eq_by_year: dict, accrual_by_year: dict, years: list, context: str) -> str:
        eq = eq_by_year.get(years[0], {})
        accrual = accrual_by_year.get(years[0])

        prompt = (
            f"Earnings quality forensic analysis for {company_name}.\n\n"
            f"EARNINGS QUALITY METRICS (FY{years[0]}):\n"
            f"  Gross Margin: {eq.get('gross_margin_pct', 'N/A')}% (Δ{eq.get('gross_margin_change', 0):+.1f}pp YoY)\n"
            f"  Net Margin: {eq.get('net_margin_pct', 'N/A')}%\n"
            f"  Effective Tax Rate: {(eq.get('effective_tax_rate') or 0)*100:.1f}%\n"
        )
        if accrual:
            prompt += (
                f"  Earnings Quality: {accrual.earnings_quality}\n"
                f"  CF Accrual Ratio: {accrual.cf_accrual_ratio:.3f}\n"
                f"  Cash Earnings Ratio (CFO/NI): {accrual.cash_earnings_ratio:.2f}x\n"
                f"  CFO/EBITDA: {accrual.cash_conversion_ratio:.2f}x\n"
                f"  ΔAR/Revenue: {accrual.revenue_accrual_ratio:.3f}\n"
            )
        if len(years) > 1:
            prompt += "\nMULTI-YEAR EARNINGS QUALITY:\n"
            for y in sorted(years[:4], reverse=True):
                a = accrual_by_year.get(y)
                prompt += f"  FY{y}: Quality={a.earnings_quality if a else 'N/A'}, Margin={eq_by_year.get(y, {}).get('net_margin_pct', 'N/A')}%\n"
        if context:
            prompt += f"\nDOCUMENT CONTEXT:\n{context[:1500]}\n"
        prompt += (
            "\nAssess earnings quality and sustainability. Flag non-GAAP adjustments, "
            "one-time gains, tax anomalies, and accrual-based inflation. "
            "Evidence → Analysis → Reasoning → Conclusion."
        )
        return prompt

    def _build_summary(self, company_name: str, eq_by_year: dict, accrual_by_year: dict, years: list, result: AgentResult) -> str:
        eq = eq_by_year.get(years[0], {})
        accrual = accrual_by_year.get(years[0])
        quality = accrual.earnings_quality if accrual else "N/A"
        return (
            f"EARNINGS QUALITY — {company_name} FY{years[0]}\n"
            f"Gross Margin: {eq.get('gross_margin_pct', 'N/A')}% (Δ{eq.get('gross_margin_change', 0):+.1f}pp) | "
            f"ETR: {(eq.get('effective_tax_rate') or 0)*100:.1f}% | Quality: {quality}\n"
            f"Risk Score: {result.risk_score:.1f}/100 | Red Flags: {len(result.red_flags)}"
        )
