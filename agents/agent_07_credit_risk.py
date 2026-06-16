"""
Agent 7 — Credit Risk Assessment Agent
========================================
Computes leverage, coverage, and liquidity ratios.
Maps to Altman EM Z-Score (non-manufacturing) and CRISIL credit metrics.
Frameworks: Altman EM (2000), CRISIL/ICRA methodology, ISA 570 Going Concern.
"""

from __future__ import annotations
from utils.helpers import safe_divide
from forensics.altman_score import AltmanZScore, AltmanInputs
from forensics.risk_scorer import RiskScorer
from llm.prompts import build_credit_risk_prompt
from .base_agent import BaseForensicAgent, AgentResult


_COVERAGE_THRESHOLDS = {
    "CRITICAL": 1.5,   # Interest coverage < 1.5x = near default (CRISIL)
    "HIGH":     2.5,
    "MODERATE": 4.0,
    "LOW":      6.0,   # > 6x = comfortable
}
_LEVERAGE_THRESHOLDS = {
    "CRITICAL": 5.0,   # Net Debt/EBITDA > 5x
    "HIGH":     3.5,
    "MODERATE": 2.5,
    "LOW":      1.5,
}
_DE_RATIO_THRESHOLDS = {
    "CRITICAL": 3.0,   # D/E > 3x with declining revenue (skill: +25 pts)
    "HIGH":     2.0,
}
_SHORT_TERM_DEBT_THRESHOLD = 0.60  # >60% short-term debt = rollover risk


class CreditRiskAgent(BaseForensicAgent):
    """
    Institutional credit risk analysis: leverage, coverage, liquidity,
    maturity profile, and covenants. Maps to credit rating analogue.
    """

    def investigate(self, company_name: str, company_id: int, financial_data: dict, **kwargs) -> AgentResult:
        self.log_info(f"Credit risk assessment for {company_name}")
        result = AgentResult(agent_id=self.agent_id, agent_name=self.agent_name)

        if not financial_data:
            result.status = "PARTIAL"
            result.error = "No financial data"
            return result

        years = sorted(financial_data.keys(), reverse=True)
        result.data_quality = self._check_data_quality(financial_data)

        ratios_by_year: dict[str, dict] = {}
        for year in years[:5]:
            ratios_by_year[year] = self._compute_credit_ratios(financial_data[year])

        self._generate_findings(result, ratios_by_year, years, company_name, financial_data)

        rag_result = self._run_agentic_rag(
            company_name,
            "Credit risk analysis: debt covenants, interest coverage ratio, Net Debt/EBITDA, "
            "liquidity ratios, short-term debt rollover risk, credit rating trajectory, "
            "and Altman EM Z-Score distress zone signals.",
            financial_data,
        )
        result.raw_analysis = rag_result.raw_text

        result.summary = self._build_summary(company_name, ratios_by_year, years, result)
        self._save_output(result, company_name)
        self.log_info(f"Credit Risk complete. Risk={result.risk_score:.1f}/100")
        return result

    # ─── Ratio Computation ────────────────────────────────────────

    def _compute_credit_ratios(self, d: dict) -> dict:
        ebit = d.get("ebit", 0) or d.get("net_income", 0) or 0
        ebitda = d.get("ebitda", 0) or (ebit * 1.2 if ebit > 0 else 1)
        interest = abs(d.get("interest_expense", 0) or 0)
        total_debt = d.get("total_debt", 0) or (
            (d.get("long_term_debt", 0) or 0) + (d.get("short_term_debt", 0) or 0)
        )
        cash = d.get("cash_equivalents", 0) or 0
        net_debt = total_debt - cash
        equity = d.get("shareholder_equity", 0) or 1
        total_assets = d.get("total_assets", 1) or 1
        total_liabilities = d.get("total_liabilities", 0) or 0
        current_assets = d.get("current_assets", 0) or 0
        current_liabilities = d.get("current_liabilities", 1) or 1
        inventory = d.get("inventory", 0) or 0
        revenue = d.get("revenue", 1) or 1
        cfo = d.get("cfo", 0) or 0

        # Coverage ratios
        interest_coverage = safe_divide(ebit, interest) if interest > 0 else None
        dscr = safe_divide(cfo, interest) if interest > 0 else None  # Debt Service Coverage

        # Leverage ratios
        net_debt_to_ebitda = safe_divide(net_debt, ebitda) if ebitda != 0 else None
        debt_to_equity = safe_divide(total_debt, equity) if equity > 0 else None
        debt_to_assets = safe_divide(total_liabilities, total_assets)

        # Liquidity ratios
        current_ratio = safe_divide(current_assets, current_liabilities)
        quick_ratio = safe_divide(current_assets - inventory, current_liabilities)
        cash_ratio = safe_divide(cash, current_liabilities)

        # Altman EM Z-Score (non-manufacturing variant, Altman 2000)
        # EM formula: Z'=6.56(WC/TA)+3.26(RE/TA)+6.72(EBIT/TA)+1.05(BVE/TL)
        # Safe>2.60, Grey 1.10-2.60, Distress<1.10
        altman_result = None
        try:
            wc = current_assets - current_liabilities
            inp = AltmanInputs(
                working_capital=wc, total_assets=total_assets,
                retained_earnings=d.get("retained_earnings", 0) or 0,
                ebit=ebit, market_cap=d.get("market_cap", 0) or 0,
                book_equity=equity, total_liabilities=total_liabilities,
                revenue=revenue, is_manufacturing=False, is_public=True,
            )
            altman_result = AltmanZScore().calculate(inp)
        except Exception:
            pass

        # Short-term debt concentration
        short_term_debt = d.get("short_term_debt", 0) or 0
        short_term_debt_ratio = safe_divide(short_term_debt, total_debt) if total_debt > 0 else None

        return {
            "interest_coverage": round(interest_coverage, 2) if interest_coverage is not None else None,
            "dscr": round(dscr, 2) if dscr is not None else None,
            "net_debt_to_ebitda": round(net_debt_to_ebitda, 2) if net_debt_to_ebitda is not None else None,
            "debt_to_equity": round(debt_to_equity, 2) if debt_to_equity is not None else None,
            "debt_to_assets": round(debt_to_assets, 3),
            "current_ratio": round(current_ratio, 2),
            "quick_ratio": round(quick_ratio, 2),
            "cash_ratio": round(cash_ratio, 3),
            "net_debt": net_debt,
            "total_debt": total_debt,
            "ebitda": ebitda,
            "short_term_debt_ratio": round(short_term_debt_ratio, 3) if short_term_debt_ratio is not None else None,
            "altman_zone": altman_result.zone if altman_result else None,
            "altman_z": altman_result.z_score if altman_result else None,
            "implied_rating": AltmanZScore().get_implied_credit_rating(altman_result.z_score) if altman_result else None,
        }

    # ─── Findings Generation ──────────────────────────────────────

    def _generate_findings(
        self, result: AgentResult, ratios_by_year: dict, years: list, company_name: str, financial_data: dict
    ) -> None:
        latest_year = years[0]
        r = ratios_by_year.get(latest_year, {})
        score_contributors = []

        # ── Interest Coverage ──────────────────────────────────────
        ic = r.get("interest_coverage")
        if ic is not None:
            if ic < _COVERAGE_THRESHOLDS["CRITICAL"]:
                f = self._create_finding(
                    "RED_FLAG",
                    f"CRITICAL: Interest Coverage {ic:.1f}x — Near-Default Stress",
                    "Interest coverage below 1.5x means EBIT barely covers interest. Near-default territory.",
                    f"EBIT/Interest Expense = {ic:.2f}x (threshold: >1.5x critical, >2.5x high risk). "
                    f"Implied credit rating: {r.get('implied_rating', 'N/A')}",
                    fiscal_year=latest_year, risk_level="CRITICAL", confidence=0.92,
                )
                result.red_flags.append(f); result.findings.append(f)
                score_contributors.append(92)
            elif ic < _COVERAGE_THRESHOLDS["HIGH"]:
                f = self._create_finding(
                    "RED_FLAG",
                    f"Weak Interest Coverage: {ic:.1f}x (threshold: >2.5x)",
                    "Limited headroom above interest obligations. Covenant breach risk.",
                    f"EBIT/Interest = {ic:.2f}x",
                    fiscal_year=latest_year, risk_level="HIGH", confidence=0.88,
                )
                result.red_flags.append(f); result.findings.append(f)
                score_contributors.append(72)
            elif ic > _COVERAGE_THRESHOLDS["LOW"]:
                f = self._create_finding(
                    "GREEN_FLAG",
                    f"Strong Interest Coverage: {ic:.1f}x",
                    "Comfortable debt service capacity.",
                    f"EBIT/Interest = {ic:.2f}x. Well above 4x+ benchmark.",
                    fiscal_year=latest_year, risk_level="POSITIVE", confidence=0.85,
                )
                result.green_flags.append(f); result.findings.append(f)
                score_contributors.append(15)

        # ── Net Debt / EBITDA ──────────────────────────────────────
        nd_ebitda = r.get("net_debt_to_ebitda")
        if nd_ebitda is not None:
            if nd_ebitda > _LEVERAGE_THRESHOLDS["CRITICAL"]:
                f = self._create_finding(
                    "RED_FLAG",
                    f"Extreme Leverage: Net Debt/EBITDA = {nd_ebitda:.1f}x",
                    "Net Debt exceeds 5x EBITDA. Junk territory. Refinancing risk is elevated.",
                    f"Net Debt = {r.get('net_debt', 0)/1e6:.0f}M / EBITDA = {r.get('ebitda', 0)/1e6:.0f}M = {nd_ebitda:.1f}x",
                    fiscal_year=latest_year, risk_level="CRITICAL", confidence=0.90,
                )
                result.red_flags.append(f); result.findings.append(f)
                score_contributors.append(90)
            elif nd_ebitda > _LEVERAGE_THRESHOLDS["HIGH"]:
                f = self._create_finding(
                    "RED_FLAG",
                    f"High Leverage: Net Debt/EBITDA = {nd_ebitda:.1f}x",
                    "Leverage above 3.5x EBITDA leaves limited room for earnings deterioration.",
                    f"Net Debt/EBITDA = {nd_ebitda:.1f}x (investment grade threshold: ≤3.0x)",
                    fiscal_year=latest_year, risk_level="HIGH", confidence=0.85,
                )
                result.red_flags.append(f); result.findings.append(f)
                score_contributors.append(68)
            elif 0 < nd_ebitda < _LEVERAGE_THRESHOLDS["LOW"]:
                f = self._create_finding(
                    "GREEN_FLAG",
                    f"Conservative Leverage: Net Debt/EBITDA = {nd_ebitda:.1f}x",
                    "Low leverage relative to earnings. Financial flexibility maintained.",
                    f"Net Debt/EBITDA = {nd_ebitda:.1f}x (below 1.5x is very conservative)",
                    fiscal_year=latest_year, risk_level="POSITIVE", confidence=0.85,
                )
                result.green_flags.append(f); result.findings.append(f)
                score_contributors.append(12)

        # ── Current Ratio ──────────────────────────────────────────
        cr = r.get("current_ratio", 1)
        if cr < 1.0:
            f = self._create_finding(
                "RED_FLAG",
                f"Current Ratio Below 1.0x: {cr:.2f}x — Short-Term Solvency Risk",
                "Current liabilities exceed current assets. Company may struggle to meet near-term obligations.",
                f"Current Ratio = {cr:.2f}x. Quick Ratio = {r.get('quick_ratio', 0):.2f}x",
                fiscal_year=latest_year, risk_level="HIGH", confidence=0.88,
            )
            result.red_flags.append(f); result.findings.append(f)
            score_contributors.append(75)

        # ── Leverage trend ─────────────────────────────────────────
        if len(years) >= 3:
            debt_trend = [(y, ratios_by_year[y].get("net_debt_to_ebitda", 0) or 0) for y in sorted(years[:4])]
            lev_values = [v for _, v in debt_trend]
            if lev_values[-1] > lev_values[0] * 1.5 and lev_values[-1] > 2:
                f = self._create_finding(
                    "RED_FLAG",
                    "Leverage Steadily Rising Over Multiple Years",
                    "Multi-year debt accumulation without proportionate EBITDA growth.",
                    f"Net Debt/EBITDA trend: {', '.join([f'{y}:{v:.1f}x' for y, v in debt_trend])}",
                    risk_level="HIGH", confidence=0.82,
                )
                result.red_flags.append(f); result.findings.append(f)
                score_contributors.append(68)

        # ── Short-Term Debt Concentration ─────────────────────────
        st_ratio = r.get("short_term_debt_ratio")
        if st_ratio is not None and st_ratio > _SHORT_TERM_DEBT_THRESHOLD:
            f = self._create_finding(
                "RED_FLAG",
                f"Short-Term Debt Concentration: {st_ratio*100:.0f}% of Total Debt",
                "Over 60% of total debt maturing in <1 year creates rollover risk. "
                "If credit markets tighten or lenders withdraw, the company faces a liquidity crisis.",
                f"Short-term debt = {st_ratio*100:.0f}% of total debt "
                f"({r.get('total_debt', 0)/1e6:.0f}M total). Threshold: >60% = rollover risk.",
                fiscal_year=latest_year, risk_level="HIGH", confidence=0.82,
            )
            result.red_flags.append(f); result.findings.append(f)
            score_contributors.append(65)

        # ── Altman EM Distress Zone ────────────────────────────────
        if r.get("altman_zone") == "DISTRESS":
            f = self._create_finding(
                "RED_FLAG",
                f"Altman EM Z-Score in Distress Zone: {r.get('altman_z', 0):.2f}",
                "Altman Emerging Market Z-Score below 1.10 (non-manufacturing variant, Altman 2000) "
                "indicates significant financial distress probability. "
                "EM formula: Z'=6.56(WC/TA)+3.26(RE/TA)+6.72(EBIT/TA)+1.05(BVE/TL).",
                f"EM Z-Score = {r.get('altman_z', 0):.3f} (EM distress threshold < 1.10). "
                f"Implied rating: {r.get('implied_rating', 'N/A')}",
                fiscal_year=latest_year, risk_level="CRITICAL", confidence=0.88,
            )
            result.red_flags.append(f); result.findings.append(f)
            score_contributors.append(85)
        elif r.get("altman_zone") == "GREY":
            f = self._create_finding(
                "RED_FLAG",
                f"Altman EM Z-Score in Grey Zone: {r.get('altman_z', 0):.2f}",
                "Altman EM Z-Score in grey zone (1.10–2.60). Elevated distress risk. Monitor trajectory.",
                f"EM Z-Score = {r.get('altman_z', 0):.3f} (safe > 2.60, grey 1.10-2.60, distress < 1.10).",
                fiscal_year=latest_year, risk_level="HIGH", confidence=0.78,
            )
            result.red_flags.append(f); result.findings.append(f)
            score_contributors.append(58)

        # Compute credit risk score
        scorer = RiskScorer()
        z = r.get("altman_z")
        if z is not None:
            score_contributors.append(scorer.altman_to_score(z))

        result.risk_score = (sum(score_contributors) / len(score_contributors)) if score_contributors else 35.0
        result.risk_score = max(10.0, min(95.0, result.risk_score))

    def _build_prompt(self, company_name: str, ratios_by_year: dict, years: list, context: str) -> str:
        r = ratios_by_year.get(years[0], {})
        trend_lines = []
        if len(years) > 1:
            for y in sorted(ratios_by_year.keys(), reverse=True)[:4]:
                rv = ratios_by_year[y]
                trend_lines.append(
                    f"  FY{y}: NDebt/EBITDA={rv.get('net_debt_to_ebitda', 'N/A')}, "
                    f"IC={rv.get('interest_coverage', 'N/A')}x, CR={rv.get('current_ratio', 'N/A')}x"
                )
        return build_credit_risk_prompt(
            company=company_name,
            interest_coverage=r.get("interest_coverage"),
            net_debt_ebitda=r.get("net_debt_to_ebitda"),
            debt_equity=r.get("debt_to_equity"),
            altman_z=r.get("altman_z"),
            altman_zone=r.get("altman_zone"),
            short_term_debt_ratio=r.get("short_term_debt_ratio"),
            trend_summary="\n".join(trend_lines) if trend_lines else "Single year data only.",
            document_context=context or "",
        )

    def _build_summary(self, company_name: str, ratios_by_year: dict, years: list, result: AgentResult) -> str:
        r = ratios_by_year.get(years[0], {})
        return (
            f"CREDIT RISK — {company_name} FY{years[0]}\n"
            f"Interest Coverage: {r.get('interest_coverage', 'N/A')}x | "
            f"Net Debt/EBITDA: {r.get('net_debt_to_ebitda', 'N/A')}x | "
            f"Current Ratio: {r.get('current_ratio', 'N/A')}x\n"
            f"Altman Zone: {r.get('altman_zone', 'N/A')} | "
            f"Implied Rating: {r.get('implied_rating', 'N/A')} | "
            f"Risk Score: {result.risk_score:.1f}/100"
        )
