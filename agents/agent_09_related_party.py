"""
Agent 9 — Related Party Forensics Agent
=========================================
Detects: self-dealing, promoter loans, RPT revenue inflation, disclosure opacity,
director remuneration misalignment, inter-company guarantees.
"""

from __future__ import annotations
import re
from .base_agent import BaseForensicAgent, AgentResult
from utils.helpers import safe_divide


class RelatedPartyAgent(BaseForensicAgent):
    """
    Dedicated related-party transaction (RPT) forensics.
    Combines NLP pattern detection on disclosures + quantitative ratio checks.
    """

    # RPT revenue > this share of total revenue = concentration risk
    RPT_REVENUE_THRESHOLD = 0.15      # 15%
    # Loans to directors/promoters > this share of net worth
    PROMOTER_LOAN_THRESHOLD = 0.10
    # Director remuneration > this % of net profit = governance concern
    DIR_REMU_THRESHOLD = 0.10

    # Keywords indicating vague/opaque RPT disclosures
    _OPACITY_PHRASES = [
        "arm's length", "prevailing market rates", "approved by board",
        "customary terms", "similar to terms", "at cost",
        "approved by shareholders", "on normal commercial terms",
    ]
    # Keywords that should accompany disclosures but are often missing
    _GOOD_DISCLOSURE_PHRASES = [
        "independent valuation", "fairness opinion", "third party benchmarked",
        "audit committee reviewed", "market rate confirmed",
    ]
    # Red-flag language in RPT disclosures
    _RED_FLAG_PHRASES = [
        "waived", "forgiven", "written off", "converted to equity",
        "interest-free", "unsecured loan", "no interest",
        "below market", "concessional", "subsidised",
    ]

    def investigate(
        self, company_name: str, company_id: int, financial_data: dict, **kwargs
    ) -> AgentResult:
        self.log_info(f"Related party forensics for {company_name}")
        result = AgentResult(agent_id=self.agent_id, agent_name=self.agent_name)
        result.data_quality = self._check_data_quality(financial_data)

        years = sorted(financial_data.keys(), reverse=True)
        scores: list[float] = []

        # ── 1. RPT Revenue Concentration ─────────────────────────
        for yr in years[:3]:
            d = financial_data[yr]
            rpt_rev = d.get("related_party_revenue", 0) or 0
            total_rev = d.get("revenue", 0) or 0
            if total_rev <= 0 or rpt_rev <= 0:
                continue
            share = safe_divide(rpt_rev, total_rev)
            if share > self.RPT_REVENUE_THRESHOLD:
                sev = "CRITICAL" if share > 0.40 else "HIGH" if share > 0.25 else "MEDIUM"
                f = self._create_finding(
                    finding_type="RED_FLAG",
                    title=f"RPT Revenue = {share*100:.1f}% of Total Revenue ({yr})",
                    detail=(
                        f"Related-party transactions account for {share*100:.1f}% of revenue "
                        f"in FY{yr}. High RPT revenue concentration can indicate circular "
                        f"transactions, fictitious revenue, or captive-customer dependency."
                    ),
                    evidence=(
                        f"FY{yr}: RPT Revenue {rpt_rev/1e6:.1f}M / Total Revenue {total_rev/1e6:.1f}M "
                        f"= {share*100:.1f}% (threshold: {self.RPT_REVENUE_THRESHOLD*100:.0f}%)"
                    ),
                    fiscal_year=yr,
                    risk_level=sev,
                    confidence=0.88,
                    calculation=f"RPT_share={share*100:.2f}%",
                )
                result.red_flags.append(f)
                result.findings.append(f)
                scores.append(70 if sev == "CRITICAL" else 58 if sev == "HIGH" else 45)

        # ── 2. Promoter / Director Loans ─────────────────────────
        for yr in years[:2]:
            d = financial_data[yr]
            promoter_loan = d.get("loans_to_promoters", 0) or d.get("loans_to_directors", 0) or 0
            net_worth = d.get("shareholder_equity", 0) or 0
            if net_worth <= 0 or promoter_loan <= 0:
                continue
            ratio = safe_divide(promoter_loan, net_worth)
            if ratio > self.PROMOTER_LOAN_THRESHOLD:
                f = self._create_finding(
                    finding_type="RED_FLAG",
                    title=f"Promoter/Director Loans = {ratio*100:.1f}% of Net Worth ({yr})",
                    detail=(
                        f"Loans of {promoter_loan/1e6:.1f}M to promoters/directors represent "
                        f"{ratio*100:.1f}% of net worth. This raises expropriation risk — "
                        f"funds being tunnelled to insiders at the company's expense."
                    ),
                    evidence=(
                        f"FY{yr}: Promoter Loans {promoter_loan/1e6:.1f}M | "
                        f"Net Worth {net_worth/1e6:.1f}M | Ratio {ratio*100:.1f}%"
                    ),
                    fiscal_year=yr,
                    risk_level="CRITICAL" if ratio > 0.25 else "HIGH",
                    confidence=0.90,
                    calculation=f"ratio={promoter_loan}/{net_worth}={ratio*100:.2f}%",
                )
                result.red_flags.append(f)
                result.findings.append(f)
                scores.append(75 if ratio > 0.25 else 60)

        # ── 3. Director Remuneration vs Net Profit ────────────────
        for yr in years[:2]:
            d = financial_data[yr]
            dir_remu = d.get("director_remuneration", 0) or 0
            net_income = d.get("net_income", 0) or 0
            if net_income <= 0 or dir_remu <= 0:
                continue
            ratio = safe_divide(dir_remu, net_income)
            if ratio > self.DIR_REMU_THRESHOLD:
                f = self._create_finding(
                    finding_type="RED_FLAG",
                    title=f"Director Remuneration = {ratio*100:.1f}% of Net Profit ({yr})",
                    detail=(
                        f"Director pay of {dir_remu/1e6:.1f}M is {ratio*100:.1f}% of net profit. "
                        f"Excessive managerial pay relative to earnings is a governance red flag — "
                        f"insiders extracting value at shareholders' expense."
                    ),
                    evidence=(
                        f"FY{yr}: Director Remu {dir_remu/1e6:.1f}M | "
                        f"Net Profit {net_income/1e6:.1f}M | Ratio {ratio*100:.1f}%"
                    ),
                    fiscal_year=yr,
                    risk_level="HIGH" if ratio > 0.20 else "MEDIUM",
                    confidence=0.82,
                    calculation=f"ratio={dir_remu}/{net_income}={ratio*100:.2f}%",
                )
                result.red_flags.append(f)
                result.findings.append(f)
                scores.append(55 if ratio > 0.20 else 42)

        # ── 4. RPT Disclosure Quality NLP ─────────────────────────
        context = self._retrieve_context(
            company_name,
            "related party transactions promoter director loans guarantees disclosures RPT",
        )
        opacity_hits = sum(1 for p in self._OPACITY_PHRASES if p.lower() in context.lower())
        red_flag_hits = sum(1 for p in self._RED_FLAG_PHRASES if p.lower() in context.lower())
        good_hits = sum(1 for p in self._GOOD_DISCLOSURE_PHRASES if p.lower() in context.lower())

        if red_flag_hits >= 2:
            f = self._create_finding(
                finding_type="RED_FLAG",
                title=f"Concerning RPT Disclosure Language ({red_flag_hits} Red-Flag Phrases)",
                detail=(
                    f"Disclosure text contains {red_flag_hits} concerning phrases: "
                    f"{', '.join(p for p in self._RED_FLAG_PHRASES if p.lower() in context.lower())[:200]}. "
                    f"This suggests non-arm's-length terms in related party dealings."
                ),
                evidence=f"Pattern matches in disclosure text: {red_flag_hits} red-flag / {opacity_hits} opacity / {good_hits} good-disclosure phrases",
                fiscal_year=years[0] if years else "",
                risk_level="HIGH" if red_flag_hits >= 3 else "MEDIUM",
                confidence=0.70,
            )
            result.red_flags.append(f)
            result.findings.append(f)
            scores.append(55 if red_flag_hits >= 3 else 42)

        if opacity_hits >= 4 and good_hits == 0:
            f = self._create_finding(
                finding_type="RED_FLAG",
                title="RPT Disclosures Are Boilerplate — No Independent Validation",
                detail=(
                    f"All {opacity_hits} RPT disclosures use generic phrases like "
                    "'arm's length' and 'board approved' with no independent valuation, "
                    "fairness opinion, or audit committee review cited. "
                    "This is the hallmark of opaque RPT disclosure."
                ),
                evidence=f"Opacity phrases: {opacity_hits} | Good-disclosure phrases: {good_hits}",
                fiscal_year=years[0] if years else "",
                risk_level="MEDIUM",
                confidence=0.65,
            )
            result.red_flags.append(f)
            result.findings.append(f)
            scores.append(42)

        if good_hits >= 3 and red_flag_hits == 0:
            f = self._create_finding(
                finding_type="GREEN_FLAG",
                title="RPT Disclosures Include Independent Validation Processes",
                detail=f"Disclosure text references {good_hits} independent-validation practices, suggesting robust RPT governance.",
                evidence=f"Good-disclosure phrases found: {good_hits}",
                fiscal_year=years[0] if years else "",
                risk_level="POSITIVE",
                confidence=0.65,
            )
            result.green_flags.append(f)
            result.findings.append(f)
            scores.append(25)

        # ── 5. LLM Synthesis ─────────────────────────────────────
        quant_summary = (
            f"{len(result.red_flags)} red flags from quantitative checks. "
            + ("; ".join(f.title for f in result.red_flags[:5]) if result.red_flags else "No major RPT flags.")
        )
        years_data = {y: financial_data[y] for y in years[:3]}
        prompt = (
            f"RELATED PARTY FORENSICS — {company_name}\n\n"
            f"Quantitative Pre-Analysis:\n{quant_summary}\n\n"
            f"Financial Data:\n{str(years_data)[:1500]}\n\n"
            f"Document Context:\n{context[:2000]}\n\n"
            "Investigate:\n"
            "1. Map all significant related party relationships (entities, nature, amounts).\n"
            "2. Are RPT prices arm's length? Is there evidence of tunnelling or self-dealing?\n"
            "3. Loans, guarantees, and pledges by/to insiders — terms and adequacy of disclosure.\n"
            "4. Director remuneration vs performance — is pay aligned with shareholder returns?\n"
            "5. Any circular transactions or revenue round-tripping via related entities?\n"
            "Format: Evidence → Analysis → Conclusion. Flag CRITICAL / HIGH items explicitly."
        )
        result.raw_analysis = self._analyze_with_llm(prompt, "governance_specialist", max_tokens=2048)

        # ── Scoring ───────────────────────────────────────────────
        base = sum(scores) / len(scores) if scores else 38.0
        result.risk_score = max(10.0, min(95.0, base))
        result.summary = (
            f"RELATED PARTY FORENSICS — {company_name}: Risk={result.risk_score:.1f}/100 | "
            f"Red Flags={len(result.red_flags)} | Green Flags={len(result.green_flags)}"
        )
        self._save_output(result, company_name)
        self.log_info(f"Related party forensics complete. Risk={result.risk_score:.1f}/100")
        return result
