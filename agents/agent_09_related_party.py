"""
Agent 9 — Related Party Forensics Agent
=========================================
Detects: self-dealing, promoter loans, RPT revenue inflation, disclosure opacity,
director remuneration misalignment, inter-company guarantees, tunnelling patterns.
Frameworks: SEBI RPT Regulations 2021 / OECD Corporate Governance Principles (2023) / ISA 550.
"""

from __future__ import annotations
import re
from .base_agent import BaseForensicAgent, AgentResult
from llm.prompts import build_rpt_prompt
from utils.helpers import safe_divide


class RelatedPartyAgent(BaseForensicAgent):
    """
    Dedicated related-party transaction (RPT) forensics.
    Combines NLP pattern detection on disclosures + quantitative ratio checks.
    Applies SEBI RPT Regulations 2021 materiality thresholds and OECD Principle VI.C tunnelling detection.
    """

    # RPT revenue > this share of total revenue = concentration risk (SEBI RPT 2021: >10% turnover = material)
    RPT_REVENUE_THRESHOLD = 0.15      # 15% — elevated concern level
    # Loans to directors/promoters > this share of net worth (SEBI RPT 2021 materiality: 10%)
    PROMOTER_LOAN_THRESHOLD = 0.10
    # Director remuneration > this % of net profit = governance concern
    DIR_REMU_THRESHOLD = 0.10
    # Circular transaction / tunnelling risk threshold
    CIRCULAR_TRANSACTION_THRESHOLD = 0.20  # RPT purchases + sales both >20% revenue = circular risk

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
        "special resolution", "minority shareholder approval",
    ]
    # Red-flag language in RPT disclosures
    _RED_FLAG_PHRASES = [
        "waived", "forgiven", "written off", "converted to equity",
        "interest-free", "unsecured loan", "no interest",
        "below market", "concessional", "subsidised",
    ]
    # Tunnelling / circular transaction signals (OECD Principle VI.C)
    _TUNNELLING_PHRASES = [
        "inter-company loan", "inter-company advance", "loan to subsidiary",
        "purchase from promoter", "sale to promoter group",
        "guarantee on behalf of", "corporate guarantee",
        "investment in associate", "unsecured inter-corporate deposit",
        "inter-corporate deposit", "icd",
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

        # ── 5. Tunnelling / Circular Transaction Detection ────────
        tunnelling_hits = sum(1 for p in self._TUNNELLING_PHRASES if p.lower() in context.lower())
        if tunnelling_hits >= 3:
            f = self._create_finding(
                finding_type="RED_FLAG",
                title=f"Tunnelling / Circular Transaction Signals Detected ({tunnelling_hits} indicators)",
                detail=(
                    f"Disclosure text contains {tunnelling_hits} signals associated with tunnelling patterns "
                    "(OECD Principle VI.C): inter-corporate loans, guarantees for related entities, "
                    "or inter-company deposits. These structures can transfer value from the listed entity "
                    "to the promoter group at the expense of minority shareholders."
                ),
                evidence=(
                    f"Tunnelling phrase matches: {tunnelling_hits}. "
                    f"Matching terms: {', '.join(p for p in self._TUNNELLING_PHRASES if p.lower() in context.lower())[:300]}"
                ),
                fiscal_year=years[0] if years else "",
                risk_level="CRITICAL" if tunnelling_hits >= 5 else "HIGH",
                confidence=0.65,
            )
            result.red_flags.append(f)
            result.findings.append(f)
            scores.append(72 if tunnelling_hits >= 5 else 58)

        # ── 6. SEBI RPT 2021 Materiality Check ───────────────────
        # Check if any year had promoter loans above SEBI's 10% net worth materiality threshold
        # that should have required shareholder special resolution
        for yr in years[:2]:
            d = financial_data[yr]
            rpt_rev = d.get("related_party_revenue", 0) or 0
            total_rev = d.get("revenue", 1) or 1
            rpt_purchases = d.get("related_party_purchases", 0) or 0
            # SEBI RPT 2021: if both RPT sales AND purchases are >10% of revenue → circular risk
            if (rpt_rev / total_rev > 0.10) and (rpt_purchases / total_rev > 0.10):
                f = self._create_finding(
                    finding_type="RED_FLAG",
                    title=f"Potential Circular RPT Transactions ({yr}): Both Sales and Purchases via Related Parties",
                    detail=(
                        f"FY{yr}: RPT revenue = {rpt_rev/total_rev*100:.1f}% of total revenue AND "
                        f"RPT purchases = {rpt_purchases/total_rev*100:.1f}% of revenue. "
                        "When both sales and purchases flow through related parties at similar scales, "
                        "this is a classic indicator of revenue round-tripping / circular transactions "
                        "(SEBI RPT Regulations 2021 — heightened scrutiny trigger)."
                    ),
                    evidence=(
                        f"FY{yr}: RPT Revenue={rpt_rev/1e6:.1f}M ({rpt_rev/total_rev*100:.1f}% of rev), "
                        f"RPT Purchases={rpt_purchases/1e6:.1f}M ({rpt_purchases/total_rev*100:.1f}% of rev)"
                    ),
                    fiscal_year=yr,
                    risk_level="CRITICAL",
                    confidence=0.75,
                )
                result.red_flags.append(f)
                result.findings.append(f)
                scores.append(78)

        # ── 7. LLM Synthesis (SEBI RPT 2021 / OECD framework) ────
        rpt_revenue_share = 0.0
        if years and financial_data[years[0]].get("revenue", 0):
            rpt_revenue_share = safe_divide(
                financial_data[years[0]].get("related_party_revenue", 0) or 0,
                financial_data[years[0]].get("revenue", 1) or 1,
            )
        promoter_loan_ratio_latest = 0.0
        if years:
            d0 = financial_data[years[0]]
            pl = d0.get("loans_to_promoters", 0) or d0.get("loans_to_directors", 0) or 0
            nw = d0.get("shareholder_equity", 1) or 1
            promoter_loan_ratio_latest = safe_divide(pl, nw)

        quant_flags_str = "; ".join(f.title for f in result.red_flags[:6]) or "none"
        rag_result = self._run_agentic_rag(
            company_name,
            (
                f"Related party transaction forensics. "
                f"RPT revenue share={rpt_revenue_share*100:.1f}%, "
                f"promoter loan ratio={promoter_loan_ratio_latest*100:.1f}%. "
                f"Red flags: {quant_flags_str}. "
                "Apply SEBI RPT Regulations 2021 materiality thresholds and OECD Principle VI.C tunnelling detection."
            ),
            financial_data,
        )
        result.raw_analysis = rag_result.raw_text

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
