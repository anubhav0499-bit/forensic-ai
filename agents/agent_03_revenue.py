"""
Agent 3 — Revenue Forensics Agent
==================================
Detects: channel stuffing, premature recognition, deferred-revenue pull-forward,
AR/revenue divergence, Q4 skew, fictitious revenue signals.
"""

from __future__ import annotations
import re
from .base_agent import BaseForensicAgent, AgentResult
from utils.helpers import safe_divide


def _extract_year_int(key: str) -> int:
    """Extract a 4-digit calendar year from any fiscal-year key format."""
    m = re.search(r'20\d{2}', key)
    return int(m.group()) if m else 0


class RevenueForensicsAgent(BaseForensicAgent):
    """
    Dedicated revenue forensics engine.
    Uses quantitative checks first; LLM synthesis second.
    """

    # Q4 revenue > this fraction of annual revenue = channel stuffing signal
    Q4_SKEW_THRESHOLD = 0.35
    # AR growing this many pp faster than revenue = receivables manipulation signal
    AR_REV_GAP_THRESHOLD = 15.0
    # Deferred revenue shrinking YoY by > this % while revenue is growing = pull-forward signal
    DEFERRED_REV_SHRINK_THRESHOLD = -0.10
    # Revenue CAGR > this while EBIT CAGR negative = revenue without margin signal
    REV_EBIT_DIVERGENCE_THRESHOLD = 0.20

    def investigate(
        self, company_name: str, company_id: int, financial_data: dict, **kwargs
    ) -> AgentResult:
        self.log_info(f"Revenue forensics for {company_name}")
        result = AgentResult(agent_id=self.agent_id, agent_name=self.agent_name)
        result.data_quality = self._check_data_quality(financial_data)

        if not financial_data:
            result.status = "FAILED"
            result.error = "No financial data"
            return result

        years = sorted(financial_data.keys(), reverse=True)
        scores: list[float] = []

        # ── 1. AR vs Revenue Growth Gap ──────────────────────────
        for i in range(min(3, len(years) - 1)):
            curr_y, prev_y = years[i], years[i + 1]
            curr = financial_data[curr_y]
            prev = financial_data[prev_y]

            rev_curr = curr.get("revenue", 0) or 0
            rev_prev = prev.get("revenue", 0) or 0
            ar_curr  = curr.get("accounts_receivable", 0) or 0
            ar_prev  = prev.get("accounts_receivable", 0) or 0

            if rev_prev <= 0 or ar_prev <= 0:
                continue

            rev_growth = safe_divide(rev_curr - rev_prev, abs(rev_prev)) * 100
            ar_growth  = safe_divide(ar_curr  - ar_prev,  abs(ar_prev))  * 100
            gap = ar_growth - rev_growth

            if gap > self.AR_REV_GAP_THRESHOLD:
                severity = "CRITICAL" if gap > 40 else "HIGH" if gap > 25 else "MEDIUM"
                f = self._create_finding(
                    finding_type="RED_FLAG",
                    title=f"AR Growing {gap:.1f}pp Faster Than Revenue ({curr_y})",
                    detail=(
                        f"Accounts receivable grew {ar_growth:.1f}% while revenue grew "
                        f"{rev_growth:.1f}%. A persistent gap suggests channel stuffing, "
                        f"aggressive recognition, or collection problems."
                    ),
                    evidence=(
                        f"FY{curr_y}: Revenue {rev_curr/1e6:.1f}M (growth {rev_growth:.1f}%) | "
                        f"AR {ar_curr/1e6:.1f}M (growth {ar_growth:.1f}%) | Gap {gap:.1f}pp"
                    ),
                    fiscal_year=curr_y,
                    risk_level=severity,
                    confidence=0.85,
                    calculation=f"AR_growth={ar_growth:.2f}%, Rev_growth={rev_growth:.2f}%, Gap={gap:.2f}pp",
                )
                result.red_flags.append(f)
                result.findings.append(f)
                scores.append(70 if severity == "CRITICAL" else 55 if severity == "HIGH" else 42)

        # ── 2. Deferred Revenue Shrinking While Revenue Rises ────
        for i in range(min(2, len(years) - 1)):
            curr_y, prev_y = years[i], years[i + 1]
            curr = financial_data[curr_y]
            prev = financial_data[prev_y]

            def_rev_curr = curr.get("deferred_revenue", 0) or 0
            def_rev_prev = prev.get("deferred_revenue", 0) or 0
            rev_curr = curr.get("revenue", 0) or 0
            rev_prev = prev.get("revenue", 0) or 0

            if def_rev_prev <= 0:
                continue

            def_chg = safe_divide(def_rev_curr - def_rev_prev, def_rev_prev)
            rev_chg = safe_divide(rev_curr - rev_prev, abs(rev_prev) or 1)

            if def_chg < self.DEFERRED_REV_SHRINK_THRESHOLD and rev_chg > 0.05:
                f = self._create_finding(
                    finding_type="RED_FLAG",
                    title=f"Deferred Revenue Declining While Revenue Rises ({curr_y})",
                    detail=(
                        f"Deferred revenue fell {abs(def_chg)*100:.1f}% while total revenue rose "
                        f"{rev_chg*100:.1f}%. This pattern suggests acceleration (pull-forward) "
                        f"of previously deferred recognition."
                    ),
                    evidence=(
                        f"FY{curr_y}: Deferred Rev {def_rev_curr/1e6:.1f}M "
                        f"(Δ {def_chg*100:.1f}%) | Revenue {rev_curr/1e6:.1f}M "
                        f"(Δ {rev_chg*100:.1f}%)"
                    ),
                    fiscal_year=curr_y,
                    risk_level="HIGH",
                    confidence=0.78,
                    calculation=f"Deferred_chg={def_chg*100:.2f}%, Rev_chg={rev_chg*100:.2f}%",
                )
                result.red_flags.append(f)
                result.findings.append(f)
                scores.append(55)

        # ── 3. Revenue CAGR vs EBIT CAGR Divergence ─────────────
        if len(years) >= 3:
            latest_y = years[0]
            oldest_y = years[min(2, len(years) - 1)]
            _ly, _oy = _extract_year_int(latest_y), _extract_year_int(oldest_y)
            n_years  = (_ly - _oy) if (_ly and _oy and _ly > _oy) else 2

            rev_l = financial_data[latest_y].get("revenue", 0) or 0
            rev_o = financial_data[oldest_y].get("revenue", 0) or 0
            ebit_l = financial_data[latest_y].get("ebit", 0) or 0
            ebit_o = financial_data[oldest_y].get("ebit", 0) or 0

            if rev_o > 0 and ebit_o > 0 and n_years > 0:
                rev_cagr  = (rev_l  / rev_o)  ** (1 / n_years) - 1
                ebit_cagr = (ebit_l / ebit_o) ** (1 / n_years) - 1

                if rev_cagr > self.REV_EBIT_DIVERGENCE_THRESHOLD and ebit_cagr < 0:
                    f = self._create_finding(
                        finding_type="RED_FLAG",
                        title=f"Revenue CAGR {rev_cagr*100:.1f}% With Negative EBIT CAGR {ebit_cagr*100:.1f}%",
                        detail=(
                            "Revenue is growing rapidly but operating profit is declining. "
                            "This can indicate aggressive revenue recognition with costs hidden "
                            "off-income-statement, or a fundamental deterioration in unit economics."
                        ),
                        evidence=(
                            f"FY{oldest_y}→{latest_y}: Revenue CAGR {rev_cagr*100:.1f}% | "
                            f"EBIT CAGR {ebit_cagr*100:.1f}% | "
                            f"Revenue {rev_l/1e6:.1f}M vs {rev_o/1e6:.1f}M | "
                            f"EBIT {ebit_l/1e6:.1f}M vs {ebit_o/1e6:.1f}M"
                        ),
                        fiscal_year=latest_y,
                        risk_level="HIGH",
                        confidence=0.80,
                        calculation=(
                            f"Rev CAGR=({rev_l}/{rev_o})^(1/{n_years})-1={rev_cagr*100:.2f}% | "
                            f"EBIT CAGR=({ebit_l}/{ebit_o})^(1/{n_years})-1={ebit_cagr*100:.2f}%"
                        ),
                    )
                    result.red_flags.append(f)
                    result.findings.append(f)
                    scores.append(58)

        # ── 4. Q4 Revenue Skew Detection ─────────────────────────
        latest = financial_data.get(years[0], {})
        q4_rev   = latest.get("q4_revenue", 0) or 0
        ann_rev  = latest.get("revenue",   0) or 0
        if q4_rev > 0 and ann_rev > 0:
            q4_share = safe_divide(q4_rev, ann_rev)
            if q4_share > self.Q4_SKEW_THRESHOLD:
                f = self._create_finding(
                    finding_type="RED_FLAG",
                    title=f"Q4 Revenue = {q4_share*100:.1f}% of Annual Revenue ({years[0]})",
                    detail=(
                        f"Q4 accounts for {q4_share*100:.1f}% of annual revenue — above the "
                        f"{self.Q4_SKEW_THRESHOLD*100:.0f}% channel-stuffing threshold. "
                        "Suggests year-end deals pushed to meet targets."
                    ),
                    evidence=(
                        f"FY{years[0]}: Q4 Rev {q4_rev/1e6:.1f}M / Annual Rev {ann_rev/1e6:.1f}M "
                        f"= {q4_share*100:.1f}% (threshold: {self.Q4_SKEW_THRESHOLD*100:.0f}%)"
                    ),
                    fiscal_year=years[0],
                    risk_level="HIGH",
                    confidence=0.72,
                    calculation=f"Q4_share={q4_share*100:.2f}%",
                )
                result.red_flags.append(f)
                result.findings.append(f)
                scores.append(60)

        # ── 5. Revenue Stability / Positive Quality Signals ──────
        if len(years) >= 3:
            rev_growth_rates = []
            for i in range(min(3, len(years) - 1)):
                c = financial_data[years[i]].get("revenue", 0) or 0
                p = financial_data[years[i + 1]].get("revenue", 0) or 0
                if p > 0:
                    rev_growth_rates.append(safe_divide(c - p, p))

            if rev_growth_rates:
                avg_growth = sum(rev_growth_rates) / len(rev_growth_rates)
                volatility = (max(rev_growth_rates) - min(rev_growth_rates))

                if 0.05 <= avg_growth <= 0.30 and volatility < 0.15:
                    f = self._create_finding(
                        finding_type="GREEN_FLAG",
                        title=f"Stable Revenue Growth: {avg_growth*100:.1f}% Avg, Low Volatility",
                        detail="Consistent, moderate revenue growth with low year-to-year volatility indicates organic demand rather than recognition manipulation.",
                        evidence=f"Avg growth: {avg_growth*100:.1f}% | Volatility (max-min range): {volatility*100:.1f}pp",
                        fiscal_year=years[0],
                        risk_level="POSITIVE",
                        confidence=0.75,
                    )
                    result.green_flags.append(f)
                    result.findings.append(f)
                    scores.append(30)

        # ── 6. LLM Synthesis ─────────────────────────────────────
        context = self._retrieve_context(
            company_name,
            "revenue recognition policy deferred revenue channel stuffing quarterly sales breakdown",
        )
        quant_summary = (
            f"Quantitative checks found {len(result.red_flags)} red flags. "
            + ("; ".join(f.title for f in result.red_flags[:5]) if result.red_flags else "No major quantitative red flags.")
        )
        years_data = {y: financial_data[y] for y in years[:3]}
        prompt = (
            f"REVENUE FORENSICS — {company_name}\n\n"
            f"Quantitative Pre-Analysis:\n{quant_summary}\n\n"
            f"Financial Data (last 3Y):\n{str(years_data)[:2000]}\n\n"
            f"Document Context:\n{context}\n\n"
            "Investigate:\n"
            "1. Revenue recognition policies — are they conservative or aggressive?\n"
            "2. Any channel-stuffing patterns, return provisions, or customer concentration?\n"
            "3. Deferred revenue and unearned revenue trends.\n"
            "4. Qualitative management language around revenue guidance.\n"
            "Format: Evidence → Analysis → Conclusion for each finding."
        )
        result.raw_analysis = self._analyze_with_llm(prompt, "forensic_accountant", max_tokens=2048)

        # ── Scoring ───────────────────────────────────────────────
        base = sum(scores) / len(scores) if scores else 35.0
        result.risk_score = max(10.0, min(95.0, base))
        result.summary = (
            f"REVENUE FORENSICS — {company_name}: Risk={result.risk_score:.1f}/100 | "
            f"Red Flags={len(result.red_flags)} | Green Flags={len(result.green_flags)}"
        )
        self._save_output(result, company_name)
        self.log_info(f"Revenue forensics complete. Risk={result.risk_score:.1f}/100")
        return result
