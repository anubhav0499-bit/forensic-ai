"""
Cross-Validation Engine
========================

References
----------
ISA 520 (IAASB): Analytical Procedures — mandatory use of analytical
    procedures in the overall review stage; ratio comparison across
    periods to identify unexpected fluctuations (§A5–A7).
SA 520 (ICAI): Indian equivalent of ISA 520.
PCAOB AS 2305: Substantive Analytical Procedures — guidance on
    expectation-vs-actual comparisons as audit evidence.

Dechow, P. M., & Skinner, D. J. (2000). "Earnings Management:
    Reconciling the Views of Accounting Academics, Practitioners,
    and Regulators." Accounting Horizons, 14(2), 235–250.
    Discusses cross-statement inconsistencies as fraud indicators.

The 10 Cross-Validation Rules
--------------------------------
1. REVENUE_CFO_DIVERGENCE    — Revenue CAGR 40%+ above CFO CAGR (3+ yrs)
2. AR_REVENUE_DIVERGENCE     — AR growing 25%+ faster than revenue (1 yr)
3. INVENTORY_COGS_DIVERGENCE — Inventory growing 30%+ faster than COGS
4. RETAINED_EARNINGS_INCONSISTENCY — RE delta ≠ NI − Dividends by >25%
5. POOR_EBITDA_CFO_CONVERSION — CFO/EBITDA < 0.40x in any year
6. GROSS_MARGIN_JUMP         — Gross margin changes >7pp in one year
7. TAX_RATE_ANOMALY          — ETR deviates >12pp from company average
8. UNDERINVESTMENT           — CapEx/Depreciation < 0.40x
9. DEBT_REVENUE_DIVERGENCE   — Debt CAGR 50%+ above revenue CAGR (3+ yrs)
10. BALANCE_SHEET_IMBALANCE  — Assets ≠ Liabilities+Equity by >5%

Checks internal consistency of financial statements across years.
Catches errors and manipulation that pass individual model tests
but fail when ratios are compared against each other.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from utils.helpers import safe_divide


@dataclass
class CrossValidationIssue:
    issue_type: str
    severity: str        # CRITICAL / HIGH / MODERATE / LOW
    description: str
    evidence: str
    metric_a: str
    value_a: float
    metric_b: str
    value_b: float
    fiscal_year: str = ""
    confidence: float = 0.85


class CrossValidator:
    """
    Checks 10 cross-statement consistency rules:
    1.  Revenue vs CFO divergence (multi-year)
    2.  AR growth vs revenue growth gap
    3.  Inventory growth vs COGS growth gap
    4.  Net income vs retained earnings consistency
    5.  EBITDA vs CFO conversion gap
    6.  Gross profit margin jump (single year)
    7.  Effective tax rate outlier
    8.  Depreciation vs CapEx ratio (under-investment)
    9.  Debt growth vs revenue growth gap
    10. Balance sheet identity check (Assets = Liabilities + Equity)
    """

    def validate(self, financial_data: dict) -> list[CrossValidationIssue]:
        """
        Run all cross-validation checks.
        Returns list of issues sorted by severity.
        """
        if not financial_data or len(financial_data) < 2:
            return []

        years = sorted(financial_data.keys())
        issues: list[CrossValidationIssue] = []

        issues.extend(self._check_revenue_cfo_divergence(financial_data, years))
        issues.extend(self._check_ar_vs_revenue(financial_data, years))
        issues.extend(self._check_inventory_vs_cogs(financial_data, years))
        issues.extend(self._check_retained_earnings_consistency(financial_data, years))
        issues.extend(self._check_ebitda_cfo_conversion(financial_data, years))
        issues.extend(self._check_gross_margin_jump(financial_data, years))
        issues.extend(self._check_tax_rate_outlier(financial_data, years))
        issues.extend(self._check_depreciation_capex(financial_data, years))
        issues.extend(self._check_debt_vs_revenue(financial_data, years))
        issues.extend(self._check_balance_sheet_identity(financial_data, years))

        # Sort by severity
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3}
        issues.sort(key=lambda x: severity_order.get(x.severity, 4))
        return issues

    # ─── Rule 1: Revenue vs CFO multi-year divergence ─────────────

    def _check_revenue_cfo_divergence(self, fd: dict, years: list) -> list[CrossValidationIssue]:
        issues = []
        if len(years) < 3:
            return issues

        rev_start = fd[years[0]].get("revenue", 0) or 0
        rev_end = fd[years[-1]].get("revenue", 0) or 0
        cfo_start = fd[years[0]].get("cfo", 0) or 0
        cfo_end = fd[years[-1]].get("cfo", 0) or 0

        if rev_start <= 0 or cfo_start <= 0:
            return issues

        rev_cagr = safe_divide(rev_end - rev_start, rev_start)
        cfo_cagr = safe_divide(cfo_end - cfo_start, cfo_start)
        divergence = rev_cagr - cfo_cagr

        if divergence > 0.40:  # Revenue growing 40%+ faster than CFO cumulatively
            issues.append(CrossValidationIssue(
                issue_type="REVENUE_CFO_DIVERGENCE",
                severity="CRITICAL" if divergence > 0.70 else "HIGH",
                description="Revenue growing significantly faster than CFO over multi-year period",
                evidence=(
                    f"Revenue growth {rev_cagr*100:.1f}% vs CFO growth {cfo_cagr*100:.1f}% "
                    f"over {len(years)-1} years. Gap: {divergence*100:.1f}pp. "
                    f"Classic earnings quality deterioration signal."
                ),
                metric_a="Revenue Growth",
                value_a=rev_cagr,
                metric_b="CFO Growth",
                value_b=cfo_cagr,
                fiscal_year=f"{years[0]}–{years[-1]}",
                confidence=0.88,
            ))
        return issues

    # ─── Rule 2: AR growth vs revenue growth ─────────────────────

    def _check_ar_vs_revenue(self, fd: dict, years: list) -> list[CrossValidationIssue]:
        issues = []
        for i in range(1, len(years)):
            curr, prev = fd[years[i]], fd[years[i - 1]]
            ar_curr = curr.get("accounts_receivable", 0) or 0
            ar_prev = prev.get("accounts_receivable", 0) or 0
            rev_curr = curr.get("revenue", 1) or 1
            rev_prev = prev.get("revenue", 1) or 1

            if ar_prev <= 0:
                continue

            ar_growth = safe_divide(ar_curr - ar_prev, ar_prev)
            rev_growth = safe_divide(rev_curr - rev_prev, rev_prev)
            gap = ar_growth - rev_growth

            if gap > 0.25:
                issues.append(CrossValidationIssue(
                    issue_type="AR_REVENUE_DIVERGENCE",
                    severity="CRITICAL" if gap > 0.50 else "HIGH",
                    description="Accounts receivable growing significantly faster than revenue",
                    evidence=(
                        f"AR grew {ar_growth*100:.1f}% vs Revenue {rev_growth*100:.1f}% in FY{years[i]}. "
                        f"Gap: {gap*100:.1f}pp. Suggests channel stuffing or fictitious revenue."
                    ),
                    metric_a="AR Growth",
                    value_a=ar_growth,
                    metric_b="Revenue Growth",
                    value_b=rev_growth,
                    fiscal_year=years[i],
                    confidence=0.90,
                ))
        return issues

    # ─── Rule 3: Inventory vs COGS ────────────────────────────────

    def _check_inventory_vs_cogs(self, fd: dict, years: list) -> list[CrossValidationIssue]:
        issues = []
        for i in range(1, len(years)):
            curr, prev = fd[years[i]], fd[years[i - 1]]
            inv_curr = curr.get("inventory", 0) or 0
            inv_prev = prev.get("inventory", 0) or 0
            cogs_curr = curr.get("cogs", 0) or curr.get("revenue", 1) * 0.6
            cogs_prev = prev.get("cogs", 0) or prev.get("revenue", 1) * 0.6

            if inv_prev <= 0 or cogs_prev <= 0:
                continue

            inv_growth = safe_divide(inv_curr - inv_prev, inv_prev)
            cogs_growth = safe_divide(cogs_curr - cogs_prev, cogs_prev)
            gap = inv_growth - cogs_growth

            if gap > 0.30:
                issues.append(CrossValidationIssue(
                    issue_type="INVENTORY_COGS_DIVERGENCE",
                    severity="HIGH",
                    description="Inventory growing substantially faster than COGS",
                    evidence=(
                        f"Inventory grew {inv_growth*100:.1f}% vs COGS {cogs_growth*100:.1f}% in FY{years[i]}. "
                        f"Gap: {gap*100:.1f}pp. Possible overproduction or inventory manipulation."
                    ),
                    metric_a="Inventory Growth",
                    value_a=inv_growth,
                    metric_b="COGS Growth",
                    value_b=cogs_growth,
                    fiscal_year=years[i],
                    confidence=0.85,
                ))
        return issues

    # ─── Rule 4: Net Income vs Retained Earnings ──────────────────

    def _check_retained_earnings_consistency(self, fd: dict, years: list) -> list[CrossValidationIssue]:
        issues = []
        for i in range(1, len(years)):
            curr, prev = fd[years[i]], fd[years[i - 1]]
            re_curr = curr.get("retained_earnings", None)
            re_prev = prev.get("retained_earnings", None)
            ni = curr.get("net_income", 0) or 0
            div = abs(curr.get("dividends_paid", 0) or 0)

            if re_curr is None or re_prev is None or ni == 0:
                continue

            re_change = re_curr - re_prev
            expected_re_change = ni - div
            discrepancy = abs(re_change - expected_re_change)
            pct_discrepancy = safe_divide(discrepancy, abs(ni)) if ni != 0 else 0

            if pct_discrepancy > 0.25 and discrepancy > 1e6:
                issues.append(CrossValidationIssue(
                    issue_type="RETAINED_EARNINGS_INCONSISTENCY",
                    severity="HIGH" if pct_discrepancy > 0.5 else "MODERATE",
                    description="Retained earnings change inconsistent with net income minus dividends",
                    evidence=(
                        f"FY{years[i]}: RE change = {re_change/1e6:.1f}M, "
                        f"Expected (NI {ni/1e6:.1f}M - Div {div/1e6:.1f}M) = {expected_re_change/1e6:.1f}M. "
                        f"Discrepancy: {discrepancy/1e6:.1f}M ({pct_discrepancy*100:.1f}%). "
                        f"Unexplained equity adjustments or prior period corrections possible."
                    ),
                    metric_a="Actual RE Change",
                    value_a=re_change,
                    metric_b="Expected RE Change (NI-Div)",
                    value_b=expected_re_change,
                    fiscal_year=years[i],
                    confidence=0.82,
                ))
        return issues

    # ─── Rule 5: EBITDA vs CFO conversion ────────────────────────

    def _check_ebitda_cfo_conversion(self, fd: dict, years: list) -> list[CrossValidationIssue]:
        issues = []
        for year in years:
            d = fd[year]
            ebitda = d.get("ebitda", 0) or 0
            cfo = d.get("cfo", 0) or 0
            if ebitda <= 0:
                continue
            conversion = safe_divide(cfo, ebitda)
            if conversion < 0.40:
                issues.append(CrossValidationIssue(
                    issue_type="POOR_EBITDA_CFO_CONVERSION",
                    severity="HIGH" if conversion < 0.25 else "MODERATE",
                    description="Poor EBITDA-to-CFO conversion — working capital absorbing cash",
                    evidence=(
                        f"FY{year}: CFO/EBITDA = {conversion:.2f}x "
                        f"(CFO={cfo/1e6:.0f}M, EBITDA={ebitda/1e6:.0f}M). "
                        f"Threshold: >0.85x. Possible accounts receivable build-up or deferred revenue rundown."
                    ),
                    metric_a="CFO",
                    value_a=cfo,
                    metric_b="EBITDA",
                    value_b=ebitda,
                    fiscal_year=year,
                    confidence=0.85,
                ))
        return issues

    # ─── Rule 6: Gross margin jump ───────────────────────────────

    def _check_gross_margin_jump(self, fd: dict, years: list) -> list[CrossValidationIssue]:
        issues = []
        for i in range(1, len(years)):
            curr, prev = fd[years[i]], fd[years[i - 1]]
            gp_curr = curr.get("gross_profit", 0) or 0
            rev_curr = curr.get("revenue", 1) or 1
            gp_prev = prev.get("gross_profit", 0) or 0
            rev_prev = prev.get("revenue", 1) or 1

            if gp_prev <= 0 or rev_prev <= 0:
                continue

            gm_curr = gp_curr / rev_curr
            gm_prev = gp_prev / rev_prev
            gm_delta = gm_curr - gm_prev

            if abs(gm_delta) > 0.07:
                issues.append(CrossValidationIssue(
                    issue_type="GROSS_MARGIN_JUMP",
                    severity="HIGH" if abs(gm_delta) > 0.12 else "MODERATE",
                    description=f"Gross margin {'increased' if gm_delta > 0 else 'collapsed'} {abs(gm_delta)*100:.1f}pp in one year",
                    evidence=(
                        f"FY{years[i-1]}: {gm_prev*100:.1f}% → FY{years[i]}: {gm_curr*100:.1f}%. "
                        f"Change: {gm_delta*100:+.1f}pp. "
                        f"{'Rising margins without explanation could mask cost capitalisation.' if gm_delta > 0 else 'Falling margins suggest pricing pressure or cost inflation.'}"
                    ),
                    metric_a=f"Gross Margin FY{years[i]}",
                    value_a=gm_curr,
                    metric_b=f"Gross Margin FY{years[i-1]}",
                    value_b=gm_prev,
                    fiscal_year=years[i],
                    confidence=0.82,
                ))
        return issues

    # ─── Rule 7: Tax rate outlier ─────────────────────────────────

    def _check_tax_rate_outlier(self, fd: dict, years: list) -> list[CrossValidationIssue]:
        issues = []
        etrs = []
        for year in years:
            d = fd[year]
            ni = d.get("net_income", 0) or 0
            tax = d.get("income_tax_expense", 0) or 0
            pretax = ni + tax
            if pretax > 0:
                etrs.append((year, safe_divide(tax, pretax)))

        if len(etrs) < 2:
            return issues

        etr_values = [e for _, e in etrs]
        avg_etr = sum(etr_values) / len(etr_values)

        for year, etr in etrs:
            deviation = abs(etr - avg_etr)
            if deviation > 0.12:
                issues.append(CrossValidationIssue(
                    issue_type="TAX_RATE_ANOMALY",
                    severity="MODERATE",
                    description=f"Effective tax rate in FY{year} deviates significantly from company average",
                    evidence=(
                        f"FY{year} ETR: {etr*100:.1f}% vs company avg: {avg_etr*100:.1f}%. "
                        f"Deviation: {deviation*100:.1f}pp. "
                        f"Investigate: deferred tax reversals, R&D credits, or offshore tax planning."
                    ),
                    metric_a=f"ETR FY{year}",
                    value_a=etr,
                    metric_b="Average ETR",
                    value_b=avg_etr,
                    fiscal_year=year,
                    confidence=0.75,
                ))
        return issues

    # ─── Rule 8: Depreciation vs CapEx ───────────────────────────

    def _check_depreciation_capex(self, fd: dict, years: list) -> list[CrossValidationIssue]:
        issues = []
        for year in years:
            d = fd[year]
            dep = d.get("depreciation", 0) or 0
            capex = abs(d.get("capex", 0) or 0)
            if dep <= 0 or capex <= 0:
                continue
            ratio = safe_divide(capex, dep)
            if ratio < 0.40:
                issues.append(CrossValidationIssue(
                    issue_type="UNDERINVESTMENT",
                    severity="MODERATE",
                    description="CapEx significantly below depreciation — possible asset under-investment",
                    evidence=(
                        f"FY{year}: CapEx {capex/1e6:.0f}M / Depreciation {dep/1e6:.0f}M = {ratio:.2f}x. "
                        f"Ratio below 1.0x suggests net asset base eroding. "
                        f"Below 0.4x is severe under-investment."
                    ),
                    metric_a="CapEx",
                    value_a=capex,
                    metric_b="Depreciation",
                    value_b=dep,
                    fiscal_year=year,
                    confidence=0.80,
                ))
        return issues

    # ─── Rule 9: Debt growth vs revenue growth ───────────────────

    def _check_debt_vs_revenue(self, fd: dict, years: list) -> list[CrossValidationIssue]:
        issues = []
        if len(years) < 3:
            return issues

        debt_start = fd[years[0]].get("total_debt", 0) or 0
        debt_end = fd[years[-1]].get("total_debt", 0) or 0
        rev_start = fd[years[0]].get("revenue", 1) or 1
        rev_end = fd[years[-1]].get("revenue", 1) or 1

        if debt_start <= 0:
            return issues

        debt_growth = safe_divide(debt_end - debt_start, debt_start)
        rev_growth = safe_divide(rev_end - rev_start, rev_start)
        gap = debt_growth - rev_growth

        if gap > 0.50 and debt_end > 1e6:
            issues.append(CrossValidationIssue(
                issue_type="DEBT_REVENUE_DIVERGENCE",
                severity="HIGH" if gap > 1.0 else "MODERATE",
                description="Debt growing much faster than revenue over multi-year period",
                evidence=(
                    f"Debt grew {debt_growth*100:.1f}% vs Revenue {rev_growth*100:.1f}% "
                    f"over {len(years)-1} years. Gap: {gap*100:.1f}pp. "
                    f"Debt {debt_start/1e6:.0f}M → {debt_end/1e6:.0f}M. Leverage creep risk."
                ),
                metric_a="Debt Growth",
                value_a=debt_growth,
                metric_b="Revenue Growth",
                value_b=rev_growth,
                fiscal_year=f"{years[0]}–{years[-1]}",
                confidence=0.85,
            ))
        return issues

    # ─── Rule 10: Balance sheet identity ─────────────────────────

    def _check_balance_sheet_identity(self, fd: dict, years: list) -> list[CrossValidationIssue]:
        issues = []
        for year in years:
            d = fd[year]
            total_assets = d.get("total_assets", 0) or 0
            total_liabilities = d.get("total_liabilities", 0) or 0
            equity = d.get("shareholder_equity", 0) or 0

            if total_assets <= 0 or total_liabilities <= 0 or equity == 0:
                continue

            expected_assets = total_liabilities + equity
            discrepancy = abs(total_assets - expected_assets)
            pct = safe_divide(discrepancy, total_assets)

            if pct > 0.05:
                issues.append(CrossValidationIssue(
                    issue_type="BALANCE_SHEET_IMBALANCE",
                    severity="CRITICAL" if pct > 0.15 else "HIGH",
                    description="Balance sheet fails Assets = Liabilities + Equity identity check",
                    evidence=(
                        f"FY{year}: Total Assets {total_assets/1e6:.0f}M ≠ "
                        f"Liabilities ({total_liabilities/1e6:.0f}M) + Equity ({equity/1e6:.0f}M) "
                        f"= {expected_assets/1e6:.0f}M. Discrepancy: {discrepancy/1e6:.0f}M ({pct*100:.1f}%). "
                        f"Possible data extraction error or off-balance-sheet exposure."
                    ),
                    metric_a="Total Assets",
                    value_a=total_assets,
                    metric_b="Liabilities + Equity",
                    value_b=expected_assets,
                    fiscal_year=year,
                    confidence=0.92,
                ))
        return issues

    # ─── Utility ─────────────────────────────────────────────────

    def summary_for_prompt(self, issues: list[CrossValidationIssue]) -> str:
        """Format issues as a concise block for LLM prompts."""
        if not issues:
            return "No cross-validation issues detected."
        lines = [f"CROSS-VALIDATION ISSUES ({len(issues)} found):"]
        for iss in issues[:8]:
            lines.append(
                f"  [{iss.severity}] {iss.issue_type} (FY{iss.fiscal_year}): {iss.description}"
            )
        return "\n".join(lines)
