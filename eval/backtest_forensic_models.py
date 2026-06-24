"""
Backtest — Forensic Scoring Engine Accuracy on Synthetic Fraud-Pattern Data
=============================================================================

Tests the deterministic scoring engines (Beneish, Altman, Dechow, Piotroski,
Accrual) against synthetic two-year financial datasets engineered to mirror
the publicly documented modus operandi of well-known fraud cases, plus a
control group of clean companies with normal growth profiles.

These are NOT reconstructions of real companies' actual filed financials —
exact historical figures were not available to verify, so inventing precise
numbers and labeling them "Enron's 2001 balance sheet" would be presenting
unverified claims as fact. Instead, each fraud-pattern case is built to
exhibit the specific ratio distortion the real case is known for (e.g. the
WorldCom case is documented to have capitalized ~$3.8B of operating
expenses as capex — the synthetic "worldcom_pattern" case reproduces that
distortion's effect on AQI/DEPI, not WorldCom's literal balance sheet).

Purpose: validate that the scoring engines' formulas and thresholds
actually separate manipulated-pattern data from clean data, independent of
any LLM call — pure deterministic Python, runs in seconds, no API cost.

Run: python -m eval.backtest_forensic_models
"""

from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from forensics.beneish_score import BeneishMScore, BeneishInputs
from forensics.altman_score import AltmanZScore, AltmanInputs
from forensics.dechow_score import DechowFScore
from forensics.piotroski_score import PiotroskiFScore, PiotroskiInputs
from forensics.accrual_analysis import AccrualAnalyzer


@dataclass
class CaseData:
    name: str
    label: str           # "FRAUD" or "CLEAN"
    pattern: str          # which documented MO this models
    revenue_t: float
    revenue_tm1: float
    cogs_t: float
    cogs_tm1: float
    sga_t: float
    sga_tm1: float
    net_income_t: float
    net_income_tm1: float
    total_assets_t: float
    total_assets_tm1: float
    current_assets_t: float
    current_assets_tm1: float
    ppe_t: float
    ppe_tm1: float
    depreciation_t: float
    depreciation_tm1: float
    receivables_t: float
    receivables_tm1: float
    inventory_t: float
    inventory_tm1: float
    cash_t: float
    cash_tm1: float
    current_liabilities_t: float
    current_liabilities_tm1: float
    long_term_debt_t: float
    long_term_debt_tm1: float
    working_capital_t: float
    working_capital_tm1: float
    taxes_payable_t: float
    cfo_t: float
    cfo_tm1: float
    ebitda_t: float
    ebit_t: float
    retained_earnings_t: float
    market_cap_t: float
    shares_outstanding_t: float
    shares_outstanding_tm1: float
    shares_issued: bool = False
    debt_issued: bool = False


# ═══════════════════════════════════════════════════════════════════════
# FRAUD-PATTERN CASES (8) — each models one documented MO
# ═══════════════════════════════════════════════════════════════════════

FRAUD_CASES = [
    CaseData(
        name="revenue_inflation_pattern", label="FRAUD",
        pattern="Satyam-style: fictitious receivables/cash growing far faster than revenue",
        revenue_t=2200, revenue_tm1=2000, cogs_t=1450, cogs_tm1=1300,
        sga_t=300, sga_tm1=280, net_income_t=420, net_income_tm1=380,
        total_assets_t=3400, total_assets_tm1=2900,
        current_assets_t=2100, current_assets_tm1=1650,
        ppe_t=900, ppe_tm1=850, depreciation_t=95, depreciation_tm1=90,
        receivables_t=950, receivables_tm1=520,        # AR up 83% vs revenue up 10% → high DSRI
        inventory_t=180, inventory_tm1=170,
        cash_t=480, cash_tm1=460,
        current_liabilities_t=700, current_liabilities_tm1=650,
        long_term_debt_t=300, long_term_debt_tm1=320,
        working_capital_t=1400, working_capital_tm1=1000,
        taxes_payable_t=40, cfo_t=130, cfo_tm1=340,    # CFO collapses despite rising NI
        ebitda_t=560, ebit_t=465, retained_earnings_t=1100,
        market_cap_t=3000, shares_outstanding_t=500, shares_outstanding_tm1=500,
    ),
    CaseData(
        name="opex_capitalization_pattern", label="FRAUD",
        pattern="WorldCom-style: operating expenses capitalized as capex, depreciation slowed",
        revenue_t=4200, revenue_tm1=4100, cogs_t=2900, cogs_tm1=2750,
        sga_t=480, sga_tm1=470, net_income_t=380, net_income_tm1=120,
        total_assets_t=8200, total_assets_tm1=6500,     # assets balloon from capitalized opex
        current_assets_t=2000, current_assets_tm1=1950,
        ppe_t=5600, ppe_tm1=4000,                        # PP&E jump = capitalized line costs
        depreciation_t=280, depreciation_tm1=360,        # depreciation rate slows (DEPI signal)
        receivables_t=820, receivables_tm1=800,
        inventory_t=90, inventory_tm1=85,
        cash_t=350, cash_tm1=400,
        current_liabilities_t=1500, current_liabilities_tm1=1400,
        long_term_debt_t=3200, long_term_debt_tm1=2400,
        working_capital_t=500, working_capital_tm1=550,
        taxes_payable_t=60, cfo_t=900, cfo_tm1=850,
        ebitda_t=1500, ebit_t=1220, retained_earnings_t=2100,
        market_cap_t=15000, shares_outstanding_t=2900, shares_outstanding_tm1=2900,
        debt_issued=True,
    ),
    CaseData(
        name="round_trip_growth_pattern", label="FRAUD",
        pattern="Enron-style: explosive implausible sales growth via round-trip trades, margin erosion",
        revenue_t=9800, revenue_tm1=4400,                # SGI ~2.2x — far above 1.6 threshold
        cogs_t=9300, cogs_tm1=4000,                       # margin collapses as low-margin trades dominate
        sga_t=520, sga_tm1=380, net_income_t=410, net_income_tm1=380,
        total_assets_t=7200, total_assets_tm1=5200,
        current_assets_t=4800, current_assets_tm1=3100,
        ppe_t=1400, ppe_tm1=1300, depreciation_t=110, depreciation_tm1=105,
        receivables_t=1900, receivables_tm1=750,
        inventory_t=60, inventory_tm1=55,
        cash_t=600, cash_tm1=580,
        current_liabilities_t=3100, current_liabilities_tm1=1900,
        long_term_debt_t=1800, long_term_debt_tm1=1600,
        working_capital_t=1700, working_capital_tm1=1200,
        taxes_payable_t=50, cfo_t=90, cfo_tm1=300,
        ebitda_t=620, ebit_t=510, retained_earnings_t=1400,
        market_cap_t=20000, shares_outstanding_t=750, shares_outstanding_tm1=750,
    ),
    CaseData(
        name="phantom_assets_pattern", label="FRAUD",
        pattern="Wirecard-style: fictitious cash/trustee balances, soft-asset inflation",
        revenue_t=2100, revenue_tm1=1750, cogs_t=1300, cogs_tm1=1050,
        sga_t=380, sga_tm1=330, net_income_t=290, net_income_tm1=260,
        total_assets_t=4400, total_assets_tm1=3100,
        current_assets_t=2600, current_assets_tm1=1500,  # phantom cash inflates current assets
        ppe_t=350, ppe_tm1=340, depreciation_t=40, depreciation_tm1=38,
        receivables_t=620, receivables_tm1=420,
        inventory_t=20, inventory_tm1=18,
        cash_t=1850, cash_tm1=950,                        # "trustee cash" that doesn't reconcile
        current_liabilities_t=900, current_liabilities_tm1=750,
        long_term_debt_t=400, long_term_debt_tm1=380,
        working_capital_t=1700, working_capital_tm1=750,
        taxes_payable_t=25, cfo_t=60, cfo_tm1=240,
        ebitda_t=370, ebit_t=330, retained_earnings_t=900,
        market_cap_t=9000, shares_outstanding_t=120, shares_outstanding_tm1=120,
    ),
    CaseData(
        name="leverage_concealment_pattern", label="FRAUD",
        pattern="IL&FS-style: debt round-tripped through SPVs, holdco leverage understated then snaps",
        revenue_t=1600, revenue_tm1=1500, cogs_t=1100, cogs_tm1=1000,
        sga_t=220, sga_tm1=210, net_income_t=90, net_income_tm1=130,
        total_assets_t=5200, total_assets_tm1=4600,
        current_assets_t=1400, current_assets_tm1=1500,
        ppe_t=900, ppe_tm1=880, depreciation_t=70, depreciation_tm1=68,
        receivables_t=700, receivables_tm1=600,
        inventory_t=40, inventory_tm1=38,
        cash_t=110, cash_tm1=220,
        current_liabilities_t=2200, current_liabilities_tm1=1500,
        long_term_debt_t=2600, long_term_debt_tm1=1900,    # leverage spikes once SPVs consolidated
        working_capital_t=-800, working_capital_tm1=0,
        taxes_payable_t=15, cfo_t=-40, cfo_tm1=110,
        ebitda_t=220, ebit_t=150, retained_earnings_t=300,
        market_cap_t=1800, shares_outstanding_t=200, shares_outstanding_tm1=200,
        debt_issued=True,
    ),
    CaseData(
        name="sales_overstatement_pattern", label="FRAUD",
        pattern="Luckin Coffee-style: fabricated transaction volume, implausible same-period sales growth",
        revenue_t=3300, revenue_tm1=1450,                  # SGI ~2.3x, far above threshold
        cogs_t=2050, cogs_tm1=950, sga_t=900, sga_tm1=620,
        net_income_t=-180, net_income_tm1=-260,
        total_assets_t=3600, total_assets_tm1=2200,
        current_assets_t=2200, current_assets_tm1=1300,
        ppe_t=1100, ppe_tm1=750, depreciation_t=90, depreciation_tm1=70,
        receivables_t=480, receivables_tm1=180,
        inventory_t=160, inventory_tm1=110,
        cash_t=900, cash_tm1=700,
        current_liabilities_t=1300, current_liabilities_tm1=800,
        long_term_debt_t=600, long_term_debt_tm1=400,
        working_capital_t=900, working_capital_tm1=500,
        taxes_payable_t=10, cfo_t=-300, cfo_tm1=-200,
        ebitda_t=-90, ebit_t=-180, retained_earnings_t=-400,
        market_cap_t=4000, shares_outstanding_t=330, shares_outstanding_tm1=300,
        shares_issued=True,
    ),
    CaseData(
        name="deferred_writedown_pattern", label="FRAUD",
        pattern="Carillion-style: rising leverage/dividends maintained while goodwill impairment deferred",
        revenue_t=5200, revenue_tm1=5100, cogs_t=4950, cogs_tm1=4800,
        sga_t=220, sga_tm1=210, net_income_t=10, net_income_tm1=140,
        total_assets_t=6200, total_assets_tm1=6300,
        current_assets_t=2600, current_assets_tm1=2700,
        ppe_t=900, ppe_tm1=950, depreciation_t=60, depreciation_tm1=65,
        receivables_t=1800, receivables_tm1=1600,
        inventory_t=80, inventory_tm1=85,
        cash_t=120, cash_tm1=260,
        current_liabilities_t=2400, current_liabilities_tm1=2100,
        long_term_debt_t=2200, long_term_debt_tm1=1700,
        working_capital_t=200, working_capital_tm1=600,
        taxes_payable_t=20, cfo_t=-60, cfo_tm1=180,
        ebitda_t=180, ebit_t=120, retained_earnings_t=-200,
        market_cap_t=900, shares_outstanding_t=800, shares_outstanding_tm1=800,
        debt_issued=True,
    ),
    CaseData(
        name="fictitious_transactions_pattern", label="FRAUD",
        pattern="Steinhoff-style: fictitious inter-company deals booked as revenue, complex structure",
        revenue_t=4600, revenue_tm1=3700, cogs_t=3300, cogs_tm1=2550,
        sga_t=480, sga_tm1=420, net_income_t=520, net_income_tm1=400,
        total_assets_t=8800, total_assets_tm1=6900,
        current_assets_t=3600, current_assets_tm1=2700,
        ppe_t=2200, ppe_tm1=1950, depreciation_t=140, depreciation_tm1=135,
        receivables_t=1700, receivables_tm1=950,
        inventory_t=900, inventory_tm1=750,
        cash_t=380, cash_tm1=420,
        current_liabilities_t=2100, current_liabilities_tm1=1600,
        long_term_debt_t=2600, long_term_debt_tm1=1900,
        working_capital_t=1500, working_capital_tm1=1100,
        taxes_payable_t=45, cfo_t=80, cfo_tm1=460,
        ebitda_t=900, ebit_t=760, retained_earnings_t=1900,
        market_cap_t=11000, shares_outstanding_t=2900, shares_outstanding_tm1=2900,
        debt_issued=True,
    ),
]

# ═══════════════════════════════════════════════════════════════════════
# CLEAN CONTROL CASES (8) — normal growth, no manipulation pattern
# ═══════════════════════════════════════════════════════════════════════

CLEAN_CASES = [
    CaseData(
        name=f"clean_control_{i+1}", label="CLEAN",
        pattern="Stable organic growth, no manipulation signal",
        revenue_t=rev_t, revenue_tm1=rev_tm1,
        cogs_t=rev_t * 0.62, cogs_tm1=rev_tm1 * 0.62,
        sga_t=rev_t * 0.14, sga_tm1=rev_tm1 * 0.14,
        net_income_t=rev_t * 0.11, net_income_tm1=rev_tm1 * 0.11,
        total_assets_t=rev_t * 1.55, total_assets_tm1=rev_tm1 * 1.5,
        current_assets_t=rev_t * 0.65, current_assets_tm1=rev_tm1 * 0.64,
        ppe_t=rev_t * 0.55, ppe_tm1=rev_tm1 * 0.55,
        depreciation_t=rev_t * 0.05, depreciation_tm1=rev_tm1 * 0.05,
        receivables_t=rev_t * 0.21, receivables_tm1=rev_tm1 * 0.21,   # in line with revenue
        inventory_t=rev_t * 0.10, inventory_tm1=rev_tm1 * 0.10,
        cash_t=rev_t * 0.18, cash_tm1=rev_tm1 * 0.17,
        current_liabilities_t=rev_t * 0.28, current_liabilities_tm1=rev_tm1 * 0.27,
        long_term_debt_t=rev_t * 0.30, long_term_debt_tm1=rev_tm1 * 0.31,
        working_capital_t=rev_t * 0.37, working_capital_tm1=rev_tm1 * 0.37,
        taxes_payable_t=rev_t * 0.02,
        cfo_t=rev_t * 0.13, cfo_tm1=rev_tm1 * 0.125,        # CFO tracks/exceeds NI
        ebitda_t=rev_t * 0.18, ebit_t=rev_t * 0.13,
        retained_earnings_t=rev_t * 0.9,
        market_cap_t=rev_t * 3.2,
        shares_outstanding_t=500, shares_outstanding_tm1=500,
    )
    for i, (rev_t, rev_tm1) in enumerate([
        (2400, 2150), (5600, 4950), (1200, 1080), (8800, 7900),
        (3300, 2950), (980, 870), (6200, 5500), (4100, 3650),
    ])
]

ALL_CASES = FRAUD_CASES + CLEAN_CASES


def run_engines(c: CaseData) -> dict:
    out = {"name": c.name, "label": c.label, "pattern": c.pattern}

    # ── Beneish M-Score ──────────────────────────────────────
    beneish = BeneishMScore().calculate(BeneishInputs(
        net_receivables_t=c.receivables_t, sales_t=c.revenue_t, cogs_t=c.cogs_t,
        current_assets_t=c.current_assets_t, ppe_t=c.ppe_t, total_assets_t=c.total_assets_t,
        depreciation_t=c.depreciation_t, sga_t=c.sga_t, total_debt_t=c.long_term_debt_t,
        current_liabilities_t=c.current_liabilities_t, working_capital_t=c.working_capital_t,
        cash_t=c.cash_t, taxes_payable_t=c.taxes_payable_t,
        net_receivables_tm1=c.receivables_tm1, sales_tm1=c.revenue_tm1, cogs_tm1=c.cogs_tm1,
        current_assets_tm1=c.current_assets_tm1, ppe_tm1=c.ppe_tm1, total_assets_tm1=c.total_assets_tm1,
        depreciation_tm1=c.depreciation_tm1, sga_tm1=c.sga_tm1, total_debt_tm1=c.long_term_debt_tm1,
        current_liabilities_tm1=c.current_liabilities_tm1, working_capital_tm1=c.working_capital_tm1,
        cash_tm1=c.cash_tm1,
    ))
    out["beneish_m"] = beneish.m_score
    out["beneish_flag"] = beneish.manipulation_likely or beneish.risk_level in ("HIGH", "CRITICAL")

    # ── Altman Z-Score (manufacturing/public variant) ───────
    total_liabilities = c.current_liabilities_t + c.long_term_debt_t
    altman = AltmanZScore().calculate(AltmanInputs(
        working_capital=c.working_capital_t, total_assets=c.total_assets_t,
        retained_earnings=c.retained_earnings_t, ebit=c.ebit_t,
        market_cap=c.market_cap_t, total_liabilities=total_liabilities,
        revenue=c.revenue_t, is_manufacturing=True, is_public=True,
    ))
    out["altman_z"] = altman.z_score
    out["altman_flag"] = altman.zone == "DISTRESS"

    # ── Dechow F-Score ───────────────────────────────────────
    dechow = DechowFScore().calculate(
        total_assets=c.total_assets_t, total_assets_prev=c.total_assets_tm1,
        cash=c.cash_t, accounts_receivable=c.receivables_t,
        accounts_receivable_prev=c.receivables_tm1, inventory=c.inventory_t,
        inventory_prev=c.inventory_tm1, ppe_net=c.ppe_t,
        revenue=c.revenue_t, revenue_prev=c.revenue_tm1,
        net_income=c.net_income_t, net_income_prev=c.net_income_tm1,
        cfo=c.cfo_t, shares_issued=c.shares_issued, debt_issued=c.debt_issued,
    )
    out["dechow_f"] = dechow.f_score
    out["dechow_flag"] = dechow.f_score > 0.025

    # ── Piotroski F-Score ────────────────────────────────────
    piotroski = PiotroskiFScore().calculate(PiotroskiInputs(
        net_income=c.net_income_t, total_assets=c.total_assets_t, cfo=c.cfo_t,
        total_debt=c.long_term_debt_t, current_ratio=safe_ratio(c.current_assets_t, c.current_liabilities_t),
        shares_outstanding=c.shares_outstanding_t, gross_profit=c.revenue_t - c.cogs_t,
        revenue=c.revenue_t, long_term_debt=c.long_term_debt_t,
        net_income_prev=c.net_income_tm1, total_assets_prev=c.total_assets_tm1,
        total_debt_prev=c.long_term_debt_tm1,
        current_ratio_prev=safe_ratio(c.current_assets_tm1, c.current_liabilities_tm1),
        shares_outstanding_prev=c.shares_outstanding_tm1,
        gross_profit_prev=c.revenue_tm1 - c.cogs_tm1, revenue_prev=c.revenue_tm1,
        long_term_debt_prev=c.long_term_debt_tm1,
    ))
    out["piotroski_f"] = piotroski.f_score
    out["piotroski_flag"] = piotroski.f_score <= 3

    # ── Accrual Analysis (Sloan) ─────────────────────────────
    accrual = AccrualAnalyzer().calculate(
        net_income=c.net_income_t, cfo=c.cfo_t, ebitda=c.ebitda_t,
        total_assets=c.total_assets_t, total_assets_prev=c.total_assets_tm1,
        working_capital=c.working_capital_t, working_capital_prev=c.working_capital_tm1,
        cash=c.cash_t, cash_prev=c.cash_tm1, depreciation=c.depreciation_t,
        revenue=c.revenue_t, accounts_receivable=c.receivables_t,
        accounts_receivable_prev=c.receivables_tm1,
    )
    out["accrual_ratio"] = accrual.cf_accrual_ratio
    out["accrual_flag"] = abs(accrual.cf_accrual_ratio) > 0.10

    # Composite: flagged if >=2 of the 4 fraud-relevant models agree
    # (Altman tests distress, not fraud — reported separately, excluded from composite)
    fraud_votes = sum([out["beneish_flag"], out["dechow_flag"], out["piotroski_flag"], out["accrual_flag"]])
    out["composite_flag"] = fraud_votes >= 2
    out["fraud_votes"] = fraud_votes
    return out


def safe_ratio(a: float, b: float) -> float:
    return a / b if b else 0.0


def main():
    results = [run_engines(c) for c in ALL_CASES]

    print("\n" + "=" * 100)
    print("FORENSIC SCORING ENGINE BACKTEST - synthetic fraud-pattern vs. clean-control data")
    print("=" * 100)
    header = f"{'Case':<32}{'Label':<7}{'Beneish':>9}{'Altman Z':>10}{'Dechow':>9}{'Piotroski':>10}{'Accrual%':>10}{'Votes':>7}{'Composite':>11}"
    print(header)
    print("-" * 100)
    for r in results:
        print(
            f"{r['name']:<32}{r['label']:<7}"
            f"{r['beneish_m']:>9.2f}{r['altman_z']:>10.2f}{r['dechow_f']:>9.3f}"
            f"{r['piotroski_f']:>10}{r['accrual_ratio']*100:>9.1f}%{r['fraud_votes']:>7}"
            f"{'FLAGGED' if r['composite_flag'] else 'clear':>11}"
        )

    print("-" * 100)

    # ── Accuracy metrics ──────────────────────────────────────
    frauds = [r for r in results if r["label"] == "FRAUD"]
    clean = [r for r in results if r["label"] == "CLEAN"]

    def rate(cases, key, expect_true):
        matches = sum(1 for c in cases if c[key] == expect_true)
        return matches, len(cases), (matches / len(cases) * 100 if cases else 0)

    print("\nPER-MODEL ACCURACY (fraud-pattern sensitivity / clean-control specificity):\n")
    for model, key in [
        ("Beneish M-Score", "beneish_flag"),
        ("Dechow F-Score", "dechow_flag"),
        ("Piotroski F-Score", "piotroski_flag"),
        ("Accrual Ratio (Sloan)", "accrual_flag"),
        ("Altman Z (distress, not fraud)", "altman_flag"),
    ]:
        sens_n, sens_d, sens_pct = rate(frauds, key, True)
        spec_n, spec_d, spec_pct = rate(clean, key, False)
        print(f"  {model:<32} sensitivity {sens_n}/{sens_d} ({sens_pct:.0f}%)   "
              f"specificity {spec_n}/{spec_d} ({spec_pct:.0f}%)")

    comp_sens_n, comp_sens_d, comp_sens_pct = rate(frauds, "composite_flag", True)
    comp_spec_n, comp_spec_d, comp_spec_pct = rate(clean, "composite_flag", False)
    print(f"\n  {'COMPOSITE (>=2 of 4 fraud models agree)':<32} "
          f"sensitivity {comp_sens_n}/{comp_sens_d} ({comp_sens_pct:.0f}%)   "
          f"specificity {comp_spec_n}/{comp_spec_d} ({comp_spec_pct:.0f}%)")
    print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    main()
