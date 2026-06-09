"""
Working Capital Analysis - Inventory, Receivables, Payables Forensics
"""

from __future__ import annotations
from dataclasses import dataclass, field
from utils.helpers import safe_divide
from config import FORENSIC_THRESHOLDS


@dataclass
class WorkingCapitalResult:
    # Days metrics
    dso: float = 0.0              # Days Sales Outstanding (Receivable Days)
    dso_prev: float = 0.0
    dso_change: float = 0.0

    dio: float = 0.0              # Days Inventory Outstanding
    dio_prev: float = 0.0
    dio_change: float = 0.0

    dpo: float = 0.0              # Days Payable Outstanding
    dpo_prev: float = 0.0
    dpo_change: float = 0.0

    ccc: float = 0.0              # Cash Conversion Cycle (DSO + DIO - DPO)
    ccc_prev: float = 0.0
    ccc_change: float = 0.0

    # Ratios
    inventory_turnover: float = 0.0
    receivable_turnover: float = 0.0
    payable_turnover: float = 0.0

    # Flags
    flags: list = field(default_factory=list)
    risk_level: str = "LOW"
    interpretation: str = ""


class WorkingCapitalAnalyzer:
    """Full working capital forensics with trend analysis."""

    def calculate(
        self,
        # Current Year
        accounts_receivable: float,
        inventory: float,
        accounts_payable: float,
        revenue: float,
        cogs: float,
        # Prior Year
        accounts_receivable_prev: float = 0.0,
        inventory_prev: float = 0.0,
        accounts_payable_prev: float = 0.0,
        revenue_prev: float = 0.0,
        cogs_prev: float = 0.0,
    ) -> WorkingCapitalResult:
        r = WorkingCapitalResult()

        # ── DSO: Days Sales Outstanding ────────────────────────
        # AR / (Revenue / 365) = Average collection period
        r.dso = safe_divide(accounts_receivable * 365, revenue)
        r.dso_prev = safe_divide(accounts_receivable_prev * 365, revenue_prev)
        r.dso_change = r.dso - r.dso_prev

        # ── DIO: Days Inventory Outstanding ───────────────────
        # Inventory / (COGS / 365)
        r.dio = safe_divide(inventory * 365, cogs) if cogs else safe_divide(inventory * 365, revenue * 0.7)
        r.dio_prev = safe_divide(inventory_prev * 365, cogs_prev) if cogs_prev else safe_divide(inventory_prev * 365, revenue_prev * 0.7)
        r.dio_change = r.dio - r.dio_prev

        # ── DPO: Days Payable Outstanding ────────────────────
        # AP / (COGS / 365)
        r.dpo = safe_divide(accounts_payable * 365, cogs) if cogs else safe_divide(accounts_payable * 365, revenue * 0.7)
        r.dpo_prev = safe_divide(accounts_payable_prev * 365, cogs_prev) if cogs_prev else safe_divide(accounts_payable_prev * 365, revenue_prev * 0.7)
        r.dpo_change = r.dpo - r.dpo_prev

        # ── CCC: Cash Conversion Cycle ────────────────────────
        r.ccc = r.dso + r.dio - r.dpo
        r.ccc_prev = r.dso_prev + r.dio_prev - r.dpo_prev
        r.ccc_change = r.ccc - r.ccc_prev

        # ── Turnover ratios ───────────────────────────────────
        r.receivable_turnover = safe_divide(revenue, accounts_receivable)
        r.inventory_turnover = safe_divide(cogs, inventory) if cogs else safe_divide(revenue * 0.7, inventory)
        r.payable_turnover = safe_divide(cogs, accounts_payable) if cogs else safe_divide(revenue * 0.7, accounts_payable)

        # ── Forensic Flags ────────────────────────────────────
        flags = []

        # Receivable spike - major fraud indicator
        if r.dso_change > FORENSIC_THRESHOLDS.dso_spike_threshold:
            flags.append({
                "type": "DSO_SPIKE",
                "severity": "CRITICAL" if r.dso_change > 60 else "HIGH",
                "value": r.dso_change,
                "message": (
                    f"DSO increased by {r.dso_change:.0f} days YoY ({r.dso_prev:.0f}→{r.dso:.0f} days). "
                    f"Receivables growing significantly faster than revenue. "
                    f"Classic channel stuffing or fictitious revenue indicator."
                )
            })
        elif r.dso_change > 15:
            flags.append({
                "type": "DSO_RISING",
                "severity": "MODERATE",
                "value": r.dso_change,
                "message": f"DSO rising {r.dso_change:.0f} days. Monitor for collection quality deterioration."
            })

        # Inventory build-up
        if revenue_prev > 0:
            inv_growth = safe_divide(inventory - inventory_prev, inventory_prev)
            rev_growth = safe_divide(revenue - revenue_prev, revenue_prev)
            if inv_growth > rev_growth + FORENSIC_THRESHOLDS.inventory_growth_threshold:
                flags.append({
                    "type": "INVENTORY_INFLATION",
                    "severity": "HIGH",
                    "value": inv_growth - rev_growth,
                    "message": (
                        f"Inventory grew {inv_growth*100:.1f}% vs revenue growth of {rev_growth*100:.1f}%. "
                        f"Excess inventory build-up ({r.dio_change:.0f} days). "
                        f"Possible overproduction to inflate assets or channel stuffing."
                    )
                })

        # Payable stretching (could indicate liquidity stress)
        if r.dpo_change > FORENSIC_THRESHOLDS.dpo_stretch_threshold:
            flags.append({
                "type": "PAYABLE_STRETCHING",
                "severity": "MODERATE",
                "value": r.dpo_change,
                "message": (
                    f"DPO increased {r.dpo_change:.0f} days ({r.dpo_prev:.0f}→{r.dpo:.0f} days). "
                    f"Company taking longer to pay suppliers. Could indicate liquidity stress."
                )
            })

        # Worsening CCC
        if r.ccc_change > 30:
            flags.append({
                "type": "CCC_DETERIORATING",
                "severity": "HIGH",
                "value": r.ccc_change,
                "message": (
                    f"Cash Conversion Cycle worsened by {r.ccc_change:.0f} days "
                    f"({r.ccc_prev:.0f}→{r.ccc:.0f} days). "
                    f"Increasing working capital intensity. Cash drain risk."
                )
            })

        r.flags = flags

        # ── Overall Assessment ────────────────────────────────
        critical_flags = [f for f in flags if f["severity"] == "CRITICAL"]
        high_flags = [f for f in flags if f["severity"] == "HIGH"]

        if critical_flags:
            r.risk_level = "CRITICAL"
        elif len(high_flags) >= 2:
            r.risk_level = "HIGH"
        elif flags:
            r.risk_level = "MODERATE"
        else:
            r.risk_level = "LOW"

        r.interpretation = (
            f"DSO: {r.dso:.0f}d (Δ{r.dso_change:+.0f}d) | "
            f"DIO: {r.dio:.0f}d (Δ{r.dio_change:+.0f}d) | "
            f"DPO: {r.dpo:.0f}d (Δ{r.dpo_change:+.0f}d) | "
            f"CCC: {r.ccc:.0f}d (Δ{r.ccc_change:+.0f}d)"
        )
        return r
