"""
Dechow F-Score — Probability of Financial Misreporting
=======================================================

References
----------
Dechow, P. M., Ge, W., Larson, C. R., & Sloan, R. G. (2011).
    "Predicting Material Accounting Misstatements."
    Contemporary Accounting Research, 28(1), 17–82.
    Logistic model trained on SEC AAER sample 1982–2005.
    Formula:
      ln(p/1-p) = -7.893 + 0.790×RSST + 2.518×ΔAR + 1.191×ΔInv
                  + 1.979×SoftAssets + 0.171×ΔCashSales
                  - 0.932×ΔROA + 1.029×Issued
    Base rate ≈ 0.0037; F-Score > 0.025 → 6.7× base rate misstatement risk.

Richardson, S. A., Sloan, R. G., Soliman, M. T., & Tuna, I. (2005).
    "Accrual Reliability, Earnings Persistence, and Stock Prices."
    Journal of Accounting and Economics, 39(3), 437–485.
    RSST Accrual = ΔWorking Capital + ΔNon-current + ΔFinancial.

Audit Standards
---------------
PCAOB AS 2401 §66: "Fraud Risk Factors Relating to Misstatements
    Arising from Fraudulent Financial Reporting."
ISA 240 §A1–A6: Risk factors related to incentives/pressures.
Companies Act 2013 §143(12): Reporting of suspected fraud by auditors.

Uses financial statement characteristics to predict AAER (SEC enforcement actions).
Higher F-Score = Higher probability of misreporting.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from utils.helpers import safe_divide
import math


@dataclass
class DechowResult:
    f_score: float = 0.0
    raw_score: float = 0.0
    manipulation_likely: bool = False
    risk_level: str = ""
    interpretation: str = ""
    components: dict = field(default_factory=dict)


class DechowFScore:
    """
    Dechow F-Score based on the logistic regression model.
    Predicts probability of accounting misreporting (AAER).

    Variables:
    - rsst_accrual: Revenue & Inventory accrual
    - change_receivables: ΔAccounts Receivable / avg. assets
    - change_inventory: ΔInventory / avg. assets
    - soft_assets: Soft assets ratio (Total assets - Cash - PP&E) / Total assets
    - change_cash_sales: Change in cash sales ratio
    - change_roa: Change in ROA
    - issue: External financing indicator (issuing equity or debt)
    """

    # Logistic regression coefficients from Dechow et al. (2011)
    COEFFICIENTS = {
        "intercept": -7.893,
        "rsst_accrual": 0.790,
        "change_receivables": 2.518,
        "change_inventory": 1.191,
        "soft_assets": 1.979,
        "change_cash_sales": 0.171,
        "change_roa": -0.932,
        "issue": 3.129,
    }

    def calculate(
        self,
        # Balance sheet
        total_assets: float,
        total_assets_prev: float,
        cash: float,
        accounts_receivable: float,
        accounts_receivable_prev: float,
        inventory: float,
        inventory_prev: float,
        ppe_net: float,
        # Income
        revenue: float,
        revenue_prev: float,
        net_income: float,
        net_income_prev: float,
        cfo: float,
        # Financing
        shares_issued: bool = False,
        debt_issued: bool = False,
    ) -> DechowResult:
        result = DechowResult()
        avg_assets = (total_assets + total_assets_prev) / 2 if total_assets_prev else total_assets

        # ── RSST Accrual ──────────────────────────────────────
        # (ΔWorking Capital + ΔNon-current Operating Assets - ΔNon-current Operating Liabilities) / avg. assets
        # Simplified proxy: (Net Income - CFO) / avg. assets
        rsst_accrual = safe_divide(net_income - cfo, avg_assets)

        # ── Change in Receivables ─────────────────────────────
        change_receivables = safe_divide(accounts_receivable - accounts_receivable_prev, avg_assets)

        # ── Change in Inventory ───────────────────────────────
        change_inventory = safe_divide(inventory - inventory_prev, avg_assets)

        # ── Soft Assets ───────────────────────────────────────
        # (Total Assets - Cash - PP&E) / Total Assets
        # Higher = more intangibles/deferrals = more manipulable assets
        soft_assets = safe_divide(total_assets - cash - ppe_net, total_assets)

        # ── Change in Cash Sales ──────────────────────────────
        # Δ(Revenue - ΔAR) / avg. assets
        cash_sales_t = revenue - (accounts_receivable - accounts_receivable_prev)
        cash_sales_tm1 = revenue_prev
        change_cash_sales = safe_divide(cash_sales_t - cash_sales_tm1, avg_assets)

        # ── Change in ROA ─────────────────────────────────────
        roa_t = safe_divide(net_income, total_assets)
        roa_tm1 = safe_divide(net_income_prev, total_assets_prev)
        change_roa = roa_t - roa_tm1

        # ── External Financing Indicator ─────────────────────
        issue = 1.0 if (shares_issued or debt_issued) else 0.0

        # ── Logistic Score ────────────────────────────────────
        c = self.COEFFICIENTS
        linear_combo = (
            c["intercept"]
            + c["rsst_accrual"] * rsst_accrual
            + c["change_receivables"] * change_receivables
            + c["change_inventory"] * change_inventory
            + c["soft_assets"] * soft_assets
            + c["change_cash_sales"] * change_cash_sales
            + c["change_roa"] * change_roa
            + c["issue"] * issue
        )

        try:
            result.f_score = 1 / (1 + math.exp(-linear_combo))
        except OverflowError:
            result.f_score = 0.0 if linear_combo < 0 else 1.0

        result.raw_score = linear_combo

        # ── Interpretation ────────────────────────────────────
        # Dechow et al. used ~1% as the base rate of misreporting in the population
        # F-Score > 1% (0.01) means company is above average risk
        # F-Score > 0.10 is high risk (10x base rate)
        result.manipulation_likely = result.f_score > 0.01

        if result.f_score > 0.10:
            result.risk_level = "CRITICAL"
            result.interpretation = f"Dechow F-Score {result.f_score:.4f} (>{0.10:.2f}): EXTREME misreporting risk. 10x+ base rate."
        elif result.f_score > 0.025:
            result.risk_level = "HIGH"
            result.interpretation = f"Dechow F-Score {result.f_score:.4f}: HIGH misreporting probability. 2.5x+ base rate."
        elif result.f_score > 0.010:
            result.risk_level = "MODERATE"
            result.interpretation = f"Dechow F-Score {result.f_score:.4f}: Elevated misreporting probability. Above baseline."
        else:
            result.risk_level = "LOW"
            result.interpretation = f"Dechow F-Score {result.f_score:.4f}: Low misreporting probability. Below baseline rate."

        result.components = {
            "RSST Accrual": rsst_accrual,
            "Change in Receivables": change_receivables,
            "Change in Inventory": change_inventory,
            "Soft Assets Ratio": soft_assets,
            "Change in Cash Sales": change_cash_sales,
            "Change in ROA": change_roa,
            "External Financing": issue,
        }
        return result
