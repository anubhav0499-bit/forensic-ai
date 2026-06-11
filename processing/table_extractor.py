"""
Table Extractor - Financial table extraction with Camelot and Tabula
Specialized for income statements, balance sheets, and cash flow statements
"""

from __future__ import annotations
import re
import pandas as pd
from pathlib import Path
from typing import Optional
from loguru import logger

from utils.helpers import safe_divide, detect_currency_unit
from config import PROCESSING_CONFIG


class TableExtractor:
    """
    Extract and parse financial tables from PDFs.
    Cascade: Camelot (lattice) → Camelot (stream) → Tabula → pdfplumber
    """

    # Key financial metrics to look for in table rows
    INCOME_STATEMENT_ROWS = [
        "revenue", "sales", "turnover", "net revenue", "total revenue",
        "cost of goods sold", "cogs", "cost of revenue", "cost of sales",
        "gross profit", "gross margin",
        "ebitda", "operating profit", "ebit", "operating income",
        "depreciation", "amortization",
        "profit before tax", "pbt", "pretax income",
        "tax", "income tax",
        "profit after tax", "pat", "net income", "net profit",
        "earnings per share", "eps", "basic eps", "diluted eps",
    ]

    BALANCE_SHEET_ROWS = [
        "total assets", "current assets", "non-current assets",
        "cash", "cash and cash equivalents", "short term investments",
        "accounts receivable", "trade receivables", "debtors",
        "inventories", "inventory", "stock",
        "property plant equipment", "ppe", "fixed assets",
        "goodwill", "intangible assets",
        "total liabilities", "current liabilities",
        "accounts payable", "trade payables", "creditors",
        "short term debt", "long term debt", "borrowings",
        "shareholders equity", "stockholders equity", "net worth",
        "retained earnings", "reserves",
    ]

    CASH_FLOW_ROWS = [
        "cash from operations", "operating activities",
        "capital expenditure", "capex",
        "free cash flow",
        "cash from investing", "investing activities",
        "cash from financing", "financing activities",
        "net change in cash", "opening cash", "closing cash",
        "dividends paid",
    ]

    def __init__(self):
        self._camelot_available = self._check_camelot()
        self._tabula_available = self._check_tabula()

    def _check_camelot(self) -> bool:
        try:
            import camelot
            return True
        except ImportError:
            return False

    def _check_tabula(self) -> bool:
        try:
            import tabula
            return True
        except ImportError:
            return False

    def extract_from_pdf(self, pdf_path: Path) -> dict:
        """
        Extract all financial tables from a PDF.
        Returns dict with income_statement, balance_sheet, cash_flow.
        """
        result = {
            "income_statement": None,
            "balance_sheet": None,
            "cash_flow": None,
            "all_tables": [],
        }

        logger.info(f"Extracting tables from: {pdf_path.name}")

        # Try Camelot (best for tables with borders)
        if self._camelot_available:
            tables = self._extract_camelot(pdf_path)
            if tables:
                result["all_tables"].extend(tables)

        # Try Tabula if Camelot failed or got fewer tables
        if self._tabula_available and len(result["all_tables"]) < 3:
            tabula_tables = self._extract_tabula(pdf_path)
            result["all_tables"].extend(tabula_tables)

        # Classify tables into financial statement types
        for table in result["all_tables"]:
            if table is None or table.empty:
                continue
            stmt_type = self._classify_table(table)
            if stmt_type and result[stmt_type] is None:
                result[stmt_type] = table

        return result

    def _extract_camelot(self, pdf_path: Path) -> list[pd.DataFrame]:
        tables = []
        try:
            import camelot
            # Try lattice mode first (tables with borders)
            camelot_tables = camelot.read_pdf(
                str(pdf_path),
                flavor="lattice",
                pages="all",
                line_scale=PROCESSING_CONFIG.lattice_line_scale,
            )
            for t in camelot_tables:
                if t.accuracy > 70:
                    tables.append(t.df)

            # Also try stream mode
            if len(tables) < 3:
                stream_tables = camelot.read_pdf(
                    str(pdf_path),
                    flavor="stream",
                    pages="all",
                    edge_tol=50,
                )
                for t in stream_tables:
                    if t.accuracy > 60:
                        tables.append(t.df)

            logger.debug(f"Camelot extracted {len(tables)} tables")
        except Exception as e:
            logger.debug(f"Camelot extraction failed: {e}")
        return tables

    def _extract_tabula(self, pdf_path: Path) -> list[pd.DataFrame]:
        tables = []
        try:
            import tabula
            dfs = tabula.read_pdf(
                str(pdf_path),
                pages="all",
                multiple_tables=True,
                silent=True,
            )
            tables = [df for df in dfs if not df.empty and len(df) > 2]
            logger.debug(f"Tabula extracted {len(tables)} tables")
        except Exception as e:
            logger.debug(f"Tabula extraction failed: {e}")
        return tables

    def _classify_table(self, df: pd.DataFrame) -> Optional[str]:
        """Classify a table as income_statement, balance_sheet, or cash_flow."""
        if df is None or df.empty:
            return None

        # Convert all cells to lowercase string
        text = " ".join(str(v).lower() for v in df.values.flatten() if v)

        income_hits = sum(1 for kw in ["revenue", "profit", "ebitda", "eps", "pat"] if kw in text)
        bs_hits = sum(1 for kw in ["total assets", "equity", "liabilities", "receivable", "inventory"] if kw in text)
        cf_hits = sum(1 for kw in ["cash from", "operating activities", "capex", "investing"] if kw in text)

        max_hits = max(income_hits, bs_hits, cf_hits)
        if max_hits == 0:
            return None

        if income_hits == max_hits:
            return "income_statement"
        elif bs_hits == max_hits:
            return "balance_sheet"
        else:
            return "cash_flow"

    def parse_financial_table(self, df: pd.DataFrame, statement_type: str) -> dict:
        """
        Parse a financial table DataFrame into a structured dict.
        Handles multi-year columns.
        Returns: {year: {metric: value}}
        """
        if df is None or df.empty:
            return {}

        result = {}

        # Try to identify year columns
        header_row = df.iloc[0] if len(df) > 0 else pd.Series()
        year_pattern = re.compile(r"20\d{2}|FY\s*\d{2,4}|Q[1-4]", re.I)

        # Identify which columns contain years
        year_cols = {}
        for col_idx, val in enumerate(df.columns):
            val_str = str(val)
            if year_pattern.search(val_str):
                year = re.search(r"20\d{2}", val_str)
                if year:
                    year_cols[col_idx] = year.group()

        # Also check first row
        for col_idx in range(len(df.columns)):
            val_str = str(df.iloc[0, col_idx] if len(df) > 0 else "")
            if year_pattern.search(val_str):
                year = re.search(r"20\d{2}", val_str)
                if year:
                    year_cols[col_idx] = year.group()

        if not year_cols:
            # Use column positions as fallback years
            years_guess = ["2024", "2023", "2022", "2021", "2020"]
            for i, col in enumerate(df.columns[1:], 1):
                if i - 1 < len(years_guess):
                    year_cols[i] = years_guess[i - 1]

        # Initialize year dicts
        for year in year_cols.values():
            result[year] = {}

        # Parse each row
        metric_rows = self.INCOME_STATEMENT_ROWS if statement_type == "income_statement" else (
            self.BALANCE_SHEET_ROWS if statement_type == "balance_sheet" else self.CASH_FLOW_ROWS
        )

        for _, row in df.iterrows():
            row_label = str(row.iloc[0]).lower().strip() if len(row) > 0 else ""
            if not row_label:
                continue

            # Find matching metric
            matched_metric = self._match_metric(row_label, metric_rows)
            if not matched_metric:
                continue

            # Extract values for each year
            for col_idx, year in year_cols.items():
                try:
                    val_str = str(row.iloc[col_idx]).replace(",", "").replace("(", "-").replace(")", "")
                    value = float(val_str)
                    result[year][matched_metric] = value
                except (ValueError, IndexError):
                    pass

        return result

    def _match_metric(self, row_label: str, metric_list: list) -> Optional[str]:
        """Fuzzy match a row label to a known financial metric."""
        from fuzzywuzzy import fuzz
        best_score = 0
        best_metric = None
        for metric in metric_list:
            score = fuzz.partial_ratio(row_label.lower(), metric.lower())
            if score > best_score and score > 75:
                best_score = score
                best_metric = metric.replace(" ", "_")
        return best_metric

    def merge_financial_data(self, tables_result: dict) -> dict:
        """Merge all extracted tables into a unified financial dict by year."""
        unified = {}

        for stmt_type, df in tables_result.items():
            if stmt_type == "all_tables" or df is None:
                continue
            parsed = self.parse_financial_table(df, stmt_type)
            for year, metrics in parsed.items():
                if year not in unified:
                    unified[year] = {}
                unified[year].update(metrics)

        return unified
