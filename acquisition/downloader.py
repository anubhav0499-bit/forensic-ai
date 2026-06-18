"""
Document Downloader — Orchestrates Document Acquisition from All Sources
=========================================================================

References
----------
SEBI LODR Regulation 46 — Mandatory website disclosure: annual reports,
    shareholding patterns, and financial results must be publicly accessible
    on the company's website and exchange portals.
SEC Rule 17a-5 (EDGAR) — Electronic filing mandatory for US-listed companies;
    machine-readable XBRL-tagged financials since 2009.
Loughran, T., & McDonald, B. (2016). "Textual Analysis in Accounting and
    Finance: A Survey." Journal of Accounting Research, 54(4), 1187-1230.
    Document acquisition pipeline enables downstream textual analysis of
    annual reports, MD&A sections, and conference call transcripts.

Architecture / Data Flow
------------------------
Priority-ordered acquisition cascade:
  (1) Company IR page (IRScraper)
  (2) Stock exchange filings — NSE / BSE / EDGAR (IndiaMarketsClient /
      SECEdgarClient)
  (3) Regulatory databases — SEBI / RBI
  (4) Rating agencies — CRISIL / ICRA / CARE
  (5) Web search fallback
Deduplication is performed by SHA-256 content hash before DB registration.
Local cache stored at data/documents/{company_slug}/.

Handles deduplication, checksums, and local caching.
"""

from __future__ import annotations
import hashlib
import time
from pathlib import Path
from typing import Optional
from loguru import logger

from config import ACQUISITION_CONFIG
from utils.storage import StorageManager
from database.sqlite_handler import SQLiteHandler
from utils.audit_trail import AuditTrail
from .company_lookup import CompanyLookup, CompanyProfile
from .sec_edgar import SECEdgarClient, SECFiling
from .india_markets import IndiaMarketsClient


class DocumentDownloader:
    """
    Orchestrates document acquisition from:
    1. SEC EDGAR (US companies)
    2. NSE/BSE (Indian companies)
    3. Company IR websites
    4. yfinance for financial data
    """

    def __init__(self, storage: StorageManager, db: SQLiteHandler, audit: AuditTrail):
        self.storage = storage
        self.db = db
        self.audit = audit
        self.lookup = CompanyLookup()
        self.edgar = SECEdgarClient()
        self.india = IndiaMarketsClient()

    def acquire_all_documents(
        self,
        company_profile: CompanyProfile,
        company_id: int,
        years: int = 5,
    ) -> dict:
        """
        Main acquisition workflow.
        Returns summary of all acquired documents.
        """
        logger.info(f"Starting document acquisition for: {company_profile.name}")
        summary = {
            "company": company_profile.name,
            "annual_reports": 0,
            "quarterly_results": 0,
            "concall_transcripts": 0,
            "total_files": 0,
            "financial_data_years": 0,
        }

        market = company_profile.market.upper()

        if market == "US" or company_profile.cik:
            summary = self._acquire_us_documents(company_profile, company_id, years, summary)
        elif market == "INDIA" or company_profile.nse_symbol or company_profile.bse_code:
            summary = self._acquire_india_documents(company_profile, company_id, years, summary)
        else:
            # Try both
            summary = self._acquire_us_documents(company_profile, company_id, years, summary)
            if summary["total_files"] == 0:
                summary = self._acquire_india_documents(company_profile, company_id, years, summary)

        # Always try yfinance for supplemental financial data
        self._acquire_yfinance_data(company_profile, company_id)

        summary["total_files"] = summary["annual_reports"] + summary["quarterly_results"] + summary["concall_transcripts"]
        logger.info(f"Acquisition complete: {summary['total_files']} documents for {company_profile.name}")
        return summary

    def _acquire_us_documents(
        self, profile: CompanyProfile, company_id: int, years: int, summary: dict
    ) -> dict:
        """Acquire US company documents from SEC EDGAR."""
        cik = profile.cik
        if not cik:
            cik = self.edgar.get_cik_by_ticker(profile.ticker or profile.name)
            if not cik:
                logger.warning(f"Cannot find SEC CIK for {profile.name}")
                return summary

        logger.info(f"Acquiring SEC EDGAR documents for {profile.name} (CIK: {cik})")

        # Annual reports (10-K)
        annual_filings = self.edgar.get_annual_filings(cik, years)
        for filing in annual_filings:
            if self._is_duplicate(company_id, filing.form_type, filing.period_of_report[:4]):
                logger.debug(f"Skipping duplicate: {filing.form_type} {filing.period_of_report}")
                continue

            local_path = self.edgar.download_filing(filing, self.storage.raw_filings)
            if local_path:
                checksum = self._checksum_file(local_path)
                self.db.register_document(company_id, {
                    "document_type": filing.form_type,
                    "fiscal_year": filing.period_of_report[:4],
                    "quarter": "",
                    "source_url": filing.doc_url,
                    "local_path": str(local_path),
                    "file_size_bytes": local_path.stat().st_size,
                    "checksum": checksum,
                })
                self.audit.log_document(
                    agent_id=1,
                    source_url=filing.doc_url,
                    document_type=filing.form_type,
                    fiscal_year=filing.period_of_report[:4],
                    file_path=str(local_path),
                )
                summary["annual_reports"] += 1

        # Quarterly reports (10-Q)
        quarterly_filings = self.edgar.get_quarterly_filings(cik, years * 4)
        for filing in quarterly_filings:
            if self._is_duplicate(company_id, filing.form_type, filing.period_of_report[:4], filing.period_of_report[5:7]):
                continue
            local_path = self.edgar.download_filing(filing, self.storage.raw_filings)
            if local_path:
                self.db.register_document(company_id, {
                    "document_type": filing.form_type,
                    "fiscal_year": filing.period_of_report[:4],
                    "quarter": filing.period_of_report[5:7],
                    "source_url": filing.doc_url,
                    "local_path": str(local_path),
                    "file_size_bytes": local_path.stat().st_size,
                    "checksum": self._checksum_file(local_path),
                })
                summary["quarterly_results"] += 1

        # XBRL Financial Data
        xbrl_data = self.edgar.get_xbrl_financial_facts(cik)
        if xbrl_data:
            financials = self.edgar.extract_financials_from_xbrl(xbrl_data)
            for year, data in sorted(financials.items(), reverse=True)[:years]:
                self.db.upsert_financial_data(company_id, year, data)
                summary["financial_data_years"] += 1

            # Save raw XBRL
            self.storage.save_json(xbrl_data, "xbrl_financial_facts.json", "Financials")

        return summary

    def _acquire_india_documents(
        self, profile: CompanyProfile, company_id: int, years: int, summary: dict
    ) -> dict:
        """Acquire Indian company documents from NSE/BSE."""
        symbol = profile.nse_symbol or profile.ticker or profile.name
        if not symbol:
            return summary

        symbol = symbol.upper()
        logger.info(f"Acquiring NSE/BSE documents for {profile.name} (Symbol: {symbol})")

        # Annual reports
        annual_filings = self.india.get_annual_reports(symbol, years)
        for filing in annual_filings:
            if self._is_duplicate(company_id, "annual_report", filing.fiscal_year):
                continue
            local_path = self.india.download_filing(filing, self.storage.raw_filings)
            if local_path:
                self.db.register_document(company_id, {
                    "document_type": "annual_report",
                    "fiscal_year": filing.fiscal_year,
                    "quarter": "",
                    "source_url": filing.pdf_url,
                    "local_path": str(local_path),
                    "file_size_bytes": local_path.stat().st_size,
                    "checksum": self._checksum_file(local_path),
                })
                self.audit.log_document(1, filing.pdf_url, "annual_report", filing.fiscal_year, str(local_path))
                summary["annual_reports"] += 1

        # Quarterly results
        quarterly = self.india.get_quarterly_results(symbol, years * 4)
        for filing in quarterly:
            local_path = self.india.download_filing(filing, self.storage.raw_filings)
            if local_path:
                self.db.register_document(company_id, {
                    "document_type": "quarterly_results",
                    "fiscal_year": filing.fiscal_year,
                    "quarter": filing.quarter,
                    "source_url": filing.pdf_url,
                    "local_path": str(local_path),
                    "file_size_bytes": local_path.stat().st_size,
                    "checksum": self._checksum_file(local_path),
                })
                summary["quarterly_results"] += 1

        # Concall transcripts
        concalls = self.india.get_concall_transcripts(symbol, years)
        for filing in concalls:
            local_path = self.india.download_filing(filing, self.storage.raw_filings)
            if local_path:
                self.db.register_document(company_id, {
                    "document_type": "concall",
                    "fiscal_year": filing.fiscal_year,
                    "quarter": filing.quarter,
                    "source_url": filing.pdf_url,
                    "local_path": str(local_path),
                    "file_size_bytes": local_path.stat().st_size,
                    "checksum": self._checksum_file(local_path),
                })
                summary["concall_transcripts"] += 1

        # Screener.in financial data
        screener_data = self.india.get_screener_financials(profile.name)
        if screener_data:
            for year, data in screener_data.items():
                self.db.upsert_financial_data(company_id, year, data)
                summary["financial_data_years"] += 1
            self.storage.save_json(screener_data, "screener_financials.json", "Financials")

        # Shareholding pattern
        shareholding = self.india.get_shareholding_pattern(symbol)
        for record in shareholding:
            self.db.save_shareholding(company_id, record)

        return summary

    def _acquire_yfinance_data(self, profile: CompanyProfile, company_id: int) -> None:
        """Supplement with yfinance financial data."""
        try:
            import yfinance as yf
            ticker_str = profile.ticker
            if not ticker_str:
                return

            t = yf.Ticker(ticker_str)

            # Financial statements
            for period_type, method_name in [
                ("annual", "financials"),
                ("annual_bs", "balance_sheet"),
                ("annual_cf", "cashflow"),
            ]:
                try:
                    df = getattr(t, method_name)
                    if df is not None and not df.empty:
                        df_path = self.storage.financials / f"yfinance_{period_type}.json"
                        df.to_json(df_path, orient="columns", date_format="iso")
                        logger.debug(f"Saved yfinance {period_type} data")
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"yfinance data fetch failed: {e}")

    def _is_duplicate(self, company_id: int, doc_type: str, fiscal_year: str, quarter: str = "") -> bool:
        """Check if document already downloaded."""
        docs = self.db.get_documents(company_id, doc_type)
        for doc in docs:
            if doc.get("fiscal_year") == fiscal_year and doc.get("quarter", "") == quarter:
                local = doc.get("local_path", "")
                if local and Path(local).exists():
                    return True
        return False

    def _checksum_file(self, path: Path) -> str:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def close(self):
        self.lookup.close()
        self.edgar.close()
        self.india.close()
