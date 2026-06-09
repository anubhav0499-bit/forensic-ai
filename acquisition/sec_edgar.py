"""
SEC EDGAR Client - Full document acquisition for US-listed companies
Free API: https://www.sec.gov/cgi-bin/browse-edgar
XBRL Financial Facts: https://data.sec.gov/api/xbrl/companyfacts/{CIK}.json
"""

from __future__ import annotations
import re
import time
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
import httpx
from loguru import logger
from config import ACQUISITION_CONFIG


@dataclass
class SECFiling:
    cik: str
    company_name: str
    form_type: str
    filing_date: str
    period_of_report: str
    accession_number: str
    primary_doc: str
    filing_url: str
    doc_url: str = ""


class SECEdgarClient:
    """
    Full SEC EDGAR API client.
    Downloads: 10-K, 10-Q, DEF 14A, 8-K filings.
    Extracts: XBRL financial data (structured financial statements).
    """

    BASE_URL = "https://data.sec.gov"
    SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
    COMPANY_SEARCH = "https://efts.sec.gov/LATEST/search-index"

    def __init__(self):
        self.client = httpx.Client(
            timeout=ACQUISITION_CONFIG.download_timeout,
            headers={
                "User-Agent": ACQUISITION_CONFIG.sec_user_agent,
                "Accept": "application/json",
            },
            follow_redirects=True,
        )
        self._rate_limit_delay = 1.0 / ACQUISITION_CONFIG.rate_limit

    def get_cik_by_ticker(self, ticker: str) -> Optional[str]:
        """Get SEC CIK number for a ticker."""
        try:
            resp = self.client.get(
                f"{self.BASE_URL}/submissions/CIK{ticker.upper()}.json",
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                return str(data.get("cik", "")).zfill(10)
        except Exception:
            pass

        # Try company search
        try:
            resp = self.client.get(
                f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms=10-K",
                timeout=10,
            )
            if resp.status_code == 200:
                hits = resp.json().get("hits", {}).get("hits", [])
                if hits:
                    cik = hits[0].get("_source", {}).get("entity_id", "")
                    return str(cik).zfill(10) if cik else None
        except Exception:
            pass
        return None

    def get_submissions(self, cik: str) -> dict:
        """Get full filing history for a company."""
        cik = str(cik).zfill(10)
        try:
            resp = self.client.get(f"{self.BASE_URL}/submissions/CIK{cik}.json", timeout=20)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to get submissions for CIK {cik}: {e}")
            return {}

    def get_annual_filings(self, cik: str, years: int = 5) -> list[SECFiling]:
        """Get 10-K filings for the last N years."""
        submissions = self.get_submissions(cik)
        if not submissions:
            return []

        filings = []
        company_name = submissions.get("name", "")
        recent = submissions.get("filings", {}).get("recent", {})

        if not recent:
            return []

        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        periods = recent.get("reportDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        for i, form in enumerate(forms):
            if form in ("10-K", "10-K/A", "20-F"):
                if len(filings) >= years:
                    break
                accession = accessions[i].replace("-", "")
                primary = primary_docs[i]
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/"
                doc_url = filing_url + primary

                filings.append(SECFiling(
                    cik=cik,
                    company_name=company_name,
                    form_type=form,
                    filing_date=dates[i],
                    period_of_report=periods[i] if i < len(periods) else "",
                    accession_number=accessions[i],
                    primary_doc=primary,
                    filing_url=filing_url,
                    doc_url=doc_url,
                ))

        logger.info(f"Found {len(filings)} annual filings for {company_name} (CIK: {cik})")
        return filings

    def get_quarterly_filings(self, cik: str, quarters: int = 8) -> list[SECFiling]:
        """Get 10-Q filings."""
        submissions = self.get_submissions(cik)
        if not submissions:
            return []

        filings = []
        company_name = submissions.get("name", "")
        recent = submissions.get("filings", {}).get("recent", {})
        if not recent:
            return []

        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        periods = recent.get("reportDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        for i, form in enumerate(forms):
            if form in ("10-Q", "10-Q/A"):
                if len(filings) >= quarters:
                    break
                accession = accessions[i].replace("-", "")
                primary = primary_docs[i]
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/"
                doc_url = filing_url + primary

                filings.append(SECFiling(
                    cik=cik,
                    company_name=company_name,
                    form_type=form,
                    filing_date=dates[i],
                    period_of_report=periods[i] if i < len(periods) else "",
                    accession_number=accessions[i],
                    primary_doc=primary,
                    filing_url=filing_url,
                    doc_url=doc_url,
                ))
        return filings

    def get_xbrl_financial_facts(self, cik: str) -> dict:
        """
        Get structured XBRL financial data from SEC.
        Returns complete financial history in JSON format.
        This is the most reliable source for US company financials.
        """
        cik = str(cik).zfill(10)
        url = f"{self.BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"
        try:
            time.sleep(self._rate_limit_delay)
            resp = self.client.get(url, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            logger.info(f"Retrieved XBRL facts for CIK {cik}: {len(data.get('facts', {}))} taxonomies")
            return data
        except Exception as e:
            logger.error(f"XBRL facts retrieval failed for CIK {cik}: {e}")
            return {}

    def extract_financials_from_xbrl(self, xbrl_data: dict) -> dict:
        """
        Parse XBRL data into a structured financial dictionary by year.
        Maps standard US-GAAP taxonomy to our financial model.
        """
        if not xbrl_data:
            return {}

        us_gaap = xbrl_data.get("facts", {}).get("us-gaap", {})
        dei = xbrl_data.get("facts", {}).get("dei", {})

        # Mapping: our field name → XBRL concept name
        field_mapping = {
            "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
            "cogs": ["CostOfGoodsSold", "CostOfRevenue", "CostOfGoodsAndServicesSold"],
            "gross_profit": ["GrossProfit"],
            "ebit": ["OperatingIncomeLoss"],
            "ebitda": ["EarningsBeforeInterestTaxesDepreciationAndAmortization"],
            "net_income": ["NetIncomeLoss", "ProfitLoss"],
            "eps": ["EarningsPerShareBasic"],
            "eps_diluted": ["EarningsPerShareDiluted"],
            "total_assets": ["Assets"],
            "current_assets": ["AssetsCurrent"],
            "cash_equivalents": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsAndShortTermInvestments"],
            "accounts_receivable": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
            "inventory": ["InventoryNet"],
            "ppe_net": ["PropertyPlantAndEquipmentNet"],
            "goodwill": ["Goodwill"],
            "total_liabilities": ["Liabilities"],
            "current_liabilities": ["LiabilitiesCurrent"],
            "accounts_payable": ["AccountsPayableCurrent"],
            "long_term_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
            "shareholder_equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
            "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
            "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
            "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsForCapitalImprovements"],
            "dividends_paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
            "shares_outstanding": ["CommonStockSharesOutstanding"],
            "depreciation": ["DepreciationDepletionAndAmortization", "Depreciation"],
            "interest_expense": ["InterestExpense"],
            "sga": ["SellingGeneralAndAdministrativeExpense"],
            "rd_expense": ["ResearchAndDevelopmentExpense"],
        }

        annual_data = {}

        for our_field, xbrl_concepts in field_mapping.items():
            for concept in xbrl_concepts:
                if concept in us_gaap:
                    units = us_gaap[concept].get("units", {})
                    # Get USD annual values
                    values = units.get("USD", units.get("shares", []))
                    for entry in values:
                        if entry.get("form") in ("10-K", "20-F") and entry.get("fp") == "FY":
                            year = str(entry.get("end", ""))[:4]
                            if year not in annual_data:
                                annual_data[year] = {}
                            if our_field not in annual_data[year]:  # Don't overwrite
                                annual_data[year][our_field] = entry.get("val", 0)
                    break  # Use first matching concept

        return annual_data

    def download_filing(self, filing: SECFiling, save_path: Path) -> Optional[Path]:
        """Download a filing document."""
        try:
            time.sleep(self._rate_limit_delay)
            resp = self.client.get(filing.doc_url, timeout=ACQUISITION_CONFIG.download_timeout)
            resp.raise_for_status()

            # Check file size
            content_length = len(resp.content)
            if content_length > ACQUISITION_CONFIG.max_file_size_mb * 1024 * 1024:
                logger.warning(f"File too large: {content_length/(1024*1024):.1f} MB, skipping")
                return None

            # Determine filename
            ext = ".pdf" if "pdf" in filing.primary_doc.lower() else ".htm"
            filename = f"{filing.form_type}_{filing.period_of_report}{ext}"
            file_path = save_path / filename

            with open(file_path, "wb") as f:
                f.write(resp.content)

            checksum = hashlib.md5(resp.content).hexdigest()
            logger.info(f"Downloaded: {filename} ({content_length/1024:.0f}KB, MD5: {checksum[:8]})")
            return file_path

        except Exception as e:
            logger.error(f"Failed to download {filing.doc_url}: {e}")
            return None

    def search_full_text(self, company_name: str, form_type: str = "10-K") -> list[dict]:
        """Full-text search EDGAR for filings."""
        try:
            params = {
                "q": f'"{company_name}"',
                "forms": form_type,
                "dateRange": "custom",
                "startdt": "2019-01-01",
            }
            resp = self.client.get(
                "https://efts.sec.gov/LATEST/search-index",
                params=params,
                timeout=20,
            )
            if resp.status_code == 200:
                return resp.json().get("hits", {}).get("hits", [])
        except Exception as e:
            logger.error(f"EDGAR search failed: {e}")
        return []

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
