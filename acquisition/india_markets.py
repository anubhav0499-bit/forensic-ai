"""
India Markets Client - NSE/BSE document acquisition
Sources: NSE EDGAR, BSE filing system, Screener.in
"""

from __future__ import annotations
import re
import time
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import httpx
from bs4 import BeautifulSoup
from loguru import logger
from config import ACQUISITION_CONFIG


@dataclass
class IndiaFiling:
    company_name: str
    symbol: str
    filing_type: str        # annual_report, quarterly, concall, etc.
    fiscal_year: str
    quarter: str = ""
    filing_date: str = ""
    pdf_url: str = ""
    source: str = "NSE"


class IndiaMarketsClient:
    """
    Document acquisition for NSE/BSE listed companies.
    Sources: NSE filing system, BSE Listing Center, Screener.in
    """

    NSE_BASE = "https://www.nseindia.com"
    BSE_BASE = "https://api.bseindia.com"
    SCREENER_BASE = "https://www.screener.in"

    def __init__(self):
        self.session = httpx.Client(
            timeout=ACQUISITION_CONFIG.download_timeout,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json, text/html,*/*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.nseindia.com",
            },
            follow_redirects=True,
        )
        self._rate_delay = 1.0 / ACQUISITION_CONFIG.rate_limit
        self._init_session()

    def _init_session(self):
        """Initialize NSE session (required for API access)."""
        try:
            self.session.get(f"{self.NSE_BASE}/", timeout=10)
            logger.debug("NSE session initialized")
        except Exception as e:
            logger.debug(f"NSE session init: {e}")

    def get_annual_reports(self, symbol: str, years: int = 5) -> list[IndiaFiling]:
        """Get NSE annual report links."""
        filings = []

        # Try NSE annual reports API
        try:
            url = f"{self.NSE_BASE}/api/annual-reports"
            params = {"index": "equities", "symbol": symbol.upper()}
            resp = self.session.get(url, params=params, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                for item in data[:years]:
                    filings.append(IndiaFiling(
                        company_name=symbol,
                        symbol=symbol,
                        filing_type="annual_report",
                        fiscal_year=str(item.get("fromYear", "")),
                        filing_date=item.get("date", ""),
                        pdf_url=item.get("fileName", ""),
                        source="NSE",
                    ))
        except Exception as e:
            logger.debug(f"NSE annual reports failed: {e}")

        # Try BSE
        if not filings:
            filings = self._get_bse_annual_reports(symbol, years)

        return filings

    def _get_bse_annual_reports(self, symbol: str, years: int = 5) -> list[IndiaFiling]:
        """Get BSE annual report links."""
        filings = []
        try:
            bse_code = self._get_bse_code(symbol)
            if not bse_code:
                return []

            url = f"{self.BSE_BASE}/BseIndiaAPI/api/AnnualReport/w"
            params = {"scripcode": bse_code, "flag": "C"}
            resp = self.session.get(url, params=params, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                for item in (data.get("Table", []) or [])[:years]:
                    filings.append(IndiaFiling(
                        company_name=symbol,
                        symbol=symbol,
                        filing_type="annual_report",
                        fiscal_year=str(item.get("YEAR", "")),
                        pdf_url=item.get("ANREP_ATTCH", ""),
                        source="BSE",
                    ))
        except Exception as e:
            logger.debug(f"BSE annual reports failed: {e}")
        return filings

    def get_quarterly_results(self, symbol: str, quarters: int = 8) -> list[IndiaFiling]:
        """Get NSE quarterly results PDFs."""
        filings = []
        try:
            url = f"{self.NSE_BASE}/api/corporates-financial-results"
            params = {"index": "equities", "symbol": symbol.upper(), "type": "Quarterly"}
            resp = self.session.get(url, params=params, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("data", [])[:quarters]
                for item in items:
                    filings.append(IndiaFiling(
                        company_name=symbol,
                        symbol=symbol,
                        filing_type="quarterly",
                        fiscal_year=item.get("period", ""),
                        quarter=item.get("period", ""),
                        filing_date=item.get("filingDate", ""),
                        pdf_url=item.get("attchmnt", ""),
                        source="NSE",
                    ))
        except Exception as e:
            logger.debug(f"NSE quarterly results failed: {e}")
        return filings

    def get_concall_transcripts(self, symbol: str, years: int = 5) -> list[IndiaFiling]:
        """Search for earnings call transcripts."""
        filings = []
        try:
            # Try NSE filings
            url = f"{self.NSE_BASE}/api/corporates-financial-results"
            params = {"index": "equities", "symbol": symbol.upper()}
            resp = self.session.get(url, params=params, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("data", []):
                    if "transcript" in str(item).lower() or "concall" in str(item).lower():
                        filings.append(IndiaFiling(
                            company_name=symbol,
                            symbol=symbol,
                            filing_type="concall",
                            fiscal_year=item.get("period", ""),
                            pdf_url=item.get("attchmnt", ""),
                            source="NSE",
                        ))
        except Exception as e:
            logger.debug(f"Concall transcript search failed: {e}")
        return filings

    def get_screener_financials(self, company_name: str) -> dict:
        """
        Scrape Screener.in for structured financial data.
        Screener provides 10-year financial history in a clean format.
        """
        financials = {}
        try:
            # Search for company
            search_url = f"{self.SCREENER_BASE}/api/company/search/?q={company_name}"
            resp = self.session.get(search_url, timeout=15)
            if resp.status_code != 200:
                return {}

            results = resp.json()
            if not results:
                return {}

            # Get first result
            company_url = results[0].get("url", "")
            if not company_url:
                return {}

            # Fetch company page
            time.sleep(self._rate_delay)
            page_url = f"{self.SCREENER_BASE}{company_url}"
            page_resp = self.session.get(page_url, timeout=20)
            if page_resp.status_code != 200:
                return {}

            financials = self._parse_screener_page(page_resp.text, company_name)
            logger.info(f"Scraped Screener.in for {company_name}: {len(financials)} years")
        except Exception as e:
            logger.debug(f"Screener.in scraping failed for {company_name}: {e}")
        return financials

    def _parse_screener_page(self, html: str, company_name: str) -> dict:
        """Parse Screener.in HTML to extract financial tables."""
        soup = BeautifulSoup(html, "lxml")
        financials = {}

        # Find consolidated financial tables
        tables = soup.find_all("table", {"class": re.compile("data-table|responsive-holder", re.I)})

        for table in tables:
            headers = [th.get_text(strip=True) for th in table.find_all("th")]
            rows = table.find_all("tr")

            for row in rows:
                cells = [td.get_text(strip=True).replace(",", "") for td in row.find_all("td")]
                if not cells or len(cells) < 2:
                    continue

                metric = cells[0].lower().strip()
                for i, header in enumerate(headers[1:], 1):
                    if i < len(cells) and cells[i]:
                        year = header
                        if year not in financials:
                            financials[year] = {}
                        try:
                            value = float(cells[i].replace(",", ""))
                            # Map metric names
                            metric_map = {
                                "sales": "revenue",
                                "net profit": "net_income",
                                "operating profit": "ebit",
                                "total assets": "total_assets",
                                "net cash flow": "cfo",
                                "eps in rs": "eps",
                            }
                            mapped = metric_map.get(metric, metric.replace(" ", "_"))
                            financials[year][mapped] = value
                        except (ValueError, TypeError):
                            pass
        return financials

    def get_shareholding_pattern(self, symbol: str, quarters: int = 8) -> list[dict]:
        """Get quarterly shareholding pattern."""
        patterns = []
        try:
            url = f"{self.NSE_BASE}/api/shareholding-patterns"
            params = {"symbol": symbol.upper(), "from": "2020-01-01"}
            resp = self.session.get(url, params=params, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("data", [])[:quarters]:
                    patterns.append({
                        "quarter": item.get("quarter", ""),
                        "promoter_holding": item.get("promoterAndPromoterGroupTotal", 0),
                        "fii_holding": item.get("fiiTotal", 0),
                        "dii_holding": item.get("diiTotal", 0),
                        "public_holding": item.get("publicTotal", 0),
                        "promoter_pledge": item.get("promoterPledge", 0),
                    })
        except Exception as e:
            logger.debug(f"Shareholding pattern failed: {e}")
        return patterns

    def _get_bse_code(self, nse_symbol: str) -> Optional[str]:
        """Get BSE code from NSE symbol."""
        try:
            url = f"{self.NSE_BASE}/api/quote-equity"
            params = {"symbol": nse_symbol.upper()}
            resp = self.session.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return str(data.get("metadata", {}).get("pdSectorPe", {}).get("bseCode", ""))
        except Exception:
            pass
        return None

    def download_filing(self, filing: IndiaFiling, save_path: Path) -> Optional[Path]:
        """Download an India filing PDF."""
        if not filing.pdf_url:
            return None
        try:
            time.sleep(self._rate_delay)
            url = filing.pdf_url
            if not url.startswith("http"):
                url = self.NSE_BASE + url

            resp = self.session.get(url, timeout=ACQUISITION_CONFIG.download_timeout)
            resp.raise_for_status()

            filename = f"{filing.filing_type}_{filing.fiscal_year}_{filing.quarter}.pdf"
            file_path = save_path / filename
            with open(file_path, "wb") as f:
                f.write(resp.content)

            logger.info(f"Downloaded: {filename} ({len(resp.content)//1024:.0f}KB)")
            return file_path
        except Exception as e:
            logger.error(f"Download failed for {filing.pdf_url}: {e}")
            return None

    def close(self):
        self.session.close()
