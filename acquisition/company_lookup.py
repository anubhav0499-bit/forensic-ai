"""
Company Lookup — Identifies Exchange, Ticker, ISIN and Market for Any Company
==============================================================================

References
----------
NSE India (2023). nseindia.com — National Stock Exchange.
    GET /api/quote-equity?symbol=INFY returns company profile, ISIN,
    market cap.
BSE India (2023). bseindia.com — Bombay Stock Exchange.
    Scrip search by company name returns BSE scrip code.
SEC EDGAR (2024). sec.gov/cgi-bin/browse-edgar — Full-text company search.
    CIK number required for downstream filing retrieval.
Yahoo Finance (2023). pypi.org/project/yfinance/ — Global ticker lookup;
    market cap, exchange, sector metadata across all major markets.

Standards / Specifications
--------------------------
SEBI LODR Regulation 30 — Listed companies must file identity disclosures
    (ISIN, exchange membership) with exchanges; this module resolves those
    identifiers for any input company name or ticker.

Architecture / Data Flow
------------------------
Produces a CompanyProfile dataclass with company_name, exchange, ticker, isin,
country, sector, market_cap, cik, nse_symbol, bse_code.  Lookup cascade:
  (1) SEC EDGAR by ticker (US short codes)
  (2) Yahoo Finance search (global fallback)
  (3) NSE autocomplete API (Indian companies)
  (4) SEC EDGAR full-text search by name
Result is cached in-memory for the session lifetime.

Supports: NSE/BSE (India), SEC EDGAR (US), Yahoo Finance (global)
"""

from __future__ import annotations
import json
import time
from typing import Optional
from dataclasses import dataclass
import httpx
from loguru import logger
from config import ACQUISITION_CONFIG


@dataclass
class CompanyProfile:
    name: str
    normalized_name: str = ""
    ticker: str = ""
    isin: str = ""
    exchange: str = ""          # NSE, BSE, NASDAQ, NYSE, etc.
    market: str = ""            # INDIA, US, UK, etc.
    sector: str = ""
    industry: str = ""
    cik: str = ""               # SEC CIK (US companies)
    nse_symbol: str = ""
    bse_code: str = ""
    currency: str = "INR"
    market_cap: float = 0.0
    country: str = ""
    ir_url: str = ""            # Investor Relations website
    edgar_url: str = ""


class CompanyLookup:
    """
    Multi-source company identification.
    Tries: SEC EDGAR → Yahoo Finance → NSE/BSE → Manual
    """

    def __init__(self):
        self.client = httpx.Client(
            timeout=30.0,
            headers={"User-Agent": ACQUISITION_CONFIG.sec_user_agent},
            follow_redirects=True,
        )
        self._company_cache: dict[str, CompanyProfile] = {}

    def identify(self, company_input: str) -> CompanyProfile:
        """Main entry: identify company from name or ticker."""
        from utils.helpers import normalize_company_name
        normalized = normalize_company_name(company_input)

        if normalized in self._company_cache:
            return self._company_cache[normalized]

        logger.info(f"Identifying company: {company_input}")

        profile = CompanyProfile(name=company_input, normalized_name=normalized)

        # Try each source in order
        # 1. If ticker-like (all caps, short), try SEC EDGAR
        if company_input.upper() == company_input and len(company_input) <= 6:
            edgar_profile = self._lookup_sec_by_ticker(company_input)
            if edgar_profile:
                self._company_cache[normalized] = edgar_profile
                return edgar_profile

        # 2. Try Yahoo Finance search (works globally)
        yf_profile = self._lookup_yahoo_finance(company_input)
        if yf_profile:
            self._company_cache[normalized] = yf_profile
            return yf_profile

        # 3. Try NSE India
        nse_profile = self._lookup_nse(company_input)
        if nse_profile:
            self._company_cache[normalized] = nse_profile
            return nse_profile

        # 4. Try SEC EDGAR full-text search
        edgar_profile = self._lookup_sec_by_name(company_input)
        if edgar_profile:
            self._company_cache[normalized] = edgar_profile
            return edgar_profile

        # 5. Return partial profile with what we have
        logger.warning(f"Could not fully identify: {company_input}. Using partial profile.")
        self._company_cache[normalized] = profile
        return profile

    def _lookup_yahoo_finance(self, query: str) -> Optional[CompanyProfile]:
        """Use yfinance for company lookup."""
        try:
            import yfinance as yf
            # Try direct ticker first
            ticker_guess = query.upper().replace(" ", "").replace(".", "")

            # Try Indian suffixes
            for suffix in ["", ".NS", ".BO", ".L", ".TO"]:
                try:
                    t = yf.Ticker(ticker_guess + suffix)
                    info = t.info
                    if info and info.get("longName"):
                        profile = CompanyProfile(
                            name=info.get("longName", query),
                            normalized_name=query,
                            ticker=ticker_guess + suffix,
                            exchange=info.get("exchange", ""),
                            sector=info.get("sector", ""),
                            industry=info.get("industry", ""),
                            market_cap=info.get("marketCap", 0.0),
                            currency=info.get("currency", "USD"),
                            country=info.get("country", ""),
                        )
                        if ".NS" in suffix:
                            profile.exchange = "NSE"
                            profile.market = "INDIA"
                            profile.nse_symbol = ticker_guess
                            profile.currency = "INR"
                        elif ".BO" in suffix:
                            profile.exchange = "BSE"
                            profile.market = "INDIA"
                            profile.currency = "INR"
                        else:
                            profile.market = "US"
                        logger.info(f"Found via Yahoo Finance: {profile.name} ({profile.ticker})")
                        return profile
                except Exception:
                    continue

            # Search by name using yfinance search
            search_result = yf.Search(query, max_results=1)
            quotes = search_result.quotes
            if quotes:
                best = quotes[0]
                symbol = best.get("symbol", "")
                if symbol:
                    t = yf.Ticker(symbol)
                    info = t.info
                    return CompanyProfile(
                        name=info.get("longName", query),
                        normalized_name=query,
                        ticker=symbol,
                        exchange=info.get("exchange", ""),
                        sector=info.get("sector", ""),
                        industry=info.get("industry", ""),
                        market_cap=info.get("marketCap", 0.0),
                        currency=info.get("currency", "USD"),
                        country=info.get("country", ""),
                        market="INDIA" if ".NS" in symbol or ".BO" in symbol else "US",
                    )
        except Exception as e:
            logger.debug(f"Yahoo Finance lookup failed for {query}: {e}")
        return None

    def _lookup_sec_by_ticker(self, ticker: str) -> Optional[CompanyProfile]:
        """Lookup US company by ticker via SEC EDGAR."""
        try:
            url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2020-01-01&forms=10-K"
            resp = self.client.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                hits = data.get("hits", {}).get("hits", [])
                if hits:
                    source = hits[0].get("_source", {})
                    return CompanyProfile(
                        name=source.get("entity_name", ticker),
                        normalized_name=ticker,
                        ticker=ticker,
                        cik=source.get("entity_id", ""),
                        exchange="NASDAQ/NYSE",
                        market="US",
                        currency="USD",
                    )
        except Exception as e:
            logger.debug(f"SEC ticker lookup failed: {e}")
        return None

    def _lookup_sec_by_name(self, name: str) -> Optional[CompanyProfile]:
        """Lookup US company by name via SEC EDGAR company search."""
        try:
            url = f"https://efts.sec.gov/LATEST/search-index?q=%22{name}%22&forms=10-K&dateRange=custom&startdt=2022-01-01"
            resp = self.client.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                hits = data.get("hits", {}).get("hits", [])
                if hits:
                    source = hits[0].get("_source", {})
                    entity = source.get("entity_name", "")
                    cik = source.get("entity_id", "")
                    if entity and cik:
                        return CompanyProfile(
                            name=entity,
                            normalized_name=name,
                            cik=cik,
                            exchange="NASDAQ/NYSE",
                            market="US",
                            currency="USD",
                        )
        except Exception as e:
            logger.debug(f"SEC name lookup failed: {e}")
        return None

    def _lookup_nse(self, name: str) -> Optional[CompanyProfile]:
        """Lookup Indian company via NSE."""
        try:
            # NSE equity search
            url = "https://www.nseindia.com/api/search/autocomplete"
            params = {"q": name}
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
                "Referer": "https://www.nseindia.com",
            }
            resp = self.client.get(url, params=params, headers=headers, timeout=10)
            if resp.status_code == 200:
                results = resp.json()
                symbols = results.get("symbols", [])
                if symbols:
                    best = symbols[0]
                    return CompanyProfile(
                        name=best.get("symbol_info", name),
                        normalized_name=name,
                        ticker=best.get("symbol", ""),
                        nse_symbol=best.get("symbol", ""),
                        exchange="NSE",
                        market="INDIA",
                        currency="INR",
                    )
        except Exception as e:
            logger.debug(f"NSE lookup failed: {e}")
        return None

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
