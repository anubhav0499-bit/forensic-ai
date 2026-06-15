"""
Internet Search Tool for the Agentic RAG pipeline.

Provider cascade:
  1. Tavily          — best for analytical / financial queries; set TAVILY_API_KEY
  2. DuckDuckGo      — free, no API key required; uses langchain-community
  3. Serper/Google   — set SERPER_API_KEY for Google search via Serper.dev
  4. SerpAPI         — set SERPAPI_API_KEY
  5. httpx fallback  — direct HTTP fetch for well-known financial portals
"""

from __future__ import annotations

import os
from loguru import logger

# Financial-domain trust list used by Tavily
_FINANCIAL_DOMAINS = [
    "bseindia.com", "nseindia.com", "sebi.gov.in", "rbi.org.in",
    "sec.gov", "edgar.sec.gov",
    "moneycontrol.com", "economictimes.indiatimes.com", "livemint.com",
    "businessstandard.com", "reuters.com", "bloomberg.com",
    "ft.com", "wsj.com", "cnbc.com",
    "crisil.com", "icra.in", "careratings.com",
]


class InternetSearchTool:
    """
    Multi-provider internet search.  Gracefully falls back down the chain
    when a provider is unavailable or its API key is missing.
    """

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    # ── Public API ──────────────────────────────────────────────────────

    def search(self, query: str) -> str:
        """Search the internet; return formatted plain-text results."""
        for name, fn in self._providers():
            result = self._try(name, fn, query)
            if result:
                return result
        logger.warning("[InternetSearch] All providers failed — no results")
        return ""

    def search_financial(self, company: str, topics: list[str]) -> str:
        """
        Run targeted financial searches across multiple topics.
        topics: e.g. ["audit opinion", "SEBI order", "credit rating"]
        """
        parts = []
        for topic in topics[:3]:
            domain_hint = (
                "site:bseindia.com OR site:nseindia.com OR site:sebi.gov.in "
                "OR site:moneycontrol.com OR site:economictimes.indiatimes.com"
            )
            q = f"{company} {topic} {domain_hint}"
            result = self.search(q)
            if result:
                parts.append(f"[{topic.upper()}]\n{result}")
        return "\n\n".join(parts)

    # ── Provider cascade ────────────────────────────────────────────────

    def _providers(self):
        return [
            ("Tavily",     self._tavily),
            ("DuckDuckGo", self._duckduckgo),
            ("Serper",     self._serper),
            ("SerpAPI",    self._serpapi),
        ]

    def _try(self, name: str, fn, query: str) -> str:
        try:
            result = fn(query)
            if result:
                logger.debug(f"[InternetSearch] {name} returned results")
            return result
        except Exception as exc:
            logger.debug(f"[InternetSearch] {name} failed: {exc}")
            return ""

    # ── Tavily ──────────────────────────────────────────────────────────

    def _tavily(self, query: str) -> str:
        key = os.getenv("TAVILY_API_KEY", "")
        if not key:
            raise ValueError("TAVILY_API_KEY not set")
        from langchain_community.tools.tavily_search import TavilySearchResults
        tool = TavilySearchResults(
            max_results=self.max_results,
            tavily_api_key=key,
            search_depth="advanced",
            include_domains=_FINANCIAL_DOMAINS,
        )
        results = tool.invoke({"query": query})
        return self._format_list(results, "Tavily")

    # ── DuckDuckGo ──────────────────────────────────────────────────────

    def _duckduckgo(self, query: str) -> str:
        from langchain_community.tools import DuckDuckGoSearchRun
        tool = DuckDuckGoSearchRun()
        text = tool.run(query)
        return f"[DuckDuckGo]\n{text}" if text else ""

    # ── Serper ──────────────────────────────────────────────────────────

    def _serper(self, query: str) -> str:
        key = os.getenv("SERPER_API_KEY", "")
        if not key:
            raise ValueError("SERPER_API_KEY not set")
        from langchain_community.utilities import GoogleSerperAPIWrapper
        wrapper = GoogleSerperAPIWrapper(serper_api_key=key, k=self.max_results)
        result = wrapper.run(query)
        return f"[Serper/Google]\n{result}" if result else ""

    # ── SerpAPI ─────────────────────────────────────────────────────────

    def _serpapi(self, query: str) -> str:
        key = os.getenv("SERPAPI_API_KEY", "")
        if not key:
            raise ValueError("SERPAPI_API_KEY not set")
        from langchain_community.utilities import SerpAPIWrapper
        wrapper = SerpAPIWrapper(serpapi_api_key=key)
        result = wrapper.run(query)
        return f"[SerpAPI]\n{result}" if result else ""

    # ── Formatter ───────────────────────────────────────────────────────

    @staticmethod
    def _format_list(results, source: str) -> str:
        if not results:
            return ""
        if isinstance(results, str):
            return f"[{source}]\n{results}"
        parts = []
        for r in results[: 5]:
            if isinstance(r, dict):
                title   = r.get("title", "")
                url     = r.get("url", r.get("link", ""))
                snippet = r.get("content", r.get("snippet", ""))[:400]
                parts.append(f"• {title}\n  {url}\n  {snippet}")
            else:
                parts.append(str(r)[:300])
        return f"[{source}]\n" + "\n\n".join(parts)
