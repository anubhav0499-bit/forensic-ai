"""
IR Website Scraper — scrapes company Investor Relations pages for annual
reports, earnings presentations, and governance documents.
"""

from __future__ import annotations
import re
import time
import hashlib
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import Optional
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup
from loguru import logger

from config import ACQUISITION_CONFIG


@dataclass
class IRDocument:
    url: str
    title: str
    doc_type: str          # annual_report | earnings_ppt | concall | governance
    year: Optional[int]
    source_page: str
    file_size_bytes: int = 0
    checksum: str = ""


_IR_PAGE_PATTERNS = [
    "/investor-relations",
    "/investors",
    "/ir",
    "/investor",
    "/financials",
    "/annual-report",
    "/corporate/investors",
    "/en/investors",
]

_DOC_KEYWORDS = {
    "annual_report": [
        "annual report", "annual-report", "annual_report",
        "10-k", "10k", "20-f", "form 20f",
    ],
    "earnings_ppt": [
        "earnings presentation", "investor presentation",
        "results presentation", "q4", "q3", "q2", "q1",
        "quarterly results", "analyst day",
    ],
    "concall": [
        "earnings call", "conference call", "concall",
        "transcript", "q&a",
    ],
    "governance": [
        "corporate governance", "proxy statement", "def 14a",
        "sustainability report", "esg report", "annual proxy",
    ],
}


class IRScraper:
    """Discovers and downloads documents from company IR websites."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; ForensicAI/1.0; research)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })
        self._rate_delay = 1.0 / ACQUISITION_CONFIG.rate_limit if ACQUISITION_CONFIG.rate_limit else 1.0

    # ─── Public API ────────────────────────────────────────────────────────

    def find_ir_page(self, company_website: str) -> Optional[str]:
        """Try common IR URL patterns to find the investor relations page."""
        base = company_website.rstrip("/")
        for pattern in _IR_PAGE_PATTERNS:
            candidate = base + pattern
            try:
                resp = self.session.get(candidate, timeout=10, allow_redirects=True)
                if resp.status_code == 200 and len(resp.text) > 1000:
                    logger.info(f"IR page found: {candidate}")
                    return resp.url
                time.sleep(self._rate_delay)
            except Exception:
                continue
        return None

    def discover_documents(self, ir_url: str, max_docs: int = 30) -> list[IRDocument]:
        """Scrape an IR page and return a list of downloadable documents."""
        docs: list[IRDocument] = []
        try:
            resp = self.session.get(ir_url, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"IR page fetch failed: {ir_url} — {e}")
            return docs

        soup = BeautifulSoup(resp.text, "html.parser")
        links = soup.find_all("a", href=True)

        for link in links:
            href = link["href"].strip()
            if not href or href.startswith("#"):
                continue

            full_url = urljoin(ir_url, href)
            text = (link.get_text(separator=" ") or "").strip().lower()

            doc_type = self._classify_link(text, full_url)
            if not doc_type:
                continue

            if not self._is_downloadable(full_url):
                continue

            year = self._extract_year(text + " " + full_url)
            title = link.get_text(separator=" ").strip()[:200] or Path(urlparse(full_url).path).stem

            doc = IRDocument(
                url=full_url,
                title=title,
                doc_type=doc_type,
                year=year,
                source_page=ir_url,
            )
            if not any(d.url == full_url for d in docs):
                docs.append(doc)

            if len(docs) >= max_docs:
                break

        logger.info(f"Discovered {len(docs)} IR documents at {ir_url}")
        return docs

    def download_document(self, doc: IRDocument, dest_dir: Path) -> Optional[Path]:
        """Download a single IR document. Returns the local path."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        filename = self._safe_filename(doc)
        dest_path = dest_dir / filename

        if dest_path.exists():
            logger.debug(f"Skip (exists): {filename}")
            return dest_path

        try:
            time.sleep(self._rate_delay)
            resp = self.session.get(doc.url, timeout=60, stream=True)
            resp.raise_for_status()

            data = b""
            for chunk in resp.iter_content(chunk_size=65536):
                data += chunk

            doc.file_size_bytes = len(data)
            doc.checksum = hashlib.md5(data).hexdigest()

            with open(dest_path, "wb") as f:
                f.write(data)

            logger.info(f"Downloaded IR doc ({doc.file_size_bytes//1024}KB): {filename}")
            return dest_path

        except Exception as e:
            logger.warning(f"IR download failed: {doc.url} — {e}")
            return None

    def scrape_ir_site(self, company_website: str, dest_dir: Path, max_docs: int = 30) -> list[Path]:
        """Full pipeline: find IR page → discover docs → download all. Returns saved paths."""
        ir_url = self.find_ir_page(company_website)
        if not ir_url:
            logger.info(f"No IR page found for: {company_website}")
            return []

        documents = self.discover_documents(ir_url, max_docs=max_docs)
        paths = []
        for doc in documents:
            path = self.download_document(doc, dest_dir / doc.doc_type)
            if path:
                paths.append(path)

        return paths

    # ─── Helpers ──────────────────────────────────────────────────────────

    def _classify_link(self, text: str, url: str) -> Optional[str]:
        combined = (text + " " + url).lower()
        for doc_type, keywords in _DOC_KEYWORDS.items():
            if any(kw in combined for kw in keywords):
                return doc_type
        return None

    def _is_downloadable(self, url: str) -> bool:
        path = urlparse(url).path.lower()
        allowed_exts = {".pdf", ".xlsx", ".xls", ".pptx", ".ppt", ".docx", ".doc", ".zip"}
        return any(path.endswith(ext) for ext in allowed_exts)

    def _extract_year(self, text: str) -> Optional[int]:
        matches = re.findall(r"\b(20\d{2}|19\d{2})\b", text)
        if matches:
            # Return most recent year found
            return max(int(m) for m in matches)
        return None

    def _safe_filename(self, doc: IRDocument) -> str:
        year_prefix = f"{doc.year}_" if doc.year else ""
        slug = re.sub(r"[^\w\-]+", "_", doc.title)[:60].strip("_")
        ext = Path(urlparse(doc.url).path).suffix or ".pdf"
        return f"{year_prefix}{slug}{ext}"
