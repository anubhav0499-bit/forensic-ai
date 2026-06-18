"""
PDF Processor — Extracts Text and Structure from Financial Documents
====================================================================

References
----------
Artifex Software (2006-2023). "PyMuPDF (fitz)." pymupdf.readthedocs.io.
    Fast native PDF text extraction; preserves layout and font metadata;
    best performance on born-digital annual report PDFs.
Kluyver, T., et al. (2022). "pdfplumber." github.com/jsvine/pdfplumber.
    Accurate text extraction with character / word / line bounding boxes;
    better for complex multi-column layouts and scanned-quality PDFs.
Smith, R. (2007). "An Overview of the Tesseract OCR Engine." ICDAR 2007.
    Tesseract v5 — primary OCR engine for image-PDF annual reports.
EasyOCR (JaidedAI, 2020). github.com/JaidedAI/EasyOCR.
    Secondary OCR fallback; better on mixed-language (English + Hindi)
    Indian annual reports where Tesseract accuracy degrades.

Architecture / Data Flow
------------------------
Extraction cascade logic:
  (1) PyMuPDF  — fast; best for born-digital PDFs (confidence > 0.7 → return)
  (2) pdfplumber — fallback for complex layouts (confidence > 0.5 → return)
  (3) OCR: Tesseract → EasyOCR — final fallback for scanned documents
Output: ParsedDocument with list of {page_num, text, tables} per page plus
aggregate word_count, tables_found, extraction_method, and confidence score.

Cascade: PyMuPDF → pdfplumber → OCR (Tesseract/EasyOCR)
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from loguru import logger

from config import PROCESSING_CONFIG


@dataclass
class ParsedDocument:
    file_path: str
    total_pages: int = 0
    text: str = ""
    pages: list[dict] = field(default_factory=list)   # [{page_num, text, tables}]
    metadata: dict = field(default_factory=dict)
    extraction_method: str = ""
    confidence: float = 0.0
    tables_found: int = 0
    word_count: int = 0


class PDFProcessor:
    """
    Multi-strategy PDF text extraction.
    For financial reports: prioritizes table accuracy over raw speed.
    """

    def __init__(self):
        self._pymupdf_available = self._check_pymupdf()
        self._pdfplumber_available = self._check_pdfplumber()
        self._ocr_available = self._check_ocr()

    def _check_pymupdf(self) -> bool:
        try:
            import fitz
            return True
        except ImportError:
            return False

    def _check_pdfplumber(self) -> bool:
        try:
            import pdfplumber
            return True
        except ImportError:
            return False

    def _check_ocr(self) -> bool:
        try:
            import pytesseract
            return True
        except ImportError:
            return False

    def process(self, pdf_path: Path) -> ParsedDocument:
        """Main processing pipeline."""
        doc = ParsedDocument(file_path=str(pdf_path))

        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return doc

        logger.info(f"Processing PDF: {pdf_path.name}")

        # Strategy 1: PyMuPDF (fastest, good for digital PDFs)
        if self._pymupdf_available:
            doc = self._extract_pymupdf(pdf_path, doc)
            if doc.confidence > 0.7:
                return self._post_process(doc)

        # Strategy 2: pdfplumber (better for complex layouts)
        if self._pdfplumber_available:
            doc = self._extract_pdfplumber(pdf_path, doc)
            if doc.confidence > 0.5:
                return self._post_process(doc)

        # Strategy 3: OCR (for scanned documents)
        if self._ocr_available and doc.word_count < 100:
            doc = self._extract_ocr(pdf_path, doc)

        return self._post_process(doc)

    def _extract_pymupdf(self, pdf_path: Path, doc: ParsedDocument) -> ParsedDocument:
        try:
            import fitz
            pdf = fitz.open(str(pdf_path))
            doc.total_pages = len(pdf)
            doc.metadata = pdf.metadata or {}

            pages = []
            all_text = []

            for page_num in range(len(pdf)):
                page = pdf[page_num]
                text = page.get_text("text")
                blocks = page.get_text("dict").get("blocks", [])

                # Extract tables using PyMuPDF
                tables = []
                try:
                    page_tabs = page.find_tables()
                    if page_tabs and page_tabs.tables:
                        for tab in page_tabs.tables:
                            tables.append(tab.extract())
                except Exception:
                    pass

                pages.append({
                    "page_num": page_num + 1,
                    "text": text,
                    "tables": tables,
                })
                all_text.append(text)

            doc.text = "\n".join(all_text)
            doc.pages = pages
            doc.word_count = len(doc.text.split())
            doc.tables_found = sum(len(p["tables"]) for p in pages)
            doc.extraction_method = "PyMuPDF"
            doc.confidence = min(1.0, doc.word_count / 500)

            pdf.close()
            logger.debug(f"PyMuPDF: {doc.word_count} words, {doc.tables_found} tables from {doc.total_pages} pages")
        except Exception as e:
            logger.warning(f"PyMuPDF failed: {e}")
        return doc

    def _extract_pdfplumber(self, pdf_path: Path, doc: ParsedDocument) -> ParsedDocument:
        try:
            import pdfplumber
            with pdfplumber.open(str(pdf_path)) as pdf:
                doc.total_pages = len(pdf.pages)
                pages = []
                all_text = []

                for page in pdf.pages:
                    text = page.extract_text() or ""
                    tables = page.extract_tables() or []

                    pages.append({
                        "page_num": page.page_number,
                        "text": text,
                        "tables": tables,
                    })
                    all_text.append(text)

                doc.text = "\n".join(all_text)
                doc.pages = pages
                doc.word_count = len(doc.text.split())
                doc.tables_found = sum(len(p["tables"]) for p in pages)
                doc.extraction_method = "pdfplumber"
                doc.confidence = min(1.0, doc.word_count / 500)

            logger.debug(f"pdfplumber: {doc.word_count} words, {doc.tables_found} tables")
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}")
        return doc

    def _extract_ocr(self, pdf_path: Path, doc: ParsedDocument) -> ParsedDocument:
        """OCR extraction for scanned PDFs."""
        try:
            import pytesseract
            from pdf2image import convert_from_path
            from PIL import Image

            logger.info(f"Applying OCR to {pdf_path.name}")
            images = convert_from_path(str(pdf_path), dpi=200)
            pages = []
            all_text = []

            for i, image in enumerate(images):
                text = pytesseract.image_to_string(
                    image,
                    lang=PROCESSING_CONFIG.ocr_language,
                    config="--psm 1",
                )
                confidence = self._get_ocr_confidence(image)
                pages.append({
                    "page_num": i + 1,
                    "text": text,
                    "tables": [],
                    "ocr_confidence": confidence,
                })
                all_text.append(text)

            doc.text = "\n".join(all_text)
            doc.pages = pages
            doc.word_count = len(doc.text.split())
            doc.extraction_method = "OCR-Tesseract"
            doc.confidence = 0.6
            logger.info(f"OCR complete: {doc.word_count} words")
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            # Try EasyOCR as last resort
            try:
                import easyocr
                reader = easyocr.Reader(["en"], gpu=False)
                from pdf2image import convert_from_path
                images = convert_from_path(str(pdf_path), dpi=150)
                all_text = []
                for image in images[:20]:  # Limit to 20 pages
                    results = reader.readtext(image, detail=0)
                    all_text.append(" ".join(results))
                doc.text = "\n".join(all_text)
                doc.word_count = len(doc.text.split())
                doc.extraction_method = "OCR-EasyOCR"
                doc.confidence = 0.55
            except Exception as e2:
                logger.error(f"All OCR methods failed: {e2}")
        return doc

    def _get_ocr_confidence(self, image) -> float:
        try:
            import pytesseract
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [c for c in data["conf"] if c > 0]
            return sum(confidences) / len(confidences) / 100 if confidences else 0.0
        except Exception:
            return 0.0

    def _post_process(self, doc: ParsedDocument) -> ParsedDocument:
        """Clean and normalize extracted text."""
        if doc.text:
            # Remove excessive whitespace
            doc.text = re.sub(r"\n{3,}", "\n\n", doc.text)
            doc.text = re.sub(r" {2,}", " ", doc.text)
            # Remove page numbers (common patterns)
            doc.text = re.sub(r"^\s*\d+\s*$", "", doc.text, flags=re.MULTILINE)
            doc.word_count = len(doc.text.split())
        return doc

    def extract_financial_sections(self, doc: ParsedDocument) -> dict:
        """
        Identify and extract key financial sections from document.
        Returns dict of section_name → text.
        """
        sections = {}
        text = doc.text

        section_patterns = {
            "income_statement": [
                r"(?:consolidated\s+)?(?:statement\s+of\s+)?(?:profit\s+(?:and|&)\s+loss|income\s+statement)",
                r"statement\s+of\s+operations",
            ],
            "balance_sheet": [
                r"(?:consolidated\s+)?(?:statement\s+of\s+)?(?:financial\s+position|balance\s+sheet)",
            ],
            "cash_flow": [
                r"(?:consolidated\s+)?(?:statement\s+of\s+)?cash\s+flow",
            ],
            "mda": [
                r"management(?:'s)?\s+discussion\s+(?:and\s+)?analysis",
                r"MD&A",
            ],
            "auditor_report": [
                r"(?:independent\s+)?(?:auditor(?:s)?(?:'s?)?\s+report|audit\s+report)",
            ],
            "related_party": [
                r"related\s+party\s+(?:transactions|disclosures)",
                r"related\s+parties",
            ],
            "notes": [
                r"notes?\s+to\s+(?:the\s+)?(?:financial\s+statements|accounts)",
            ],
        }

        for section_name, patterns in section_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    # Extract ~3000 chars from section start
                    start = match.start()
                    sections[section_name] = text[start:start + 5000]
                    break

        return sections
