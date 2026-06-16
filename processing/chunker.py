"""
Document Chunker - Semantic and table-aware chunking for RAG pipeline
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from config import PROCESSING_CONFIG


@dataclass
class DocumentChunk:
    chunk_id: str
    content: str
    chunk_type: str           # text, table, financial_statement, mda, audit
    source_document: str
    page_num: int = 0
    fiscal_year: str = ""
    section: str = ""
    metadata: dict = field(default_factory=dict)
    word_count: int = 0


class DocumentChunker:
    """
    Smart chunking that preserves:
    - Table integrity (tables never split mid-row)
    - Section context (each chunk knows its document section)
    - Financial statement coherence
    """

    SECTION_HEADERS = [
        "balance sheet", "income statement", "profit and loss",
        "cash flow", "notes to accounts", "management discussion",
        "auditors report", "corporate governance", "related party",
        "segment", "key audit matters", "significant accounting policies",
    ]

    def __init__(self):
        self.chunk_size = PROCESSING_CONFIG.chunk_size
        self.chunk_overlap = PROCESSING_CONFIG.chunk_overlap

    def chunk_document(self, text: str, source: str, fiscal_year: str = "") -> list[DocumentChunk]:
        """Main chunking entry point."""
        chunks = []

        # Split into sections first
        sections = self._split_into_sections(text)

        for section_name, section_text in sections.items():
            section_chunks = self._chunk_section(section_text, source, section_name, fiscal_year)
            chunks.extend(section_chunks)

        return chunks

    def _split_into_sections(self, text: str) -> dict:
        """Split document into named sections."""
        sections = {"general": ""}
        current_section = "general"
        lines = text.split("\n")

        for line in lines:
            line_lower = line.lower().strip()

            # Check if this line is a section header
            for header in self.SECTION_HEADERS:
                if header in line_lower and len(line_lower) < 100:
                    current_section = header.replace(" ", "_")
                    if current_section not in sections:
                        sections[current_section] = ""
                    break

            sections.setdefault(current_section, "")
            sections[current_section] += line + "\n"

        return {k: v for k, v in sections.items() if len(v.strip()) > 50}

    # Sentence-ending punctuation for boundary snapping
    _SENT_END = re.compile(r"[.!?][\"'\)\]]*\s*$")

    def _chunk_section(
        self,
        text: str,
        source: str,
        section: str,
        fiscal_year: str,
    ) -> list[DocumentChunk]:
        """Chunk a section with overlap, snapping boundaries to sentence ends."""
        chunks = []
        words = text.split()

        if not words:
            return chunks

        chunk_type = self._classify_section(section)
        step    = self.chunk_size * 2 if chunk_type == "table" else self.chunk_size
        overlap = self.chunk_overlap

        i = 0
        chunk_idx = 0
        while i < len(words):
            end = min(i + step, len(words))

            # Snap end boundary forward to the next sentence end (up to 30 words)
            if end < len(words):
                snap_limit = min(end + 30, len(words))
                for j in range(end, snap_limit):
                    if self._SENT_END.search(words[j]):
                        end = j + 1
                        break

            chunk_words = words[i:end]
            content = " ".join(chunk_words)

            chunks.append(DocumentChunk(
                chunk_id=f"{source}_{section}_{chunk_idx}",
                content=content,
                chunk_type=chunk_type,
                source_document=source,
                section=section,
                fiscal_year=fiscal_year,
                word_count=len(chunk_words),
                metadata={"section": section, "fiscal_year": fiscal_year},
            ))

            i += max(1, end - i - overlap)
            chunk_idx += 1

        return chunks

    def _classify_section(self, section: str) -> str:
        if any(s in section for s in ["balance", "income", "profit", "cash_flow"]):
            return "financial_statement"
        elif "management" in section or "mda" in section:
            return "mda"
        elif "auditor" in section or "audit" in section:
            return "audit"
        elif "related_party" in section:
            return "related_party"
        elif "governance" in section:
            return "governance"
        return "text"

    def chunk_table(self, table_data: list[list], source: str, fiscal_year: str) -> list[DocumentChunk]:
        """Convert table to string chunks, preserving row integrity."""
        if not table_data:
            return []

        # Convert table to text representation
        rows = []
        for row in table_data:
            rows.append(" | ".join(str(cell) for cell in row if cell))

        table_text = "\n".join(rows)

        return [DocumentChunk(
            chunk_id=f"{source}_table_{hash(table_text) % 10000}",
            content=table_text,
            chunk_type="table",
            source_document=source,
            fiscal_year=fiscal_year,
            word_count=len(table_text.split()),
            metadata={"table_rows": len(table_data)},
        )]
