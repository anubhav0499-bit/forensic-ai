"""
Document Chunker - Semantic, recursive, and table-aware chunking for RAG pipeline.

Modes:
  DocumentChunker          — section-aware, sentence-boundary-snapped (original)
  RecursiveChunker         — hierarchical parent→child splitting for dense passages
  SemanticChunker          — embedding-similarity-guided splits (requires embedder)
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional
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

    # ── Recursive chunking convenience method ─────────────────────────

    def chunk_recursive(
        self,
        text: str,
        source: str,
        fiscal_year: str = "",
    ) -> list[DocumentChunk]:
        """
        Delegate to RecursiveChunker with the same source/fiscal_year metadata.
        Useful when callers want both modes without maintaining two instances.
        """
        rc = RecursiveChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return rc.chunk(text, source, fiscal_year)

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


# ─────────────────────────────────────────────────────────────────────────────
# Recursive Chunker
# ─────────────────────────────────────────────────────────────────────────────

class RecursiveChunker:
    """
    Hierarchical recursive text splitter that mirrors LangChain's
    RecursiveCharacterTextSplitter logic, adapted for financial documents.

    Split hierarchy (coarsest → finest):
      1. Double newline   (paragraph boundary)
      2. Single newline   (line boundary)
      3. Sentence end     (. ! ?)
      4. Word boundary    (space)

    When a segment at level N fits within chunk_size, it is emitted as a leaf
    chunk. When it exceeds chunk_size, it is recursively split at level N+1.
    This produces natural, semantically-coherent chunks without hard word cuts.

    Parent-child links:
      Each child chunk stores metadata["parent_id"] pointing to the parent
      chunk_id. This enables multi-vector retrieval of full parent context
      when a child chunk is matched.
    """

    _SEPARATORS = ["\n\n", "\n", r"(?<=[.!?])\s+", r"\s+"]

    def __init__(
        self,
        chunk_size: int = PROCESSING_CONFIG.chunk_size,
        chunk_overlap: int = PROCESSING_CONFIG.chunk_overlap,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        text: str,
        source: str,
        fiscal_year: str = "",
        parent_id: str = "",
        depth: int = 0,
    ) -> list[DocumentChunk]:
        """Recursively split text into semantically coherent chunks."""
        words = text.split()
        if len(words) <= self.chunk_size or depth >= len(self._SEPARATORS):
            # Leaf: emit as a single chunk
            if not words:
                return []
            cid = f"{source}_rc_{depth}_{abs(hash(text[:60])) % 100000}"
            return [DocumentChunk(
                chunk_id=cid,
                content=text.strip(),
                chunk_type="text",
                source_document=source,
                fiscal_year=fiscal_year,
                word_count=len(words),
                metadata={
                    "source":    source,
                    "fiscal_year": fiscal_year,
                    "depth":     depth,
                    "parent_id": parent_id,
                },
            )]

        # Split at the current separator level
        sep = self._SEPARATORS[depth]
        segments = [s.strip() for s in re.split(sep, text) if s.strip()]

        # Merge tiny segments up to chunk_size, then recurse on each merged block
        merged = self._merge_segments(segments)
        chunks: list[DocumentChunk] = []
        parent_cid = f"{source}_rc_{depth}_{abs(hash(text[:60])) % 100000}"

        for seg in merged:
            sub_chunks = self.chunk(
                seg, source, fiscal_year,
                parent_id=parent_cid,
                depth=depth + 1,
            )
            chunks.extend(sub_chunks)

        return chunks

    def _merge_segments(self, segments: list[str]) -> list[str]:
        """Merge consecutive small segments until they reach chunk_size."""
        merged: list[str] = []
        current_words = 0
        current_parts: list[str] = []

        for seg in segments:
            seg_words = len(seg.split())
            if current_words + seg_words > self.chunk_size and current_parts:
                merged.append(" ".join(current_parts))
                # overlap: keep last chunk_overlap words
                overlap_text = " ".join(current_parts[-1].split()[-self.chunk_overlap:])
                current_parts = [overlap_text] if overlap_text else []
                current_words = len(current_parts[0].split()) if current_parts else 0
            current_parts.append(seg)
            current_words += seg_words

        if current_parts:
            merged.append(" ".join(current_parts))
        return merged


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Chunker (embedding-guided splits)
# ─────────────────────────────────────────────────────────────────────────────

class SemanticChunker:
    """
    Embedding-similarity-guided chunking.
    Splits at sentence boundaries where the cosine similarity between
    adjacent sentences drops below a threshold, indicating a topic shift.

    Requires an EmbeddingModel instance. Falls back to RecursiveChunker
    if the embedding model is not available.
    """

    def __init__(
        self,
        embedder=None,
        similarity_threshold: float = 0.75,
        chunk_size: int = PROCESSING_CONFIG.chunk_size,
        chunk_overlap: int = PROCESSING_CONFIG.chunk_overlap,
    ):
        self.embedder = embedder
        self.threshold = similarity_threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._fallback = RecursiveChunker(chunk_size, chunk_overlap)

    def chunk(
        self,
        text: str,
        source: str,
        fiscal_year: str = "",
    ) -> list[DocumentChunk]:
        if self.embedder is None or not self.embedder.is_available():
            return self._fallback.chunk(text, source, fiscal_year)

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) < 2:
            return self._fallback.chunk(text, source, fiscal_year)

        # Compute embeddings for all sentences
        try:
            embeddings = self.embedder.encode(sentences)
        except Exception:
            return self._fallback.chunk(text, source, fiscal_year)

        # Find split points where similarity drops below threshold
        split_indices: list[int] = []
        for i in range(len(sentences) - 1):
            sim = self.embedder.similarity(embeddings[i], embeddings[i + 1])
            if sim < self.threshold:
                split_indices.append(i + 1)

        # Build segments from split points
        boundaries = [0] + split_indices + [len(sentences)]
        chunks: list[DocumentChunk] = []
        for seg_idx, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            seg_sentences = sentences[start:end]
            # Merge tiny segments with their neighbours
            if len(" ".join(seg_sentences).split()) < 30 and chunks:
                # Append to the last chunk
                last = chunks[-1]
                merged = last.content + " " + " ".join(seg_sentences)
                chunks[-1] = DocumentChunk(
                    chunk_id=last.chunk_id,
                    content=merged.strip(),
                    chunk_type=last.chunk_type,
                    source_document=source,
                    fiscal_year=fiscal_year,
                    word_count=len(merged.split()),
                    metadata=last.metadata,
                )
                continue

            content = " ".join(seg_sentences)
            cid = f"{source}_sem_{seg_idx}_{abs(hash(content[:60])) % 100000}"
            chunks.append(DocumentChunk(
                chunk_id=cid,
                content=content,
                chunk_type="text",
                source_document=source,
                fiscal_year=fiscal_year,
                word_count=len(content.split()),
                metadata={
                    "source":      source,
                    "fiscal_year": fiscal_year,
                    "seg_idx":     seg_idx,
                    "chunker":     "semantic",
                },
            ))

        return chunks
