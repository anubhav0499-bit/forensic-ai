"""
LangGraph State Schema — Agentic RAG 12-step loop.

Diagram mapping:
  Step 1  → query_rewriter_node   (rewrite query)
  Step 2  → rewritten_query field
  Step 3  → detail_checker_node   (need more details?)
  Step 4  → if No  → generator_node (direct LLM call)
  Step 5  → if Yes → source_router_node (pick sources)
  Step 6  → retriever_node        (vector DB / internet / tools)
  Step 7  → retrieved_context field
  Step 8  → generator_node prompt
  Step 9  → response field
  Step 10 → relevance_checker_node
  Step 11 → if Yes → END (final response)
  Step 12 → if No  → back to query_rewriter_node
"""

from __future__ import annotations
from typing import Any, TypedDict, List, Optional


class AgenticRAGState(TypedDict, total=False):
    """Shared mutable state carried through the LangGraph nodes."""

    # ── Input ──────────────────────────────────────────────────────────
    company_name: str
    agent_name: str
    agent_id: int
    original_query: str
    financial_data: dict          # {fiscal_year: {metric: value}}
    retriever: Any                # HybridRetriever instance passed from base_agent

    # ── Query rewriting (Steps 1-2) ────────────────────────────────────
    rewritten_query: str
    query_history: List[str]      # All rewrites so far (loop detection)

    # ── Routing (Steps 3, 5) ───────────────────────────────────────────
    needs_more_details: bool
    selected_sources: List[str]   # "vector_db" | "internet" | "tools_api"

    # ── Retrieval (Steps 6-7) ──────────────────────────────────────────
    retrieved_context: str
    source_citations: List[str]

    # ── Generation (Steps 8-9) ─────────────────────────────────────────
    response: str

    # ── Validation (Steps 10-12) ───────────────────────────────────────
    is_relevant: bool
    relevance_score: float
    iteration: int
    max_iterations: int           # Safety cap — default 3

    # ── Final output accumulators ──────────────────────────────────────
    final_response: str
    findings: List[dict]          # Parsed ParsedFinding dicts
    risk_score: float
