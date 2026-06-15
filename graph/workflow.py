"""
LangGraph Agentic RAG Workflow
================================
Implements the exact 12-step architecture diagram:

  [User Query]
       │
   ①  query_rewriter  ─────────────────────────────────────────┐
       │                                                         │
   ②  rewritten_query                                           │  (loop back
       │                                                         │   on NOT_RELEVANT)
   ③  detail_checker                                            │
       ├─ needs_more ──► ⑤ source_router ──► ⑥ retriever ─┐  │
       └─ sufficient ──────────────────────────────────────┘ │  │
                                                              │  │
   ⑦  retrieved_context + rewritten_query                    │  │
       │                                                      ◄──┘
   ⑧-⑨ generator  (LLM call with full prompt + context)
       │
   ⑩  relevance_checker
       ├─ RELEVANT ──────────────────────────────────► END (final response)
       └─ NOT_RELEVANT + iterations < max ──────────► ① query_rewriter

Integration points:
  - LlamaIndex : source_router → retriever → vector_db path
  - LangChain  : generator_node LLM call via langchain_client
  - LangGraph  : StateGraph with conditional edges for the loop
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import StateGraph, END

from .state import AgenticRAGState
from .nodes import (
    query_rewriter_node,
    detail_checker_node,
    source_router_node,
    retriever_node,
    generator_node,
    relevance_checker_node,
)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


# ── Routing functions ─────────────────────────────────────────────────────

def _route_detail_check(state: AgenticRAGState) -> str:
    """Step 3 → Step 5 (need details) or → Step 8 (sufficient context)."""
    return "source_router" if state.get("needs_more_details", True) else "generator"


def _route_relevance_check(state: AgenticRAGState) -> str:
    """Step 10 → END (relevant) or → Step 1 (retry)."""
    if state.get("is_relevant", False):
        return END
    iteration = state.get("iteration", 1)
    max_iter  = state.get("max_iterations", 3)
    if iteration >= max_iter:
        return END
    return "query_rewriter"


# ── Graph builder ─────────────────────────────────────────────────────────

def build_agentic_rag_graph() -> "CompiledStateGraph":
    """
    Compile the full Agentic RAG LangGraph StateGraph.

    Returns a compiled graph ready to call via .invoke() / .stream().
    """
    g = StateGraph(AgenticRAGState)

    # Nodes
    g.add_node("query_rewriter",     query_rewriter_node)      # Steps 1-2
    g.add_node("detail_checker",     detail_checker_node)      # Step 3
    g.add_node("source_router",      source_router_node)       # Step 5
    g.add_node("retriever",          retriever_node)           # Steps 6-7
    g.add_node("generator",          generator_node)           # Steps 8-9
    g.add_node("relevance_checker",  relevance_checker_node)   # Steps 10-12

    # Fixed edges
    g.add_edge("query_rewriter",    "detail_checker")
    g.add_edge("source_router",     "retriever")
    g.add_edge("retriever",         "generator")
    g.add_edge("generator",         "relevance_checker")

    # Conditional: detail check
    g.add_conditional_edges(
        "detail_checker",
        _route_detail_check,
        {"source_router": "source_router", "generator": "generator"},
    )

    # Conditional: relevance check (loop or END)
    g.add_conditional_edges(
        "relevance_checker",
        _route_relevance_check,
        {END: END, "query_rewriter": "query_rewriter"},
    )

    g.set_entry_point("query_rewriter")
    return g.compile()


# ── Singleton ─────────────────────────────────────────────────────────────

_GRAPH: "CompiledStateGraph | None" = None


def get_agentic_rag_graph() -> "CompiledStateGraph":
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_agentic_rag_graph()
    return _GRAPH


# ── Convenience runner ────────────────────────────────────────────────────

def run_agentic_rag(
    company_name: str,
    agent_name: str,
    agent_id: int,
    query: str,
    financial_data: dict | None = None,
    max_iterations: int = 3,
) -> AgenticRAGState:
    """
    Run the full 12-step Agentic RAG pipeline for one agent query.

    Args:
        company_name:   e.g. "Infosys Limited"
        agent_name:     e.g. "Revenue Quality Agent"
        agent_id:       integer agent number (1-17)
        query:          the forensic investigation question
        financial_data: {fiscal_year: {metric: value}} dict
        max_iterations: maximum rewrite-retrieve-generate cycles (default 3)

    Returns:
        Final AgenticRAGState with 'final_response', 'findings', 'risk_score'.
    """
    graph = get_agentic_rag_graph()

    initial: AgenticRAGState = {
        "company_name":     company_name,
        "agent_name":       agent_name,
        "agent_id":         agent_id,
        "original_query":   query,
        "rewritten_query":  query,
        "financial_data":   financial_data or {},
        "query_history":    [],
        "needs_more_details": True,
        "selected_sources": [],
        "retrieved_context": "",
        "source_citations": [],
        "response":         "",
        "is_relevant":      False,
        "relevance_score":  0.0,
        "iteration":        0,
        "max_iterations":   max_iterations,
        "final_response":   "",
        "findings":         [],
        "risk_score":       50.0,
    }

    return graph.invoke(initial)
