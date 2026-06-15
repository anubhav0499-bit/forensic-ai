"""
LangGraph Orchestration Package
Implements the 12-step Agentic RAG loop from the architecture diagram.

Imports are lazy — if langgraph is not installed the package can still be
imported and the ImportError is only raised when you actually call a function.
"""

from .state import AgenticRAGState


def build_agentic_rag_graph():
    from .workflow import build_agentic_rag_graph as _build
    return _build()


def get_agentic_rag_graph():
    from .workflow import get_agentic_rag_graph as _get
    return _get()


def run_agentic_rag(*args, **kwargs):
    from .workflow import run_agentic_rag as _run
    return _run(*args, **kwargs)


__all__ = [
    "AgenticRAGState",
    "build_agentic_rag_graph",
    "get_agentic_rag_graph",
    "run_agentic_rag",
]
