"""
LangGraph Node Functions — Agentic RAG 12-step loop.

Each function takes the current AgenticRAGState and returns a dict of
updated fields (LangGraph merges it with the existing state).

Step mapping:
  query_rewriter_node    → Steps 1-2
  detail_checker_node    → Step 3
  source_router_node     → Step 5
  retriever_node         → Steps 6-7
  generator_node         → Steps 8-9
  relevance_checker_node → Steps 10-12
"""

from __future__ import annotations

import re
from loguru import logger

from .state import AgenticRAGState


# ── Step 1-2: Query Rewriter ────────────────────────────────────────────

def query_rewriter_node(state: AgenticRAGState) -> dict:
    """
    Rewrite the original query for better retrieval precision.
    On retries, reformulates using synonyms / different angles to escape
    the vocabulary mismatch that caused the relevance check to fail.
    """
    from llm.langchain_client import lc_invoke

    company   = state.get("company_name", "")
    original  = state.get("original_query", "")
    history   = state.get("query_history", [])
    iteration = state.get("iteration", 0)

    if not history:
        prompt = (
            f"You are a forensic financial analyst. Rewrite this query to be "
            f"more specific and retrieval-optimised for financial document analysis.\n\n"
            f"Company: {company}\n"
            f"Original query: {original}\n\n"
            f"Rewrite rules:\n"
            f"1. Include specific financial metrics (e.g. 'revenue recognition' → "
            f"'revenue recognition policy, deferred revenue, contract assets')\n"
            f"2. Add relevant frameworks (ISA 240, SEBI, PCAOB, Beneish)\n"
            f"3. Include temporal context (FY2023, FY2022)\n"
            f"4. Decompose compound topics into key sub-terms\n\n"
            f"Return ONLY the rewritten query, no explanation."
        )
    else:
        recent = history[-2:]
        prompt = (
            f"Previous retrieval queries failed to return relevant results:\n"
            f"{chr(10).join(f'  {i+1}. {q}' for i, q in enumerate(recent))}\n\n"
            f"Original query: {original}\n"
            f"Company: {company}\n\n"
            f"Reformulate using different terminology, synonyms, or a different "
            f"analytical angle that might match different document sections.\n"
            f"Return ONLY the new query, no explanation."
        )

    rewritten = lc_invoke(prompt, fast=True).strip()
    logger.info(f"[QueryRewriter] iteration={iteration+1} → '{rewritten[:80]}…'")

    return {
        "rewritten_query": rewritten,
        "query_history":   history + [rewritten],
        "iteration":       iteration + 1,
    }


# ── Step 3: Detail Checker ───────────────────────────────────────────────

def detail_checker_node(state: AgenticRAGState) -> dict:
    """
    Decide whether additional context retrieval is needed.
    First iteration with no context always returns True.
    Subsequent iterations ask the LLM whether existing context is sufficient.
    """
    from llm.langchain_client import lc_invoke

    context   = state.get("retrieved_context", "")
    query     = state.get("rewritten_query") or state.get("original_query", "")
    iteration = state.get("iteration", 1)

    # Always retrieve on the first pass
    if not context or len(context) < 200:
        return {"needs_more_details": True}

    # On retries ask the LLM
    prompt = (
        f"Assess whether the retrieved context is sufficient to answer the forensic query.\n\n"
        f"Query: {query}\n\n"
        f"Retrieved context (first 500 chars):\n{context[:500]}\n\n"
        f"Reply with exactly one word: YES (need more retrieval) or NO (context is sufficient)."
    )
    answer = lc_invoke(prompt, fast=True).strip().upper()
    needs_more = "YES" in answer and "NO" not in answer[:3]
    logger.debug(f"[DetailChecker] needs_more={needs_more}")
    return {"needs_more_details": needs_more}


# ── Step 5: Source Router ────────────────────────────────────────────────

def source_router_node(state: AgenticRAGState) -> dict:
    """
    Select which data sources to query: vector_db, internet, tools_api.
    Uses a fast LLM call to decide based on the query and company name.
    """
    from llm.langchain_client import lc_invoke

    query   = state.get("rewritten_query") or state.get("original_query", "")
    company = state.get("company_name", "")

    prompt = (
        f"You are a source-routing agent for forensic financial research.\n\n"
        f"Company: {company}\n"
        f"Query: {query}\n\n"
        f"Available sources:\n"
        f"  vector_db  — indexed annual reports, filings, financial statements (already loaded)\n"
        f"  internet   — real-time news, regulatory orders, analyst reports\n"
        f"  tools_api  — live financial data: stock price, ratios, market cap via yfinance\n\n"
        f"Reply with a comma-separated list of the sources that would best answer this query.\n"
        f"Example: 'vector_db, internet'"
    )
    answer  = lc_invoke(prompt, fast=True).lower()
    sources = []
    if "vector_db" in answer:
        sources.append("vector_db")
    if "internet" in answer:
        sources.append("internet")
    if "tools_api" in answer:
        sources.append("tools_api")
    if not sources:
        sources = ["vector_db"]

    logger.info(f"[SourceRouter] sources={sources}")
    return {"selected_sources": sources}


# ── Steps 6-7: Retriever ─────────────────────────────────────────────────

def retriever_node(state: AgenticRAGState) -> dict:
    """
    Fetch context from each selected source and combine into retrieved_context.
    Each source section is clearly labelled so the generator can cite them.
    """
    company  = state.get("company_name", "")
    query    = state.get("rewritten_query") or state.get("original_query", "")
    sources  = state.get("selected_sources", ["vector_db"])
    fin_data = state.get("financial_data", {})

    context_parts: list[str] = []
    citations: list[str]     = []

    # ── Vector DB (hybrid retriever passed from base_agent) ───────────
    if "vector_db" in sources:
        retriever = state.get("retriever")
        if retriever is not None:
            try:
                results = retriever.search(company_name=company, query=query, n_results=8)
                texts = [
                    r.get("content", r.get("text", "")).strip()
                    for r in results
                    if r.get("content", r.get("text", "")).strip()
                ]
                if texts:
                    context_parts.append("[INDEXED DOCUMENTS]\n" + "\n\n".join(texts))
                    citations.extend(t[:100] for t in texts[:3])
            except Exception as exc:
                logger.warning(f"[Retriever] hybrid search failed: {exc}")
        else:
            # fallback: LlamaIndex when no retriever in state (standalone use)
            try:
                from agentic_rag.rag_pipeline import LlamaIndexRAGPipeline
                pipeline = LlamaIndexRAGPipeline(company_name=company)
                result   = pipeline.query(company, query)
                if result.response:
                    context_parts.append(f"[INDEXED DOCUMENTS]\n{result.response}")
                    citations.extend(
                        str(n.node.get_content()[:100]) for n in (result.source_nodes or [])[:3]
                    )
            except Exception as exc:
                logger.warning(f"[Retriever] vector_db failed: {exc}")

    # ── Internet search ────────────────────────────────────────────────
    if "internet" in sources:
        try:
            from agentic_rag.internet_search import InternetSearchTool
            searcher = InternetSearchTool()
            web      = searcher.search(f"{company} {query}")
            if web:
                context_parts.append(f"[INTERNET SEARCH]\n{web}")
        except Exception as exc:
            logger.warning(f"[Retriever] internet search failed: {exc}")

    # ── Live financial data (yfinance) ────────────────────────────────
    if "tools_api" in sources:
        try:
            context_parts.append(_fetch_live_financials(company, fin_data))
        except Exception as exc:
            logger.warning(f"[Retriever] tools_api failed: {exc}")

    combined = "\n\n".join(p for p in context_parts if p)
    if not combined:
        combined = "No additional context retrieved."

    return {"retrieved_context": combined, "source_citations": citations}


def _fetch_live_financials(company: str, fin_data: dict) -> str:
    """Try yfinance for real-time financial metadata."""
    import yfinance as yf
    # Guess ticker: use company_name as-is (works for plain tickers like INFY, AAPL)
    ticker = company.upper().split()[0]
    t      = yf.Ticker(ticker)
    info   = t.info or {}
    if not info.get("marketCap"):
        return ""
    lines = [
        f"[LIVE FINANCIAL DATA — {ticker}]",
        f"Market Cap:     {info.get('marketCap', 'N/A'):,}",
        f"Revenue (TTM):  {info.get('totalRevenue', 'N/A')}",
        f"P/E Ratio:      {info.get('trailingPE', 'N/A')}",
        f"Debt/Equity:    {info.get('debtToEquity', 'N/A')}",
        f"Current Ratio:  {info.get('currentRatio', 'N/A')}",
        f"Quick Ratio:    {info.get('quickRatio', 'N/A')}",
        f"Return on Eq:   {info.get('returnOnEquity', 'N/A')}",
        f"52-week high:   {info.get('fiftyTwoWeekHigh', 'N/A')}",
        f"52-week low:    {info.get('fiftyTwoWeekLow', 'N/A')}",
    ]
    return "\n".join(lines)


# ── Steps 8-9: Generator ─────────────────────────────────────────────────

def generator_node(state: AgenticRAGState) -> dict:
    """
    Build a forensic analysis prompt and call the primary (capable) LLM.
    Parses the response through OutputHarness for structured findings.
    """
    from llm.langchain_client import get_langchain_llm
    from llm.prompts import SYSTEM_PROMPTS, build_analysis_prompt
    from llm.output_harness import OutputHarness
    from langchain_core.messages import SystemMessage, HumanMessage

    llm      = get_langchain_llm(fast=False)
    harness  = OutputHarness()
    company  = state.get("company_name", "")
    query    = state.get("rewritten_query") or state.get("original_query", "")
    context  = state.get("retrieved_context", "")
    fin_data = state.get("financial_data", {})
    agent_nm = state.get("agent_name", "forensic_accountant")

    years    = sorted(fin_data.keys(), reverse=True)[:3]
    prompt   = build_analysis_prompt(
        agent_role=agent_nm,
        company_name=company,
        fiscal_years=years,
        financial_data={y: fin_data[y] for y in years},
        extracted_text=context,
        question=query,
        structured_output=True,
    )
    prompt += harness.structured_output_suffix()

    system   = SYSTEM_PROMPTS.get("forensic_accountant", "")
    messages = [SystemMessage(content=system), HumanMessage(content=prompt)]

    response     = llm.invoke(messages)
    response_txt = response.content if hasattr(response, "content") else str(response)

    harness_result = harness.extract(response_txt, company_name=company)
    findings = [
        {
            "flag_type":  f.flag_type,
            "risk_level": f.risk_level,
            "title":      f.title,
            "detail":     f.detail,
            "evidence":   f.evidence,
            "confidence": f.confidence,
        }
        for f in harness_result.findings
    ]
    risk = harness_result.extracted_risk_score or harness.estimate_risk_score(response_txt)

    logger.info(f"[Generator] response_words={len(response_txt.split())} risk={risk:.1f}")
    return {"response": response_txt, "findings": findings, "risk_score": risk}


# ── Steps 10-12: Relevance Checker ───────────────────────────────────────

def relevance_checker_node(state: AgenticRAGState) -> dict:
    """
    Evaluate whether the generated response adequately answers the original query.
    If not (and iterations remain), trigger a new query-rewrite cycle.
    """
    from llm.langchain_client import lc_invoke

    response  = state.get("response", "")
    query     = state.get("original_query", "")
    company   = state.get("company_name", "")
    iteration = state.get("iteration", 1)
    max_iter  = state.get("max_iterations", 3)

    # Trivially irrelevant: empty or very short response
    if len(response.split()) < 30:
        return {"is_relevant": False, "relevance_score": 0.0}

    # At max iterations, accept what we have
    if iteration >= max_iter:
        logger.info(f"[RelevanceChecker] max iterations reached — accepting response")
        return {"is_relevant": True, "relevance_score": 0.5, "final_response": response}

    prompt = (
        f"You are a quality-gate agent for forensic financial analysis.\n\n"
        f"Original query: {query}\n"
        f"Company: {company}\n\n"
        f"Response excerpt (first 600 chars):\n{response[:600]}\n\n"
        f"Rate relevance:\n"
        f"  RELEVANT — directly addresses the query with specific financial data or risk findings\n"
        f"  NOT_RELEVANT — generic, off-topic, or missing key evidence\n\n"
        f"Reply with: RELEVANT <score 0.0-1.0>  or  NOT_RELEVANT <score 0.0-1.0>\n"
        f"Example: 'RELEVANT 0.85'"
    )
    answer = lc_invoke(prompt, fast=True).strip()

    is_relevant  = "RELEVANT" in answer.upper() and "NOT_RELEVANT" not in answer.upper()
    score_match  = re.search(r"(\d+\.\d+)", answer)
    score        = float(score_match.group(1)) if score_match else (0.75 if is_relevant else 0.25)

    logger.info(f"[RelevanceChecker] is_relevant={is_relevant} score={score:.2f}")
    final = response if is_relevant else state.get("final_response", "")
    return {"is_relevant": is_relevant, "relevance_score": score, "final_response": final}
