"""
Base Forensic Agent — Foundation class for all investigation agents
====================================================================

References
----------
ISA 315 (Revised 2019): Identifying and Assessing the Risks of Material
    Misstatement — §A200: "Understanding the Entity and Its Environment."
    Basis for Evidence → Analysis → Reasoning → Conclusion framework.

ISA 240: The Auditor's Responsibilities Relating to Fraud — §A1–A6:
    Fraud risk factors; used to structure red-flag detection logic.

PCAOB AS 2101: Audit Planning — systematic evidence-gathering approach
    mirrored in each agent's investigation() method.

Honovich, O., et al. (2022). "TRUE: Re-evaluating Factual Consistency
    Evaluation." NAACL 2022. Basis for Guardrails integration.

Es, S., et al. (2023). "RAGAS." arXiv:2309.15217.
    HarnessResult + Guardrails pipeline analogous to RAGAS faithfulness.

Agent Design Pattern
--------------------
All agents inherit from BaseForensicAgent:
  _retrieve_context()     — multi-query RAG via ContextBuilder
  _analyze_and_extract()  — LLM call + OutputHarness + Guardrails
  _run_agentic_rag()      — LangGraph 12-step pipeline with fallback
  _create_finding()       — log to audit trail + persist to SQLite
  _save_output()          — write agent JSON to Agent_Outputs/

Implements: Evidence → Analysis → Reasoning → Conclusion framework
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from llm.llm_manager import LLMManager
from llm.prompts import SYSTEM_PROMPTS, get_knowledge_context
from llm.context_builder import ContextBuilder
from llm.output_harness import OutputHarness, HarnessResult
from rag.hybrid_retriever import HybridRetriever
from database.sqlite_handler import SQLiteHandler
from utils.audit_trail import AuditTrail
from utils.guardrails import Guardrails
from utils.storage import StorageManager
from config import AGENT_NAMES, AGENT_CONTEXT_QUERIES, CONTEXT_CONFIG, HARNESS_CONFIG, AGENTIC_RAG_CONFIG


@dataclass
class AgentFinding:
    """Structured finding from a forensic agent."""
    agent_id: int
    agent_name: str
    finding_type: str       # RED_FLAG / GREEN_FLAG / OBSERVATION / CALCULATION
    title: str
    detail: str
    evidence: str
    source_document: str = ""
    fiscal_year: str = ""
    risk_level: str = "MEDIUM"   # CRITICAL / HIGH / MEDIUM / LOW / POSITIVE
    confidence: float = 0.75
    calculation: str = ""
    recommendation: str = ""


@dataclass
class AgentResult:
    """Complete output from an agent's investigation."""
    agent_id: int
    agent_name: str
    status: str = "COMPLETE"    # COMPLETE / PARTIAL / FAILED
    findings: list[AgentFinding] = field(default_factory=list)
    red_flags: list[AgentFinding] = field(default_factory=list)
    green_flags: list[AgentFinding] = field(default_factory=list)
    risk_score: float = 50.0    # 0-100
    summary: str = ""
    raw_analysis: str = ""
    data_quality: float = 0.5   # How complete was the underlying data?
    error: str = ""


class BaseForensicAgent(ABC):
    """
    Base class for all forensic investigation agents.
    Provides: LLM access, RAG retrieval, database persistence, audit logging.
    """

    def __init__(
        self,
        agent_id: int,
        llm: LLMManager,
        retriever: HybridRetriever,
        db: SQLiteHandler,
        audit: AuditTrail,
        storage: StorageManager,
        company_id: int = 0,
    ):
        self.agent_id = agent_id
        self.agent_name = AGENT_NAMES.get(agent_id, f"Agent {agent_id}")
        self.llm = llm
        self.retriever = retriever
        self.db = db
        self.audit = audit
        self.storage = storage
        self.company_id = company_id
        self._ctx = ContextBuilder(retriever)
        self._harness = OutputHarness()
        self._guardrails = Guardrails(groundedness_threshold=0.20, block_on_fail=False)

    @abstractmethod
    def investigate(
        self,
        company_name: str,
        company_id: int,
        financial_data: dict,
        **kwargs,
    ) -> AgentResult:
        """Main investigation method. Must be implemented by each agent."""
        pass

    # ── Helper Methods ─────────────────────────────────────

    def _retrieve_context(self, company_name: str, query: str, max_tokens: int = 2500) -> str:
        """
        Multi-query RAG context retrieval.

        If this agent has a dedicated query set in AGENT_CONTEXT_QUERIES,
        that set is merged with the caller-supplied query for richer coverage.
        Falls back to single-query retrieval for agents without a query set.
        """
        agent_queries = AGENT_CONTEXT_QUERIES.get(self.agent_id, [])
        all_queries = ([query] + agent_queries) if agent_queries else [query]
        return self._ctx.build(
            company_name=company_name,
            queries=all_queries,
            budget_tokens=max_tokens,
            n_per_query=CONTEXT_CONFIG.n_results_per_query,
        )

    def _analyze_with_llm(
        self,
        prompt: str,
        system_role: str = "forensic_accountant",
        max_tokens: int = 2048,
    ) -> str:
        """
        Run LLM analysis with appropriate system prompt.

        Appends a structured JSON output instruction (if harness config is on)
        and validates the response. Returns the raw text regardless of
        validation so callers can still use it for narrative output, but logs
        a warning when quality is low.
        """
        system_prompt = SYSTEM_PROMPTS.get(system_role, SYSTEM_PROMPTS["forensic_accountant"])
        if HARNESS_CONFIG.request_structured_output:
            prompt = prompt + self._harness.structured_output_suffix()
        raw = self.llm.generate(prompt, system_prompt, max_tokens=max_tokens)
        result = self._harness.extract(raw, company_name="")
        if not result.is_valid:
            self.log_warning(f"LLM output failed quality check (role={system_role})")
        elif result.quality_score < 0.5:
            self.log_warning(
                f"Low-quality LLM output: score={result.quality_score:.2f} (role={system_role})"
            )
        return raw

    def _analyze_and_extract(
        self,
        prompt: str,
        system_role: str = "forensic_accountant",
        max_tokens: int = 2048,
        company_name: str = "",
    ) -> HarnessResult:
        """
        Run LLM analysis and return a fully parsed HarnessResult.

        Use this instead of _analyze_with_llm() when you want structured
        findings (ParsedFinding list) and a numeric risk score extracted
        automatically from the LLM output.
        Prepends agent-specific knowledge context (standards + historical cases).
        """
        knowledge = get_knowledge_context(self.agent_id)
        if knowledge:
            prompt = f"{knowledge}\n\n{prompt}"
        system_prompt = SYSTEM_PROMPTS.get(system_role, SYSTEM_PROMPTS["forensic_accountant"])
        if HARNESS_CONFIG.request_structured_output:
            prompt = prompt + self._harness.structured_output_suffix()
        raw = self.llm.generate(prompt, system_prompt, max_tokens=max_tokens)
        result = self._harness.extract(raw, company_name=company_name)
        # Run guardrails on extracted output; attach scores to HarnessResult
        if result.raw_text:
            self._guardrails.enrich_harness_result(result, context=prompt, query="")
        return result

    def _run_agentic_rag(
        self,
        company_name: str,
        query: str,
        financial_data: dict | None = None,
    ) -> HarnessResult:
        """
        Full Agentic RAG pipeline (LangGraph + LlamaIndex + LangChain).

        Runs the 12-step loop:
          query rewrite → detail check → source routing →
          retrieval (vector DB / internet / APIs) → generation → relevance check → loop

        Falls back to the classic _analyze_with_llm() path when LangGraph is
        not installed or any step errors out, so existing agents never break.

        Returns a HarnessResult so the call site is identical to
        _analyze_and_extract().
        """
        if not AGENTIC_RAG_CONFIG.enabled:
            context = self._retrieve_context(company_name, query)
            return self._analyze_and_extract(context + "\n\n" + query, company_name=company_name)

        try:
            from graph.workflow import run_agentic_rag
            final_state = run_agentic_rag(
                company_name=company_name,
                agent_name=self.agent_name,
                agent_id=self.agent_id,
                query=query,
                financial_data=financial_data or {},
                max_iterations=AGENTIC_RAG_CONFIG.max_iterations,
                retriever=self.retriever,
            )
            raw_text = final_state.get("final_response") or final_state.get("response", "")
            harness_result = self._harness.extract(raw_text, company_name=company_name)
            # Merge structured findings from the graph state
            if final_state.get("findings"):
                from llm.output_harness import ParsedFinding
                graph_findings = [
                    ParsedFinding(
                        flag_type =f.get("flag_type", "RED_FLAG"),
                        risk_level=f.get("risk_level", "MEDIUM"),
                        title     =f.get("title", ""),
                        detail    =f.get("detail", ""),
                        evidence  =f.get("evidence", ""),
                        confidence=f.get("confidence", 0.65),
                    )
                    for f in final_state["findings"]
                ]
                harness_result.findings = graph_findings
            if final_state.get("risk_score"):
                harness_result.extracted_risk_score = final_state["risk_score"]
            return harness_result
        except Exception as exc:
            self.log_warning(f"Agentic RAG pipeline error — falling back to classic path: {exc}")
            context = self._retrieve_context(company_name, query)
            return self._analyze_and_extract(
                context + "\n\nQuery: " + query, company_name=company_name
            )

    def _create_finding(
        self,
        finding_type: str,
        title: str,
        detail: str,
        evidence: str,
        source_document: str = "",
        fiscal_year: str = "",
        risk_level: str = "MEDIUM",
        confidence: float = 0.75,
        calculation: str = "",
    ) -> AgentFinding:
        finding = AgentFinding(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            finding_type=finding_type,
            title=title,
            detail=detail,
            evidence=evidence,
            source_document=source_document,
            fiscal_year=fiscal_year,
            risk_level=risk_level,
            confidence=confidence,
            calculation=calculation,
        )

        # Log to audit trail
        if finding_type == "RED_FLAG":
            self.audit.log_red_flag(
                self.agent_id, self.agent_name, title, evidence,
                severity=risk_level, source_document=source_document,
            )
        elif finding_type == "GREEN_FLAG":
            self.audit.log_green_flag(
                self.agent_id, self.agent_name, title, evidence,
                source_document=source_document,
            )
        else:
            self.audit.log(
                self.agent_id, self.agent_name,
                action=f"{finding_type}: {title}",
                finding=detail, evidence=evidence,
                source_document=source_document,
                confidence=confidence, risk_level=risk_level,
            )

        # Persist to database
        self.db.save_finding(
            company_id=self.company_id,
            finding={
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "fiscal_year": fiscal_year,
                "finding_type": finding_type,
                "finding_title": title,
                "finding_detail": detail,
                "evidence": evidence,
                "source_document": source_document,
                "risk_level": risk_level,
                "confidence": confidence,
            },
        )
        return finding

    def _save_output(self, result: AgentResult, company_name: str) -> None:
        """Save agent output to disk."""
        from dataclasses import asdict
        output = {
            "agent_id": result.agent_id,
            "agent_name": result.agent_name,
            "status": result.status,
            "risk_score": result.risk_score,
            "summary": result.summary,
            "red_flags": len(result.red_flags),
            "green_flags": len(result.green_flags),
            "findings": [
                {
                    "type": f.finding_type,
                    "title": f.title,
                    "risk_level": f.risk_level,
                    "evidence": f.evidence[:500],
                }
                for f in result.findings
            ],
        }
        filename = f"agent_{self.agent_id:02d}_{self.agent_name.replace(' ', '_')[:30]}.json"
        self.storage.save_json(output, filename, "Agent_Outputs")

    def _get_financial_trend(self, financial_data: dict, metric: str) -> list[tuple]:
        """Extract year-over-year trend for a metric."""
        trend = []
        for year in sorted(financial_data.keys()):
            val = financial_data[year].get(metric)
            if val is not None:
                trend.append((year, val))
        return trend

    def _calculate_cagr(self, start_val: float, end_val: float, years: int) -> Optional[float]:
        """Calculate Compound Annual Growth Rate."""
        if start_val <= 0 or years <= 0:
            return None
        from utils.helpers import safe_divide
        return ((end_val / start_val) ** (1 / years) - 1) * 100

    def _check_data_quality(self, financial_data: dict) -> float:
        """Rate how complete the financial data is (0-1)."""
        required_fields = [
            "revenue", "net_income", "total_assets", "cfo",
            "accounts_receivable", "inventory", "total_debt",
        ]
        if not financial_data:
            return 0.0

        years = list(financial_data.keys())
        if not years:
            return 0.0

        latest_year = max(years)
        latest = financial_data[latest_year]
        filled = sum(1 for f in required_fields if latest.get(f) is not None and latest.get(f) != 0)
        return filled / len(required_fields)

    def log_info(self, message: str) -> None:
        logger.info(f"[{self.agent_name}] {message}")

    def log_warning(self, message: str) -> None:
        logger.warning(f"[{self.agent_name}] {message}")

    def log_error(self, message: str) -> None:
        logger.error(f"[{self.agent_name}] {message}")
