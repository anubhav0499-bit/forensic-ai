"""
Forensic Orchestrator — 17-Agent Multi-Phase Workflow Controller
=================================================================

References
----------
Multi-Agent Orchestration
~~~~~~~~~~~~~~~~~~~~~~~~~~
LangChain (Chase, H., 2022). Multi-agent orchestration patterns:
    parallel agent pools with ThreadPoolExecutor, phase gates, and
    result aggregation.
    github.com/langchain-ai/langchain

Park, J. S., et al. (2023). "Generative Agents: Interactive Simulacra
    of Human Behavior." UIST 2023. Multi-agent state sharing pattern:
    later phases receive all prior phase outputs as context.

Li, G., et al. (2023). "MetaGPT: Meta Programming for a Multi-Agent
    Collaborative Framework." ICLR 2024. Structured agent roles with
    sequential gating and result injection.

Parallelism Strategy
---------------------
Phase B: ThreadPoolExecutor(max_workers=8) for cloud LLM backends.
         Sequential fallback for local Ollama/LM Studio (no rate limits).
Phase A → Phase B → Phase C → Phase D: strictly sequential (each phase
receives the complete output of all prior phases).

Phase Architecture
-------------------
Phase A: Agent 6 (Fraud Detection) — establishes forensic baseline
Phase B: Agents 3,4,5,7,8,9,10,11 — parallel specialist agents (threaded)
Phase C: Agents 12,14 — synthesis/LLM agents with full prior context
Phase D: Agent 17 (Director) — iterative synthesis with refinement loop
         (triggers extra "Resolve Ambiguity" pass if score in 38–62 range)
"""

from __future__ import annotations
import concurrent.futures
import time
import json
from typing import TypedDict
from pathlib import Path
from loguru import logger

from config import LLM_CONFIG, HARNESS_CONFIG, AGENT_CONTEXT_QUERIES, AGENTIC_RAG_CONFIG
from llm.context_builder import ContextBuilder
from llm.output_harness import OutputHarness
from llm.llm_manager import LLMManager


def _llm_generate(prompt: str, system: str = "", max_tokens: int = 2048, llm=None) -> str:
    """
    Unified LLM call: tries LangChain client first (all 12 providers),
    falls back to legacy LLMManager if needed.
    """
    try:
        from llm.langchain_client import lc_invoke
        return lc_invoke(prompt, system=system, fast=False)
    except Exception:
        if llm is not None:
            return llm.generate(prompt, system, max_tokens=max_tokens)
        return "[LLM unavailable]"
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from rag.bm25_retriever import BM25Retriever
from rag.hybrid_retriever import HybridRetriever
from database.sqlite_handler import SQLiteHandler
# DuckDBHandler intentionally not imported here.
# Import it directly inside Agent 12 when that agent is implemented.
from utils.storage import StorageManager
from utils.audit_trail import AuditTrail

from acquisition.company_lookup import CompanyLookup, CompanyProfile
from acquisition.downloader import DocumentDownloader
from processing.pdf_processor import PDFProcessor
from processing.table_extractor import TableExtractor
from processing.chunker import DocumentChunker

from forensics.cross_validator import CrossValidator, CrossValidationIssue

from .base_agent import AgentResult, AgentFinding
from .agent_03_revenue import RevenueForensicsAgent
from .agent_04_cashflow import CashFlowForensicsAgent
from .agent_05_working_capital import WorkingCapitalAgent
from .agent_06_fraud_detection import FraudDetectionAgent
from .agent_07_credit_risk import CreditRiskAgent
from .agent_08_earnings_quality import EarningsQualityAgent
from .agent_09_related_party import RelatedPartyAgent
from .agent_10_auditor import AuditorIntelligenceAgent
from .agent_11_management_nlp import ManagementNLPAgent
from .agent_17_director import ChiefInvestigationDirector


class InvestigationState(TypedDict):
    """LangGraph state shared across all agents."""
    company_name: str
    company_id: int
    company_profile: dict
    financial_data: dict
    documents_acquired: int
    chunks_indexed: int
    agent_results: dict
    cross_validation_issues: list
    risk_score: float
    verdict: str
    report_path: str
    errors: list
    start_time: str


# ─── Shared utilities ────────────────────────────────────────────

_harness = OutputHarness()  # singleton for orchestrator-level use


def _build_inter_agent_context(agent_results: dict[int, AgentResult]) -> str:
    """
    Build structured inter-agent context for injection into Phase C/D prompts.

    Format:
      [Agent N — Name]  Risk: XX.X/100 | Red: N | Green: N | Status: S
        Summary (up to 300 chars)
        ⚠ [LEVEL] Finding title
        ⚠ [LEVEL] Finding title  (top 4)
    """
    if not agent_results:
        return ""

    lines = ["=== PRIOR AGENT FINDINGS (cross-agent intelligence) ==="]
    for agent_id in sorted(agent_results.keys()):
        r = agent_results[agent_id]
        lines.append(
            f"\n[Agent {agent_id} — {r.agent_name}]  "
            f"Risk: {r.risk_score:.1f}/100 | Red Flags: {len(r.red_flags)} | "
            f"Green Flags: {len(r.green_flags)} | Status: {r.status}"
        )
        if r.summary:
            # Include first 300 chars of summary, but strip the redundant header
            summary_text = r.summary[:300].split("\n")[0]
            lines.append(f"  Summary: {summary_text}")
        # Top 4 red flags by severity
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        sorted_flags = sorted(
            r.red_flags, key=lambda f: severity_order.get(f.risk_level, 4)
        )
        for flag in sorted_flags[:4]:
            evidence_snippet = (f" | Evidence: {flag.evidence[:60]}" if flag.evidence else "")
            lines.append(f"  ⚠ [{flag.risk_level}] {flag.title[:90]}{evidence_snippet}")
        if r.green_flags:
            lines.append(f"  ✓ [{len(r.green_flags)} green flag(s)] {r.green_flags[0].title[:80]}")
    lines.append("\n=== END PRIOR AGENT FINDINGS ===\n")
    return "\n".join(lines)


# ─── Main Orchestrator ───────────────────────────────────────────

class ForensicOrchestrator:
    """
    Main orchestration engine for the Forensic AI platform.
    Manages: document acquisition → processing → RAG → agents → reporting.
    Uses thread-pool parallelism for Phase B agents (5–10x faster than sequential).
    """

    def __init__(self):
        logger.info("Initializing Forensic AI Platform...")
        self.llm = LLMManager()
        self.embedding_model = EmbeddingModel()
        self.bm25 = BM25Retriever()
        self.db = SQLiteHandler()
        self.cross_validator = CrossValidator()
        logger.info(f"LLM Backend: {self.llm.get_backend_info()}")
        logger.info(f"Embeddings: {self.embedding_model.model_name}")

    def investigate(self, company_name: str, ticker: str = "", isin: str = "") -> dict:
        start_time = time.time()
        logger.info(f"\n{'='*60}\nStarting Forensic Investigation: {company_name}\n{'='*60}")

        # ── Setup ───────────────────────────────────────────────────
        storage = StorageManager(company_name, ticker)
        audit = AuditTrail(storage.base_path, company_name)
        vector_store = VectorStore(self.embedding_model, storage.knowledge_base)
        retriever = HybridRetriever(vector_store, self.bm25)

        # ── Phase 1: Company Identification ────────────────────────
        logger.info("Phase 1: Company Identification")
        with CompanyLookup() as lookup:
            profile = lookup.identify(company_name)
        if ticker and not profile.ticker:
            profile.ticker = ticker.upper()
        if isin and not profile.isin:
            profile.isin = isin

        company_id = self.db.upsert_company({
            "name": profile.name, "ticker": profile.ticker, "isin": profile.isin,
            "exchange": profile.exchange, "sector": profile.sector,
            "industry": profile.industry, "country": profile.country,
            "currency": profile.currency, "market_cap": profile.market_cap,
        })
        session_id = self.db.start_session(company_id)
        audit.log(0, "Orchestrator", "COMPANY_IDENTIFIED",
                  finding=f"Identified as {profile.name} ({profile.ticker}) on {profile.exchange}",
                  source_url=profile.ir_url)

        # ── Phase 2: Document Acquisition ──────────────────────────
        logger.info("Phase 2: Document Acquisition")
        downloader = DocumentDownloader(storage, self.db, audit)
        try:
            acq_summary = downloader.acquire_all_documents(profile, company_id, years=5)
            logger.info(f"Acquisition: {acq_summary['total_files']} files, {acq_summary['financial_data_years']} years")
        except Exception as e:
            logger.error(f"Acquisition error: {e}")
            acq_summary = {"total_files": 0, "financial_data_years": 0}
        finally:
            downloader.close()

        # ── Phase 3: Processing & Indexing ──────────────────────────
        logger.info("Phase 3: Document Processing and RAG Indexing")
        all_chunks = self._process_documents(storage, retriever, company_name)

        # ── Phase 4: Financial Data Assembly ────────────────────────
        logger.info("Phase 4: Financial Data Assembly")
        financial_data = self._assemble_financial_data(company_id, company_name, storage, ticker=profile.ticker)
        if not financial_data:
            logger.warning("No structured financial data. Agents will work from text only.")

        # ── Phase 4b: Cross-Validation ──────────────────────────────
        logger.info("Phase 4b: Cross-Validating Financial Statements")
        cv_issues = self.cross_validator.validate(financial_data)
        if cv_issues:
            logger.warning(f"Cross-validation found {len(cv_issues)} issues ({sum(1 for i in cv_issues if i.severity == 'CRITICAL')} CRITICAL)")
            self._log_cv_issues(cv_issues, audit, company_id)
        else:
            logger.info("Cross-validation: No inconsistencies detected")

        # ── Phase 5: Multi-Agent Investigation (Parallel) ───────────
        logger.info("Phase 5: Multi-Agent Forensic Investigation (Parallel)")
        agent_results = self._run_all_agents(
            company_name=company_name, company_id=company_id,
            financial_data=financial_data, retriever=retriever,
            storage=storage, audit=audit, cv_issues=cv_issues,
        )

        # ── Phase 6: Director Synthesis + Refinement ────────────────
        logger.info("Phase 6: Chief Investigation Director Synthesis")
        director = ChiefInvestigationDirector(17, self.llm, retriever, self.db, audit, storage, company_id=company_id)
        director_result = director.investigate(
            company_name=company_name, company_id=company_id,
            financial_data=financial_data, all_agent_results=agent_results,
        )
        agent_results[17] = director_result

        # ── Iterative refinement: re-run grey-zone agents ───────────
        director_result = self._iterative_refinement(
            director_result, agent_results, company_name, company_id,
            financial_data, retriever, storage, audit,
        )
        agent_results[17] = director_result

        # ── Phase 7: Reports ────────────────────────────────────────
        logger.info("Phase 7: Generating Reports")
        report_paths = self._generate_reports(
            company_name, company_id, financial_data,
            agent_results, director_result, storage, profile, cv_issues,
        )

        # ── Close Session ───────────────────────────────────────────
        elapsed = time.time() - start_time
        self.db.close_session(
            session_id=session_id,
            risk_score=director_result.risk_score,
            report_path=str(report_paths.get("pdf", "")),
            summary=director_result.summary[:500],
        )
        audit_summary = audit.export_summary()
        storage.save_json(audit_summary, "audit_summary.json", "Audit_Trail")

        verdict = self._extract_verdict(director_result)
        result = {
            "company_name": company_name,
            "company_profile": profile.__dict__,
            "investigation_duration_seconds": round(elapsed, 1),
            "documents_acquired": acq_summary.get("total_files", 0),
            "chunks_indexed": len(all_chunks),
            "financial_years": list(financial_data.keys()),
            "cross_validation_issues": len(cv_issues),
            "critical_cv_issues": sum(1 for i in cv_issues if i.severity == "CRITICAL"),
            "agent_results": {k: v.summary for k, v in agent_results.items() if hasattr(v, "summary")},
            "overall_risk_score": director_result.risk_score,
            "verdict": verdict,
            "red_flags": len(director_result.red_flags),
            "green_flags": len(director_result.green_flags),
            "top_red_flags": [
                {"title": f.title, "risk_level": f.risk_level, "evidence": f.evidence[:200]}
                for f in sorted(director_result.red_flags, key=lambda x: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}.get(x.risk_level, 3))[:10]
            ],
            "risk_components": self._extract_risk_components(agent_results),
            "report_paths": {k: str(v) for k, v in report_paths.items()},
            "storage_path": str(storage.base_path),
        }

        logger.info(
            f"\n{'='*60}\nInvestigation Complete: {company_name}\n"
            f"Duration: {elapsed:.0f}s | Risk Score: {director_result.risk_score:.1f}/100\n"
            f"Red Flags: {len(director_result.red_flags)} | CV Issues: {len(cv_issues)}\n"
            f"{'='*60}\n"
        )
        return result

    # ─── Phase 5: Parallel Agent Execution ───────────────────────

    def _run_all_agents(
        self, company_name, company_id, financial_data, retriever, storage, audit, cv_issues
    ) -> dict[int, AgentResult]:
        results: dict[int, AgentResult] = {}
        cv_context = self.cross_validator.summary_for_prompt(cv_issues)

        # ── Phase A: Agent 6 (Fraud) — must run first ─────────────
        logger.info("  Phase A: Agent 6 (Fraud Detection)")
        try:
            fraud_agent = FraudDetectionAgent(6, self.llm, retriever, self.db, audit, storage, company_id=company_id)
            results[6] = fraud_agent.investigate(company_name, company_id, financial_data,
                                                  cv_context=cv_context)
            logger.info(f"  ✓ Agent 6 (Fraud Detection): Risk={results[6].risk_score:.1f}")
        except Exception as e:
            logger.error(f"  ✗ Agent 6 failed: {e}")
            results[6] = AgentResult(6, "Fraud Detection Agent", status="FAILED", error=str(e))

        # ── Phase B: 8 specialist agents in parallel threads ──────
        logger.info("  Phase B: Running 8 specialist agents in parallel...")
        phase_b_specs = [
            (3,  "Revenue Forensics Agent",    RevenueForensicsAgent),
            (4,  "Cash Flow Forensics Agent",  CashFlowForensicsAgent),
            (5,  "Working Capital Agent",      WorkingCapitalAgent),
            (7,  "Credit Risk Agent",          CreditRiskAgent),
            (8,  "Earnings Quality Agent",     EarningsQualityAgent),
            (9,  "Related Party Agent",        RelatedPartyAgent),
            (10, "Auditor Intelligence Agent", AuditorIntelligenceAgent),
            (11, "Management NLP Agent",       ManagementNLPAgent),
        ]

        phase_b_results = self._run_agents_parallel(
            phase_b_specs, company_name, company_id, financial_data,
            retriever, storage, audit, cv_context=cv_context,
        )
        for agent_id, r in phase_b_results.items():
            results[agent_id] = r
            status = "✓" if r.status != "FAILED" else "✗"
            logger.info(f"  {status} Agent {agent_id} ({r.agent_name}): Risk={r.risk_score:.1f}")

        # ── Build inter-agent context for Phase C ─────────────────
        inter_context = _build_inter_agent_context(results)
        logger.info("  Phase C: Running synthesis agents with full prior context...")

        # ── Phase C: Agent 12 (Peer) — still generic LLM ──────────
        # Agents 14, 15, 16 have been merged into _run_perspectives()
        # below — same 3 LLM calls, zero extra orchestration overhead.
        phase_c_configs = [
            (12, "peer", "Peer Comparison Agent"),
        ]
        phase_c_results = self._run_generic_agents_parallel(
            phase_c_configs, company_name, company_id, financial_data,
            retriever, storage, audit, inter_context, cv_context,
        )
        for agent_id, r in phase_c_results.items():
            results[agent_id] = r
            status = "✓" if r.status != "FAILED" else "✗"
            logger.info(f"  {status} Agent {agent_id} ({r.agent_name}): Risk={r.risk_score:.1f}")

        # ── Phase C: Investment Committee Perspectives ──────────────
        # Replaces the 3 separate Short Seller / Bull Case / Devil's
        # Advocate agents.  Same prompts, one AgentResult, no overhead.
        logger.info("  Phase C: Investment committee perspectives (bear/bull/devil)...")
        try:
            results[14] = self._run_perspectives(
                company_name, company_id, financial_data,
                retriever, storage, audit, inter_context, cv_context,
            )
            logger.info("  ✓ Investment committee perspectives complete")
        except Exception as e:
            logger.error(f"  ✗ Perspectives failed: {e}")
            results[14] = AgentResult(14, "Investment Committee Perspectives",
                                      status="FAILED", error=str(e))

        return results

    # ── Backend capability ────────────────────────────────────────

    def _supports_parallel_requests(self) -> bool:
        """
        Cloud APIs (Groq, OpenAI, Anthropic, Gemini, Together, OpenRouter)
        process concurrent requests independently — threading gives a real
        speed-up here.

        Local backends (Ollama, LM Studio, HuggingFace) serialise requests
        at the inference engine anyway, so spawning threads adds overhead
        with zero throughput gain.
        """
        return self.llm.backend in (
            "groq", "openai", "anthropic", "gemini", "together", "openrouter"
        )

    # ── Shared parallel dispatcher ────────────────────────────────

    def _run_parallel(self, items: list, run_fn, max_workers: int = 8) -> dict[int, AgentResult]:
        """
        Cloud → ThreadPoolExecutor with per-agent timeout; local → sequential loop.
        Each element of `items` is unpacked as positional args to run_fn,
        which must return (agent_id, AgentResult).
        """
        timeout = HARNESS_CONFIG.agent_timeout_seconds
        results: dict[int, AgentResult] = {}
        if self._supports_parallel_requests():
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as executor:
                futures = {executor.submit(run_fn, *item): item[0] for item in items}
                for future in concurrent.futures.as_completed(futures, timeout=timeout * len(items)):
                    aid = futures[future]
                    try:
                        aid, r = future.result(timeout=timeout)
                    except concurrent.futures.TimeoutError:
                        logger.error(f"Agent {aid} timed out after {timeout}s")
                        from config import AGENT_NAMES
                        r = AgentResult(aid, AGENT_NAMES.get(aid, f"Agent {aid}"),
                                        status="FAILED", error=f"Timed out after {timeout}s")
                    except Exception as e:
                        logger.error(f"Agent {aid} raised: {e}")
                        from config import AGENT_NAMES
                        r = AgentResult(aid, AGENT_NAMES.get(aid, f"Agent {aid}"),
                                        status="FAILED", error=str(e))
                    results[aid] = r
        else:
            for item in items:
                aid, r = run_fn(*item)
                results[aid] = r
        return results

    # ── Specialized agent runner ──────────────────────────────────

    def _run_agents_parallel(
        self, specs: list, company_name, company_id, financial_data,
        retriever, storage, audit, cv_context: str = "",
    ) -> dict[int, AgentResult]:
        def run_one(agent_id, agent_name, agent_cls):
            try:
                agent = agent_cls(agent_id, self.llm, retriever, self.db, audit, storage, company_id=company_id)
                return agent_id, agent.investigate(
                    company_name, company_id, financial_data, cv_context=cv_context
                )
            except Exception as e:
                logger.error(f"Agent {agent_id} ({agent_name}) failed: {e}")
                return agent_id, AgentResult(agent_id, agent_name, status="FAILED", error=str(e))

        return self._run_parallel(specs, run_one, max_workers=8)

    # ── Generic (LLM-only) agent runner ──────────────────────────

    def _run_generic_agents_parallel(
        self, configs: list, company_name, company_id, financial_data,
        retriever, storage, audit, inter_context: str, cv_context: str,
    ) -> dict[int, AgentResult]:
        def run_one(agent_id, agent_type, agent_name):
            try:
                r = self._run_generic_agent(
                    agent_id=agent_id, agent_name=agent_name, agent_type=agent_type,
                    company_name=company_name, company_id=company_id,
                    financial_data=financial_data, retriever=retriever,
                    storage=storage, audit=audit,
                    inter_agent_context=inter_context, cv_context=cv_context,
                )
                return agent_id, r
            except Exception as e:
                logger.error(f"Generic agent {agent_id} ({agent_name}) failed: {e}")
                return agent_id, AgentResult(agent_id, agent_name, status="FAILED", error=str(e))

        return self._run_parallel(configs, run_one, max_workers=6)

    def _run_generic_agent(
        self, agent_id, agent_name, agent_type, company_name, company_id,
        financial_data, retriever, storage, audit,
        inter_agent_context: str = "", cv_context: str = "",
    ) -> AgentResult:
        """
        LLM-based generic agent with:
        - Multi-query context retrieval (ContextBuilder)
        - Structured output extraction (OutputHarness)
        - Inter-agent context injection
        """
        from llm.prompts import SYSTEM_PROMPTS, build_analysis_prompt

        role_map = {
            "revenue": "forensic_accountant", "related_party": "governance_specialist",
            "peer": "equity_analyst", "short_seller": "short_seller",
            "bull_case": "bull_case", "devils_advocate": "devils_advocate",
        }
        question_map = {
            "revenue": "Investigate revenue recognition quality, channel stuffing indicators, and premature recognition risks.",
            "related_party": "Map related party transactions, loans, guarantees, and disclosure quality. Identify self-dealing.",
            "peer": "Benchmark financial metrics against industry peers. Identify outliers and suspicious divergences.",
            "short_seller": "Build the strongest possible institutional bear case with all available evidence.",
            "bull_case": "Build the strongest possible institutional bull case with all available evidence.",
            "devils_advocate": "Challenge all previous agent conclusions. Find counterevidence. Stress-test every red flag.",
        }

        role = role_map.get(agent_type, "forensic_accountant")
        question = question_map.get(agent_type, "Perform forensic analysis.")

        # Multi-query context retrieval
        ctx_builder = ContextBuilder(retriever)
        agent_queries = AGENT_CONTEXT_QUERIES.get(agent_id, [question])
        context = ctx_builder.build(
            company_name=company_name,
            queries=([question] + agent_queries) if agent_queries else [question],
            budget_tokens=2200,
        )
        years = sorted(financial_data.keys(), reverse=True)

        prompt = build_analysis_prompt(
            agent_role=agent_name, company_name=company_name,
            fiscal_years=years[:3],
            financial_data={y: financial_data[y] for y in years[:3]} if years else {},
            extracted_text=context, question=question,
        )
        if inter_agent_context:
            prompt = inter_agent_context + "\n" + prompt
        if cv_context:
            prompt += f"\n\n{cv_context}"
        # Append structured output instruction
        prompt += _harness.structured_output_suffix()

        raw_analysis = _llm_generate(
            prompt,
            system=SYSTEM_PROMPTS.get(role, SYSTEM_PROMPTS["forensic_accountant"]),
            max_tokens=2048,
            llm=self.llm,
        )

        # Structured extraction via harness
        harness_result = _harness.extract(raw_analysis, company_name=company_name)
        result = AgentResult(agent_id=agent_id, agent_name=agent_name)
        result.raw_analysis = raw_analysis
        result.risk_score = (
            harness_result.extracted_risk_score
            if harness_result.extracted_risk_score is not None
            else _harness.estimate_risk_score(raw_analysis)
        )
        result.summary = f"{agent_name}: {raw_analysis[:400]}..."
        result.findings, result.red_flags, result.green_flags = self._findings_from_harness(
            harness_result, agent_id, agent_name, audit
        )

        filename = f"agent_{agent_id:02d}_{agent_name.replace(' ', '_')[:30]}.json"
        storage.save_json({
            "analysis": raw_analysis,
            "risk_score": result.risk_score,
            "parse_method": harness_result.parse_method,
            "quality_score": harness_result.quality_score,
        }, filename, "Agent_Outputs")
        return result

    # ─── Investment Committee Perspectives ───────────────────────────

    def _run_perspectives(
        self, company_name, company_id, financial_data,
        retriever, storage, audit, inter_context: str, cv_context: str,
    ) -> AgentResult:
        """
        Bear case, bull case, and devil's advocate in one method.

        Previously these were agents 14, 15, 16 — three separate
        _run_generic_agent() invocations with identical boilerplate.
        Now it's three llm.generate() calls and one AgentResult.

        The Director still receives the full text of all three perspectives
        (via inter_agent_context in the refinement pass or directly in
        all_agent_results[14].raw_analysis).
        """
        from llm.prompts import SYSTEM_PROMPTS, build_analysis_prompt

        years = sorted(financial_data.keys(), reverse=True)
        ctx_builder = ContextBuilder(retriever)
        context = ctx_builder.build(
            company_name=company_name,
            queries=AGENT_CONTEXT_QUERIES.get(14, [
                "investment thesis valuation risks opportunities competitive position",
            ]),
            budget_tokens=2000,
        )
        base_prompt = build_analysis_prompt(
            agent_role="Investment Committee",
            company_name=company_name,
            fiscal_years=years[:3],
            financial_data={y: financial_data[y] for y in years[:3]} if years else {},
            extracted_text=context,
            question="",
        )
        if inter_context:
            base_prompt = inter_context + "\n" + base_prompt
        if cv_context:
            base_prompt += f"\n\n{cv_context}"

        role_configs = [
            ("bear",  "short_seller",    "Build the strongest institutional BEAR case. Cite specific evidence for every claim."),
            ("bull",  "bull_case",       "Build the strongest institutional BULL case. Challenge bear assumptions with counterevidence."),
            ("devil", "devils_advocate", "Challenge ALL prior agent conclusions. Find counterevidence. Stress-test every red flag."),
        ]
        perspectives: dict[str, str] = {}
        for label, role, task in role_configs:
            prompt = f"{base_prompt}\n\nYOUR TASK: {task}"
            try:
                perspectives[label] = _llm_generate(
                    prompt,
                    system=SYSTEM_PROMPTS.get(role, SYSTEM_PROMPTS["forensic_accountant"]),
                    max_tokens=1500,
                    llm=self.llm,
                )
            except Exception as e:
                logger.error(f"Perspectives/{label} failed: {e}")
                perspectives[label] = f"[{label.upper()} PERSPECTIVE FAILED: {e}]"

        # Single AgentResult combining all three views
        result = AgentResult(agent_id=14, agent_name="Investment Committee Perspectives")
        result.raw_analysis = (
            f"=== BEAR CASE (Short Seller) ===\n{perspectives.get('bear', '')}\n\n"
            f"=== BULL CASE ===\n{perspectives.get('bull', '')}\n\n"
            f"=== DEVIL'S ADVOCATE ===\n{perspectives.get('devil', '')}"
        )
        result.summary = (
            f"Investment Committee — {company_name}: "
            f"Bear/Bull/Devil perspectives compiled."
        )
        # Score intentionally neutral — perspectives are informational,
        # not independent risk assessors.
        result.risk_score = 50.0

        storage.save_json(
            {"bear_case": perspectives.get("bear", ""),
             "bull_case": perspectives.get("bull", ""),
             "devils_advocate": perspectives.get("devil", "")},
            "agent_14_investment_committee_perspectives.json",
            "Agent_Outputs",
        )
        audit.log(14, "Investment Committee Perspectives",
                  "PERSPECTIVES_COMPLETE",
                  finding=f"Bear/bull/devil perspectives generated for {company_name}")
        return result

    # ─── Iterative Refinement (Director → targeted re-investigation) ─

    def _iterative_refinement(
        self, director_result: AgentResult, agent_results: dict,
        company_name, company_id, financial_data, retriever, storage, audit,
    ) -> AgentResult:
        """
        If risk score is in the grey zone (38–62), the Director triggers one
        targeted 'Devil's Advocate' deep dive to stress-test the conclusions.
        Prevents inconclusive verdicts.
        """
        score = director_result.risk_score
        if not (38 <= score <= 62):
            return director_result

        logger.info(f"  Iterative refinement triggered: risk={score:.1f} in grey zone")
        inter_context = _build_inter_agent_context(agent_results)

        refinement_prompt = (
            f"The current investigation of {company_name} has a GREY ZONE risk score of {score:.1f}/100. "
            f"The committee is UNDECIDED. Your job is to resolve this ambiguity.\n\n"
            f"{inter_context}\n\n"
            "Based on ALL prior agent findings above, make a DEFINITIVE determination:\n"
            "1. Which red flags are CONFIRMED (high evidence) vs UNCONFIRMED (weak evidence)?\n"
            "2. What additional signal would move this to AVOID vs BUY?\n"
            "3. Give a DEFINITIVE VERDICT: MONITOR CLOSELY / CAUTIOUS BUY / AVOID with specific conditions.\n"
            "Be direct. Do not hedge. This is the final word."
        )

        from llm.prompts import SYSTEM_PROMPTS
        refinement_analysis = _llm_generate(
            refinement_prompt,
            system=SYSTEM_PROMPTS.get("investment_director", SYSTEM_PROMPTS["forensic_accountant"]),
            max_tokens=1500,
            llm=self.llm,
        )

        # Append refinement to director result
        director_result.raw_analysis += f"\n\n=== GREY ZONE REFINEMENT ===\n{refinement_analysis}"
        director_result.summary += f"\n\n[Refinement] {refinement_analysis[:300]}"

        # Adjust score slightly toward a definitive verdict
        adjust_up = any(kw in refinement_analysis.lower() for kw in ["avoid", "caution", "red flag confirmed"])
        adjust_down = any(kw in refinement_analysis.lower() for kw in ["buy", "strong", "bullish", "positive"])
        if adjust_up and not adjust_down:
            director_result.risk_score = min(72, score + 10)
        elif adjust_down and not adjust_up:
            director_result.risk_score = max(28, score - 10)

        logger.info(f"  Refinement complete. Score adjusted: {score:.1f} → {director_result.risk_score:.1f}")
        return director_result

    # ─── Cross-Validation Logging ────────────────────────────────

    def _log_cv_issues(self, issues: list[CrossValidationIssue], audit: AuditTrail, company_id: int) -> None:
        for issue in issues:
            audit.log_red_flag(
                agent_id=0, agent_name="CrossValidator",
                flag_title=f"[CV] {issue.issue_type}",
                evidence=issue.evidence,
                severity=issue.severity,
            )
            # Save to DB as agent finding
            self.db.save_finding(
                company_id=company_id,
                finding={
                    "agent_id": 0,
                    "agent_name": "Cross-Validator",
                    "fiscal_year": issue.fiscal_year,
                    "finding_type": "RED_FLAG",
                    "finding_title": f"[CV] {issue.issue_type}: {issue.description[:80]}",
                    "finding_detail": issue.evidence,
                    "evidence": issue.evidence,
                    "risk_level": issue.severity,
                    "confidence": issue.confidence,
                },
            )

    # ─── Document Processing ─────────────────────────────────────

    def _process_documents(self, storage: StorageManager, retriever: HybridRetriever, company_name: str) -> list:
        pdf_processor = PDFProcessor()
        table_extractor = TableExtractor()
        chunker = DocumentChunker()
        all_chunks = []

        raw_files = (
            storage.list_files("Raw_Filings", "pdf") +
            storage.list_files("Raw_Filings", "htm") +
            storage.list_files("Raw_Filings", "html")
        )

        for file_path in raw_files[:20]:
            try:
                parsed = pdf_processor.process(file_path)
                if parsed.word_count > 100:
                    storage.save_text(parsed.text, f"{file_path.stem}.txt", "Parsed_Data/Text")
                    fiscal_year = self._extract_year_from_filename(file_path.name)
                    chunks = chunker.chunk_document(parsed.text, file_path.name, fiscal_year)
                    all_chunks.extend(chunks)
                    tables = table_extractor.extract_from_pdf(file_path)
                    for _, df in tables.items():
                        if df is not None and not df.empty:
                            all_chunks.extend(chunker.chunk_table(df.values.tolist(), file_path.name, fiscal_year))
            except Exception as e:
                logger.warning(f"Failed to process {file_path.name}: {e}")

        if all_chunks:
            retriever.index(company_name, all_chunks)
            logger.info(f"Indexed {len(all_chunks)} chunks")
        return all_chunks

    # ─── Financial Data Assembly ────────────────────────────────

    def _assemble_financial_data(self, company_id: int, company_name: str, storage: StorageManager, ticker: str = "") -> dict:
        db_records = self.db.get_financial_history(company_id, years=5)
        financial_data: dict = {}

        for record in db_records:
            year = str(record.get("fiscal_year", ""))
            if year:
                financial_data[year] = {k: v for k, v in record.items() if v is not None and v != 0}

        for yf_file in ["yfinance_annual.json", "screener_financials.json"]:
            yf_path = storage.financials / yf_file
            if yf_path.exists():
                try:
                    with open(yf_path) as f:
                        saved_data = json.load(f)
                    for year, metrics in saved_data.items():
                        if year not in financial_data:
                            financial_data[year] = {}
                        for k, v in metrics.items():
                            if v and k not in financial_data[year]:
                                financial_data[year][k] = v
                except Exception:
                    pass

        if not financial_data:
            financial_data = self._fetch_yfinance_financials(company_name, ticker=ticker)
        return financial_data

    def _fetch_yfinance_financials(self, company_name: str, ticker: str = "") -> dict:
        try:
            import yfinance as yf
            import pandas as pd

            name_base = company_name.upper().replace(" ", "")
            tickers_to_try = []
            # Try provided ticker first (with exchange suffixes for Indian stocks)
            if ticker:
                t = ticker.upper()
                tickers_to_try += [t, t + ".NS", t + ".BO"]
            # Then guess from company name
            tickers_to_try += [
                name_base,
                name_base + ".NS",
                name_base + ".BO",
            ]
            # Deduplicate while preserving order
            seen = set()
            tickers_to_try = [x for x in tickers_to_try if not (x in seen or seen.add(x))]
            for ticker_str in tickers_to_try:
                try:
                    t = yf.Ticker(ticker_str)
                    financials = t.financials
                    balance_sheet = t.balance_sheet
                    cashflow = t.cashflow
                    if financials is not None and not financials.empty:
                        result = {}
                        for col in financials.columns[:5]:
                            year = str(col.year) if hasattr(col, "year") else str(col)[:4]
                            year_data: dict = {}
                            for yf_name, our_name in {
                                "Total Revenue": "revenue", "Cost Of Revenue": "cogs",
                                "Gross Profit": "gross_profit", "Operating Income": "ebit",
                                "EBITDA": "ebitda", "Net Income": "net_income", "Basic EPS": "eps",
                            }.items():
                                if yf_name in financials.index:
                                    val = financials.loc[yf_name, col]
                                    if pd.notna(val):
                                        year_data[our_name] = float(val)
                            if balance_sheet is not None and not balance_sheet.empty and col in balance_sheet.columns:
                                for yf_name, our_name in {
                                    "Total Assets": "total_assets", "Current Assets": "current_assets",
                                    "Cash And Cash Equivalents": "cash_equivalents",
                                    "Accounts Receivable": "accounts_receivable",
                                    "Inventory": "inventory",
                                    "Total Liab": "total_liabilities",
                                    "Current Liabilities": "current_liabilities",
                                    "Accounts Payable": "accounts_payable",
                                    "Long Term Debt": "long_term_debt",
                                    "Total Stockholder Equity": "shareholder_equity",
                                    "Retained Earnings": "retained_earnings",
                                }.items():
                                    if yf_name in balance_sheet.index:
                                        val = balance_sheet.loc[yf_name, col]
                                        if pd.notna(val):
                                            year_data[our_name] = float(val)
                            if cashflow is not None and not cashflow.empty and col in cashflow.columns:
                                for yf_name, our_name in {
                                    "Total Cash From Operating Activities": "cfo",
                                    "Capital Expenditures": "capex",
                                    "Dividends Paid": "dividends_paid",
                                }.items():
                                    if yf_name in cashflow.index:
                                        val = cashflow.loc[yf_name, col]
                                        if pd.notna(val):
                                            year_data[our_name] = float(abs(val) if "capex" in our_name else val)
                            if year_data:
                                result[year] = year_data
                        if result:
                            logger.info(f"yfinance data for {ticker_str}: {len(result)} years")
                            return result
                except Exception:
                    continue
        except ImportError:
            pass
        return {}

    # ─── Helpers ─────────────────────────────────────────────────

    def _extract_verdict(self, director_result: AgentResult) -> str:
        if "VERDICT:" in director_result.summary:
            return director_result.summary.split("VERDICT:")[1].split("\n")[0].strip()
        score = director_result.risk_score
        if score >= 75: return "STRONG AVOID"
        if score >= 60: return "AVOID"
        if score >= 50: return "CAUTION"
        if score >= 38: return "MONITOR"
        if score >= 25: return "CAUTIOUS BUY"
        return "BUY"

    def _extract_risk_components(self, agent_results: dict) -> dict:
        """Extract per-dimension risk scores from agent results."""
        component_map = {
            3: "revenue_quality",    4: "cash_flow_quality",    5: "working_capital",
            6: "fraud_indicators",   7: "credit_risk",          8: "earnings_quality",
            9: "governance",        10: "auditor_risk",         11: "management_credibility",
        }
        return {
            dim: round(agent_results[aid].risk_score, 1)
            for aid, dim in component_map.items()
            if aid in agent_results and hasattr(agent_results[aid], "risk_score")
        }

    def _extract_year_from_filename(self, filename: str) -> str:
        import re
        match = re.search(r"20\d{2}", filename)
        return match.group() if match else ""

    def _estimate_risk_from_text(self, text: str) -> float:
        """Delegate to the OutputHarness which extracts numeric score first."""
        return _harness.estimate_risk_score(text)

    def _findings_from_harness(
        self, harness_result, agent_id: int, agent_name: str, audit: AuditTrail
    ):
        """
        Convert OutputHarness ParsedFinding list to AgentFinding objects
        and log each to the audit trail.
        """
        findings, red_flags, green_flags = [], [], []
        for pf in harness_result.findings[:HARNESS_CONFIG.max_findings_per_agent]:
            f = AgentFinding(
                agent_id=agent_id,
                agent_name=agent_name,
                finding_type=pf.flag_type,
                title=pf.title,
                detail=pf.detail,
                evidence=pf.evidence,
                risk_level=pf.risk_level,
                confidence=pf.confidence,
            )
            findings.append(f)
            if pf.flag_type == "RED_FLAG":
                red_flags.append(f)
                audit.log_red_flag(
                    agent_id, agent_name, pf.title, pf.evidence,
                    severity=pf.risk_level,
                )
            elif pf.flag_type == "GREEN_FLAG":
                green_flags.append(f)
                audit.log_green_flag(agent_id, agent_name, pf.title, pf.evidence)
        return findings, red_flags, green_flags

    def _parse_llm_findings(self, text: str, agent_id: int, agent_name: str, audit: AuditTrail):
        """Legacy wrapper — routes through OutputHarness for backward compatibility."""
        harness_result = _harness.extract(text)
        return self._findings_from_harness(harness_result, agent_id, agent_name, audit)

    def _generate_reports(
        self, company_name, company_id, financial_data, agent_results,
        director_result, storage, profile, cv_issues,
    ) -> dict:
        from reporting.report_compiler import ReportCompiler
        compiler = ReportCompiler(storage, self.db)
        return compiler.generate_all(
            company_name=company_name, company_id=company_id,
            financial_data=financial_data, agent_results=agent_results,
            director_result=director_result,
            profile=profile.__dict__ if hasattr(profile, "__dict__") else {},
            extra_data={"cross_validation_issues": [
                {
                    "issue_type": i.issue_type,
                    "description": i.description,
                    "fiscal_year": i.fiscal_year,
                    "severity": i.severity,
                    "evidence": i.evidence,
                }
                for i in cv_issues
            ]},
        )

    def investigate_batch(self, companies: list[str]) -> list[dict]:
        results = []
        for i, company in enumerate(companies, 1):
            logger.info(f"\nCompany {i}/{len(companies)}: {company}")
            try:
                results.append(self.investigate(company))
            except Exception as e:
                logger.error(f"Investigation failed for {company}: {e}")
                results.append({"company_name": company, "error": str(e)})
        return results
