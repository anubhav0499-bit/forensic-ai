"""
Base Forensic Agent - Foundation class for all investigation agents
Implements: Evidence → Analysis → Reasoning → Conclusion framework
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from llm.llm_manager import LLMManager
from llm.prompts import SYSTEM_PROMPTS
from rag.hybrid_retriever import HybridRetriever
from database.sqlite_handler import SQLiteHandler
from utils.audit_trail import AuditTrail
from utils.storage import StorageManager
from config import AGENT_NAMES


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
    ):
        self.agent_id = agent_id
        self.agent_name = AGENT_NAMES.get(agent_id, f"Agent {agent_id}")
        self.llm = llm
        self.retriever = retriever
        self.db = db
        self.audit = audit
        self.storage = storage

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
        """Get relevant document context from RAG."""
        return self.retriever.get_context_for_agent(company_name, query, max_tokens)

    def _analyze_with_llm(
        self,
        prompt: str,
        system_role: str = "forensic_accountant",
        max_tokens: int = 2048,
    ) -> str:
        """Run LLM analysis with appropriate system prompt."""
        system_prompt = SYSTEM_PROMPTS.get(system_role, SYSTEM_PROMPTS["forensic_accountant"])
        return self.llm.generate(prompt, system_prompt, max_tokens=max_tokens)

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
            company_id=0,  # Will be set by orchestrator
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
