"""
Audit Trail — Complete, auditable record of all investigation steps
====================================================================
Every agent action, data source, calculation, and conclusion is logged.

Standards
---------
ISA 230 / SA 230: Audit Documentation — auditor must document "sufficient
    appropriate audit evidence to support the audit opinion." Audit trail
    implements this requirement digitally.
PCAOB AS 1215: Audit Documentation — documentation must be completed within 60
    days of audit report date; permanent record of all findings and conclusions.
Companies Act 2013 §134(5)(e) — Board responsibility for adequate internal
    financial controls; audit trail provides evidence of control operation.
SEBI LODR Regulation 4(2)(f) — Books of account and records preserved for 5 years;
    audit trail is a mandatory component.

Role
----
Append-only investigation log consumed by ReportCompiler for the Audit Trail XLSX
tab and JSON summary. Log format per entry: timestamp (ISO 8601), agent_id,
agent_name, action, finding, evidence (first 200 chars), source_document,
confidence (0-1), risk_level (CRITICAL/HIGH/MEDIUM/LOW). Persisted to:
data/audit_trail_{company_slug}.jsonl.
"""

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Any
from dataclasses import dataclass, field, asdict
from loguru import logger


@dataclass
class AuditEntry:
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    agent_id: int = 0
    agent_name: str = ""
    action: str = ""
    source_url: str = ""
    source_document: str = ""
    evidence: str = ""
    calculation: str = ""
    finding: str = ""
    confidence: float = 0.0
    risk_level: str = "UNKNOWN"
    metadata: dict = field(default_factory=dict)


class AuditTrail:
    """
    Immutable audit log for the forensic investigation.
    Every finding is traceable to source evidence.
    """

    def __init__(self, storage_path: Path, company_name: str):
        self.storage_path = storage_path
        self.company_name = company_name
        self.entries: list[AuditEntry] = []
        self.log_file = storage_path / "Audit_Trail" / "investigation_log.jsonl"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._session_start = datetime.utcnow().isoformat()
        self._write_header()

    def _write_header(self) -> None:
        header = {
            "type": "SESSION_START",
            "company": self.company_name,
            "timestamp": self._session_start,
            "platform": "ForensicAI v1.0",
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(header) + "\n")

    def log(
        self,
        agent_id: int,
        agent_name: str,
        action: str,
        finding: str = "",
        evidence: str = "",
        source_url: str = "",
        source_document: str = "",
        calculation: str = "",
        confidence: float = 0.0,
        risk_level: str = "INFO",
        **metadata,
    ) -> AuditEntry:
        entry = AuditEntry(
            agent_id=agent_id,
            agent_name=agent_name,
            action=action,
            source_url=source_url,
            source_document=source_document,
            evidence=evidence,
            calculation=calculation,
            finding=finding,
            confidence=confidence,
            risk_level=risk_level,
            metadata=metadata,
        )
        self.entries.append(entry)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry)) + "\n")
        logger.debug(f"[Agent {agent_id}] {action}: {finding[:100] if finding else ''}")
        return entry

    def log_calculation(
        self,
        agent_id: int,
        agent_name: str,
        metric_name: str,
        value: float,
        formula: str,
        inputs: dict,
        interpretation: str,
    ) -> None:
        calc_str = f"{metric_name} = {value:.4f} | Formula: {formula} | Inputs: {json.dumps(inputs, default=str)}"
        self.log(
            agent_id=agent_id,
            agent_name=agent_name,
            action=f"CALCULATION:{metric_name}",
            calculation=calc_str,
            finding=interpretation,
            confidence=1.0,
            metric_name=metric_name,
            metric_value=value,
            formula=formula,
            inputs=inputs,
        )

    def log_document(
        self,
        agent_id: int,
        source_url: str,
        document_type: str,
        fiscal_year: str,
        file_path: str,
    ) -> None:
        self.log(
            agent_id=agent_id,
            agent_name="Document Acquisition Agent",
            action="DOCUMENT_ACQUIRED",
            source_url=source_url,
            source_document=file_path,
            finding=f"Downloaded {document_type} for FY{fiscal_year}",
            document_type=document_type,
            fiscal_year=fiscal_year,
        )

    def log_red_flag(
        self,
        agent_id: int,
        agent_name: str,
        flag_title: str,
        evidence: str,
        severity: str = "HIGH",
        source_document: str = "",
    ) -> None:
        self.log(
            agent_id=agent_id,
            agent_name=agent_name,
            action="RED_FLAG",
            finding=flag_title,
            evidence=evidence,
            source_document=source_document,
            risk_level=severity,
            confidence=0.85,
            flag_type="RED",
        )
        logger.warning(f"RED FLAG [{severity}]: {flag_title}")

    def log_green_flag(
        self,
        agent_id: int,
        agent_name: str,
        flag_title: str,
        evidence: str,
        source_document: str = "",
    ) -> None:
        self.log(
            agent_id=agent_id,
            agent_name=agent_name,
            action="GREEN_FLAG",
            finding=flag_title,
            evidence=evidence,
            source_document=source_document,
            risk_level="POSITIVE",
            confidence=0.80,
            flag_type="GREEN",
        )

    def get_red_flags(self) -> list[AuditEntry]:
        return [e for e in self.entries if e.metadata.get("flag_type") == "RED"]

    def get_green_flags(self) -> list[AuditEntry]:
        return [e for e in self.entries if e.metadata.get("flag_type") == "GREEN"]

    def get_agent_findings(self, agent_id: int) -> list[AuditEntry]:
        return [e for e in self.entries if e.agent_id == agent_id]

    def export_summary(self) -> dict:
        """Lightweight session summary. Full entry data lives in investigation_log.jsonl."""
        return {
            "company": self.company_name,
            "session_start": self._session_start,
            "session_end": datetime.utcnow().isoformat(),
            "total_entries": len(self.entries),
            "red_flags": len(self.get_red_flags()),
            "green_flags": len(self.get_green_flags()),
            "agents_run": sorted({e.agent_id for e in self.entries}),
            "audit_log": str(self.log_file),
        }
