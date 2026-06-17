"""
Forensic AI — Knowledge Base Module
Provides authoritative standards, historical fraud cases, fraud indicators,
audit procedure templates, and reasoning framework for all agents.
"""

from .forensic_knowledge import (
    REASONING_FRAMEWORK,
    HALLUCINATION_CONTROL,
    EVIDENCE_PRIORITY,
    AUDIT_STANDARDS_CATALOG,
    INDIAN_REGULATORY_PROVISIONS,
    FRAUD_INDICATORS_CATALOG,
    AUDIT_PROCEDURE_TEMPLATES,
    AGENT_STANDARDS_MAP,
    AGENT_CASE_MAP,
    get_agent_knowledge_block,
    get_audit_procedures,
    get_reasoning_preamble,
)

__all__ = [
    "REASONING_FRAMEWORK",
    "HALLUCINATION_CONTROL",
    "EVIDENCE_PRIORITY",
    "AUDIT_STANDARDS_CATALOG",
    "INDIAN_REGULATORY_PROVISIONS",
    "FRAUD_INDICATORS_CATALOG",
    "AUDIT_PROCEDURE_TEMPLATES",
    "AGENT_STANDARDS_MAP",
    "AGENT_CASE_MAP",
    "get_agent_knowledge_block",
    "get_audit_procedures",
    "get_reasoning_preamble",
]
