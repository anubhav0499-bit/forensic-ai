# FORENSIC AI — TECHNICAL REFERENCE MANUAL
### Institutional-Grade Multi-Agent Forensic Accounting & Financial Intelligence Platform

---

**Classification:** Internal — Restricted  
**Version:** 1.3  
**Platform:** Forensic AI v1.3  
**Maintained by:** Platform Owner  
**Last Updated:** June 2026

**v1.3 Changes (current) — Full codebase audit, 23 bugs fixed across 17 files:**
- `acquisition/india_markets.py` — ZeroDivisionError guard on `rate_limit == 0`
- `acquisition/ir_scraper.py` — fixed rate-limit sleep: was sleeping `rate_limit` seconds (a frequency value) instead of `1/rate_limit`; also fixed after prior `settings` → `ACQUISITION_CONFIG` fix
- `acquisition/sec_edgar.py` — IndexError guard when `dates`/`accessions`/`primary_docs` arrays are shorter than `forms`; applied to both annual and quarterly filing loops
- `processing/table_extractor.py` — `NameError: PROCESSING_CONFIG is not defined`; added missing `from config import PROCESSING_CONFIG` import
- `processing/chunker.py` — infinite loop when `chunk_overlap >= chunk_size`; clamped advance to `max(1, step - overlap)`
- `database/sqlite_handler.py` — `PRAGMA foreign_keys=ON` was only set on the init connection; moved both PRAGMAs into `_conn()` so every connection enforces them
- `database/duckdb_handler.py` — `KeyError`/`ValueError` in `melt()`: `ticker` column does not exist in `financial_data` table; removed from `id_cols`
- `forensics/beneish_score.py` — TATA formula: (a) added `working_capital_tm1` and `cash_tm1` fields to `BeneishInputs`; (b) fixed working capital change to use symmetric `WC_t - WC_tm1`; (c) fixed cash term to use delta `(cash_t - cash_tm1)` not absolute level; (d) added `taxes_payable_t` to the formula
- `utils/storage.py` — `NameError: Optional is not defined`; crashes on import; added `Optional` to `from typing import Any, Optional`
- `rag/vector_store.py` — `1 - distance` similarity only correct for cosine metric; now uses `1/(1+d)` for non-negative distances (L2), cosine fallback preserved
- `rag/hybrid_retriever.py` — word count accumulated pre-truncation length causing loop to exit early after first truncated chunk; now counts words of `content` after truncation
- `agents/agent_04_cashflow.py` — multi-year CFO divergence finding missing `fiscal_year` argument; finding was stored with empty fiscal year in DB
- `agents/agent_05_working_capital.py` — multi-year DSO deterioration finding missing `fiscal_year` argument
- `agents/agent_08_earnings_quality.py` — low-ETR check (`etr < 0.08`) was inside `if etr_change is not None` guard; never fired when no prior-year ETR exists; moved to separate `if` block
- `agents/agent_11_management_nlp.py` — (a) `self.retriever.search()` called with non-existent method; replaced with `self._retrieve_context()`; (b) `concall_record` dict built but never persisted to DB; added `self.db.save_concall()` call; (c) `risk_score` had upper cap but no lower cap (returned 0.0 for clean companies); added `max(10.0, ...)` consistent with all other agents
- `agents/agent_17_director.py` — red flags sorted alphabetically by string (`"LOW"` sorts after `"HIGH"`); replaced with severity-priority dict sort
- `reporting/xlsx_generator.py` — `from openpyxl.styles import ... numbers` raises `ImportError`; `numbers` does not exist in `openpyxl.styles`; removed
- `reporting/docx_generator.py` — class-level `RGBColor(...)` attributes evaluated at import time when `HAS_DOCX=False`; raised `NameError`; guarded with `if HAS_DOCX else None`
- `reporting/pdf_generator.py` — `color.hexval()` method does not exist on ReportLab `HexColor`; `hasattr` always returned False, cover-page risk color always hardcoded red; fixed to use `color._hexval`
- `reporting/html_generator.py` — Piotroski scores 3–6 (neutral) shown in same red as scores <3 (failing); added three-tier coloring: green ≥7, orange 3–6, red <3
- `llm/llm_manager.py` — 429 retry logic now parses wait time from error message (e.g. "try again in 3.26s") instead of fixed 2s/4s backoff that was shorter than Groq's rate-limit reset window
- `acquisition/ir_scraper.py` — `from config import settings` (settings undefined); fixed to `from config import ACQUISITION_CONFIG` (previous session)

**v1.2 Changes:**
- 10 bug fixes: `company_id` threading through all agents; CAUTIOUS BUY verdict ordering; CV issue field names; governance agent lookup; `EmbeddingConfig.device` GPU default; CAGR fiscal-year key parsing; Gemini safety-filter crash; risk-score keyword false negatives; HF backend availability flag; `_extract_perspective()` regex for merged agent 14
- Refactoring: `_run_agents_parallel` and `_run_generic_agents_parallel` unified into a single `_run_parallel(items, run_fn, max_workers)` dispatcher
- `AuditTrail.export_summary()` now writes a lightweight stats summary with a pointer to `investigation_log.jsonl` instead of duplicating all entries
- Removed LangGraph / LangChain from runtime dependencies (were listed but never used)
- `.gitignore` added; `app.py` LLM selector replaced with auto-detected backend display
- PPTX and HTML report generators disabled by default (re-enable by uncommenting in `report_compiler.py`)

**v1.1 Changes:**
- 9-provider LLM support (Groq, OpenAI, Anthropic, Gemini, Together, OpenRouter, LM Studio, Ollama, HF)
- Agent 3 (Revenue Forensics) — promoted to dedicated class with quantitative checks
- Agent 9 (Related Party) — promoted to dedicated class with RPT forensics
- Phase B now runs 8 agents in parallel (was 6)
- Agents 14/15/16 (Short Seller / Bull Case / Devil's Advocate) merged into a single `_run_perspectives()` call returning one `AgentResult(agent_id=14)`
- Google Colab notebook (`colab_setup.ipynb`) for cloud-based execution
- Industry-specific threshold overrides framework
- `requirements-minimal.txt` for quick-start installs

---

## TABLE OF CONTENTS

1. [Executive Overview](#1-executive-overview)
2. [System Architecture](#2-system-architecture)
3. [Agent Roster — All 17 Agents](#3-agent-roster)
4. [Investigation Pipeline — 7 Phases](#4-investigation-pipeline)
5. [Forensic Scoring Models](#5-forensic-scoring-models)
6. [Risk Scoring Framework](#6-risk-scoring-framework)
7. [Cross-Validation Engine](#7-cross-validation-engine)
8. [Data Sources & Acquisition](#8-data-sources--acquisition)
9. [Output Formats — 4 Active Report Types](#9-output-formats)
10. [Configuration Reference](#10-configuration-reference)
11. [Operational Guide — Running Investigations](#11-operational-guide)
12. [Control Points & Override Procedures](#12-control-points--override-procedures)
13. [Audit Trail & Evidence Chain](#13-audit-trail--evidence-chain)
14. [Limitations & Caveats](#14-limitations--caveats)
15. [Escalation & Review Procedures](#15-escalation--review-procedures)
16. [Glossary](#16-glossary)

---

## 1. EXECUTIVE OVERVIEW

### What is Forensic AI?

Forensic AI is a Python-based, multi-agent forensic accounting platform designed for institutional-grade financial due diligence, investment risk assessment, and fraud investigation. It operates fully locally with no paid API dependencies, no cloud data transmission, and no third-party subscriptions.

### What It Does — End-to-End

When a user submits a company name (e.g., "Infosys", "AAPL", "Reliance"):

```
User Input → Company ID → Document Acquisition → Knowledge Base → 
17 Agents (Parallel) → Cross-Validation → Director Synthesis → 
4 Report Formats (PDF/DOCX/XLSX/JSON) → Risk Score (0–100) + Investment Verdict
```

The platform automatically:
- Identifies the company and locates all public filings (SEC EDGAR for US; NSE/BSE for India)
- Downloads annual reports, quarterly results, earnings transcripts, and governance filings
- Parses PDFs (including scanned documents via OCR), extracts financial tables
- Builds a searchable forensic knowledge base (ChromaDB + BM25 hybrid RAG)
- Runs 17 specialist AI agents in parallel, each investigating a specific domain
- Cross-validates financial statements for internal consistency across 10 rules
- Synthesizes all findings into a single investigation verdict
- Generates 4 report formats by default: PDF, DOCX, XLSX, JSON (PPTX and HTML available, disabled by default)

### Key Principles

| Principle | Implementation |
|-----------|---------------|
| **Evidence-First** | Every finding follows: Evidence → Analysis → Reasoning → Conclusion |
| **No Hallucinations** | Quantitative models (Beneish, Altman, Piotroski, Dechow) run on actual financial data |
| **Full Traceability** | Immutable JSONL audit trail; every finding cites its source document and calculation |
| **Provider-Agnostic** | 9 LLM providers supported; works on Colab (free API), VSCode (local), or cloud |
| **Latest Data** | yfinance + SEC EDGAR + NSE/BSE pull real-time financial data at investigation time |
| **Graceful Degradation** | Groq→OpenAI→Anthropic→Gemini→LM Studio→Ollama→HuggingFace→Template; always produces output |

### Who It Is For

| User Role | Use Case |
|-----------|---------|
| **Forensic Analyst** | Primary investigator; runs investigations, reviews agent findings |
| **Senior Analyst / Fund Manager** | Reviews final reports; makes investment decisions |
| **Risk Committee** | Reviews aggregated risk scores; approves verdicts |
| **Compliance Officer** | Reviews audit trail; ensures evidence chain integrity |
| **Platform Administrator** | Manages configuration, LLM models, storage, access |

---

## 2. SYSTEM ARCHITECTURE

### High-Level Component Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FORENSIC AI PLATFORM                        │
├──────────────┬──────────────┬──────────────┬───────────────────────┤
│  ACQUISITION │  PROCESSING  │     RAG      │    FORENSIC ENGINES   │
│              │              │              │                        │
│ SEC EDGAR    │ PyMuPDF      │ ChromaDB     │ Beneish M-Score       │
│ NSE/BSE      │ pdfplumber   │ BM25         │ Altman Z-Score        │
│ IR Scraper   │ Camelot      │ SentenceTransf│ Piotroski F-Score    │
│ yfinance     │ Tabula       │ Hybrid RRF   │ Dechow F-Score        │
│ Screener.in  │ Tesseract OCR│              │ Accrual Analysis      │
│              │              │              │ Working Capital        │
│              │              │              │ Cross-Validator        │
├──────────────┴──────────────┴──────────────┴───────────────────────┤
│                         17-AGENT LAYER                              │
│                                                                     │
│  Phase A         Phase B (8 agents, parallel)        Phase C (Parallel)│
│  ─────────       ───────────────────────────────     ─────────────────│
│  Agent 6:        Agent 3: Revenue Forensics           Agent 12: Peer  │
│  Fraud Detect    Agent 4: Cash Flow Forensics         Agent 14: Short │
│                  Agent 5: Working Capital             Agent 15: Bull  │
│                  Agent 7: Credit Risk                 Agent 16: Devil │
│                  Agent 8: Earnings Quality                            │
│                  Agent 9: Related Party (NEW)                        │
│                  Agent 10: Auditor Intelligence                       │
│                  Agent 11: Management NLP                             │
│                                                                     │
│  Phase D: Agent 17 — Chief Investigation Director                  │
│           (Synthesis + Iterative Refinement Loop)                  │
├─────────────────────────────────────────────────────────────────────┤
│                      DATA LAYER                                     │
│  SQLite (findings, companies, sessions)  DuckDB (analytics)        │
│  ChromaDB (vector embeddings)            JSONL (audit trail)        │
├─────────────────────────────────────────────────────────────────────┤
│                    REPORTING LAYER                                  │
│   PDF  │  DOCX  │  XLSX  │  PPTX  │  HTML Dashboard  │  JSON      │
└─────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM (Cloud Free)** | Groq (Llama 70B) / Gemini 2.0 Flash | Ultra-fast free-tier cloud inference; ideal for Colab |
| **LLM (Cloud Paid)** | OpenAI GPT-4o / Anthropic Claude / Together AI | Highest quality; paid APIs |
| **LLM (Local)** | Ollama + Qwen2.5:7B / LM Studio | Fully offline; no data transmission |
| **LLM (GPU)** | HuggingFace Transformers | GPU-accelerated (Colab T4/A100, local GPU) |
| **LLM (Last Resort)** | Template Mode | Deterministic output without LLM |
| **Embeddings** | BAAI/bge-large-en-v1.5 | Semantic document search |
| **Vector DB** | ChromaDB | Persistent embedding store |
| **Sparse Retrieval** | BM25Okapi | Keyword-based document retrieval |
| **Hybrid Fusion** | RRF (k=60) | Combines BM25 + dense with 0.4/0.6 weights |
| **Relational DB** | SQLite (WAL mode) | Findings, companies, sessions |
| **Analytics DB** | DuckDB | Trend analysis, peer benchmarking |
| **PDF Parsing** | PyMuPDF → pdfplumber → Tesseract | Cascade OCR fallback |
| **Table Extraction** | Camelot → Tabula | Financial table parsing |
| **Orchestration** | ThreadPoolExecutor | Parallel agent execution (cloud backends); sequential fallback for local |
| **UI** | Streamlit | Web interface |
| **CLI** | Python argparse | Command-line interface |

### Directory Structure

```
forensic_ai/
│
├── main.py                    ← CLI entry point
├── app.py                     ← Streamlit web interface
├── config.py                  ← All configuration (single source of truth)
├── colab_setup.ipynb          ← Google Colab notebook (free API, Drive mount)
├── requirements.txt           ← Full dependencies (local + cloud)
├── requirements-minimal.txt   ← Minimal install (cloud API only)
├── .env.example               ← API keys template (copy to .env)
│
├── acquisition/               ← Document acquisition
│   ├── company_lookup.py      ← Company identification (Yahoo, EDGAR, NSE)
│   ├── sec_edgar.py           ← SEC EDGAR API client
│   ├── india_markets.py       ← NSE/BSE/Screener.in client
│   ├── ir_scraper.py          ← IR website scraper
│   └── downloader.py          ← Orchestrates all acquisition
│
├── processing/                ← Document processing
│   ├── pdf_processor.py       ← PDF parsing cascade
│   ├── table_extractor.py     ← Financial table extraction
│   └── chunker.py             ← RAG chunking
│
├── rag/                       ← Retrieval-Augmented Generation
│   ├── embeddings.py          ← Sentence transformer wrapper
│   ├── vector_store.py        ← ChromaDB wrapper
│   ├── bm25_retriever.py      ← BM25 keyword retrieval
│   └── hybrid_retriever.py    ← RRF fusion + reranking
│
├── forensics/                 ← Quantitative forensic engines
│   ├── beneish_score.py       ← Beneish M-Score (8 variables)
│   ├── altman_score.py        ← Altman Z-Score (3 model variants)
│   ├── piotroski_score.py     ← Piotroski F-Score (9 criteria)
│   ├── dechow_score.py        ← Dechow F-Score (logistic regression)
│   ├── accrual_analysis.py    ← Sloan accrual decomposition
│   ├── working_capital_analysis.py  ← DSO/DIO/DPO/CCC
│   ├── risk_scorer.py         ← Composite 0-100 risk scorer
│   └── cross_validator.py     ← 10-rule internal consistency engine
│
├── agents/                    ← AI investigation agents
│   ├── base_agent.py          ← Abstract base class
│   ├── agent_03_revenue.py    ← Revenue Forensics (AR/Rev gap, Q4 skew, deferred rev)
│   ├── agent_04_cashflow.py
│   ├── agent_05_working_capital.py
│   ├── agent_06_fraud_detection.py
│   ├── agent_07_credit_risk.py
│   ├── agent_08_earnings_quality.py
│   ├── agent_09_related_party.py  ← Related Party (RPT concentration, promoter loans)
│   ├── agent_10_auditor.py
│   ├── agent_11_management_nlp.py
│   ├── agent_17_director.py
│   └── orchestrator.py        ← Main workflow controller
│
├── database/
│   ├── schema.py              ← 13-table SQLite schema
│   ├── sqlite_handler.py      ← SQLite CRUD
│   └── duckdb_handler.py      ← Analytical queries
│
├── llm/
│   ├── llm_manager.py         ← LLM backend manager
│   └── prompts.py             ← System prompts for each agent role
│
├── reporting/
│   ├── report_compiler.py     ← Aggregates all data for reports
│   ├── xlsx_generator.py      ← Excel workbook (9 tabs)
│   ├── docx_generator.py      ← Word document (25 sections)
│   ├── pdf_generator.py       ← PDF (ReportLab)
│   ├── pptx_generator.py      ← PowerPoint (7 slides)
│   └── html_generator.py      ← HTML dashboard
│
└── utils/
    ├── storage.py             ← File storage manager
    ├── audit_trail.py         ← Immutable JSONL audit log
    └── helpers.py             ← Shared utilities
```

---

## 3. AGENT ROSTER

The platform deploys 17 specialist agents organized into 4 execution phases. Each agent has a defined domain, data sources, and output type.

### Phase A — Forensic Baseline (Sequential, First)

| # | Agent | Class | Primary Role | Engines Used |
|---|-------|-------|-------------|-------------|
| **6** | **Fraud Detection Agent** | `FraudDetectionAgent` | Runs all 5 quantitative forensic models; compares against known fraud case database | Beneish, Altman, Piotroski, Dechow, AccrualAnalyzer |

> **Why Agent 6 runs first:** Its scores (M-Score, Z-Score, accrual ratios) are injected into the prompts of all subsequent agents as baseline context.

---

### Phase B — Specialist Agents (Parallel, 8 agents)

| # | Agent | Class | Primary Role | Engines Used |
|---|-------|-------|-------------|-------------|
| **3** | **Revenue Forensics** | `RevenueForensicsAgent` | AR/Revenue gap, Q4 skew, deferred revenue pull-forward, CAGR divergence | Quantitative + RAG |
| **4** | **Cash Flow Forensics** | `CashFlowForensicsAgent` | FCF quality, CFO-vs-NI divergence, CapEx sustainability | AccrualAnalyzer |
| **5** | **Working Capital** | `WorkingCapitalAgent` | DSO/DIO/DPO/CCC with multi-year trend | WorkingCapitalAnalyzer |
| **7** | **Credit Risk** | `CreditRiskAgent` | Leverage, coverage, liquidity, implied credit rating | AltmanZScore |
| **8** | **Earnings Quality** | `EarningsQualityAgent` | Accruals, tax rate anomalies, margin consistency | AccrualAnalyzer |
| **9** | **Related Party** | `RelatedPartyAgent` | Self-dealing, RPT concentration, promoter loans, disclosure quality | Quantitative + RAG |
| **10** | **Auditor Intelligence** | `AuditorIntelligenceAgent` | Going concern, material weakness, restatements, KAMs | RAG + NLP |
| **11** | **Management NLP** | `ManagementNLPAgent` | Evasion, uncertainty, non-GAAP overemphasis in disclosures | Word-list NLP + RAG |

---

### Phase C — Synthesis Agents (with inter-agent context)

These agents receive the complete output of Phase A and Phase B agents before forming their own views. This prevents echo-chamber conclusions.

| # | Agent | LLM Role | Primary Question |
|---|-------|---------|-----------------|
| **12** | **Peer Comparison** | Equity Analyst | How do metrics compare to industry peers? Are any outliers suspicious? |
| **14** | **Investment Committee Perspectives** | Bear / Bull / Devil's Advocate | Three perspectives combined in one `AgentResult`: `_run_perspectives()` makes three sequential LLM calls and concatenates them under `=== BEAR CASE ===`, `=== BULL CASE ===`, `=== DEVIL'S ADVOCATE ===` headers. Agent 17 extracts each section via regex. |

> **Note on agents 3, 9:** These were Phase C LLM-only agents in v1.0. They were promoted to dedicated quantitative classes in v1.1 and now run in Phase B alongside the other specialist agents.
>
> **Note on agents 15, 16:** These were standalone agents in v1.0. They were merged into agent 14 in v1.1 to eliminate boilerplate. The Director still receives all three perspectives via `all_agent_results[14].raw_analysis`.

---

### Phase D — Director Synthesis (Sequential, Last)

| # | Agent | Class | Primary Role |
|---|-------|-------|-------------|
| **17** | **Chief Investigation Director** | `ChiefInvestigationDirector` | Aggregates all 16 agents; computes composite risk score; issues final verdict; generates 50–80 management questions; sets monitoring triggers |

> **Iterative Refinement:** If risk score lands in the 38–62 grey zone, the Director automatically triggers a targeted "Resolve the Ambiguity" LLM pass and adjusts the score toward a definitive verdict.

---

### Agent Data Flow

```
Financial Data ──────────────────────────────────────────────┐
                                                             ↓
RAG Context ──────────────────────────────────────────────── Agent
                                                             ↓
Cross-Validation Issues ──────────────────────────────────── Prompt
                                                             ↓
Inter-Agent Context (Phase C only) ──────────────────────── LLM
                                                             ↓
                                                          AgentResult
                                                             │
                ┌────────────────┬───────────────────────────┘
                ↓                ↓
         AgentFinding       risk_score (0-100)
         (RED_FLAG /        summary
          GREEN_FLAG /      raw_analysis
          OBSERVATION)
```

---

## 4. INVESTIGATION PIPELINE

### The 7 Phases

```
Phase 1: Company Identification     (~5 sec)
Phase 2: Document Acquisition       (~2–10 min, network-dependent)
Phase 3: Document Processing + RAG  (~3–15 min, size-dependent)
Phase 4: Financial Data Assembly    (~30 sec)
Phase 4b: Cross-Validation          (~2 sec)
Phase 5: Multi-Agent Investigation  (~5–30 min, LLM-dependent)
Phase 6: Director Synthesis         (~3–5 min)
Phase 7: Report Generation          (~1–2 min)
```

### Phase 1 — Company Identification

**What happens:** The platform uses a cascade of lookups to identify the company:
1. Yahoo Finance search (returns ticker, exchange, sector)
2. SEC EDGAR ticker lookup (if US company)
3. NSE India autocomplete API (if Indian company)
4. Manual override via `--company` flag with `--ticker` if needed

**Output:** `CompanyProfile` dataclass:
```
name, ticker, isin, exchange, sector, industry,
country, currency, market_cap, cik (SEC), ir_url
```

**Control Point:** If company identification fails or picks the wrong entity, see [Section 12 — Override Procedures](#12-control-points--override-procedures).

---

### Phase 2 — Document Acquisition

**US Companies (SEC EDGAR):**
- Annual Reports: 10-K / 20-F (last 5 years)
- Quarterly Reports: 10-Q (last 8 quarters)
- Financial Data: XBRL JSON API (`data.sec.gov/api/xbrl/companyfacts/{CIK}.json`)
- Supplemental: Proxy statements, 8-K material events

**Indian Companies (NSE/BSE):**
- NSE Annual Reports API
- BSE filing API
- Screener.in structured financials scraping
- NSE Quarterly Results
- Concall transcripts

**IR Website Scraping:**
- Discovers company IR page via common URL patterns
- Downloads PDFs, PPTX, DOCX from the IR page

**Supplemental:**
- `yfinance` for live pricing, historical data, ratios

**Document Deduplication:** MD5 checksums prevent re-downloading identical files.

**Rate Limiting:** 0.5 requests/second by default (configurable). SEC EDGAR enforces 10 req/sec limit; this platform stays well within it.

---

### Phase 3 — Document Processing & RAG Indexing

**PDF Processing Cascade:**
```
PyMuPDF (text-based PDF)
    ↓ (if fails or <100 words)
pdfplumber (layout-aware)
    ↓ (if still fails)
Tesseract OCR (scanned documents)
    ↓ (if still fails)
EasyOCR (alternative OCR engine)
```

**Table Extraction Cascade:**
```
Camelot (lattice mode — bordered tables)
    ↓ (if fails)
Camelot (stream mode — borderless tables)
    ↓ (if fails)
Tabula
```

**Chunking & Indexing:**
- Documents split into 512-token chunks with 64-token overlap
- Section-aware: income_statement / balance_sheet / cash_flow / mda / audit / related_party / governance
- Each chunk stored in ChromaDB (dense) and BM25 in-memory index

---

### Phase 4 — Financial Data Assembly

**Priority cascade for financial data:**
```
1. SQLite database (from prior runs or XBRL extraction)
2. Saved yfinance JSON files
3. Saved Screener.in JSON files
4. Live yfinance API call (last resort)
```

**Metrics extracted (30+ per year, up to 5 years):**

| Category | Metrics |
|----------|---------|
| Income Statement | revenue, cogs, gross_profit, ebit, ebitda, net_income, eps, sga, depreciation |
| Balance Sheet | total_assets, current_assets, inventory, accounts_receivable, cash_equivalents, total_liabilities, current_liabilities, accounts_payable, long_term_debt, shareholder_equity, retained_earnings, ppe_net |
| Cash Flow | cfo, capex, dividends_paid, free_cash_flow |
| Derived | working_capital, net_debt, gross_margin, net_margin, roa, roe |

---

### Phase 4b — Cross-Validation

10 internal consistency rules run automatically before agents start. Issues are:
- Logged to the audit trail
- Saved to the database
- Passed as context to all agents
- Included in reports

See [Section 7 — Cross-Validation Engine](#7-cross-validation-engine) for full rule details.

---

## 5. FORENSIC SCORING MODELS

### 5.1 Beneish M-Score

**Purpose:** Detect earnings manipulation using 8 financial ratios.

**Formula:**
```
M = -4.84 + 0.920×DSRI + 0.528×GMI + 0.404×AQI + 0.892×SGI
         + 0.115×DEPI - 0.172×SGAI + 4.679×TATA - 0.327×LVGI
```

**Variables:**

| Variable | Formula | What It Detects |
|----------|---------|----------------|
| DSRI | (AR_t/Sales_t) / (AR_t-1/Sales_t-1) | Days Sales Receivable Index — rising = channel stuffing |
| GMI | GrossMargin_t-1 / GrossMargin_t | Gross Margin Index — deterioration = pressure to manipulate |
| AQI | (1 - (CA+PPE)/TA)_t / (1 - (CA+PPE)/TA)_t-1 | Asset Quality Index — rising = soft asset accumulation |
| SGI | Sales_t / Sales_t-1 | Sales Growth Index — high growth = incentive to manipulate |
| DEPI | (Dep/(Dep+PPE))_t-1 / (Dep/(Dep+PPE))_t | Depreciation Index — declining = slowing depreciation |
| SGAI | (SGA/Sales)_t / (SGA/Sales)_t-1 | SGA Index — rising = disproportionate expense |
| TATA | (Net Income - CFO) / Total Assets | Total Accruals to Assets — high = earnings not cash-backed |
| LVGI | (LTD+CL)_t/TA_t / (LTD+CL)_t-1/TA_t-1 | Leverage Index — rising = incentive to manipulate |

**Interpretation:**

| M-Score | Risk Level | Meaning |
|---------|-----------|---------|
| > -1.78 | **MANIPULATOR** (CRITICAL) | Strong statistical probability of earnings manipulation |
| -2.22 to -1.78 | **GREY ZONE** (HIGH) | Watch carefully |
| < -2.22 | **NON-MANIPULATOR** (LOW) | Within normal range |

**Manipulation Probability** (logistic function):
```
P = 1 / (1 + e^(-M - (-0.92)))
```

---

### 5.2 Altman Z-Score

**Purpose:** Predict bankruptcy and financial distress.

**Three Model Variants:**

| Model | Companies | Formula |
|-------|-----------|---------|
| Manufacturing/Public | Listed manufacturing | Z = 1.2×X1 + 1.4×X2 + 3.3×X3 + 0.6×X4 + 1.0×X5 |
| Non-Manufacturing | Services, tech | Z' = 6.56×X1 + 3.26×X2 + 6.72×X3 + 1.05×X4 |
| Private | Unlisted | Z'' = 0.717×X1 + 0.847×X2 + 3.107×X3 + 0.420×X4 + 0.998×X5 |

**Variables:**

| Variable | Formula |
|----------|---------|
| X1 | Working Capital / Total Assets |
| X2 | Retained Earnings / Total Assets |
| X3 | EBIT / Total Assets |
| X4 | Market Cap (or Book Equity) / Total Liabilities |
| X5 | Revenue / Total Assets |

**Zones (Manufacturing/Public model):**

| Z-Score | Zone | Implied Action |
|---------|------|---------------|
| > 2.99 | **SAFE** | Low distress probability |
| 1.81 – 2.99 | **GREY** | Monitor leverage |
| < 1.81 | **DISTRESS** | High bankruptcy probability |

**Implied Credit Ratings:**

| Z-Score | Implied Rating |
|---------|---------------|
| > 8.0 | AAA |
| 6.0 – 8.0 | AA |
| 4.5 – 6.0 | A |
| 3.5 – 4.5 | BBB |
| 2.5 – 3.5 | BB |
| 1.5 – 2.5 | B |
| 1.0 – 1.5 | CCC |
| < 1.0 | D (Default) |

---

### 5.3 Piotroski F-Score

**Purpose:** Assess financial strength across 9 binary criteria.

| Criterion | Category | Pass Condition |
|-----------|---------|---------------|
| F1: ROA | Profitability | ROA > 0 |
| F2: CFO | Profitability | CFO > 0 |
| F3: ΔROA | Profitability | ROA_t > ROA_t-1 |
| F4: Accrual | Profitability | CFO/Assets > ROA |
| F5: Leverage | Leverage/Liquidity | Long-term Debt ratio decreased |
| F6: Liquidity | Leverage/Liquidity | Current Ratio improved |
| F7: Dilution | Leverage/Liquidity | No new shares issued |
| F8: Gross Margin | Efficiency | Gross Margin improved |
| F9: Asset Turnover | Efficiency | Asset Turnover improved |

**Interpretation:**

| F-Score | Classification |
|---------|---------------|
| 8 – 9 | **STRONG** — high financial quality |
| 5 – 7 | **AVERAGE** |
| 3 – 4 | **WEAK** |
| 0 – 2 | **DISTRESSED** |

---

### 5.4 Dechow F-Score

**Purpose:** Logistic regression predicting probability of SEC Accounting and Auditing Enforcement Release (AAER).

**Formula:**
```
ln(p / 1-p) = -7.893 + 0.790×RSST + 2.518×ΔAR + 1.191×ΔInv
              + 1.979×SoftAssets + 0.171×ΔCashSales + (-0.932)×ΔROA
              + 1.029×Issued
```

**Key Variables:**

| Variable | Meaning |
|----------|---------|
| RSST Accrual | (ΔWC + ΔNon-current + ΔFin) / Avg Assets |
| ΔAR | Change in receivables / Avg Assets |
| ΔInventory | Change in inventory / Avg Assets |
| Soft Assets | (TA - Cash - PPE) / TA |
| ΔCash Sales | % change in cash sales |
| ΔROA | Change in Return on Assets |
| Issued | Binary: new equity or debt issued |

**Risk Thresholds:**

| F-Score | Risk Level |
|---------|-----------|
| > 0.10 | **VERY HIGH** — 10× base rate |
| > 0.025 | **HIGH** — 2.5× base rate |
| > 0.01 | **ELEVATED** — above base rate |
| ≤ 0.01 | **LOW** — at or below base rate (~1%) |

---

### 5.5 Accrual Analysis (Sloan 1996)

**Two Methods:**

**Balance Sheet Accrual:**
```
BS Accrual = ΔWorking Capital − ΔCash − Depreciation
BS Accrual Ratio = BS Accrual / Average Total Assets
```

**Cash Flow Accrual:**
```
CF Accrual = Net Income − CFO
CF Accrual Ratio = CF Accrual / Average Total Assets
```

**Quality Ratios:**

| Ratio | Formula | Good Level | Concern Level |
|-------|---------|-----------|---------------|
| Cash Earnings Ratio | CFO / Net Income | > 1.0x | < 0.70x |
| CFO/EBITDA | CFO / EBITDA | > 0.85x | < 0.70x |
| ΔAR/Revenue | Change in AR / Revenue | < 3% | > 5% |

---

## 6. RISK SCORING FRAMEWORK

### Composite Risk Score (0–100)

The platform produces a single 0–100 composite risk score. Higher = more risk.

**Score Composition:**

| Dimension | Weight | Derived From |
|-----------|--------|-------------|
| Fraud Indicators | 25% | Agent 6 (Beneish, Dechow, Piotroski, Altman, Accrual) |
| Earnings Quality | 20% | Agent 8 (accrual ratios, tax consistency, margin) |
| Cash Flow Quality | 20% | Agent 4 (FCF, CFO/NI, EBITDA conversion) |
| Governance | 15% | Agents 9, 10, 11 (RPT, auditor, management NLP) |
| Credit Risk | 10% | Agent 7 (leverage, coverage, liquidity) |
| Auditor Risk | 5% | Agent 10 (going concern, material weakness) |
| Management Credibility | 5% | Agent 11 (evasion, overconfidence, non-GAAP emphasis) |

**Investment Verdicts:**

| Risk Score | Verdict | Meaning |
|-----------|---------|---------|
| 0 – 24 | **BUY** | Strong fundamentals, clean governance, high earnings quality |
| 25 – 37 | **CAUTIOUS BUY** | Investable with monitoring; some minor flags |
| 38 – 49 | **MONITOR** | No immediate action; watch quarterly triggers |
| 50 – 59 | **CAUTION** | Multiple concerns; reduce position or avoid new entry |
| 60 – 74 | **AVOID** | Significant fraud or credit risk indicators |
| 75 – 100 | **STRONG AVOID** | High manipulation probability; systematic red flags |

### Risk Bands (Color Coding)

| Band | Score Range | Color |
|------|------------|-------|
| VERY LOW | 0–20 | Dark Green |
| LOW | 21–40 | Green |
| MODERATE | 41–60 | Orange |
| HIGH | 61–80 | Red |
| EXTREME | 81–100 | Dark Red |

### Finding Severity Levels

| Severity | Meaning | Examples |
|----------|---------|---------|
| **CRITICAL** | Imminent fraud or insolvency signal | Going concern; M-Score > -1.0; negative CFO with positive NI |
| **HIGH** | Serious red flag requiring investigation | M-Score > -1.78; DSO spike >30 days; interest coverage < 2.5x |
| **MEDIUM** | Monitor; investigate if persistent | Gross margin change >5pp; auditor tenure >10 years |
| **LOW** | Note-worthy; not immediately actionable | Minor accrual elevation; non-Big 4 auditor |
| **POSITIVE** | Green flag | Strong F-Score; clean Big 4 opinion; CFO/NI > 1.2x |

---

## 7. CROSS-VALIDATION ENGINE

The `CrossValidator` runs 10 internal consistency checks on financial statements before any agent analyzes them. It catches manipulation that passes individual model tests.

| Rule | Issue Type | Trigger |
|------|-----------|---------|
| 1 | `REVENUE_CFO_DIVERGENCE` | Revenue CAGR 40%+ above CFO CAGR over 3+ years |
| 2 | `AR_REVENUE_DIVERGENCE` | AR growing 25%+ faster than revenue in any single year |
| 3 | `INVENTORY_COGS_DIVERGENCE` | Inventory growing 30%+ faster than COGS |
| 4 | `RETAINED_EARNINGS_INCONSISTENCY` | RE change deviates 25%+ from (NI − Dividends) |
| 5 | `POOR_EBITDA_CFO_CONVERSION` | CFO/EBITDA < 0.40x in any year |
| 6 | `GROSS_MARGIN_JUMP` | Gross margin changes >7pp in one year |
| 7 | `TAX_RATE_ANOMALY` | Effective tax rate deviates 12%+ from company average |
| 8 | `UNDERINVESTMENT` | CapEx/Depreciation < 0.40x |
| 9 | `DEBT_REVENUE_DIVERGENCE` | Debt CAGR 50%+ above revenue CAGR over 3+ years |
| 10 | `BALANCE_SHEET_IMBALANCE` | Assets ≠ Liabilities + Equity by >5% |

**Output per issue:**
- `issue_type` — machine-readable identifier
- `severity` — CRITICAL / HIGH / MODERATE / LOW
- `description` — plain English explanation
- `evidence` — specific numbers and calculation
- `fiscal_year` — which year the issue occurs
- `confidence` — 0.75–0.92

---

## 8. DATA SOURCES & ACQUISITION

### US Companies (SEC EDGAR)

| Data | API Endpoint | Notes |
|------|-------------|-------|
| Company CIK | `efts.sec.gov/LATEST/search-index` | Free, no auth required |
| Filings list | `data.sec.gov/submissions/CIK{10-digit}.json` | 40+ year history |
| Financial facts (XBRL) | `data.sec.gov/api/xbrl/companyfacts/CIK{}.json` | 30+ financial metrics |
| Filing documents | `www.sec.gov/Archives/edgar/data/...` | Full filings |
| Full-text search | `efts.sec.gov/LATEST/search-index?q=...` | For specific content |

**Rate limit:** Configured to 0.5 req/sec; SEC allows up to 10 req/sec for compliant user agents.

### Indian Companies (NSE/BSE)

| Data | Source | Notes |
|------|--------|-------|
| Annual Reports | NSE India API | Last 5 years |
| Quarterly Results | NSE filings API | Last 8 quarters |
| Screener.in | Web scraping | Structured 10-year P&L, Balance Sheet, Cash Flow |
| Shareholding | BSE / NSE quarterly | Promoter, FII, DII holdings |
| BSE filings | BSE corporate filings API | Announcements, results |

### Financial Data Cascade

```
Priority 1: SQLite database (previous XBRL/Screener extraction)
Priority 2: Saved yfinance_annual.json
Priority 3: Saved screener_financials.json
Priority 4: Live yfinance API (last resort)
```

### Supported Company Types

| Type | Coverage |
|------|---------|
| US Listed (NYSE/NASDAQ) | Full EDGAR access; XBRL structured financials |
| Indian Listed (NSE) | NSE API + Screener.in; strong coverage |
| Indian Listed (BSE-only) | BSE API + Screener.in; good coverage |
| International (non-US/India) | yfinance supplemental only; limited |
| Private Companies | No acquisition; manual data entry required |

---

## 9. OUTPUT FORMATS

Four formats are generated by default. PPTX and HTML are implemented but disabled — uncomment their blocks in `reporting/report_compiler.py` to re-enable.

### 9.1 JSON Output *(always generated first)*
- **Format:** Structured JSON with all investigation data
- **Use case:** API integration; programmatic processing; archival; basis for all other formats

### 9.2 XLSX Report (Excel)
- **Generator:** openpyxl
- **Tabs:**

| Tab | Content |
|-----|---------|
| Executive Summary | One-page scorecard |
| Financial Statements | 5-year multi-section P&L, Balance Sheet, Cash Flow |
| Forensic Scores | Beneish, Altman, Piotroski, Dechow by year |
| Red Flags Registry | All red flags, sortable by severity |
| Beneish M-Score | 8-component breakdown with thresholds |
| Altman Z-Score | X1–X5 variables by year |
| Risk Dashboard | Composite risk with component breakdown |
| Management Questions | 50–80 management questions |
| Monitoring Framework | 8 quarterly metric triggers |

- **Use case:** Analyst workbook; further calculations

### 9.3 DOCX Report (Word)
- **Generator:** python-docx
- **Sections:** 25 sections covering full investigation
- **Target length:** 15,000–40,000 words
- **Use case:** Detailed due diligence documentation; legal record

### 9.4 PDF Report
- **Generator:** ReportLab
- **Sections:** Cover page, Executive Summary, Forensic Scores, Red Flags (sorted by severity), Green Flags, Financial Data Table (5-year), Agent Summaries, Final Verdict
- **Length:** 20–60 pages depending on findings
- **Use case:** Formal presentation to investment committee

### 9.5 PPTX Report (PowerPoint) — *disabled by default*
- **Generator:** python-pptx
- **Slides:** 7 (Title, Executive Summary, Risk Scorecard, Forensic Scores, Top Red Flags, Financial Highlights, Final Verdict)
- **Enable:** Uncomment the PPTX block in `reporting/report_compiler.py`
- **Use case:** Investment committee presentation

### 9.6 HTML Dashboard — *disabled by default*
- **Generator:** Custom HTML/CSS
- **Type:** Self-contained single-file (no external dependencies)
- **Contents:** Risk score banner, forensic scores, agent scores, red/green flag cards with severity badges, financial table, management questionnaire (top 20)
- **Enable:** Uncomment the HTML block in `reporting/report_compiler.py`
- **Use case:** Quick review; sharing via email

---

## 10. CONFIGURATION REFERENCE

All configuration is centralized in `config.py`. No configuration is scattered across files.

### LLM Configuration — Multi-Provider

The platform auto-detects providers in priority order. Set a key to enable that provider.

**Provider priority cascade:**
```
Groq → OpenAI → Anthropic → Gemini → Together → OpenRouter → LM Studio → Ollama → HF → Template
```

**Environment variables (set in `.env` file or shell):**
```bash
# ── Free Cloud Providers (recommended for Colab) ──────────────────
GROQ_API_KEY=gsk_...          # Groq (free) — 14,400 req/day, Llama 70B
GOOGLE_API_KEY=AIza...        # Gemini (free tier) — 1,500 req/day

# ── Paid Cloud Providers ──────────────────────────────────────────
OPENAI_API_KEY=sk-proj-...    # GPT-4o / GPT-4o-mini
ANTHROPIC_API_KEY=sk-ant-...  # Claude Opus / Haiku
TOGETHER_API_KEY=...          # Together AI (hosted open-source)
OPENROUTER_API_KEY=sk-or-...  # OpenRouter (100+ models)

# ── Local Providers (no key needed — auto-detected) ───────────────
# Ollama: install from https://ollama.com → ollama pull qwen2.5:7b
# LM Studio: install from https://lmstudio.ai → start local server

# ── Force a specific provider ──────────────────────────────────────
LLM_PROVIDER=groq             # Options: auto|groq|openai|anthropic|gemini|
                              #          together|openrouter|lmstudio|ollama|hf

# ── Override model names per provider ──────────────────────────────
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_FAST_MODEL=llama-3.1-8b-instant
GEMINI_MODEL=gemini-1.5-pro-latest
GEMINI_FAST_MODEL=gemini-2.0-flash
OLLAMA_MODEL=qwen2.5:7b
```

**LLM generation settings (in `config.py`):**
```python
temperature    = 0.1    # Low = deterministic; keep ≤ 0.2 for forensic analysis
max_tokens     = 4096
context_window = 32768
timeout        = 300    # seconds per LLM call
max_retries    = 3
```

**Override via CLI:**
```bash
python main.py "Infosys" --model qwen2.5:14b    # Ollama model name
```

### Forensic Thresholds (`ForensicThresholds`)

These are the academic thresholds used in scoring. Change only with documented rationale.

```python
beneish_threshold      = -1.78   # Manipulation above this
altman_safe_zone       = 2.99    # Below = grey/distress
altman_distress_zone   = 1.81
piotroski_strong       = 7       # 7–9 = strong
piotroski_weak         = 2       # 0–2 = distressed
dso_spike_threshold    = 30      # days YoY change
inventory_growth_threshold = 0.30 # % above revenue growth
dpo_stretch_threshold  = 30      # days YoY change
accrual_ratio_high     = 0.10
accrual_ratio_moderate = 0.05
cash_conversion_warning = 0.70   # CFO/EBITDA below this
```

### Risk Scoring Weights (`RISK_SCORE_WEIGHTS`)

```python
RISK_SCORE_WEIGHTS = {
    "fraud_indicators":      0.25,
    "earnings_quality":      0.20,
    "cash_flow_quality":     0.20,
    "governance":            0.15,
    "credit_risk":           0.10,
    "auditor_risk":          0.05,
    "management_credibility": 0.05,
}
```

> **Senior override:** Weights can be adjusted for specific mandates (e.g., credit-focused analysis may increase `credit_risk` weight to 0.25 and reduce `management_credibility` to 0.02).

### Acquisition Configuration (`AcquisitionConfig`)

```python
rate_limit = 0.5        # requests per second (SEC allows 10/sec)
years      = 5          # years of annual filings to acquire
```

**Override via CLI:**
```bash
python main.py "Infosys" --years 7
```

### Storage Configuration

```python
DATA_DIR    = Path("forensic_ai_data")     # or Google Drive on Colab
REPORTS_DIR = DATA_DIR / "Investigations"
```

**Override via environment:**
```bash
FORENSIC_AI_OUTPUT_DIR=/mnt/nas/forensic python main.py "Infosys"
```

### Known Fraud Case Database (`FRAUD_CASE_DATABASE`)

Pre-loaded reference cases for pattern-matching:

| Case | Year | Type | Key Signals |
|------|------|------|------------|
| Enron | 2001 | Revenue overstatement; special purpose entities | SPE abuse, off-balance-sheet, aggressive revenue |
| Wirecard | 2020 | Fictitious cash; fabricated revenue | Missing cash balance, trustee accounts, phantom revenue |
| Satyam | 2009 | Bank balance fabrication; fake receivables | Fictitious cash, false receivables, promoter loans |
| Luckin Coffee | 2020 | Fabricated sales figures | Revenue inflation, related party, opaque disclosures |
| Carillion | 2018 | Working capital manipulation; goodwill inflation | Negative CFO, stretched payables, goodwill |
| Steinhoff | 2017 | Multi-jurisdiction fraud; off-balance-sheet | Revenue manipulation, complex structures |

---

## 11. OPERATIONAL GUIDE

### A. Google Colab (Free — Recommended First-Time Setup)

1. Open `colab_setup.ipynb` in Google Colab
2. Get a free key: Groq at [console.groq.com](https://console.groq.com) or Gemini at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
3. Add key to Colab Secrets (🔑 icon in left sidebar): `GROQ_API_KEY` or `GOOGLE_API_KEY`
4. Run cells in order — reports save to `MyDrive/Forensic_Reports/`

### B. VSCode / Local — Quick Start (Cloud API)

```bash
# 1. Create virtual environment and install
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements-minimal.txt

# 2. Configure API key
cp .env.example .env
# Edit .env: add GROQ_API_KEY or GOOGLE_API_KEY

# 3. Check and run
python main.py --check
python main.py "Infosys"
```

### C. VSCode / Local — Full Offline Setup (Ollama)

```bash
# Install Ollama: https://ollama.com/download
ollama pull qwen2.5:7b         # Primary (7B, ~4.5 GB)
ollama pull phi3.5:3.8b        # Fast model (3.8B, ~2.3 GB)

pip install -r requirements.txt
python main.py "Infosys"        # Ollama auto-detected, no .env needed
```

### D. LM Studio (Local GUI Model Manager)

1. Install [LM Studio](https://lmstudio.ai), download any GGUF model via browser
2. Start local server: LM Studio → Local Server → Start Server
3. No API key needed — auto-detected at `localhost:1234`
4. `python main.py "Infosys"`

### Running Investigations

**Single Company (CLI):**
```bash
python main.py "Infosys"
python main.py "AAPL"
python main.py "RELIANCE INDUSTRIES" --verbose
```

**Multiple Companies (CLI):**
```bash
python main.py "Infosys" "TCS" "Wipro" "HCL Technologies"
```

**Batch File:**
```bash
python main.py --batch companies.txt --verbose
```

**Web Interface:**
```bash
streamlit run app.py            # Opens at http://localhost:8501
```

**Python API:**
```python
from agents.orchestrator import ForensicOrchestrator
orchestrator = ForensicOrchestrator()
result = orchestrator.investigate("Infosys", ticker="INFY")
print(f"Risk Score: {result['overall_risk_score']}/100")
print(f"Verdict: {result['verdict']}")
```

### Output Location

```
forensic_ai_data/Investigations/{CompanyName}/
├── Raw_Filings/           ← Downloaded PDFs, HTMLs
├── Parsed_Data/Text/      ← Extracted text
├── Financials/            ← Structured financial JSON
├── Agent_Outputs/         ← Per-agent JSON outputs
├── Charts/                ← Plotly chart exports
├── Reports/               ← Final reports (PDF, DOCX, XLSX, etc.)
├── Audit_Trail/           ← JSONL audit log + summary
├── Final_Output/          ← report_index.json
└── Knowledge_Base/        ← ChromaDB embeddings
```

### Expected Run Times

| Environment | LLM Backend | LLM Time | Total (incl. data) |
|-------------|------------|---------|-------------------|
| Colab (Groq Llama 70B) | Free cloud | 3–5 min | 8–15 min |
| Colab (Gemini 2.0 Flash) | Free cloud | 2–4 min | 7–12 min |
| VSCode (GPT-4o) | Paid cloud | 5–10 min | 10–18 min |
| VSCode (Ollama 7B, CPU) | Local | 15–35 min | 20–45 min |
| VSCode (Ollama 7B, GPU) | Local | 5–12 min | 10–20 min |
| Template mode (no LLM) | None | 0 min | 5–10 min |

> Document acquisition (download + parse) adds 3–8 min regardless of LLM backend. EDGAR-listed US companies are faster than Indian companies due to structured XBRL data.

---

## 12. CONTROL POINTS & OVERRIDE PROCEDURES

### Control Point 1: Company Identity Override

If the platform identifies the wrong company:

```bash
# Force a specific ticker
python main.py "HDFC" --company "HDFC Bank" 

# The CompanyLookup will fall back to manual entry
# Edit acquisition/company_lookup.py:
# Add to MANUAL_OVERRIDES dict:
# "HDFC": CompanyProfile(name="HDFC Bank", ticker="HDFCBANK.NS", exchange="NSE", ...)
```

### Control Point 2: Verdict Override

If the Director's verdict needs to be overridden by a senior analyst, the recommended procedure is:

1. Note the investigation ID from the audit trail
2. Document the override reason
3. Adjust the final risk score in `{company}/Final_Output/report_index.json`
4. Re-run report generation only:
   ```python
   from reporting.report_compiler import ReportCompiler
   # Load existing investigation data and regenerate with adjusted score
   ```
5. Log the override in the audit trail with `audit.log_manual_override()`

> **Important:** The platform's quantitative scores should never be altered. Only the final investment verdict can be overridden, with documented rationale.

### Control Point 3: Threshold Adjustment

If industry-specific thresholds are needed (e.g., banks have different DSO norms):

```python
# config.py — create an industry override
INDUSTRY_THRESHOLD_OVERRIDES = {
    "BANKING": {
        "beneish_threshold": -1.5,    # Banks have higher accruals structurally
        "dso_spike_threshold": 60,     # Loan portfolios change DSO norms
    },
    "REAL_ESTATE": {
        "accrual_ratio_high": 0.20,   # Project-based revenue recognition
    }
}
```

### Control Point 4: Agent Disable / Enable

To disable a specific Phase B agent (e.g., skip Related Party for a credit-only mandate):

```python
# In orchestrator.py, _run_all_agents(), edit phase_b_specs:
phase_b_specs = [
    (3,  "Revenue Forensics Agent",    RevenueForensicsAgent),
    (4,  "Cash Flow Forensics Agent",  CashFlowForensicsAgent),
    (5,  "Working Capital Agent",      WorkingCapitalAgent),
    (7,  "Credit Risk Agent",          CreditRiskAgent),
    (8,  "Earnings Quality Agent",     EarningsQualityAgent),
    # (9, "Related Party Agent", RelatedPartyAgent),  ← disabled
    (10, "Auditor Intelligence Agent", AuditorIntelligenceAgent),
    (11, "Management NLP Agent",       ManagementNLPAgent),
]
```

To disable the Investment Committee Perspectives (bear/bull/devil):

```python
# In orchestrator.py, _run_all_agents():
# Comment out the perspectives block (lines starting with "results[14] = ...")
# results[14] = self._run_perspectives(...)
```

Phase C currently runs only Agent 12 (Peer Comparison) via `phase_c_configs`. To add a new LLM-only agent to Phase C, add an entry to `phase_c_configs` and extend `role_map` / `question_map` in `_run_generic_agent()`.

### Control Point 5: LLM Model Override

For high-stakes investigations requiring maximum LLM capability:

```bash
# Use a larger model for the primary analysis
python main.py "Infosys" --model qwen2.5:14b

# Or set the OLLAMA_PRIMARY_MODEL environment variable:
set OLLAMA_PRIMARY_MODEL=qwen2.5:14b
python main.py "Infosys"
```

### Control Point 6: Data Quality Threshold

If financial data quality is too low (< 50% of required fields), agents log a warning and operate from text only. To enforce a minimum data quality threshold:

```python
# In orchestrator.py investigate():
if financial_data_quality < 0.50:
    raise InsufficientDataError(
        f"Financial data quality {financial_data_quality:.0%} below minimum threshold"
    )
```

### Control Point 7: Rate Limiting

If the platform is being used in a shared environment:

```python
# config.py
@dataclass
class AcquisitionConfig:
    rate_limit: float = 0.5   # Increase to 1.0 for slower acquisition
```

---

## 13. AUDIT TRAIL & EVIDENCE CHAIN

### Audit Trail Design

Every finding, calculation, and agent action is logged to an **immutable JSONL file**. This creates a complete, tamper-evident evidence chain.

**File location:** `{company}/Audit_Trail/investigation_log.jsonl`

At session close, `audit_summary.json` is written alongside it. This is a **lightweight stats file** (entry counts, agent list, timing, path to the JSONL) — it does not duplicate the full entry list, which lives only in the JSONL.

**Each JSONL entry:**
```json
{
  "timestamp": "2026-06-09T14:23:45.123456",
  "session_id": "sess_20260609_142340",
  "agent_id": 6,
  "agent_name": "Fraud Detection Agent",
  "action": "RED_FLAG: Beneish M-Score: -1.23 - Manipulation Likely",
  "finding": "M-Score of -1.23 exceeds -1.78 threshold...",
  "evidence": "M-Score = -1.2340. Components: DSRI=1.42, GMI=0.89...",
  "source_document": "infosys_10k_2024.pdf",
  "confidence": 0.85,
  "risk_level": "HIGH",
  "calculation": "M = -4.84 + 0.920×1.42 + 0.528×0.89 + ... = -1.2340"
}
```

### What is Logged

| Event Type | When | Fields |
|-----------|------|-------|
| `COMPANY_IDENTIFIED` | Phase 1 | name, ticker, exchange, source_url |
| `DOCUMENT_ACQUIRED` | Phase 2 | filename, source_url, checksum, size |
| `CALCULATION` | Phase 5 | model, inputs, result, formula |
| `RED_FLAG` | Phase 5 | agent, title, evidence, severity, source_doc |
| `GREEN_FLAG` | Phase 5 | agent, title, evidence, source_doc |
| `OBSERVATION` | Phase 5 | agent, finding, context |
| `VERDICT` | Phase 6 | risk_score, verdict, rationale |

### Database Schema (SQLite)

13 tables store all investigation data:

```
companies          — Company master record
documents          — Downloaded filing metadata
financial_data     — 30+ metrics per company per year
forensic_scores    — Beneish, Altman, Piotroski, Dechow scores
agent_findings     — All RED_FLAG / GREEN_FLAG / OBSERVATION findings
red_flags          — Filtered red flags with severity
green_flags        — Positive indicators
audit_trail        — Redundant copy of JSONL for SQL queries
related_party_transactions — RPT amounts and parties
auditor_history    — Auditor name, tenure, qualification history
shareholding_data  — Promoter, FII, DII holdings quarterly
concall_intelligence — Management statement analysis
investigation_sessions — Session timing and outcomes
```

### Querying the Audit Trail

```python
from database.sqlite_handler import SQLiteHandler
db = SQLiteHandler()

# All red flags for a company
flags = db.execute_query("""
    SELECT af.finding_title, af.risk_level, af.confidence, af.fiscal_year
    FROM agent_findings af
    JOIN companies c ON af.company_id = c.id
    WHERE c.name LIKE '%Infosys%' AND af.finding_type = 'RED_FLAG'
    ORDER BY af.risk_level, af.confidence DESC
""")

# Historical forensic scores
scores = db.execute_query("""
    SELECT fiscal_year, beneish_m_score, altman_z_score, piotroski_f_score
    FROM forensic_scores
    WHERE company_id = ?
    ORDER BY fiscal_year DESC
""", (company_id,))
```

---

## 14. LIMITATIONS & CAVEATS

### Data Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| **Indian companies pre-2018** | Screener.in coverage limited | Manual data entry or Bloomberg supplement |
| **Private companies** | No public filings; no acquisition | Must provide financial data manually |
| **Non-English filings** | OCR + NLP quality degrades | Multi-language models not yet integrated |
| **Restatements** | Historical data may contain pre-restatement figures | Always verify against latest filing |
| **Non-standard fiscal years** | Year mapping may be imprecise | Verify fiscal year end dates |

### Model Limitations

| Model | Limitation |
|-------|-----------|
| **Beneish M-Score** | Calibrated on 1990s US GAAP; may have false positives for rapidly growing tech companies |
| **Altman Z-Score** | Originally calibrated on manufacturing firms; financial sector companies require different model |
| **Piotroski F-Score** | Not designed for financial companies (banks, NBFCs) |
| **Dechow F-Score** | Logistic model based on SEC enforcement actions; regulatory differences outside US may affect base rate |

### LLM Limitations

| Risk | Mitigation |
|------|-----------|
| **Hallucination in LLM narrative** | All factual claims backed by quantitative engine output; LLM provides interpretation only |
| **Model knowledge cutoff** | Cloud models (GPT-4o, Gemini, Groq Llama) have recent cutoffs. Financial data comes from live sources (yfinance, EDGAR) so numerical analysis reflects current figures even if LLM general knowledge lags. |
| **Template mode quality** | If no LLM is available, narrative sections are replaced by structured templates |
| **Prompt sensitivity** | System prompts are fixed per agent role; not user-editable without code changes |
| **Free-tier rate limits** | Groq: 14,400 req/day; Gemini: 1,500 req/day. Batch investigations > 5 companies may hit limits. Use paid tiers or local Ollama for high volume. |
| **Data privacy (cloud APIs)** | Financial data is sent to the cloud provider if using Groq/OpenAI/Anthropic/Gemini. For sensitive mandates use Ollama or LM Studio (local only). |

### Not Intended For

- **High-frequency or real-time analysis** — this is a point-in-time forensic investigation tool
- **Credit scoring** — not a substitute for formal credit ratings
- **Legal evidence** — output must be reviewed by qualified professionals before use in legal proceedings
- **Retail investor use** — designed for institutional analysts with domain expertise

---

## 15. ESCALATION & REVIEW PROCEDURES

### When to Escalate to Senior Review

| Condition | Action |
|-----------|--------|
| Risk score > 70 | Mandatory senior analyst review before verdict is communicated |
| CRITICAL cross-validation issue detected | Pause investigation; verify raw data before proceeding |
| Going concern or material weakness detected | Immediate escalation; do not share preliminary report |
| Risk score in 38–62 grey zone after refinement | Investment committee discussion required |
| Any finding with confidence < 0.60 | Flag as unverified; independent verification required |

### Review Checklist for Senior Analyst

When reviewing a completed investigation, the senior analyst should:

**Step 1 — Verify Company Identity**
- [ ] Confirm company name, ticker, and exchange are correct
- [ ] Check that the fiscal year dates match the company's actual fiscal year

**Step 2 — Verify Data Quality**
- [ ] Review `Agent_Outputs/` folder for data quality scores
- [ ] If data quality < 60%, treat quantitative scores as indicative only
- [ ] Check cross-validation issues for CRITICAL balance sheet imbalances

**Step 3 — Review Quantitative Scores**
- [ ] Beneish M-Score: are the 8 component values plausible?
- [ ] Altman Z-Score: check X4 (Market Cap / Liabilities) — is market cap current?
- [ ] Piotroski: note which specific criteria (F1–F9) are failing

**Step 4 — Review Agent Findings**
- [ ] Sort red flags by severity; review all CRITICAL flags personally
- [ ] Check if any CRITICAL flag has a confidence below 0.75 — if so, verify the source document
- [ ] Review Agent 16 (Devil's Advocate) output to understand the counter-case

**Step 5 — Verify Audit Trail**
- [ ] Spot-check 3–5 red flag source document citations
- [ ] Confirm all quantitative calculations have `calculation` field populated

**Step 6 — Issue or Modify Verdict**
- [ ] If verdict requires modification, use the override procedure in Section 12
- [ ] Document override rationale in the audit trail

### Management Questions (Agent 17 Output)

The Director generates 50–80 tailored questions for management. These are organized into categories:

| Category | Examples |
|----------|---------|
| Revenue Quality | "What percentage of FY{year} revenue was recognized in Q4?" |
| Receivables | "What is the age-wise breakdown of outstanding receivables?" |
| Cash Flow | "Why has CFO grown slower than net income over the past 3 years?" |
| Related Parties | "What is the business rationale for the {transaction} with {related entity}?" |
| Auditor | "What specifically triggered the Key Audit Matter on revenue recognition?" |
| Governance | "Has the board's audit committee reviewed the related party transactions?" |

### Quarterly Monitoring Triggers

The Director also sets 8 quarterly triggers. Investigation should be re-run if any trigger fires:

| Metric | Alert Threshold |
|--------|----------------|
| DSO change | > +15 days QoQ |
| Inventory days change | > +20 days QoQ |
| CFO/NI ratio | Drops below 0.70x |
| Net Debt/EBITDA | Increases by > 0.5x in one quarter |
| Gross Margin | Changes > 3pp QoQ |
| Auditor qualification | Any new qualification |
| Promoter shareholding | Drops > 2% in one quarter |
| Cash balance | Drops > 20% with no disclosed reason |

---

## 16. GLOSSARY

| Term | Definition |
|------|-----------|
| **AAER** | Accounting and Auditing Enforcement Release — SEC enforcement action against fraudulent reporting |
| **Accrual** | Recognized revenue/expense not yet received/paid as cash. High accruals = earnings quality risk |
| **BM25** | Best Match 25 — probabilistic keyword search algorithm used for sparse retrieval |
| **CCC** | Cash Conversion Cycle = DSO + DIO − DPO. Lower = better working capital efficiency |
| **CFO** | Cash Flow from Operations — actual cash generated from business operations |
| **CIK** | Central Index Key — SEC's unique company identifier (e.g., Infosys = CIK 1067491) |
| **Cross-Validation** | Internal consistency checks across financial statements to catch manipulation |
| **Dechow F-Score** | Logistic regression model predicting AAER probability (misreporting probability) |
| **DIO** | Days Inventory Outstanding = Inventory / (COGS/365). Rising = potential inflation |
| **DPO** | Days Payable Outstanding = Payables / (COGS/365). Rising suddenly = liquidity stress |
| **DSO** | Days Sales Outstanding = Receivables / (Revenue/365). Spike = channel stuffing risk |
| **EBITDA** | Earnings Before Interest, Tax, Depreciation, and Amortisation |
| **ETR** | Effective Tax Rate = Income Tax Expense / Pre-tax Income |
| **FCF** | Free Cash Flow = CFO − CapEx |
| **Going Concern** | Audit qualification: doubt about company's ability to survive next 12 months |
| **ICFR** | Internal Controls over Financial Reporting (SOX Section 404) |
| **KAM** | Key Audit Matter — area where auditors exercised significant judgment |
| **LLM** | Large Language Model (e.g., Qwen2.5, Phi-4 Mini, Llama 3.2) |
| **M-Score** | Beneish M-Score — earnings manipulation indicator |
| **Material Weakness** | Deficiency in internal controls that could allow material misstatement |
| **NLP** | Natural Language Processing — used in Agent 11 for management language analysis |
| **OCR** | Optical Character Recognition — used for scanned PDF documents |
| **Ollama** | Local LLM server; runs models on user's machine without internet |
| **RAG** | Retrieval-Augmented Generation — search-then-generate approach using document knowledge base |
| **Restatement** | Correction of previously issued financial statements |
| **RRF** | Reciprocal Rank Fusion — combines sparse (BM25) and dense (vector) search results |
| **RSST Accrual** | Richardson-Sloan-Soliman-Tuna accrual measure used in Dechow F-Score |
| **RPT** | Related Party Transaction — dealings between company and connected persons/entities |
| **TATA** | Total Accruals to Total Assets — key Beneish variable |
| **XBRL** | eXtensible Business Reporting Language — structured financial data format used by SEC |
| **Z-Score** | Altman Z-Score — financial distress and bankruptcy prediction model |

---

*End of Forensic AI Technical Reference Manual v1.2*

*This document is classified as Internal — Restricted. Distribution is limited to authorized analysts and reviewers. This platform produces research-grade output to assist human analysts; all findings must be reviewed by qualified professionals before being acted upon. This is not financial advice.*

---
**Document Control**

| Field | Value |
|-------|-------|
| Version | 1.2 |
| Status | Active |
| Review Cycle | Quarterly |
| Owner | Platform Administrator |
| Distribution | Forensic Analysts, Senior Analysts, Risk Committee, Compliance |
