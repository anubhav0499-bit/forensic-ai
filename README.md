# Forensic AI — Autonomous Forensic Accounting Platform

**Institutional-grade 17-agent autonomous forensic accounting system.**  
Give it a company name. It downloads public filings, runs 17 specialist AI agents across a multi-layer RAG pipeline, and returns a 0–100 risk score, an investment verdict, and full PDF/DOCX/XLSX/JSON reports.

---

## Table of Contents

1. [What It Does](#1-what-it-does)
2. [System Architecture](#2-system-architecture)
3. [RAG Pipeline — 13 Advanced Modules](#3-rag-pipeline)
4. [17 Specialist Agents](#4-18-specialist-agents)
5. [Choose Your Setup Path](#5-choose-your-setup-path)
6. [Path A — Google Colab (Free, No Install)](#6-path-a--google-colab-free-no-install)
7. [Path B — Local with Cloud API](#7-path-b--local-with-cloud-api)
8. [Path C — Fully Offline with Ollama](#8-path-c--fully-offline-with-ollama)
9. [Running an Investigation](#9-running-an-investigation)
10. [Understanding the Output](#10-understanding-the-output)
11. [Configuration Reference](#11-configuration-reference)
12. [Output Files and Folders](#12-output-files-and-folders)
13. [Forensic Scoring Models](#13-forensic-scoring-models)
14. [Technical References](#14-technical-references)
15. [Troubleshooting](#15-troubleshooting)
16. [FAQ](#16-faq)

---

## 1. What It Does

When you submit a company name (e.g. `"Infosys"`, `"AAPL"`, `"RELIANCE"`):

```
Your input
    → Company identification (Yahoo Finance / SEC EDGAR / NSE)
    → Document acquisition (annual reports, 10-Ks, quarterly results, transcripts)
    → PDF parsing + financial table extraction
    → Core Knowledge Base (5-layer: global standards, Indian regulations, fraud library)
    → Hybrid RAG (FAISS + ChromaDB + BM25 + BGE embeddings + Multi-Vector + RRF)
    → 17 specialist AI agents in 4 phases
    → Cross-validation of financial statements (10 internal consistency rules)
    → Guardrails (groundedness + hallucination detection + standard validation)
    → RAGAS evaluation of context quality
    → Chief Director synthesises all findings with iterative refinement
    → PDF, DOCX, XLSX, and JSON reports
    → Risk score 0–100 + Investment verdict
```

**Investment verdicts:**

| Score | Verdict | Meaning |
|-------|---------|---------|
| 0–24 | **BUY** | Clean financials, strong governance |
| 25–37 | **CAUTIOUS BUY** | Investable; minor flags to watch |
| 38–49 | **MONITOR** | No immediate action; quarterly review |
| 50–59 | **CAUTION** | Multiple concerns; avoid new entry |
| 60–74 | **AVOID** | Significant fraud or credit risk signals |
| 75–100 | **STRONG AVOID** | High manipulation probability |

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         FORENSIC AI PLATFORM v1.4                        │
├──────────────┬───────────────┬───────────────────┬────────────────────────┤
│  ACQUISITION │  PROCESSING   │     RAG LAYER     │   FORENSIC ENGINES     │
│              │               │                   │                        │
│ SEC EDGAR    │ PyMuPDF       │ FAISS (primary)   │ Beneish M-Score       │
│ NSE/BSE      │ pdfplumber    │ ChromaDB (backup) │ Altman Z-Score        │
│ IR Scraper   │ Camelot/Tabula│ BM25Okapi         │ Piotroski F-Score     │
│ yfinance     │ Tesseract OCR │ BGE-large (1024d) │ Dechow F-Score        │
│ Screener.in  │ RecursiveChunk│ Multi-Vector      │ Accrual Analysis      │
│              │ SemanticChunk │ HybridRRF         │ Working Capital        │
│              │               │ CrossEncoder      │ Cross-Validator (10)  │
│              │               │ ContextCompressor │ Risk Scorer           │
│              │               │ ConvMemory        │                        │
│              │               │ Guardrails        │                        │
├──────────────┴───────────────┴───────────────────┴────────────────────────┤
│                    CORE KNOWLEDGE BASE (5 Layers)                         │
│  ISA/SA/PCAOB standards · SEBI/MCA regulations · Fraud case library (10) │
│  Reasoning framework · Audit procedure templates · Agent-to-standard map  │
├───────────────────────────────────────────────────────────────────────────┤
│                         17-AGENT LAYER                                    │
│                                                                           │
│  Phase A         Phase B (8 agents, parallel)       Phase C (parallel)   │
│  ---------       --------------------------------    ------------------- │
│  Agent 6:        Agent 3: Revenue Forensics          Agent 12: Peer Comp │
│  Fraud Detect    Agent 4: Cash Flow Forensics        Agent 14: Inv. Cmte │
│                  Agent 5: Working Capital                                 │
│                  Agent 7: Credit Risk                                     │
│                  Agent 8: Earnings Quality            Phase D (Director)  │
│                  Agent 9: Related Party               -----------------   │
│                  Agent 10: Auditor Intelligence       Agent 17: Chief Dir │
│                  Agent 11: Management NLP                                  │
├───────────────────────────────────────────────────────────────────────────┤
│                   AGENTIC RAG (LangGraph 12-step loop)                    │
│  query_rewriter → detail_check → source_router → retriever →             │
│  generator → relevance_check → (loop up to 3 iterations)                │
├───────────────────────────────────────────────────────────────────────────┤
│                        DATA LAYER                                         │
│  SQLite (WAL) — 13 tables  ·  DuckDB — analytics  ·  JSONL audit trail   │
│  FAISS — local vector index  ·  ChromaDB — persistent embeddings         │
├───────────────────────────────────────────────────────────────────────────┤
│                      REPORTING LAYER                                      │
│   PDF  │  DOCX  │  XLSX (9 tabs)  │  PPTX*  │  HTML*  │  JSON           │
│                                              (* disabled by default)      │
└───────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM (Cloud Free)** | Groq (Llama 70B) / Gemini 2.0 Flash | Ultra-fast free-tier; ideal for Colab |
| **LLM (Cloud Paid)** | OpenAI GPT-4o / Anthropic Claude / Together AI | Highest quality |
| **LLM (Local)** | Ollama + Qwen2.5:7B / LM Studio | Fully offline |
| **LLM (Fallback)** | Template Mode | Deterministic output without LLM |
| **Embeddings (primary)** | BAAI/bge-large-en-v1.5 (1024-dim) | Best quality embedding |
| **Embeddings (fast)** | BAAI/bge-small-en-v1.5 (384-dim, ~130 MB) | 4x faster |
| **Vector DB (primary)** | FAISS IndexFlatIP | High-speed local similarity search |
| **Vector DB (backup)** | ChromaDB | Persistent cosine-distance store |
| **Sparse Retrieval** | BM25Okapi | Keyword-based retrieval |
| **Hybrid Fusion** | RRF (k=60) | BM25 (0.4) + Dense (0.6) weights |
| **Reranking** | ms-marco-MiniLM-L-6-v2 | Cross-encoder precision reranking |
| **Multi-Vector** | Raw + LLM summary embeddings | Dual-representation retrieval |
| **Chunking** | Recursive + Semantic + Section-aware | Three chunking modes |
| **Agentic RAG** | LangGraph StateGraph | 12-step query→retrieve→generate loop |
| **Relational DB** | SQLite (WAL mode, 13 tables) | Findings, companies, sessions |
| **Analytics DB** | DuckDB | Trend analysis, peer benchmarking |
| **Orchestration** | ThreadPoolExecutor | Parallel Phase B agent execution |
| **UI** | Streamlit | Web interface at localhost:8501 |
| **CLI** | Python argparse | Command-line interface |

---

## 3. RAG Pipeline

The platform implements 13 advanced RAG structures:

### 3.1 Embeddings — BAAI/bge-large-en-v1.5
Three-tier waterfall:
- `EmbeddingModel()` → bge-large-en-v1.5 (1024-dim, best quality)
- `EmbeddingModel(fast=True)` → bge-small-en-v1.5 (384-dim, ~4x faster, ~130 MB)
- Automatic fallback to all-MiniLM-L6-v2 if BGE unavailable

### 3.2 FAISS Vector Store
- `IndexFlatIP` with L2-normalised vectors (inner product = cosine similarity)
- Persistent as `.faiss` binary + `_meta.pkl` metadata file
- Opt-in via `USE_FAISS=true` in `.env`; drop-in alternative to ChromaDB

### 3.3 Hybrid Retrieval (BM25 + Dense + RRF)
- BM25Okapi weight: **0.4**; Dense BGE weight: **0.6**
- Reciprocal Rank Fusion constant k=60 (Cormack et al. 2009)
- `search_multi()`: multi-query RRF merge for query diversity
- `search_multi_reranked()`: + cross-encoder precision pass

### 3.4 Smart Chunking — Three Modes
- **DocumentChunker**: section-aware + sentence-boundary snapping (default)
- **RecursiveChunker**: 4-level separator hierarchy (`\n\n` → `\n` → sentence → word)
  with parent→child ID links enabling multi-vector parent retrieval
- **SemanticChunker**: splits at cosine-similarity drops < 0.75 between adjacent
  sentences (requires BGE embedder; auto-falls back to Recursive)

### 3.5 Multi-Vector Retrieval
- Each chunk embedded twice: **raw content** + **LLM 1-sentence summary**
- Summary embedding captures gist; raw embedding captures exact phrasing
- Results merged via RRF (raw weight 0.6, summary weight 0.4)

### 3.6 Cross-Encoder Reranking
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (lazy-loaded, cached per instance)
- Applied after RRF: rescores candidate pool with query-passage pair scores
- Available via `search_with_reranking()` and `search_multi_reranked()`

### 3.7 Context Compression
- **LLM path**: extracts only query-relevant sentences per source chunk
- **Keyword path** (no LLM): TF-overlap scoring; keeps top 50% sentences by relevance
- Preserves `[Source N: doc, FY, section]` provenance headers through compression
- Enable: `ContextBuilder(compressor=ContextCompressor(llm))`

### 3.8 Query Rewriting — Agentic RAG (LangGraph)
- 12-step LangGraph loop: `query_rewriter → detail_check → source_router →
  retriever → generator → relevance_check` (up to 3 iterations)
- HyDE support in LlamaIndex pipeline (hypothetical document embedding)
- Falls back to classic RAG if LangGraph unavailable; enable via `AGENTIC_RAG_ENABLED=true`

### 3.9 Conversation Memory
- SQLite-backed per `(session_id, company_name, agent_id)`
- `make_session_id()` — one session per company per day
- `build_history_context()` — formatted history block for LLM injection
- `MAX_CONTENT_CHARS=800` per turn to bound context window usage

### 3.10 Guardrails — Three Layers
- **Layer 1 — Groundedness**: token overlap ≥ 0.20 between answer and context
- **Layer 2 — Standard validation**: ISA/SA/IAS/PCAOB number validation against
  60+ valid codes; invented standards flagged as HALLUCINATION_RISK
- **Layer 3 — Overconfidence detection**: keyword scan for "definitely", "certainly", etc.
- Risk levels: CRITICAL / HIGH / MEDIUM / LOW; scores attached to every `HarnessResult`

### 3.11 RAGAS Evaluation
- 10 built-in forensic test questions per company
- Metrics: context_relevancy (30%), faithfulness (35%), answer_relevancy (25%), context_recall (10%)
- Uses ragas library when installed; custom overlap-based fallback otherwise
- Results saved to `DATA_DIR/eval_results/ragas_{company}.json`

### 3.12 SSE Streaming
- `stream_sse()` → `AsyncGenerator[str, None]` for FastAPI `StreamingResponse`
- `stream_sync()` → `Generator[str, None, None]` for Streamlit `st.write_stream()`
- Frame format: `data: {"token": "...", "done": false}\n\n`
- Native provider streaming first (Groq/OpenAI/Anthropic); chunked fallback for others

### 3.13 Core Knowledge Base
Injected into every agent's LLM call via `get_agent_knowledge_block(agent_id)`:
- **Layer 1**: Global Audit Standards (ISA 240, 315, 520, 570; PCAOB AS 2401)
- **Layer 2**: Indian Regulatory Standards (SEBI LODR, CARO 2020, Companies Act 2013)
- **Layer 3**: Historical Fraud Repository (10 cases: Satyam, IL&FS, DHFL, Yes Bank,
  Enron, WorldCom, Wirecard, Luckin Coffee, Carillion, Steinhoff)
- **Layer 4**: Financial Statement Intelligence (Beneish, Dechow, Sloan thresholds)
- **Layer 5**: Regulatory Intelligence (SEBI SAST, RBI stressed asset framework, IBC 2016)

---

## 4. 17 Specialist Agents

### Phase A — Forensic Baseline (Sequential, First)

| Agent | Role | Engines |
|-------|------|---------|
| **6 · Fraud Detection** | Beneish M-Score, Altman Z-Score, Piotroski F-Score, Dechow F-Score, Sloan Accruals; compares against 10-case fraud library | Quantitative engines + RAG |

### Phase B — Specialist Agents (8 agents, parallel)

| Agent | Domain | Key Signals | Standard |
|-------|--------|------------|----------|
| **3 · Revenue Forensics** | AR/revenue gap, Q4 skew, deferred revenue, CAGR divergence | DSO spike, channel stuffing | ISA 240 §A.26 |
| **4 · Cash Flow Forensics** | CFO vs. NI divergence, FCF quality, CapEx sustainability | Negative FCF, CFO/EBITDA < 0.4 | PCAOB AS 2401 §66 |
| **5 · Working Capital** | DSO / DIO / DPO / CCC multi-year trends | Rising CCC, DEPI < 1 | Beneish DSRI/DEPI |
| **7 · Credit Risk** | Leverage, interest coverage, liquidity, implied credit rating | ICR < 1.5x, Z < 1.81 | Altman EM Z-Score |
| **8 · Earnings Quality** | Accrual ratios, ETR anomalies, margin consistency | Accrual ratio > 0.10 | Dechow & Dichev (2002) |
| **9 · Related Party** | RPT concentration, promoter loans, pledge levels, disclosure quality | Pledge > 50%, undisclosed RPT | SEBI RPT Reg. 2021 |
| **10 · Auditor Intelligence** | Going concern, material weaknesses, KAMs, auditor changes | Big-4 → small-firm switch | ISA 570, 240 §A50 |
| **11 · Management NLP** | Evasion language, hedging, non-GAAP overemphasis | Word-list NLP + RAG | ISA 240 §A4 |

### Phase C — Synthesis Agents (Full prior context)

| Agent | Role |
|-------|------|
| **12 · Peer Comparison** | Benchmarks all metrics against industry peers; flags statistical outliers |
| **14 · Investment Committee** | Three perspectives in one: Bear case + Bull case + Devil's Advocate (three sequential LLM calls) |

### Phase D — Director Synthesis

| Agent | Role |
|-------|------|
| **17 · Chief Director** | Aggregates all 17 agents → composite risk score → final verdict → 50–80 management questions → 8 quarterly monitoring triggers. Triggers extra "Resolve Ambiguity" pass if score in 38–62 grey zone. |

---

## 5. Choose Your Setup Path

| Path | Where | LLM Cost | Setup Time | Best For |
|------|-------|----------|-----------|---------|
| **A — Google Colab** | Browser | Free | ~5 min | First-time use; no local install |
| **B — Local + Cloud API** | Your machine | Free (Groq/Gemini) or paid | ~10 min | Daily use; faster than Colab |
| **C — Local + Ollama** | Your machine | Free, offline | ~20 min | Privacy; no API keys; air-gapped |

---

## 6. Path A — Google Colab (Free, No Install)

**Requirements:** A Google account. Nothing else.

### Step 1 — Get a free API key

**Option 1 — Groq** (recommended: 14,400 free requests/day):
1. Go to [console.groq.com](https://console.groq.com) → sign up → Create API key (starts with `gsk_`)

**Option 2 — Gemini** (1,500 free requests/day):
1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) → Create API key (starts with `AIza`)

### Step 2 — Open the notebook

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. File → Upload notebook → upload `colab_setup.ipynb`

### Step 3 — Add your API key to Colab Secrets

1. Click the **key icon** in the left sidebar → **+ Add new secret**
2. Name: `GROQ_API_KEY` (or `GOOGLE_API_KEY`) → Value: paste key → toggle **Notebook access** ON

### Step 4 — Run the cells in order

| Cell | Action | Time |
|------|--------|------|
| 1 | Install packages | ~2 min |
| 2 | Mount Drive, load API key | ~30 sec |
| 3 | Import platform, verify LLM | ~30 sec |
| 4 | Run investigation | 8–15 min |

Reports save to `My Drive → Forensic_Reports → {CompanyName}/`

---

## 7. Path B — Local with Cloud API

**Requirements:** Python 3.10+, pip. A Groq or Gemini API key (free).

```bash
# Clone
git clone https://github.com/anubhav0499-bit/forensic-ai.git forensic_ai
cd forensic_ai

# Virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Mac/Linux

# Install
pip install -r requirements-minimal.txt

# Configure
copy .env.example .env       # Windows
cp .env.example .env         # Mac/Linux
# Edit .env: uncomment GROQ_API_KEY=gsk_your_key_here

# Verify
python main.py --check

# Run
python main.py "Infosys"
```

Reports save to `~/Documents/Forensic_Reports/Infosys/`

---

## 8. Path C — Fully Offline with Ollama

**Requirements:** Python 3.10+, ~8 GB RAM, ~10 GB disk. No internet after setup.

```bash
# Install Ollama from https://ollama.com/download

# Download a model
ollama pull qwen2.5:7b       # 7B, ~4.5 GB (recommended)
# or
ollama pull phi3.5:3.8b      # 3.8B, ~2.3 GB (low-RAM machines)

# Install packages
cd forensic_ai
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements-minimal.txt

# Run (no .env needed — Ollama auto-detected at localhost:11434)
python main.py "Infosys"
```

*Ollama on CPU: 15–35 min per investigation. GPU: 5–12 min.*

---

## 9. Running an Investigation

### CLI

```bash
# Single company
python main.py "Infosys"
python main.py "AAPL"
python main.py "RELIANCE INDUSTRIES" --verbose

# Multiple companies
python main.py "Infosys" "TCS" "Wipro"

# Batch file
python main.py --batch companies.txt

# CLI options
python main.py "Infosys" --years 7 --output /mnt/reports --model qwen2.5:14b
```

### Web Interface

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### Python API

```python
from agents.orchestrator import ForensicOrchestrator

orchestrator = ForensicOrchestrator()
result = orchestrator.investigate("Infosys", ticker="INFY")

print(f"Risk Score : {result['overall_risk_score']:.1f}/100")
print(f"Verdict    : {result['verdict']}")
print(f"Red Flags  : {result['red_flags']}")
print(f"Reports    : {result['report_paths']}")
```

### RAGAS Evaluation

```bash
python -m eval.ragas_eval --company "Infosys" --save --method auto
```

Or set `RAGAS_ENABLED=true` in `.env` to run automatically on every investigation.

### SSE Streaming (FastAPI)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from api.streaming import StreamingManager

app = FastAPI()
mgr = StreamingManager(llm_manager)

@app.get("/stream")
async def stream_analysis(query: str):
    return StreamingResponse(
        mgr.stream_sse(query, system_role="forensic_accountant"),
        media_type="text/event-stream"
    )
```

---

## 10. Understanding the Output

### Terminal summary

```
-----------------------------------------------------------
  Company:        Infosys
  Risk Score:     [XXXXXXXX................] 42.3/100 (MONITOR)
  Verdict:        MONITOR
  Red Flags:      7
  Green Flags:    4
  Documents:      23 acquired and analyzed

  Reports Generated:
    OK  [JSON]  Infosys_Investigation_20260617_1430.json
    OK  [XLSX]  Infosys_Forensic_Analysis_20260617_1430.xlsx
    OK  [DOCX]  Infosys_Forensic_Report_20260617_1430.docx
    OK  [PDF]   Infosys_Forensic_Report_20260617_1430.pdf
-----------------------------------------------------------
```

### Risk score components (with `--verbose`)

| Dimension | Weight | Source Agent |
|-----------|--------|-------------|
| Fraud Indicators | 25% | Agent 6 (Beneish, Altman, Dechow, Piotroski) |
| Earnings Quality | 20% | Agent 8 |
| Cash Flow Quality | 20% | Agent 4 |
| Governance | 15% | Agents 9, 10, 11 |
| Credit Risk | 10% | Agent 7 |
| Auditor Risk | 5% | Agent 10 |
| Management Credibility | 5% | Agent 11 |

### Red flag severity levels

| Severity | Meaning |
|----------|---------|
| **CRITICAL** | Imminent signal — going concern, M-Score > -1.0, negative CFO with positive NI |
| **HIGH** | Serious — M-Score > -1.78, DSO spike > 30 days, ICR < 2.5x |
| **MEDIUM** | Monitor — gross margin change > 5pp, auditor tenure > 10 years |
| **LOW** | Note-worthy — minor accrual elevation |

### Guardrail scores on findings

Every LLM-generated finding carries:
- `grounding_score`: token overlap with retrieved context (0–1)
- `hallucination_risk`: CRITICAL / HIGH / MEDIUM / LOW
- `faithfulness_score`: claim entailment in source context

---

## 11. Configuration Reference

All settings in `.env` (copy from `.env.example`). No code changes needed.

### LLM Provider

```bash
LLM_PROVIDER=groq    # groq | openai | anthropic | gemini | together
                     # openrouter | lmstudio | ollama | hf | auto
```

Auto-cascade order: `Groq → OpenAI → Anthropic → Gemini → Together → OpenRouter → LM Studio → Ollama → HuggingFace → Template`

### API Keys

```bash
GROQ_API_KEY=gsk_...          # Free — 14,400 req/day
GOOGLE_API_KEY=AIza...        # Free — 1,500 req/day
OPENAI_API_KEY=sk-proj-...    # Paid
ANTHROPIC_API_KEY=sk-ant-...  # Paid
TOGETHER_API_KEY=...          # Paid
OPENROUTER_API_KEY=sk-or-...  # Paid (100+ models)
```

### RAG Feature Toggles

```bash
# FAISS vector store (opt-in — 3-5x faster than ChromaDB)
USE_FAISS=true

# BGE fast mode (bge-small instead of bge-large; 4x faster, less accurate)
EMBEDDING_FAST=true

# Multi-vector retrieval (raw + LLM summary embeddings)
USE_MULTI_VECTOR=true

# Context compression (LLM-based sentence extraction)
USE_CONTEXT_COMPRESSION=true

# Conversation memory (SQLite-backed session history)
USE_CONVERSATION_MEMORY=true

# Guardrails (always active; these tune thresholds)
GUARDRAILS_GROUNDEDNESS_THRESHOLD=0.20
GUARDRAILS_BLOCK_ON_FAIL=false

# SSE Streaming
STREAMING_CHUNK_SIZE=5
STREAMING_DELAY=0.01

# RAGAS Evaluation (run on every investigation when true)
RAGAS_ENABLED=false

# Agentic RAG (LangGraph 12-step loop)
AGENTIC_RAG_ENABLED=false
AGENTIC_RAG_MAX_ITERATIONS=3
```

### Model Overrides

```bash
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_FAST_MODEL=llama-3.1-8b-instant
GEMINI_MODEL=gemini-1.5-flash
OLLAMA_MODEL=qwen2.5:7b
ANTHROPIC_MODEL=claude-opus-4-8
```

---

## 12. Output Files and Folders

```
forensic_ai_data/Investigations/{CompanyName}/
├── Raw_Filings/                   <- Downloaded PDFs, HTML filings
├── Parsed_Data/Text/              <- Extracted plain text
├── Financials/                    <- Structured financial JSON
├── Agent_Outputs/                 <- Per-agent JSON (18 files)
│   ├── agent_06_Fraud_Detection_Agent.json
│   └── agent_17_director_final_output.json   <- management questions here
├── Audit_Trail/
│   ├── investigation_log.jsonl    <- Immutable timestamped evidence chain
│   └── audit_summary.json         <- Lightweight stats + pointer to JSONL
├── Final_Output/
│   ├── {Company}_Investigation_{timestamp}.json
│   ├── {Company}_Forensic_Analysis_{timestamp}.xlsx
│   ├── {Company}_Forensic_Report_{timestamp}.docx
│   └── {Company}_Forensic_Report_{timestamp}.pdf
├── Knowledge_Base/                <- ChromaDB vector embeddings
└── faiss_db/                      <- FAISS index + metadata (if USE_FAISS=true)
```

### Key files to review

| File | Contents |
|------|---------|
| `Final_Output/*.pdf` | Executive summary + full narrative |
| `Final_Output/*.xlsx` | 9-tab analyst workbook including management questions |
| `Agent_Outputs/agent_17_*.json` | Verdict rationale, 50–80 management questions, monitoring triggers |
| `Audit_Trail/investigation_log.jsonl` | Full evidence chain — every calculation with source citations |

---

## 13. Forensic Scoring Models

| Model | Reference | Key Threshold |
|-------|-----------|--------------|
| **Beneish M-Score** | Beneish (1999) | > -1.78 = likely manipulator |
| **Altman Z-Score** | Altman (1968, 2000 EM) | < 1.81 = distress zone |
| **Piotroski F-Score** | Piotroski (2000) | 0–2 = distressed; 8–9 = strong |
| **Dechow F-Score** | Dechow et al. (2011) | > 0.025 = 6.7x base rate misstatement risk |
| **Sloan Accruals** | Sloan (1996) | CF accrual ratio > 0.10 = critical |

### Cross-Validation Rules (10)

| Rule | Trigger |
|------|---------|
| REVENUE_CFO_DIVERGENCE | Revenue CAGR 40%+ above CFO CAGR over 3+ years |
| AR_REVENUE_DIVERGENCE | AR growing 25%+ faster than revenue |
| INVENTORY_COGS_DIVERGENCE | Inventory growing 30%+ faster than COGS |
| RETAINED_EARNINGS_INCONSISTENCY | RE delta deviates >25% from NI minus Dividends |
| POOR_EBITDA_CFO_CONVERSION | CFO/EBITDA < 0.40x in any year |
| GROSS_MARGIN_JUMP | Gross margin changes >7pp in one year |
| TAX_RATE_ANOMALY | ETR deviates >12pp from company average |
| UNDERINVESTMENT | CapEx/Depreciation < 0.40x |
| DEBT_REVENUE_DIVERGENCE | Debt CAGR 50%+ above revenue CAGR over 3+ years |
| BALANCE_SHEET_IMBALANCE | Assets do not equal Liabilities plus Equity by >5% |

---

## 14. Technical References

### Forensic Accounting Models

| Model | Citation |
|-------|---------|
| Beneish M-Score | Beneish, M. D. (1999). "The Detection of Earnings Manipulation." *Financial Analysts Journal*, 55(5), 24–36. |
| Dechow F-Score | Dechow, P. M., Ge, W., Larson, C. R., & Sloan, R. G. (2011). "Predicting Material Accounting Misstatements." *Contemporary Accounting Research*, 28(1), 17–82. |
| Altman Z-Score | Altman, E. I. (1968). "Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy." *Journal of Finance*, 23(4), 589–609. |
| Altman EM Z-Score | Altman, E. I. (2000). "Predicting Financial Distress of Companies: Revisiting the Z-Score and ZETA Models." Stern School of Business Working Paper. |
| Piotroski F-Score | Piotroski, J. D. (2000). "Value Investing: The Use of Historical Financial Statement Information." *Journal of Accounting Research*, 38(Supplement), 1–41. |
| Sloan Accruals | Sloan, R. G. (1996). "Do Stock Prices Fully Reflect Information in Accruals and Cash Flows?" *The Accounting Review*, 71(3), 289–315. |
| RSST Accruals | Richardson, S. A., Sloan, R. G., Soliman, M. T., & Tuna, I. (2005). "Accrual Reliability, Earnings Persistence, and Stock Prices." *Journal of Accounting and Economics*, 39(3), 437–485. |

### RAG and LLM Frameworks

| Component | Citation |
|-----------|---------|
| BM25 | Robertson, S., & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond." *Foundations and Trends in Information Retrieval*, 3(4), 333–389. |
| Reciprocal Rank Fusion | Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods." SIGIR 2009, 758–759. |
| FAISS | Johnson, J., Douze, M., & Jegou, H. (2019). "Billion-Scale Similarity Search with GPUs." *IEEE Transactions on Big Data*, 7(3), 535–547. |
| BGE Embeddings | Xiao, S., et al. (2023). "C-Pack: Packaged Resources to Advance General Chinese Embedding." arXiv:2309.07597. |
| RAGAS | Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). "RAGAS: Automated Evaluation of Retrieval Augmented Generation." arXiv:2309.15217. |
| HyDE | Gao, L., et al. (2022). "Precise Zero-Shot Dense Retrieval without Relevance Labels." arXiv:2212.10496. |
| ReAct | Yao, S., et al. (2022). "ReAct: Synergizing Reasoning and Acting in Language Models." ICLR 2023. |
| Context Compression | Ma, X., et al. (2023). "Query Rewriting for Retrieval-Augmented Large Language Models." arXiv:2305.14283. |
| LangGraph | LangChain Inc. (2024). LangGraph: Build stateful, multi-actor applications with LLMs. github.com/langchain-ai/langgraph. |
| LlamaIndex | Liu, J. (2022). LlamaIndex: A data framework for LLM applications. github.com/run-llama/llama_index. |
| Cross-Encoder Reranking | Nogueira, R., & Cho, K. (2019). "Passage Re-ranking with BERT." arXiv:1901.04085. |
| Guardrails / Faithfulness | Min, S., et al. (2023). "FActScoring: Fine-grained Atomic Evaluation of Factual Precision." EMNLP 2023. |
| Conversation Memory | Shinn, N., et al. (2023). "Reflexion: Language Agents with Verbal Reinforcement Learning." NeurIPS 2023. |
| Multi-Agent Orchestration | Li, G., et al. (2023). "MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework." ICLR 2024. |

### Audit Standards Referenced

| Standard | Body | Relevance |
|----------|------|-----------|
| ISA 240 | IAASB | Fraud in an Audit of Financial Statements |
| ISA 315 (Revised 2019) | IAASB | Identifying and Assessing Risks of Material Misstatement |
| ISA 520 | IAASB | Analytical Procedures |
| ISA 570 | IAASB | Going Concern |
| SA 240, SA 315, SA 520, SA 570 | ICAI | Indian equivalents of above ISAs |
| PCAOB AS 2401 | PCAOB | Consideration of Fraud in a Financial Statement Audit (US) |
| PCAOB AS 2305 | PCAOB | Substantive Analytical Procedures |
| CARO 2020 | MCA (India) | Companies (Auditor's Report) Order — 21 reporting requirements |

### Indian Regulatory Framework

| Regulation | Authority | Relevance |
|-----------|-----------|-----------|
| SEBI LODR Regulations 2015 | SEBI | Listed entity disclosure obligations (Reg. 33, 34, 46) |
| SEBI SAST Regulations 2011 | SEBI | Promoter pledge thresholds and mandatory disclosure |
| SEBI RPT Regulations 2021 | SEBI | Related party transaction approval thresholds |
| Companies Act 2013 ss. 134, 143, 177 | MCA | Board responsibility, auditor duties, audit committee |
| Insolvency and Bankruptcy Code 2016 | MCA | Financial distress resolution (ss. 7, 9) |
| RBI Resolution Framework 2018 | RBI | Revised framework for resolution of stressed assets |

---

## 15. Troubleshooting

### "No LLM provider available — running in template mode"

```bash
python main.py --check
cat .env | grep _API_KEY   # Verify key is set correctly
```

### Investigation too slow

| Backend | Typical Time | Fix |
|---------|-------------|-----|
| Ollama CPU | 20–45 min | Use `phi3.5:3.8b` or `EMBEDDING_FAST=true` |
| Template mode | 5–10 min | Add an API key |
| Groq / Gemini | 8–15 min | Check free-tier rate limit |

### Missing packages

```bash
# Core RAG
pip install chromadb sentence-transformers faiss-cpu

# Optional RAGAS evaluation
pip install ragas datasets

# Report formats
pip install openpyxl python-docx reportlab
```

### Groq rate limit (batch mode)

Groq free tier: 14,400 requests/day. An investigation uses ~17–25 requests (~600 investigations/day before hitting limit). For bulk use, switch to `OPENAI_API_KEY` or use `OLLAMA_MODEL`.

---

## 16. FAQ

**Q: Which LLM gives the best results?**  
Quality: `claude-opus-4-8` (Anthropic) or `gpt-4o` (OpenAI). Free: `llama-3.3-70b-versatile` on Groq. Local: `qwen2.5:7b` on Ollama. Quantitative scores (Beneish, Altman etc.) are identical regardless of LLM — only narrative interpretation differs.

**Q: Can I use FAISS instead of ChromaDB?**  
Yes. Set `USE_FAISS=true` in `.env` and `pip install faiss-cpu`. FAISS is 3–5x faster for similarity queries.

**Q: How do I enable semantic chunking?**

```python
from processing.chunker import SemanticChunker
from rag.embeddings import EmbeddingModel
chunker = SemanticChunker(embedder=EmbeddingModel(), similarity_threshold=0.75)
chunks = chunker.chunk(text, source="doc.pdf", fiscal_year="2024")
```

**Q: How do I run RAGAS evaluation?**

```bash
python -m eval.ragas_eval --company "Infosys" --save --method auto
```

**Q: Can I add my own agent?**  
Yes. Subclass `BaseForensicAgent`, implement `investigate()`, and add it to `phase_b_specs` in `orchestrator.py`. The base class provides multi-vector RAG, guardrails, conversation memory, audit logging, and database persistence automatically.

**Q: What is the difference between the three chunking modes?**  
`DocumentChunker` is section-aware and sentence-snapped (best for financial statements). `RecursiveChunker` builds parent-child hierarchies (best for dense narrative). `SemanticChunker` splits at topic boundaries by cosine similarity (best for MDA and earnings call text).

**Q: Does this work for private companies?**  
Partially. Document acquisition only works for publicly listed companies (SEC EDGAR for US; NSE/BSE for India). For private companies, provide a financial data dict directly via the Python API.

**Q: Is this financial advice?**  
No. This platform produces research-grade output to assist human analysts. All findings must be reviewed by qualified investment professionals before being acted upon.

---

*Forensic AI v1.4 — For full technical documentation, see [FORENSIC_AI_TECHNICAL_REFERENCE.md](FORENSIC_AI_TECHNICAL_REFERENCE.md)*

*Repository: [github.com/anubhav0499-bit/forensic-ai](https://github.com/anubhav0499-bit/forensic-ai)*
