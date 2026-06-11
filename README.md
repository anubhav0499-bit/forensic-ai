# Forensic AI — User Guide

**Institutional-grade multi-agent forensic accounting platform.**  
Give it a company name. It downloads public filings, runs 17 specialist AI agents, and returns a 0–100 risk score, an investment verdict, and full PDF/DOCX/XLSX/JSON reports.

---

## Table of Contents

1. [What It Does](#1-what-it-does)
2. [Choose Your Setup Path](#2-choose-your-setup-path)
3. [Path A — Google Colab (Free, No Install)](#3-path-a--google-colab-free-no-install)
4. [Path B — Local with Cloud API](#4-path-b--local-with-cloud-api)
5. [Path C — Fully Offline with Ollama](#5-path-c--fully-offline-with-ollama)
6. [Running an Investigation](#6-running-an-investigation)
7. [Understanding the Output](#7-understanding-the-output)
8. [The 17 Agents — What Each One Does](#8-the-17-agents--what-each-one-does)
9. [Configuration Reference](#9-configuration-reference)
10. [Output Files and Folders](#10-output-files-and-folders)
11. [Troubleshooting](#11-troubleshooting)
12. [FAQ](#12-faq)

---

## 1. What It Does

When you submit a company name (e.g. `"Infosys"`, `"AAPL"`, `"RELIANCE"`):

```
Your input
    → Company identification (Yahoo Finance / SEC EDGAR / NSE)
    → Document acquisition (annual reports, 10-Ks, quarterly results, transcripts)
    → PDF parsing + financial table extraction
    → RAG knowledge base (ChromaDB + BM25)
    → 17 specialist AI agents run in parallel
    → Cross-validation of financial statements (10 internal consistency rules)
    → Chief Director synthesises all findings
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

## 2. Choose Your Setup Path

| Path | Where | LLM Cost | Setup Time | Best For |
|------|-------|----------|-----------|---------|
| **A — Google Colab** | Browser | Free | ~5 min | First-time use; no local install |
| **B — Local + Cloud API** | Your machine | Free (Groq/Gemini) or paid | ~10 min | Daily use; faster than Colab |
| **C — Local + Ollama** | Your machine | Free, offline | ~20 min | Privacy; no API keys; air-gapped |

---

## 3. Path A — Google Colab (Free, No Install)

**Requirements:** A Google account. Nothing else.

### Step 1 — Get a free API key

**Option 1 — Groq** (recommended: fastest, 14,400 free requests/day):
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free)
3. Create API key → copy it (starts with `gsk_`)

**Option 2 — Gemini** (1,500 free requests/day):
1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with Google → Create API key → copy it (starts with `AIza`)

### Step 2 — Open the notebook

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. File → Upload notebook → upload `colab_setup.ipynb` from this folder
   *(Or: File → Open notebook → Google Drive if you've already uploaded it)*

### Step 3 — Add your API key to Colab Secrets

1. Click the **🔑 key icon** in the left sidebar
2. Click **+ Add new secret**
3. Name: `GROQ_API_KEY` (or `GOOGLE_API_KEY`)
4. Value: paste your key
5. Toggle **Notebook access** ON

### Step 4 — Run the cells in order

| Cell | What it does | Time |
|------|-------------|------|
| **Cell 1** | Installs all packages | ~2 min |
| **Cell 2** | Mounts Drive, loads your API key | ~30 sec |
| **Cell 3** | Imports the platform, verifies LLM connection | ~30 sec |
| **Cell 4** | Runs the investigation | 3–8 min |

### Step 5 — Find your reports

Reports save to `My Drive → Forensic_Reports → {CompanyName}/`

Cell 4 prints the exact paths at the end.

---

## 4. Path B — Local with Cloud API

**Requirements:** Python 3.10+, pip. A Groq or Gemini API key (free).

### Step 1 — Clone / download the project

```bash
# If you have git:
git clone <your-repo-url> forensic_ai
cd forensic_ai

# Or unzip the downloaded folder and cd into it:
cd "path/to/forensic_ai"
```

### Step 2 — Create a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac / Linux
python -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install packages

```bash
pip install -r requirements-minimal.txt
```

*This installs ~500 MB of packages. Takes 2–5 minutes.*

### Step 4 — Add your API key

```bash
# Copy the template
copy .env.example .env       # Windows
cp .env.example .env         # Mac/Linux

# Open .env in any text editor and uncomment + fill in ONE line:
# GROQ_API_KEY=gsk_your_key_here
# (or GOOGLE_API_KEY=AIza_your_key_here)
```

### Step 5 — Verify the setup

```bash
python main.py --check
```

Expected output:
```
✓ ChromaDB
✓ SentenceTransformers
✓ PyMuPDF
...
```

### Step 6 — Run your first investigation

```bash
python main.py "Infosys"
```

Reports save to `~/Documents/Forensic_Reports/Infosys/`

---

## 5. Path C — Fully Offline with Ollama

**Requirements:** Python 3.10+, ~8 GB RAM, ~10 GB disk. No internet needed after setup.

### Step 1 — Install Ollama

Download from [ollama.com](https://ollama.com/download) and install it.

### Step 2 — Download a model

```bash
# Recommended (7B, ~4.5 GB) — good quality/speed balance:
ollama pull qwen2.5:7b

# Smaller/faster (3.8B, ~2.3 GB) — for low-RAM machines:
ollama pull phi3.5:3.8b
```

Ollama runs as a background service automatically after install.

### Step 3 — Install packages

```bash
cd forensic_ai
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Mac/Linux

pip install -r requirements-minimal.txt
```

*(No `.env` file needed — Ollama is auto-detected at `localhost:11434`)*

### Step 4 — Verify

```bash
python main.py --check
```

You should see `✓ Ollama: X models available` in the terminal.

### Step 5 — Run

```bash
python main.py "Infosys"
```

*Ollama on CPU takes 15–35 min per investigation. GPU (if available) takes 5–12 min.*

---

## 6. Running an Investigation

### CLI — Single company

```bash
python main.py "Infosys"
python main.py "AAPL"
python main.py "RELIANCE INDUSTRIES"
```

### CLI — Multiple companies

```bash
python main.py "Infosys" "TCS" "Wipro"
```

### CLI — From a file

Create `companies.txt`:
```
Infosys
TCS
HDFC Bank
# Lines starting with # are ignored
```

```bash
python main.py --batch companies.txt
```

### CLI — Options

| Flag | What it does | Example |
|------|-------------|---------|
| `--verbose` / `-v` | Show risk component breakdown in terminal | `python main.py "AAPL" -v` |
| `--years` / `-y` | Years of historical data to analyse (default: 5) | `--years 7` |
| `--output` / `-o` | Override report output directory | `--output /mnt/reports` |
| `--check` | Check dependencies and exit | `python main.py --check` |
| `--model` | Override Ollama model | `--model qwen2.5:14b` |

### Web Interface

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Enter company names in the text box, click **Start Forensic Investigation**. Download buttons appear when reports are ready.

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

---

## 7. Understanding the Output

### Terminal summary (after each investigation)

```
─────────────────────────────────────────────────────────────
  Company:        Infosys
  Risk Score:     [████████░░░░░░░░░░░░] 42.3/100 (MODERATE)
  Verdict:        MONITOR
  Red Flags:      7
  Green Flags:    4
  Documents:      23 acquired and analyzed

  Reports Generated:
    ✓  [JSON]  Infosys_Investigation_20260611_1430.json
    ✓  [XLSX]  Infosys_Forensic_Analysis_20260611_1430.xlsx
    ✓  [DOCX]  Infosys_Forensic_Report_20260611_1430.docx
    ✓  [PDF]   Infosys_Forensic_Report_20260611_1430.pdf
─────────────────────────────────────────────────────────────
```

### Risk score components (with `--verbose`)

The composite score is built from 7 weighted dimensions:

| Dimension | Weight | Source Agent |
|-----------|--------|-------------|
| Fraud Indicators | 25% | Agent 6 (Beneish, Altman, Dechow, Piotroski) |
| Earnings Quality | 20% | Agent 8 |
| Cash Flow Quality | 20% | Agent 4 |
| Governance | 15% | Agent 9 (Related Party) |
| Credit Risk | 10% | Agent 7 |
| Auditor Risk | 5% | Agent 10 |
| Management Credibility | 5% | Agent 11 |

### Forensic scoring models

| Model | What it measures | Key threshold |
|-------|-----------------|---------------|
| **Beneish M-Score** | Earnings manipulation probability | > −1.78 = likely manipulator |
| **Altman Z-Score** | Bankruptcy risk | < 1.81 = distress zone |
| **Piotroski F-Score** | Financial strength (0–9) | < 3 = distressed |
| **Dechow F-Score** | SEC enforcement action probability | > 0.025 = high risk |

### Red flag severity levels

| Severity | Meaning |
|----------|---------|
| **CRITICAL** | Imminent signal — going concern, M-Score > −1.0, negative CFO with positive net income |
| **HIGH** | Serious — M-Score > −1.78, DSO spike > 30 days, interest coverage < 2.5× |
| **MEDIUM** | Monitor — gross margin change > 5pp, auditor tenure > 10 years |
| **LOW** | Note-worthy — minor accrual elevation |

---

## 8. The 17 Agents — What Each One Does

**Phase A — Runs first, output feeds all later agents**

| Agent | What it investigates |
|-------|---------------------|
| 6 · Fraud Detection | Runs all 5 forensic models (Beneish, Altman, Piotroski, Dechow, Sloan accruals). Compares against 6 historical fraud cases (Enron, Wirecard, Satyam, Luckin Coffee, Carillion, Steinhoff) |

**Phase B — 8 agents run in parallel**

| Agent | What it investigates |
|-------|---------------------|
| 3 · Revenue Forensics | AR/revenue growth gap, Q4 revenue skew (channel stuffing), deferred revenue pull-forward, revenue-EBIT divergence |
| 4 · Cash Flow Forensics | CFO vs net income divergence, free cash flow quality, CapEx sustainability |
| 5 · Working Capital | DSO / DIO / DPO / Cash Conversion Cycle trends over 5 years |
| 7 · Credit Risk | Leverage ratios, interest coverage, liquidity, implied credit rating |
| 8 · Earnings Quality | Accrual ratios, effective tax rate anomalies, margin consistency |
| 9 · Related Party | RPT concentration, promoter loans, pledge levels, disclosure quality |
| 10 · Auditor Intelligence | Going concern opinions, material weaknesses, auditor changes, Key Audit Matters |
| 11 · Management NLP | Evasion language, excessive hedging, non-GAAP overemphasis in earnings calls |

**Phase C — Runs after Phase B, with full prior context**

| Agent | What it investigates |
|-------|---------------------|
| 12 · Peer Comparison | Benchmarks all metrics against industry peers; flags outliers |
| 14 · Investment Committee | Three perspectives in one: Bear case (short seller), Bull case, Devil's Advocate |

**Phase D — Final synthesis**

| Agent | What it investigates |
|-------|---------------------|
| 17 · Chief Director | Aggregates all 16 agents, computes composite risk score, issues final verdict, generates 50–80 management questions, sets 8 quarterly monitoring triggers. If score lands in the 38–62 grey zone, triggers one extra "Resolve the Ambiguity" LLM pass |

---

## 9. Configuration Reference

All settings live in `.env` (copy from `.env.example`). No code changes needed.

### LLM provider

```bash
# Force a specific provider (default: auto-cascade)
LLM_PROVIDER=groq      # groq | openai | anthropic | gemini | together
                       # openrouter | lmstudio | ollama | hf | auto
```

The auto-cascade tries providers in this order until one responds:
`Groq → OpenAI → Anthropic → Gemini → Together → OpenRouter → LM Studio → Ollama → HuggingFace → Template`

### API keys (set only the ones you have)

```bash
GROQ_API_KEY=gsk_...          # Free — 14,400 req/day
GOOGLE_API_KEY=AIza...        # Free — 1,500 req/day
OPENAI_API_KEY=sk-proj-...    # Paid
ANTHROPIC_API_KEY=sk-ant-...  # Paid
TOGETHER_API_KEY=...          # Paid (hosted open-source models)
OPENROUTER_API_KEY=sk-or-...  # Paid (100+ models)
```

### Model overrides (optional)

```bash
GROQ_MODEL=llama-3.3-70b-versatile      # default
GROQ_FAST_MODEL=llama-3.1-8b-instant
GEMINI_MODEL=gemini-1.5-pro-latest
GEMINI_FAST_MODEL=gemini-2.0-flash
OLLAMA_MODEL=qwen2.5:7b
ANTHROPIC_MODEL=claude-opus-4-8
```

### Output directory

```bash
REPORTS_DIR=/path/to/your/reports    # default: ~/Documents/Forensic_Reports
```

---

## 10. Output Files and Folders

Every investigation creates a folder at `~/Documents/Forensic_Reports/{CompanyName}/`:

```
{CompanyName}/
├── Raw_Filings/               ← Downloaded PDFs, HTML filings
├── Parsed_Data/Text/          ← Extracted plain text
├── Financials/                ← Structured financial JSON (yfinance, Screener)
├── Agent_Outputs/             ← JSON output from each of the 17 agents
│   ├── agent_06_Fraud_Detection_Agent.json
│   ├── agent_17_director_final_output.json   ← management questions here
│   └── ...
├── Audit_Trail/
│   ├── investigation_log.jsonl   ← Immutable timestamped evidence chain
│   └── audit_summary.json        ← Lightweight stats + pointer to JSONL
├── Final_Output/
│   ├── {Company}_Investigation_{timestamp}.json   ← Full data dump
│   ├── {Company}_Forensic_Analysis_{timestamp}.xlsx
│   ├── {Company}_Forensic_Report_{timestamp}.docx
│   ├── {Company}_Forensic_Report_{timestamp}.pdf
│   └── report_index.json         ← Index of all generated reports
└── Knowledge_Base/            ← ChromaDB vector embeddings
```

### Key files to review

| File | What's in it |
|------|-------------|
| `Final_Output/*.pdf` | Executive summary + full narrative report |
| `Final_Output/*.xlsx` | 9-tab analyst workbook including management questions |
| `Agent_Outputs/agent_17_director_final_output.json` | Verdict rationale, 50–80 management questions, monitoring triggers |
| `Audit_Trail/investigation_log.jsonl` | Full evidence chain — every calculation and finding with source citations |

---

## 11. Troubleshooting

### "No LLM provider available — running in template mode"

The platform found no usable LLM. Fix:
1. Check your `.env` file has a valid API key (no extra spaces, correct variable name)
2. Run `python main.py --check` to see which packages are missing
3. If using Ollama: run `ollama list` to confirm a model is installed

### "Failed to import orchestrator"

You're running from the wrong directory. Fix:
```bash
cd forensic_ai       # make sure you're inside the project folder
python main.py "Infosys"
```

### Investigation takes very long

| LLM backend | Typical time | Fix |
|-------------|-------------|-----|
| Ollama on CPU | 20–45 min | Normal. Use `phi3.5:3.8b` for speed |
| Template mode | 5–10 min | Normal (no LLM). Add an API key to get narrative |
| Groq / Gemini | 8–15 min | If slower, check free-tier rate limit |

### "ChromaDB not installed" / "sentence-transformers not installed"

```bash
pip install chromadb sentence-transformers
```

### Reports not generating (XLSX / DOCX / PDF errors)

Each report format fails independently — others still generate. Check the terminal for:
- `XLSX generation failed: ...` → `pip install openpyxl`
- `DOCX generation failed: ...` → `pip install python-docx`
- `PDF generation failed: ...` → `pip install reportlab`

### Groq rate limit hit (batch investigations)

Groq free tier: 14,400 requests/day. An investigation uses ~17–25 requests. You can run ~600 investigations/day before hitting the limit. For bulk use, switch to `OPENAI_API_KEY` or `OLLAMA_MODEL`.

### Wrong company identified

Force the correct ticker:
```bash
python main.py "HDFC" --company "HDFC Bank"
```

Or add a manual override in `acquisition/company_lookup.py` if the company is consistently misidentified.

---

## 12. FAQ

**Q: Does this send my data to the cloud?**  
A: Only if you use a cloud LLM (Groq, OpenAI, Anthropic, Gemini). The financial data included in prompts goes to whichever provider you configure. For sensitive mandates, use Ollama or LM Studio — fully local, no data leaves your machine.

**Q: Which LLM gives the best results?**  
A: For quality: `claude-opus-4-8` (Anthropic) or `gpt-4o` (OpenAI). For free: `llama-3.3-70b-versatile` on Groq. For local: `qwen2.5:7b` on Ollama. The quantitative forensic scores (Beneish, Altman etc.) are identical regardless of LLM — only the narrative interpretation differs.

**Q: Can I investigate private companies?**  
A: Partially. Document acquisition only works for publicly listed companies (SEC EDGAR for US; NSE/BSE for India). For private companies, the agents work from text only — you can provide a financial data dict via the Python API and the agents will analyse it.

**Q: How do I re-run only the reports without re-investigating?**  
A: Load the existing JSON output and call the report compiler directly:
```python
from reporting.report_compiler import ReportCompiler
from utils.storage import StorageManager
from database.sqlite_handler import SQLiteHandler

storage = StorageManager("CompanyName", "TICKER")
db = SQLiteHandler()
compiler = ReportCompiler(storage, db)
# Load existing data and call compiler.generate_all(...)
```

**Q: How do I enable the PPTX and HTML reports?**  
A: Open `reporting/report_compiler.py` and uncomment the PPTX block (around line 89) and/or the HTML block (around line 99). Both generators are fully implemented — they're just disabled by default to keep the default run lighter.

**Q: The risk score seems too high / too low. Can I adjust the thresholds?**  
A: Yes. In `config.py`, edit `ForensicThresholds` (academic model thresholds) and/or `RISK_SCORE_WEIGHTS` (the 7 dimension weights). Industry-specific overrides are already configured for BANKING, REAL_ESTATE, SOFTWARE_SAAS, and INSURANCE in `INDUSTRY_THRESHOLD_OVERRIDES`.

**Q: Can I add my own agent?**  
A: Yes. Subclass `BaseForensicAgent`, implement `investigate()`, and add it to `phase_b_specs` or `phase_c_configs` in `orchestrator.py`. The base class provides LLM access, RAG retrieval, database persistence, and audit logging automatically.

**Q: Is this financial advice?**  
A: No. This platform produces research-grade output to assist human analysts. All findings must be reviewed by qualified investment professionals before being acted upon.

---

*Forensic AI v1.2 — For full technical documentation, see [FORENSIC_AI_TECHNICAL_REFERENCE.md](FORENSIC_AI_TECHNICAL_REFERENCE.md)*
