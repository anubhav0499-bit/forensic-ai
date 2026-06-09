# FORENSIC AI — DEVELOPER HANDOFF
### Everything you need to understand, change, and extend the system

---

**For:** Developer / Platform Maintainer  
**Assumes:** Comfortable reading Python; basic familiarity with the platform  
**Read first:** `FORENSIC_AI_TECHNICAL_REFERENCE.md` for system context  

---

## QUICK ORIENTATION

```
61 Python files | 3 entry points | 8 forensic engines | 17 agents (10 dedicated, 4 generic-LLM)
9 LLM providers | Colab / VSCode / local — all supported
```

**The three files you will touch most often:**

| File | Why you'll touch it |
|------|-------------------|
| `config.py` | Thresholds, model names, weights, agent names — all centralized here |
| `agents/orchestrator.py` | Agent wiring, execution phases, parallel config |
| `llm/prompts.py` | Every agent's system prompt and analysis prompt |

**The one rule:** `config.py` is the single source of truth. Never hardcode a threshold, model name, or weight anywhere else.

---

## TABLE OF CONTENTS

1. [Current System State — What's Fully Built vs Stub](#1-current-system-state)
2. [How to Add a New Agent (Step-by-Step)](#2-how-to-add-a-new-agent)
3. [How to Promote a Generic Agent to Dedicated](#3-how-to-promote-a-generic-agent-to-dedicated)
4. [How to Modify Forensic Thresholds](#4-how-to-modify-forensic-thresholds)
5. [How to Adjust Risk Score Weights](#5-how-to-adjust-risk-score-weights)
6. [How to Change or Add an LLM](#6-how-to-change-or-add-an-llm)
7. [How to Add a New Data Source](#7-how-to-add-a-new-data-source)
8. [How to Add a New Forensic Model](#8-how-to-add-a-new-forensic-model)
9. [How to Add or Modify a Report Section](#9-how-to-add-or-modify-a-report-section)
10. [How to Add a New Cross-Validation Rule](#10-how-to-add-a-new-cross-validation-rule)
11. [Key Architecture Decisions & Why](#11-key-architecture-decisions--why)
12. [Known Incomplete Parts & TODOs](#12-known-incomplete-parts--todos)
13. [Common Gotchas & Bugs to Avoid](#13-common-gotchas--bugs-to-avoid)
14. [How to Test Changes](#14-how-to-test-changes)
15. [Dependency Map — What Imports What](#15-dependency-map)

---

## 1. CURRENT SYSTEM STATE

### What is fully implemented (with dedicated classes + engines):

| # | Agent | File | Phase | Status |
|---|-------|------|-------|--------|
| 3  | Revenue Forensics       | `agent_03_revenue.py`         | B | ✅ **NEW** — AR/Rev gap, deferred rev, Q4 skew, CAGR divergence |
| 4  | Cash Flow Forensics     | `agent_04_cashflow.py`        | B | ✅ Full — uses AccrualAnalyzer |
| 5  | Working Capital         | `agent_05_working_capital.py` | B | ✅ Full — uses WorkingCapitalAnalyzer |
| 6  | Fraud Detection         | `agent_06_fraud_detection.py` | A | ✅ Full — Beneish + Altman + Piotroski + Dechow |
| 7  | Credit Risk             | `agent_07_credit_risk.py`     | B | ✅ Full — uses AltmanZScore + coverage ratios |
| 8  | Earnings Quality        | `agent_08_earnings_quality.py`| B | ✅ Full — uses AccrualAnalyzer |
| 9  | Related Party Forensics | `agent_09_related_party.py`   | B | ✅ **NEW** — RPT concentration, promoter loans, NLP opacity |
| 10 | Auditor Intelligence    | `agent_10_auditor.py`         | B | ✅ Full — NLP pattern matching on audit text |
| 11 | Management NLP          | `agent_11_management_nlp.py`  | B | ✅ Full — word-list NLP on disclosures |
| 17 | Chief Investigation Director | `agent_17_director.py`   | D | ✅ Full — synthesis + refinement loop |

### What runs via `_run_generic_agent` (LLM-only, no dedicated class):

| # | Agent | Type Key | Notes |
|---|-------|----------|-------|
| 12 | Peer Comparison   | `"peer"`            | DuckDB peer benchmarking exists but not wired to an agent class yet |
| 14 | Short Seller      | `"short_seller"`    | Pure LLM; works well |
| 15 | Bull Case         | `"bull_case"`       | Pure LLM; works well |
| 16 | Devil's Advocate  | `"devils_advocate"` | Pure LLM; works well |

Agents 0, 1, 2 (Historical Data, Document Acquisition, Financial Extraction) are defined in `AGENT_NAMES` in `config.py` but have **no implementation at all** — they were spec'd but not yet built.

### Forensic Engines:

| Engine | File | Status |
|--------|------|--------|
| Beneish M-Score | `forensics/beneish_score.py` | ✅ Complete — all 8 variables |
| Altman Z-Score | `forensics/altman_score.py` | ✅ Complete — 3 model variants |
| Piotroski F-Score | `forensics/piotroski_score.py` | ✅ Complete — all 9 criteria |
| Dechow F-Score | `forensics/dechow_score.py` | ✅ Complete — logistic regression |
| Accrual Analysis | `forensics/accrual_analysis.py` | ✅ Complete — Sloan + CF methods |
| Working Capital | `forensics/working_capital_analysis.py` | ✅ Complete — DSO/DIO/DPO/CCC |
| Risk Scorer | `forensics/risk_scorer.py` | ✅ Complete — composite 0–100 |
| Cross-Validator | `forensics/cross_validator.py` | ✅ Complete — 10 rules |

---

## 2. HOW TO ADD A NEW AGENT

This is the most common change you'll make. Follow these 5 steps exactly.

### Step 1 — Register the agent in `config.py`

```python
# config.py  ← line ~249
AGENT_NAMES = {
    # ... existing entries ...
    18: "Supply Chain Risk Agent",   # ← add your new agent here
}
```

### Step 2 — Create the agent file

Copy this template to `agents/agent_18_supply_chain.py`:

```python
"""
Agent 18 — Supply Chain Risk Agent
Investigates: supplier concentration, geographic risk, single-source dependency.
"""
from __future__ import annotations
from .base_agent import BaseForensicAgent, AgentResult


class SupplyChainRiskAgent(BaseForensicAgent):
    """
    Forensic supply chain analysis.
    """

    def investigate(
        self, company_name: str, company_id: int, financial_data: dict, **kwargs
    ) -> AgentResult:
        self.log_info(f"Supply chain risk analysis for {company_name}")
        result = AgentResult(agent_id=self.agent_id, agent_name=self.agent_name)

        # 1. Retrieve relevant document context
        context = self._retrieve_context(
            company_name,
            "supplier concentration geographic risk single source procurement"
        )

        # 2. Run any quantitative analysis you want here
        #    (use financial_data dict — keys like "revenue", "cogs", "inventory")
        years = sorted(financial_data.keys(), reverse=True)

        # 3. Create findings
        if some_condition:
            f = self._create_finding(
                finding_type="RED_FLAG",      # or "GREEN_FLAG" or "OBSERVATION"
                title="Single Supplier Concentration Risk",
                detail="Detailed explanation of the finding.",
                evidence="Specific numbers and source: 'Per FY2024 annual report p.47...'",
                fiscal_year=years[0] if years else "",
                risk_level="HIGH",            # CRITICAL / HIGH / MEDIUM / LOW / POSITIVE
                confidence=0.80,              # 0.0 – 1.0
                calculation="Optional formula trace",
            )
            result.red_flags.append(f)
            result.findings.append(f)

        # 4. Run LLM analysis
        prompt = f"Analyze supply chain risk for {company_name}.\n\nContext:\n{context}"
        result.raw_analysis = self._analyze_with_llm(prompt, "forensic_accountant")

        # 5. Set risk score
        result.risk_score = 45.0  # Replace with your calculation

        # 6. Write summary
        result.summary = f"SUPPLY CHAIN RISK — {company_name}: {result.risk_score:.1f}/100"

        # 7. Save output file
        self._save_output(result, company_name)
        self.log_info(f"Supply Chain complete. Risk={result.risk_score:.1f}/100")
        return result
```

### Step 3 — Wire the agent into the orchestrator

Open `agents/orchestrator.py`. Choose which phase:

- **Phase B** (parallel, engine-based, no need for prior agent context) → add to `phase_b_specs`
- **Phase C** (LLM-heavy, benefits from seeing prior findings) → add to `phase_c_configs`

**For a dedicated class (Phase B):**
```python
# orchestrator.py  ← around line 267
# Add import at top of file:
from .agent_18_supply_chain import SupplyChainRiskAgent

# Then in _run_all_agents():
phase_b_specs = [
    (4, "Cash Flow Forensics Agent", CashFlowForensicsAgent),
    (5, "Working Capital Agent", WorkingCapitalAgent),
    (7, "Credit Risk Agent", CreditRiskAgent),
    (8, "Earnings Quality Agent", EarningsQualityAgent),
    (10, "Auditor Intelligence Agent", AuditorIntelligenceAgent),
    (11, "Management NLP Agent", ManagementNLPAgent),
    (18, "Supply Chain Risk Agent", SupplyChainRiskAgent),  # ← add here
]
```

**For a generic/LLM agent (Phase C):**
```python
# orchestrator.py  ← around line 290
phase_c_configs = [
    (3, "revenue", "Revenue Forensics Agent"),
    (9, "related_party", "Related Party Agent"),
    (12, "peer", "Peer Comparison Agent"),
    (14, "short_seller", "Short Seller Agent"),
    (15, "bull_case", "Bull Case Agent"),
    (16, "devils_advocate", "Devil's Advocate Agent"),
    (18, "supply_chain", "Supply Chain Risk Agent"),  # ← add here
]
```

If Phase C generic, also add to the `role_map` and `question_map` in `_run_generic_agent()`:

```python
# orchestrator.py  ← around line 384
role_map = {
    # ... existing ...
    "supply_chain": "forensic_accountant",   # ← add
}
question_map = {
    # ... existing ...
    "supply_chain": "Investigate supplier concentration, geographic risk, and single-source dependencies.",  # ← add
}
```

### Step 4 — Add a system prompt (optional but recommended)

```python
# llm/prompts.py  ← inside SYSTEM_PROMPTS dict
SYSTEM_PROMPTS = {
    # ... existing ...
    "supply_chain_analyst": """You are an institutional supply chain risk analyst.
Your job is to identify: single-source supplier risk, geographic concentration,
commodity price exposure, and logistics vulnerabilities.
Follow Evidence → Analysis → Reasoning → Conclusion.""",
}
```

Then in your agent's LLM call:
```python
result.raw_analysis = self._analyze_with_llm(prompt, "supply_chain_analyst")
```

### Step 5 — Add risk component extraction (optional)

```python
# orchestrator.py  ← _extract_risk_components()
component_map = {
    4: "cash_flow_quality",
    # ... existing ...
    18: "supply_chain_risk",   # ← add
}
```

That's it. The agent will now run in the correct phase, receive inter-agent context (if Phase C), be included in the Director's synthesis, and appear in all 6 report formats automatically.

---

## 3. HOW TO PROMOTE A GENERIC AGENT TO DEDICATED

Agents 3, 9, 12 currently run via `_run_generic_agent`. Here's how to give Agent 3 (Revenue Forensics) a full dedicated implementation.

### Current state of Agent 3

```python
# orchestrator.py  phase_c_configs
(3, "revenue", "Revenue Forensics Agent"),
```

It runs with a generic LLM prompt. No quantitative engine exists for it.

### Steps to promote it

**1. Create `agents/agent_03_revenue.py`** with your forensic logic.  
   Key checks to implement:
   - Revenue CAGR vs industry peers
   - AR days trend (already available from Agent 5, but run independently here)
   - Deferred revenue changes (shrinking deferred revenue = pull-forward recognition)
   - Q4 revenue as % of full-year (channel stuffing indicator: >35% is suspicious)
   - Revenue from top 5 customers as % (concentration risk)
   - Geographic revenue mix shifts

**2. Move it from `phase_c_configs` to `phase_b_specs`:**

```python
# orchestrator.py

# REMOVE from phase_c_configs:
# (3, "revenue", "Revenue Forensics Agent"),

# ADD import at top:
from .agent_03_revenue import RevenueForen sicsAgent

# ADD to phase_b_specs:
(3, "Revenue Forensics Agent", RevenueForensicsAgent),
```

**3. No other changes needed.** The orchestrator automatically handles the rest.

### Agent 12 — Peer Comparison

Agent 12 (`peer`) has a working engine already: `database/duckdb_handler.py` has `peer_benchmarking()` and `get_trend_analysis()`. To promote it:

```python
# agents/agent_12_peer.py
from database.duckdb_handler import DuckDBHandler

class PeerComparisonAgent(BaseForensicAgent):
    def investigate(self, ...):
        ddb = DuckDBHandler()
        # Load peer data
        peer_results = ddb.peer_benchmarking(
            company_id=company_id,
            metric="net_margin",
            peer_company_ids=[...]   # populate from DB
        )
        # anomaly detection
        anomalies = ddb.detect_anomalies(company_id)
        # ... create findings from anomalies
```

---

## 4. HOW TO MODIFY FORENSIC THRESHOLDS

**All thresholds live in one place:** `config.py`, `ForensicThresholds` dataclass.

```python
# config.py  ← around line 130
@dataclass
class ForensicThresholds:
    beneish_threshold: float = -1.78          # M-Score: above = manipulator
    altman_safe_zone: float = 2.99            # Z-Score: above = safe
    altman_distress_zone: float = 1.81        # Z-Score: below = distress
    piotroski_strong: int = 7                 # F-Score: >= = strong
    piotroski_weak: int = 2                   # F-Score: <= = weak/distressed
    dso_spike_threshold: float = 30.0         # DSO YoY change (days)
    inventory_growth_threshold: float = 0.30  # Inventory growth above revenue growth
    dpo_stretch_threshold: float = 30.0       # DPO YoY change (days)
    accrual_ratio_high: float = 0.10          # CF Accrual ratio high risk
    accrual_ratio_moderate: float = 0.05      # CF Accrual ratio moderate
    cash_conversion_warning: float = 0.70     # CFO/EBITDA warning

FORENSIC_THRESHOLDS = ForensicThresholds()
```

### Industry-Specific Threshold Overrides

When analysing banks, REITs, or other non-standard sectors:

```python
# config.py  ← add after FORENSIC_THRESHOLDS
INDUSTRY_THRESHOLD_OVERRIDES: dict[str, dict] = {
    "BANKING": {
        "beneish_threshold": -1.50,      # Banks have structurally higher accruals
        "dso_spike_threshold": 90.0,     # Loan portfolio DSO norms differ
        "accrual_ratio_high": 0.20,      # Higher accrual threshold for banks
    },
    "REAL_ESTATE": {
        "accrual_ratio_high": 0.18,      # Project-based revenue recognition
        "dso_spike_threshold": 60.0,     # Property sales have longer cycles
    },
    "SOFTWARE_SAAS": {
        "dso_spike_threshold": 45.0,     # Subscription billing cycles
        "piotroski_strong": 6,           # SaaS burns cash early, lower F-score normal
    },
}
```

Then in each forensic engine, check the override:

```python
# In any forensic engine that uses FORENSIC_THRESHOLDS:
from config import FORENSIC_THRESHOLDS, INDUSTRY_THRESHOLD_OVERRIDES

def _get_threshold(key: str, industry: str = "") -> float:
    if industry in INDUSTRY_THRESHOLD_OVERRIDES:
        return INDUSTRY_THRESHOLD_OVERRIDES[industry].get(key, getattr(FORENSIC_THRESHOLDS, key))
    return getattr(FORENSIC_THRESHOLDS, key)
```

> **Warning:** The academic Beneish threshold of -1.78 is validated on US GAAP companies. If you change it, document the validation study you're using as a reference.

---

## 5. HOW TO ADJUST RISK SCORE WEIGHTS

```python
# config.py  ← around line 155
RISK_SCORE_WEIGHTS: dict[str, float] = {
    "fraud_indicators":       0.25,
    "earnings_quality":       0.20,
    "cash_flow_quality":      0.20,
    "governance":             0.15,
    "credit_risk":            0.10,
    "auditor_risk":           0.05,
    "management_credibility": 0.05,
}
# These must sum to 1.0 — the RiskScorer enforces this.
```

The `RiskScorer.calculate()` in `forensics/risk_scorer.py` reads these directly. No other change needed.

**Example — Credit-focused mandate:**
```python
RISK_SCORE_WEIGHTS = {
    "fraud_indicators":       0.20,
    "earnings_quality":       0.15,
    "cash_flow_quality":      0.20,
    "governance":             0.10,
    "credit_risk":            0.25,   # ← increased for credit mandate
    "auditor_risk":           0.05,
    "management_credibility": 0.05,
}
```

**The verdict bands** (BUY/MONITOR/AVOID/etc.) are also in `config.py`:

```python
# config.py
RISK_BANDS: dict[str, tuple] = {
    "BUY":          (0,  24),
    "CAUTIOUS_BUY": (25, 37),
    "MONITOR":      (38, 49),
    "CAUTION":      (50, 59),
    "AVOID":        (60, 74),
    "STRONG_AVOID": (75, 100),
}
```

To add a new verdict band, add it here. The Director's `_determine_verdict()` in `agent_17_director.py` reads this dict.

---

## 6. HOW TO CHANGE OR ADD AN LLM

### Provider Auto-Detection Priority

The platform now supports **9 providers** and auto-detects them in this order:

```
Groq → OpenAI → Anthropic → Gemini → Together → OpenRouter → LM Studio → Ollama → HuggingFace → Template
```

No code changes are needed to switch providers. Just set the right env variable.

### Switching providers — zero code changes

**Option A: Set in `.env` file (recommended for VSCode / local)**
```bash
# Copy .env.example to .env and fill in your key
cp .env.example .env
# Then edit .env:
GROQ_API_KEY=gsk_xxxx
```

**Option B: Set in shell before running**
```bash
GROQ_API_KEY=gsk_xxxx python main.py "Infosys"
```

**Option C: Force a specific provider**
```bash
LLM_PROVIDER=gemini GOOGLE_API_KEY=AIza... python main.py "Infosys"
```

**Option D: In Colab** — add to Secrets (🔑 icon), then Cell 2 in `colab_setup.ipynb` loads them.

### Free Provider Setup

| Provider | Get Key | Free Limit | Best For |
|----------|---------|-----------|----------|
| Groq | [console.groq.com](https://console.groq.com) | 14,400 req/day | Colab, CI |
| Gemini | [aistudio.google.com](https://aistudio.google.com/app/apikey) | 1,500 req/day | Colab |
| Ollama | Install at [ollama.com](https://ollama.com) | Unlimited (local) | VSCode, laptop |
| LM Studio | [lmstudio.ai](https://lmstudio.ai) | Unlimited (local) | VSCode, laptop |

### Change the model for any provider

All model names are in `config.py` under `PROVIDER_MODELS`. Override any one:

```python
# config.py  ← PROVIDER_MODELS dict
PROVIDER_MODELS = {
    "groq":      {"primary": "llama-3.3-70b-versatile", "fast": "llama-3.1-8b-instant"},
    "openai":    {"primary": "gpt-4o",                  "fast": "gpt-4o-mini"},
    "anthropic": {"primary": "claude-opus-4-8",         "fast": "claude-haiku-4-5-20251001"},
    "gemini":    {"primary": "gemini-1.5-pro-latest",   "fast": "gemini-2.0-flash"},
    # ...
}
```

Or via env var without editing code:
```bash
GROQ_MODEL=llama-3.1-8b-instant python main.py "Infosys"    # faster, cheaper
OPENAI_MODEL=gpt-4-turbo         python main.py "Infosys"    # specific version
```

### Add a brand-new provider

The manager (`llm/llm_manager.py`) uses two patterns:

**Pattern 1 — OpenAI-compatible API** (same request format as OpenAI):
Add it to the `_OPENAI_COMPAT` dict at the top of `llm_manager.py`:
```python
_OPENAI_COMPAT: dict[str, dict] = {
    # existing entries ...
    "fireworks": {"base_url": "https://api.fireworks.ai/inference/v1",
                  "api_key": os.getenv("FIREWORKS_API_KEY", "")},
}
```
Also add it to `PROVIDER_MODELS` in `config.py` and to `_AUTO_ORDER` in `llm_manager.py`.

**Pattern 2 — Custom API** (non-OpenAI format):
Add `_check_<name>()` and `_generate_<name>()` methods, then add the dispatch in `_try_provider()` and `_dispatch()`.

### Local model management (Ollama)

```bash
ollama pull qwen2.5:14b       # Better quality (14B vs 7B)
ollama pull deepseek-r1:8b    # Reasoning-focused model
ollama pull mistral:7b        # Alternative 7B
ollama list                   # Show installed models
```

Model is auto-selected from available models based on the preference list in `_check_ollama()`.

### Local model management (LM Studio)

1. Download [LM Studio](https://lmstudio.ai)
2. Download any GGUF model from the model browser
3. Start the local server (LM Studio → Local Server tab → Start)
4. Platform auto-detects at `http://localhost:1234`

### HuggingFace Transformers (Colab GPU)

When no API key is available but a GPU is detected (Colab T4/A100), the platform loads HF models:
```bash
# Set HF model explicitly:
HF_MODEL=Qwen/Qwen2.5-14B-Instruct python main.py "Infosys"
```
This is the slowest option but completely free with Colab GPU.

---

## 7. HOW TO ADD A NEW DATA SOURCE

### Pattern: every acquisition source follows the same shape

```python
# acquisition/my_new_source.py
import requests
from loguru import logger

class MyNewSourceClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "ForensicAI/1.0"
        self._rate_limit = 0.5  # seconds between requests

    def get_annual_data(self, ticker: str, years: int = 5) -> dict:
        """Return {year: {metric: value}} dict."""
        ...

    def download_filing(self, url: str, dest_path: Path) -> Optional[Path]:
        """Download a document. Return path or None."""
        ...
```

### Wire into the downloader

```python
# acquisition/downloader.py  ← acquire_all_documents()
def acquire_all_documents(self, profile: CompanyProfile, company_id: int, years: int = 5):
    # existing code ...

    # Add your new source:
    if profile.country == "SG":   # Example: Singapore source
        from .my_new_source import MyNewSourceClient
        sg_client = MyNewSourceClient()
        data = sg_client.get_annual_data(profile.ticker, years)
        if data:
            self.storage.save_json(data, "sg_financials.json", "Financials")
```

### Wire financial data into the cascade

```python
# agents/orchestrator.py  ← _assemble_financial_data()
for yf_file in ["yfinance_annual.json", "screener_financials.json", "sg_financials.json"]:
    #                                                                  ^^^^ add here
    ...
```

---

## 8. HOW TO ADD A NEW FORENSIC MODEL

### Example: Adding Montier C-Score (Accounting Quality)

**Step 1 — Create the engine:**

```python
# forensics/montier_cscore.py
from __future__ import annotations
from dataclasses import dataclass
from utils.helpers import safe_divide


@dataclass
class CScoreResult:
    c_score: int          # 0–6 (higher = worse accounting quality)
    interpretation: str
    flags: list


class MontierCScore:
    """
    Montier C-Score: 6-variable accounting quality check.
    Score 0–6. Above 4 = likely accounting manipulation.
    """

    def calculate(self, ...) -> CScoreResult:
        score = 0
        flags = []

        # Variable 1: Growing net income with shrinking CFO
        if net_income > net_income_prev and cfo < cfo_prev:
            score += 1
            flags.append("NI_RISING_CFO_FALLING")

        # ... add other 5 variables ...

        result = CScoreResult(c_score=score, interpretation=self._interpret(score), flags=flags)
        return result

    def _interpret(self, score: int) -> str:
        if score >= 4: return "HIGH ACCOUNTING RISK"
        if score >= 2: return "MODERATE RISK"
        return "LOW RISK"
```

**Step 2 — Register in `forensics/__init__.py`:**

```python
from .montier_cscore import MontierCScore
```

**Step 3 — Call it from the relevant agent:**

```python
# agents/agent_08_earnings_quality.py  ← inside investigate()
from forensics.montier_cscore import MontierCScore

cscore_result = MontierCScore().calculate(
    net_income=d.get("net_income", 0),
    cfo=d.get("cfo", 0),
    # ...
)
scores["montier_c_score"] = cscore_result.c_score
```

**Step 4 — Save to database:**

Add the column to the `forensic_scores` table in `database/schema.py`:

```sql
-- database/schema.py  ← CREATE TABLE forensic_scores
CREATE TABLE IF NOT EXISTS forensic_scores (
    ...
    montier_c_score REAL,       -- ← add
    ...
);
```

Then add to the DB save in Agent 6's `_run_all_models()` or the relevant agent:

```python
self.db.save_forensic_scores(company_id, year, {
    # ... existing ...
    "montier_c_score": scores.get("montier_c_score"),
})
```

**Step 5 — Add to the XLSX report:**

```python
# reporting/xlsx_generator.py  ← wherever forensic scores are written
ws.cell(row=r, column=1, value="Montier C-Score")
ws.cell(row=r, column=2, value=scores.get("montier_c_score", "N/A"))
```

That's the full chain for a new forensic model.

---

## 9. HOW TO ADD OR MODIFY A REPORT SECTION

### Add a section to the DOCX (Word) report

```python
# reporting/docx_generator.py

# 1. Find the _add_* method for the section closest to where you want to insert.
#    Sections are called in generate() in order.

# 2. Add a new method:
def _add_supply_chain_section(self, doc, report_data: dict) -> None:
    doc.add_heading("SUPPLY CHAIN RISK ANALYSIS", level=1)
    sc_data = report_data.get("agent_analyses", {}).get("18", {})
    if not sc_data:
        doc.add_paragraph("Supply chain analysis not available.")
        return
    doc.add_paragraph(sc_data.get("summary", ""))

# 3. Call it in generate() at the right position:
def generate(self, report_data: dict, output_path: Path) -> Path:
    doc = Document()
    # ... existing sections ...
    self._add_supply_chain_section(doc, report_data)   # ← add here
    # ... more sections ...
    doc.save(str(output_path))
```

### Add a tab to the XLSX (Excel) report

```python
# reporting/xlsx_generator.py

def _add_supply_chain_tab(self, wb, report_data: dict) -> None:
    ws = wb.create_sheet("Supply Chain Risk")
    ws["A1"] = "SUPPLY CHAIN RISK ANALYSIS"
    # ... populate cells ...

def generate(self, report_data: dict, output_path: Path) -> Path:
    wb = openpyxl.Workbook()
    # ... existing tabs ...
    self._add_supply_chain_tab(wb, report_data)   # ← add here
    wb.save(str(output_path))
```

### Add a slide to the PPTX report

```python
# reporting/pptx_generator.py

def _add_supply_chain_slide(self, prs, data: dict) -> None:
    slide = self._blank_slide(prs)
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12), Inches(0.8))
    title_box.text_frame.paragraphs[0].text = "SUPPLY CHAIN RISK"
    # ... add content ...

def generate(self, report_data: dict, output_path: Path) -> Path:
    prs = Presentation()
    # ... existing slides ...
    self._add_supply_chain_slide(prs, report_data)   # ← add here
    prs.save(str(output_path))
```

### Expose new agent data to all report generators

Report data is assembled in `reporting/report_compiler.py`, `_compile_report_data()`. If Agent 18 output needs to be in reports:

```python
# reporting/report_compiler.py  ← _compile_report_data()
report_data = {
    # ... existing keys ...
    "supply_chain_analysis": agent_results.get(18, AgentResult(18, "")).summary,
}
```

---

## 10. HOW TO ADD A NEW CROSS-VALIDATION RULE

Open `forensics/cross_validator.py`. The pattern for every rule is identical:

```python
# forensics/cross_validator.py

# 1. Add your rule method:
def _check_capex_revenue_gap(self, fd: dict, years: list) -> list[CrossValidationIssue]:
    """Rule 11: CapEx growing much faster than revenue (possible asset inflation)."""
    issues = []
    for i in range(1, len(years)):
        curr, prev = fd[years[i]], fd[years[i - 1]]
        capex_curr = abs(curr.get("capex", 0) or 0)
        capex_prev = abs(prev.get("capex", 0) or 0)
        rev_curr = curr.get("revenue", 1) or 1
        rev_prev = prev.get("revenue", 1) or 1

        if capex_prev <= 0:
            continue

        capex_growth = safe_divide(capex_curr - capex_prev, capex_prev)
        rev_growth = safe_divide(rev_curr - rev_prev, rev_prev)

        if capex_growth > rev_growth + 0.40:
            issues.append(CrossValidationIssue(
                issue_type="CAPEX_REVENUE_GAP",
                severity="MODERATE",
                description="CapEx growing significantly faster than revenue",
                evidence=(
                    f"FY{years[i]}: CapEx growth {capex_growth*100:.1f}% "
                    f"vs Revenue growth {rev_growth*100:.1f}%. Gap: {(capex_growth-rev_growth)*100:.1f}pp."
                ),
                metric_a="CapEx Growth", value_a=capex_growth,
                metric_b="Revenue Growth", value_b=rev_growth,
                fiscal_year=years[i],
                confidence=0.78,
            ))
    return issues

# 2. Call it from validate():
def validate(self, financial_data: dict) -> list[CrossValidationIssue]:
    ...
    issues.extend(self._check_balance_sheet_identity(financial_data, years))
    issues.extend(self._check_capex_revenue_gap(financial_data, years))   # ← add here
    ...
```

That's all. The issue will automatically appear in:
- The audit trail log
- The SQLite database
- All 6 report formats
- The agent context (injected as `cv_context`)

---

## 11. KEY ARCHITECTURE DECISIONS & WHY

Understanding these will save you from making breaking changes.

### Decision 1: `config.py` as single source of truth

**Why:** Every threshold, weight, model name, and URL is in `config.py`. This means a senior can adjust `FORENSIC_THRESHOLDS.beneish_threshold = -1.60` in one place and it propagates everywhere. No hunting through 56 files.

**Don't break this by:** hardcoding `-1.78` inside an agent file instead of `FORENSIC_THRESHOLDS.beneish_threshold`.

---

### Decision 2: `BaseForensicAgent` ABC with `_create_finding()`

**Why:** Every finding must be routed through `_create_finding()`. This method:
1. Creates the `AgentFinding` dataclass
2. Logs it to the immutable JSONL audit trail
3. Saves it to SQLite

If you create findings without `_create_finding()`, they won't appear in the audit trail or database. This breaks the evidence chain.

**Don't break this by:** building `AgentFinding(...)` objects directly and appending them to `result.red_flags` without calling `_create_finding()`.

---

### Decision 3: Phase A → B → C execution order

**Why:**
- **Phase A** (Fraud Detection) runs first because its M-Score, Z-Score, and accrual scores are objective and fast. Later agents use these as a baseline.
- **Phase B** (specialist engines) runs in parallel because they're independent — each uses its own engine on `financial_data`. They don't need each other's output.
- **Phase C** (synthesis agents) runs last because they receive the complete `inter_agent_context` from Phase A+B. The Short Seller agent seeing the Fraud Detection and Credit Risk output before writing is what makes the synthesis coherent.

**Don't break this by:** putting a Phase C synthesis agent in Phase B. It will miss all the context and produce weaker analysis.

---

### Decision 4: Thread-pool parallelism with blocking LLM calls

**Why:** The LLM `generate()` calls are synchronous HTTP requests to Ollama (or blocking HF inference). Python's `concurrent.futures.ThreadPoolExecutor` releases the GIL for I/O-bound blocking calls, so true parallelism is achieved for the Ollama case.

**SQLite thread safety:** The database uses WAL (Write-Ahead Log) mode, which handles concurrent reads and serializes writes. All agents can call `db.save_finding()` concurrently without corruption.

**Don't break this by:** using `asyncio` coroutines inside agent `investigate()` methods without `asyncio.to_thread()`. The agents are designed to be synchronous. The thread-pool wrapping is all handled in the orchestrator.

---

### Decision 5: Hybrid RAG (BM25 + Dense) with RRF

**Why:** Financial documents contain precise terminology (e.g., "DSRI", "trade receivables", "CARO 2020"). Dense-only retrieval misses exact keyword matches. BM25-only misses semantic relationships. The 0.4/0.6 weighted RRF fusion gives the best of both. The `k=60` in RRF is the standard from the original paper and shouldn't need changing.

**Don't break this by:** replacing the hybrid retriever with vector-only search. You'll get worse recall on specific accounting terms.

---

### Decision 6: JSONL audit trail (append-only)

**Why:** The audit trail is a `.jsonl` file opened in append mode (`"a"`). This makes it immutable — you can only add entries, never modify them. This is important for legal defensibility.

**Don't break this by:** opening the audit file in write mode (`"w"`), which would truncate and destroy the evidence chain.

---

## 12. KNOWN INCOMPLETE PARTS & TODOs

### FIXED IN THIS SESSION ✅

| Previously Missing | What Was Done |
|-------------------|--------------|
| Agent 3 (Revenue) pure LLM | Dedicated `agent_03_revenue.py` — AR/Rev gap, Q4 skew, deferred rev, CAGR divergence |
| Agent 9 (Related Party) pure LLM | Dedicated `agent_09_related_party.py` — RPT concentration, promoter loans, NLP opacity |
| LLM locked to Ollama only | 9-provider cascade: Groq, OpenAI, Anthropic, Gemini, Together, OpenRouter, LM Studio, Ollama, HF |
| No Colab support | `colab_setup.ipynb` — full Colab notebook with Drive mount + free API setup |
| No env template | `.env.example` — all keys + instructions |
| INDUSTRY_THRESHOLD_OVERRIDES unused | Added `get_threshold(key, sector)` helper in `config.py` |
| requirements.txt missing cloud providers | Added `openai`, `anthropic`, `google-generativeai` + created `requirements-minimal.txt` |

### Still Pending (High Priority)

| Gap | File | What's Needed |
|-----|------|--------------|
| **Agent 12 (Peer) — DuckDB not wired** | `agents/orchestrator.py` | Move Agent 12 from Phase C generic to Phase B; connect `DuckDBHandler.peer_benchmarking()` |
| **Agents 0, 1, 2 — no implementation** | `config.py` AGENT_NAMES | Reserved slots; not called anywhere — stub or remove |
| **Plotly charts not generated** | `agents/orchestrator.py` | Add chart generation phase between Phase 5 and Phase 7 |
| **`get_threshold()` not called in agents** | All forensic engines | Agents still read `FORENSIC_THRESHOLDS` directly; need to pass sector to use overrides |

### Still Pending (Medium Priority)

| Gap | File | What's Needed |
|-----|------|--------------|
| **Screener.in parser fragile** | `acquisition/india_markets.py` | Add exponential backoff for HTTP 429; fallback to yfinance |
| **No retry on rate limits** | `acquisition/india_markets.py` | Add `tenacity.retry` decorator |
| **report_compiler extra_data** | `reporting/report_compiler.py` | Verify the `extra_data` kwarg is accepted (see Gotcha 8) |

### Still Pending (Low Priority)

| Gap | File | What's Needed |
|-----|------|--------------|
| **No unit tests** | (missing) | `tests/` directory with pytest tests for each forensic engine |
| **VSCode launch config** | (missing) | `.vscode/launch.json` for debugging with F5 |

---

## 13. COMMON GOTCHAS & BUGS TO AVOID

### Gotcha 1: Financial data values are sometimes 0, sometimes None

```python
# WRONG — will fail or produce NaN
gross_margin = data["gross_profit"] / data["revenue"]

# RIGHT — always use safe_divide() and handle None/0
from utils.helpers import safe_divide
gross_margin = safe_divide(data.get("gross_profit", 0) or 0, data.get("revenue", 1) or 1)
```

### Gotcha 2: yfinance CapEx is always negative

yfinance returns CapEx as a negative number. Always `abs()` it:
```python
capex = abs(data.get("capex", 0) or 0)
```

### Gotcha 3: Fiscal year keys are strings, not ints

Financial data dict keys are strings (e.g., `"2024"`, not `2024`):
```python
# WRONG
year = 2024
data[year]   # KeyError

# RIGHT
year = "2024"
data[year]

# Or when sorting:
years = sorted(financial_data.keys(), reverse=True)   # sorts strings lexically — works for "2020"–"2024"
```

### Gotcha 4: `_create_finding()` sets `company_id=0` in base class

The `BaseForensicAgent._create_finding()` hardcodes `company_id=0` in the DB save:
```python
# base_agent.py line ~152
self.db.save_finding(company_id=0, ...)   # ← known limitation
```

The orchestrator doesn't currently pass `company_id` down to individual agents. If you need findings properly attributed to companies in the DB, you can fix this by storing `company_id` on the agent during `__init__` and using it in `_create_finding()`.

### Gotcha 5: ChromaDB collections are per-company

Each company gets its own ChromaDB collection named `forensic_ai_{normalized_company_name}`. If you run two investigations concurrently with the same company name (shouldn't happen normally), they'll share the same collection and may produce incorrect RAG results.

### Gotcha 6: Thread safety in `_parse_llm_findings()`

`_parse_llm_findings()` in the orchestrator creates `AgentFinding` objects and calls `audit.log_red_flag()`. The `AuditTrail` class opens the JSONL file in append mode, which is thread-safe on most OS file systems. But if you add shared state to `AuditTrail`, make it thread-safe with a `threading.Lock`.

### Gotcha 7: The Altman Z-Score variant matters

The orchestrator always uses `is_manufacturing=True, is_public=True`. For financial companies, REIT, or private companies, this is wrong:

```python
# agents/agent_06_fraud_detection.py  ← _run_all_models()
inp_z = AltmanInputs(
    ...
    is_manufacturing=False,  # set False for services/tech/financial
    is_public=False,         # set False for private companies
)
```

Consider passing company sector from the `CompanyProfile` to agents so they can set this correctly.

### Gotcha 8: `report_compiler.py` `generate_all()` signature mismatch

The orchestrator calls:
```python
compiler.generate_all(..., extra_data={...})
```

But the original `ReportCompiler.generate_all()` signature may not include `extra_data`. If you get a `TypeError`, add `**kwargs` to `generate_all()`:

```python
# reporting/report_compiler.py
def generate_all(self, ..., extra_data: dict = None, **kwargs) -> dict:
    ...
```

---

## 14. HOW TO TEST CHANGES

### Quick smoke test (no Ollama needed)

```python
# Run this in a Python shell from the forensic_ai/ directory:
import sys; sys.path.insert(0, ".")

# Test forensic engines directly (no LLM, no network)
from forensics.beneish_score import BeneishMScore, BeneishInputs

inp = BeneishInputs(
    net_receivables_t=150, sales_t=1000, cogs_t=600,
    current_assets_t=400, ppe_t=300, total_assets_t=800,
    depreciation_t=30, sga_t=100, total_debt_t=200,
    current_liabilities_t=150, working_capital_t=250,
    cash_t=50, taxes_payable_t=20,
    net_receivables_tm1=120, sales_tm1=900, cogs_tm1=550,
    current_assets_tm1=350, ppe_tm1=320, total_assets_tm1=750,
    depreciation_tm1=28, sga_tm1=95, total_debt_tm1=180,
    current_liabilities_tm1=130,
)
result = BeneishMScore().calculate(inp)
print(f"M-Score: {result.m_score:.4f}")
print(f"Manipulation: {result.manipulation_likely}")
assert isinstance(result.m_score, float), "M-Score should be float"
print("✓ Beneish OK")
```

### Test cross-validator

```python
from forensics.cross_validator import CrossValidator

test_data = {
    "2022": {"revenue": 100e6, "cfo": 90e6, "net_income": 10e6,
             "accounts_receivable": 20e6, "total_assets": 500e6,
             "total_liabilities": 300e6, "shareholder_equity": 200e6},
    "2023": {"revenue": 120e6, "cfo": 40e6, "net_income": 25e6,   # ← CFO drops while NI rises
             "accounts_receivable": 50e6,                            # ← AR spikes
             "total_assets": 550e6,
             "total_liabilities": 340e6, "shareholder_equity": 210e6},
}
issues = CrossValidator().validate(test_data)
for i in issues:
    print(f"[{i.severity}] {i.issue_type}: {i.description}")
assert len(issues) > 0, "Should detect AR-revenue divergence"
print(f"✓ CrossValidator found {len(issues)} issues")
```

### Test the full pipeline without downloading documents

```python
from agents.orchestrator import ForensicOrchestrator

# Monkey-patch document acquisition to skip it
import agents.orchestrator as orch
orch.DocumentDownloader = lambda *a, **kw: type("FakeDownloader", (), {
    "acquire_all_documents": lambda self, *a, **kw: {"total_files": 0, "financial_data_years": 0},
    "close": lambda self: None,
})()

orchestrator = ForensicOrchestrator()

# Inject synthetic financial data instead of downloading
SYNTHETIC_DATA = {
    "2024": {"revenue": 1500e6, "net_income": 200e6, "cfo": 180e6, "capex": 80e6,
             "total_assets": 3000e6, "current_assets": 900e6, "current_liabilities": 600e6,
             "accounts_receivable": 300e6, "inventory": 150e6, "total_debt": 800e6,
             "shareholder_equity": 1200e6, "gross_profit": 600e6, "ebit": 250e6},
    "2023": {"revenue": 1300e6, "net_income": 170e6, "cfo": 160e6, "capex": 70e6,
             "total_assets": 2800e6, "current_assets": 850e6, "current_liabilities": 580e6,
             "accounts_receivable": 240e6, "inventory": 130e6, "total_debt": 750e6,
             "shareholder_equity": 1100e6, "gross_profit": 530e6, "ebit": 220e6},
}
orchestrator._fetch_yfinance_financials = lambda name: SYNTHETIC_DATA

result = orchestrator.investigate("TestCo")
print(f"Risk Score: {result['overall_risk_score']:.1f}/100")
print(f"Verdict: {result['verdict']}")
print(f"Red Flags: {result['red_flags']}")
```

### Test a single agent in isolation

```python
from llm.llm_manager import LLMManager
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from rag.bm25_retriever import BM25Retriever
from rag.hybrid_retriever import HybridRetriever
from database.sqlite_handler import SQLiteHandler
from utils.storage import StorageManager
from utils.audit_trail import AuditTrail
from pathlib import Path

# Minimal setup
llm = LLMManager()
em = EmbeddingModel()
storage = StorageManager("TestCo", "TEST")
audit = AuditTrail(storage.base_path, "TestCo")
vs = VectorStore(em, storage.knowledge_base)
bm25 = BM25Retriever()
retriever = HybridRetriever(vs, bm25)
db = SQLiteHandler()

# Test Agent 5 (Working Capital)
from agents.agent_05_working_capital import WorkingCapitalAgent
agent = WorkingCapitalAgent(5, llm, retriever, db, audit, storage)
result = agent.investigate("TestCo", company_id=1, financial_data=SYNTHETIC_DATA)
print(f"Agent 5 Risk Score: {result.risk_score:.1f}")
print(f"Red Flags: {len(result.red_flags)}")
for flag in result.red_flags:
    print(f"  [{flag.risk_level}] {flag.title}")
```

### Syntax check all files after changes

```bash
# Windows PowerShell:
cd forensic_ai
$files = Get-ChildItem -Recurse -Filter "*.py" | Select-Object -ExpandProperty FullName
$errors = @()
foreach ($f in $files) {
    $result = python -m py_compile $f 2>&1
    if ($LASTEXITCODE -ne 0) { $errors += "$($f.Split('\')[-1]): $result" }
}
if ($errors.Count -eq 0) { "All $($files.Count) files OK" } else { $errors }
```

---

## 15. DEPENDENCY MAP

**Who imports from whom** — useful for understanding what breaks if you change a file.

```
config.py
  ← imported by: EVERYTHING (all 56 files)
  ← DO NOT import from any forensic_ai module (circular import risk)

utils/helpers.py
  ← imported by: forensics/*.py, agents/*.py, processing/*.py

utils/audit_trail.py
  ← imported by: agents/base_agent.py, agents/orchestrator.py

forensics/accrual_analysis.py
  ← used by: agent_04_cashflow.py, agent_06_fraud_detection.py, agent_08_earnings_quality.py

forensics/working_capital_analysis.py
  ← used by: agent_05_working_capital.py

forensics/altman_score.py
  ← used by: agent_06_fraud_detection.py, agent_07_credit_risk.py

forensics/risk_scorer.py
  ← used by: agent_06_fraud_detection.py, agent_07_credit_risk.py, agent_17_director.py

forensics/cross_validator.py
  ← used by: agents/orchestrator.py (Phase 4b)

agents/base_agent.py
  ← imported by: ALL agent_*.py files

agents/orchestrator.py
  ← imports: ALL agent_*.py files, ALL forensics/*.py, ALL rag/*.py, ALL acquisition/*.py

llm/prompts.py
  ← imported by: agents/base_agent.py, agents/orchestrator.py

reporting/report_compiler.py
  ← imported by: agents/orchestrator.py

database/sqlite_handler.py
  ← imported by: agents/base_agent.py, agents/orchestrator.py, acquisition/downloader.py

rag/hybrid_retriever.py
  ← imported by: agents/base_agent.py, agents/orchestrator.py
```

**Safe to change without ripple effects:**
- Any `agent_*.py` file (only imported by `orchestrator.py`)
- Any `reporting/*_generator.py` (only imported by `report_compiler.py`)
- Any `acquisition/*.py` except `downloader.py` (only used via `downloader.py`)

**High-risk to change (affects many files):**
- `config.py` — everything imports it; changes to dataclass field names break imports
- `agents/base_agent.py` — all agents inherit from it; signature changes cascade
- `database/sqlite_handler.py` — all agents write findings through it
- `llm/llm_manager.py` — all agents call `self.llm.generate()` through it

---

*End of HANDOFF.md*

*If something in this document is wrong or out of date after you've made changes, update this file. Future-you will thank you.*
