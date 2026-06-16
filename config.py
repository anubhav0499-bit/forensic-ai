"""
FORENSIC AI - Central Configuration
====================================
All platform settings, model configs, thresholds, and paths.
"""

from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# BASE PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", str(Path.home() / "Documents" / "Forensic_Reports")))
COLAB_DRIVE = Path("/content/drive/MyDrive/Forensic_Accounting_Reports")

# Detect environment
IS_COLAB = "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ
IS_KAGGLE = "KAGGLE_URL_BASE" in os.environ
OUTPUT_DIR = COLAB_DRIVE if (IS_COLAB or IS_KAGGLE) else REPORTS_DIR

# ─────────────────────────────────────────────
# API KEYS  (set in .env or environment)
# ─────────────────────────────────────────────
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
TOGETHER_API_KEY    = os.getenv("TOGETHER_API_KEY", "")
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
MISTRAL_API_KEY     = os.getenv("MISTRAL_API_KEY", "")
COHERE_API_KEY      = os.getenv("COHERE_API_KEY", "")
TAVILY_API_KEY      = os.getenv("TAVILY_API_KEY", "")
SERPER_API_KEY      = os.getenv("SERPER_API_KEY", "")
SERPAPI_API_KEY     = os.getenv("SERPAPI_API_KEY", "")

# AWS Bedrock
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION_NAME       = os.getenv("AWS_REGION_NAME", "us-east-1")

# Azure OpenAI
AZURE_OPENAI_API_KEY  = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")

# Force a provider or use cascade (auto).
# Values: auto | openai | anthropic | google | groq | together | openrouter |
#         mistral | cohere | ollama | lmstudio | bedrock | azure | hf
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")

# Per-provider model names (override individually via env if needed)
PROVIDER_MODELS: dict[str, dict[str, str]] = {
    "groq":        {"primary": os.getenv("GROQ_MODEL",        "llama-3.3-70b-versatile"),
                    "fast":    os.getenv("GROQ_FAST_MODEL",    "llama-3.1-8b-instant")},
    "openai":      {"primary": os.getenv("OPENAI_MODEL",       "gpt-4o"),
                    "fast":    os.getenv("OPENAI_FAST_MODEL",   "gpt-4o-mini")},
    "anthropic":   {"primary": os.getenv("ANTHROPIC_MODEL",    "claude-sonnet-4-6"),
                    "fast":    os.getenv("ANTHROPIC_FAST_MODEL","claude-haiku-4-5-20251001")},
    "gemini":      {"primary": os.getenv("GEMINI_MODEL",       "gemini-1.5-pro-latest"),
                    "fast":    os.getenv("GEMINI_FAST_MODEL",   "gemini-2.0-flash")},
    "together":    {"primary": os.getenv("TOGETHER_MODEL",     "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
                    "fast":    os.getenv("TOGETHER_FAST_MODEL", "meta-llama/Llama-3.1-8B-Instruct-Turbo")},
    "openrouter":  {"primary": os.getenv("OPENROUTER_MODEL",   "anthropic/claude-3.5-sonnet"),
                    "fast":    os.getenv("OPENROUTER_FAST_MODEL","google/gemini-flash-1.5")},
    "mistral":     {"primary": os.getenv("MISTRAL_MODEL",      "mistral-large-latest"),
                    "fast":    os.getenv("MISTRAL_FAST_MODEL",  "mistral-small-latest")},
    "cohere":      {"primary": os.getenv("COHERE_MODEL",       "command-r-plus"),
                    "fast":    os.getenv("COHERE_FAST_MODEL",   "command-r")},
    "bedrock":     {"primary": os.getenv("BEDROCK_MODEL",      "anthropic.claude-3-5-sonnet-20241022-v2:0"),
                    "fast":    os.getenv("BEDROCK_FAST_MODEL",  "anthropic.claude-3-haiku-20240307-v1:0")},
    "azure":       {"primary": os.getenv("AZURE_OPENAI_DEPLOYMENT",      "gpt-4o"),
                    "fast":    os.getenv("AZURE_OPENAI_FAST_DEPLOYMENT",  "gpt-4o-mini")},
    "lmstudio":    {"primary": os.getenv("LMSTUDIO_MODEL",     "local-model"),
                    "fast":    os.getenv("LMSTUDIO_FAST_MODEL", "local-model")},
    "ollama":      {"primary": os.getenv("OLLAMA_MODEL",        "qwen2.5:7b"),
                    "fast":    os.getenv("OLLAMA_FAST_MODEL",   "phi3.5:3.8b")},
    "hf":          {"primary": os.getenv("HF_MODEL",           "Qwen/Qwen2.5-7B-Instruct"),
                    "fast":    os.getenv("HF_FAST_MODEL",       "microsoft/phi-3.5-mini-instruct")},
}

# ─────────────────────────────────────────────
# LLM CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class LLMConfig:
    # Local provider URLs
    ollama_base_url:   str = os.getenv("OLLAMA_BASE_URL",   "http://localhost:11434")
    lmstudio_base_url: str = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234")

    # Legacy: kept for backward-compat with any existing code reading these fields
    primary_model: str = PROVIDER_MODELS["ollama"]["primary"]
    fast_model:    str = PROVIDER_MODELS["ollama"]["fast"]
    nlp_model:     str = "llama3.2:3b"
    hf_primary_model: str = PROVIDER_MODELS["hf"]["primary"]
    hf_fast_model:    str = PROVIDER_MODELS["hf"]["fast"]

    # Generation parameters
    temperature:    float = 0.1
    max_tokens:     int   = 4096
    context_window: int   = 32768
    top_p:          float = 0.9

    # Timeout / retries
    timeout:     int   = 300
    max_retries: int   = 3
    retry_delay: float = 2.0

LLM_CONFIG = LLMConfig()

# ─────────────────────────────────────────────
# EMBEDDING CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-large-en-v1.5"
    fallback_model: str = "all-MiniLM-L6-v2"
    device: str = "cuda" if os.getenv("USE_GPU", "").lower() in ("cuda", "1", "true", "yes") else "cpu"
    batch_size: int = 32
    normalize_embeddings: bool = True
    chunk_size: int = 512
    chunk_overlap: int = 64

EMBEDDING_CONFIG = EmbeddingConfig()

# ─────────────────────────────────────────────
# VECTOR DATABASE CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class VectorDBConfig:
    persist_dir: Path = DATA_DIR / "chroma_db"
    collection_prefix: str = "forensic_ai"
    distance_metric: str = "cosine"
    n_results: int = 10

VECTOR_DB_CONFIG = VectorDBConfig()

# ─────────────────────────────────────────────
# DATABASE CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class DatabaseConfig:
    sqlite_path: Path = DATA_DIR / "forensic_ai.db"
    duckdb_path: Path = DATA_DIR / "forensic_analytics.duckdb"

DB_CONFIG = DatabaseConfig()

# ─────────────────────────────────────────────
# DOCUMENT ACQUISITION CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class AcquisitionConfig:
    # SEC EDGAR
    sec_base_url: str = "https://efts.sec.gov/LATEST/search-index"
    sec_submissions_url: str = "https://data.sec.gov/submissions"
    sec_company_facts_url: str = "https://data.sec.gov/api/xbrl/companyfacts"
    sec_user_agent: str = "ForensicAI forensicai@analysis.com"

    # NSE India
    nse_base_url: str = "https://www.nseindia.com"
    nse_filings_url: str = "https://www.nseindia.com/companies-listing/corporate-filings-annual-reports"

    # BSE India
    bse_base_url: str = "https://www.bseindia.com"
    bse_search_url: str = "https://api.bseindia.com/BseIndiaAPI/api/AnnualReport/w"

    # Download settings
    download_timeout: int = 60
    max_file_size_mb: int = 200

    # Historical depth
    years_history: int = 5

    # Rate limiting (requests per second)
    rate_limit: float = 0.5

ACQUISITION_CONFIG = AcquisitionConfig()

# ─────────────────────────────────────────────
# PROCESSING CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class ProcessingConfig:
    # PDF processing
    ocr_language: str = "eng"
    ocr_confidence_threshold: float = 60.0

    # Chunking — 400 words ≈ 500 tokens; large enough for multi-sentence financial
    # reasoning, small enough to fit several chunks in a 4096-token context window.
    chunk_size: int = 400
    chunk_overlap: int = 80
    table_chunk_size: int = 1200

    # Table extraction
    table_accuracy_threshold: float = 0.85
    lattice_line_scale: int = 15
    stream_edge_tol: int = 2

PROCESSING_CONFIG = ProcessingConfig()

# ─────────────────────────────────────────────
# FORENSIC THRESHOLDS
# ─────────────────────────────────────────────
@dataclass
class ForensicThresholds:
    # Beneish M-Score
    beneish_manipulation_threshold: float = -1.78
    beneish_high_risk_threshold: float = -1.0

    # Altman Z-Score (EM non-manufacturing variant — Altman 2000)
    altman_safe_zone: float = 2.60   # EM: >2.60 safe (manufacturing: 2.99)
    altman_grey_zone: float = 1.10   # EM: <1.10 distress (manufacturing: 1.81)

    # Piotroski F-Score
    piotroski_strong: int = 7
    piotroski_weak: int = 2

    # Accruals ratio thresholds
    accrual_ratio_high: float = 0.10
    accrual_ratio_moderate: float = 0.05

    # Revenue growth anomaly (YoY %)
    revenue_growth_spike_threshold: float = 0.50   # 50% YoY = investigate
    revenue_growth_collapse_threshold: float = -0.20

    # Cash conversion ratio
    cash_conversion_healthy: float = 0.85
    cash_conversion_warning: float = 0.70

    # DSO (Days Sales Outstanding) spike
    dso_spike_threshold: float = 30  # days increase

    # Inventory build-up
    inventory_growth_threshold: float = 0.30  # 30% more than revenue growth

    # Payable stretching
    dpo_stretch_threshold: float = 30  # days increase

    # Debt/Equity threshold
    de_ratio_concern: float = 2.0

    # Interest Coverage
    interest_coverage_distress: float = 1.5
    interest_coverage_concern: float = 2.5

FORENSIC_THRESHOLDS = ForensicThresholds()

# Industry-specific threshold overrides.
# Keys must match sector strings returned by CompanyLookup.
INDUSTRY_THRESHOLD_OVERRIDES: dict[str, dict] = {
    "BANKING": {
        "beneish_manipulation_threshold": -1.50,
        "accrual_ratio_high": 0.20,
        "dso_spike_threshold": 90.0,
    },
    "REAL_ESTATE": {
        "accrual_ratio_high": 0.18,
        "dso_spike_threshold": 60.0,
    },
    "SOFTWARE_SAAS": {
        "dso_spike_threshold": 45.0,
        "piotroski_strong": 6,
    },
    "INSURANCE": {
        "beneish_manipulation_threshold": -1.40,
        "accrual_ratio_high": 0.25,
        "dso_spike_threshold": 75.0,
    },
}

def get_threshold(key: str, sector: str = "") -> float:
    """Return the sector-adjusted threshold for `key`, falling back to the default."""
    overrides = INDUSTRY_THRESHOLD_OVERRIDES.get(sector.upper(), {})
    if key in overrides:
        return float(overrides[key])
    return float(getattr(FORENSIC_THRESHOLDS, key))

# ─────────────────────────────────────────────
# RISK SCORING WEIGHTS
# ─────────────────────────────────────────────
RISK_SCORE_WEIGHTS = {
    "fraud_indicators":      0.20,   # Beneish M-Score, Dechow F-Score, accruals
    "earnings_quality":      0.20,   # Accrual ratios, cash conversion, NOA
    "cash_flow_quality":     0.15,   # OCF/EBITDA, FCF, financing dependency
    "governance":            0.15,   # Board, SEBI LODR, promoter, RPT (COSO)
    "credit_risk":           0.10,   # Altman EM Z-Score, ICR, D/E
    "auditor_risk":          0.10,   # ISA 240 §A50, PCAOB AS 2101, non-audit fees
    "management_credibility": 0.10,  # ISA 240 §A4, guidance accuracy, NLP
}

RISK_BANDS = {
    (0, 20): "VERY LOW RISK",
    (21, 40): "LOW RISK",
    (41, 60): "MODERATE RISK",
    (61, 80): "HIGH RISK",
    (81, 100): "EXTREME RISK",
}

# ─────────────────────────────────────────────
# REPORT CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class ReportConfig:
    min_word_count: int = 15000
    target_word_count: int = 25000

    # Branding
    firm_name: str = "Forensic AI Intelligence"
    firm_tagline: str = "Evidence-Backed Financial Investigation"

    # Fonts
    heading_font: str = "Calibri"
    body_font: str = "Calibri"

    # Colors (hex)
    primary_color: str = "#1a237e"    # Deep blue
    accent_color: str = "#b71c1c"     # Deep red (for risks)
    safe_color: str = "#1b5e20"       # Deep green (for positives)
    warning_color: str = "#e65100"    # Deep orange (warnings)

REPORT_CONFIG = ReportConfig()

# ─────────────────────────────────────────────
# AGENT CONFIGURATION
# ─────────────────────────────────────────────
AGENT_NAMES = {
    0: "Historical Data Intelligence Agent",
    1: "Document Acquisition Agent",
    2: "Financial Statement Extraction Agent",
    3: "Revenue Forensics Agent",
    4: "Cash Flow Forensics Agent",
    5: "Working Capital Investigation Agent",
    6: "Fraud Detection Agent",
    7: "Credit Risk Agent",
    8: "Earnings Quality Agent",
    9: "Related Party Forensics Agent",
    10: "Auditor Intelligence Agent",
    11: "Management NLP Agent",
    12: "Peer Comparison Agent",
    13: "Historical Fraud Pattern Agent",
    14: "Short Seller Agent",
    15: "Bull Case Agent",
    16: "Devil's Advocate Agent",
    17: "Chief Investigation Director",
}

# ─────────────────────────────────────────────
# KNOWN FRAUD CASES (Reference Database)
# ─────────────────────────────────────────────
FRAUD_CASE_DATABASE = {
    "enron": {
        "company": "Enron Corporation",
        "year": 2001,
        "fraud_type": "Mark-to-Market Accounting, SPE Manipulation",
        "key_signals": [
            "Revenue recognition manipulation",
            "Off-balance-sheet entities",
            "Complex financial structures",
            "Aggressive revenue growth",
            "Auditor conflicts (Arthur Andersen)",
            "Management stock sales preceding collapse",
        ],
        "m_score": -1.0,  # Would have been high
    },
    "wirecard": {
        "company": "Wirecard AG",
        "year": 2020,
        "fraud_type": "Fictitious Revenue, Phantom Cash",
        "key_signals": [
            "Cash on balance sheet not verified",
            "Rapid revenue growth in opaque markets",
            "Third-party acquiring business",
            "Auditor qualifications ignored",
            "Related party transactions",
            "Short seller reports dismissed",
        ],
        "m_score": -1.2,
    },
    "satyam": {
        "company": "Satyam Computer Services",
        "year": 2009,
        "fraud_type": "Cash Fraud, Fake Invoices",
        "key_signals": [
            "Inflated cash and bank balances",
            "Fictitious debtors",
            "Understated liabilities",
            "Promoter credibility issues",
            "Related party concerns",
            "Auditor (PwC) failure",
        ],
        "m_score": -1.5,
    },
    "luckin_coffee": {
        "company": "Luckin Coffee",
        "year": 2020,
        "fraud_type": "Fabricated Revenue, Inflated Transactions",
        "key_signals": [
            "Rapid store expansion with low per-store economics",
            "Inflated same-store sales",
            "Related party transactions",
            "Management credibility",
            "Short seller report (Muddy Waters)",
            "CFO misconduct",
        ],
        "m_score": -0.8,
    },
    "carillion": {
        "company": "Carillion PLC",
        "year": 2018,
        "fraud_type": "Revenue Recognition, Pension Fraud",
        "key_signals": [
            "Aggressive revenue recognition on contracts",
            "Rising debt with maintained dividend",
            "Pension deficit understated",
            "Goodwill impairment deferred",
            "Management bonuses misaligned",
            "Auditor (KPMG) failure",
        ],
        "m_score": -1.3,
    },
    "steinhoff": {
        "company": "Steinhoff International",
        "year": 2017,
        "fraud_type": "Fictitious Transactions, Revenue Overstatement",
        "key_signals": [
            "Complex multi-jurisdictional structure",
            "Related party transactions",
            "Revenue from fictitious deals",
            "Frequent auditor changes",
            "CEO credibility",
            "Governance failures",
        ],
        "m_score": -1.1,
    },
}

# ─────────────────────────────────────────────
# LOGGING CONFIGURATION
# ─────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = DATA_DIR / "logs" / "forensic_ai.log"

# ─────────────────────────────────────────────
# SUPPORTED DOCUMENT TYPES
# ─────────────────────────────────────────────
SUPPORTED_DOCUMENT_TYPES = {
    "annual_report": ["10-K", "20-F", "Annual Report"],
    "quarterly": ["10-Q", "Quarterly Results", "Q1", "Q2", "Q3", "Q4"],
    "concall": ["Earnings Call", "Conference Call", "Transcript"],
    "investor_presentation": ["Investor Presentation", "Analyst Day"],
    "esg": ["ESG Report", "Sustainability Report", "CSR Report"],
    "governance": ["Corporate Governance", "Proxy Statement", "DEF 14A"],
    "credit_rating": ["Credit Rating", "Rating Report"],
    "auditor": ["Audit Report", "Statutory Auditors Report"],
}

# ─────────────────────────────────────────────
# PEER INDUSTRY MAPPING
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# CONTEXT BUILDER CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class ContextConfig:
    # Per-query result count fed into deduplication
    n_results_per_query: int = 6
    # Jaccard overlap threshold for near-duplicate suppression (0-1)
    dedup_threshold: float = 0.65
    # Approximate chars per LLM token (used for budget capping)
    chars_per_token: int = 4
    # Default context budget in tokens when not specified
    default_budget_tokens: int = 3000

CONTEXT_CONFIG = ContextConfig()

# Per-agent retrieval query sets.
# Each agent gets 2–4 targeted sub-queries instead of one generic string.
# Keys are agent IDs matching AGENT_NAMES above.
AGENT_CONTEXT_QUERIES: dict[int, list[str]] = {
    3: [  # Revenue Forensics
        "revenue recognition policy deferred revenue contract assets",
        "channel stuffing year-end revenue spike reversals",
        "accounts receivable growth DSO days sales outstanding",
        "revenue breakdown segment geographic concentration",
    ],
    4: [  # Cash Flow Forensics
        "operating cash flow OCF EBITDA non-cash reconciliation",
        "capital expenditure capex free cash flow FCF",
        "working capital changes cash from operations",
        "financing activities debt issuance equity dilution dividends",
    ],
    5: [  # Working Capital
        "accounts receivable inventory days DIO DPO DSO",
        "cash conversion cycle working capital efficiency",
        "payables stretching creditor days payable outstanding",
        "depreciation asset useful life DEPI",
    ],
    6: [  # Fraud Detection
        "accruals non-cash income earnings management manipulation",
        "revenue recognition aggressive accounting restatement",
        "goodwill impairment asset write-offs intangibles",
        "related party transactions audit qualifications",
    ],
    7: [  # Credit Risk
        "debt leverage interest coverage ratio ICR EBIT interest expense",
        "credit rating downgrade watch negative outlook",
        "long term debt maturity schedule covenant breach refinancing",
        "going concern liquidity cash reserves short-term obligations",
    ],
    8: [  # Earnings Quality
        "accruals net operating assets NOA balance sheet inflation",
        "EBITDA operating profit margin conversion OCF",
        "one-time gains exceptional items non-recurring income",
        "earnings per share guidance vs actual delivery",
    ],
    9: [  # Related Party
        "related party transactions promoter loans inter-company",
        "director remuneration management fees transfer pricing",
        "corporate guarantees pledged shares promoter group entities",
        "SEBI RPT disclosure approval audit committee",
    ],
    10: [  # Auditor Intelligence
        "auditor report going concern opinion material uncertainty",
        "key audit matters KAM emphasis of matter paragraph",
        "material weakness internal control ICFR Section 404",
        "non-audit fees qualified adverse opinion restatement",
    ],
    11: [  # Management NLP
        "MD&A management discussion analysis forward guidance",
        "earnings call transcript CEO CFO commentary analyst questions",
        "risk factors litigation regulatory disclosures",
        "insider transactions promoter buying selling shares",
    ],
    12: [  # Peer Comparison
        "industry average revenue growth margin EBITDA peer",
        "sector benchmark working capital efficiency comparison",
        "competitive position market share business model",
    ],
    14: [  # Investment Committee
        "investment thesis valuation risks opportunities",
        "competitive advantage moat management quality track record",
        "growth catalysts risks bull bear case",
    ],
    17: [  # Director
        "overall risk verdict recommendation summary",
        "convergent red flags fraud governance credit evidence",
        "material findings cross-validation synthesis",
    ],
}

# ─────────────────────────────────────────────
# OUTPUT HARNESS CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class HarnessConfig:
    # Minimum words for an LLM response to be considered valid
    min_response_words: int = 25
    # Whether to append JSON schema instruction to all generic prompts
    request_structured_output: bool = True
    # Max findings to extract per agent from LLM output
    max_findings_per_agent: int = 12
    # Per-agent LLM call timeout in seconds (for ThreadPoolExecutor)
    agent_timeout_seconds: int = 90

HARNESS_CONFIG = HarnessConfig()

# ─────────────────────────────────────────────
# AGENTIC RAG CONFIGURATION
# LangGraph + LlamaIndex + LangChain pipeline
# ─────────────────────────────────────────────
@dataclass
class AgenticRAGConfig:
    # Enable the full LangGraph pipeline; set False to use classic path
    enabled: bool = os.getenv("AGENTIC_RAG_ENABLED", "true").lower() == "true"
    # Max query-rewrite cycles per agent before accepting response
    max_iterations: int = int(os.getenv("AGENTIC_RAG_MAX_ITER", "3"))
    # Enable LlamaIndex for vector retrieval (vs. existing HybridRetriever)
    use_llamaindex: bool = os.getenv("LLAMAINDEX_ENABLED", "true").lower() == "true"
    # Enable internet search in the agentic pipeline
    internet_search_enabled: bool = os.getenv("INTERNET_SEARCH_ENABLED", "true").lower() == "true"
    # Enable live financial data via yfinance / tools_api source
    tools_api_enabled: bool = os.getenv("TOOLS_API_ENABLED", "true").lower() == "true"
    # Preferred internet search provider: tavily | duckduckgo | serper | serpapi
    search_provider: str = os.getenv("SEARCH_PROVIDER", "duckduckgo")

AGENTIC_RAG_CONFIG = AgenticRAGConfig()

# ─────────────────────────────────────────────
# PEER INDUSTRY MAPPING
# ─────────────────────────────────────────────
INDUSTRY_PEER_GROUPS = {
    "IT_SERVICES": ["Infosys", "TCS", "Wipro", "HCL Technologies", "Tech Mahindra", "Accenture", "IBM"],
    "BANKING": ["HDFC Bank", "ICICI Bank", "State Bank of India", "Axis Bank", "Kotak Mahindra"],
    "OIL_GAS": ["Reliance Industries", "ONGC", "Indian Oil", "BPCL", "ExxonMobil", "Shell"],
    "FMCG": ["HUL", "Nestle", "P&G", "Britannia", "Dabur", "Marico"],
    "PHARMA": ["Sun Pharma", "Dr Reddy", "Cipla", "Lupin", "Aurobindo", "Pfizer"],
    "TECHNOLOGY": ["Apple", "Microsoft", "Google", "Amazon", "Meta", "Nvidia"],
    "AUTOMOTIVE": ["Maruti Suzuki", "Tata Motors", "M&M", "Toyota", "Hyundai"],
    "TELECOM": ["Reliance Jio", "Airtel", "Vodafone Idea", "AT&T", "Verizon"],
    "REAL_ESTATE": ["DLF", "Godrej Properties", "Prestige", "Sobha"],
    "STEEL": ["Tata Steel", "JSW Steel", "SAIL", "ArcelorMittal"],
}
