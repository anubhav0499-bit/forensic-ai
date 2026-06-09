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

# Force a provider or use cascade (auto).
# Values: auto | groq | openai | anthropic | gemini | together | openrouter | lmstudio | ollama | hf
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")

# Per-provider model names (override individually via env if needed)
PROVIDER_MODELS: dict[str, dict[str, str]] = {
    "groq":        {"primary": os.getenv("GROQ_MODEL",        "llama-3.3-70b-versatile"),
                    "fast":    os.getenv("GROQ_FAST_MODEL",    "llama-3.1-8b-instant")},
    "openai":      {"primary": os.getenv("OPENAI_MODEL",       "gpt-4o"),
                    "fast":    os.getenv("OPENAI_FAST_MODEL",   "gpt-4o-mini")},
    "anthropic":   {"primary": os.getenv("ANTHROPIC_MODEL",    "claude-opus-4-8"),
                    "fast":    os.getenv("ANTHROPIC_FAST_MODEL","claude-haiku-4-5-20251001")},
    "gemini":      {"primary": os.getenv("GEMINI_MODEL",       "gemini-1.5-pro-latest"),
                    "fast":    os.getenv("GEMINI_FAST_MODEL",   "gemini-2.0-flash")},
    "together":    {"primary": os.getenv("TOGETHER_MODEL",     "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
                    "fast":    os.getenv("TOGETHER_FAST_MODEL", "meta-llama/Llama-3.1-8B-Instruct-Turbo")},
    "openrouter":  {"primary": os.getenv("OPENROUTER_MODEL",   "anthropic/claude-3.5-sonnet"),
                    "fast":    os.getenv("OPENROUTER_FAST_MODEL","google/gemini-flash-1.5")},
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
    device: str = "cuda" if os.getenv("USE_GPU", "auto") == "auto" else "cpu"
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

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64
    table_chunk_size: int = 1024

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

    # Altman Z-Score (manufacturing)
    altman_safe_zone: float = 2.99
    altman_grey_zone: float = 1.81

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
    "fraud_indicators": 0.25,
    "earnings_quality": 0.20,
    "cash_flow_quality": 0.20,
    "governance": 0.15,
    "credit_risk": 0.10,
    "auditor_risk": 0.05,
    "management_credibility": 0.05,
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
