"""
Database Schema Definitions
"""

SCHEMA_SQL = """
-- Companies master
CREATE TABLE IF NOT EXISTS companies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    ticker TEXT,
    isin TEXT,
    exchange TEXT,
    sector TEXT,
    industry TEXT,
    country TEXT,
    market_cap REAL,
    currency TEXT DEFAULT 'INR',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Documents registry
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER REFERENCES companies(id),
    document_type TEXT NOT NULL,
    fiscal_year TEXT,
    quarter TEXT,
    source_url TEXT,
    local_path TEXT,
    file_size_bytes INTEGER,
    download_date TEXT DEFAULT (datetime('now')),
    parsed BOOLEAN DEFAULT FALSE,
    parse_date TEXT,
    checksum TEXT,
    UNIQUE(company_id, document_type, fiscal_year, quarter)
);

-- Financial statements (annual)
CREATE TABLE IF NOT EXISTS financial_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER REFERENCES companies(id),
    fiscal_year TEXT NOT NULL,
    period_end_date TEXT,
    -- Income Statement
    revenue REAL,
    revenue_growth REAL,
    gross_profit REAL,
    gross_margin REAL,
    ebitda REAL,
    ebitda_margin REAL,
    ebit REAL,
    ebit_margin REAL,
    pbt REAL,
    tax REAL,
    pat REAL,
    pat_margin REAL,
    eps REAL,
    eps_diluted REAL,
    -- Balance Sheet
    total_assets REAL,
    current_assets REAL,
    cash_equivalents REAL,
    accounts_receivable REAL,
    inventory REAL,
    prepaid_expenses REAL,
    ppe_gross REAL,
    ppe_net REAL,
    goodwill REAL,
    intangible_assets REAL,
    total_liabilities REAL,
    current_liabilities REAL,
    accounts_payable REAL,
    short_term_debt REAL,
    long_term_debt REAL,
    total_debt REAL,
    deferred_revenue REAL,
    shareholder_equity REAL,
    retained_earnings REAL,
    -- Cash Flow
    cfo REAL,
    cfi REAL,
    cff REAL,
    capex REAL,
    fcf REAL,
    dividends_paid REAL,
    -- Derived Ratios
    current_ratio REAL,
    quick_ratio REAL,
    de_ratio REAL,
    interest_coverage REAL,
    roe REAL,
    roa REAL,
    roce REAL,
    asset_turnover REAL,
    inventory_days REAL,
    receivable_days REAL,
    payable_days REAL,
    cash_conversion_cycle REAL,
    -- Source
    source_document TEXT,
    extraction_confidence REAL DEFAULT 0.0,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Forensic scores
CREATE TABLE IF NOT EXISTS forensic_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER REFERENCES companies(id),
    fiscal_year TEXT NOT NULL,
    -- Beneish M-Score
    beneish_m_score REAL,
    beneish_dsri REAL,
    beneish_gmi REAL,
    beneish_aqi REAL,
    beneish_sgi REAL,
    beneish_depi REAL,
    beneish_sgai REAL,
    beneish_lvgi REAL,
    beneish_tata REAL,
    beneish_manipulation_flag BOOLEAN,
    -- Altman Z-Score
    altman_z_score REAL,
    altman_x1 REAL,
    altman_x2 REAL,
    altman_x3 REAL,
    altman_x4 REAL,
    altman_x5 REAL,
    altman_zone TEXT,
    -- Piotroski F-Score
    piotroski_f_score INTEGER,
    piotroski_roa_positive BOOLEAN,
    piotroski_cfo_positive BOOLEAN,
    piotroski_roa_change BOOLEAN,
    piotroski_accrual BOOLEAN,
    piotroski_leverage_change BOOLEAN,
    piotroski_liquidity_change BOOLEAN,
    piotroski_shares_issued BOOLEAN,
    piotroski_gross_margin_change BOOLEAN,
    piotroski_asset_turnover_change BOOLEAN,
    -- Dechow F-Score
    dechow_f_score REAL,
    dechow_manipulation_flag BOOLEAN,
    -- Accruals
    accrual_ratio REAL,
    cash_earnings_ratio REAL,
    -- Overall Risk
    overall_risk_score REAL,
    fraud_risk_score REAL,
    earnings_quality_score REAL,
    credit_risk_score REAL,
    governance_score REAL,
    auditor_risk_score REAL,
    cash_flow_quality_score REAL,
    management_credibility_score REAL,
    risk_band TEXT,
    calculated_at TEXT DEFAULT (datetime('now'))
);

-- Agent findings
CREATE TABLE IF NOT EXISTS agent_findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER REFERENCES companies(id),
    agent_id INTEGER NOT NULL,
    agent_name TEXT NOT NULL,
    fiscal_year TEXT,
    finding_type TEXT,
    finding_title TEXT NOT NULL,
    finding_detail TEXT,
    evidence TEXT,
    source_document TEXT,
    risk_level TEXT DEFAULT 'MEDIUM',
    confidence REAL DEFAULT 0.5,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Red flags registry
CREATE TABLE IF NOT EXISTS red_flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER REFERENCES companies(id),
    agent_id INTEGER,
    agent_name TEXT,
    flag_title TEXT NOT NULL,
    flag_detail TEXT,
    evidence TEXT,
    severity TEXT DEFAULT 'HIGH',
    fiscal_year TEXT,
    source_document TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Green flags registry
CREATE TABLE IF NOT EXISTS green_flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER REFERENCES companies(id),
    agent_id INTEGER,
    agent_name TEXT,
    flag_title TEXT NOT NULL,
    flag_detail TEXT,
    evidence TEXT,
    fiscal_year TEXT,
    source_document TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Audit trail
CREATE TABLE IF NOT EXISTS audit_trail (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER REFERENCES companies(id),
    timestamp TEXT DEFAULT (datetime('now')),
    agent_id INTEGER,
    agent_name TEXT,
    action TEXT NOT NULL,
    detail TEXT,
    source_url TEXT,
    source_document TEXT
);

-- Related party transactions
CREATE TABLE IF NOT EXISTS related_party_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER REFERENCES companies(id),
    fiscal_year TEXT,
    party_name TEXT,
    relationship TEXT,
    transaction_type TEXT,
    amount REAL,
    currency TEXT,
    disclosure_quality TEXT,
    risk_flag BOOLEAN DEFAULT FALSE,
    source_document TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Auditor history
CREATE TABLE IF NOT EXISTS auditor_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER REFERENCES companies(id),
    fiscal_year TEXT,
    auditor_name TEXT,
    audit_firm_tier TEXT,
    auditor_changed BOOLEAN DEFAULT FALSE,
    change_reason TEXT,
    qualification TEXT,
    key_audit_matters TEXT,
    going_concern_flag BOOLEAN DEFAULT FALSE,
    emphasis_of_matter TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Shareholding data
CREATE TABLE IF NOT EXISTS shareholding_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER REFERENCES companies(id),
    fiscal_year TEXT,
    quarter TEXT,
    promoter_holding REAL,
    promoter_pledge REAL,
    fii_holding REAL,
    dii_holding REAL,
    public_holding REAL,
    promoter_holding_change REAL,
    pledge_change REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Concall intelligence
CREATE TABLE IF NOT EXISTS concall_intelligence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER REFERENCES companies(id),
    fiscal_year TEXT,
    quarter TEXT,
    call_date TEXT,
    evasion_score REAL,
    optimism_score REAL,
    uncertainty_score REAL,
    narrative_consistency_score REAL,
    key_phrases TEXT,
    red_flag_phrases TEXT,
    sentiment_score REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Investigation sessions
CREATE TABLE IF NOT EXISTS investigation_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER REFERENCES companies(id),
    session_start TEXT DEFAULT (datetime('now')),
    session_end TEXT,
    status TEXT DEFAULT 'IN_PROGRESS',
    overall_risk_score REAL,
    report_path TEXT,
    summary TEXT
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_financial_company_year ON financial_data(company_id, fiscal_year);
CREATE INDEX IF NOT EXISTS idx_forensic_company_year ON forensic_scores(company_id, fiscal_year);
CREATE INDEX IF NOT EXISTS idx_findings_company ON agent_findings(company_id, agent_id);
CREATE INDEX IF NOT EXISTS idx_documents_company ON documents(company_id, document_type);
"""
