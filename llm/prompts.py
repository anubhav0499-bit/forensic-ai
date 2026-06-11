"""
Forensic AI - System Prompts and Prompt Builders
Frameworks: ISA 240, PCAOB AS 2401/2101, COSO Internal Control Framework,
            SEBI LODR/SAST/RPT Regulations 2021, OECD Corporate Governance Principles,
            Beneish (1999) M-Score, Dechow et al. (2011) F-Score, Altman EM Z-Score.
"""

SYSTEM_PROMPTS = {
    "forensic_accountant": """You are a senior forensic accountant with 20+ years of experience at Big Four firms, applying ISA 240 (Auditor's Responsibilities Relating to Fraud) and PCAOB AS 2401.
Your mandate is to uncover accounting manipulation, fraud indicators, and earnings quality issues.
Apply the Dechow et al. (2011) F-Score framework: accrual ratio = (Net Income – OCF) / Avg Total Assets. >5% = elevated; >10% = critical misreporting risk.
Apply the Beneish M-Score: M > -1.78 = likely manipulator; -2.22 to -1.78 = grey zone. Count indices above paper thresholds: 3+ = elevated risk regardless of M-Score.
You follow the Evidence → Analysis → Reasoning → Conclusion framework.
Never speculate. Only make conclusions supported by numerical evidence.
Always cite specific financial statement line items, years, and percentage changes.
Flag any finding with: SEVERITY (CRITICAL/HIGH/MEDIUM/LOW), EVIDENCE, ANALYSIS, CONCLUSION.""",

    "credit_analyst": """You are a senior credit analyst at a major rating agency applying the Altman Emerging Market Z-Score framework.
EM Z' = 6.56(WC/TA) + 3.26(RE/TA) + 6.72(EBIT/TA) + 1.05(BVE/TL). Safe: Z'>2.60; Grey: 1.10-2.60; Distress: Z'<1.10.
You specialize in solvency analysis, debt serviceability, and credit deterioration signals.
Key thresholds — Interest Coverage (ICR): <1.5x = stress; <1.0x = default risk. Debt/EBITDA: >4x = leveraged; >6x = highly leveraged.
Short-term debt >60% of total debt = rollover risk. Net Debt/Equity >3x with declining revenue = critical.
Assess: liquidity ratios, debt structure, interest coverage, covenant risks, refinancing risks.
Follow CRISIL/ICRA/Moody's/S&P analytical framework. Cite specific numbers and thresholds.""",

    "fraud_investigator": """You are a forensic fraud investigator applying ISA 240 (The Auditor's Responsibilities Relating to Fraud in an Audit of Financial Statements) and PCAOB AS 2401.
You identify financial statement fraud using Beneish M-Score, Dechow F-Score, Benford's Law, and PCAOB red flag frameworks.
Beneish M-Score > -1.78 = manipulation likely. Count Beneish indices above paper thresholds: 3+ = elevated risk regardless of M-Score.
Dechow F-Score: base rate = 1%; F-Score > 1% = above average; > 10% = extreme risk (10x base rate).
Apply COSO fraud triangle: pressure, opportunity, rationalisation.
Every accusation requires three independent corroborating data points. Cross-reference management disclosures with financial data.""",

    "equity_analyst": """You are a senior equity research analyst applying Dechow et al. (2011) F-Score earnings quality framework.
Accrual ratio = (Net Income – OCF) / Average Total Assets. >5% = elevated; >10% = critical misreporting risk.
EBITDA-to-OCF conversion: healthy firms convert >80%. Below 60% sustained = earnings quality concern.
Net Operating Assets (NOA) trend: sustained increase signals balance sheet inflation.
You specialize in earnings quality, cash flow analysis, and identification of accounting adjustments.
Produce institutional-grade analysis suitable for investment committee presentation.""",

    "auditor_intelligence": """You are an expert in audit quality assessment applying ISA 240 §A50 and PCAOB AS 2101 (Audit Planning and Supervision).
Auditor independence standards (PCAOB Rule 3521): non-audit fees >40% of total fees = independence risk.
Auditor downgrade (Big 4 → smaller firm) = highest-risk governance signal.
Going concern emphasis of matter paragraphs (ISA 570) = CRITICAL flag regardless of other indicators.
Qualified opinions (ISA 705) = immediate escalation. Material weakness in ICFR = precedes restatements.
Analyze Key Audit Matters (KAMs), auditor qualifications, audit firm history, and going concern issues.
Compare against industry standards and historical patterns at comparable firms.""",

    "governance_specialist": """You are a corporate governance expert applying COSO Internal Control Framework, SEBI LODR Regulations (Reg. 17-27), SEBI SAST, SEBI RPT Regulations 2021, and OECD Corporate Governance Principles.
SEBI LODR: board must be ≥50% independent directors (Reg. 17). Audit committee must be fully independent (Reg. 18).
Promoter pledge risk bands (SEBI SAST): >30% = elevated; >50% = high; >70% = critical with collateral-call risk.
Director exits ≥2 in 12 months = governance stress (COSO Control Environment deficiency signal).
CFO change is correlated with earnings manipulation per Dechow et al. (2010).
SEBI RPT Regulations 2021: all material RPTs require shareholder approval; related-party loans >10% net worth = SEBI materiality threshold.
Benchmark against SEBI/SEC governance codes. Identify governance red flags per OECD Principles of Corporate Governance (2023).""",

    "management_nlp": """You are an expert in detecting linguistic patterns in financial communications applying ISA 240 §A4 (management override risk indicators).
Guidance miss rate >40% over 3 years = low management credibility (ISA 240 §A4 fraud risk factor).
Insider selling while making bullish public statements = management credibility red flag.
Promoter remuneration >10% of PAT (especially in loss years) = governance and credibility concern.
Analyze MD&A, earnings calls, and shareholder letters for: evasive language, over-optimism,
selective disclosure, narrative inconsistencies, hedging language, and deflection patterns.
Use Loughran-McDonald financial word lists and deception detection frameworks.""",

    "short_seller": """You are a professional short seller analyst at a major short-selling research firm.
Build the strongest possible bear case using only evidence from public disclosures.
Identify: overvaluation, accounting red flags, governance issues, competitive threats.
Apply PCAOB AS 2401 §66 evidence evaluation: are red flags isolated or systemic? Systemic patterns = highest-confidence fraud indicators.
This is for investment research purposes. Cite every claim with specific evidence.""",

    "bull_case": """You are an institutional bull-case equity analyst.
Present the strongest possible investment thesis using evidence from disclosures.
Identify: competitive moats, management quality, growth catalysts, financial strength.
Challenge bear case assumptions with counterevidence. Document genuine green flags:
long auditor tenure with clean opinions, zero promoter pledge, consistent guidance accuracy, strong OCF conversion.""",

    "devils_advocate": """You are the Devil's Advocate on the investment committee.
Challenge every conclusion made by other analysts. Find counterevidence.
Apply Bayesian calibration: a single red flag in an otherwise clean company is categorically different from 3 red flags in a weak governance environment.
Ask: What could we be wrong about? What evidence contradicts our thesis?
Force the team to defend every significant conclusion.""",

    "investment_director": """You are the Chief Investment Officer chairing the investment committee, applying PCAOB AS 2401 §66 and ISA 330 evidence evaluation.
Synthesize findings from 16 specialist analysts. Apply dimension weights: Fraud Indicators 20%, Earnings Quality 20%, Cash Flow Quality 15%, Governance 15%, Credit Risk 10%, Auditor Risk 10%, Management Credibility 10%.
Evaluate signal convergence: isolated vs. systemic red flags. Resolve conflicts. Weigh evidence.
Produce a final verdict: STRONG AVOID / AVOID / CAUTION / MONITOR / CAUTIOUS BUY / BUY with clear reasoning.
Risk-adjust the recommendation. This memo goes to the portfolio manager.""",
}


def build_analysis_prompt(
    agent_role: str,
    company_name: str,
    fiscal_years: list,
    financial_data: dict,
    extracted_text: str = "",
    question: str = "",
    additional_context: str = "",
) -> str:
    """Build a structured analysis prompt for forensic investigation."""
    years_str = ", ".join(fiscal_years) if fiscal_years else "Available"

    prompt = f"""
INVESTIGATION BRIEF
===================
Company: {company_name}
Fiscal Years Under Investigation: {years_str}
Role: {agent_role}

FINANCIAL DATA:
{_format_financial_data(financial_data)}

{f"SOURCE DOCUMENT EXCERPTS:{chr(10)}{extracted_text[:3000]}" if extracted_text else ""}

{f"ADDITIONAL CONTEXT:{chr(10)}{additional_context}" if additional_context else ""}

INVESTIGATION TASK:
{question}

REQUIRED OUTPUT FORMAT:
1. KEY FINDINGS (numbered list with evidence)
2. RISK INDICATORS (classify each: CRITICAL/HIGH/MEDIUM/LOW)
3. EVIDENCE CITATIONS (specific numbers, years, source documents)
4. CONCLUSION
5. RECOMMENDED FOLLOW-UP QUESTIONS

Remember: Every finding must be supported by specific financial data.
"""
    return prompt.strip()


def _format_financial_data(data: dict) -> str:
    """Format financial data dict for prompt injection."""
    if not data:
        return "No structured financial data available."
    lines = []
    for year, metrics in sorted(data.items(), reverse=True):
        lines.append(f"\n{'─'*40}")
        lines.append(f"FISCAL YEAR: {year}")
        lines.append(f"{'─'*40}")
        for metric, value in metrics.items():
            if value is not None and value != 0:
                if isinstance(value, float):
                    lines.append(f"  {metric}: {value:,.2f}")
                else:
                    lines.append(f"  {metric}: {value}")
    return "\n".join(lines) if lines else "No data available."


def build_fraud_detection_prompt(
    company: str,
    m_score: float,
    z_score: float,
    f_score: int,
    accrual_ratio: float,
    flags_above_paper_threshold: int = 0,
) -> str:
    # Altman EM thresholds: Safe>2.60, Grey 1.10-2.60, Distress<1.10
    altman_label = "DISTRESS ZONE" if z_score < 1.10 else "GREY ZONE" if z_score < 2.60 else "SAFE ZONE"
    # Dechow F-Score: base rate 1%; >5% elevated; >10% extreme
    accrual_label = "CRITICAL (>10% — 10x base rate)" if abs(accrual_ratio) > 0.10 else "ELEVATED (>5%)" if abs(accrual_ratio) > 0.05 else "Acceptable"
    return f"""
FRAUD DETECTION ANALYSIS - {company}
=====================================
Frameworks: ISA 240 / PCAOB AS 2401 / Beneish (1999) / Dechow et al. (2011) F-Score / Altman EM (2000)

QUANTITATIVE FRAUD INDICATORS:
- Beneish M-Score:               {m_score:.4f}  {'ABOVE MANIPULATION THRESHOLD (-1.78) — MANIPULATOR LIKELY' if m_score > -1.78 else 'GREY ZONE (-2.22 to -1.78) — MONITOR' if m_score > -2.22 else 'Below manipulation threshold'}
- Beneish Indices Above Paper Threshold: {flags_above_paper_threshold}/8  {'ELEVATED BENEISH RISK (3+ indices)' if flags_above_paper_threshold >= 3 else 'MODERATE (1-2 indices above threshold)' if flags_above_paper_threshold >= 1 else 'Clean'}
- Altman EM Z-Score (non-mfg):   {z_score:.4f}  {altman_label}
  EM Thresholds: Safe>2.60, Grey 1.10-2.60, Distress<1.10
- Piotroski F-Score:             {f_score}/9   {'STRONG' if f_score >= 7 else 'WEAK' if f_score <= 2 else 'NEUTRAL'}
- Dechow Accrual Ratio (F-Score): {accrual_ratio:.4f}  {accrual_label}
  Base rate of misstatement: 1%. Ratio >5% = elevated; >10% = 10x base rate.

Based on these quantitative signals (ISA 240 §26 / PCAOB AS 2401), provide:
1. Overall fraud risk assessment with COSO fraud triangle (pressure / opportunity / rationalisation)
2. Which specific Beneish components breach the Beneish (1999) paper thresholds
3. Dechow F-Score interpretation and accruals analysis
4. Historical fraud cases with similar quantitative profiles (Satyam, Wirecard, Enron, Luckin)
5. Recommended investigation areas under ISA 240 §27-32
6. Confidence level in the assessment
"""


def build_management_nlp_prompt(company: str, concall_text: str, mda_text: str) -> str:
    return f"""
MANAGEMENT COMMUNICATION ANALYSIS - {company}
=============================================

Analyze the following management communications for:
1. Evasive language and non-answers to analyst questions
2. Over-optimistic forward guidance relative to historical delivery
3. Selective disclosure (emphasizing positives, burying negatives)
4. Changes in language from prior periods (tone shift analysis)
5. Frequent use of non-GAAP metrics to obscure GAAP performance
6. Specific vague phrases that avoid accountability

EARNINGS CALL EXCERPT:
{concall_text[:2000] if concall_text else "Not available"}

MD&A EXCERPT:
{mda_text[:2000] if mda_text else "Not available"}

Score each dimension 0-10 (10 = most concerning).
Provide specific quotes as evidence for each finding.
"""


def build_peer_comparison_prompt(company: str, company_metrics: dict, peer_metrics: dict) -> str:
    return f"""
PEER BENCHMARKING ANALYSIS - {company}
=======================================

COMPANY METRICS:
{_format_financial_data({company: company_metrics})}

PEER GROUP METRICS:
{_format_financial_data(peer_metrics)}

Analyze:
1. Which metrics are significant positive outliers vs peers? (Could indicate manipulation)
2. Which metrics are significant negative outliers? (Risk indicators)
3. Margin profile vs industry: is the company structurally more profitable? Why?
4. Working capital efficiency vs peers
5. Cash conversion vs peers
6. Revenue growth vs peers

For each outlier, assess: (a) legitimate business model advantage, or (b) potential accounting manipulation?
"""


def build_rpt_prompt(
    company: str,
    rpt_revenue_share: float,
    promoter_loan_ratio: float,
    disclosure_context: str,
    quantitative_flags: list,
) -> str:
    """Related-party forensics prompt — SEBI RPT Regulations 2021 / OECD Principles."""
    flags_text = "\n".join(f"  - {f}" for f in quantitative_flags) if quantitative_flags else "  None detected quantitatively."
    return f"""
RELATED PARTY FORENSICS — {company}
=====================================
Frameworks: SEBI RPT Regulations 2021 (amended) / OECD Corporate Governance Principles (2023) / ISA 550

PRE-COMPUTED QUANTITATIVE FLAGS:
{flags_text}

KEY METRICS:
- RPT Revenue as % of Total Revenue: {rpt_revenue_share*100:.1f}%  {'CRITICAL (>40%)' if rpt_revenue_share > 0.40 else 'HIGH (>25%)' if rpt_revenue_share > 0.25 else 'ELEVATED (>15%)' if rpt_revenue_share > 0.15 else 'Within threshold'}
  Threshold: >15% = concentration risk; >25% = captive-customer/circular risk; >40% = critical
- Promoter/Director Loans as % of Net Worth: {promoter_loan_ratio*100:.1f}%
  SEBI RPT 2021 materiality threshold: >10% net worth requires shareholder approval

SEBI RPT REGULATIONS 2021 — KEY PROVISIONS TO CHECK:
1. All material RPTs (value > lower of Rs.1,000 cr or 10% of annual consolidated turnover) require
   prior shareholder approval via special resolution (Reg. 23, LODR 2021 amendment)
2. RPTs with promoters/promoter group entities must be disclosed in the annual report
3. Audit committee must pre-approve or ratify all RPTs above materiality thresholds
4. Listed entities cannot enter into RPTs that benefit the promoter group to the detriment of minority shareholders

TUNNELLING PATTERNS (OECD Principle VI.C) TO DETECT:
- Loans from company to promoter entities at below-market rates or unsecured
- Goods/services sold to related parties at below-market prices (revenue understatement)
- Goods/services purchased from related parties at above-market prices (cost inflation)
- Circular revenue transactions (company → related entity → back, inflating both entities' revenues)
- Guarantees given by company for promoter group debt without commercial benefit
- Acquisitions of promoter group assets at inflated valuations (wealth transfer to promoter)

DISCLOSURE CONTEXT:
{disclosure_context[:2500] if disclosure_context else "Not available."}

Investigate (Evidence → Analysis → Conclusion):
1. Map all significant RPTs by entity, nature, amount, and whether audit-committee-approved
2. Are RPT prices arm's length? Compare to third-party terms where discernible
3. Identify any tunnelling patterns — wealth flowing from listed entity to promoter group
4. Assess disclosure quality: SEBI-compliant vs. boilerplate vs. opacity
5. Circular transaction risk: revenue round-tripping between related entities
6. Flag each finding CRITICAL / HIGH / MEDIUM with specific SEBI regulation cited
"""


def build_credit_risk_prompt(
    company: str,
    interest_coverage: float | None,
    net_debt_ebitda: float | None,
    debt_equity: float | None,
    altman_z: float | None,
    altman_zone: str | None,
    short_term_debt_ratio: float | None,
    trend_summary: str,
    document_context: str,
) -> str:
    """Credit risk prompt — Altman EM Z-Score / CRISIL metrics / ISA 570."""
    ic_label = "STRESS (<1.5x)" if (interest_coverage or 999) < 1.5 else "HIGH RISK (<2.5x)" if (interest_coverage or 999) < 2.5 else "Adequate"
    z_label = altman_zone or ("DISTRESS" if (altman_z or 999) < 1.10 else "GREY" if (altman_z or 999) < 2.60 else "SAFE")
    return f"""
CREDIT RISK ASSESSMENT — {company}
=====================================
Frameworks: Altman EM Z-Score (2000) / CRISIL Credit Metrics / ISA 570 Going Concern

ALTMAN EMERGING MARKET Z-SCORE (NON-MANUFACTURING):
Formula: Z' = 6.56(WC/TA) + 3.26(RE/TA) + 6.72(EBIT/TA) + 1.05(BVE/TL)
- Z-Score: {f"{altman_z:.3f}" if altman_z is not None else "N/A"}  Zone: {z_label}
- Thresholds: Safe > 2.60 | Grey 1.10-2.60 | Distress < 1.10

CRISIL/ICRA CREDIT METRICS (LATEST YEAR):
- Interest Coverage Ratio (EBIT/Interest): {f"{interest_coverage:.2f}x" if interest_coverage is not None else "N/A"}  [{ic_label}]
  Thresholds: >4x = comfortable | 2.5-4x = adequate | 1.5-2.5x = stressed | <1.5x = near-default | <1.0x = default risk
- Net Debt/EBITDA: {f"{net_debt_ebitda:.1f}x" if net_debt_ebitda is not None else "N/A"}
  Thresholds: <2x = low leverage | 2-4x = moderate | 4-6x = highly leveraged | >6x = distressed
- Debt/Equity: {f"{debt_equity:.2f}x" if debt_equity is not None else "N/A"}
  Thresholds: manufacturing <2x healthy | >3x with declining revenue = critical
- Short-Term Debt as % of Total Debt: {f"{short_term_debt_ratio*100:.1f}%" if short_term_debt_ratio is not None else "N/A"}
  Threshold: >60% = rollover / refinancing risk

MULTI-YEAR TREND:
{trend_summary}

DOCUMENT CONTEXT (rating agency / annual report debt section):
{document_context[:1500] if document_context else "Not available."}

Assess (Evidence → Analysis → Reasoning → Conclusion):
1. Altman EM Z-Score zone and trajectory — is the company moving toward or away from distress?
2. Debt serviceability: can EBIT cover interest? What headroom exists above covenant thresholds?
3. Liquidity risk: current ratio, quick ratio, short-term debt refinancing requirements
4. Rollover risk: concentration of maturities in near-term
5. Rating trajectory if disclosed (CRISIL/ICRA downgrades are early warning signals)
6. Going concern assessment per ISA 570: at what point do current ratios trigger a going concern qualification?
7. Compare implied credit rating vs. any formal rating agency assessment
"""


def build_auditor_risk_prompt(
    company: str,
    auditor_name: str,
    is_big_four: bool,
    going_concern: bool,
    material_weakness: bool,
    restatement: bool,
    qualified_opinion: bool,
    emphasis_of_matter: bool,
    kam_count: int,
    tenure_years: int | None,
    non_audit_fee_ratio: float | None,
    audit_context: str,
) -> str:
    """Auditor risk prompt — ISA 240 §A50 / PCAOB AS 2101 / PCAOB Rule 3521."""
    non_audit_label = (
        f"{non_audit_fee_ratio*100:.1f}% — INDEPENDENCE RISK (>40% threshold, PCAOB Rule 3521)"
        if non_audit_fee_ratio is not None and non_audit_fee_ratio > 0.40
        else f"{non_audit_fee_ratio*100:.1f}% — Within threshold" if non_audit_fee_ratio is not None
        else "Not disclosed"
    )
    return f"""
AUDITOR INTELLIGENCE ASSESSMENT — {company}
==========================================
Frameworks: ISA 240 §A50 / PCAOB AS 2101 (Audit Planning) / PCAOB Rule 3521 / ISA 570 / ISA 705

AUDIT SIGNALS DETECTED:
- Auditor: {auditor_name}  (Big 4: {is_big_four})
  [Big 4 → smaller firm downgrade = CRITICAL governance signal per ISA 240 §A50]
- Going Concern Flag (ISA 570): {going_concern}  {'CRITICAL — immediate escalation required' if going_concern else 'Not detected'}
- Material Weakness in ICFR (PCAOB AS 2201): {material_weakness}  {'CRITICAL — precedes restatements' if material_weakness else 'Not detected'}
- Prior Period Restatement: {restatement}  {'CRITICAL — strongest single predictor of fraud' if restatement else 'Not detected'}
- Qualified / Adverse Opinion (ISA 705): {qualified_opinion}  {'HIGH risk — exceptions to fair presentation' if qualified_opinion else 'Not detected'}
- Emphasis of Matter Paragraph: {emphasis_of_matter}
- Key Audit Matters (KAMs) count: {kam_count}  {'HIGH (>4 KAMs = multiple areas of significant auditor judgment)' if kam_count >= 5 else 'Normal range' if kam_count >= 2 else 'Low'}
- Estimated Auditor Tenure: {f"{tenure_years} years" if tenure_years is not None else "Unknown"}
  [>10 years = independence review warranted; note SEBI rotation for listed Indian companies]
- Non-Audit Fees as % of Total Audit Fees: {non_audit_label}
  [PCAOB Rule 3521: >40% ratio creates independence threat]

PCAOB AS 2101 / ISA 240 §A50 AUDIT QUALITY INDICATORS:
- Auditor firm size relative to client complexity
- Industry specialisation of audit partner
- Audit committee engagement with auditor
- Changes in accounting estimates or audit approaches year-over-year

AUDITOR REPORT EXCERPTS:
{audit_context[:2500] if audit_context else "Not available."}

Assess (Evidence → Analysis → Reasoning → Conclusion):
1. Audit quality and independence: does the auditor have the capacity and independence to catch management fraud?
2. What do KAMs reveal about areas of highest financial statement risk?
3. Going concern / qualified opinion: what specific facts triggered this and what does it signal about management integrity?
4. Non-audit fee structure: does the revenue dependency impair independence (PCAOB Rule 3521)?
5. Auditor tenure and rotation: independence risk vs. institutional knowledge trade-off
6. What should a forensic investigator focus on given these auditor signals (ISA 240 §A50 guidance)?
"""
