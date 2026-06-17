"""
Forensic AI — Core Knowledge Base
===================================
Single source of truth for:
  - Reasoning framework (Evidence → Conclusion 6-step chain)
  - Hallucination control protocol
  - Evidence priority order
  - ISA / SA / PCAOB / COSO standards catalog
  - Indian regulatory provisions (CARO 2020, Companies Act 2013, SEBI)
  - Fraud detection indicator catalog with thresholds
  - Audit procedure templates per risk type
  - Agent-to-standards and agent-to-case mappings

Usage:
  from knowledge import get_agent_knowledge_block, get_audit_procedures
"""

from __future__ import annotations

# ─── Reasoning Framework ─────────────────────────────────────────────────────

REASONING_FRAMEWORK = """
MANDATORY REASONING CHAIN — apply to every finding:
  Step 1 EVIDENCE    Identify the specific factual data point (number, ratio, disclosure)
  Step 2 STANDARD    Map to the applicable standard (ISA/SA/PCAOB/Ind AS/Companies Act)
  Step 3 RISK        State the manipulation mechanism or governance failure implied
  Step 4 HISTORY     Compare with known fraud patterns (Satyam, IL&FS, Enron, Wirecard…)
  Step 5 REGULATORY  State the regulatory implication (SEBI/Companies Act/CARO/LODR)
  Step 6 CONCLUSION  Confidence-weighted finding: CRITICAL/HIGH/MEDIUM/LOW + severity reason
Conclusions that skip steps are speculation, not findings.
""".strip()

HALLUCINATION_CONTROL = """
HALLUCINATION CONTROL — never invent:
  - Audit standard numbers or provisions not in this prompt
  - Financial figures not present in retrieved data or the financial data table
  - Enforcement actions, penalties, or court orders not explicitly disclosed
  - Fraud allegations without direct supporting evidence
If evidence is insufficient state exactly:
  "Available evidence insufficient for [dimension]. Recommended procedure: [specific test]."
""".strip()

EVIDENCE_PRIORITY = (
    "Evidence priority (highest → lowest): "
    "(1) Actual financial data/filings "
    "(2) Applicable audit standards "
    "(3) Accounting standards "
    "(4) Regulatory guidance "
    "(5) Historical fraud patterns "
    "(6) Secondary commentary"
)

# ─── ISA / PCAOB / COSO Standards Catalog ────────────────────────────────────

AUDIT_STANDARDS_CATALOG: dict[str, dict] = {
    "ISA_200": {
        "title": "Overall Objectives of the Independent Auditor",
        "relevance": "Professional scepticism — auditor must not accept explanations without independent evidence",
    },
    "ISA_240": {
        "title": "The Auditor's Responsibilities Relating to Fraud",
        "relevance": "Primary standard for fraud risk. §26: revenue recognition is presumed a fraud risk. §A4: management override indicators. §A50: auditor independence red flags.",
        "key_sections": {
            "§26": "Revenue recognition presumed fraud risk in every audit",
            "§A4": "Management override: guidance miss rate >40%, insider selling, compensation misalignment",
            "§A50": "Auditor signals: Big4→smaller firm switch, going concern, non-audit fees >40%",
            "§27-32": "Required investigation areas when fraud indicators present",
            "§66": "Evaluation of audit evidence — isolated vs systemic red flags",
        },
    },
    "ISA_250": {
        "title": "Consideration of Laws and Regulations",
        "relevance": "SEBI/regulatory breach indicators; illegal acts by clients (PCAOB AS 2405 equivalent)",
    },
    "ISA_315": {
        "title": "Identifying and Assessing Risks of Material Misstatement",
        "relevance": "Risk assessment procedures; entity-level controls; fraud risk factors",
    },
    "ISA_330": {
        "title": "The Auditor's Responses to Assessed Risks",
        "relevance": "Links risk assessment to specific substantive procedures; §66 evidence evaluation",
    },
    "ISA_500": {
        "title": "Audit Evidence",
        "relevance": "Sufficiency and appropriateness; weak/management-arranged evidence ≠ conclusion",
    },
    "ISA_505": {
        "title": "External Confirmations",
        "relevance": "Bank confirmations must go direct — not routed through management (Satyam failure point)",
    },
    "ISA_520": {
        "title": "Analytical Procedures",
        "relevance": "Ratio analysis, trend analysis, reasonableness testing; base for all agent scoring",
    },
    "ISA_540": {
        "title": "Auditing Accounting Estimates",
        "relevance": "Management bias in fair value, goodwill impairment, provisions, depreciation rates",
    },
    "ISA_550": {
        "title": "Related Parties",
        "relevance": "Identifying undisclosed RPTs; arm's-length testing; SEBI RPT 2021 framework",
    },
    "ISA_570": {
        "title": "Going Concern",
        "relevance": "Going concern qualification = CRITICAL forensic flag regardless of other indicators",
    },
    "ISA_705": {
        "title": "Modifications to the Opinion",
        "relevance": "Qualified / Adverse / Disclaimer opinions — each requires escalation",
    },
    "PCAOB_AS2101": {
        "title": "Audit Planning and Supervision",
        "relevance": "Auditor independence, tenure, competence assessment",
    },
    "PCAOB_AS2201": {
        "title": "Internal Control over Financial Reporting (SOX 404)",
        "relevance": "Material weakness — strongest single predictor of financial restatement",
    },
    "PCAOB_AS2401": {
        "title": "Fraud Risk Considerations",
        "relevance": "§26 revenue fraud presumption; §66 isolated vs systemic signal evaluation",
    },
    "PCAOB_Rule3521": {
        "title": "Independence — Non-Audit Fees",
        "relevance": "Non-audit fees >40% of total fees = independence threat",
    },
    "COSO_ICFR": {
        "title": "Internal Control — Integrated Framework",
        "relevance": "5 components: Control Environment (tone at top), Risk Assessment, Control Activities, Information & Communication, Monitoring",
    },
}

# ─── Indian Regulatory Provisions ────────────────────────────────────────────

INDIAN_REGULATORY_PROVISIONS: dict[str, dict] = {
    "SA_240": {
        "title": "Fraud in Financial Statements (ICAI)",
        "key_provision": "§143(12) Companies Act 2013: auditor must report fraud ≥ Rs 1 crore to Central Government",
    },
    "COMPANIES_ACT_143_12": {
        "title": "Companies Act 2013 §143(12)",
        "key_provision": "Mandatory fraud reporting to Ministry of Corporate Affairs if material fraud detected",
        "threshold": "Rs 1 crore or above",
    },
    "COMPANIES_ACT_177": {
        "title": "Audit Committee — Composition and Functions",
        "key_provision": "Mandatory for listed companies; minimum 3 directors all independent; at least 1 financially literate",
    },
    "COMPANIES_ACT_188": {
        "title": "Related Party Transactions",
        "key_provision": "Board approval required for RPTs above threshold; shareholder approval for material RPTs",
    },
    "SEBI_LODR_REG17": {
        "title": "Board of Directors Composition",
        "key_provision": "≥50% independent directors mandatory; non-compliance = structural governance failure",
    },
    "SEBI_LODR_REG18": {
        "title": "Audit Committee",
        "key_provision": "All members must be independent; at least one financially literate",
    },
    "SEBI_LODR_REG23": {
        "title": "Related Party Transactions",
        "key_provision": "Material RPTs (>lower of Rs 1,000 crore or 10% of annual consolidated turnover) require prior shareholder approval via special resolution",
    },
    "SEBI_RPT_2021": {
        "title": "SEBI RPT Regulations 2021 (Amendment to LODR)",
        "key_provision": "Expanded related-party definition; mandatory arm's-length testing; audit committee pre-approval; listed entity cannot enter RPTs that benefit promoter group to minority's detriment",
    },
    "SEBI_SAST_2011": {
        "title": "Substantial Acquisition of Shares and Takeovers",
        "key_provision": "Promoter crossing 25%/75% triggers open offer; promoter pledge disclosure mandatory",
        "pledge_thresholds": ">30% elevated | >50% high | >70% critical (collateral sale risk)",
    },
    "SEBI_INSIDER_TRADING_2015": {
        "title": "Prohibition of Insider Trading",
        "key_provision": "Trading window closure during UPSI; connected persons definition; disclosure of trades >Rs 10 lakh",
    },
    "CARO_2020": {
        "title": "Companies (Auditor's Report) Order 2020",
        "key_clauses": {
            "3(ii)(b)": "Inventory: physical verification frequency and material discrepancies",
            "3(ix)(a)": "Default on loans from banks/financial institutions — details required",
            "3(ix)(b)": "Willful defaulter status under RBI guidelines",
            "3(xv)": "Term loans utilisation — end-use verification",
            "3(xvi)": "Short-term funds used for long-term purposes (fund diversion signal)",
            "3(xix)": "Cash losses in current and immediately preceding year",
            "3(xx)": "Whether auditor reported fraud under §143(12) of Companies Act",
        },
    },
    "IND_AS_115": {
        "title": "Revenue from Contracts with Customers",
        "key_provision": "5-step recognition model; percentage-of-completion revenue — verify stage against independent progress certificate",
    },
    "IND_AS_36": {
        "title": "Impairment of Assets",
        "key_provision": "Annual impairment test for goodwill; recoverable amount must be supported by cash flow projections",
    },
    "RBI_PCA": {
        "title": "RBI Prompt Corrective Action Framework",
        "key_provision": "Banks with CRAR <10.25% or NPA >6% or ROA <0 placed under PCA; restricts lending and dividends",
    },
}

# ─── Fraud Detection Indicator Catalog ───────────────────────────────────────

FRAUD_INDICATORS_CATALOG: dict[str, list[dict]] = {
    "revenue_manipulation": [
        {"indicator": "Revenue-to-OCF divergence sustained >2 years", "standard": "ISA 240 §26", "threshold": "OCF/Revenue ratio declining while revenue grows >10% YoY"},
        {"indicator": "DSO expanding >15% YoY", "standard": "ISA 240 / Ind AS 115", "threshold": ">15 days increase annually"},
        {"indicator": "Q4 revenue spike reversing in Q1", "standard": "ISA 240 §A26", "threshold": ">20% of annual revenue in Q4"},
        {"indicator": "Related-party sales growing faster than third-party", "standard": "ISA 550 / SEBI RPT 2021", "threshold": "RPT revenue >10% of total"},
        {"indicator": "Percentage of completion above physical progress", "standard": "Ind AS 115", "threshold": "Stage >actual physical progress by independent engineer"},
    ],
    "expense_manipulation": [
        {"indicator": "Capex/revenue ratio rising without capacity growth", "standard": "ISA 540 / IAS 16", "threshold": "Capex CAGR > Revenue CAGR + 10%"},
        {"indicator": "R&D capitalisation rising as % of revenue", "standard": "IAS 38 / Ind AS 38", "threshold": "Capitalised R&D >50% of total R&D spend"},
        {"indicator": "Depreciation rate declining vs. asset base growth", "standard": "ISA 540", "threshold": "DEPI >1.083 (Beneish threshold)"},
        {"indicator": "Provisions reversing regularly (cookie-jar reversals)", "standard": "ISA 540", "threshold": "2+ consecutive years of material provision write-back"},
        {"indicator": "Goodwill not impaired despite business decline", "standard": "IAS 36 / Ind AS 36", "threshold": "Static goodwill with falling segment revenue"},
    ],
    "cash_flow_manipulation": [
        {"indicator": "Operating cash inflows classified as investing", "standard": "IAS 7 / Ind AS 7", "threshold": "Unusual investing inflows from operating-type sources"},
        {"indicator": "Bill discounting reclassified to reduce payables", "standard": "Ind AS 109", "threshold": "Sudden payable reduction without revenue decline"},
        {"indicator": "Financing CF masks operating weakness", "standard": "PCAOB AS 2401", "threshold": "Financing inflows > OCF for 2+ years"},
    ],
    "related_party_abuse": [
        {"indicator": "Loans to promoter entities below market rate", "standard": "ISA 550 / SEBI RPT 2021", "threshold": "Balance >10% net worth; rate below RBI repo"},
        {"indicator": "Sales to SPVs at inflated price", "standard": "SA 550 / Ind AS 24", "threshold": "Price > comparable third-party transactions"},
        {"indicator": "Circular revenue between related entities", "standard": "SEBI RPT 2021 / OECD VI.C", "threshold": "Revenue loop through 2+ related entities"},
        {"indicator": "Guarantees for promoter group debt", "standard": "Ind AS 24", "threshold": "Contingent liability >5% of net worth"},
    ],
    "governance_failure": [
        {"indicator": "Board independence below 50%", "standard": "SEBI LODR Reg 17", "threshold": "<50% independent directors"},
        {"indicator": "≥2 independent director exits in 12 months", "standard": "COSO Control Environment", "threshold": "2+ resignations in rolling 12 months"},
        {"indicator": "CFO change within investigation period", "standard": "ISA 240 §A4 / Dechow (2010)", "threshold": "Any change — correlated with earnings manipulation"},
        {"indicator": "Promoter pledge >50%", "standard": "SEBI SAST 2011", "threshold": ">50% = high; >70% = critical with collateral-sale risk"},
    ],
}

# ─── Audit Procedure Templates ────────────────────────────────────────────────

AUDIT_PROCEDURE_TEMPLATES: dict[str, dict] = {
    "revenue_inflation": {
        "standards": ["ISA 240", "SA 240", "ISA 500", "ISA 505", "Ind AS 115"],
        "procedures": [
            "Customer confirmation letters — direct send by auditor, NOT routed through management (ISA 505)",
            "Cut-off testing: examine sales 2 weeks pre/post year-end against delivery notes and dispatch records",
            "Contract review: for long-term contracts, verify % completion against independent engineer certificate",
            "Journal entry testing: search for round-number entries, period-end entries, entries made by senior management",
            "Subsequent receipts testing: verify year-end receivables collected within 90 days post year-end",
        ],
        "required_evidence": ["Invoices", "Signed contracts", "Delivery notes", "Bank receipts", "Customer confirmations"],
    },
    "asset_inflation": {
        "standards": ["ISA 540", "IAS 16", "IAS 36", "IAS 38", "Ind AS 16", "Ind AS 36"],
        "procedures": [
            "Physical verification of major capital additions — do not rely on management-prepared list",
            "Review nature of capitalised items: recurring, maintenance-type, or labour-intensive items = opex masquerading as capex",
            "Vouch depreciation rates against asset register and prior year — flag any rate changes without stated rationale",
            "Test impairment: compare recoverable amount (higher of VIU or FVLCD) vs. carrying value per CGU",
            "Trace capital work-in-progress (CWIP) ageing — prolonged CWIP with no completion may indicate fictitious assets",
        ],
        "required_evidence": ["CWIP schedules", "Purchase orders", "Site inspection notes", "Impairment model with assumptions"],
    },
    "related_party_fraud": {
        "standards": ["ISA 550", "SA 550", "SEBI RPT 2021", "Ind AS 24", "Companies Act §188"],
        "procedures": [
            "Obtain full related-party list from management + independently verify via MCA21 (director DINs, common shareholders)",
            "Examine all RPT transactions for arm's-length pricing — obtain comparable third-party transactions",
            "Review board minutes and audit committee minutes for RPT approval resolutions (SEBI LODR Reg 23)",
            "Trace fund flows: loans given to RPTs — are they being repaid or evergreened at maturity?",
            "Map holding structure using MCA21 to build corporate tree and identify undisclosed entities",
            "Search for circular transactions: revenue flows A → RPT entity → back to A inflating both entities' P&L",
        ],
        "required_evidence": ["Board/AC minutes", "RPT register", "Fund transfer confirmations", "Comparable arm's-length transactions"],
    },
    "going_concern": {
        "standards": ["ISA 570", "SA 570", "IAS 1", "Ind AS 1"],
        "procedures": [
            "Review cash flow projections for next 12 months — stress-test key assumptions (revenue -20%, cost +10%)",
            "Obtain bank facility confirmation letters directly from banks (NOT management-arranged)",
            "Review all debt agreements for covenant compliance — obtain waivers documentation where breached",
            "Examine post-balance-sheet events: default notices, rating downgrades, bank facility cancellations",
            "Assess management's mitigation plans: distinguish committed actions from aspirational plans",
        ],
        "required_evidence": ["Cash flow model with assumptions", "Direct bank confirmation letters", "Facility agreements", "Post-period board minutes"],
    },
    "management_override": {
        "standards": ["ISA 240 §A4", "PCAOB AS 2401 §66", "SA 240"],
        "procedures": [
            "Journal entry testing across full year — filter for period-end, round-number, management-initiated entries",
            "Review accounting estimates for consistency — year-over-year changes in estimates without stated rationale = red flag",
            "Reconcile management representations against documentary evidence independently",
            "Analyse management compensation: high bonus tied to earnings metrics creates manipulation incentive",
            "Interview audit committee members separately from management",
        ],
        "required_evidence": ["Journal entry population with SAP/ERP export", "Estimate revision memos", "Compensation structure documents"],
    },
}

# ─── Agent-to-Standards Mapping ──────────────────────────────────────────────

AGENT_STANDARDS_MAP: dict[int, list[str]] = {
    3: ["ISA_240", "PCAOB_AS2401"],                         # Revenue Forensics
    4: ["ISA_240", "PCAOB_AS2401", "ISA_520"],              # Cash Flow
    5: ["ISA_240", "ISA_520"],                              # Working Capital
    6: ["ISA_240", "PCAOB_AS2401", "ISA_315", "ISA_520"],  # Fraud Detection
    7: ["ISA_570", "PCAOB_AS2101"],                         # Credit Risk
    8: ["ISA_240", "ISA_520", "PCAOB_AS2401"],              # Earnings Quality
    9: ["ISA_550", "ISA_240"],                              # Related Party
    10: ["ISA_240", "PCAOB_AS2101", "PCAOB_Rule3521", "ISA_570", "ISA_705"],  # Auditor
    11: ["ISA_240", "ISA_500"],                             # Management NLP
    12: ["ISA_520"],                                        # Peer Comparison
    17: ["ISA_240", "ISA_330", "PCAOB_AS2401", "COSO_ICFR"],  # Director
}

# ─── Agent-to-Historical-Cases Mapping ───────────────────────────────────────

AGENT_CASE_MAP: dict[int, list[str]] = {
    3:  ["satyam", "luckin_coffee", "worldcom"],    # Revenue Forensics
    4:  ["worldcom", "enron", "carillion"],         # Cash Flow
    5:  ["satyam", "dhfl", "luckin_coffee"],        # Working Capital
    6:  ["satyam", "wirecard", "enron", "luckin_coffee"],  # Fraud Detection
    7:  ["ilfs", "yes_bank", "carillion"],          # Credit Risk
    8:  ["enron", "worldcom", "carillion"],         # Earnings Quality
    9:  ["ilfs", "dhfl", "steinhoff", "satyam"],   # Related Party
    10: ["satyam", "wirecard", "carillion"],        # Auditor Intelligence
    11: ["enron", "luckin_coffee", "satyam"],       # Management NLP
    12: ["luckin_coffee", "wirecard"],              # Peer Comparison
    17: ["satyam", "wirecard", "enron", "ilfs"],   # Director synthesis
}

# ─── Historical Fraud Case Database (extended) ────────────────────────────────

FRAUD_CASE_LIBRARY: dict[str, dict] = {
    "satyam": {
        "company": "Satyam Computer Services", "country": "India", "year": 2009,
        "fraud_type": "Fictitious Cash & Debtors",
        "mechanism": "Rs 5,040 crore cash fabricated; bank certificates forged; fictitious fixed deposits; inflated headcount",
        "accounting_technique": "Assets (cash) and revenue overstated simultaneously via forged bank statements",
        "audit_failure": "PwC accepted management-prepared bank statements; did not obtain direct bank confirmation per ISA 505",
        "detection_indicators": [
            "Cash/bank balance >> reported interest income (cash not earning market rate)",
            "Revenue growth far exceeded cash inflows — persistent OCF/revenue divergence",
            "Analyst discovered Rs 1,000 crore acquisition overpayment before confession",
        ],
        "regulatory_response": "SEBI debarment of auditors; MCA appointed new board; criminal prosecution of promoter",
        "pattern_watch": "High reported cash + low interest income; auditor reliance on management-arranged confirmations",
    },
    "ilfs": {
        "company": "Infrastructure Leasing & Financial Services", "country": "India", "year": 2018,
        "fraud_type": "Round-Tripping, Leverage Concealment",
        "mechanism": "Loans round-tripped between 250+ group entities to show repayment capacity; consolidated leverage hidden behind SPV structure",
        "accounting_technique": "Intra-group loans recorded as performing; SPVs kept off-consolidated balance sheet",
        "audit_failure": "Group-level leverage not assessed; rating agencies relied on holdco-only financials",
        "detection_indicators": [
            "Debt/equity at holdco vastly lower than economic reality",
            "Loan repayments circular: entity A repaid using funds borrowed from entity B",
            "Audited holdco appeared clean; all stress sat in subsidiary SPVs",
            "Rating cliff: AAA → junk within weeks",
        ],
        "regulatory_response": "NCLAT appointed new board; SFIO investigation; ICAI disciplinary action against auditors",
        "pattern_watch": "Complex holding structures with many SPVs; intra-group treasury operations; sudden rating cliff",
    },
    "dhfl": {
        "company": "Dewan Housing Finance Corporation", "country": "India", "year": 2019,
        "fraud_type": "Shell Company Diversion, Fictitious Loans",
        "mechanism": "Rs 31,000 crore diverted via shell companies; fictitious housing loans to non-existent borrowers in rural areas",
        "accounting_technique": "Loan book appeared healthy; fictitious borrowers scattered geographically to impede field verification",
        "audit_failure": "Loan book not independently sampled by geography; related shell entities not identified",
        "detection_indicators": [
            "Geographic concentration of loans in areas with no business presence",
            "Coralline Capital report identified shell company structure before collapse",
            "Spread between borrowing and lending rate compressed despite stated growth",
        ],
        "regulatory_response": "SEBI debarment; NCLT resolution proceedings; criminal prosecution",
        "pattern_watch": "NBFC with rapid loan book growth + weak field verification + high promoter borrowings from company",
    },
    "yes_bank": {
        "company": "Yes Bank", "country": "India", "year": 2020,
        "fraud_type": "NPA Suppression, Quid-Pro-Quo Lending",
        "mechanism": "Loans to distressed entities in exchange for kickbacks; NPA recognition suppressed; evergreening at maturity",
        "accounting_technique": "Loans restructured to avoid NPA classification; RBI divergence report revealed actual GNPA 100%+ above reported",
        "detection_indicators": [
            "RBI divergence report: actual NPA significantly higher than disclosed",
            "Stressed sector concentration: IL&FS, DHFL, Cox & Kings exposures",
            "Promoter salary vs bank performance mismatch",
            "Provision coverage ratio declining while NPAs worsened",
        ],
        "regulatory_response": "RBI-imposed moratorium; rescue by SBI-led consortium; criminal proceedings against MD",
        "pattern_watch": "Bank with divergence between reported and RBI-assessed NPAs; rapid growth in stressed sectors",
    },
    "enron": {
        "company": "Enron Corporation", "country": "USA", "year": 2001,
        "fraud_type": "SPE Accounting, Mark-to-Market Inflation",
        "mechanism": "SPVs used to move losses off-balance-sheet; energy contracts marked to inflated model values; CFO personally invested in SPVs",
        "accounting_technique": "GAAP technically complied with; economic substance vs legal form gap exploited. SPVs not consolidated despite effective control",
        "audit_failure": "Arthur Andersen — document shredding; non-audit fees exceeded audit fees (independence compromise)",
        "detection_indicators": [
            "EBITDA growing but FCF consistently negative across 3+ years",
            "Complex legal structures with no apparent business rationale",
            "CFO personally invested in SPVs = undisclosed conflict of interest",
            "Energy trading revenues grew implausibly fast vs peers",
        ],
        "regulatory_response": "Criminal conviction of executives; dissolution of Arthur Andersen; SOX legislation enacted",
        "pattern_watch": "Complex SPV structure; non-audit fees > audit fees; EBITDA/FCF divergence",
    },
    "worldcom": {
        "company": "WorldCom Inc", "country": "USA", "year": 2002,
        "fraud_type": "Opex Capitalisation",
        "mechanism": "$3.8 billion operating expenses (network line costs) capitalised as capital expenditure, boosting EBITDA and understating expenses",
        "accounting_technique": "Line costs reclassified from operating expense to capex. Internal audit team discovered scheme",
        "audit_failure": "Arthur Andersen — audit procedures did not test nature of capitalised items independently",
        "detection_indicators": [
            "Capex growing faster than revenue with no network capacity expansion explanation",
            "OCF/EBITDA ratio very low: expenses secretly capitalised don't flow through OCF",
            "Asset turnover declining despite flat capex-to-revenue narrative",
        ],
        "regulatory_response": "Largest US bankruptcy filing at time; criminal convictions; $11 billion SEC civil penalty",
        "pattern_watch": "Rising capex + flat revenue + stable reported EBITDA = possible expense capitalisation (Worldcom pattern)",
    },
    "wirecard": {
        "company": "Wirecard AG", "country": "Germany", "year": 2020,
        "fraud_type": "Phantom Cash, Fictitious Revenue",
        "mechanism": "€1.9 billion in trustee accounts that did not exist; Asia-Pacific acquiring business entirely fabricated",
        "audit_failure": "EY (10-year tenure) never obtained direct bank confirmation; management-arranged statements accepted",
        "detection_indicators": [
            "Cash held in 'trustee accounts' never directly confirmed by auditor",
            "Asia revenues grew faster than any comparable regional competitor",
            "FT journalists and internal whistleblowers raised concerns 2 years before collapse",
            "Short-seller Zatarra/Muddy Waters report dismissed without independent investigation",
        ],
        "regulatory_response": "BaFin reformed; EY under criminal investigation; CEO arrested",
        "pattern_watch": "Cash held offshore or by third-party trustees; auditor reliance on management-arranged bank statements; rapid growth in low-transparency jurisdictions",
    },
    "luckin_coffee": {
        "company": "Luckin Coffee", "country": "China/USA", "year": 2020,
        "fraud_type": "Fabricated Revenue",
        "mechanism": "RMB 2.2 billion revenue fabricated; inflated expenses to mask fraud; coordinated across multiple entity levels",
        "detection_indicators": [
            "Revenue per store implausibly high vs foot-traffic data (alternative data)",
            "Muddy Waters short report used store visits and coupon tracking to disprove unit economics",
            "Anonymous whistleblower package sent to SEC with detailed evidence",
            "Management sold shares at peak valuation preceding disclosure",
        ],
        "regulatory_response": "SEC enforcement; Nasdaq delisting; Chinese regulatory action",
        "pattern_watch": "High-growth consumer brand with unverifiable unit economics; VIE structure; management selling post-peak",
    },
    "carillion": {
        "company": "Carillion PLC", "country": "UK", "year": 2018,
        "fraud_type": "Revenue Recognition, Pension Understatement",
        "mechanism": "Aggressive revenue recognition on construction contracts; pension deficit understated; goodwill impairment deferred; dividend maintained while cash was collapsing",
        "audit_failure": "KPMG (19-year tenure) — non-audit fees concern; aggressive estimates not challenged",
        "detection_indicators": [
            "Revenue recognised ahead of cash receipts (high average debtor days)",
            "Debt rising while dividend was maintained — cash flow inconsistency",
            "Goodwill never impaired despite contract losses in same divisions",
            "5 profit warnings in 9 months before collapse",
        ],
        "regulatory_response": "KPMG fined; FRC investigation; Parliamentary inquiry into Big 4",
        "pattern_watch": "Dividend maintained with rising debt; goodwill static despite division losses; auditor long tenure with non-audit fee concentration",
    },
    "steinhoff": {
        "company": "Steinhoff International", "country": "South Africa/Germany", "year": 2017,
        "fraud_type": "Fictitious Transactions, Revenue Overstatement",
        "mechanism": "Fictitious revenue from related-party deals across multiple jurisdictions; complex multi-layered holding structure to obscure fraud",
        "detection_indicators": [
            "Revenue from transactions that lacked economic substance",
            "Frequent auditor changes and late filing of accounts",
            "CEO credibility — prior investigations in Germany",
            "Governance failures: audit committee did not challenge complex transactions",
        ],
        "regulatory_response": "South African FSCA investigation; criminal proceedings against CEO; KPMG resigned",
        "pattern_watch": "Multi-jurisdictional holding structure; revenue from non-arm's-length deals; frequent auditor changes",
    },
}

# ─── Public Helper Functions ──────────────────────────────────────────────────

def get_reasoning_preamble() -> str:
    """Returns a compact reasoning + hallucination control block for prompt injection."""
    return f"{REASONING_FRAMEWORK}\n\n{HALLUCINATION_CONTROL}\n\n{EVIDENCE_PRIORITY}"


def get_agent_knowledge_block(agent_id: int) -> str:
    """
    Build a knowledge context block for a specific agent.
    Returns relevant standards and historical case patterns formatted
    for injection into the agent's LLM prompt.
    Kept concise (<400 words) to avoid bloating context windows.
    """
    lines: list[str] = []

    # Standards relevant to this agent
    standard_keys = AGENT_STANDARDS_MAP.get(agent_id, [])
    if standard_keys:
        lines.append("APPLICABLE STANDARDS:")
        for key in standard_keys:
            s = AUDIT_STANDARDS_CATALOG.get(key, {})
            if s:
                lines.append(f"  {key}: {s.get('title', '')} — {s.get('relevance', '')[:120]}")

    # Indian-specific provisions for governance/RPT/auditor agents
    if agent_id in (9, 10, 11):
        lines.append("\nINDIAN REGULATORY PROVISIONS:")
        india_keys = {
            9:  ["SEBI_RPT_2021", "SEBI_LODR_REG23", "COMPANIES_ACT_188", "CARO_2020"],
            10: ["SA_240", "COMPANIES_ACT_143_12", "CARO_2020"],
            11: ["SEBI_INSIDER_TRADING_2015", "SEBI_SAST_2011"],
        }.get(agent_id, [])
        for key in india_keys:
            p = INDIAN_REGULATORY_PROVISIONS.get(key, {})
            if p:
                lines.append(f"  {key}: {p.get('title', '')} — {p.get('key_provision', '')[:120]}")

    # Historical case patterns relevant to this agent
    case_keys = AGENT_CASE_MAP.get(agent_id, [])
    if case_keys:
        lines.append("\nRELEVANT FRAUD PATTERNS (for case-based comparison):")
        for key in case_keys[:3]:  # limit to 3 to control token count
            c = FRAUD_CASE_LIBRARY.get(key, {})
            if c:
                indicators = " | ".join(c.get("detection_indicators", [])[:2])
                lines.append(
                    f"  {c.get('company', key)} ({c.get('year', '')}): "
                    f"{c.get('fraud_type', '')} — Watch: {indicators[:150]}"
                )

    return "\n".join(lines)


def get_audit_procedures(risk_type: str) -> str:
    """
    Returns formatted audit procedure recommendations for a given risk type.
    risk_type: one of 'revenue_inflation', 'asset_inflation', 'related_party_fraud',
                       'going_concern', 'management_override'
    """
    template = AUDIT_PROCEDURE_TEMPLATES.get(risk_type)
    if not template:
        return f"No procedure template for risk type: {risk_type}"

    stds = ", ".join(template.get("standards", []))
    procs = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(template.get("procedures", [])))
    evidence = ", ".join(template.get("required_evidence", []))
    return (
        f"RECOMMENDED AUDIT PROCEDURES — {risk_type.replace('_', ' ').upper()}\n"
        f"Standards: {stds}\n"
        f"Procedures:\n{procs}\n"
        f"Required evidence: {evidence}"
    )
