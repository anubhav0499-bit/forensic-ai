"""
Agent 6 - Fraud Detection Agent
================================
Calculates: Beneish M-Score, Dechow F-Score, Piotroski F-Score, Accrual Analysis
Compares with known fraud case signatures.
"""

from __future__ import annotations
from .base_agent import BaseForensicAgent, AgentResult, AgentFinding
from forensics.beneish_score import BeneishMScore, BeneishInputs
from forensics.altman_score import AltmanZScore, AltmanInputs
from forensics.piotroski_score import PiotroskiFScore, PiotroskiInputs
from forensics.dechow_score import DechowFScore
from forensics.accrual_analysis import AccrualAnalyzer
from forensics.risk_scorer import RiskScorer
from config import FRAUD_CASE_DATABASE
from utils.helpers import safe_divide


class FraudDetectionAgent(BaseForensicAgent):
    """
    Quantitative fraud detection using multiple academic models.
    Every score is calculated with full formula tracing.
    """

    def investigate(self, company_name: str, company_id: int, financial_data: dict, **kwargs) -> AgentResult:
        self.log_info(f"Starting fraud detection analysis for {company_name}")
        result = AgentResult(agent_id=self.agent_id, agent_name=self.agent_name)

        if not financial_data:
            result.status = "FAILED"
            result.error = "No financial data available for forensic scoring"
            return result

        years = sorted(financial_data.keys(), reverse=True)
        if len(years) < 2:
            result.status = "PARTIAL"
            result.error = "Need at least 2 years of data for M-Score calculation"

        # ── Run All Forensic Models ────────────────────────────
        all_scores = {}

        for i, year in enumerate(years[:3]):  # Last 3 years
            year_data = financial_data[year]
            prev_year = years[i + 1] if i + 1 < len(years) else None
            prev_data = financial_data.get(prev_year, {}) if prev_year else {}

            scores = self._run_all_models(year, year_data, prev_data)
            all_scores[year] = scores

            # Save to database
            self.db.save_forensic_scores(company_id, year, {
                k: v for k, v in scores.items()
                if not isinstance(v, dict)
            })

        # ── Generate Findings ──────────────────────────────────
        latest_year = years[0]
        latest_scores = all_scores.get(latest_year, {})

        findings = []

        # Beneish M-Score finding
        m_score = latest_scores.get("beneish_m_score")
        if m_score is not None:
            manipulation_likely = m_score > -1.78
            finding = self._create_finding(
                finding_type="RED_FLAG" if manipulation_likely else "OBSERVATION",
                title=f"Beneish M-Score: {m_score:.3f} - {'⚠️ MANIPULATION LIKELY' if manipulation_likely else '✅ Below Threshold'}",
                detail=latest_scores.get("beneish_interpretation", ""),
                evidence=f"M-Score = {m_score:.4f} (threshold: -1.78). Components: DSRI={latest_scores.get('beneish_dsri', 0):.3f}, GMI={latest_scores.get('beneish_gmi', 0):.3f}, AQI={latest_scores.get('beneish_aqi', 0):.3f}, TATA={latest_scores.get('beneish_tata', 0):.3f}",
                fiscal_year=latest_year,
                risk_level="CRITICAL" if m_score > -1.0 else ("HIGH" if manipulation_likely else "LOW"),
                confidence=0.85,
                calculation=f"M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI = {m_score:.4f}",
            )
            findings.append(finding)
            if manipulation_likely:
                result.red_flags.append(finding)

        # Dechow F-Score finding
        d_score = latest_scores.get("dechow_f_score")
        if d_score is not None:
            if d_score > 0.025:
                finding = self._create_finding(
                    finding_type="RED_FLAG",
                    title=f"Dechow F-Score: {d_score:.4f} - ELEVATED Misreporting Probability",
                    detail=latest_scores.get("dechow_interpretation", ""),
                    evidence=f"F-Score = {d_score:.4f} (base rate ~0.01). High soft assets, rising receivables, or external financing detected.",
                    fiscal_year=latest_year,
                    risk_level="HIGH" if d_score > 0.025 else "MODERATE",
                    confidence=0.80,
                )
                findings.append(finding)
                result.red_flags.append(finding)

        # Piotroski F-Score finding
        piotroski = latest_scores.get("piotroski_f_score")
        if piotroski is not None:
            if piotroski <= 2:
                finding = self._create_finding(
                    finding_type="RED_FLAG",
                    title=f"Piotroski F-Score: {piotroski}/9 - WEAK Financial Strength",
                    detail="Low Piotroski score indicates deteriorating financial fundamentals across profitability, leverage, and efficiency.",
                    evidence=f"F-Score = {piotroski}/9. Failing: {latest_scores.get('piotroski_failing', [])}",
                    fiscal_year=latest_year,
                    risk_level="HIGH",
                    confidence=0.80,
                )
                findings.append(finding)
                result.red_flags.append(finding)
            elif piotroski >= 7:
                finding = self._create_finding(
                    finding_type="GREEN_FLAG",
                    title=f"Piotroski F-Score: {piotroski}/9 - STRONG Financial Position",
                    detail="High Piotroski score indicates strong financial fundamentals.",
                    evidence=f"F-Score = {piotroski}/9. Passing all major profitability, leverage, and efficiency criteria.",
                    fiscal_year=latest_year,
                    risk_level="POSITIVE",
                    confidence=0.80,
                )
                findings.append(finding)
                result.green_flags.append(finding)

        # Altman Z-Score finding
        z_score = latest_scores.get("altman_z_score")
        if z_score is not None:
            zone = latest_scores.get("altman_zone", "")
            if zone == "DISTRESS":
                finding = self._create_finding(
                    finding_type="RED_FLAG",
                    title=f"Altman Z-Score: {z_score:.3f} - DISTRESS ZONE",
                    detail=latest_scores.get("altman_interpretation", ""),
                    evidence=f"Z-Score = {z_score:.4f} (< 1.81 distress zone). Implied credit rating: {latest_scores.get('implied_credit_rating', 'N/A')}",
                    fiscal_year=latest_year,
                    risk_level="CRITICAL" if z_score < 1.0 else "HIGH",
                    confidence=0.85,
                    calculation="Z = 1.2*WC/TA + 1.4*RE/TA + 3.3*EBIT/TA + 0.6*MktCap/TL + 1.0*Sales/TA",
                )
                findings.append(finding)
                result.red_flags.append(finding)

        # Multi-year trend in M-Score
        m_trend = [(y, v["beneish_m_score"]) for y, v in all_scores.items() if "beneish_m_score" in v]
        if len(m_trend) >= 2:
            m_trend_sorted = sorted(m_trend, key=lambda x: x[0])
            if m_trend_sorted[-1][1] > m_trend_sorted[0][1]:
                finding = self._create_finding(
                    finding_type="RED_FLAG",
                    title="M-Score Trending Up - Increasing Manipulation Risk Over Time",
                    detail=f"Beneish M-Score has been rising over the investigation period. Trend: {', '.join([f'{y}:{s:.2f}' for y, s in m_trend_sorted])}",
                    evidence="Multi-year M-Score trend indicates worsening earnings quality.",
                    risk_level="HIGH",
                    confidence=0.75,
                )
                findings.append(finding)
                result.red_flags.append(finding)

        # ── Historical Fraud Pattern Comparison ───────────────
        fraud_similarity = self._compare_with_known_frauds(latest_scores)
        if fraud_similarity:
            for case_name, similarity in fraud_similarity.items():
                if similarity > 0.6:
                    case = FRAUD_CASE_DATABASE[case_name]
                    finding = self._create_finding(
                        finding_type="RED_FLAG",
                        title=f"Financial Profile Similar to {case['company']} ({case['year']})",
                        detail=(
                            f"The company's fraud indicator profile shows {similarity*100:.0f}% similarity "
                            f"to {case['company']}, a confirmed {case['fraud_type']} case. "
                            f"Key matching signals: {', '.join(case['key_signals'][:3])}"
                        ),
                        evidence=f"Similarity score: {similarity:.2f}. Fraud type: {case['fraud_type']}",
                        risk_level="CRITICAL" if similarity > 0.75 else "HIGH",
                        confidence=similarity * 0.7,  # Scale confidence by similarity
                    )
                    findings.append(finding)
                    result.red_flags.append(finding)

        # ── LLM Deep Analysis ─────────────────────────────────
        context = self._retrieve_context(company_name, "fraud manipulation accounting irregularities restatement")
        m_score_val = latest_scores.get("beneish_m_score", -2.5)
        z_score_val = latest_scores.get("altman_z_score", 3.0)
        p_score_val = latest_scores.get("piotroski_f_score", 5)
        accrual_val = latest_scores.get("accrual_ratio", 0.0)

        from llm.prompts import build_fraud_detection_prompt
        fraud_prompt = build_fraud_detection_prompt(company_name, m_score_val, z_score_val, p_score_val, accrual_val)
        if context:
            fraud_prompt += f"\n\nDOCUMENT EVIDENCE:\n{context}"

        raw_analysis = self._analyze_with_llm(fraud_prompt, "fraud_investigator")
        result.raw_analysis = raw_analysis

        # ── Overall Fraud Risk Score ───────────────────────────
        result.risk_score = self._calculate_fraud_risk_score(latest_scores)
        result.findings = findings

        result.summary = (
            f"FRAUD DETECTION SUMMARY - {company_name}\n"
            f"{'='*50}\n"
            f"Beneish M-Score: {m_score_val:.3f} ({'⚠️ MANIPULATOR' if m_score_val > -1.78 else '✅ Clean'})\n"
            f"Altman Z-Score: {z_score_val:.3f} ({latest_scores.get('altman_zone', 'N/A')})\n"
            f"Piotroski F-Score: {p_score_val}/9\n"
            f"Dechow F-Score: {latest_scores.get('dechow_f_score', 0):.4f}\n"
            f"Accrual Ratio: {accrual_val:.3f}\n"
            f"Fraud Risk Score: {result.risk_score:.1f}/100\n"
            f"Red Flags: {len(result.red_flags)} | Green Flags: {len(result.green_flags)}"
        )

        self._save_output(result, company_name)
        self.log_info(f"Fraud detection complete. Risk score: {result.risk_score:.1f}/100")
        return result

    def _run_all_models(self, year: str, data: dict, prev_data: dict) -> dict:
        """Run all forensic models for a given year."""
        scores = {}

        # ── Beneish M-Score ────────────────────────────────────
        try:
            inp = BeneishInputs(
                net_receivables_t=data.get("accounts_receivable", 0),
                sales_t=data.get("revenue", 0),
                cogs_t=data.get("cogs", data.get("revenue", 0) * 0.6),
                current_assets_t=data.get("current_assets", 0),
                ppe_t=data.get("ppe_net", 0),
                total_assets_t=data.get("total_assets", 1),
                depreciation_t=data.get("depreciation", 0),
                sga_t=data.get("sga", data.get("revenue", 0) * 0.15),
                total_debt_t=data.get("total_debt", 0),
                current_liabilities_t=data.get("current_liabilities", 0),
                working_capital_t=data.get("current_assets", 0) - data.get("current_liabilities", 0),
                cash_t=data.get("cash_equivalents", 0),
                taxes_payable_t=data.get("tax", 0),
                net_receivables_tm1=prev_data.get("accounts_receivable", data.get("accounts_receivable", 0) * 0.9),
                sales_tm1=prev_data.get("revenue", data.get("revenue", 0) * 0.9),
                cogs_tm1=prev_data.get("cogs", data.get("cogs", data.get("revenue", 0) * 0.6) * 0.9),
                current_assets_tm1=prev_data.get("current_assets", data.get("current_assets", 0) * 0.9),
                ppe_tm1=prev_data.get("ppe_net", data.get("ppe_net", 0) * 0.9),
                total_assets_tm1=prev_data.get("total_assets", data.get("total_assets", 1) * 0.9),
                depreciation_tm1=prev_data.get("depreciation", data.get("depreciation", 0)),
                sga_tm1=prev_data.get("sga", data.get("sga", 0) * 0.9),
                total_debt_tm1=prev_data.get("total_debt", data.get("total_debt", 0)),
                current_liabilities_tm1=prev_data.get("current_liabilities", data.get("current_liabilities", 0) * 0.9),
            )
            beneish = BeneishMScore().calculate(inp)
            scores.update({
                "beneish_m_score": beneish.m_score,
                "beneish_dsri": beneish.dsri,
                "beneish_gmi": beneish.gmi,
                "beneish_aqi": beneish.aqi,
                "beneish_sgi": beneish.sgi,
                "beneish_depi": beneish.depi,
                "beneish_sgai": beneish.sgai,
                "beneish_lvgi": beneish.lvgi,
                "beneish_tata": beneish.tata,
                "beneish_manipulation_flag": beneish.manipulation_likely,
                "beneish_interpretation": beneish.interpretation,
                "beneish_risk_level": beneish.risk_level,
            })
        except Exception as e:
            self.log_warning(f"Beneish calculation failed: {e}")

        # ── Altman Z-Score ─────────────────────────────────────
        try:
            wc = data.get("current_assets", 0) - data.get("current_liabilities", 0)
            inp_z = AltmanInputs(
                working_capital=wc,
                total_assets=data.get("total_assets", 1),
                retained_earnings=data.get("retained_earnings", 0),
                ebit=data.get("ebit", data.get("net_income", 0) * 1.3),
                market_cap=data.get("market_cap", 0),
                book_equity=data.get("shareholder_equity", 0),
                total_liabilities=data.get("total_liabilities", 0),
                revenue=data.get("revenue", 0),
                is_manufacturing=True,
                is_public=True,
            )
            altman = AltmanZScore().calculate(inp_z)
            scores.update({
                "altman_z_score": altman.z_score,
                "altman_x1": altman.x1,
                "altman_x2": altman.x2,
                "altman_x3": altman.x3,
                "altman_x4": altman.x4,
                "altman_x5": altman.x5,
                "altman_zone": altman.zone,
                "altman_interpretation": altman.interpretation,
                "implied_credit_rating": AltmanZScore().get_implied_credit_rating(altman.z_score),
            })
        except Exception as e:
            self.log_warning(f"Altman calculation failed: {e}")

        # ── Piotroski F-Score ──────────────────────────────────
        try:
            avg_assets = (data.get("total_assets", 1) + prev_data.get("total_assets", data.get("total_assets", 1))) / 2
            roa = safe_divide(data.get("net_income", 0), avg_assets)
            roa_prev = safe_divide(prev_data.get("net_income", 0), prev_data.get("total_assets", avg_assets))

            inp_p = PiotroskiInputs(
                net_income=data.get("net_income", 0),
                total_assets=data.get("total_assets", 1),
                cfo=data.get("cfo", 0),
                roa=roa,
                long_term_debt=data.get("long_term_debt", 0),
                current_ratio=safe_divide(data.get("current_assets", 0), data.get("current_liabilities", 1)),
                shares_outstanding=data.get("shares_outstanding", 1000),
                gross_profit=data.get("gross_profit", 0),
                revenue=data.get("revenue", 0),
                net_income_prev=prev_data.get("net_income", 0),
                total_assets_prev=prev_data.get("total_assets", data.get("total_assets", 1)),
                roa_prev=roa_prev,
                long_term_debt_prev=prev_data.get("long_term_debt", 0),
                current_ratio_prev=safe_divide(prev_data.get("current_assets", 0), prev_data.get("current_liabilities", 1)),
                shares_outstanding_prev=prev_data.get("shares_outstanding", 1000),
                gross_profit_prev=prev_data.get("gross_profit", 0),
                revenue_prev=prev_data.get("revenue", 0),
            )
            piotroski = PiotroskiFScore().calculate(inp_p)
            failing = [k for k, v in {
                "F1-ROA": piotroski.criteria.f1_roa_positive,
                "F2-CFO": piotroski.criteria.f2_cfo_positive,
                "F3-ΔROA": piotroski.criteria.f3_roa_increasing,
                "F4-Accrual": piotroski.criteria.f4_accrual_quality,
                "F5-Leverage": piotroski.criteria.f5_leverage_decreasing,
                "F6-Liquidity": piotroski.criteria.f6_liquidity_improving,
                "F7-NoDilution": piotroski.criteria.f7_no_dilution,
                "F8-GrossMargin": piotroski.criteria.f8_gross_margin_improving,
                "F9-Turnover": piotroski.criteria.f9_asset_turnover_improving,
            }.items() if not v]
            scores.update({
                "piotroski_f_score": piotroski.f_score,
                "piotroski_classification": piotroski.classification,
                "piotroski_failing": failing,
                "piotroski_interpretation": piotroski.interpretation,
            })
        except Exception as e:
            self.log_warning(f"Piotroski calculation failed: {e}")

        # ── Dechow F-Score ─────────────────────────────────────
        try:
            dechow_calc = DechowFScore()
            d_result = dechow_calc.calculate(
                total_assets=data.get("total_assets", 1),
                total_assets_prev=prev_data.get("total_assets", data.get("total_assets", 1)),
                cash=data.get("cash_equivalents", 0),
                accounts_receivable=data.get("accounts_receivable", 0),
                accounts_receivable_prev=prev_data.get("accounts_receivable", 0),
                inventory=data.get("inventory", 0),
                inventory_prev=prev_data.get("inventory", 0),
                ppe_net=data.get("ppe_net", 0),
                revenue=data.get("revenue", 0),
                revenue_prev=prev_data.get("revenue", 0),
                net_income=data.get("net_income", 0),
                net_income_prev=prev_data.get("net_income", 0),
                cfo=data.get("cfo", 0),
                shares_issued=False,
                debt_issued=data.get("total_debt", 0) > prev_data.get("total_debt", 0) * 1.1,
            )
            scores.update({
                "dechow_f_score": d_result.f_score,
                "dechow_interpretation": d_result.interpretation,
                "dechow_manipulation_flag": d_result.manipulation_likely,
            })
        except Exception as e:
            self.log_warning(f"Dechow calculation failed: {e}")

        # ── Accrual Analysis ───────────────────────────────────
        try:
            accrual = AccrualAnalyzer().calculate(
                net_income=data.get("net_income", 0),
                cfo=data.get("cfo", 0),
                ebitda=data.get("ebitda", data.get("net_income", 0) * 1.5),
                total_assets=data.get("total_assets", 1),
                total_assets_prev=prev_data.get("total_assets", data.get("total_assets", 1)),
                working_capital=data.get("current_assets", 0) - data.get("current_liabilities", 0),
                working_capital_prev=(prev_data.get("current_assets", 0) - prev_data.get("current_liabilities", 0)),
                cash=data.get("cash_equivalents", 0),
                cash_prev=prev_data.get("cash_equivalents", 0),
                depreciation=data.get("depreciation", 0),
                revenue=data.get("revenue", 0),
                accounts_receivable=data.get("accounts_receivable", 0),
                accounts_receivable_prev=prev_data.get("accounts_receivable", 0),
            )
            scores.update({
                "accrual_ratio": accrual.cf_accrual_ratio,
                "cash_earnings_ratio": accrual.cash_earnings_ratio,
                "cash_conversion_ratio": accrual.cash_conversion_ratio,
                "accrual_earnings_quality": accrual.earnings_quality,
            })
        except Exception as e:
            self.log_warning(f"Accrual calculation failed: {e}")

        return scores

    def _compare_with_known_frauds(self, scores: dict) -> dict:
        """Compare current fraud indicators with known fraud case signatures."""
        m_score = scores.get("beneish_m_score", -2.5)
        similarities = {}

        for case_name, case in FRAUD_CASE_DATABASE.items():
            case_m = case.get("m_score", -2.5)
            # Simple similarity: proximity of M-Score + flag overlap
            m_sim = max(0, 1 - abs(m_score - case_m) / 2)
            similarities[case_name] = m_sim

        return similarities

    def _calculate_fraud_risk_score(self, scores: dict) -> float:
        """Aggregate 0-100 fraud risk score from all models."""
        scorer = RiskScorer()
        subscores = []

        m_score = scores.get("beneish_m_score")
        if m_score is not None:
            subscores.append(scorer.beneish_to_score(m_score))

        z_score = scores.get("altman_z_score")
        if z_score is not None:
            subscores.append(scorer.altman_to_score(z_score) * 0.5)  # Half weight for credit

        p_score = scores.get("piotroski_f_score")
        if p_score is not None:
            subscores.append(scorer.piotroski_to_score(p_score))

        accrual = scores.get("accrual_ratio")
        if accrual is not None:
            subscores.append(scorer.accrual_to_score(accrual))

        d_score = scores.get("dechow_f_score")
        if d_score is not None:
            d_risk = min(100.0, d_score / 0.10 * 90)
            subscores.append(d_risk)

        return sum(subscores) / len(subscores) if subscores else 50.0
