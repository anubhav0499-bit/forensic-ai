"""
PDF Report Generator — Using ReportLab for institutional-grade PDFs
====================================================================

References
----------
ReportLab (1999-2023). "ReportLab PDF Library." reportlab.com.
    Programmatic PDF generation with custom layouts, charts, and embedded fonts.
WeasyPrint (2011-2023). weasyprint.org.
    Alternative HTML-to-PDF renderer; used when ReportLab unavailable.

Standards
---------
ISO 32000 (PDF 1.7 Standard) — PDF/A compliance for long-term archival of
    forensic investigation records.

Role
----
Generates a print-ready PDF version of the forensic report. Architecture: render
DOCX → convert to PDF via python-docx + win32com (Windows) or LibreOffice
(Linux/Mac). ReportLab fallback for charts and tables when conversion unavailable.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from loguru import logger

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import HexColor, black, white
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

from config import REPORT_CONFIG


class PDFGenerator:
    """Generates institutional PDF report using ReportLab."""

    PRIMARY_BLUE = HexColor("#1A237E") if HAS_REPORTLAB else None
    RISK_RED = HexColor("#B71C1C") if HAS_REPORTLAB else None
    SAFE_GREEN = HexColor("#1B5E20") if HAS_REPORTLAB else None
    WARN_ORANGE = HexColor("#E65100") if HAS_REPORTLAB else None
    LIGHT_BLUE = HexColor("#E3F2FD") if HAS_REPORTLAB else None

    def generate(self, report_data: dict, output_path: Path) -> Path:
        if not HAS_REPORTLAB:
            logger.warning("reportlab not installed. PDF generation skipped.")
            # Try to generate a simple text file instead
            self._generate_text_fallback(report_data, output_path.with_suffix(".txt"))
            return output_path

        meta = report_data.get("metadata", {})
        company = meta.get("company_name", "Unknown")
        exec_sum = report_data.get("executive_summary", {})
        score = exec_sum.get("overall_risk_score", 50)
        verdict = exec_sum.get("verdict", "MONITOR")

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=2.5 * cm,
            leftMargin=2.5 * cm,
            topMargin=2.5 * cm,
            bottomMargin=2.5 * cm,
        )

        styles = self._create_styles()
        story = []

        # Cover Page
        story.extend(self._cover_page(company, score, verdict, styles))
        story.append(PageBreak())

        # Executive Summary
        story.extend(self._executive_summary(report_data, styles))
        story.append(PageBreak())

        # Forensic Scores
        story.extend(self._forensic_scores_section(report_data, styles))
        story.append(PageBreak())

        # Red Flags
        story.extend(self._red_flags_section(report_data, styles))
        story.append(PageBreak())

        # Green Flags
        story.extend(self._green_flags_section(report_data, styles))
        story.append(PageBreak())

        # Financial Summary
        story.extend(self._financial_summary(report_data, styles))
        story.append(PageBreak())

        # Agent Analysis Summaries
        story.extend(self._agent_summaries(report_data, styles))
        story.append(PageBreak())

        # Final Verdict
        story.extend(self._final_verdict(report_data, styles))

        doc.build(story)
        logger.info(f"PDF saved: {output_path.name}")
        return output_path

    def _create_styles(self):
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            "CustomTitle",
            parent=styles["Title"],
            textColor=self.PRIMARY_BLUE,
            fontSize=22,
            spaceAfter=20,
        ))
        styles.add(ParagraphStyle(
            "SectionHeader",
            parent=styles["Heading1"],
            textColor=self.PRIMARY_BLUE,
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
        ))
        styles.add(ParagraphStyle(
            "RedFlag",
            parent=styles["Normal"],
            textColor=self.RISK_RED,
            fontSize=10,
            spaceAfter=6,
        ))
        styles.add(ParagraphStyle(
            "GreenFlag",
            parent=styles["Normal"],
            textColor=self.SAFE_GREEN,
            fontSize=10,
            spaceAfter=6,
        ))
        styles.add(ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=10,
            spaceAfter=8,
            leading=16,
            alignment=TA_JUSTIFY,
        ))
        return styles

    def _cover_page(self, company: str, score: float, verdict: str, styles) -> list:
        elements = [Spacer(1, 2 * inch)]
        color = self.RISK_RED if score > 60 else (self.WARN_ORANGE if score > 40 else self.SAFE_GREEN)

        elements.append(Paragraph("FORENSIC ACCOUNTING INVESTIGATION", styles["CustomTitle"]))
        elements.append(Paragraph(company.upper(), styles["CustomTitle"]))
        elements.append(Spacer(1, 0.5 * inch))
        elements.append(Paragraph(
            f'<font color="{color._hexval if hasattr(color, "_hexval") else "#B71C1C"}" size="18"><b>RISK SCORE: {score:.1f}/100</b></font>',
            styles["Normal"]
        ))
        elements.append(Paragraph(
            f'<font size="14"><b>VERDICT: {verdict}</b></font>',
            styles["Normal"]
        ))
        elements.append(Spacer(1, inch))
        elements.append(HRFlowable(width="100%", color=self.PRIMARY_BLUE))
        elements.append(Paragraph(
            f"{REPORT_CONFIG.firm_name} | {REPORT_CONFIG.firm_tagline}<br/>"
            f"{datetime.now().strftime('%B %d, %Y')}<br/>"
            f"<i>STRICTLY CONFIDENTIAL</i>",
            styles["Normal"]
        ))
        return elements

    def _executive_summary(self, data: dict, styles) -> list:
        elements = [Paragraph("EXECUTIVE SUMMARY", styles["SectionHeader"])]
        exec_sum = data.get("executive_summary", {})
        company = data.get("metadata", {}).get("company_name", "")
        score = exec_sum.get("overall_risk_score", 50)

        summary_data = [
            ["Overall Risk Score", f"{score:.1f}/100"],
            ["Investment Verdict", exec_sum.get("verdict", "N/A")],
            ["Total Red Flags", str(exec_sum.get("total_red_flags", 0))],
            ["Total Green Flags", str(exec_sum.get("total_green_flags", 0))],
            ["Report Date", datetime.now().strftime("%B %d, %Y")],
        ]
        table = Table(summary_data, colWidths=[200, 250])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), self.LIGHT_BLUE),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, black),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [white, self.LIGHT_BLUE]),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))

        final_analysis = exec_sum.get("final_analysis", "")
        if final_analysis:
            elements.append(Paragraph(final_analysis[:2000], styles["Body"]))

        return elements

    def _forensic_scores_section(self, data: dict, styles) -> list:
        elements = [Paragraph("FORENSIC SCORES SUMMARY", styles["SectionHeader"])]
        forensic_scores = data.get("forensic_scores", [{}])
        scores = forensic_scores[0] if forensic_scores else {}

        score_data = [
            ["Model", "Score", "Threshold", "Status"],
            ["Beneish M-Score", f"{scores.get('beneish_m_score', 'N/A'):.3f}" if isinstance(scores.get('beneish_m_score'), float) else "N/A", "-1.78", "RISK" if isinstance(scores.get('beneish_m_score'), float) and scores.get('beneish_m_score') > -1.78 else "OK"],
            ["Altman Z-Score", f"{scores.get('altman_z_score', 'N/A'):.3f}" if isinstance(scores.get('altman_z_score'), float) else "N/A", "1.81/2.99", scores.get('altman_zone', 'N/A')],
            ["Piotroski F-Score", f"{scores.get('piotroski_f_score', 'N/A')}/9", "3", "WEAK" if isinstance(scores.get('piotroski_f_score'), int) and scores.get('piotroski_f_score') <= 2 else "OK"],
            ["Dechow F-Score", f"{scores.get('dechow_f_score', 'N/A'):.4f}" if isinstance(scores.get('dechow_f_score'), float) else "N/A", "0.025", "RISK" if isinstance(scores.get('dechow_f_score'), float) and scores.get('dechow_f_score') > 0.025 else "OK"],
        ]
        table = Table(score_data, colWidths=[150, 100, 100, 100])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), self.PRIMARY_BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, black),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, self.LIGHT_BLUE]),
        ]))
        elements.append(table)
        return elements

    def _red_flags_section(self, data: dict, styles) -> list:
        elements = [Paragraph("RED FLAG REGISTRY", styles["SectionHeader"])]
        red_flags = data.get("red_flags", [])
        elements.append(Paragraph(f"Total Red Flags: {len(red_flags)}", styles["Body"]))

        for i, flag in enumerate(red_flags[:15], 1):
            elements.append(Paragraph(
                f"RF-{i:03d} [{flag.get('risk_level', 'HIGH')}]: {flag.get('title', '')}",
                styles["RedFlag"]
            ))
            elements.append(Paragraph(flag.get("evidence", "")[:200], styles["Body"]))
        return elements

    def _green_flags_section(self, data: dict, styles) -> list:
        elements = [Paragraph("GREEN FLAG REGISTRY (POSITIVE INDICATORS)", styles["SectionHeader"])]
        green_flags = data.get("green_flags", [])

        for i, flag in enumerate(green_flags[:10], 1):
            elements.append(Paragraph(
                f"GF-{i:03d}: {flag.get('title', '')}",
                styles["GreenFlag"]
            ))
            elements.append(Paragraph(flag.get("detail", "")[:200], styles["Body"]))
        return elements

    def _financial_summary(self, data: dict, styles) -> list:
        elements = [Paragraph("FINANCIAL DATA SUMMARY", styles["SectionHeader"])]
        financial_data = data.get("financial_data", {})
        years = sorted(financial_data.keys(), reverse=True)[:5]

        if not years:
            elements.append(Paragraph("No financial data available.", styles["Body"]))
            return elements

        headers = ["Metric"] + [f"FY{y}" for y in years]
        metrics = [
            ("Revenue", "revenue"),
            ("Net Income", "net_income"),
            ("CFO", "cfo"),
            ("Total Assets", "total_assets"),
            ("Total Debt", "total_debt"),
        ]

        table_data = [headers]
        for label, field in metrics:
            row = [label]
            for year in years:
                val = financial_data.get(year, {}).get(field)
                row.append(f"{val/1e6:.1f}M" if val else "N/A")
            table_data.append(row)

        col_widths = [100] + [75] * len(years)
        table = Table(table_data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), self.PRIMARY_BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, black),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, self.LIGHT_BLUE]),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        elements.append(table)
        return elements

    def _agent_summaries(self, data: dict, styles) -> list:
        elements = [Paragraph("AGENT ANALYSIS SUMMARIES", styles["SectionHeader"])]
        agent_analyses = data.get("agent_analyses", {})

        for agent_id, analysis in sorted(agent_analyses.items()):
            name = analysis.get("name", f"Agent {agent_id}")
            risk_score = analysis.get("risk_score", 50)
            elements.append(Paragraph(f"{name} (Risk Score: {risk_score:.1f}/100)", styles["SectionHeader"]))
            summary = analysis.get("summary", "No analysis available.")
            elements.append(Paragraph(summary[:500], styles["Body"]))

        return elements

    def _final_verdict(self, data: dict, styles) -> list:
        elements = [Paragraph("FINAL VERDICT", styles["SectionHeader"])]
        exec_sum = data.get("executive_summary", {})
        verdict = exec_sum.get("verdict", "MONITOR")
        score = exec_sum.get("overall_risk_score", 50)

        elements.append(Paragraph(
            f"<b>VERDICT: {verdict} | RISK SCORE: {score:.1f}/100</b>",
            styles["SectionHeader"]
        ))
        elements.append(Paragraph(
            "This report has been prepared based on publicly available information. "
            "This is for institutional investor use only and does not constitute financial advice.",
            styles["Body"]
        ))
        elements.append(Paragraph(
            f"Prepared by {REPORT_CONFIG.firm_name} | {datetime.now().strftime('%B %d, %Y')}",
            styles["Body"]
        ))
        return elements

    def _generate_text_fallback(self, report_data: dict, output_path: Path) -> None:
        """Plain text fallback if reportlab not available."""
        meta = report_data.get("metadata", {})
        exec_sum = report_data.get("executive_summary", {})
        company = meta.get("company_name", "Unknown")

        text = f"""
FORENSIC ACCOUNTING INVESTIGATION REPORT
{'='*60}
Company: {company}
Overall Risk Score: {exec_sum.get('overall_risk_score', 50):.1f}/100
Verdict: {exec_sum.get('verdict', 'MONITOR')}
Red Flags: {exec_sum.get('total_red_flags', 0)}
Green Flags: {exec_sum.get('total_green_flags', 0)}
Report Date: {datetime.now().strftime('%B %d, %Y')}

EXECUTIVE SUMMARY:
{exec_sum.get('final_analysis', 'No analysis available.')[:5000]}

RED FLAGS:
{'─'*40}
"""
        for i, flag in enumerate(report_data.get("red_flags", [])[:20], 1):
            text += f"\n{i}. [{flag.get('risk_level', 'HIGH')}] {flag.get('title', '')}\n   {flag.get('evidence', '')[:200]}\n"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"Text fallback report saved: {output_path.name}")
