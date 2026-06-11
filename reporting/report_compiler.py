"""
Report Compiler - Orchestrates generation of all output formats
PDF, DOCX, XLSX, PPTX, HTML, JSON
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
from loguru import logger
from datetime import datetime

from utils.storage import StorageManager
from database.sqlite_handler import SQLiteHandler
from config import REPORT_CONFIG


class ReportCompiler:
    """Generates all report formats from investigation results."""

    def __init__(self, storage: StorageManager, db: SQLiteHandler):
        self.storage = storage
        self.db = db

    def generate_all(
        self,
        company_name: str,
        company_id: int,
        financial_data: dict,
        agent_results: dict,
        director_result,
        profile: dict = None,
        extra_data: dict = None,
        **kwargs,
    ) -> dict:
        """Generate all report formats. Returns dict of format→path."""
        report_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        logger.info(f"Generating reports for {company_name}")

        # Compile report data
        report_data = self._compile_report_data(
            company_name, company_id, financial_data, agent_results,
            director_result, profile, extra_data,
        )

        # Save JSON findings first (always works)
        json_path = self.storage.final_output / f"{company_name}_Investigation_{timestamp}.json"
        import json
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, default=str, ensure_ascii=False)
        report_paths["json"] = json_path
        logger.info(f"JSON report saved: {json_path.name}")

        # XLSX report
        try:
            from .xlsx_generator import XLSXGenerator
            xlsx_gen = XLSXGenerator()
            xlsx_path = self.storage.final_output / f"{company_name}_Forensic_Analysis_{timestamp}.xlsx"
            xlsx_gen.generate(report_data, xlsx_path)
            report_paths["xlsx"] = xlsx_path
            logger.info(f"XLSX report saved: {xlsx_path.name}")
        except Exception as e:
            logger.error(f"XLSX generation failed: {e}")

        # DOCX report
        try:
            from .docx_generator import DOCXGenerator
            docx_gen = DOCXGenerator()
            docx_path = self.storage.final_output / f"{company_name}_Forensic_Report_{timestamp}.docx"
            docx_gen.generate(report_data, docx_path)
            report_paths["docx"] = docx_path
            logger.info(f"DOCX report saved: {docx_path.name}")
        except Exception as e:
            logger.error(f"DOCX generation failed: {e}")

        # PDF report (via DOCX conversion or reportlab)
        try:
            from .pdf_generator import PDFGenerator
            pdf_gen = PDFGenerator()
            pdf_path = self.storage.final_output / f"{company_name}_Forensic_Report_{timestamp}.pdf"
            pdf_gen.generate(report_data, pdf_path)
            report_paths["pdf"] = pdf_path
            logger.info(f"PDF report saved: {pdf_path.name}")
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")

        # PPTX deck — disabled until a user requests it.
        # To re-enable: uncomment the block below.
        # try:
        #     from .pptx_generator import PPTXGenerator
        #     pptx_path = self.storage.final_output / f"{company_name}_Executive_Deck_{timestamp}.pptx"
        #     PPTXGenerator().generate(report_data, pptx_path)
        #     report_paths["pptx"] = pptx_path
        #     logger.info(f"PPTX deck saved: {pptx_path.name}")
        # except Exception as e:
        #     logger.error(f"PPTX generation failed: {e}")

        # HTML dashboard — disabled until a web server deployment is set up.
        # To re-enable: uncomment the block below.
        # try:
        #     from .html_generator import HTMLGenerator
        #     html_path = self.storage.final_output / f"{company_name}_Dashboard_{timestamp}.html"
        #     HTMLGenerator().generate(report_data, html_path)
        #     report_paths["html"] = html_path
        #     logger.info(f"HTML dashboard saved: {html_path.name}")
        # except Exception as e:
        #     logger.error(f"HTML generation failed: {e}")

        # Save report index
        index = {
            "company": company_name,
            "generated_at": datetime.now().isoformat(),
            "reports": {k: str(v) for k, v in report_paths.items()},
        }
        self.storage.save_json(index, "report_index.json", "Final_Output")

        logger.info(f"Report generation complete. {len(report_paths)} formats generated.")
        return report_paths

    def _compile_report_data(self, company_name, company_id, financial_data, agent_results, director_result, profile, extra_data=None) -> dict:
        """Compile all investigation data into a single report dict."""
        # Get stored data from DB
        red_flags = self.db.get_red_flags(company_id)
        all_findings = self.db.get_all_findings(company_id)
        forensic_scores = self.db.get_forensic_scores(company_id)

        # Director findings
        director_summary = getattr(director_result, "summary", "")
        director_raw = getattr(director_result, "raw_analysis", "")
        overall_score = getattr(director_result, "risk_score", 50.0)
        director_red = getattr(director_result, "red_flags", [])
        director_green = getattr(director_result, "green_flags", [])

        return {
            "metadata": {
                "company_name": company_name,
                "company_id": company_id,
                "profile": profile or {},
                "report_date": datetime.now().isoformat(),
                "report_version": "1.0",
                "platform": "Forensic AI v1.0",
            },
            "executive_summary": {
                "overall_risk_score": overall_score,
                "verdict": self._extract_verdict(director_summary),
                "total_red_flags": len(director_red),
                "total_green_flags": len(director_green),
                "key_findings": director_summary,
                "final_analysis": director_raw,
            },
            "financial_data": financial_data,
            "forensic_scores": forensic_scores,
            "red_flags": [
                {
                    "agent": f.agent_name,
                    "title": f.title,
                    "detail": f.detail,
                    "evidence": f.evidence,
                    "risk_level": f.risk_level,
                    "fiscal_year": f.fiscal_year,
                }
                for f in director_red
            ],
            "green_flags": [
                {
                    "agent": f.agent_name,
                    "title": f.title,
                    "detail": f.detail,
                }
                for f in director_green
            ],
            "agent_analyses": {
                str(agent_id): {
                    "name": getattr(result, "agent_name", ""),
                    "risk_score": getattr(result, "risk_score", 50.0),
                    "summary": getattr(result, "summary", ""),
                    "red_flag_count": len(getattr(result, "red_flags", [])),
                }
                for agent_id, result in agent_results.items()
            },
            "all_findings": all_findings[:100],  # Top 100 from DB
            "management_questions": self._get_management_questions(agent_results),
            "monitoring_framework": self._get_monitoring_framework(agent_results),
            "cross_validation": (extra_data or {}).get("cross_validation_issues", []),
        }

    def _extract_verdict(self, summary: str) -> str:
        if "STRONG AVOID" in summary:
            return "STRONG AVOID"
        elif "AVOID" in summary:
            return "AVOID"
        elif "CAUTIOUS BUY" in summary:
            return "CAUTIOUS BUY"
        elif "CAUTION" in summary:
            return "CAUTION"
        elif "MONITOR" in summary:
            return "MONITOR"
        elif "BUY" in summary:
            return "BUY"
        return "MONITOR"

    def _get_management_questions(self, agent_results: dict) -> list[str]:
        director = agent_results.get(17)
        if director:
            output = self.storage.load_json("director_final_output.json", "Agent_Outputs")
            if output:
                return output.get("management_questions", [])
        return []

    def _get_monitoring_framework(self, agent_results: dict) -> list[dict]:
        output = self.storage.load_json("director_final_output.json", "Agent_Outputs")
        if output:
            return output.get("monitoring_triggers", [])
        return []
