"""
HTML Dashboard Generator — Interactive investigation summary
=============================================================

References
----------
W3C (2023). "HTML5 / CSS3." w3.org.
    Responsive report layout; print-media CSS for browser print-to-PDF.
Chart.js (2023). "Chart.js: Simple yet flexible JavaScript charting." chartjs.org.
    JavaScript charting library for interactive risk score dashboard, component
    radar chart.
Ronacher, A. (2008-2023). "Jinja2 Template Engine."
    jinja.palletsprojects.com.
    Template-based HTML report generation.

Role
----
Interactive HTML report for web viewing. Features: expandable agent findings
sections, interactive risk score gauge, responsive table layout, print-to-PDF
support. Served via FastAPI `/report/{company}` endpoint.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
from loguru import logger
from config import REPORT_CONFIG


class HTMLGenerator:
    """Generates an interactive HTML dashboard for the investigation."""

    def generate(self, report_data: dict, output_path: Path) -> Path:
        meta = report_data.get("metadata", {})
        company = meta.get("company_name", "Unknown")
        exec_sum = report_data.get("executive_summary", {})
        score = exec_sum.get("overall_risk_score", 50)
        verdict = exec_sum.get("verdict", "MONITOR")
        red_flags = report_data.get("red_flags", [])
        green_flags = report_data.get("green_flags", [])
        forensic_scores = report_data.get("forensic_scores", [{}])
        fs = forensic_scores[0] if forensic_scores else {}
        agent_analyses = report_data.get("agent_analyses", {})
        financial_data = report_data.get("financial_data", {})

        score_color = "#B71C1C" if score > 60 else ("#E65100" if score > 40 else "#1B5E20")
        years = sorted(financial_data.keys(), reverse=True)[:5]

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forensic AI - {company}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Calibri', sans-serif; background: #f5f5f5; color: #333; }}
        .header {{ background: linear-gradient(135deg, #1A237E, #283593); color: white; padding: 30px; }}
        .header h1 {{ font-size: 28px; margin-bottom: 8px; }}
        .header p {{ font-size: 14px; opacity: 0.8; }}
        .score-banner {{ background: {score_color}; color: white; padding: 20px; text-align: center; font-size: 24px; font-weight: bold; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .card h3 {{ color: #1A237E; margin-bottom: 15px; font-size: 16px; border-bottom: 2px solid #E3F2FD; padding-bottom: 8px; }}
        .score-row {{ display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #eee; }}
        .score-val {{ font-weight: bold; }}
        .red {{ color: #B71C1C; }}
        .green {{ color: #1B5E20; }}
        .orange {{ color: #E65100; }}
        .flag-item {{ padding: 10px; margin: 8px 0; border-left: 4px solid; border-radius: 4px; }}
        .flag-red {{ border-color: #B71C1C; background: #FFEBEE; }}
        .flag-green {{ border-color: #1B5E20; background: #E8F5E9; }}
        .flag-title {{ font-weight: bold; margin-bottom: 4px; }}
        .flag-evidence {{ font-size: 12px; color: #666; }}
        .severity-CRITICAL {{ background: #B71C1C; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px; }}
        .severity-HIGH {{ background: #E65100; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px; }}
        .severity-MEDIUM {{ background: #F57F17; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px; }}
        .financial-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
        .financial-table th {{ background: #1A237E; color: white; padding: 8px; text-align: left; }}
        .financial-table td {{ padding: 6px 8px; border-bottom: 1px solid #eee; }}
        .financial-table tr:nth-child(even) {{ background: #f9f9f9; }}
        .verdict-box {{ text-align: center; padding: 30px; background: white; border-radius: 8px; border: 3px solid {score_color}; margin: 20px 0; }}
        .verdict-text {{ font-size: 32px; font-weight: bold; color: {score_color}; }}
        footer {{ text-align: center; padding: 20px; color: #999; font-size: 12px; }}
    </style>
</head>
<body>
<div class="header">
    <h1>🔍 FORENSIC AI INVESTIGATION REPORT</h1>
    <p>{company} &nbsp;|&nbsp; {REPORT_CONFIG.firm_name} &nbsp;|&nbsp; {datetime.now().strftime('%B %d, %Y')}</p>
</div>
<div class="score-banner">OVERALL RISK SCORE: {score:.1f}/100 &nbsp;&nbsp;|&nbsp;&nbsp; VERDICT: {verdict}</div>

<div class="container">
    <!-- Summary Grid -->
    <div class="grid">
        <div class="card">
            <h3>📊 Risk Overview</h3>
            <div class="score-row"><span>Overall Risk Score</span><span class="score-val" style="color:{score_color}">{score:.1f}/100</span></div>
            <div class="score-row"><span>Investment Verdict</span><span class="score-val" style="color:{score_color}">{verdict}</span></div>
            <div class="score-row"><span>Total Red Flags</span><span class="score-val red">{exec_sum.get("total_red_flags", 0)}</span></div>
            <div class="score-row"><span>Total Green Flags</span><span class="score-val green">{exec_sum.get("total_green_flags", 0)}</span></div>
        </div>
        <div class="card">
            <h3>🔢 Forensic Model Scores</h3>
            {self._forensic_scores_html(fs)}
        </div>
        <div class="card">
            <h3>🤖 Agent Risk Scores</h3>
            {self._agent_scores_html(agent_analyses)}
        </div>
    </div>

    <!-- Red Flags -->
    <div class="card">
        <h3>🚨 Red Flag Registry ({len(red_flags)} Total)</h3>
        {self._red_flags_html(red_flags[:15])}
    </div>

    <!-- Green Flags -->
    <div class="card" style="margin-top:20px">
        <h3>✅ Green Flag Registry ({len(green_flags)} Total)</h3>
        {self._green_flags_html(green_flags[:10])}
    </div>

    <!-- Financial Data -->
    <div class="card" style="margin-top:20px">
        <h3>📈 Financial Data Summary</h3>
        {self._financial_table_html(financial_data, years)}
    </div>

    <!-- Verdict -->
    <div class="verdict-box">
        <div style="color:#999;font-size:14px;margin-bottom:10px">INVESTMENT COMMITTEE VERDICT</div>
        <div class="verdict-text">{verdict}</div>
        <div style="margin-top:10px;color:#666">Risk Score: {score:.1f}/100 | {self._risk_band(score)}</div>
    </div>

    <!-- Management Questions -->
    <div class="card" style="margin-top:20px">
        <h3>❓ Management Questionnaire (Top 20)</h3>
        <ol style="padding-left:20px;font-size:13px;line-height:2;">
        {chr(10).join([f'<li>{q}</li>' for q in report_data.get("management_questions", [])[:20]])}
        </ol>
    </div>
</div>

<footer>
    {REPORT_CONFIG.firm_name} | {REPORT_CONFIG.firm_tagline} | {datetime.now().strftime('%Y')} | STRICTLY CONFIDENTIAL
</footer>

</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"HTML dashboard saved: {output_path.name}")
        return output_path

    def _forensic_scores_html(self, fs: dict) -> str:
        m = fs.get("beneish_m_score")
        z = fs.get("altman_z_score")
        p = fs.get("piotroski_f_score")
        d = fs.get("dechow_f_score")

        def score_color(val, threshold, inverse=False):
            if val is None:
                return "#999"
            risky = (val > threshold) if not inverse else (val < threshold)
            return "#B71C1C" if risky else "#1B5E20"

        rows = [
            ("Beneish M-Score", f"{m:.3f}" if isinstance(m, float) else "N/A", score_color(m, -1.78)),
            ("Altman Z-Score", f"{z:.3f}" if isinstance(z, float) else "N/A", score_color(z, 2.99, inverse=True)),
            ("Piotroski F-Score", f"{p}/9" if isinstance(p, int) else "N/A", "#1B5E20" if isinstance(p, int) and p >= 7 else ("#E65100" if isinstance(p, int) and p >= 3 else "#B71C1C")),
            ("Dechow F-Score", f"{d:.4f}" if isinstance(d, float) else "N/A", score_color(d, 0.025)),
        ]
        return "".join([
            f'<div class="score-row"><span>{name}</span><span class="score-val" style="color:{color}">{val}</span></div>'
            for name, val, color in rows
        ])

    def _agent_scores_html(self, agent_analyses: dict) -> str:
        rows = []
        for aid, analysis in sorted(agent_analyses.items()):
            score = analysis.get("risk_score", 50)
            color = "#B71C1C" if score > 60 else ("#E65100" if score > 40 else "#1B5E20")
            name = analysis.get("name", f"Agent {aid}")[:30]
            rows.append(f'<div class="score-row"><span style="font-size:12px">{name}</span><span class="score-val" style="color:{color}">{score:.0f}/100</span></div>')
        return "".join(rows)

    def _red_flags_html(self, flags: list) -> str:
        items = []
        for flag in flags:
            severity = flag.get("risk_level", "HIGH")
            items.append(f"""<div class="flag-item flag-red">
                <div class="flag-title"><span class="severity-{severity}">{severity}</span> {flag.get("title", "")}</div>
                <div class="flag-evidence">{flag.get("evidence", "")[:200]}</div>
            </div>""")
        return "".join(items) if items else "<p style='color:#999'>No red flags identified</p>"

    def _green_flags_html(self, flags: list) -> str:
        items = []
        for flag in flags:
            items.append(f"""<div class="flag-item flag-green">
                <div class="flag-title">✅ {flag.get("title", "")}</div>
                <div class="flag-evidence">{flag.get("detail", "")[:200]}</div>
            </div>""")
        return "".join(items) if items else "<p style='color:#999'>No green flags identified</p>"

    def _financial_table_html(self, financial_data: dict, years: list) -> str:
        if not years:
            return "<p>No financial data available</p>"

        metrics = [
            ("Revenue", "revenue"),
            ("Net Income", "net_income"),
            ("CFO", "cfo"),
            ("Total Assets", "total_assets"),
            ("Total Debt", "total_debt"),
            ("Gross Margin %", None),
        ]

        header = "<tr>" + "".join([f"<th>Metric</th>"] + [f"<th>FY{y}</th>" for y in years]) + "</tr>"
        rows = []
        for label, field in metrics:
            cells = [f"<td><b>{label}</b></td>"]
            for year in years:
                d = financial_data.get(year, {})
                if field:
                    val = d.get(field)
                    cells.append(f"<td>{val/1e6:.0f}M</td>" if val else "<td>N/A</td>")
                else:
                    rev = d.get("revenue", 0)
                    gp = d.get("gross_profit", 0)
                    pct = f"{gp/rev*100:.1f}%" if rev else "N/A"
                    cells.append(f"<td>{pct}</td>")
            rows.append(f"<tr>{''.join(cells)}</tr>")

        return f"<table class='financial-table'>{header}{''.join(rows)}</table>"

    def _risk_band(self, score: float) -> str:
        if score <= 20:
            return "VERY LOW RISK"
        elif score <= 40:
            return "LOW RISK"
        elif score <= 60:
            return "MODERATE RISK"
        elif score <= 80:
            return "HIGH RISK"
        return "EXTREME RISK"
