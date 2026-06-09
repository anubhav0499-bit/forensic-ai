"""
FORENSIC AI - Streamlit Web Interface
=====================================
Multi-Agent Forensic Accounting & Financial Intelligence Platform
No paid APIs. No cloud dependency. Runs locally.
"""

from __future__ import annotations
import streamlit as st
import json
import time
import sys
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="Forensic AI - Institutional Forensic Accounting Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1A237E, #283593);
        color: white; padding: 25px 30px; border-radius: 10px;
        margin-bottom: 25px;
    }
    .main-header h1 { font-size: 28px; margin-bottom: 5px; }
    .main-header p { font-size: 13px; opacity: 0.85; }
    .risk-score-box {
        text-align: center; padding: 20px; border-radius: 8px;
        border: 3px solid; margin: 10px 0;
    }
    .metric-card {
        background: white; padding: 15px; border-radius: 8px;
        border: 1px solid #E0E0E0; margin: 5px 0;
    }
    .red-flag-item {
        background: #FFEBEE; padding: 10px; border-left: 4px solid #B71C1C;
        margin: 5px 0; border-radius: 4px;
    }
    .green-flag-item {
        background: #E8F5E9; padding: 10px; border-left: 4px solid #1B5E20;
        margin: 5px 0; border-radius: 4px;
    }
    .agent-card {
        background: #F5F5F5; padding: 12px; border-radius: 6px;
        margin: 6px 0; border-left: 3px solid #1A237E;
    }
    .stButton > button {
        background: #1A237E; color: white; border-radius: 6px;
        padding: 10px 25px; font-weight: bold; border: none;
        font-size: 15px;
    }
    .stButton > button:hover { background: #283593; }
</style>
""", unsafe_allow_html=True)


def main():
    # ─── Header ────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🔍 FORENSIC AI</h1>
        <p>Multi-Agent Forensic Accounting & Financial Intelligence Platform</p>
        <p>Evidence-First | No Hallucinations | Institutional Grade | 100% Local</p>
    </div>
    """, unsafe_allow_html=True)

    # ─── Sidebar ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("# 🔍 Forensic AI")
        st.markdown("### ⚙️ Configuration")

        with st.expander("🤖 LLM Backend", expanded=True):
            backend = st.selectbox("LLM Provider", ["Ollama (Local)", "HuggingFace", "Template (No LLM)"])
            ollama_model = st.text_input("Ollama Model", value="qwen2.5:7b")
            st.info("Install Ollama: https://ollama.com\nThen: `ollama pull qwen2.5:7b`")

        with st.expander("📋 Investigation Settings"):
            years_history = st.slider("Years of History", 2, 10, 5)
            enable_ocr = st.checkbox("Enable OCR for Scanned PDFs", value=True)
            enable_peer_comparison = st.checkbox("Include Peer Benchmarking", value=True)

        with st.expander("📁 Output Settings"):
            generate_pdf = st.checkbox("Generate PDF", value=True)
            generate_docx = st.checkbox("Generate DOCX", value=True)
            generate_xlsx = st.checkbox("Generate XLSX", value=True)
            generate_pptx = st.checkbox("Generate PPTX", value=True)
            generate_html = st.checkbox("Generate HTML Dashboard", value=True)

        st.markdown("---")
        st.markdown("### 📊 Platform Status")
        try:
            import httpx
            resp = httpx.get("http://localhost:11434/api/tags", timeout=3)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                st.success(f"✅ Ollama: {len(models)} models available")
            else:
                st.warning("⚠️ Ollama: Not responding")
        except Exception:
            st.warning("⚠️ Ollama: Not installed")

        try:
            from sentence_transformers import SentenceTransformer
            st.success("✅ Embeddings: Ready")
        except ImportError:
            st.error("❌ Embeddings: sentence-transformers not installed")

        try:
            import chromadb
            st.success("✅ Vector DB: ChromaDB ready")
        except ImportError:
            st.error("❌ ChromaDB: Not installed")

    # ─── Main Content ───────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 New Investigation",
        "📊 Results Dashboard",
        "📚 Knowledge Base",
        "⚙️ System Info"
    ])

    with tab1:
        _investigation_tab(years_history)

    with tab2:
        _results_dashboard_tab()

    with tab3:
        _knowledge_base_tab()

    with tab4:
        _system_info_tab()


def _investigation_tab(years_history: int) -> None:
    st.markdown("## 🏢 Start Forensic Investigation")
    st.markdown("Enter company names below. The platform will automatically discover and analyze all public filings.")

    col1, col2 = st.columns([2, 1])
    with col1:
        companies_input = st.text_area(
            "Company Names (one per line)",
            placeholder="Infosys\nRELIANCE\nAAPL\nTCS",
            height=150,
            help="Enter company names or stock tickers. Works for Indian (NSE/BSE) and US (SEC EDGAR) listed companies."
        )

    with col2:
        st.markdown("### Quick Examples")
        for example in ["Infosys", "RELIANCE", "AAPL", "HDFC Bank", "TCS"]:
            if st.button(f"+ {example}", key=f"ex_{example}"):
                st.session_state["prefill_company"] = example

        st.markdown("---")
        st.markdown("### What's Included")
        st.markdown("""
        - 17 Forensic Agents
        - Beneish M-Score
        - Altman Z-Score
        - Piotroski F-Score
        - Dechow F-Score
        - Management NLP
        - Related Party Analysis
        - Peer Benchmarking
        - 6 Report Formats
        """)

    if not companies_input.strip():
        st.info("👆 Enter at least one company name to begin the investigation.")
        return

    companies = [c.strip() for c in companies_input.strip().split("\n") if c.strip()]

    if not companies:
        return

    st.markdown(f"**Ready to investigate:** {len(companies)} companies")
    for c in companies:
        st.write(f"  • {c}")

    if st.button("🚀 START FORENSIC INVESTIGATION", type="primary"):
        _run_investigation(companies, years_history)


def _run_investigation(companies: list[str], years_history: int) -> None:
    """Run the full forensic investigation with live progress."""
    overall_progress = st.progress(0, text="Initializing investigation...")
    status_container = st.empty()
    results_container = st.container()

    all_results = []

    for company_idx, company in enumerate(companies):
        company_progress = (company_idx) / len(companies)
        overall_progress.progress(company_progress, text=f"Investigating: {company}")

        with st.expander(f"🔍 {company} - Investigation Log", expanded=True):
            log_area = st.empty()
            log_messages = []

            def add_log(msg):
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_messages.append(f"[{timestamp}] {msg}")
                log_area.code("\n".join(log_messages[-20:]))

            try:
                add_log("Phase 1: Initializing orchestrator...")
                from agents.orchestrator import ForensicOrchestrator
                orchestrator = ForensicOrchestrator()

                add_log(f"Phase 2: Identifying company '{company}'...")
                add_log("Phase 3: Discovering documents from SEC EDGAR / NSE / BSE...")
                add_log("Phase 4: Downloading and processing documents...")
                add_log("Phase 5: Building forensic knowledge base...")
                add_log("Phase 6: Running 17 specialist agents in parallel...")
                add_log("  ├── Agent 6: Fraud Detection (Beneish, Dechow, Altman, Piotroski)")
                add_log("  ├── Agent 11: Management NLP Analysis")
                add_log("  ├── Agents 3-5: Revenue, Cash Flow, Working Capital")
                add_log("  ├── Agents 7-10: Credit, Earnings, Related Party, Auditor")
                add_log("  ├── Agents 12-16: Peer, Patterns, Short Seller, Bull, Devil's Advocate")
                add_log("  └── Agent 17: Chief Investigation Director synthesizing findings...")
                add_log("Phase 7: Generating reports (PDF, DOCX, XLSX, PPTX, HTML)...")

                result = orchestrator.investigate(company)
                all_results.append(result)

                add_log(f"✅ Investigation COMPLETE!")
                add_log(f"   Risk Score: {result.get('overall_risk_score', 0):.1f}/100")
                add_log(f"   Red Flags: {result.get('red_flags', 0)}")
                add_log(f"   Reports saved to: {result.get('storage_path', 'N/A')}")

                # Show quick results
                score = result.get("overall_risk_score", 50)
                verdict = result.get("verdict", "MONITOR")
                score_color = "#B71C1C" if score > 60 else ("#E65100" if score > 40 else "#1B5E20")

                st.markdown(f"""
                <div class="risk-score-box" style="border-color:{score_color}">
                    <h2 style="color:{score_color}">{verdict}</h2>
                    <h3>Risk Score: {score:.1f}/100</h3>
                    <p>Red Flags: {result.get('red_flags', 0)} | Green Flags: {result.get('green_flags', 0)}</p>
                </div>
                """, unsafe_allow_html=True)

                # Download buttons for reports
                report_paths = result.get("report_paths", {})
                if report_paths:
                    st.markdown("### 📥 Download Reports")
                    cols = st.columns(len(report_paths))
                    for i, (fmt, path) in enumerate(report_paths.items()):
                        path_obj = Path(path)
                        if path_obj.exists():
                            with open(path, "rb") as f:
                                with cols[i]:
                                    st.download_button(
                                        label=f"📄 {fmt.upper()}",
                                        data=f,
                                        file_name=path_obj.name,
                                        mime="application/octet-stream",
                                    )

            except Exception as e:
                add_log(f"❌ Error: {e}")
                st.error(f"Investigation failed for {company}: {e}")
                st.exception(e)

        overall_progress.progress(
            (company_idx + 1) / len(companies),
            text=f"Completed: {company}"
        )

    overall_progress.progress(1.0, text="All investigations complete!")
    st.success(f"✅ Investigated {len(companies)} companies. See results in the Results Dashboard tab.")

    if all_results:
        st.session_state["investigation_results"] = all_results


def _results_dashboard_tab() -> None:
    st.markdown("## 📊 Investigation Results Dashboard")

    results = st.session_state.get("investigation_results", [])
    if not results:
        st.info("No investigation results yet. Run an investigation in the 'New Investigation' tab.")

        # Load recent results from filesystem
        from config import OUTPUT_DIR
        if OUTPUT_DIR.exists():
            recent_dirs = sorted(OUTPUT_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:10]
            if recent_dirs:
                st.markdown("### Recent Investigations")
                for d in recent_dirs:
                    if d.is_dir():
                        index_file = d / "Final_Output" / "report_index.json"
                        if index_file.exists():
                            with open(index_file) as f:
                                idx = json.load(f)
                            st.write(f"**{d.name}** - {idx.get('generated_at', 'Unknown date')[:10]}")
        return

    for result in results:
        company = result.get("company_name", "Unknown")
        score = result.get("overall_risk_score", 50)
        score_color = "#B71C1C" if score > 60 else ("#E65100" if score > 40 else "#1B5E20")
        verdict = result.get("verdict", "MONITOR")

        with st.expander(f"📋 {company} — Score: {score:.1f}/100 | {verdict}", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Risk Score", f"{score:.1f}/100")
            with col2:
                st.metric("Red Flags", result.get("red_flags", 0))
            with col3:
                st.metric("Green Flags", result.get("green_flags", 0))
            with col4:
                st.metric("Documents", result.get("documents_acquired", 0))

            # Agent scores
            st.markdown("#### Agent Risk Scores")
            agent_results = result.get("agent_results", {})
            agent_cols = st.columns(3)
            for i, (agent_id, summary) in enumerate(list(agent_results.items())[:9]):
                with agent_cols[i % 3]:
                    st.markdown(f'<div class="agent-card"><small>{summary[:100]}...</small></div>', unsafe_allow_html=True)


def _knowledge_base_tab() -> None:
    st.markdown("## 📚 Forensic Knowledge Base")
    st.info("The knowledge base stores all processed documents, embeddings, and extracted financial data.")

    from config import OUTPUT_DIR, DB_CONFIG
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Companies Investigated")
        if OUTPUT_DIR.exists():
            companies = [d.name for d in OUTPUT_DIR.iterdir() if d.is_dir()]
            if companies:
                for c in companies:
                    st.write(f"• {c}")
            else:
                st.write("No investigations yet.")
        else:
            st.write("Output directory not created yet.")

    with col2:
        st.markdown("### Database Status")
        try:
            from database.sqlite_handler import SQLiteHandler
            db = SQLiteHandler()
            companies = db.execute_query("SELECT name, ticker FROM companies ORDER BY created_at DESC LIMIT 10")
            if companies:
                for c in companies:
                    st.write(f"• {c['name']} ({c.get('ticker', 'N/A')})")
            else:
                st.write("No companies in database.")
        except Exception as e:
            st.write(f"Database: {e}")


def _system_info_tab() -> None:
    st.markdown("## ⚙️ System Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🤖 LLM Backend")
        try:
            from llm.llm_manager import LLMManager
            llm = LLMManager()
            info = llm.get_backend_info()
            st.json(info)
        except Exception as e:
            st.error(f"LLM check failed: {e}")

        st.markdown("### 🔍 Embedding Model")
        try:
            from rag.embeddings import EmbeddingModel
            em = EmbeddingModel()
            st.success(f"Model: {em.model_name}")
            st.info(f"Dimension: {em.get_dimension()}")
        except Exception as e:
            st.error(f"Embedding check: {e}")

    with col2:
        st.markdown("### 📦 Package Status")
        packages = [
            ("langchain", "LangChain"),
            ("langgraph", "LangGraph"),
            ("chromadb", "ChromaDB"),
            ("sentence_transformers", "SentenceTransformers"),
            ("fitz", "PyMuPDF"),
            ("pdfplumber", "pdfplumber"),
            ("camelot", "Camelot"),
            ("openpyxl", "openpyxl"),
            ("docx", "python-docx"),
            ("pptx", "python-pptx"),
            ("reportlab", "reportlab"),
            ("duckdb", "DuckDB"),
            ("yfinance", "yfinance"),
            ("plotly", "Plotly"),
        ]
        for pkg, name in packages:
            try:
                __import__(pkg)
                st.success(f"✅ {name}")
            except ImportError:
                st.error(f"❌ {name}")

        st.markdown("### 🔧 Installation Help")
        st.code("pip install -r requirements.txt", language="bash")
        st.code("ollama pull qwen2.5:7b", language="bash")


if __name__ == "__main__":
    main()
