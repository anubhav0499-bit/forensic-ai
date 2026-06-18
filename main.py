"""
FORENSIC AI — CLI Entry Point
==============================
Usage:
    python main.py "Infosys"
    python main.py "AAPL" "RELIANCE" "TCS"
    python main.py --company "Infosys" --years 7 --output /path/to/output
    python main.py --batch companies.txt

References
----------
Ramirez, S. (2020-2023). "Typer: Build great CLIs. Easy to code. Based on Python
    type hints." typer.tiangolo.com.
    CLI framework for --company, --provider, --output-dir arguments.
Williamson, W. (2020-2023). "Rich: Rich text and beautiful formatting in the
    terminal." rich.readthedocs.io.
    Progress bars, colored output, tables for terminal display.

Role
----
Primary CLI entry point for the platform. Commands:
  investigate --company "Infosys" [--provider groq] [--output-dir ./reports]
      Runs full 17-agent investigation.
  eval --company "Infosys"
      Runs offline RAGAS evaluation.
  version
      Prints platform version.
"""

import argparse
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add package root to path (works in Colab/Kaggle/local)
sys.path.insert(0, str(Path(__file__).parent))

# Ensure UTF-8 output on Windows (box-drawing chars, block elements, check marks in output)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Python < 3.7 or non-TextIOWrapper stdout


def _print_banner() -> None:
    banner = """
╔══════════════════════════════════════════════════════════════╗
║              FORENSIC AI — INSTITUTIONAL EDITION             ║
║    Multi-Agent Forensic Accounting & Financial Intelligence  ║
║    Agentic RAG | LangChain + LlamaIndex + LangGraph          ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def _print_result_summary(result: dict) -> None:
    company = result.get("company_name", "Unknown")
    score = result.get("overall_risk_score", 0)
    verdict = result.get("verdict", "N/A")
    red_flags = result.get("red_flags", 0)
    green_flags = result.get("green_flags", 0)
    docs = result.get("documents_acquired", 0)
    report_paths = result.get("report_paths", {})

    score_bar_length = int(score / 5)
    score_bar = "█" * score_bar_length + "░" * (20 - score_bar_length)
    score_color = "HIGH RISK" if score > 60 else ("MODERATE" if score > 40 else "LOW RISK")

    print(f"\n{'─'*65}")
    print(f"  Company:        {company}")
    print(f"  Risk Score:     [{score_bar}] {score:.1f}/100 ({score_color})")
    print(f"  Verdict:        {verdict}")
    print(f"  Red Flags:      {red_flags}")
    print(f"  Green Flags:    {green_flags}")
    print(f"  Documents:      {docs} acquired and analyzed")

    if result.get("error"):
        print(f"  ⚠  Error:       {result['error']}")

    if report_paths:
        print(f"\n  Reports Generated:")
        for fmt, path in report_paths.items():
            p = Path(path)
            exists = "✓" if p.exists() else "✗"
            print(f"    {exists}  [{fmt.upper()}] {p.name}")
            print(f"        {path}")

    print(f"{'─'*65}\n")


def _print_risk_components(result: dict) -> None:
    components = result.get("risk_components", {})
    if not components:
        return
    print("  Risk Component Breakdown:")
    for dim, score in sorted(components.items(), key=lambda x: -x[1]):
        bar_len = int(score / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"    {dim:<30} [{bar}] {score:.1f}")
    print()


def _print_top_red_flags(result: dict, limit: int = 5) -> None:
    flags = result.get("top_red_flags", [])
    if not flags:
        return
    print(f"  Top {min(limit, len(flags))} Red Flags:")
    for i, flag in enumerate(flags[:limit], 1):
        severity = flag.get("risk_level", "HIGH")
        title = flag.get("title", "Unknown")[:70]
        print(f"    {i}. [{severity}] {title}")
    print()


def _load_companies_from_file(filepath: str) -> list[str]:
    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    companies = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                companies.append(line)
    return companies


def _save_batch_summary(results: list[dict], output_dir: Path) -> None:
    summary = {
        "batch_date": datetime.now().isoformat(),
        "total_companies": len(results),
        "completed": sum(1 for r in results if not r.get("error")),
        "failed": sum(1 for r in results if r.get("error")),
        "results": [
            {
                "company": r.get("company_name", "Unknown"),
                "risk_score": r.get("overall_risk_score", 0),
                "verdict": r.get("verdict", "N/A"),
                "red_flags": r.get("red_flags", 0),
                "green_flags": r.get("green_flags", 0),
                "error": r.get("error"),
            }
            for r in results
        ],
    }
    summary_path = output_dir / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Batch summary saved: {summary_path}")


def _check_system_requirements() -> bool:
    """Check key dependencies before running."""
    issues = []

    try:
        import httpx
        try:
            resp = httpx.get("http://localhost:11434/api/tags", timeout=3)
            if resp.status_code != 200:
                issues.append("⚠  Ollama is running but returned an error")
        except Exception:
            issues.append("ℹ  Ollama not found — will use HuggingFace or template mode")
    except ImportError:
        issues.append("⚠  httpx not installed: pip install httpx")

    try:
        import chromadb
    except ImportError:
        issues.append("⚠  chromadb not installed: pip install chromadb")

    try:
        import sentence_transformers
    except ImportError:
        issues.append("⚠  sentence-transformers not installed: pip install sentence-transformers")

    if issues:
        print("\n  System Check:")
        for issue in issues:
            print(f"    {issue}")
        print()

    hard_errors = [i for i in issues if i.startswith("⚠")]
    return len(hard_errors) == 0


def main() -> None:
    _print_banner()

    parser = argparse.ArgumentParser(
        description="FORENSIC AI — Institutional-grade forensic accounting platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Infosys"
  python main.py "AAPL" "RELIANCE" "TCS"
  python main.py --company "HDFC Bank" --years 7
  python main.py --batch companies.txt --verbose
  python main.py --check          # Check system requirements only
        """
    )

    parser.add_argument(
        "companies",
        nargs="*",
        help="Company names or tickers to investigate (e.g. 'Infosys' 'AAPL')"
    )
    parser.add_argument(
        "--company", "-c",
        action="append",
        dest="extra_companies",
        help="Additional company to investigate (can be repeated)"
    )
    parser.add_argument(
        "--batch", "-b",
        metavar="FILE",
        help="Text file with one company per line"
    )
    parser.add_argument(
        "--years", "-y",
        type=int,
        default=5,
        help="Years of historical data to analyze (default: 5)"
    )
    parser.add_argument(
        "--output", "-o",
        metavar="DIR",
        help="Output directory override"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed risk components and flags in terminal output"
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF report generation"
    )
    parser.add_argument(
        "--no-pptx",
        action="store_true",
        help="Skip PPTX report generation"
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Generate only JSON output (fastest)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check system requirements and exit"
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:7b",
        help="Ollama model to use (default: qwen2.5:7b)"
    )

    args = parser.parse_args()

    # System check mode
    if args.check:
        print("  Checking system requirements...\n")
        ok = _check_system_requirements()

        # Check individual packages
        packages = [
            ("chromadb", "ChromaDB"),
            ("sentence_transformers", "SentenceTransformers"),
            ("fitz", "PyMuPDF"),
            ("pdfplumber", "pdfplumber"),
            ("openpyxl", "openpyxl"),
            ("docx", "python-docx"),
            ("reportlab", "reportlab"),
            ("pptx", "python-pptx"),
            ("duckdb", "DuckDB"),
            ("yfinance", "yfinance"),
            ("streamlit", "Streamlit"),
            ("plotly", "Plotly"),
            ("bs4", "BeautifulSoup4"),
        ]
        print("  Package Status:")
        for pkg, name in packages:
            try:
                __import__(pkg)
                print(f"    ✓  {name}")
            except ImportError:
                print(f"    ✗  {name}  →  pip install {pkg}")

        print("\n  Install all: pip install -r requirements.txt")
        print("  Ollama:      https://ollama.com → ollama pull qwen2.5:7b\n")
        sys.exit(0 if ok else 1)

    # Gather companies list
    companies = list(args.companies or [])
    if args.extra_companies:
        companies.extend(args.extra_companies)
    if args.batch:
        batch_companies = _load_companies_from_file(args.batch)
        companies.extend(batch_companies)
        print(f"  Loaded {len(batch_companies)} companies from {args.batch}")

    if not companies:
        parser.print_help()
        print("\n  ⚠  No companies specified. Use positional args or --company / --batch.\n")
        sys.exit(1)

    # Deduplicate while preserving order
    seen = set()
    companies = [c for c in companies if not (c in seen or seen.add(c))]

    print(f"  Investigating {len(companies)} {'company' if len(companies)==1 else 'companies'}:")
    for c in companies:
        print(f"    • {c}")
    print()

    _check_system_requirements()

    # Apply output dir override to config if requested
    if args.output:
        import os
        os.environ["FORENSIC_AI_OUTPUT_DIR"] = args.output

    # Override model if specified
    if args.model != "qwen2.5:7b":
        from config import LLM_CONFIG
        LLM_CONFIG.primary_model = args.model

    # Run investigation
    try:
        from agents.orchestrator import ForensicOrchestrator
        orchestrator = ForensicOrchestrator()
    except ImportError as e:
        print(f"  ✗  Failed to import orchestrator: {e}")
        print("  Make sure you are running from the forensic_ai directory.")
        print("  Try: cd forensic_ai && python main.py 'Infosys'")
        sys.exit(1)

    all_results = []
    start_time = time.time()

    for i, company in enumerate(companies, 1):
        print(f"[{i}/{len(companies)}] Starting investigation: {company}")
        print(f"         Time: {datetime.now().strftime('%H:%M:%S')}")
        company_start = time.time()

        try:
            result = orchestrator.investigate(company)
            elapsed = time.time() - company_start
            print(f"         ✓ Completed in {elapsed:.0f}s")
            all_results.append(result)

            _print_result_summary(result)

            if args.verbose:
                _print_risk_components(result)
                _print_top_red_flags(result, limit=5)

        except KeyboardInterrupt:
            print("\n\n  Investigation interrupted by user. Saving partial results...")
            break
        except Exception as e:
            elapsed = time.time() - company_start
            print(f"         ✗ Failed after {elapsed:.0f}s: {e}")
            all_results.append({
                "company_name": company,
                "error": str(e),
                "overall_risk_score": 0,
                "verdict": "ERROR",
                "red_flags": 0,
                "green_flags": 0,
            })

    total_elapsed = time.time() - start_time

    # Print overall summary
    print(f"\n{'═'*65}")
    print(f"  BATCH COMPLETE — {len(companies)} companies in {total_elapsed/60:.1f} minutes")
    print(f"{'─'*65}")
    print(f"  {'Company':<35} {'Score':>7}  {'Verdict':<20}")
    print(f"  {'─'*35} {'─'*7}  {'─'*20}")

    for r in sorted(all_results, key=lambda x: -x.get("overall_risk_score", 0)):
        name = r.get("company_name", "Unknown")[:33]
        score = r.get("overall_risk_score", 0)
        verdict = r.get("verdict", "ERROR")[:18]
        flag = " ⚠" if r.get("error") else ""
        print(f"  {name:<35} {score:>7.1f}  {verdict:<20}{flag}")

    print(f"{'═'*65}\n")

    # Save batch summary if multiple companies
    if len(all_results) > 1:
        try:
            from config import OUTPUT_DIR
            _save_batch_summary(all_results, OUTPUT_DIR)
        except Exception:
            pass

    # Exit with non-zero code if any investigation failed
    failed = sum(1 for r in all_results if r.get("error"))
    sys.exit(failed)


if __name__ == "__main__":
    main()
