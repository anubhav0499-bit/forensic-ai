"""
RAGAS Evaluation Suite — Context relevance, faithfulness, answer relevance scoring.
Uses the ragas library when available; falls back to custom overlap metrics otherwise.

Run standalone:
    python -m eval.ragas_eval --company "Infosys" --save
"""

from __future__ import annotations
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from loguru import logger

from config import DATA_DIR


@dataclass
class RAGASResult:
    context_relevancy: float      # fraction of retrieved context relevant to question
    faithfulness: float           # fraction of answer grounded in context
    answer_relevancy: float       # fraction of answer relevant to the question
    context_recall: float         # fraction of ground-truth covered by context
    composite_score: float        # weighted average
    method: str                   # "ragas" | "custom"
    num_questions: int = 0


class RAGASEvaluator:
    """
    RAGAS-powered evaluation for Forensic AI RAG pipeline.

    Built-in test set: 10 forensic accounting questions covering all major
    agent domains (revenue, cash flow, RPT, auditor, governance, credit).

    Metric weights (composite score):
      context_relevancy   30 %
      faithfulness        35 %
      answer_relevancy    25 %
      context_recall      10 %
    """

    WEIGHTS = {
        "context_relevancy": 0.30,
        "faithfulness":      0.35,
        "answer_relevancy":  0.25,
        "context_recall":    0.10,
    }

    TEST_QUESTIONS = [
        {
            "question": "What is the revenue recognition policy and are there signs of premature recognition?",
            "ground_truth": (
                "Revenue is recognised per Ind AS 115 five-step model. "
                "Premature recognition signals include DSO expansion >15 days YoY and Q4 spikes "
                "reversing in Q1, per ISA 240 §26."
            ),
        },
        {
            "question": "Is the company at risk of a going concern qualification?",
            "ground_truth": (
                "Going concern risk per ISA 570: ICR <1.5x, negative FCF for 3+ years, "
                "covenant breach, and auditor emphasis of matter are key triggers."
            ),
        },
        {
            "question": "Are related party transactions conducted at arm's length and properly disclosed?",
            "ground_truth": (
                "SEBI RPT Regulations 2021 require audit-committee pre-approval and shareholder "
                "approval for material RPTs. Arm's-length pricing must be verified against "
                "comparable third-party transactions per ISA 550."
            ),
        },
        {
            "question": "What does the auditor's report signal about financial statement risk?",
            "ground_truth": (
                "Going concern emphasis per ISA 570 = CRITICAL. Qualified opinion per ISA 705 = HIGH. "
                "Non-audit fees >40% = PCAOB Rule 3521 independence risk. "
                "Auditor downgrade (Big4 → small firm) = highest governance red flag."
            ),
        },
        {
            "question": "What are promoter pledge levels and what risk do they represent?",
            "ground_truth": (
                "SEBI SAST 2011: >30% pledge = elevated; >50% = high; >70% = critical "
                "with collateral-call risk threatening promoter control and share price."
            ),
        },
        {
            "question": "What does the Beneish M-Score indicate about earnings manipulation?",
            "ground_truth": (
                "Beneish M-Score >-1.78 = manipulation likely. Three or more Beneish indices "
                "above paper thresholds indicate elevated risk regardless of the aggregate score."
            ),
        },
        {
            "question": "What is the quality of operating cash flow relative to reported EBITDA?",
            "ground_truth": (
                "Healthy firms convert >80% of EBITDA to OCF. Sustained conversion <60% signals "
                "earnings quality concern. Negative OCF with positive PAT for 2+ years flags "
                "possible EBITDA fiction (PCAOB AS 2401)."
            ),
        },
        {
            "question": "What are the key credit risk indicators and debt serviceability metrics?",
            "ground_truth": (
                "Altman EM Z-Score <1.10 = distress zone. ICR <1.5x = stress; <1.0x = default risk. "
                "Net Debt/EBITDA >6x = highly leveraged. Short-term debt >60% of total = rollover risk."
            ),
        },
        {
            "question": "Are there indicators of capital expenditure manipulation or asset inflation?",
            "ground_truth": (
                "Capex CAGR > Revenue CAGR + 10% without capacity expansion = WorldCom pattern. "
                "DEPI >1.083 signals slowing depreciation. Rising CWIP ageing without completion "
                "may indicate fictitious assets (IAS 16 / Ind AS 16)."
            ),
        },
        {
            "question": "What governance red flags are present based on board composition and director exits?",
            "ground_truth": (
                "SEBI LODR Reg 17: board must be ≥50% independent. "
                "≥2 independent director exits in 12 months = COSO Control Environment signal. "
                "CFO change is correlated with earnings manipulation (Dechow et al. 2010)."
            ),
        },
    ]

    def __init__(self, llm_manager=None, retriever=None):
        self.llm = llm_manager
        self.retriever = retriever
        self._ragas_ok = self._probe_ragas()

    def _probe_ragas(self) -> bool:
        try:
            import ragas  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "ragas not installed — using custom overlap metrics. "
                "Install with: pip install ragas"
            )
            return False

    # ── Public ────────────────────────────────────────────────────────

    def evaluate(
        self,
        company_name: str,
        questions: list[dict] | None = None,
        save_results: bool = True,
    ) -> RAGASResult:
        """
        Run evaluation for a company.
        questions: optional override list; defaults to TEST_QUESTIONS.
        """
        test_set = questions or self.TEST_QUESTIONS
        logger.info(
            f"RAGAS eval: {len(test_set)} questions, "
            f"method={'ragas' if self._ragas_ok and self.llm and self.retriever else 'custom'}"
        )

        if self._ragas_ok and self.llm and self.retriever:
            result = self._run_ragas(company_name, test_set)
        else:
            result = self._run_custom(company_name, test_set)

        if save_results:
            self._save(company_name, result)

        logger.info(
            f"RAGAS [{company_name}] — "
            f"context_relevancy={result.context_relevancy:.3f} "
            f"faithfulness={result.faithfulness:.3f} "
            f"answer_relevancy={result.answer_relevancy:.3f} "
            f"composite={result.composite_score:.3f}"
        )
        return result

    # ── RAGAS-library path ────────────────────────────────────────────

    def _run_ragas(self, company_name: str, test_set: list[dict]) -> RAGASResult:
        try:
            from ragas import evaluate
            from ragas.metrics import (
                context_relevancy,
                faithfulness,
                answer_relevancy,
                context_recall,
            )
            from datasets import Dataset

            rows = []
            for item in test_set:
                contexts = self._retrieve(company_name, item["question"])
                answer = self._generate(item["question"], contexts)
                rows.append({
                    "question":     item["question"],
                    "answer":       answer,
                    "contexts":     contexts,
                    "ground_truth": item.get("ground_truth", ""),
                })

            ds = Dataset.from_list(rows)
            scores = evaluate(
                ds,
                metrics=[context_relevancy, faithfulness, answer_relevancy, context_recall],
            )
            cr  = float(scores.get("context_relevancy",  0.0))
            f   = float(scores.get("faithfulness",       0.0))
            ar  = float(scores.get("answer_relevancy",   0.0))
            rec = float(scores.get("context_recall",     0.0))
            comp = (
                cr  * self.WEIGHTS["context_relevancy"]
                + f * self.WEIGHTS["faithfulness"]
                + ar * self.WEIGHTS["answer_relevancy"]
                + rec * self.WEIGHTS["context_recall"]
            )
            return RAGASResult(
                context_relevancy=cr, faithfulness=f,
                answer_relevancy=ar,  context_recall=rec,
                composite_score=round(comp, 4), method="ragas",
                num_questions=len(test_set),
            )
        except Exception as exc:
            logger.warning(f"ragas library evaluation failed ({exc}); falling back to custom")
            return self._run_custom(company_name, test_set)

    # ── Custom-metrics fallback ────────────────────────────────────────

    def _run_custom(self, company_name: str, test_set: list[dict]) -> RAGASResult:
        import re
        cr_scores, f_scores, ar_scores = [], [], []
        term_re = re.compile(r"\b[a-zA-Z0-9]{4,}\b")

        for item in test_set:
            contexts = self._retrieve(company_name, item["question"])
            answer   = self._generate(item["question"], contexts)
            ctx_text = " ".join(contexts).lower()
            ctx_set  = set(term_re.findall(ctx_text))
            q_set    = set(term_re.findall(item["question"].lower()))

            # Context relevancy: query terms present in context
            cr = len(q_set & ctx_set) / (len(q_set) + 1)
            cr_scores.append(min(1.0, cr * 2))

            # Faithfulness: answer terms present in context
            if answer:
                a_set = set(term_re.findall(answer.lower()))
                f = len(a_set & ctx_set) / (len(a_set) + 1)
                f_scores.append(min(1.0, f * 1.5))
            else:
                f_scores.append(0.0)

            # Answer relevancy: answer terms match query terms
            if answer:
                a_set = set(term_re.findall(answer.lower()))
                ar = len(q_set & a_set) / (len(q_set) + 1)
                ar_scores.append(min(1.0, ar * 2))
            else:
                ar_scores.append(0.0)

        def avg(lst):
            return round(sum(lst) / len(lst), 4) if lst else 0.0

        cr_avg, f_avg, ar_avg = avg(cr_scores), avg(f_scores), avg(ar_scores)
        comp = (
            cr_avg * self.WEIGHTS["context_relevancy"]
            + f_avg * self.WEIGHTS["faithfulness"]
            + ar_avg * self.WEIGHTS["answer_relevancy"]
        )
        return RAGASResult(
            context_relevancy=cr_avg, faithfulness=f_avg,
            answer_relevancy=ar_avg,  context_recall=0.0,
            composite_score=round(comp, 4), method="custom",
            num_questions=len(test_set),
        )

    # ── Helpers ───────────────────────────────────────────────────────

    def _retrieve(self, company_name: str, question: str) -> list[str]:
        if self.retriever:
            try:
                hits = self.retriever.search(company_name, question, n_results=5)
                return [h["content"] for h in hits]
            except Exception:
                pass
        return []

    def _generate(self, question: str, contexts: list[str]) -> str:
        if self.llm and contexts:
            try:
                ctx_text = "\n\n".join(contexts[:3])
                return self.llm.generate(
                    f"Context:\n{ctx_text}\n\nQuestion: {question}\n\nAnswer:",
                    max_tokens=200,
                ) or ""
            except Exception:
                pass
        return ""

    def _save(self, company_name: str, result: RAGASResult) -> None:
        out_dir = DATA_DIR / "eval_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        slug = company_name.lower().replace(" ", "_").replace("/", "_")
        fname = out_dir / f"ragas_{slug}.json"
        with open(fname, "w") as f:
            json.dump(asdict(result), f, indent=2)
        logger.info(f"RAGAS results saved → {fname}")


# ── CLI entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation for a company")
    parser.add_argument("--company", required=True, help="Company name")
    parser.add_argument("--save", action="store_true", help="Save results to eval_results/")
    parser.add_argument("--method", choices=["ragas", "custom", "auto"], default="auto")
    args = parser.parse_args()

    evaluator = RAGASEvaluator()
    result = evaluator.evaluate(args.company, save_results=args.save)
    print(json.dumps(asdict(result), indent=2))
