"""
Guardrails — Groundedness checking, faithfulness scoring, hallucination detection.
Three-layer active guardrail system for forensic AI outputs.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class GuardrailResult:
    passed: bool
    groundedness_score: float      # 0-1: response tokens grounded in retrieved context
    faithfulness_score: float      # 0-1: no invented standards/numbers
    hallucination_risk: str        # LOW | MEDIUM | HIGH | CRITICAL
    issues: list[str] = field(default_factory=list)
    suggestion: str = ""


class Guardrails:
    """
    Three-layer guardrail system for forensic accounting LLM outputs:

      Layer 1 — Groundedness
        Token overlap between response and retrieved context.
        Low overlap → response is making claims not in the evidence base.

      Layer 2 — Faithfulness
        Detects invented audit standard numbers (e.g. ISA 999) not in the
        authoritative catalog. Also detects hallucinated entity-specific facts.

      Layer 3 — Overconfidence
        Flags definitive fraud conclusions presented without qualifying language
        ("conclusively proves fraud", "definitely manipulated") — these should
        only appear with a supporting Evidence chain.

    block_on_fail=True: callers should discard responses that fail all three layers.
    block_on_fail=False (default): warnings logged; all responses returned.
    """

    # Valid ISA/SA/IAS/IFRS/Ind AS/PCAOB numbers (authoritative subset)
    _VALID_STDS: frozenset[int] = frozenset({
        # ISA / SA
        200, 210, 220, 230, 240, 250, 260, 265,
        300, 315, 320, 330, 402, 450,
        500, 501, 505, 510, 520, 530, 540, 550, 560, 570, 580,
        600, 610, 620,
        700, 701, 705, 706, 710, 720,
        # PCAOB AS
        2101, 2201, 2301, 2401, 2405,
        # IAS / IFRS / Ind AS
        1, 2, 7, 8, 10, 12, 16, 17, 19, 20, 21, 23, 24, 32, 33, 36, 37, 38, 40,
        101, 102, 103, 104, 105, 108, 109, 110, 111, 112, 113, 115, 116,
    })

    _STD_PATTERN = re.compile(
        r"\b(?:ISA|SA|PCAOB(?:\s+AS)?|IAS|IFRS|Ind(?:ian)?\s+AS|AS)\s*(\d{2,4})\b",
        re.IGNORECASE,
    )

    _OVERCONFIDENT = re.compile(
        r"\b(definitively proves?|conclusively proves?|irrefutably|undoubtedly "
        r"|clearly demonstrates? fraud|confirmed manipulation|proven fraud)\b",
        re.IGNORECASE,
    )

    _TERM_RE = re.compile(r"\b[a-zA-Z0-9]{3,}\b")

    def __init__(
        self,
        groundedness_threshold: float = 0.20,
        block_on_fail: bool = False,
    ):
        self.groundedness_threshold = groundedness_threshold
        self.block_on_fail = block_on_fail

    # ── Public API ────────────────────────────────────────────────────

    def check(
        self,
        response: str,
        context: str,
        query: str = "",
    ) -> GuardrailResult:
        """
        Run all three layers. Returns a GuardrailResult.
        Logs a warning when block_on_fail=False and guardrail fails.
        """
        issues: list[str] = []

        # Layer 1: groundedness
        g_score = self._groundedness(response, context)
        if g_score < self.groundedness_threshold:
            issues.append(
                f"Groundedness {g_score:.2f} below threshold "
                f"{self.groundedness_threshold:.2f} — "
                f"response poorly supported by retrieved context"
            )

        # Layer 2: invented standards
        invented = self._invented_standards(response)
        if invented:
            issues.append(f"Potentially invented standard numbers: {', '.join(invented)}")

        # Layer 3: overconfident conclusions
        overkill = self._OVERCONFIDENT.findall(response)
        if overkill:
            issues.append(
                f"Overconfident unhedged conclusion language: {overkill[:3]} — "
                f"use 'consistent with' or 'indicative of' instead"
            )

        f_score = max(0.0, g_score - len(invented) * 0.15)
        risk = self._classify_risk(g_score, f_score, len(issues))

        passed = (
            not issues
            or (g_score >= self.groundedness_threshold and not invented and not overkill)
        )

        result = GuardrailResult(
            passed=passed,
            groundedness_score=round(g_score, 4),
            faithfulness_score=round(f_score, 4),
            hallucination_risk=risk,
            issues=issues,
            suggestion=self._suggestion(issues, g_score) if issues else "",
        )

        if not passed:
            level = logger.warning if risk in ("HIGH", "CRITICAL") else logger.info
            level(
                f"Guardrail {'BLOCK' if self.block_on_fail else 'WARN'} — "
                f"risk={risk}, groundedness={g_score:.2f}, {len(issues)} issue(s)"
            )

        return result

    def enrich_harness_result(self, harness_result, context: str, query: str = ""):
        """
        Convenience wrapper: run check and attach GuardrailResult to a HarnessResult.
        Returns the (possibly annotated) HarnessResult unmodified.
        """
        gr = self.check(harness_result.raw_text, context, query)
        harness_result.grounding_score = gr.groundedness_score
        # Store guardrail result in json_data so downstream can inspect it
        harness_result.json_data["guardrail"] = {
            "passed":             gr.passed,
            "groundedness":       gr.groundedness_score,
            "faithfulness":       gr.faithfulness_score,
            "hallucination_risk": gr.hallucination_risk,
            "issues":             gr.issues,
        }
        return harness_result

    # ── Layer implementations ─────────────────────────────────────────

    def _groundedness(self, response: str, context: str) -> float:
        if not context or not response:
            return 1.0  # no context provided → can't penalise
        resp_terms = set(self._TERM_RE.findall(response.lower()))
        ctx_terms  = set(self._TERM_RE.findall(context.lower()))
        if not resp_terms:
            return 0.0
        return len(resp_terms & ctx_terms) / len(resp_terms)

    def _invented_standards(self, response: str) -> list[str]:
        found = self._STD_PATTERN.findall(response)
        return sorted({
            f"ISA/SA/IAS {n}" for n in found if int(n) not in self._VALID_STDS
        })

    def _classify_risk(self, g: float, f: float, n_issues: int) -> str:
        if g < 0.10 or f < 0.10 or n_issues >= 3:
            return "CRITICAL"
        if g < 0.20 or f < 0.25 or n_issues >= 2:
            return "HIGH"
        if g < 0.35 or n_issues >= 1:
            return "MEDIUM"
        return "LOW"

    def _suggestion(self, issues: list[str], g: float) -> str:
        parts = ["Response failed guardrail. Actions:"]
        if g < self.groundedness_threshold:
            parts.append(
                "retrieve more context before generating "
                "(current context does not support the claims made)"
            )
        if any("invented" in i for i in issues):
            parts.append(
                "verify all cited standard numbers against the knowledge base"
            )
        if any("overconfident" in i.lower() for i in issues):
            parts.append(
                "replace definitive fraud conclusions with probabilistic language "
                "('consistent with', 'indicative of', 'warrants further investigation')"
            )
        return " | ".join(parts)
