"""
Output Harness — Structured extraction from LLM free-text outputs.

Provides:
  • Robust JSON extraction with three fallback strategies
  • Finding extraction from free-text (regex → scored paragraph NLP)
  • Output quality validation (length, company mention, quantitative content)
  • Re-attempt protocol for structurally invalid outputs
  • Numeric risk-score extraction from LLM text

Use OutputHarness.extract() after every LLM call to get a HarnessResult
with a typed findings list instead of a raw string.
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field


@dataclass
class ParsedFinding:
    """A finding extracted from an LLM response."""
    flag_type: str = "RED_FLAG"       # RED_FLAG | GREEN_FLAG | OBSERVATION
    risk_level: str = "MEDIUM"        # CRITICAL | HIGH | MEDIUM | LOW | POSITIVE
    title: str = ""
    detail: str = ""
    evidence: str = ""
    confidence: float = 0.65


@dataclass
class HarnessResult:
    """Structured output of OutputHarness.extract()."""
    raw_text: str = ""
    is_valid: bool = False
    findings: list[ParsedFinding] = field(default_factory=list)
    json_data: dict = field(default_factory=dict)
    parse_method: str = "none"        # json | regex | nlp | fallback
    quality_score: float = 0.0       # 0-1
    extracted_risk_score: float | None = None  # numeric score from LLM if present


# ── Severity keyword patterns ──────────────────────────────────────

_SEVERITY_MAP = {
    "CRITICAL": re.compile(
        r"\b(critical|extreme|severe|alarming|fraudulent|manipulation detected|material misstatement)\b",
        re.IGNORECASE,
    ),
    "HIGH": re.compile(
        r"\b(high.risk|significant concern|elevated risk|major|high risk|substantial)\b",
        re.IGNORECASE,
    ),
    "LOW": re.compile(
        r"\b(low.risk|minimal|minor|acceptable|adequate|within normal)\b",
        re.IGNORECASE,
    ),
    "POSITIVE": re.compile(
        r"\b(positive|strength|strong|excellent|clean opinion|healthy|well.managed)\b",
        re.IGNORECASE,
    ),
}

_FINDING_BOUNDARY = re.compile(
    r"(?:^|\n)(?:[\*\-•]\s+|\d+[\.\)]\s+|"
    r"(?:RED\s+FLAG|GREEN\s+FLAG|FINDING|RISK|CONCERN|ISSUE|ALERT|OBSERVATION)\s*[:：]?[ \t]*)",
    re.IGNORECASE | re.MULTILINE,
)

_RISK_SCORE_PATTERN = re.compile(
    r"(?:risk.score|score|overall.risk)[:\s]+(\d{1,3}(?:\.\d)?)\s*(?:/\s*100)?",
    re.IGNORECASE,
)

_RISK_KEYWORDS: list[tuple[str, float]] = [
    ("manipulation", 6.0), ("fraudulent", 6.0), ("material misstatement", 5.0),
    ("critical", 5.0), ("going concern", 5.0), ("material weakness", 5.0),
    ("restatement", 5.0), ("red flag", 3.0), ("high risk", 4.0),
    ("concern", 2.0), ("elevated", 2.0), ("declining", 2.0), ("anomaly", 3.0),
    ("discrepancy", 3.0), ("inflated", 3.0), ("overstated", 3.0),
    ("healthy", -3.0), ("low risk", -3.0), ("clean audit", -3.0),
    ("strong", -2.0), ("well managed", -2.0),
]


class OutputHarness:
    """
    Structured extraction engine for LLM outputs.

    Pipeline for each extract() call:
      1. Validate length + basic sanity checks
      2. Try JSON extraction (3 strategies)
      3. Fall back to regex finding parser
      4. Fall back to scored paragraph NLP
      5. Always extract numeric risk score if present in text
    """

    _MIN_WORDS = 25
    _MAX_TITLE = 110

    # ── Public API ─────────────────────────────────────────────────

    def extract(
        self,
        raw_text: str,
        company_name: str = "",
        expected_keywords: list[str] | None = None,
    ) -> HarnessResult:
        result = HarnessResult(raw_text=raw_text)
        if not self._validate(raw_text):
            return result
        result.is_valid = True
        result.quality_score = self._quality(raw_text, company_name, expected_keywords)
        result.extracted_risk_score = self._extract_numeric_risk(raw_text)

        # Strategy 1 — JSON
        jdata = self._parse_json(raw_text)
        if jdata and isinstance(jdata.get("findings"), list):
            result.json_data = jdata
            result.findings = self._findings_from_json(jdata["findings"])
            result.parse_method = "json"
            if "risk_score" in jdata and result.extracted_risk_score is None:
                try:
                    result.extracted_risk_score = float(jdata["risk_score"])
                except (ValueError, TypeError):
                    pass
            return result

        # Strategy 2 — Regex boundary parser
        findings = self._regex_extract(raw_text)
        if findings:
            result.findings = findings
            result.parse_method = "regex"
            return result

        # Strategy 3 — Scored paragraph NLP
        result.findings = self._nlp_extract(raw_text)
        result.parse_method = "nlp"
        return result

    def estimate_risk_score(self, raw_text: str, base: float = 50.0) -> float:
        """
        Extract numeric risk score from LLM output; fall back to keyword
        density scoring. Returns a value in [10, 95].
        """
        numeric = self._extract_numeric_risk(raw_text)
        if numeric is not None:
            return max(10.0, min(95.0, numeric))
        score = base
        text_lower = raw_text.lower()
        for keyword, weight in _RISK_KEYWORDS:
            count = text_lower.count(keyword)
            if count:
                score += weight * min(count, 5)
        return max(10.0, min(95.0, score))

    @staticmethod
    def structured_output_suffix() -> str:
        """
        Append this to any prompt to request structured JSON output.
        All agents call this so the LLM knows the expected schema.
        """
        return """

REQUIRED OUTPUT FORMAT — return ONLY valid JSON (no markdown fences, no prose outside the JSON):
{
  "summary": "<2-3 sentence executive summary>",
  "risk_score": <float 0-100, higher = more risk>,
  "findings": [
    {
      "flag_type": "RED_FLAG|GREEN_FLAG|OBSERVATION",
      "risk_level": "CRITICAL|HIGH|MEDIUM|LOW|POSITIVE",
      "title": "<concise finding title, max 100 chars>",
      "detail": "<detailed forensic explanation>",
      "evidence": "<specific numbers, ratios, or quoted text>",
      "confidence": <float 0.0-1.0>
    }
  ],
  "recommendation": "AVOID|CAUTION|MONITOR|ACCEPTABLE"
}"""

    # ── Validation ─────────────────────────────────────────────────

    def _validate(self, text: str) -> bool:
        return bool(text and len(text.split()) >= self._MIN_WORDS)

    def _quality(
        self, text: str, company: str, keywords: list[str] | None
    ) -> float:
        score = 0.4
        if company and company.lower().split()[0] in text.lower():
            score += 0.15
        if re.search(r"\d+(?:\.\d+)?%|\d+x\b|[A-Z]-Score[:\s]+[-\d.]", text):
            score += 0.15
        if re.search(r"\b(critical|high.risk|red.flag|concern|elevated)\b", text, re.IGNORECASE):
            score += 0.10
        if len(text.split()) > 300:
            score += 0.10
        if keywords:
            hit = sum(1 for kw in keywords if kw.lower() in text.lower())
            score += 0.10 * (hit / len(keywords))
        return min(1.0, score)

    # ── Numeric Score Extraction ────────────────────────────────────

    @staticmethod
    def _extract_numeric_risk(text: str) -> float | None:
        m = _RISK_SCORE_PATTERN.search(text)
        if m:
            try:
                val = float(m.group(1))
                if 0 <= val <= 100:
                    return val
            except ValueError:
                pass
        return None

    # ── JSON Parsing ───────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> dict:
        clean = re.sub(r"```(?:json)?\s*|```", "", text).strip()
        for candidate in (clean, text):
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, ValueError):
                pass
        # Extract outermost {...}
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                pass
        return {}

    def _findings_from_json(self, items: list) -> list[ParsedFinding]:
        out: list[ParsedFinding] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            f = ParsedFinding(
                flag_type=str(item.get("flag_type", "RED_FLAG")).upper(),
                risk_level=str(item.get("risk_level", "MEDIUM")).upper(),
                title=str(item.get("title", item.get("summary", "")))[:self._MAX_TITLE],
                detail=str(item.get("detail", item.get("description", ""))),
                evidence=str(item.get("evidence", item.get("data", ""))),
                confidence=float(item.get("confidence", 0.70)),
            )
            if f.title or f.detail:
                out.append(f)
        return out

    # ── Regex Finding Parser ────────────────────────────────────────

    def _regex_extract(self, text: str) -> list[ParsedFinding]:
        segments = _FINDING_BOUNDARY.split(text)
        findings: list[ParsedFinding] = []
        for seg in segments:
            seg = seg.strip()
            if len(seg) < 30:
                continue
            flag_type = "GREEN_FLAG" if re.search(
                r"\b(green.flag|positive|strength|clean)\b", seg, re.IGNORECASE
            ) else "RED_FLAG"
            risk_level = "MEDIUM"
            for lvl, pat in _SEVERITY_MAP.items():
                if pat.search(seg):
                    risk_level = lvl
                    break
            lines = seg.strip().splitlines()
            title = lines[0].strip()[:self._MAX_TITLE] if lines else seg[:80]
            detail = " ".join(l.strip() for l in lines[1:]).strip() if len(lines) > 1 else ""
            ev_parts = re.findall(
                r"(?:\d+(?:\.\d+)?%|\d+x\b|Rs\.?\s*[\d,]+M?|[A-Z]+-Score[:\s]+[-\d.]+)",
                seg,
            )
            evidence = " | ".join(ev_parts[:4])
            findings.append(ParsedFinding(
                flag_type=flag_type,
                risk_level=risk_level,
                title=title,
                detail=detail,
                evidence=evidence,
                confidence=0.65,
            ))
        return findings[:12]

    # ── Scored Paragraph NLP ───────────────────────────────────────

    def _nlp_extract(self, text: str) -> list[ParsedFinding]:
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        scored: list[tuple[float, str]] = []
        for para in paragraphs:
            if len(para.split()) < 8:
                continue
            s = self._para_score(para)
            if s > 0:
                scored.append((s, para))
        scored.sort(reverse=True)
        findings: list[ParsedFinding] = []
        for score, para in scored[:8]:
            if score >= 6:
                risk_level = "CRITICAL"
            elif score >= 4:
                risk_level = "HIGH"
            elif score >= 2:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            flag_type = "GREEN_FLAG" if re.search(
                r"\b(positive|strength|clean|strong|healthy)\b", para, re.IGNORECASE
            ) else "RED_FLAG"
            sents = re.split(r"(?<=[.!?])\s+", para)
            title = sents[0][:self._MAX_TITLE] if sents else para[:80]
            findings.append(ParsedFinding(
                flag_type=flag_type,
                risk_level=risk_level,
                title=title,
                detail=para,
                confidence=min(0.5 + score * 0.05, 0.85),
            ))
        return findings

    @staticmethod
    def _para_score(text: str) -> float:
        score = 0.0
        text_lower = text.lower()
        for kw, w in _RISK_KEYWORDS:
            if kw in text_lower:
                score += abs(w)
        score += min(len(re.findall(r"\d+(?:\.\d+)?%|\d+x\b|[A-Z]+-Score", text)), 4) * 1.5
        return score
