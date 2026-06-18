"""
Utility Helpers — Shared across all modules
============================================

References
----------
Altman, E. I. (1968, 2000). "Financial Ratios, Discriminant Analysis and the
    Prediction of Corporate Bankruptcy." Journal of Finance, 23(4), 589-609.
    safe_divide() and classify_risk() used in all forensic score calculations to
    handle zero-denominator edge cases.

Standards
---------
ISA 520 / SA 520: Analytical Procedures — ratio calculations form the basis of
    most analytical review techniques; safe_divide prevents false alerts from zero
    denominators.

Role
----
Shared utility functions called across all 17 agents and reporting modules.
Functions: safe_divide(a, b, default=0.0) — division guard; classify_risk(score)
→ {band, color, description} — maps 0-100 score to verdict band;
format_currency(val, currency="INR") — INR/USD formatting with Cr/Bn suffixes;
pct_change(old, new) → float — signed percentage change; winsorize(val, low, high)
— outlier clamping for ratio calculations.
"""

from __future__ import annotations
import re
import unicodedata
from typing import Any, Optional
from fuzzywuzzy import fuzz


def normalize_company_name(name: str) -> str:
    """Normalize company name for consistent lookup."""
    name = unicodedata.normalize("NFKD", name)
    name = name.strip().upper()
    # Remove common suffixes for matching
    suffixes = [
        r"\bLTD\.?\b", r"\bLIMITED\b", r"\bINC\.?\b", r"\bCORP\.?\b",
        r"\bPLC\b", r"\bPVT\.?\b", r"\bPRIVATE\b", r"\bLLC\b", r"\bAG\b",
        r"\bSA\b", r"\bNV\b", r"\bBV\b", r"\bGMBH\b",
    ]
    for suffix in suffixes:
        name = re.sub(suffix, "", name, flags=re.IGNORECASE).strip()
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def fuzzy_match_company(query: str, candidates: list[str], threshold: int = 75) -> Optional[str]:
    """Find best fuzzy match for a company name."""
    query_norm = normalize_company_name(query)
    best_score = 0
    best_match = None
    for candidate in candidates:
        score = fuzz.token_sort_ratio(query_norm, normalize_company_name(candidate))
        if score > best_score and score >= threshold:
            best_score = score
            best_match = candidate
    return best_match


def format_currency(value: float, currency: str = "INR", billions: bool = False) -> str:
    """Format a number as currency string."""
    if value is None:
        return "N/A"
    if billions:
        if abs(value) >= 1_000_000:
            return f"{currency} {value/1_000_000:.2f}Tn"
        elif abs(value) >= 1_000:
            return f"{currency} {value/1_000:.2f}Bn"
        else:
            return f"{currency} {value:.2f}Mn"
    else:
        if abs(value) >= 1_00_00_00_000:  # Crore in INR
            return f"{currency} {value/1_00_00_000:.2f}Cr"
        elif abs(value) >= 1_00_000:
            return f"{currency} {value/1_00_000:.2f}L"
        else:
            return f"{currency} {value:,.0f}"


def calculate_percentage_change(current: float, previous: float) -> Optional[float]:
    """Calculate YoY percentage change safely."""
    if previous is None or current is None:
        return None
    if previous == 0:
        return None
    return ((current - previous) / abs(previous)) * 100


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division with zero-guard."""
    if denominator is None or denominator == 0:
        return default
    if numerator is None:
        return default
    return numerator / denominator


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten nested dict."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def extract_numbers_from_text(text: str) -> list[float]:
    """Extract numeric values from text (handles commas, crores, etc.)."""
    # Remove commas in numbers
    text = text.replace(",", "")
    # Match numbers including decimals and negative
    pattern = r"-?\d+\.?\d*"
    matches = re.findall(pattern, text)
    return [float(m) for m in matches if m]


def detect_currency_unit(text: str) -> tuple[str, float]:
    """Detect currency and scale factor from text."""
    text_lower = text.lower()
    if "crore" in text_lower or "cr" in text_lower:
        return "INR", 1e7
    elif "lakh" in text_lower:
        return "INR", 1e5
    elif "billion" in text_lower or "bn" in text_lower:
        return "USD", 1e9
    elif "million" in text_lower or "mn" in text_lower or "m" in text_lower:
        return "USD", 1e6
    elif "thousand" in text_lower:
        return "USD", 1e3
    return "USD", 1.0


def classify_risk(score: float) -> dict:
    """Classify a 0-100 risk score into band."""
    if score <= 20:
        return {"band": "VERY LOW RISK", "color": "#1b5e20", "emoji": "🟢"}
    elif score <= 40:
        return {"band": "LOW RISK", "color": "#388e3c", "emoji": "🟡"}
    elif score <= 60:
        return {"band": "MODERATE RISK", "color": "#f57c00", "emoji": "🟠"}
    elif score <= 80:
        return {"band": "HIGH RISK", "color": "#d32f2f", "emoji": "🔴"}
    else:
        return {"band": "EXTREME RISK", "color": "#b71c1c", "emoji": "🚨"}


def truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate text for display."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def parse_fiscal_year(text: str) -> Optional[str]:
    """Extract fiscal year from document text/filename."""
    patterns = [
        r"FY\s?(\d{2,4}[-–]\d{2,4})",
        r"FY\s?(\d{2,4})",
        r"(?:financial year|fiscal year)\s+(\d{4})",
        r"(\d{4})-(\d{2})",
        r"year ended (?:march|december|june|september)\s+\d{1,2},?\s*(\d{4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None
