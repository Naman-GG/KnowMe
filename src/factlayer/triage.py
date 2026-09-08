"""Page triage: decide which pages are worth spending an LLM call on.

The starter corpus is 511 pages. Extracting from all of them on a free-tier API
is slow and mostly wasted -- tables of contents, boilerplate and signature
blocks carry no facts. Triage scores every page on *generic* fact-density
signals and spends the budget on the densest pages first.

The signals are deliberately document-agnostic. There is no rule here about
logistics companies or central banks; a page scores well because it contains
numbers attached to units and time expressions, which is what a factual claim
looks like in any domain.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .ingest import Document, Page

# Numbers that look like measurements rather than list indices or page numbers.
_NUMERIC = re.compile(r"\d[\d,]*\.?\d*")
_UNIT_TOKENS = re.compile(
    r"(₹|\$|€|£|%|\bper\s?cent\b|\bpercent\b|\bcrore\b|\blakh\b|\bmillion\b|\bbillion\b"
    r"|\btrillion\b|\bmn\b|\bbn\b|\bcr\b|\brs\.?\b|\binr\b|\busd\b|\btons?\b|\btonnes?\b"
    r"|\bkg\b|\bgw\b|\bbps\b)",
    re.IGNORECASE,
)
_PERIOD_TOKENS = re.compile(
    r"(\bfy\s?\d{2,4}\b|\bq[1-4]\b|\b(?:19|20)\d{2}\b|\byear ended\b|\bquarter ended\b"
    r"|\bas (?:at|of)\b|\bhalf[- ]year\b|\bh[12]\b|\bfiscal\b|\bas on\b)",
    re.IGNORECASE,
)
# Statements about people/roles/places -- non-numeric facts worth extracting.
_ENTITY_TOKENS = re.compile(
    r"(\bdirector\b|\bchairman\b|\bchief \w+ officer\b|\bappointed\b|\bresigned\b"
    r"|\bceased\b|\bregistered office\b|\bincorporat\w+\b|\bcin\b|\bdin\b|\bsecretary\b)",
    re.IGNORECASE,
)
# Verbs that mark an assertion being made. These carry the narrative facts --
# a forecast, an appointment, a restatement -- that pure digit-density misses.
_ASSERTION_TOKENS = re.compile(
    r"(\bprojected\b|\bexpected\b|\bestimated?\b|\bforecast\w*\b|\bstood at\b"
    r"|\bincreased?\b|\bdeclined?\b|\bgrew\b|\brecorded\b|\breported\b|\brose\b"
    r"|\bwas\b|\bis\b|\bamounted to\b|\bregistering\b|\bcompared (?:to|with)\b"
    r"|\brevised\b|\brestated\b|\bassessed\b|\bplaced at\b)",
    re.IGNORECASE,
)

# Pages that are almost always noise.
_BOILERPLATE = re.compile(
    r"(table of contents|^\s*contents\s*$|this page (?:has been |is )?intentionally left blank)",
    re.IGNORECASE | re.MULTILINE,
)


def _log_sat(count: int, half: float) -> float:
    """Map a token count to 0..1, reaching ~0.5 at `half` and saturating after.

    Logarithmic rather than linear so that a page with 200 numbers does not
    score eight times a page with 25 -- past a point, more digits on a page say
    nothing more about whether it holds a fact.
    """
    if count <= 0:
        return 0.0
    return min(math.log1p(count) / math.log1p(half * 4), 1.0)


@dataclass
class PageScore:
    page_no: int
    score: float
    reasons: dict[str, float]

    @property
    def explain(self) -> str:
        parts = [f"{k}={v:.2f}" for k, v in self.reasons.items() if v]
        return ", ".join(parts) or "no signal"


def score_page(page: Page) -> PageScore:
    text = page.text
    n = max(len(text), 1)
    if n < 120:
        return PageScore(page.page_no, 0.0, {"too_short": 0.0})

    numbers = _NUMERIC.findall(text)
    # Each signal is normalised to roughly 0..1 so the weights stay readable.
    #
    # Numeric density uses a log curve rather than a linear one. An earlier
    # linear version let dense financial tables saturate the score and crowd
    # narrative pages out of the budget entirely -- the page stating "growth in
    # FY26 would be between 6.3 and 6.8 per cent" scored 1.58 against a table's
    # 8.00, and was dropped. Facts stated in sentences matter as much as facts
    # stated in grids, so digit count is deliberately made to saturate early.
    reasons = {
        "numbers": _log_sat(len(numbers), 25.0) * 2.0,
        "units": _log_sat(len(_UNIT_TOKENS.findall(text)), 8.0) * 2.0,
        "periods": _log_sat(len(_PERIOD_TOKENS.findall(text)), 5.0) * 2.0,
        "assertions": _log_sat(len(_ASSERTION_TOKENS.findall(text)), 6.0) * 2.0,
        "entities": _log_sat(len(_ENTITY_TOKENS.findall(text)), 3.0) * 1.5,
        # Prose still matters: a page of only digits is usually an index.
        "prose": min(len(text.split()) / 250.0, 1.0) * 0.5,
    }
    if _BOILERPLATE.search(text):
        reasons["boilerplate"] = -3.0

    return PageScore(page.page_no, round(sum(reasons.values()), 3), reasons)


def triage(doc: Document, budget: int | None = None, min_score: float = 1.5) -> list[PageScore]:
    """Rank a document's pages and return those we intend to extract from.

    By default there is no budget: triage drops pages that look empty of facts
    and keeps everything else. That is deliberate. Its job is to skip covers,
    contents pages and blank leaves, not to ration genuinely informative ones.

    An earlier version capped each document at the 40 highest-scoring pages.
    That reliably lost facts -- a 100-page filing with a long statistical annex
    has far more than 40 pages worth reading, and the dense tables crowded out
    the narrative pages carrying the growth forecasts. A stratified variant that
    reserved budget for each region of the document scored worse still, because
    the bands spent their allocation on mediocre pages.

    `budget` therefore remains available for very large corpora or quick demos,
    but it is opt-in and the caller is told what it cost via `coverage()`.
    """
    scores = [score_page(p) for p in doc.pages]
    eligible = [s for s in scores if s.score >= min_score]
    if budget is not None and len(eligible) > budget:
        eligible = sorted(eligible, key=lambda s: s.score, reverse=True)[:budget]
    return sorted(eligible, key=lambda s: s.page_no)


def coverage(doc: Document, selected: list[PageScore]) -> dict[str, float | int]:
    """Report what we chose to skip, so the honesty is measurable."""
    sel = {s.page_no for s in selected}
    skipped_chars = sum(p.char_count for p in doc.pages if p.page_no not in sel)
    total_chars = sum(p.char_count for p in doc.pages) or 1
    return {
        "pages_total": doc.page_count,
        "pages_selected": len(sel),
        "chars_total": total_chars,
        "chars_skipped": skipped_chars,
        "char_coverage": round(1 - skipped_chars / total_chars, 4),
    }
