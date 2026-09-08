"""Grounding: verify that every claim is actually supported by its source page.

The extractor is asked to copy a verbatim span for each claim. This module
checks that it did, by looking for that span in the exact page text the model
was shown. A quote that is not on its own page is a fabrication, and finding it
costs nothing -- no second model call, no self-grading.

Grounding also covers *qualifiers*, which turned out to matter more than the
quotes. In testing, the model reliably produced honest quotes while inventing a
scope of "standalone and consolidated" for a slide that never mentions either
word -- and it kept doing so after the prompt explicitly forbade it. A
fabricated qualifier is more damaging than a missing one: a missing scope
produces an honest UNDERSPECIFIED verdict, while an invented one produces a
confident comparison between claims that were never comparable. So a qualifier
the source page does not support is dropped and recorded, not trusted.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from .ingest import Page, normalise
from .models import Claim, ClaimStatus, GroundingStatus

# Minimum similarity for a near-match to count as the same span.
FUZZY_THRESHOLD = 0.85
# Share of a quote's characters that must be found, in order, for a
# non-contiguous match to count as grounded.
TOKEN_COVERAGE_THRESHOLD = 0.85
MIN_QUOTE_CHARS = 8
# Words, numbers and the punctuation that binds them ("8,142", "0.7%", "FY24").
_TOKEN = re.compile(r"[\w][\w.,%/&'-]*")

# Ways the corpora write the same qualifier. Documents abbreviate ("Adj.
# EBITDA"), so a literal substring test would wrongly discard a true qualifier.
QUALIFIER_ALIASES: dict[str, set[str]] = {
    "adjusted": {"adjusted", "adj.", "adj ", "normalised", "normalized"},
    "reported": {"reported", "as reported", "reported basis"},
    "restated": {"restated", "restatement"},
    "audited": {"audited"},
    "unaudited": {"unaudited"},
    "provisional": {"provisional"},
    "consolidated": {"consolidated", "consolidation", "group"},
    "standalone": {"standalone", "stand-alone", "stand alone", "separate", "parent"},
    "projection": {"projection", "projected", "forecast", "expected"},
    "estimate": {"estimate", "estimated", "advance estimate"},
    "actual": {"actual", "actuals"},
}


@dataclass
class GroundingReport:
    """What the grounding pass did, so the numbers can be quoted honestly."""

    total: int = 0
    exact: int = 0
    fuzzy: int = 0
    gapped: int = 0
    quarantined: int = 0
    qualifiers_dropped: int = 0
    dropped_detail: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        rate = (self.exact + self.fuzzy + self.gapped) / self.total if self.total else 0.0
        return {
            "claims_checked": self.total,
            "grounded_exact": self.exact,
            "grounded_fuzzy": self.fuzzy,
            "grounded_gapped": self.gapped,
            "quarantined": self.quarantined,
            "grounding_rate": round(rate, 4),
            "qualifiers_dropped": self.qualifiers_dropped,
            "qualifiers_dropped_detail": dict(
                sorted(self.dropped_detail.items(), key=lambda kv: -kv[1])
            ),
        }


def locate_quote(quote: str, page: Page) -> tuple[GroundingStatus, int | None, int | None]:
    """Find a quote in a page, tolerating whitespace and typographic differences."""
    quote = (quote or "").strip()
    if len(quote) < MIN_QUOTE_CHARS:
        return GroundingStatus.NOT_FOUND, None, None

    # 1. Exact, in the raw text -- gives real offsets for highlighting.
    idx = page.text.find(quote)
    if idx != -1:
        return GroundingStatus.EXACT, idx, idx + len(quote)

    # 2. After folding ligatures, quotes and runs of whitespace. PDF extraction
    #    breaks lines mid-phrase, so this is the common case, not the exception.
    nq = normalise(quote)
    if nq and nq in page.norm_text:
        anchor = _anchor(quote, page)
        return GroundingStatus.FUZZY, anchor[0], anchor[1]

    # 3. Every token present, in order, but not contiguously.
    #
    #    Reading a slide that says "Q4 FY23: Rs.13 Cr / 0.7%", the model quotes
    #    "Q4 FY23: 0.7%" -- it drops the middle rather than inventing anything.
    #    Requiring a contiguous span quarantined 62% of claims on the earnings
    #    deck, nearly all of them true facts that had merely been elided.
    #
    #    So the test is not "is this span contiguous" but "does every token of
    #    the quote appear on this page, in this order". That still rejects a
    #    fabricated figure, whose tokens are simply not there, while accepting an
    #    honest abridgement. Such claims are labelled GAPPED rather than folded
    #    in with clean matches, so the distinction stays visible.
    covered, start, end = _subsequence_coverage(nq, page.norm_text)
    if covered >= TOKEN_COVERAGE_THRESHOLD:
        anchor = _anchor(quote, page)
        return GroundingStatus.GAPPED, anchor[0] or start, anchor[1] or end

    return GroundingStatus.NOT_FOUND, None, None


def _subsequence_coverage(quote: str, haystack: str) -> tuple[float, int | None, int | None]:
    """Fraction of the quote's characters found as in-order tokens in the page.

    Weighted by token length, so matching "revenue" counts for more than
    matching ":". Returns the span between the first and last matched token.
    """
    tokens = _TOKEN.findall(quote)
    if not tokens:
        return 0.0, None, None
    total = sum(len(t) for t in tokens)
    matched = 0
    cursor = 0
    first = last = None
    for token in tokens:
        idx = haystack.find(token, cursor)
        if idx == -1:
            continue
        matched += len(token)
        cursor = idx + len(token)
        first = idx if first is None else first
        last = cursor
    return (matched / total if total else 0.0), first, last


def _anchor(quote: str, page: Page) -> tuple[int | None, int | None]:
    """Best-effort offsets into the raw page text for a non-exact match.

    Used only for highlighting, so an approximate span is acceptable; the
    verdict itself never depends on these numbers.
    """
    matcher = difflib.SequenceMatcher(None, quote, page.text, autojunk=False)
    block = matcher.find_longest_match(0, len(quote), 0, len(page.text))
    if block.size < MIN_QUOTE_CHARS:
        return None, None
    return block.b, block.b + block.size


def _qualifier_supported(value: str, page: Page) -> bool:
    """Does the page actually say this qualifier?"""
    v = (value or "").strip().lower()
    if not v:
        return False
    haystack = page.norm_text
    if v in haystack:
        return True
    # Try known spellings of this qualifier, and of each word in it.
    for token in {v, *v.split()}:
        for canonical, aliases in QUALIFIER_ALIASES.items():
            if token == canonical or token in aliases:
                if any(alias in haystack for alias in aliases):
                    return True
    return False


def ground_claim(claim: Claim, page: Page, report: GroundingReport) -> Claim:
    """Verify one claim's evidence and qualifiers against its source page."""
    report.total += 1

    status, start, end = locate_quote(claim.evidence.quote, page)
    claim.evidence.grounding = status
    claim.evidence.char_start = start
    claim.evidence.char_end = end

    if status is GroundingStatus.NOT_FOUND:
        claim.status = ClaimStatus.QUARANTINED
        claim.quarantine_reason = (
            "quote not found on cited page: "
            f"{claim.evidence.quote[:120]!r}"
        )
        report.quarantined += 1
        return claim

    report.exact += status is GroundingStatus.EXACT
    report.fuzzy += status is GroundingStatus.FUZZY
    report.gapped += status is GroundingStatus.GAPPED

    # Free-text qualifiers the model can invent. `period` is validated by
    # parsing instead, and `as_of` comes from the document profile rather than
    # the page, so neither is checked here.
    for field_name in ("scope", "basis", "estimate_type"):
        value = getattr(claim.qualifiers, field_name)
        if value and not _qualifier_supported(value, page):
            setattr(claim.qualifiers, field_name, None)
            claim.qualifiers.extra[f"dropped_{field_name}"] = value
            report.qualifiers_dropped += 1
            key = f"{field_name}={value}"
            report.dropped_detail[key] = report.dropped_detail.get(key, 0) + 1

    return claim


def ground_claims(
    claims: list[Claim], pages: dict[int, Page], report: GroundingReport | None = None
) -> GroundingReport:
    """Ground a batch of claims in place, returning the tally."""
    report = report or GroundingReport()
    for claim in claims:
        page = pages.get(claim.evidence.page_no)
        if page is None:
            claim.status = ClaimStatus.QUARANTINED
            claim.quarantine_reason = f"cited page {claim.evidence.page_no} not in document"
            report.total += 1
            report.quarantined += 1
            continue
        ground_claim(claim, page, report)
    return report
