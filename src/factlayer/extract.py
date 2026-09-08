"""Claim extraction: pages of text become structured, qualified claims.

The model is given one narrow job -- read carefully and copy out what the page
says, including the context that makes each figure mean something. It is not
asked whether facts agree, whether a number is plausible, or what a measure
"really" means. Those judgements happen later, in code that can show its
working.

Two mechanisms carry most of the weight:

* **Document profiling.** One call per document establishes the publisher, the
  date it speaks as of, and its default reporting conventions. Page-level
  claims inherit those defaults, so a table that says "81,415.38" without
  repeating "consolidated, INR million, FY24" on every row still yields fully
  qualified claims. Inheritance is recorded, never silently assumed.

* **Mandatory verbatim quotes.** Every claim must carry a span copied
  character-for-character from the page. That is what makes grounding possible:
  a quote that cannot be found in its own source page is a fabrication, and we
  can detect it without asking the model to grade itself.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from .ingest import Document, Page
from .llm import LLMClient, LLMError
from .models import (
    Claim,
    ClaimStatus,
    Evidence,
    GroundingStatus,
    Period,
    Qualifiers,
    Quantity,
    ValueKind,
)
from .normalize.periods import parse_period
from .normalize.units import canonicalise, parse_unit
from .triage import PageScore, triage

log = logging.getLogger(__name__)

MAX_CLAIMS_PER_PAGE = 20
MAX_PAGE_CHARS = 9000


PROFILE_SYSTEM = """\
You read the opening pages of a document and report what kind of document it is \
and the reporting conventions it uses. Reply with JSON only."""

PROFILE_USER = """\
Below are the opening pages of a document (filename: {filename}).

Return JSON with exactly these keys:
{{
  "publisher": string | null,        // organisation that issued it
  "subject": string | null,          // the main entity it is about
  "doc_type": string | null,         // e.g. annual report, prospectus, staff report
  "as_of_date": "YYYY-MM-DD" | null, // the date the document speaks as of
  "default_period": string | null,   // main reporting period, e.g. "FY24", "2024-25"
  "default_scope": string | null,    // e.g. "consolidated", "standalone", null
  "default_unit": string | null,     // e.g. "INR million", "Rs. Cr", "per cent"
  "default_currency": string | null  // e.g. "INR", "USD"
}}

Rules:
- Use only what the text states or clearly shows. Use null when it is not stated.
- as_of_date is when the document was issued or the date its data runs to.
  If only a month or year is given, use the last day of that month or year.
- Do not guess a scope. Many documents never state one.

TEXT:
{text}
"""


EXTRACT_SYSTEM = """\
You extract factual claims from one page of a document. You copy what the page \
says; you never infer, calculate, or combine figures. Reply with JSON only."""

EXTRACT_USER = """\
Extract the factual claims stated on this page.

Document context (claims inherit these unless the page says otherwise):
- subject: {subject}
- publisher: {publisher}
- default period: {default_period}
- default scope: {default_scope}
- default unit: {default_unit}

Return JSON: {{"claims": [ ... ]}} with at most {max_claims} claims, each:
{{
  "subject":     string,            // what the claim is about
  "measure":     string,            // what is measured, in the page's own words
  "value_kind":  "number" | "range" | "text" | "date" | "boolean",
  "value":       number | null,     // the figure; low end if a range
  "value_high":  number | null,     // high end if a range, else null
  "text_value":  string | null,     // for value_kind "text"
  "date_value":  "YYYY-MM-DD"|null, // for value_kind "date"
  "bool_value":  true|false|null,   // for value_kind "boolean"
  "unit":        string | null,     // exactly as written: "Rs. Cr", "per cent"
  "period":      string | null,     // exactly as written: "FY24", "Q4 FY24"
  "scope":       string | null,     // "consolidated", "standalone", a segment name
  "basis":       string | null,     // "reported", "adjusted", "restated", "audited"
  "estimate_type": string | null,   // "actual", "provisional", "estimate", "projection"
  "quote":       string,            // VERBATIM span from the page, 10-300 chars
  "confidence":  number             // 0.0-1.0, how sure you are of this reading
}}

Decompose each claim into its parts. This matters more than anything else:
the measure name must be the *bare* quantity, with every qualifier lifted out
into its own field.

  Page text:  "FY24 Adjusted EBITDA: Rs. 76 Cr (consolidated)"
  Correct:    measure "EBITDA", basis "adjusted", period "FY24",
              scope "consolidated", unit "Rs. Cr", value 76
  Wrong:      measure "FY24 Adjusted EBITDA (consolidated)", basis null

  Page text:  "Revenue from operations on a standalone basis for FY24 stood at
               Rs. 74,540.82 million"
  Correct:    measure "revenue from operations", scope "standalone",
              period "FY24", unit "INR million", value 74540.82

Hard rules:
- NEVER put a period in "measure". "FY24 revenue" is wrong; measure is
  "revenue" and period is "FY24". The same measure recurs across years and must
  read identically each time.
- NEVER put a basis word in "measure". If the page says adjusted, reported,
  restated, normalised, audited, provisional or like-for-like, that word belongs
  in "basis" and must be removed from the measure name.
- NEVER put a scope in "measure". Consolidated, standalone, a segment name or a
  region belongs in "scope".
- "scope" must be a SINGLE slice actually named on this page, copied in the
  page's own words. If the page does not name one, use null. Never combine two
  scopes ("standalone and consolidated" is not a scope), and never infer one
  from the kind of document this is.
- Keep the page's own vocabulary for the bare measure. Do not rename "total
  income" to "revenue" or "revenue from services" to "revenue" -- those are
  genuinely different measures and the difference is the point.
- "quote" MUST be copied character-for-character from the page text below.
  Do not tidy spacing, expand abbreviations, or join separated lines.
  A claim whose quote is not on the page will be discarded.
- Write numbers as plain numerals: 8142, not "8,142" or "8142 Cr".
  Put the scale word in "unit" instead.
- Leave a qualifier null when the page does not state it. Never invent a period,
  scope or basis to fill a gap -- a missing qualifier is useful information.
- If a figure is a projection or forecast, set estimate_type accordingly.
- Only inherit the document's default period for figures that actually cover a
  period. A cumulative or since-inception total, or a standing description, has
  no period: leave it null.
- Prefer claims that a reader would consider a fact worth checking. Skip
  headings, page furniture, and narrative with no measurable content.
- Extract non-numeric facts too: who holds a role, where an office is
  registered, when someone was appointed or resigned.

PAGE {page_no} TEXT:
{text}
"""


@dataclass
class DocProfile:
    """Document-level context that page claims inherit."""

    publisher: str | None = None
    subject: str | None = None
    doc_type: str | None = None
    as_of_date: dt.date | None = None
    default_period: str | None = None
    default_scope: str | None = None
    default_unit: str | None = None
    default_currency: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "publisher": self.publisher,
            "subject": self.subject,
            "doc_type": self.doc_type,
            "as_of_date": self.as_of_date.isoformat() if self.as_of_date else None,
            "default_period": self.default_period,
            "default_scope": self.default_scope,
            "default_unit": self.default_unit,
            "default_currency": self.default_currency,
        }


def _date(value: Any) -> dt.date | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return dt.date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


def _validate_profile(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")


def _validate_claims(data: Any) -> None:
    if not isinstance(data, dict) or "claims" not in data:
        raise ValueError('expected {"claims": [...]}')
    if not isinstance(data["claims"], list):
        raise ValueError('"claims" must be a list')
    for c in data["claims"]:
        if not isinstance(c, dict):
            raise ValueError("each claim must be an object")
        for required in ("measure", "quote"):
            if not c.get(required):
                raise ValueError(f'each claim needs a non-empty "{required}"')


async def profile_document(doc: Document, llm: LLMClient) -> DocProfile:
    """Establish document-level defaults from its opening pages."""
    head = "\n\n".join(p.text for p in doc.pages[:3])[:MAX_PAGE_CHARS]
    try:
        data = await llm.json_call(
            PROFILE_SYSTEM,
            PROFILE_USER.format(filename=doc.filename, text=head),
            validate=_validate_profile,
        )
    except LLMError as exc:
        log.warning("profiling failed for %s: %s", doc.filename, exc)
        return DocProfile()

    return DocProfile(
        publisher=data.get("publisher"),
        subject=data.get("subject"),
        doc_type=data.get("doc_type"),
        as_of_date=_date(data.get("as_of_date")),
        default_period=data.get("default_period"),
        default_scope=data.get("default_scope"),
        default_unit=data.get("default_unit"),
        default_currency=data.get("default_currency"),
    )


def _build_claim(raw: dict, doc: Document, page: Page, profile: DocProfile) -> Claim | None:
    """Turn one raw extraction into a normalised Claim, or None if unusable."""
    quote = (raw.get("quote") or "").strip()
    measure = (raw.get("measure") or "").strip()
    if not quote or not measure:
        return None

    kind_raw = (raw.get("value_kind") or "").strip().lower()
    try:
        kind = ValueKind(kind_raw)
    except ValueError:
        kind = ValueKind.NUMBER if raw.get("value") is not None else ValueKind.TEXT

    # Qualifiers: what the page said, falling back to the document's defaults.
    period_label = raw.get("period") or profile.default_period
    unit_raw = raw.get("unit") or profile.default_unit
    scope = raw.get("scope") or profile.default_scope

    quantity = None
    if kind in (ValueKind.NUMBER, ValueKind.RANGE):
        low = raw.get("value")
        if not isinstance(low, (int, float)):
            return None
        high = raw.get("value_high")
        high = high if isinstance(high, (int, float)) else low
        unit = parse_unit(unit_raw)
        quantity = Quantity(
            low=float(low),
            high=float(high),
            unit_raw=unit_raw,
            dimension=unit.dimension if unit else None,
            currency=unit.currency if unit else None,
            canonical_low=canonicalise(float(low), unit),
            canonical_high=canonicalise(float(high), unit),
        )
        if high != low:
            kind = ValueKind.RANGE

    qualifiers = Qualifiers(
        period=parse_period(period_label) if period_label else None,
        scope=scope,
        basis=raw.get("basis"),
        # A claim's as_of defaults to the date its document speaks as of. This
        # is what lets a later filing supersede an earlier one.
        as_of=profile.as_of_date,
        estimate_type=raw.get("estimate_type"),
    )

    confidence = raw.get("confidence")
    confidence = float(confidence) if isinstance(confidence, (int, float)) else 0.5

    return Claim(
        id=uuid.uuid4().hex[:16],
        doc_id=doc.doc_id,
        subject_raw=(raw.get("subject") or profile.subject or "").strip() or "unknown",
        measure_raw=measure,
        value_kind=kind,
        quantity=quantity,
        text_value=(raw.get("text_value") or None),
        date_value=_date(raw.get("date_value")),
        bool_value=raw.get("bool_value") if isinstance(raw.get("bool_value"), bool) else None,
        qualifiers=qualifiers,
        evidence=Evidence(
            doc_id=doc.doc_id,
            page_no=page.page_no,
            printed_page=page.printed_page,
            quote=quote,
            grounding=GroundingStatus.NOT_FOUND,  # set by the grounding pass
        ),
        extraction_confidence=max(0.0, min(confidence, 1.0)),
        status=ClaimStatus.ACTIVE,
    )


async def extract_page(
    doc: Document, page: Page, profile: DocProfile, llm: LLMClient
) -> list[Claim]:
    try:
        data = await llm.json_call(
            EXTRACT_SYSTEM,
            EXTRACT_USER.format(
                subject=profile.subject or "unknown",
                publisher=profile.publisher or "unknown",
                default_period=profile.default_period or "not stated",
                default_scope=profile.default_scope or "not stated",
                default_unit=profile.default_unit or "not stated",
                max_claims=MAX_CLAIMS_PER_PAGE,
                page_no=page.page_no,
                text=page.text[:MAX_PAGE_CHARS],
            ),
            validate=_validate_claims,
        )
    except LLMError as exc:
        log.warning("extraction failed on %s p%d: %s", doc.filename, page.page_no, exc)
        return []

    claims = []
    for raw in data.get("claims", [])[:MAX_CLAIMS_PER_PAGE]:
        try:
            if (claim := _build_claim(raw, doc, page, profile)) is not None:
                claims.append(claim)
        except Exception as exc:  # a single malformed claim must not kill the page
            log.debug("skipped malformed claim on p%d: %s", page.page_no, exc)
    return claims


async def extract_document(
    doc: Document,
    llm: LLMClient,
    *,
    budget: int | None = None,
    progress=None,
) -> tuple[DocProfile, list[Claim]]:
    """Profile a document, then extract claims from every page triage keeps."""
    profile = await profile_document(doc, llm)
    pages: list[PageScore] = triage(doc, budget=budget)

    async def one(score: PageScore) -> list[Claim]:
        page = doc.page(score.page_no)
        claims = await extract_page(doc, page, profile, llm) if page else []
        if progress:
            progress(score.page_no, len(claims))
        return claims

    results = await asyncio.gather(*(one(s) for s in pages))
    return profile, [c for group in results for c in group]
