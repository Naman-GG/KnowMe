"""Core data model for the fact knowledge layer.

The central idea: a fact is never just (subject, measure, value). It is a value
*plus the qualifiers that make it mean something* -- the period it covers, the
scope of the entity it describes, the accounting/statistical basis used, and the
moment at which it was asserted.

Two claims only conflict if their qualifiers are compatible and their values
disagree. Everything downstream falls out of that.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# --------------------------------------------------------------------------
# Values
# --------------------------------------------------------------------------

class ValueKind(str, Enum):
    NUMBER = "number"      # a point estimate, possibly with a unit
    RANGE = "range"        # "between 6.3 and 6.8 per cent"
    TEXT = "text"          # "Sandeep Kumar Barasia", "Bengaluru, Karnataka"
    BOOLEAN = "boolean"    # "is a director" / "is not a director"
    DATE = "date"          # "resigned with effect from July 01, 2024"


class Quantity(BaseModel):
    """A numeric value held as a closed interval.

    Point estimates are stored as a degenerate interval (low == high). That one
    decision makes the comparison logic uniform: checking whether RBI's 6.5%
    agrees with the Economic Survey's "6.3 to 6.8 per cent" is then just
    interval containment rather than a special case.
    """

    low: float
    high: float
    unit_raw: str | None = None          # exactly as written: "Rs. Cr", "per cent"
    dimension: str | None = None         # currency | percent | count | mass | ratio
    currency: str | None = None          # INR | USD | ...
    # Value rescaled to the dimension's base unit (rupees, percent, items, tonnes).
    # None when the unit could not be recognised -- we do not guess.
    canonical_low: float | None = None
    canonical_high: float | None = None

    @model_validator(mode="after")
    def _order(self) -> "Quantity":
        if self.low > self.high:
            self.low, self.high = self.high, self.low
        if (
            self.canonical_low is not None
            and self.canonical_high is not None
            and self.canonical_low > self.canonical_high
        ):
            self.canonical_low, self.canonical_high = self.canonical_high, self.canonical_low
        return self

    @property
    def is_point(self) -> bool:
        return self.low == self.high

    @property
    def is_canonical(self) -> bool:
        return self.canonical_low is not None and self.canonical_high is not None


# --------------------------------------------------------------------------
# Qualifiers
# --------------------------------------------------------------------------

class Period(BaseModel):
    """A time span, normalised to a half-open date interval where possible.

    `label` keeps the document's own words ("FY24", "year ended March 31, 2024",
    "Q4 FY24") so that explanations can quote the source rather than a
    reconstruction of it.
    """

    label: str
    start: dt.date | None = None
    end: dt.date | None = None

    @property
    def is_resolved(self) -> bool:
        return self.start is not None and self.end is not None


class Qualifiers(BaseModel):
    """Context that determines whether two claims are even comparable.

    Four slots are *deeply understood* -- the reconciler has real logic for each
    (interval algebra for periods, a subsumption lattice for scope, and so on).

    `extra` is an open dictionary for qualifiers the documents introduce that we
    did not anticipate. The reconciler still uses them, but only generically:
    same value, different value, or missing. This is how the schema grows with
    the corpus without us pretending to understand every new dimension.
    """

    period: Period | None = None
    scope: str | None = None        # consolidated | standalone | India | "Express Parcel"
    basis: str | None = None        # reported | adjusted | audited | restated | ...
    as_of: dt.date | None = None    # when the claim was asserted (usually the doc date)
    estimate_type: str | None = None  # actual | provisional | estimate | projection

    extra: dict[str, str] = Field(default_factory=dict)

    def known_slots(self) -> dict[str, Any]:
        return {
            "period": self.period.label if self.period else None,
            "scope": self.scope,
            "basis": self.basis,
            "estimate_type": self.estimate_type,
        }


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------

class GroundingStatus(str, Enum):
    EXACT = "exact"          # quote found verbatim in the page text
    FUZZY = "fuzzy"          # found after whitespace/ligature normalisation
    NOT_FOUND = "not_found"  # quote is not on the cited page -> quarantine


class Evidence(BaseModel):
    """Where a claim came from, precise enough to highlight in the source."""

    doc_id: str
    page_no: int                  # 1-based index into the PDF
    printed_page: str | None = None  # page number printed on the page, if any
    quote: str                    # verbatim span the model claims to have read
    char_start: int | None = None  # offset into that page's extracted text
    char_end: int | None = None
    grounding: GroundingStatus = GroundingStatus.NOT_FOUND


# --------------------------------------------------------------------------
# Claims
# --------------------------------------------------------------------------

class ClaimStatus(str, Enum):
    ACTIVE = "active"
    QUARANTINED = "quarantined"   # failed grounding; kept, but excluded from reconciliation


class Claim(BaseModel):
    id: str
    doc_id: str

    subject_raw: str              # "Delhivery Limited", "India", "Mr. Sandeep Kumar Barasia"
    subject_key: str | None = None  # canonicalised entity key

    measure_raw: str              # "revenue from operations", "real GDP growth"
    measure_key: str | None = None  # canonical key from the measure registry

    value_kind: ValueKind
    quantity: Quantity | None = None
    text_value: str | None = None
    date_value: dt.date | None = None
    bool_value: bool | None = None

    qualifiers: Qualifiers = Field(default_factory=Qualifiers)
    evidence: Evidence

    extraction_confidence: float = 0.5
    status: ClaimStatus = ClaimStatus.ACTIVE
    quarantine_reason: str | None = None

    def display_value(self) -> str:
        if self.value_kind == ValueKind.NUMBER and self.quantity:
            unit = f" {self.quantity.unit_raw}" if self.quantity.unit_raw else ""
            return f"{self.quantity.low:,.2f}".rstrip("0").rstrip(".") + unit
        if self.value_kind == ValueKind.RANGE and self.quantity:
            unit = f" {self.quantity.unit_raw}" if self.quantity.unit_raw else ""
            return f"{self.quantity.low:g} to {self.quantity.high:g}{unit}"
        if self.value_kind == ValueKind.DATE and self.date_value:
            return self.date_value.isoformat()
        if self.value_kind == ValueKind.BOOLEAN and self.bool_value is not None:
            return "true" if self.bool_value else "false"
        return self.text_value or ""


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------

class Verdict(str, Enum):
    """The output vocabulary of the reconciler.

    The assignment asks for three relations (corroborate / contradict /
    reconcile-through-context). `RECONCILE` splits into two mechanisms that are
    genuinely different -- disjoint context vs. the world having changed -- and
    UNDERSPECIFIED is the honest fifth answer when we cannot tell.
    """

    CORROBORATES = "corroborates"      # comparable context, values agree
    CONTRADICTS = "contradicts"        # comparable context, values disagree
    COMPLEMENTARY = "complementary"    # contexts do not overlap; both can be true
    SUPERSEDES = "supersedes"          # same claim, later as_of; the world moved on
    UNDERSPECIFIED = "underspecified"  # a qualifier we need is missing


class TraceStep(BaseModel):
    """One line of the machine-readable derivation shown to the user.

    Every verdict is explained by the actual sequence of checks that produced
    it, not by a paragraph of LLM prose written after the fact.
    """

    check: str            # "unit", "measure", "period", "scope", "value"
    outcome: Literal["pass", "fail", "info", "blocked"]
    detail: str


class Relation(BaseModel):
    id: str
    claim_a_id: str
    claim_b_id: str
    verdict: Verdict
    confidence: float
    summary: str                                   # one-line human explanation
    trace: list[TraceStep] = Field(default_factory=list)
    # Set when the reconciler deferred a judgement to the LLM adjudicator.
    adjudicated: bool = False
