"""The reconciler: decide how two claims relate, and show the working.

This is the part that is deliberately *not* an LLM judgement. Given two claims
that have already been extracted, grounded and canonicalised, deciding whether
they agree is a sequence of concrete checks -- are the units commensurable, do
the periods overlap, is one scope inside the other, do the values match once
rescaled. Each check appends a line to a trace, and the trace is the
explanation shown to the user.

The order matters. Values are compared *last*, and only after the qualifiers
have established that comparing them means anything at all. Most false
contradictions come from systems that compare the numbers first.

Two arithmetic checks go further than compatibility. When one period contains
another, or one scope subsumes another, an additive quantity must satisfy an
inequality -- a quarter cannot exceed its own year, a standalone figure cannot
exceed the consolidated one. A violation is a contradiction that can be
*proved* from the documents rather than argued about.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from .models import (
    Claim,
    ClaimStatus,
    Period,
    Relation,
    TraceStep,
    ValueKind,
    Verdict,
)
from .normalize.periods import relate as relate_period
from .normalize.scope import ScopeRelation, relate_scope
from .registry import normalise_subject, similarity

# Values within this relative distance are treated as the same figure. Documents
# round differently -- the earnings deck says "8,142 Cr" for the annual report's
# "81,415.38 million", a 0.006% gap that is presentation, not disagreement.
VALUE_TOLERANCE = 0.01

# A basis word that changes what is being counted, rather than just asserting
# the ordinary reading.
MODIFYING_BASIS = {"adjusted", "restated", "normalised", "normalized", "pro forma", "like-for-like"}
NEUTRAL_BASIS = {"reported", "actual", "audited", "as reported"}

# Estimate types whose value is expected to move as new information arrives.
PROVISIONAL = {"projection", "projected", "forecast", "estimate", "estimated", "provisional", "advance estimate"}


@dataclass
class Agreement:
    agree: bool
    detail: str


def _canonical_interval(claim: Claim) -> tuple[float, float] | None:
    q = claim.quantity
    if q is None or not q.is_canonical:
        return None
    return q.canonical_low, q.canonical_high


def compare_values(a: Claim, b: Claim, tolerance: float = VALUE_TOLERANCE) -> Agreement | None:
    """Compare two claims' values, or None when they are not comparable."""
    if a.value_kind in (ValueKind.NUMBER, ValueKind.RANGE) and b.value_kind in (
        ValueKind.NUMBER,
        ValueKind.RANGE,
    ):
        ia, ib = _canonical_interval(a), _canonical_interval(b)
        if ia is None or ib is None:
            return None
        # Overlapping intervals agree. A point inside a range is the common case:
        # RBI's 6.5% sits inside the Economic Survey's 6.3-6.8%.
        if ia[0] <= ib[1] and ib[0] <= ia[1]:
            return Agreement(True, f"{_fmt(a)} and {_fmt(b)} overlap")
        gap = min(abs(ia[0] - ib[1]), abs(ib[0] - ia[1]))
        scale = max(abs(ia[0]), abs(ia[1]), abs(ib[0]), abs(ib[1]), 1e-9)
        rel = gap / scale
        if rel <= tolerance:
            return Agreement(True, f"{_fmt(a)} vs {_fmt(b)} differ by {rel:.3%} (within rounding)")
        return Agreement(False, f"{_fmt(a)} vs {_fmt(b)} differ by {rel:.1%}")

    if a.value_kind is ValueKind.DATE and b.value_kind is ValueKind.DATE:
        same = a.date_value == b.date_value
        return Agreement(same, f"{a.date_value} vs {b.date_value}")

    if a.value_kind is ValueKind.BOOLEAN and b.value_kind is ValueKind.BOOLEAN:
        same = a.bool_value == b.bool_value
        return Agreement(same, f"{a.bool_value} vs {b.bool_value}")

    ta, tb = (a.text_value or "").strip(), (b.text_value or "").strip()
    if ta and tb:
        sim = similarity(normalise_subject(ta), normalise_subject(tb))
        return Agreement(sim >= 0.8, f"{ta!r} vs {tb!r} (similarity {sim:.2f})")

    return None


def _fmt(claim: Claim) -> str:
    unit = claim.quantity.unit_raw if claim.quantity and claim.quantity.unit_raw else ""
    return f"{claim.display_value()}".strip() + (f" {unit}" if unit and unit not in claim.display_value() else "")


def _is_additive(claim: Claim) -> bool:
    """Can this quantity be summed over sub-periods or sub-entities?

    Flows (revenue, tonnage, shipments) can. Rates, margins and ratios cannot --
    a quarterly margin is not a quarter of the annual margin.
    """
    q = claim.quantity
    if q is None or q.dimension is None:
        return False
    if q.dimension in ("percent", "ratio"):
        return False
    name = claim.measure_raw.lower()
    return not any(w in name for w in ("margin", "rate", "growth", "ratio", "per cent", "share"))


def _basis_relation(a: str | None, b: str | None) -> tuple[str, str]:
    """Return (status, detail) for a pair of basis qualifiers."""
    na = (a or "").strip().lower() or None
    nb = (b or "").strip().lower() or None
    if na == nb:
        return "same", f"both {na or 'unstated'}"
    if na and nb:
        return "different", f"{na} vs {nb}"
    stated, = (x for x in (na, nb) if x)
    if stated in MODIFYING_BASIS:
        # "adjusted" against an unlabelled figure: the unlabelled one might be
        # reported or might be adjusted too. Guessing invents a verdict.
        return "unknown", f"{stated} vs unstated"
    return "same", f"{stated} vs unstated (read as the ordinary basis)"


def _supersedes(a: Claim, b: Claim) -> Claim | None:
    """Which claim, if either, is the later observation of a changeable fact."""
    aa, ab = a.qualifiers.as_of, b.qualifiers.as_of
    if aa is None or ab is None or aa == ab:
        return None
    later, earlier = (a, b) if aa > ab else (b, a)
    est = (later.qualifiers.estimate_type or "").lower()
    changeable = (
        est in PROVISIONAL
        or (earlier.qualifiers.estimate_type or "").lower() in PROVISIONAL
        or later.value_kind in (ValueKind.TEXT, ValueKind.BOOLEAN, ValueKind.DATE)
    )
    return later if changeable else None


def reconcile_pair(a: Claim, b: Claim) -> Relation | None:
    """Decide how two claims relate. Returns None if they should not be compared."""
    if a.id == b.id or a.status is not ClaimStatus.ACTIVE or b.status is not ClaimStatus.ACTIVE:
        return None

    trace: list[TraceStep] = []

    # -- subject ---------------------------------------------------------
    sa, sb = normalise_subject(a.subject_raw), normalise_subject(b.subject_raw)
    subj_sim = similarity(sa, sb)
    if sa != sb and subj_sim < 0.6:
        return None
    trace.append(TraceStep(
        check="subject", outcome="pass",
        detail=f"{a.subject_raw!r} ≡ {b.subject_raw!r}" if sa == sb
               else f"{a.subject_raw!r} ~ {b.subject_raw!r} (similarity {subj_sim:.2f})",
    ))

    # -- measure ---------------------------------------------------------
    trace.append(TraceStep(
        check="measure", outcome="pass",
        detail=f"{a.measure_raw!r} and {b.measure_raw!r} → canonical {a.measure_key!r}",
    ))

    # -- units -----------------------------------------------------------
    qa, qb = a.quantity, b.quantity
    if qa and qb:
        if not (qa.is_canonical and qb.is_canonical):
            unknown = qa.unit_raw if not qa.is_canonical else qb.unit_raw
            trace.append(TraceStep(check="unit", outcome="blocked",
                                   detail=f"unit {unknown!r} not recognised; values cannot be rescaled"))
            return _relation(a, b, Verdict.UNDERSPECIFIED, 0.4,
                             "Cannot compare: an unrecognised unit means the figures cannot be put on the same scale.",
                             trace)
        if qa.dimension != qb.dimension or (qa.currency or "") != (qb.currency or ""):
            trace.append(TraceStep(check="unit", outcome="blocked",
                                   detail=f"{qa.unit_raw!r} ({qa.dimension}/{qa.currency}) vs "
                                          f"{qb.unit_raw!r} ({qb.dimension}/{qb.currency}); no conversion applied"))
            return _relation(a, b, Verdict.UNDERSPECIFIED, 0.4,
                             "Cannot compare: different units of measurement, and currencies are never converted.",
                             trace)
        trace.append(TraceStep(check="unit", outcome="pass",
                               detail=f"{qa.unit_raw!r} and {qb.unit_raw!r} both rescale to {qa.dimension}"))

    # -- period ----------------------------------------------------------
    pa, pb = a.qualifiers.period, b.qualifiers.period
    period_rel = relate_period(pa, pb) if (pa and pb) else None
    if pa is None or pb is None:
        trace.append(TraceStep(check="period", outcome="blocked",
                               detail=f"period missing on {'A' if pa is None else 'B'}"))
        return _relation(a, b, Verdict.UNDERSPECIFIED, 0.45,
                         "Cannot compare: one claim does not state the period it covers.",
                         trace)
    if period_rel is None:
        trace.append(TraceStep(check="period", outcome="blocked",
                               detail=f"{pa.label!r} or {pb.label!r} could not be resolved to dates"))
        return _relation(a, b, Verdict.UNDERSPECIFIED, 0.45,
                         "Cannot compare: a stated period could not be resolved to actual dates.",
                         trace)

    trace.append(TraceStep(
        check="period",
        outcome="pass" if period_rel == "equals" else "info",
        detail=f"{pa.label!r} [{pa.start}→{pa.end}] {period_rel.upper()} "
               f"{pb.label!r} [{pb.start}→{pb.end}]",
    ))

    # ---- States are scoped by when they were observed, not by a window --
    #
    # A flow (revenue, tonnage) is measured *over* a period, so two disjoint
    # periods give two complementary facts: FY23 revenue and FY24 revenue are
    # both true and neither replaces the other.
    #
    # A state (a board role, a registered address, a status) is not measured
    # over a window at all -- it holds until it changes, and the period attached
    # to it is really just when the document was speaking. Two observations of a
    # state at different times are the *same* fact seen twice, so they must go to
    # the supersession check rather than being called complementary.
    is_state = a.value_kind in (ValueKind.TEXT, ValueKind.BOOLEAN, ValueKind.DATE) and \
        b.value_kind in (ValueKind.TEXT, ValueKind.BOOLEAN, ValueKind.DATE)

    if is_state and period_rel != "equals":
        trace.append(TraceStep(
            check="period", outcome="info",
            detail="state-valued claim: the period records when it was observed, "
                   "not a span it was measured over",
        ))

    # -- scope -----------------------------------------------------------
    scope_rel = relate_scope(a.qualifiers.scope, b.qualifiers.scope)
    trace.append(TraceStep(
        check="scope",
        outcome="pass" if scope_rel is ScopeRelation.SAME else "info",
        detail=f"{a.qualifiers.scope or 'unstated'} vs {b.qualifiers.scope or 'unstated'} → {scope_rel.value}",
    ))

    # -- basis -----------------------------------------------------------
    basis_status, basis_detail = _basis_relation(a.qualifiers.basis, b.qualifiers.basis)
    trace.append(TraceStep(
        check="basis",
        outcome={"same": "pass", "different": "info", "unknown": "blocked"}[basis_status],
        detail=basis_detail,
    ))

    # -- values ----------------------------------------------------------
    agreement = compare_values(a, b)
    if agreement is None:
        trace.append(TraceStep(check="value", outcome="blocked", detail="values are not of comparable kinds"))
        return _relation(a, b, Verdict.UNDERSPECIFIED, 0.4,
                         "Cannot compare: the two claims record different kinds of value.", trace)

    # ---- Contexts that do not overlap: both claims can be true ----------
    if not is_state and period_rel in ("before", "after"):
        trace.append(TraceStep(check="value", outcome="info", detail=agreement.detail))
        return _relation(a, b, Verdict.COMPLEMENTARY, 0.9,
                         f"Different periods ({pa.label} and {pb.label} do not overlap), "
                         "so the two figures describe different spans of time and both can hold.",
                         trace)

    if not is_state and period_rel in ("contains", "during"):
        outer, inner = (a, b) if period_rel == "contains" else (b, a)
        step = _containment_check(outer, inner, "period")
        trace.append(step)
        if step.outcome == "fail":
            return _relation(a, b, Verdict.CONTRADICTS, 0.85,
                             f"The figure for {inner.qualifiers.period.label} exceeds the figure for the "
                             f"{outer.qualifiers.period.label} that contains it, which cannot both be true.",
                             trace)
        return _relation(a, b, Verdict.COMPLEMENTARY, 0.9,
                         f"{inner.qualifiers.period.label} falls inside {outer.qualifiers.period.label}: "
                         "a sub-period figure nested in the total, not a disagreement.",
                         trace)

    if scope_rel in (ScopeRelation.SUBSUMES, ScopeRelation.SUBSUMED_BY):
        outer, inner = (a, b) if scope_rel is ScopeRelation.SUBSUMES else (b, a)
        step = _containment_check(outer, inner, "scope")
        trace.append(step)
        if step.outcome == "fail":
            return _relation(a, b, Verdict.CONTRADICTS, 0.8,
                             f"The {inner.qualifiers.scope or 'narrower'} figure exceeds the "
                             f"{outer.qualifiers.scope or 'wider'} figure that includes it.",
                             trace)
        return _relation(a, b, Verdict.COMPLEMENTARY, 0.88,
                         f"Different reporting scope ({a.qualifiers.scope or 'unstated'} vs "
                         f"{b.qualifiers.scope or 'unstated'}): one is a subset of the other.",
                         trace)

    if scope_rel is ScopeRelation.DISJOINT:
        return _relation(a, b, Verdict.COMPLEMENTARY, 0.85,
                         f"Different segments ({a.qualifiers.scope} and {b.qualifiers.scope}) "
                         "describe separate parts of the business.",
                         trace)

    if scope_rel is ScopeRelation.UNKNOWN:
        return _relation(a, b, Verdict.UNDERSPECIFIED, 0.5,
                         "Cannot compare: one claim does not state its reporting scope, "
                         "so it is unclear whether the two cover the same entity.",
                         trace)

    if basis_status == "different":
        trace.append(TraceStep(check="value", outcome="info", detail=agreement.detail))
        return _relation(a, b, Verdict.COMPLEMENTARY, 0.85,
                         f"Different basis ({basis_detail}): the figures measure the same quantity "
                         "under different accounting treatments.",
                         trace)

    if basis_status == "unknown":
        return _relation(a, b, Verdict.UNDERSPECIFIED, 0.45,
                         f"Cannot compare: {basis_detail}, and an unlabelled figure "
                         "may or may not be on the same basis.",
                         trace)

    # ---- Contexts are compatible: the values now mean something ---------
    trace.append(TraceStep(check="value",
                           outcome="pass" if agreement.agree else "fail",
                           detail=agreement.detail))

    if agreement.agree:
        cross = a.doc_id != b.doc_id
        return _relation(a, b, Verdict.CORROBORATES, 0.9 if cross else 0.75,
                         ("Two documents state the same figure for the same measure, period and scope."
                          if cross else
                          "The same figure is stated twice within the document, consistently."),
                         trace)

    if (later := _supersedes(a, b)) is not None:
        earlier = b if later is a else a
        trace.append(TraceStep(
            check="as_of", outcome="info",
            detail=f"later observation {later.qualifiers.as_of} supersedes {earlier.qualifiers.as_of}",
        ))
        return _relation(a, b, Verdict.SUPERSEDES, 0.85,
                         f"Not a disagreement: the claim as of {later.qualifiers.as_of} replaces the one as of "
                         f"{earlier.qualifiers.as_of}. The underlying fact changed between the two documents.",
                         trace)

    note = ""
    if a.qualifiers.as_of and b.qualifiers.as_of and a.qualifiers.as_of != b.qualifiers.as_of:
        note = " The documents carry different dates, so this may be an unflagged restatement."
    return _relation(a, b, Verdict.CONTRADICTS, 0.85,
                     "Same measure, period, scope and basis, but the values disagree "
                     f"beyond rounding.{note}",
                     trace)


def _containment_check(outer: Claim, inner: Claim, kind: str) -> TraceStep:
    """A part cannot exceed the whole -- for quantities that add up."""
    if not (_is_additive(outer) and _is_additive(inner)):
        return TraceStep(check=f"{kind}-containment", outcome="info",
                         detail="measure is a rate or ratio, so it does not aggregate; no arithmetic check")
    io, ii = _canonical_interval(outer), _canonical_interval(inner)
    if io is None or ii is None:
        return TraceStep(check=f"{kind}-containment", outcome="info", detail="values not on a canonical scale")
    if ii[0] > io[1] and io[1] > 0:
        return TraceStep(check=f"{kind}-containment", outcome="fail",
                         detail=f"part {ii[0]:,.0f} exceeds whole {io[1]:,.0f}")
    return TraceStep(check=f"{kind}-containment", outcome="pass",
                     detail=f"part {ii[0]:,.0f} ≤ whole {io[1]:,.0f}, arithmetically consistent")


def _relation(a: Claim, b: Claim, verdict: Verdict, confidence: float,
              summary: str, trace: list[TraceStep]) -> Relation:
    return Relation(
        id=uuid.uuid4().hex[:16],
        claim_a_id=a.id,
        claim_b_id=b.id,
        verdict=verdict,
        confidence=round(confidence, 3),
        summary=summary,
        trace=list(trace),
    )


def reconcile_all(claims: list[Claim], max_group: int = 400) -> list[Relation]:
    """Compare every pair of claims that share a subject and a canonical measure.

    Blocking on (subject, measure) is what keeps this tractable: comparing all
    pairs of N claims is quadratic, but claims only ever need comparing when
    they are about the same thing.
    """
    groups: dict[tuple[str, str], list[Claim]] = {}
    for c in claims:
        if c.status is not ClaimStatus.ACTIVE:
            continue
        key = (normalise_subject(c.subject_raw), c.measure_key or c.measure_raw.lower())
        groups.setdefault(key, []).append(c)

    relations: list[Relation] = []
    for group in groups.values():
        if len(group) > max_group:
            group = sorted(group, key=lambda c: -c.extraction_confidence)[:max_group]
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                if (rel := reconcile_pair(a, b)) is not None:
                    relations.append(rel)
    return relations
