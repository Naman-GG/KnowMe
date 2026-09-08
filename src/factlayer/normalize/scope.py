"""Scope comparison: does one claim describe a subset of another's subject?

Two figures can share a measure and a period and still be entirely compatible
because they describe different slices of the entity -- the parent company
alone versus the group, one business segment versus the whole, one country
versus the world.

The lattice below is small and deliberately generic. It knows about the
containment relations that recur across corporate and statistical reporting
(standalone within consolidated, a segment within the total) and treats
everything else structurally: identical strings are the same scope, and two
distinct named slices of the same parent are presumed disjoint. Scopes it has
never seen are reported UNKNOWN rather than assumed equal, which keeps an
unfamiliar corpus from silently producing false contradictions.
"""

from __future__ import annotations

import re
from enum import Enum

# Reporting boundaries, ordered from widest to narrowest.
CONSOLIDATED = {"consolidated", "group", "consolidated basis"}
STANDALONE = {"standalone", "separate", "parent", "standalone basis", "company"}
# Words meaning "everything", which subsume any named slice.
TOTAL = {"total", "overall", "aggregate", "all", "whole", "entity", "economy"}

_NORMALISE = re.compile(r"[^a-z0-9 ]+")


class ScopeRelation(str, Enum):
    SAME = "same"                # identical slice
    SUBSUMES = "subsumes"        # a contains b
    SUBSUMED_BY = "subsumed_by"  # a is contained in b
    DISJOINT = "disjoint"        # different named slices of the same parent
    UNKNOWN = "unknown"          # not enough information to say


def normalise_scope(scope: str | None) -> str | None:
    if not scope:
        return None
    s = _NORMALISE.sub(" ", scope.strip().lower())
    s = " ".join(s.split())
    return s or None


def _bucket(s: str) -> str | None:
    if s in CONSOLIDATED:
        return "consolidated"
    if s in STANDALONE:
        return "standalone"
    if s in TOTAL:
        return "total"
    return None


def relate_scope(a: str | None, b: str | None) -> ScopeRelation:
    """Relation of scope `a` to scope `b`."""
    na, nb = normalise_scope(a), normalise_scope(b)

    # An unstated scope is not a wildcard. We do not know whether an unlabelled
    # figure is consolidated or standalone, and guessing manufactures verdicts.
    if na is None and nb is None:
        return ScopeRelation.SAME
    if na is None or nb is None:
        return ScopeRelation.UNKNOWN
    if na == nb:
        return ScopeRelation.SAME

    ba, bb = _bucket(na), _bucket(nb)

    # The group includes the parent; the parent alone is a subset of the group.
    if ba == "consolidated" and bb == "standalone":
        return ScopeRelation.SUBSUMES
    if ba == "standalone" and bb == "consolidated":
        return ScopeRelation.SUBSUMED_BY

    # "Total" subsumes any specific named slice.
    if ba == "total" and bb is None:
        return ScopeRelation.SUBSUMES
    if bb == "total" and ba is None:
        return ScopeRelation.SUBSUMED_BY
    if ba in ("consolidated", "total") and bb is None:
        return ScopeRelation.SUBSUMES
    if bb in ("consolidated", "total") and ba is None:
        return ScopeRelation.SUBSUMED_BY

    # Substring containment catches "express parcel" vs "express parcel india".
    if na in nb:
        return ScopeRelation.SUBSUMES
    if nb in na:
        return ScopeRelation.SUBSUMED_BY

    # Two distinct named slices (Express Parcel vs PTL) do not overlap. This is
    # a presumption, not a proof, and the reconciler records it as such.
    if ba is None and bb is None:
        return ScopeRelation.DISJOINT

    return ScopeRelation.UNKNOWN


def comparable(a: str | None, b: str | None) -> bool:
    """True when two scopes describe the same slice and values may be compared."""
    return relate_scope(a, b) is ScopeRelation.SAME
