from factlayer.normalize.scope import ScopeRelation as R
from factlayer.normalize.scope import comparable, relate_scope


def test_identical_scopes():
    assert relate_scope("consolidated", "Consolidated") is R.SAME
    assert relate_scope(None, None) is R.SAME


def test_consolidated_subsumes_standalone():
    """The Delhivery case: 81,415 mn consolidated vs 74,540 mn standalone."""
    assert relate_scope("consolidated", "standalone") is R.SUBSUMES
    assert relate_scope("standalone", "consolidated") is R.SUBSUMED_BY
    assert not comparable("consolidated", "standalone")


def test_distinct_segments_are_disjoint():
    assert relate_scope("Express Parcel", "PTL") is R.DISJOINT


def test_segment_within_its_own_wider_scope():
    assert relate_scope("express parcel india", "express parcel") is R.SUBSUMED_BY


def test_missing_scope_is_unknown_not_equal():
    """An unlabelled figure must not be assumed to match a labelled one.

    An unstated scope could be either side of the boundary, so the honest
    answer is UNKNOWN -- which the reconciler turns into UNDERSPECIFIED rather
    than a contradiction.
    """
    assert relate_scope("consolidated", None) is R.UNKNOWN
    assert relate_scope(None, "standalone") is R.UNKNOWN
    assert not comparable("consolidated", None)


def test_known_bucket_subsumes_named_slice():
    """A named segment sits inside the consolidated total."""
    assert relate_scope("consolidated", "Express Parcel") is R.SUBSUMES
    assert relate_scope("Express Parcel", "total") is R.SUBSUMED_BY
