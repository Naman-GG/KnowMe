"""The reconciler must reach the right verdict on each real case in the corpus.

Every claim below is built from figures that genuinely appear in the starter
documents, so these tests double as a specification of the four cases the
assignment asks for.
"""

import datetime as dt

from factlayer.models import Claim, Evidence, Period, Qualifiers, Quantity, ValueKind, Verdict
from factlayer.normalize.periods import parse_period
from factlayer.normalize.units import canonicalise, parse_unit
from factlayer.reconcile import reconcile_pair


def claim(measure, value, unit, period, doc, *, high=None, scope=None, basis=None,
          as_of=None, est=None, subject="Delhivery Limited", key=None, kind=ValueKind.NUMBER,
          text=None):
    u = parse_unit(unit)
    high = value if high is None else high
    q = None
    if value is not None:
        q = Quantity(low=value, high=high, unit_raw=unit,
                     dimension=u.dimension if u else None,
                     currency=u.currency if u else None,
                     canonical_low=canonicalise(value, u), canonical_high=canonicalise(high, u))
    return Claim(
        id=f"{doc}-{measure}-{period}-{scope}-{basis}-{value}",
        doc_id=doc, subject_raw=subject, measure_raw=measure,
        measure_key=key or measure.lower().replace(" ", "_"),
        value_kind=kind, quantity=q, text_value=text,
        qualifiers=Qualifiers(period=parse_period(period), scope=scope, basis=basis,
                              as_of=as_of, estimate_type=est),
        evidence=Evidence(doc_id=doc, page_no=1, quote="x" * 20),
    )


# --- Case 1: corroboration across documents, expressed differently ---------

def test_corroboration_across_units_and_documents():
    """Q4 deck says Rs. 8,142 Cr; the annual report says 81,415.38 million."""
    deck = claim("revenue from services", 8142, "Rs. Cr", "FY24", "deck")
    report = claim("revenue from services", 81415.38, "INR million", "FY24", "annual-report")
    rel = reconcile_pair(deck, report)
    assert rel.verdict is Verdict.CORROBORATES
    assert any(s.check == "unit" and s.outcome == "pass" for s in rel.trace)


# --- Case 2: a genuine contradiction ---------------------------------------

def test_same_context_different_values_contradicts():
    a = claim("revenue from operations", 81415.38, "INR million", "FY24", "doc-a", scope="consolidated")
    b = claim("revenue from operations", 78000.00, "INR million", "FY24", "doc-b", scope="consolidated")
    rel = reconcile_pair(a, b)
    assert rel.verdict is Verdict.CONTRADICTS
    assert any(s.check == "value" and s.outcome == "fail" for s in rel.trace)


def test_part_exceeding_whole_is_a_proved_contradiction():
    """A quarter cannot be larger than the year containing it."""
    year = claim("revenue from services", 8142, "Rs. Cr", "FY24", "deck")
    quarter = claim("revenue from services", 9000, "Rs. Cr", "Q4 FY24", "deck")
    rel = reconcile_pair(year, quarter)
    assert rel.verdict is Verdict.CONTRADICTS
    assert any(s.check == "period-containment" and s.outcome == "fail" for s in rel.trace)


# --- Case 3: apparent contradictions explained by context ------------------

def test_quarter_nested_in_year_is_complementary():
    year = claim("revenue from services", 8142, "Rs. Cr", "FY24", "deck")
    quarter = claim("revenue from services", 2076, "Rs. Cr", "Q4 FY24", "deck")
    rel = reconcile_pair(year, quarter)
    assert rel.verdict is Verdict.COMPLEMENTARY
    assert any(s.check == "period-containment" and s.outcome == "pass" for s in rel.trace)


def test_standalone_vs_consolidated_is_complementary():
    """Both appear on the same page of the annual report."""
    consolidated = claim("revenue from operations", 81415.38, "INR million", "FY24", "ar", scope="consolidated")
    standalone = claim("revenue from operations", 74540.82, "INR million", "FY24", "ar", scope="standalone")
    rel = reconcile_pair(consolidated, standalone)
    assert rel.verdict is Verdict.COMPLEMENTARY
    assert "scope" in rel.summary.lower()


def test_reported_vs_adjusted_ebitda_is_complementary():
    """FY24 EBITDA Rs.127 Cr against Adj. EBITDA Rs.76 Cr -- same period, different basis."""
    reported = claim("EBITDA", 127, "Rs. Cr", "FY24", "deck", basis="reported")
    adjusted = claim("EBITDA", 76, "Rs. Cr", "FY24", "deck", basis="adjusted")
    rel = reconcile_pair(reported, adjusted)
    assert rel.verdict is Verdict.COMPLEMENTARY
    assert "basis" in rel.summary.lower()


def test_different_years_are_complementary():
    fy23 = claim("EBITDA", -452, "Rs. Cr", "FY23", "deck")
    fy24 = claim("EBITDA", 127, "Rs. Cr", "FY24", "deck")
    assert reconcile_pair(fy23, fy24).verdict is Verdict.COMPLEMENTARY


# --- Supersession ----------------------------------------------------------

def test_later_forecast_supersedes_earlier():
    """Economic Survey (Jan 2025) and IMF (Nov 2025) on FY26 growth."""
    survey = claim("real GDP growth", 5.9, "per cent", "2025-26", "survey",
                   as_of=dt.date(2025, 1, 31), est="projection", subject="India")
    imf = claim("real GDP growth", 6.6, "per cent", "2025-26", "imf",
                as_of=dt.date(2025, 11, 25), est="projection", subject="India")
    rel = reconcile_pair(survey, imf)
    assert rel.verdict is Verdict.SUPERSEDES
    assert any(s.check == "as_of" for s in rel.trace)


def test_point_inside_a_forecast_range_corroborates():
    """RBI's 6.5% sits inside the Economic Survey's 6.3-6.8% range."""
    survey = claim("real GDP growth", 6.3, "per cent", "2025-26", "survey", high=6.8,
                   as_of=dt.date(2025, 1, 31), est="projection", subject="India")
    rbi = claim("real GDP growth", 6.5, "per cent", "2025-26", "rbi",
                as_of=dt.date(2025, 5, 29), est="projection", subject="India")
    rel = reconcile_pair(survey, rbi)
    assert rel.verdict is Verdict.CORROBORATES


def test_role_change_supersedes_rather_than_contradicts():
    """Barasia is a director in the 2022 prospectus and resigned by the FY24 report."""
    prospectus = claim("board role", None, None, "2022", "prospectus", kind=ValueKind.TEXT,
                       text="Executive Director and Chief Business Officer",
                       as_of=dt.date(2022, 5, 10), subject="Sandeep Kumar Barasia")
    annual = claim("board role", None, None, "FY24", "annual-report", kind=ValueKind.TEXT,
                   text="resigned with effect from July 01, 2024",
                   as_of=dt.date(2024, 8, 1), subject="Sandeep Kumar Barasia")
    rel = reconcile_pair(prospectus, annual)
    assert rel.verdict is Verdict.SUPERSEDES


# --- Underspecified: refusing to guess -------------------------------------

def test_missing_period_is_underspecified():
    a = claim("revenue from operations", 81415.38, "INR million", None, "doc-a")
    b = claim("revenue from operations", 74540.82, "INR million", "FY24", "doc-b")
    rel = reconcile_pair(a, b)
    assert rel.verdict is Verdict.UNDERSPECIFIED
    assert any(s.check == "period" and s.outcome == "blocked" for s in rel.trace)


def test_unstated_scope_against_stated_scope_is_underspecified():
    a = claim("revenue from operations", 81415.38, "INR million", "FY24", "doc-a", scope="consolidated")
    b = claim("revenue from operations", 74540.82, "INR million", "FY24", "doc-b", scope=None)
    assert reconcile_pair(a, b).verdict is Verdict.UNDERSPECIFIED


def test_currencies_are_not_converted():
    inr = claim("revenue from operations", 8142, "Rs. Cr", "FY24", "a")
    usd = claim("revenue from operations", 980, "US$ million", "FY24", "b")
    rel = reconcile_pair(inr, usd)
    assert rel.verdict is Verdict.UNDERSPECIFIED
    assert any(s.check == "unit" and s.outcome == "blocked" for s in rel.trace)


def test_adjusted_against_unlabelled_is_underspecified():
    adjusted = claim("EBITDA", 76, "Rs. Cr", "FY24", "deck", basis="adjusted")
    unlabelled = claim("EBITDA", 127, "Rs. Cr", "FY24", "ar", basis=None)
    assert reconcile_pair(adjusted, unlabelled).verdict is Verdict.UNDERSPECIFIED


def test_margins_are_not_arithmetically_contained():
    """A quarterly margin is not a fraction of the annual margin."""
    year = claim("EBITDA margin", 1.6, "per cent", "FY24", "deck")
    quarter = claim("EBITDA margin", 2.2, "per cent", "Q4 FY24", "deck")
    rel = reconcile_pair(year, quarter)
    assert rel.verdict is Verdict.COMPLEMENTARY
    assert any(s.check == "period-containment" and s.outcome == "info" for s in rel.trace)


# --- Guarding the arithmetic check ----------------------------------------

def test_containment_check_skipped_for_measures_that_go_negative():
    """Delhivery's FY23 EBITDA was -452 Cr, so a good quarter can beat the year."""
    from factlayer.reconcile import negative_capable

    fy23 = claim("EBITDA", -452, "Rs. Cr", "FY23", "deck", key="ebitda")
    year = claim("EBITDA", 76, "Rs. Cr", "FY24", "deck", key="ebitda")
    quarter = claim("EBITDA", 109, "Rs. Cr", "Q3 FY24", "deck", key="ebitda")

    negatives = negative_capable([fy23, year, quarter])
    assert "ebitda" in negatives

    # Without the guard this is a false contradiction.
    assert reconcile_pair(year, quarter).verdict is Verdict.CONTRADICTS
    assert reconcile_pair(year, quarter, negatives).verdict is Verdict.COMPLEMENTARY


def test_containment_still_applies_to_non_negative_measures():
    """Revenue never goes negative, so the inequality remains a real proof."""
    from factlayer.reconcile import negative_capable

    year = claim("revenue from services", 8142, "Rs. Cr", "FY24", "deck", key="rev")
    quarter = claim("revenue from services", 9000, "Rs. Cr", "Q4 FY24", "deck", key="rev")
    negatives = negative_capable([year, quarter])
    assert "rev" not in negatives
    assert reconcile_pair(year, quarter, negatives).verdict is Verdict.CONTRADICTS


def test_same_page_identical_qualifiers_is_underspecified():
    """Four expense lines all called "% of revenue" are rows, not contradictions."""
    a = claim("% of revenue", 34.1, "per cent", "Q4 FY24", "deck", key="pct_rev")
    b = claim("% of revenue", 18.8, "per cent", "Q4 FY24", "deck", key="pct_rev")
    rel = reconcile_pair(a, b)
    assert rel.verdict is Verdict.UNDERSPECIFIED
    assert any(s.check == "provenance" for s in rel.trace)
