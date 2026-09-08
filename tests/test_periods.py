import datetime as dt

import pytest

from factlayer.normalize.periods import parse_period, relate

D = dt.date


@pytest.mark.parametrize(
    "label,start,end",
    [
        # Indian fiscal year, named for the year it ends in.
        ("FY24", D(2023, 4, 1), D(2024, 3, 31)),
        ("FY 2024", D(2023, 4, 1), D(2024, 3, 31)),
        ("FY2023-24", D(2023, 4, 1), D(2024, 3, 31)),
        # IMF style: first year is the start year.
        ("FY2025/26", D(2025, 4, 1), D(2026, 3, 31)),
        # Government reports' bare fiscal span.
        ("2024-25", D(2024, 4, 1), D(2025, 3, 31)),
        # Fiscal quarters count from April, so Q4 lands in the next calendar year.
        ("Q1 FY24", D(2023, 4, 1), D(2023, 6, 30)),
        ("Q4 FY24", D(2024, 1, 1), D(2024, 3, 31)),
        ("Q4 FY23", D(2023, 1, 1), D(2023, 3, 31)),
        # Calendar quarter, IMF style -- deliberately not the same as Q2 FY25.
        ("2025Q2", D(2025, 4, 1), D(2025, 6, 30)),
        # Explicit spans.
        ("year ended March 31, 2024", D(2023, 4, 1), D(2024, 3, 31)),
        ("year ended 31 March 2024", D(2023, 4, 1), D(2024, 3, 31)),
        ("quarter ended 30 June 2023", D(2023, 4, 1), D(2023, 6, 30)),
        # Instants.
        ("as at March 31, 2024", D(2024, 3, 31), D(2024, 3, 31)),
        ("as on March 21, 2025", D(2025, 3, 21), D(2025, 3, 21)),
        # Calendar year.
        ("2024", D(2024, 1, 1), D(2024, 12, 31)),
    ],
)
def test_parse_period(label, start, end):
    p = parse_period(label)
    assert p is not None, f"failed to parse {label!r}"
    assert (p.start, p.end) == (start, end), label


def test_fy24_and_year_ended_march_2024_are_the_same_span():
    """Two documents, two phrasings, one period."""
    assert relate(parse_period("FY24"), parse_period("year ended March 31, 2024")) == "equals"


def test_quarter_nests_inside_its_fiscal_year():
    """The Delhivery case: Q4 FY24 revenue sits inside FY24 revenue."""
    assert relate(parse_period("FY24"), parse_period("Q4 FY24")) == "contains"
    assert relate(parse_period("Q4 FY24"), parse_period("FY24")) == "during"


def test_calendar_year_is_not_the_fiscal_year():
    """CY2024 and FY24 overlap but are different spans -- a real reconciliation axis."""
    assert relate(parse_period("2024"), parse_period("FY24")) == "overlaps"


def test_calendar_quarter_is_not_the_fiscal_quarter():
    assert relate(parse_period("2025Q2"), parse_period("Q2 FY25")) != "equals"


def test_consecutive_years_are_disjoint():
    assert relate(parse_period("FY23"), parse_period("FY24")) == "before"


def test_undatable_period_is_reported_not_guessed():
    p = parse_period("during the period under review")
    assert p is not None and not p.is_resolved
    assert relate(p, parse_period("FY24")) is None


def test_no_time_expression_returns_none():
    assert parse_period("") is None
    assert parse_period(None) is None


@pytest.mark.parametrize(
    "label,start,end",
    [
        ("April-December 2024", D(2024, 4, 1), D(2024, 12, 31)),
        ("April – November 2024", D(2024, 4, 1), D(2024, 11, 30)),
        ("April to December 2024", D(2024, 4, 1), D(2024, 12, 31)),
        ("November-February 2025", D(2025, 11, 1), D(2026, 2, 28)),
        ("H1 2024", D(2024, 1, 1), D(2024, 6, 30)),
        ("H2 2024", D(2024, 7, 1), D(2024, 12, 31)),
    ],
)
def test_month_ranges_and_calendar_halves(label, start, end):
    p = parse_period(label)
    assert p is not None and (p.start, p.end) == (start, end), label


def test_part_year_is_not_the_whole_year():
    """The RBI reported "April-December 2024" and the IMF "2024".

    Parsing the part-year as the full year made them compare EQUAL, so a
    nine-month figure and a twelve-month one were reported as contradicting.
    """
    part = parse_period("April-December 2024")
    whole = parse_period("2024")
    assert relate(part, whole) != "equals"
    assert relate(whole, part) == "contains"
