"""Period parsing: turn time expressions into comparable date intervals.

This is the single most valuable normaliser in the system. Most apparent
contradictions in the starter corpus are period mismatches -- an annual figure
set against a quarterly one, a fiscal year against a calendar year, one
forecast vintage against another. Once periods are intervals, the reconciler
can answer "are these even talking about the same span of time?" with interval
algebra instead of an opinion.

Conventions that matter here:

* The Indian fiscal year runs 1 April to 31 March and is named for the year it
  *ends* in: FY24 is 2023-04-01 .. 2024-03-31.
* Written as a span, the first year is the start: "2024-25" and "FY2025/26"
  begin in April of the first year.
* Fiscal quarters are numbered from April: Q1 FY24 is Apr-Jun 2023, and
  Q4 FY24 is Jan-Mar 2024 -- in a different calendar year to Q1.
* "2025Q2" (the IMF's style) is a *calendar* quarter, Apr-Jun 2025. It is not
  the same span as Q2 FY25, and conflating the two would be a real error.
"""

from __future__ import annotations

import calendar
import datetime as dt
import re
from typing import Literal

from ..models import Period

FY_START_MONTH = 4  # April

MONTHS = {
    m.lower(): i
    for i, m in enumerate(calendar.month_name)
    if m
}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})

_MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))

# "year ended March 31, 2024" / "year ended 31 March 2024"
_ENDED = re.compile(
    rf"\b(?P<kind>year|quarter|half[\s-]?year|month)\s+ended\b\s*(?:on\s+)?"
    rf"(?:(?P<d1>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<m1>{_MONTH_RE})|"
    rf"(?P<m2>{_MONTH_RE})\s+(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?)"
    rf",?\s*(?P<y>\d{{4}})",
    re.IGNORECASE,
)
# "as at March 31, 2024" / "as on 21 March 2025" -> a point in time
_AS_OF = re.compile(
    rf"\bas\s+(?:at|on|of)\b\s*"
    rf"(?:(?P<d1>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<m1>{_MONTH_RE})|"
    rf"(?P<m2>{_MONTH_RE})\s+(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?)"
    rf",?\s*(?P<y>\d{{4}})",
    re.IGNORECASE,
)
# Fiscal quarter: "Q4 FY24", "Q4FY2024", "Q1 FY 2025"
_FQ = re.compile(r"\bQ(?P<q>[1-4])\s*FY\s*(?P<y>\d{2,4})\b", re.IGNORECASE)
# Calendar quarter, IMF style: "2025Q2"
_CQ = re.compile(r"\b(?P<y>\d{4})\s*Q(?P<q>[1-4])\b", re.IGNORECASE)
# Fiscal year: "FY24", "FY 2024", "FY2023-24", "FY2025/26"
_FY = re.compile(
    r"\bFY\s*(?P<y1>\d{2,4})(?:\s*[-/–]\s*(?P<y2>\d{2,4}))?\b", re.IGNORECASE
)
# Bare fiscal span used by government reports: "2024-25", "2023-24"
_SPAN = re.compile(r"\b(?P<y1>(?:19|20)\d{2})\s*[-/–]\s*(?P<y2>\d{2,4})\b")
# Half year: "H1 FY24"
_HALF = re.compile(r"\bH(?P<h>[12])\s*FY\s*(?P<y>\d{2,4})\b", re.IGNORECASE)
# A lone calendar year.
_CY = re.compile(r"\b(?P<y>(?:19|20)\d{2})\b")

Relation = Literal[
    "equals", "contains", "during", "overlaps", "before", "after", "disjoint"
]


def _yy(raw: str, pivot: int = 2000) -> int:
    """Expand a 2-digit year. 24 -> 2024."""
    y = int(raw)
    return y if y >= 1000 else pivot + y


def _eom(year: int, month: int) -> dt.date:
    return dt.date(year, month, calendar.monthrange(year, month)[1])


def fiscal_year(end_year: int) -> tuple[dt.date, dt.date]:
    """FY named by the year it ends in: FY24 -> 2023-04-01 .. 2024-03-31."""
    return dt.date(end_year - 1, FY_START_MONTH, 1), _eom(end_year, FY_START_MONTH - 1)


def fiscal_quarter(end_year: int, q: int) -> tuple[dt.date, dt.date]:
    """Q1 of FY24 is Apr-Jun 2023; Q4 of FY24 is Jan-Mar 2024."""
    start_month = FY_START_MONTH + 3 * (q - 1)
    year = end_year - 1 + (start_month - 1) // 12
    start_month = (start_month - 1) % 12 + 1
    start = dt.date(year, start_month, 1)
    end_month = start_month + 2
    end_year_ = year + (end_month - 1) // 12
    end_month = (end_month - 1) % 12 + 1
    return start, _eom(end_year_, end_month)


def calendar_quarter(year: int, q: int) -> tuple[dt.date, dt.date]:
    start = dt.date(year, 3 * (q - 1) + 1, 1)
    return start, _eom(year, 3 * (q - 1) + 3)


def _month_day(m: re.Match) -> tuple[int, int]:
    if m.group("m1"):
        return MONTHS[m.group("m1").lower()], int(m.group("d1"))
    return MONTHS[m.group("m2").lower()], int(m.group("d2"))


def parse_period(label: str | None) -> Period | None:
    """Resolve a written period to a dated interval.

    Returns a Period with `start`/`end` set when the expression is understood,
    a Period with them left None when the words look like a period but cannot be
    dated, and None when the string carries no time expression at all. The
    middle case is what drives an UNDERSPECIFIED verdict downstream.
    """
    if not label or not label.strip():
        return None
    text = label.strip()

    # "quarter/year ended <date>" -- most precise, so tried first.
    if (m := _ENDED.search(text)):
        month, day = _month_day(m)
        end = dt.date(int(m.group("y")), month, day)
        kind = m.group("kind").lower().replace(" ", "").replace("-", "")
        months = {"year": 12, "halfyear": 6, "quarter": 3, "month": 1}[kind]
        start_month_total = (end.year * 12 + end.month) - months
        start = dt.date(start_month_total // 12, start_month_total % 12 + 1, 1)
        return Period(label=text, start=start, end=end)

    if (m := _AS_OF.search(text)):
        month, day = _month_day(m)
        d = dt.date(int(m.group("y")), month, day)
        return Period(label=text, start=d, end=d)  # an instant

    if (m := _FQ.search(text)):
        s, e = fiscal_quarter(_yy(m.group("y")), int(m.group("q")))
        return Period(label=text, start=s, end=e)

    if (m := _CQ.search(text)):
        s, e = calendar_quarter(int(m.group("y")), int(m.group("q")))
        return Period(label=text, start=s, end=e)

    if (m := _HALF.search(text)):
        end_year = _yy(m.group("y"))
        q_start = 1 if m.group("h") == "1" else 3
        s, _ = fiscal_quarter(end_year, q_start)
        _, e = fiscal_quarter(end_year, q_start + 1)
        return Period(label=text, start=s, end=e)

    if (m := _FY.search(text)):
        y1 = _yy(m.group("y1"))
        # "FY2023-24" / "FY2025/26": the first year is the start year, so the
        # fiscal year is named for the second.
        end_year = _yy(m.group("y2"), pivot=(y1 // 100) * 100) if m.group("y2") else y1
        if m.group("y2") and end_year <= y1:
            end_year = y1 + 1
        s, e = fiscal_year(end_year)
        return Period(label=text, start=s, end=e)

    # "2024-25" without an FY prefix: government reports' fiscal span.
    if (m := _SPAN.search(text)):
        y1 = int(m.group("y1"))
        y2 = _yy(m.group("y2"), pivot=(y1 // 100) * 100)
        if y2 == y1 + 1:
            s, e = fiscal_year(y2)
            return Period(label=text, start=s, end=e)

    if (m := _CY.search(text)):
        y = int(m.group("y"))
        return Period(label=text, start=dt.date(y, 1, 1), end=dt.date(y, 12, 31))

    # Looks like a period but we could not date it -- say so rather than guess.
    return Period(label=text, start=None, end=None)


def relate(a: Period, b: Period) -> Relation | None:
    """Interval relation between two resolved periods, or None if undatable."""
    if not (a.is_resolved and b.is_resolved):
        return None
    if a.start == b.start and a.end == b.end:
        return "equals"
    if a.start <= b.start and a.end >= b.end:
        return "contains"
    if b.start <= a.start and b.end >= a.end:
        return "during"
    if a.end < b.start:
        return "before"
    if b.end < a.start:
        return "after"
    return "overlaps"


def overlaps(a: Period, b: Period) -> bool:
    rel = relate(a, b)
    return rel is not None and rel not in ("before", "after", "disjoint")
