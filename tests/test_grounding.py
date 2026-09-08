"""Grounding must catch fabrications without discarding honest claims."""

from factlayer.ground import GroundingReport, ground_claim, locate_quote
from factlayer.ingest import Page
from factlayer.models import (
    Claim,
    ClaimStatus,
    Evidence,
    GroundingStatus,
    Qualifiers,
)

PAGE_TEXT = (
    "1.4 Mn Tons\nPTL freight tonnage in FY24\nYoY: 29.8%\n"
    "₹8,142 Cr\nFY24 revenue from services\nYoY: 12.7%(2)\n"
    "₹127Cr / 1.6%\nEBITDA / EBITDA margin\nFY23: ₹(452) Cr / (6.3%)\n"
    "₹76Cr / 0.9%\nAdj. EBITDA / Adj. EBITDA margin\n"
)


def page() -> Page:
    return Page(page_no=6, text=PAGE_TEXT)


def claim(quote: str, **quals) -> Claim:
    return Claim(
        id="t1",
        doc_id="d1",
        subject_raw="Delhivery Limited",
        measure_raw="revenue from services",
        value_kind="text",
        text_value="x",
        qualifiers=Qualifiers(**quals),
        evidence=Evidence(doc_id="d1", page_no=6, quote=quote),
    )


def test_exact_quote_is_located_with_offsets():
    status, start, end = locate_quote("FY24 revenue from services", page())
    assert status is GroundingStatus.EXACT
    assert PAGE_TEXT[start:end] == "FY24 revenue from services"


def test_quote_broken_across_lines_still_grounds():
    """PDF extraction splits phrases; that must not read as a fabrication."""
    status, _, _ = locate_quote("₹8,142 Cr FY24 revenue from services", page())
    assert status is GroundingStatus.FUZZY


def test_invented_quote_is_quarantined():
    c = claim("Revenue for FY24 was ₹9,900 Cr as restated")
    ground_claim(c, page(), GroundingReport())
    assert c.status is ClaimStatus.QUARANTINED
    assert "not found" in c.quarantine_reason


def test_fabricated_scope_is_dropped_and_recorded():
    """The observed failure: a scope no part of the page ever mentions."""
    c = claim("FY24 revenue from services", scope="standalone and consolidated")
    report = GroundingReport()
    ground_claim(c, page(), report)
    assert c.status is ClaimStatus.ACTIVE       # the claim itself is fine
    assert c.qualifiers.scope is None           # the invented qualifier is not
    assert c.qualifiers.extra["dropped_scope"] == "standalone and consolidated"
    assert report.qualifiers_dropped == 1


def test_abbreviated_qualifier_is_kept():
    """The page writes "Adj."; the extractor writes "adjusted". Same thing."""
    c = claim("Adj. EBITDA / Adj. EBITDA margin", basis="adjusted")
    ground_claim(c, page(), GroundingReport())
    assert c.qualifiers.basis == "adjusted"


def test_grounding_report_counts():
    report = GroundingReport()
    ground_claim(claim("FY24 revenue from services"), page(), report)
    ground_claim(claim("a figure that does not appear anywhere"), page(), report)
    d = report.as_dict()
    assert d["claims_checked"] == 2
    assert d["quarantined"] == 1
    assert d["grounding_rate"] == 0.5


def test_elided_quote_grounds_as_gapped():
    """The model drops the middle of "Q4 FY23: Rs.13 Cr / 0.7%" but invents nothing."""
    p = Page(page_no=6, text="Q4 FY23: ₹13 Cr / 0.7%\nEBITDA / EBITDA margin\n")
    status, _, _ = locate_quote("Q4 FY23: 0.7%", p)
    assert status is GroundingStatus.GAPPED


def test_gapped_matching_still_rejects_fabrication():
    """Tolerating elision must not tolerate invented figures."""
    p = Page(page_no=6, text="Q4 FY23: ₹13 Cr / 0.7%\nEBITDA / EBITDA margin\n")
    status, _, _ = locate_quote("Q4 FY23: ₹99 Cr restated under Ind AS", p)
    assert status is GroundingStatus.NOT_FOUND


def test_out_of_order_tokens_are_not_grounded():
    """In-order is part of the test; scrambled tokens are not evidence."""
    p = Page(page_no=1, text="revenue from services was 8,142 Cr in FY24")
    status, _, _ = locate_quote("FY24 8,142 services revenue from was Cr in", p)
    assert status is GroundingStatus.NOT_FOUND


def test_printed_page_ignores_numbers_in_body_text():
    """A slide reading "740 Mn" must not be cited as page 740."""
    from factlayer.ingest import _printed_page_number

    slide = "740 Mn\nExpress parcel shipments in FY24\nYoY: 11.5%\n8,142 Cr\n"
    assert _printed_page_number(slide) is None

    numbered = "Annual Report 2024-25\nsome body text here\n27\n"
    assert _printed_page_number(numbered) == "27"


def test_short_quote_is_unverifiable_not_missing():
    """"3.28" is on the page, but a bare figure is not evidence.

    Reporting this as "not found" implied the model had invented it. It had not;
    the quote simply cannot be checked, which is a different failure.
    """
    p = Page(page_no=11, text="Lowest Price - Highest Price\n148.38\n3.28\nNil - 400.00\n")
    assert "3.28" in p.text
    status, _, _ = locate_quote("3.28", p)
    assert status is GroundingStatus.UNVERIFIABLE
    assert status is not GroundingStatus.NOT_FOUND


def test_the_two_quarantine_kinds_are_counted_apart():
    p = Page(page_no=11, text="Lowest Price - Highest Price\n148.38\n3.28\n")
    report = GroundingReport()
    ground_claim(claim("3.28"), p, report)                       # too short
    ground_claim(claim("a span that is nowhere on this page"), p, report)  # absent
    d = report.as_dict()
    assert d["quarantined"] == 2
    assert d["unverifiable"] == 1
    assert d["not_found"] == 1
