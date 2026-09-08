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
