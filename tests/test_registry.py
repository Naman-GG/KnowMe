"""The registry must merge synonyms without collapsing distinct measures."""

import pytest

from factlayer.registry import BatchResolver, MeasureRegistry, normalise_measure, similarity


def test_accounting_terms_are_not_stripped():
    """"total income" is a line item, not a qualified "income"."""
    assert normalise_measure("Total Income") == "total income"
    assert normalise_measure("Net Profit") == "net profit"
    assert normalise_measure("Gross Profit") == "gross profit"


def test_period_and_basis_are_stripped_from_measure_names():
    assert normalise_measure("FY24 Adjusted EBITDA") == "ebitda"
    assert normalise_measure("Q4 FY24 revenue from services") == "revenue from services"


def test_similar_names_are_proposed_as_candidates():
    """Character n-grams alone scored this pair too low to ever be considered."""
    assert similarity("revenue from operations", "revenue from services") >= 0.45


def test_clearly_distinct_measures_are_not_proposed():
    assert similarity("net profit", "gross profit") < 0.45
    assert similarity("revenue from operations", "total income") < 0.45


@pytest.mark.asyncio
async def test_deterministic_pass_needs_no_model():
    """Exact and near-exact matches resolve with no LLM available at all."""
    reg = MeasureRegistry()
    resolver = BatchResolver(reg)
    out = await resolver.resolve_many(
        ["Revenue from Operations", "revenue from operations", "FY24 EBITDA", "EBITDA"],
        llm=None,
    )
    assert out["Revenue from Operations"] == out["revenue from operations"]
    assert out["FY24 EBITDA"] == out["EBITDA"]


@pytest.mark.asyncio
async def test_merge_chain_survives_three_spellings():
    """Three names merging pairwise must not leave a dangling key."""
    reg = MeasureRegistry()
    resolver = BatchResolver(reg)

    class FakeLLM:
        usage = None
        async def json_call(self, system, user, **kw):
            n = user.count("\n1.") + user.count("\n2.") + user.count("\n3.")
            return {"decisions": [{"n": i + 1, "same": True} for i in range(n)]}

    out = await resolver.resolve_many(
        ["revenue from services", "revenue from operations", "revenue from customers"],
        llm=FakeLLM(),
    )
    assert len(set(out.values())) == 1
    assert all(k in reg.entries for k in out.values())
