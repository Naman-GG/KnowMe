import pytest

from factlayer.normalize.units import canonicalise, parse_unit


@pytest.mark.parametrize(
    "raw,dimension,base,scale",
    [
        ("Rs. Cr", "currency", "INR", 1e7),
        ("₹ Cr", "currency", "INR", 1e7),
        ("₹ crore", "currency", "INR", 1e7),
        ("INR million", "currency", "INR", 1e6),
        ("₹ million", "currency", "INR", 1e6),
        ("Rs lakh", "currency", "INR", 1e5),
        ("US$ billion", "currency", "USD", 1e9),
        ("per cent", "percent", "percent", 1.0),
        ("%", "percent", "percent", 1.0),
        ("percent", "percent", "percent", 1.0),
        ("bps", "percent", "percent", 0.01),
        ("Mn Tons", "mass", "tonnes", 1e6),
        ("tonnes", "mass", "tonnes", 1.0),
        ("Mn parcels", "count", "items", 1e6),
        ("Mn", "count", "items", 1e6),
    ],
)
def test_parse_unit(raw, dimension, base, scale):
    u = parse_unit(raw)
    assert u is not None, f"failed to parse {raw!r}"
    assert (u.dimension, u.base, u.scale) == (dimension, base, scale)


def test_unrecognised_units_return_none():
    # We would rather report "cannot compare" than invent a scale.
    for raw in ("", None, "widgets per fortnight", "   "):
        assert parse_unit(raw) is None


def test_crore_and_million_agree():
    """The corroboration case: same revenue, two scales, two documents."""
    deck = canonicalise(8142.0, parse_unit("Rs. Cr"))            # Q4 FY24 deck
    report = canonicalise(81415.38, parse_unit("INR million"))   # annual report
    assert deck is not None and report is not None
    # Within 0.1% -- the deck rounds to the nearest crore.
    assert abs(deck - report) / report < 0.001


def test_currencies_are_never_converted():
    inr, usd = parse_unit("Rs. Cr"), parse_unit("US$ million")
    assert not inr.comparable_with(usd)


def test_bps_converts_to_percent():
    assert canonicalise(1048.0, parse_unit("bps")) == pytest.approx(10.48)
