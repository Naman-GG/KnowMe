"""Subject resolution, and the failure it is currently hiding.

Reconciliation blocks on (subject, measure), so a wrong subject produces no
output at all rather than a wrong answer. These tests pin the behaviour that
caused the RBI and IMF documents to share no relations, so that the prompt fix
described in docs/LIMITATIONS.md can be verified when it is applied.
"""

from factlayer.registry import normalise_subject, similarity


def test_company_suffixes_and_honorifics_are_stripped():
    assert normalise_subject("Delhivery Limited") == normalise_subject("Delhivery Ltd")
    assert normalise_subject("Mr. Sandeep Kumar Barasia") == "sandeep kumar barasia"


def test_publisher_and_subject_do_not_match_today():
    """The observed failure, pinned.

    The RBI document's claims were labelled with the publisher's name while the
    IMF document's were labelled with the country. Their similarity is below the
    0.6 threshold reconcile_pair requires, so no comparison is ever attempted --
    which is why the two macro documents produced zero relations between them.
    """
    rbi = normalise_subject("Reserve Bank of India")
    imf = normalise_subject("India")
    assert rbi != imf
    assert similarity(rbi, imf) < 0.6


def test_naive_containment_would_be_worse():
    """Why the tempting shortcut is not taken.

    Treating a shorter subject as matching any longer one that contains it
    would link "India" to "Reserve Bank of India" -- and equally would link
    "America" to "Bank of America", inventing agreement between a country and
    a company. The fix belongs in extraction, not in matching.
    """
    assert "america" in normalise_subject("Bank of America")
    assert normalise_subject("Bank of America") != normalise_subject("America")
