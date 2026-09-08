# Limitations, assumptions and honest anomalies

Written adversarially against my own system. The assignment asks for an
extraction or reasoning failure and how it was handled; this goes further and
separates the fixes that are **principled** from the ones that are **tuned to
this particular corpus**, because that distinction decides whether the thing
generalises.

---

## 1. What is genuinely principled

These would behave correctly on documents I have never seen. Nothing about them
encodes an answer.

| Mechanism | Why it generalises |
|---|---|
| Interval algebra over periods | Allen relations over dates; no vocabulary involved |
| Unit canonicalisation | Scale words map to multipliers; unknown units return `None` rather than a guess |
| Quote grounding | Structural: is this text on this page, in this order |
| Qualifier grounding | Structural: did the page actually say this qualifier |
| Currencies never converted | A refusal, not a rule that can be wrong |
| Unstated qualifier ≠ wildcard | `UNKNOWN`, never a match |
| Same-page / same-qualifier suppression | Structural, uses no domain words |
| `negative_capable()` | **Read off the corpus at runtime** — any measure ever seen negative is exempt from the containment check. New domains get this right with no new code |
| Incremental blocking on (subject, measure) | Structural |
| Content-hash document ids | Structural |

## 2. Domain assumptions that would produce *wrong* answers elsewhere

These are the ones that worry me, because they fail silently rather than
loudly.

### 2.1 The fiscal year is hardcoded to April–March

`normalize/periods.py` sets `FY_START_MONTH = 4`, and `fiscal_year(2024)`
returns `2023-04-01 .. 2024-03-31`.

That is correct for every document in this corpus and for Indian filings
generally. It is **wrong** for a US 10-K on an October–September year, and it
would not error — it would confidently produce a mis-dated interval and then
reason correctly from a false premise. A wrong answer is worse than a missing
one, and this is the clearest example of that risk in the codebase.

*What it should be:* the fiscal-year convention inferred per document during
profiling (the document usually states its year end), stored on the profile,
and threaded into period parsing. The parser already accepts explicit forms
like "year ended 31 March 2024" correctly regardless of convention; only the
bare `FY24` shorthand depends on the assumption.

### 2.2 Few-shot examples in the prompts come from these corpora

The measure-adjudication prompt tells the model that `revenue from operations`
vs `total income`, `EBITDA` vs `EBIT`, `GDP` vs `GVA` and `headline` vs `core`
inflation are distinct. Those are real accounting and macroeconomic
distinctions, not invented ones — but I chose them **after** reading these six
documents, and they are exactly the distinctions the demo depends on.

This is the most "tuned to pass" thing in the repository. It is defensible as
domain guidance, and the *conservative default* ("when unsure, answer false")
does generalise. But a fair reading is that measure disambiguation is
better-prompted here than it would be on an unfamiliar domain.

The extraction prompt has the same issue in milder form: its worked example
(`FY24 Adjusted EBITDA: Rs. 76 Cr`) is close to a real line in the earnings
deck.

### 2.3 Curated word lists

* `QUALIFIER_ALIASES` (`adjusted` ↔ `adj.`, `consolidated` ↔ `group`, …) — used
  to avoid dropping a true qualifier that the page abbreviates. On a new domain
  it under-matches, so grounding would discard *correct* qualifiers.
* `_QUALIFIER_WORDS` in the registry, the scope lattice's
  `CONSOLIDATED`/`STANDALONE`/`TOTAL` sets, and triage's entity tokens
  (`director`, `resigned`, `registered office`, `CIN`, `DIN`) are all
  corporate-filing vocabulary chosen by reading this corpus.
* `_is_additive()` decides whether a quantity aggregates by looking for
  `margin`, `rate`, `growth`, `ratio`, `share` in the measure name. English
  only, and it would misjudge an unfamiliar phrasing.

### 2.4 Constants tuned by looking at results

| Constant | Value | Risk if wrong |
|---|---|---|
| `VALUE_TOLERANCE` | 1% | **A genuine restatement smaller than 1% is reported as agreement.** Real risk: restatements are often small |
| `TOKEN_COVERAGE_THRESHOLD` | 0.85 | Too low admits loose paraphrase as evidence; too high quarantines honest claims |
| `_SIM_THRESHOLD` | 0.45 | Too low wastes adjudication calls; too high silently loses cross-document links |
| triage `min_score` | 1.5 | Too high drops fact-bearing pages |

None of these were derived; all were chosen because they produced sensible
output on these six documents.

## 3. Deliberate precision/recall trade-offs

Both of these *suppress* findings and could hide something real:

* **Same-page suppression.** Two claims from one page with identical qualifiers
  and different values are reported UNDERSPECIFIED, on the reasoning that a
  table lost its row labels. If a document genuinely contradicts itself twice on
  one page, this system will not say so.
* **Negative-capable exemption.** Excluding profit-like measures from the
  containment check removes ~38 false contradictions, but it also means a
  *genuine* part-exceeds-whole error in a profit measure now goes undetected.

Both cut false positives sharply. Both cost recall, and I chose precision
because a knowledge layer that cries wolf is worse than one that is quiet.

## 4. Weak evaluation — the biggest gap

**There is no labelled ground truth, so there are no real precision or recall
numbers for extraction or reconciliation anywhere in this project.**

`scripts/eval_triage.py` reports 9/9 recall, but those nine targets are spans
**I chose myself** after reading the documents. That is a regression smoke test
— it proves a change did not break something I already knew about. It is not a
benchmark, and quoting it as one would be misleading.

Everything I claim about quality below the triage layer rests on hand-inspection
of samples. With more time the first thing I would build is a small labelled set
(a few hundred claims with correct qualifiers, and a few dozen labelled pairs)
so that tuning is measured instead of eyeballed.

## 4a. The most consequential failure found: publisher mistaken for subject

**Symptom.** The RBI Annual Report and the IMF Article IV report produce
**zero** cross-document relations between them, despite both reporting Indian
GDP growth, inflation and the fiscal position for overlapping periods.

**Diagnosis.**

```
RBI claims  ->  subject_key = 'reserve bank of india'   (194 of 229)
IMF claims  ->  subject_key = 'india'                   (196 of 199)
```

The extractor recorded the RBI document's subject as its **publisher** rather
than as the entity its facts describe. A central bank's annual report is about
the economy, not about the bank. Because reconciliation blocks on
(subject, measure), that one mislabel silently suppressed every comparison
between the two documents.

**Why it is worth reporting.** This is the failure mode this project is least
able to see. A wrong *value* shows up as a contradiction; a wrong *subject*
produces no output at all, and an empty result looks exactly like "these
documents have nothing in common". It was found only by asking why an expected
comparison was missing — which is why `scripts/audit.py` exists.

**Fix, not yet applied.** Two prompt changes: profiling must distinguish the
organisation that *published* a document from the entity its facts *describe*,
and extraction must be told that a claim's subject is what the figure measures,
never the publisher's name.

It is not applied in this commit for an honest reason: changing the extraction
prompt changes the cache key for every call, and with the provider's daily
token allowance spent, re-extracting the corpus is impossible tonight. Applying
it would leave a repository whose code could not reproduce its own stored
results. The diagnosis is recorded here, and
`tests/test_subject_resolution.py` pins the current behaviour so the fix is
verifiable when quota allows.

**A partial mitigation was considered and rejected.** Subject matching could
treat "India" as contained in "Reserve Bank of India" by token overlap. That
would also merge "Bank of America" with "America", inventing agreement between
a company and a country -- a worse failure than the one it fixes.

## 5. Known anomalies that remain

Run `python scripts/audit.py` to reproduce these against the current database.

1. **Fabricated scopes are frequent.** The model invented
   `scope = "standalone and consolidated"` on every claim of one slide and kept
   doing so after the prompt forbade it. Grounding catches these, but the
   underlying behaviour is unfixed — I am detecting a lie rather than preventing
   it, and a fabricated qualifier that *happens* to appear on the page would
   pass.
2. **Table row labels are lost.** Four expense lines each extracted as
   `"% of revenue"` with identical qualifiers. Suppressed rather than solved;
   the real fix is table-aware extraction that carries the row heading into the
   measure name.
3. **Quotes are elided, not verbatim.** Despite an explicit instruction, the
   model abridges (`"Q4 FY23: 0.7%"` from `"Q4 FY23: ₹13 Cr / 0.7%"`). Handled
   by the `GAPPED` grounding tier, which is a deliberate weakening of the
   verbatim requirement.
4. **Free-tier rate limiting loses pages.** Sustained extraction trips Groq's
   limiter; with insufficient backoff, whole pages failed and were silently
   skipped. Mitigated with global request pacing and longer retries, but a run
   can still lose pages, and the pipeline reports this rather than pretending to
   full coverage.
5. **`as_of` is inherited from the document, not the claim.** A single filing
   restating a prior-year figure carries the filing's date on both claims, so
   supersession cannot distinguish them.
6. **Subject resolution is shallow.** Normalised string similarity merges
   `Delhivery Limited` / `Delhivery Ltd`, but would not connect a subsidiary to
   its parent, and could over-merge two similarly-named entities.
7. **Coverage is incomplete for this run.** Groq's free tier allows 200,000
   tokens per day; a page of extraction costs roughly 4,000, and the corpus is
   511 pages -- about six days of allowance. Two documents (the Delhivery
   annual report and the Economic Survey) have **zero** claims because their
   pages were never extracted. The pipeline reports this as
   `pages_skipped_no_quota` rather than implying full coverage, but every
   corpus-level number below is drawn from four of six documents.
8. **Only one PDF text layer is used.** Scanned or image-only PDFs yield
   nothing; there is no OCR fallback, and such a document would report zero
   claims rather than an error.
