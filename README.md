# Fact Knowledge Layer

Extracts qualified factual claims from PDFs, grounds every claim in the exact
text that supports it, and reconciles claims against each other across
documents — reporting not just *that* two figures differ, but *why*, with the
derivation shown.

> **The idea in one sentence.** A contradiction is a claim about **qualifiers**,
> not about values. Two numbers differing is only a conflict if they cover the
> same period, scope, basis and entity — so this system extracts the context
> that makes a number mean something, and makes the comparison auditable rather
> than a matter of opinion.

---

## Setup and run instructions

Requires Python 3.11+ and an API key for any OpenAI-compatible endpoint. The
default targets **Groq's free tier**, so this runs without a paid account.

```bash
git clone <this repo> && cd KnowMe

# 1. environment
uv venv --python 3.12 && uv pip install -e ".[dev]"
#    (or: python -m venv .venv && .venv/bin/pip install -e ".[dev]")

# 2. credentials -- .env is gitignored
cp .env.example .env
#    then set FACTLAYER_LLM_API_KEY (GROQ_API_KEY / OPENAI_API_KEY also work)

# 3. tests -- 80 of them, none of which need the network
.venv/bin/python -m pytest -q

# 4. ingest the starter corpora
.venv/bin/python scripts/ingest.py --all

# 5. browse the results
.venv/bin/python -m uvicorn factlayer.api:app --app-dir src --reload
#    then open http://127.0.0.1:8000
```

Upload further PDFs through the UI, or:

```bash
.venv/bin/python scripts/ingest.py path/to/another.pdf
```

Useful extras:

```bash
.venv/bin/python scripts/demo.py                 # the four required cases, from live data
.venv/bin/python scripts/audit.py                # adversarial audit of the results
.venv/bin/python scripts/rebuild.py              # re-normalise + re-reconcile, no model calls
.venv/bin/python scripts/refresh_subjects.py     # re-profile documents, propagate subjects
.venv/bin/python scripts/eval_triage.py --sweep  # page-selection recall
```

**A note on rate limits.** Groq's free tier allows 8,000 tokens per minute, and
a page of extraction costs roughly 3,000, so a full 511-page ingest takes a few
hours. Every LLM call is cached to disk by a hash of its inputs, so re-runs are
instant and an interrupted run resumes where it stopped. Use `--budget 30` to
cap pages per document for a faster look.

## Video demo

*(link to be added)*

---

## Approach

### The problem with the obvious design

The default shape for this task is: chunk the PDFs, extract triples, embed
them, pair up neighbours, and ask an LLM "do these contradict?". That produces
a system whose central judgement is unauditable — it cannot tell you why it
decided anything — and it is wrong in a specific, predictable way: it compares
**values** while ignoring the **context** that makes values comparable.

In this corpus, Delhivery's FY24 revenue appears as all of these:

| Figure | Qualifier that explains it | Source |
|---|---|---|
| ₹8,142 Cr | revenue from *services*, consolidated | Q4 FY24 deck |
| ₹81,415.38 mn | revenue from *operations*, consolidated | Annual report |
| ₹74,540.82 mn | revenue from operations, **standalone** | Annual report, same page |
| ₹85,942.34 mn | **total income** (adds other income) | Annual report, MD&A |
| ₹2,076 Cr | **Q4 only**, not the full year | Q4 FY24 deck |

A value-first system reports four contradictions. Every one of them is
explained by a qualifier, and the first two are the *same fact* written in
different units.

### The claim model

The atom is not a triple. It is a value plus the qualifiers that give it
meaning:

```python
Claim(
  subject   = "Delhivery Limited",
  measure   = "revenue from operations",     # canonicalised against a growing registry
  quantity  = Quantity(low=81415.38, high=81415.38, unit_raw="INR million",
                       canonical_low=8.141538e10),   # rescaled to rupees
  qualifiers = Qualifiers(
      period = Period("FY24", 2023-04-01, 2024-03-31),
      scope  = "consolidated",
      basis  = None,
      as_of  = 2024-08-01),                  # inherited from the document
  evidence  = Evidence(doc, page=22, quote="...stood at ₹ 74,540.82 million...",
                       grounding=EXACT),
)
```

Two design decisions do a lot of work:

**Numeric values are closed intervals.** A point estimate is stored as
`low == high`. That makes "does RBI's 6.5% agree with the Economic Survey's
6.3–6.8%?" plain interval containment instead of a special case. Periods get
the same treatment, so period comparison becomes interval algebra.

**Four deep qualifier slots, plus an open extension space.** `period`, `scope`,
`basis` and `as_of` have real logic behind them. Anything else a document
introduces lands in `extra` and is compared generically — same, different, or
missing. The schema grows with the corpus without pretending to understand
every new dimension.

### The reconciler decides; the model only reads

Given two claims, deciding whether they agree is a sequence of concrete checks,
each appending a line to a trace. **Values are compared last**, and only once
the qualifiers have established that comparing them means anything.

```
A: revenue from services = 8,142 Rs. Cr        [q4-deck]
B: revenue from services = 81,415.38 INR million  [annual-report]
  ✓ subject     'Delhivery Limited' ≡ 'Delhivery Limited'
  ✓ measure     both → canonical 'revenue_from_services'
  ✓ unit        'Rs. Cr' and 'INR million' both rescale to currency
  ✓ period      'FY24' [2023-04-01→2024-03-31] EQUALS 'FY24' [2023-04-01→2024-03-31]
  ✓ scope       unstated vs unstated → same
  ✓ value       differ by 0.006% (within rounding)
  ⇒ CORROBORATES
```

The verdict vocabulary:

| Verdict | Condition |
|---|---|
| `CORROBORATES` | qualifiers compatible, values agree |
| `CONTRADICTS` | qualifiers compatible, values disagree |
| `COMPLEMENTARY` | qualifiers disjoint — both can be true |
| `SUPERSEDES` | same claim, later `as_of`; the world changed |
| `UNDERSPECIFIED` | a qualifier needed for the judgement is missing |

The assignment asks for three relations. *Reconcile* splits into two genuinely
different mechanisms — disjoint context, versus time having moved on — and
`UNDERSPECIFIED` is the honest fifth answer that most systems guess at.

### Contradiction by arithmetic, not by opinion

When one period contains another, or one scope subsumes another, a non-negative
additive quantity **must** satisfy an inequality: a quarter cannot exceed its
own year, a standalone figure cannot exceed the consolidated one. A violation is
a contradiction *proved* from the documents.

```
  i period              'FY24' CONTAINS 'Q4 FY24'
  ✕ period-containment  part 90,000,000,000 exceeds whole 81,420,000,000
  ⇒ CONTRADICTS
```

This check knows its own limits. It does not apply to margins, rates or ratios
(a quarterly margin is not a fraction of the annual one), and it does not apply
to measures that can go negative — see *Findings* below.

### Grounding, including of qualifiers

Every claim carries a verbatim span, checked against the exact page text the
model was shown. No second model call, no self-grading. Matching has three
tiers — exact, whitespace-normalised, and `GAPPED` (all tokens present in
order, where the model abridged) — and anything else is **quarantined**, kept
for inspection but excluded from reconciliation.

Grounding also covers **qualifiers**, which turned out to matter more. See
*Findings*.

### Deliberately not built

- **No graph visualisation.** The brief says a graph alone is not the solution,
  and a force-directed blob of a few thousand claims shows *that* things connect
  without showing *why*. The UI is a review queue instead.
- **No vector database.** This corpus yields ~10³ claims; even 100 documents is
  ~30k vectors — a 46 MB array and a few milliseconds of brute-force cosine. A
  vector DB earns its keep north of a million.
- **No embedding model deciding measure identity.** "Revenue from operations",
  "revenue from services" and "total income" are near-identical in any embedding
  space and are *genuinely different measures*. Similarity proposes candidates;
  it never decides.
- **Not RAG.** RAG answers a question from retrieved chunks. Contradiction
  detection needs exhaustive pairwise comparison over normalised claims — asking
  a RAG system "is there a contradiction here?" only works if both conflicting
  statements happen to land in the same context window. Retrieval is used here
  as a *blocking strategy*, not an answer generator.

### Architecture

```
PDF → ingest ─→ triage ─→ profile ─→ extract ─→ ground ─→ canonicalise ─→ reconcile → SQLite
      pages,     drop      doc-level   LLM       verify     growing        five
      offsets    empty     defaults    (only     quotes +   measure        verdicts
                 pages                 paid      qualifiers registry       + traces
                                       stage)
```

Only `extract` and the registry's ambiguous-pair adjudication cost money.
Everything else is deterministic and unit-tested.

**Improving the deterministic layers is free.** Every claim keeps the period and
unit strings the document actually used, so a better parser can be re-applied to
claims already stored. `scripts/rebuild.py` re-normalises and re-reconciles the
whole corpus with **no model calls** — when month-range periods started parsing
correctly, it corrected three false contradictions across 1,249 stored claims in
under a second. Only extraction ever costs money, and it never has to be repeated
to benefit from a fix downstream of it.

**Incrementality** falls out of two earlier decisions: document ids are content
hashes (re-adding a file is a no-op), and reconciliation is blocked on
(subject, measure), so a new document is only compared against claims sharing
those keys. Adding the sixth document does not re-examine the first five.

### AI tools used

- **Claude Code (Opus)** as the coding agent throughout — design discussion,
  implementation, and the test suite.
- **`openai/gpt-oss-120b` via Groq** as the extraction and adjudication model,
  deliberately on a free tier so reviewers can run this without an account.

---

## Results on the starter corpus

```
6 documents · 1,700 claims (1,589 active, 111 quarantined)
1,144 relations · 307 cross-document
grounding 93.5%  (435 exact · 754 normalised · 400 elided · 111 rejected)
verdicts: 611 underspecified · 488 complementary · 28 contradicts · 17 corroborates
```

Run `python scripts/demo.py` to print the four required cases from the live
database. All four are found by searching the stored relations for the *shapes*
the brief describes — no document, figure or page is hardcoded.

**Case 1 — corroboration across documents.** Two independent institutions on the
same fact, phrased and spelled differently:

```
[CORROBORATES] core inflation
   3.5 per cent   RBI Annual Report   period = '2024-25'
   3.5 percent    IMF Article IV      period = 'FY2024/25 average'
   period check: '2024-25' [2024-04-01→2025-03-31] EQUALS 'FY2024/25 average' [same]
```

and across units, between Delhivery's results deck and its annual report — with
the reported/adjusted distinction preserved on both sides:

```
[CORROBORATES] reported EBITDA    127 ₹ Cr  vs  1,266.41 ₹ Million
[CORROBORATES] adjusted EBITDA     76 ₹ Cr  vs    757.86 ₹ Million
```

**Case 2 — a genuine contradiction.** Same measure, period, scope and basis;
values 8.7% apart, with nothing in the qualifiers to explain it:

```
[CONTRADICTS] loss before tax
   -8,987.45 ₹ million   prospectus p.94   "Restated loss before tax (8,987.45)"
   -9,839.09 ₹ million   prospectus p.103  "Loss before tax (V= III+IV) (9,839.09)"
```

**Case 3 — apparent contradictions explained by context.** Three distinct
mechanisms, all demonstrable: period containment (Q4 FY24 nested in FY24, with
the arithmetic check confirming 20.76bn ≤ 81.42bn), reporting basis (reported vs
adjusted EBITDA), and disjoint periods.

**Case 4 — failures found and handled.** 111 quarantined claims, 16/16 fabricated
scopes caught on a single slide, and the extraction error below that only
cross-document reconciliation could expose.

## Findings

The debugging record is in [`docs/DECISIONS.md`](docs/DECISIONS.md); the honest
self-audit is in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md). Four highlights:

### The model spent 86% of its output budget thinking

`finish_reason: length` — 2,653 of 3,072 completion tokens were *reasoning*
tokens on one page, truncating the JSON. The dangerous part: **the truncated
output still parsed as valid JSON**, so 8 of 16 claims vanished with no error.

Fixed with `reasoning_effort: low` (extraction is careful copying, not
deduction), a raised ceiling, and treating `finish_reason == "length"` as a hard
failure. Extraction went 6 → 16 claims on the test page while completion tokens
*fell* from 3,072 to 1,781.

Immediately after, the fix appeared to do nothing — the cache key hashed the
model and prompt but **not the decoding parameters**, so it kept serving the
truncated results.

### The model fabricated a qualifier and would not stop

It tagged **all 16** claims on one slide with `scope = "standalone and
consolidated"`. Neither word appears on that page, and the value is not even a
coherent scope. It kept doing so after the prompt explicitly forbade it.

Prompting was the wrong tool. Grounding was extended from quotes to
**qualifiers**: a qualifier the source page does not support is dropped and
recorded. 16/16 caught, while `basis="adjusted"` was correctly *kept* — the page
writes "Adj.", and an alias table knows those are the same.

The principle: **a fabricated qualifier is more dangerous than a missing one.**
A missing scope yields an honest `UNDERSPECIFIED`; an invented one yields a
confident comparison between claims that were never comparable.

### Requiring verbatim quotes discarded 62% of true claims

The model quotes `"Q4 FY23: 0.7%"` from a page reading
`"Q4 FY23: ₹13 Cr / 0.7%"` — it *elides*, it does not invent. Demanding a
contiguous span quarantined 173 of 280 claims, nearly all of them true.

The right test is not "is this span contiguous" but "**do all the quote's
tokens appear on the page, in order**". Grounding went **38% → 85%**. It still
rejects fabricated figures and scrambled tokens, both of which are tested.

### An extraction error that only cross-document reconciliation could catch

The earnings deck yielded `express parcel shipments = 7,224 Mn` for FY24; the
annual report says **740 million**. Flagged CONTRADICTS — correctly, because one
of them is wrong. The claim's own quote shows which:

```
'Express Parcel shipments\n(₹ Cr)\nPTL freight tonnage(2)\n(‘000 Tons)\n8\nYoY: 11%'
```

The model read a figure off a slide of several charts and attached it to the
wrong label — the quote even carries `(₹ Cr)`, a currency unit, against a claim
recording `Mn` shipments.

**Grounding could not have caught this.** The quote really is on that page, so it
verified as `exact`. Confirming that evidence exists says nothing about whether
the model attached it to the right thing. It took a second document disagreeing.
That is the argument for building a knowledge layer at all: it finds errors that
per-document validation structurally cannot.

### Publisher mistaken for subject — the failure that produces silence

The RBI Annual Report and the IMF Article IV report share **zero** relations,
despite both reporting Indian GDP growth and inflation over overlapping
periods. The cause:

```
RBI claims  ->  subject_key = 'reserve bank of india'   (194 of 229)
IMF claims  ->  subject_key = 'india'                   (196 of 199)
```

The extractor recorded the RBI report's subject as its **publisher** rather
than the entity its facts describe — a central bank's annual report is about
the economy, not the bank. Reconciliation blocks on (subject, measure), so that
single mislabel silently suppressed every comparison between the two documents.

This is the failure mode the system is least able to see. A wrong *value*
surfaces as a contradiction; a wrong *subject* produces **no output at all**,
and an empty result is indistinguishable from "these documents have nothing in
common". It was found only by asking why an expected comparison was missing,
which is what `scripts/audit.py` is for.

The fix is two prompt changes (profiling must separate publisher from subject).
It is deliberately **not applied in this commit**: changing the extraction
prompt changes every cache key, and with the daily token allowance spent it
cannot be re-run — applying it would leave a repository whose code cannot
reproduce its own stored results. The diagnosis and the rejected shortcut are in
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md), and
`tests/test_subject_resolution.py` pins the behaviour so the fix is verifiable.

### An arithmetic check that was wrong for profit

"A part cannot exceed the whole" fired on `Q3 FY24 EBITDA = ₹109 Cr` vs
`FY24 EBITDA = ₹76 Cr` — but that inequality only holds for **non-negative**
quantities. Delhivery's FY23 EBITDA was **−452 Cr**, so one good quarter can
legitimately exceed the full year.

Rather than hardcode which measures behave this way, the system **reads it off
the corpus**: any measure ever observed negative is exempt. Contradictions fell
**41 → 3** on the earnings deck, and a new domain gets this right with no new
code.

---

## Limitations and next steps

Full version in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md), which separates
what is principled from what is tuned to this corpus. The most important:

**Corpus coverage is partial.** Groq's free tier allows 200,000 tokens per day;
extraction costs ~4,000 tokens per page and the corpus is 511 pages — roughly
six days of allowance. Five of six documents are represented, most of them from
a capped page budget rather than in full, and the Economic Survey has **zero**
claims because its pages were never extracted. The pipeline reports this as
`pages_skipped_no_quota` rather than implying it read what it did not, but the
corpus numbers below describe what was actually processed, not the whole 511
pages.

**There is no labelled ground truth, so there are no real precision or recall
numbers for extraction or reconciliation.** `scripts/eval_triage.py` reports 9/9,
but those targets are spans I chose myself after reading the documents — a
regression smoke test, not a benchmark. The first thing I would build next is a
small labelled set so that tuning is measured instead of eyeballed.

**The fiscal year is hardcoded to April–March.** Correct for this corpus and for
Indian filings; **wrong** for a US 10-K on an October–September year — and it
would not error, it would silently mis-date the interval. It should be inferred
per document during profiling.

**Some prompt examples come from these corpora.** The measure-adjudication
prompt teaches the model that `revenue from operations` vs `total income` and
`GDP` vs `GVA` are distinct. Real distinctions, but chosen after reading these
documents — the most "tuned to pass" thing in the repository.

**Two fixes deliberately hide things.** Same-page suppression means a document
that genuinely contradicts itself twice on one page will not be flagged; the
negative-capable exemption means a real part-exceeds-whole error in a profit
measure now goes undetected. Both cut false positives sharply; both cost recall.

**Next, in order:** a labelled evaluation set; per-document fiscal-year
inference; table-aware extraction so row labels survive (four expense lines each
became `"% of revenue"`); claim-level rather than document-level `as_of`; and
OCR for scanned PDFs, which currently yield nothing.

---

## Repository layout

| Path | What it does |
|---|---|
| `src/factlayer/models.py` | Claim / Qualifiers / Evidence / Verdict schema |
| `src/factlayer/ingest.py` | PDF → pages with character offsets |
| `src/factlayer/triage.py` | Generic fact-density scoring; skips empty pages |
| `src/factlayer/extract.py` | Document profiling + per-page claim extraction |
| `src/factlayer/ground.py` | Verifies quotes **and qualifiers** against source |
| `src/factlayer/normalize/` | Units, periods (interval algebra), scope lattice |
| `src/factlayer/registry.py` | Measure vocabulary that grows with the corpus |
| `src/factlayer/reconcile.py` | The five verdicts, and the derivation traces |
| `src/factlayer/pipeline.py` | End-to-end, incremental ingestion |
| `src/factlayer/db.py` | SQLite store |
| `src/factlayer/api.py` + `web/` | API and the disagreement inbox |
| `scripts/` | `ingest`, `demo`, `audit`, `rebuild`, `refresh_subjects`, `eval_triage` |
| `docs/` | `DECISIONS.md`, `LIMITATIONS.md` |
