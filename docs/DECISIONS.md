# Decisions, findings and dead ends

A running log kept while building. Findings are recorded when they happen,
including the ones that did not work out. This is the raw material for the
README's *Approach* and *Limitations* sections.

---

## The thesis

**A contradiction is a claim about qualifiers, not about values.**

Two numbers differing is not a conflict. `Revenue = ₹412 Cr` and
`Revenue = ₹98 Cr` only conflict if they cover the same period, the same entity
scope, the same unit and the same accounting basis. So the extractor's primary
job is not grabbing the number — it is capturing the context that makes the
number mean something.

The consequence: comparison becomes **deterministic and explainable**. The LLM
reads; code judges, and shows its working. Most systems ask a model "do these
contradict?" and cannot show why it answered as it did.

### The verdict vocabulary

| Verdict | Condition |
|---|---|
| `CORROBORATES` | qualifiers compatible, values agree |
| `CONTRADICTS` | qualifiers compatible, values disagree |
| `COMPLEMENTARY` | qualifiers disjoint — both can be true |
| `SUPERSEDES` | same claim, later `as_of`; the world changed |
| `UNDERSPECIFIED` | a qualifier needed for the judgement is missing |

The assignment asks for three relations. `RECONCILE` splits into two genuinely
different mechanisms (disjoint context vs. time moving on), and
`UNDERSPECIFIED` is the honest fifth answer.

---

## Design decisions

### Numeric values are stored as closed intervals
A point estimate is a degenerate interval (`low == high`). This makes "does
RBI's 6.5% agree with the Economic Survey's 6.3–6.8%?" plain interval
containment rather than a special case. Periods get the same treatment, so
period comparison is interval algebra.

### Four deep qualifier slots plus an open extension space
`period`, `scope`, `basis` and `as_of` have real logic behind them. Anything
else the documents introduce lands in `extra` and is compared generically
(same / different / missing). The schema grows with the corpus without
pretending to understand every new dimension.

### Currencies are never converted
Exchange rates move, filings rarely state the rate they used, and converting
would manufacture agreement or disagreement that is not in the documents.
Cross-currency pairs are reported as non-comparable.

### An unstated qualifier is not a wildcard
`relate_scope("consolidated", None)` returns UNKNOWN, not a match. An
unlabelled figure could be either side of the boundary. This is the single
most common source of false contradictions in naive systems.

### No vector database
The corpus yields on the order of 10³ claims; even at 100 documents it is ~30k
vectors, which is a 46 MB numpy array and a few milliseconds of brute-force
cosine. A dedicated vector DB earns its keep somewhere north of a million
vectors. Similarity is used for *candidate blocking* only, behind an interface
that could be swapped for an ANN index.

### Embeddings are not used to decide measure identity
"Revenue from operations", "revenue from services" and "total income" are
semantically near-identical and **genuinely different measures** — that
difference is the whole insight. An embedding model would collapse them.
Cheap similarity proposes candidates; the decision is made elsewhere.

### Why this is not RAG
RAG answers a question from retrieved chunks. Contradiction detection needs
exhaustive pairwise comparison over normalised claims, not top-k retrieval —
asking a RAG system "is there a contradiction here?" only works if both
conflicting statements happen to land in the same context window. Retrieval is
used here as a blocking strategy, not an answer generator.

---

## Findings

### Triage over-rewarded numeric density (fixed)
The first page scorer ranked by digit density. It kept every financial table
and dropped the **prose** pages: the Barasia biography scored 3.50 and the
Economic Survey's "growth in FY26 would be between 6.3 and 6.8 per cent"
scored 1.58, against a table's 8.00. Both are required for the demo cases.

Fix: log-saturate the number count (a 200-number table should not outscore a
25-number one eightfold) and add **assertion verbs** — `projected`, `stood at`,
`resigned` — since narrative facts are carried by verbs, not digits.
Recall 6/9 → 7/9.

### Stratified page selection made it worse (reverted)
Reserving budget per region of the document scored **6/9**, below plain global
ranking, because the bands spent their allocation on mediocre pages. Deleted
rather than kept as a clever-looking complication.

The real problem was the *budget*, not the scorer: a 100-page filing genuinely
has more than 40 fact-bearing pages. Triage now only drops empty pages.

| budget | recall |
|---|---|
| 20 | 6/9 = 67% |
| 40 | 7/9 = 78% |
| 60 | 8/9 = 89% |
| none (default) | **9/9 = 100%** |

487 of 511 pages selected, 98% character coverage.

### The model spent 86% of its output budget reasoning (fixed)
`finish_reason: length` — 2,653 of 3,072 completion tokens were *reasoning*
tokens on a single page, truncating the JSON. The dangerous part: **the
truncated output still parsed as valid JSON**, so 8 of 16 claims vanished with
no error raised.

Fix: `reasoning_effort: low` (extraction is careful copying, not deduction), a
16K token ceiling, and treating `finish_reason == "length"` as a hard failure
rather than a successful parse. Extraction went 6 → 16 claims on the test page
while completion tokens *fell* from 3,072 to 1,781.

### The cache key omitted decoding parameters (fixed)
After the truncation fix, nothing changed — the key hashed model, prompt and
temperature but not `max_tokens` or `reasoning_effort`, so it kept serving the
truncated results. The fix appeared to have failed. Any parameter that changes
the output must be in the key.

### The model fabricated a qualifier and would not stop (handled in code)
It tagged **all 16** claims on the Q4 deck's page 6 with
`scope = "standalone and consolidated"`. Neither word appears anywhere on that
page, and the value is not even a coherent scope. It kept doing so after the
prompt explicitly forbade inventing or combining scopes.

Prompting was the wrong tool. Grounding was extended from quotes to
**qualifiers**: a qualifier the source page does not support is dropped and
recorded. 16/16 fabrications caught, while `basis="adjusted"` was correctly
kept — the page writes "Adj.", and an alias table knows those are the same.

The principle behind it: **a fabricated qualifier is more dangerous than a
missing one.** A missing scope yields an honest UNDERSPECIFIED; an invented one
yields a confident comparison between claims that were never comparable.

### Extraction decomposition needed a worked example
The model initially produced `measure = "FY24 Adjusted EBITDA"`, burying the
period and basis inside the measure name — which would make the same measure
unmatchable across years and documents. A worked example in the prompt showing
correct vs. incorrect decomposition fixed it: `measure "EBITDA", basis
"adjusted", period "FY24"`.

---

## Cases located in the starter corpus

| Case | Evidence |
|---|---|
| Corroboration | FY24 revenue: `₹8,142 Cr` (Q4 deck) vs `₹81,415.38 mn` (annual report) — different units, different wording, agree to 0.006% |
| Reconciled by scope | `81,415.38 mn` consolidated vs `74,540.82 mn` standalone, same page |
| Reconciled by measure | revenue from operations `81,415.38 mn` vs total income `85,942.34 mn` |
| Reconciled by basis | FY24 EBITDA `₹127 Cr` vs Adj. EBITDA `₹76 Cr` |
| Reconciled by period | FY24 revenue `₹8,142 Cr` vs Q4 FY24 `₹2,076 Cr` (containment) |
| Supersession | Barasia "is an Executive Director" (prospectus 2022) vs "resigned w.e.f. July 01, 2024" (AR FY24) |
| Cross-institution | India real GDP growth FY26: Survey 6.3–6.8% (Jan 2025), RBI 6.5% (May 2025), IMF 6.6% (Nov 2025) |
