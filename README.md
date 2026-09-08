# Fact Knowledge Layer

Extracts qualified factual claims from PDFs, grounds every claim in its source
text, and reconciles claims against each other across documents.

**Status:** in progress. Full write-up (approach, trade-offs, demo, limitations)
lands before submission.

## The idea in one sentence

A contradiction is a claim about *qualifiers*, not about values — two numbers
differing is only a conflict if they cover the same period, scope, basis and
entity — so this system extracts the context that makes a number mean something
and makes the comparison auditable rather than a matter of opinion.

## Quick start

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
cp .env.example .env      # add your API key
```

## Layout

| Path | What it does |
|------|--------------|
| `src/factlayer/models.py` | Claim / Qualifiers / Evidence / Verdict schema |
| `src/factlayer/ingest.py` | PDF → pages with character offsets |
| `src/factlayer/triage.py` | Generic fact-density scoring; skips empty pages |
| `src/factlayer/llm.py`    | Provider-agnostic client, disk-cached, JSON-repairing |
| `scripts/eval_triage.py`  | Page-selection recall harness |

## Datasets

`starter-datasets/` holds the two provided corpora (Delhivery — 3 documents;
India macroeconomy — 3 documents), 511 pages total.
