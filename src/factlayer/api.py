"""HTTP API and UI for the fact knowledge layer.

The interface is a *disagreement inbox* rather than a graph. The assignment is
explicit that a visualisation alone is not the solution, and a force-directed
graph of a few thousand claims is decoration: it shows that things are
connected without showing why, and it is unusable for the actual task, which is
judging whether a particular pair of figures really conflicts.

So the primary view is a review queue. Each row is one pair of claims, the
verdict, and the derivation that produced it -- expandable to the source
sentence from each document. It reads like a code review for facts.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from .config import settings
from .db import Store
from .ingest import load_pdf
from .llm import LLMClient
from .models import Claim, Relation
from .pipeline import ingest_pdf
from .registry import MeasureRegistry

app = FastAPI(title="Fact Knowledge Layer", version="0.1.0")

settings.ensure_dirs()
store = Store(settings.db_path)
llm = LLMClient()
_ingest_lock = asyncio.Lock()

UI = Path(__file__).parent / "web" / "index.html"


def _claim_view(claim: Claim | None) -> dict | None:
    """A claim shaped for display, with its evidence and qualifiers flattened."""
    if claim is None:
        return None
    q = claim.qualifiers
    return {
        "id": claim.id,
        "doc_id": claim.doc_id,
        "subject": claim.subject_raw,
        "measure": claim.measure_raw,
        "measure_key": claim.measure_key,
        "value": claim.display_value(),
        "canonical": claim.quantity.canonical_low if claim.quantity else None,
        "unit": claim.quantity.unit_raw if claim.quantity else None,
        "qualifiers": {
            "period": q.period.label if q.period else None,
            "period_start": q.period.start.isoformat() if q.period and q.period.start else None,
            "period_end": q.period.end.isoformat() if q.period and q.period.end else None,
            "scope": q.scope,
            "basis": q.basis,
            "as_of": q.as_of.isoformat() if q.as_of else None,
            "estimate_type": q.estimate_type,
            "dropped": {k: v for k, v in q.extra.items() if k.startswith("dropped_")},
        },
        "evidence": {
            "page": claim.evidence.page_no,
            "printed_page": claim.evidence.printed_page,
            "quote": claim.evidence.quote,
            "grounding": claim.evidence.grounding.value,
        },
        "confidence": claim.extraction_confidence,
        "status": claim.status.value,
    }


def _relation_view(rel: Relation, docs: dict[str, str]) -> dict:
    claims = store.claims_by_id([rel.claim_a_id, rel.claim_b_id])
    a, b = claims.get(rel.claim_a_id), claims.get(rel.claim_b_id)
    va, vb = _claim_view(a), _claim_view(b)
    for view in (va, vb):
        if view:
            view["document"] = docs.get(view["doc_id"], view["doc_id"])
    return {
        "id": rel.id,
        "verdict": rel.verdict.value,
        "confidence": rel.confidence,
        "summary": rel.summary,
        "cross_doc": bool(a and b and a.doc_id != b.doc_id),
        "trace": [s.model_dump() for s in rel.trace],
        "a": va,
        "b": vb,
    }


def _doc_names() -> dict[str, str]:
    return {d["doc_id"]: d["filename"] for d in store.documents()}


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    if not UI.exists():
        return "<h1>UI not built</h1>"
    return UI.read_text()


@app.get("/api/summary")
async def summary() -> dict:
    s = store.summary()
    s["llm_usage"] = llm.usage.as_dict()
    reg = MeasureRegistry.from_dict(store.load_registry() or {})
    s["registry"] = reg.stats()
    return s


@app.get("/api/documents")
async def documents() -> list[dict]:
    return store.documents()


@app.get("/api/relations")
async def relations(
    verdict: str | None = Query(None),
    cross_doc: bool = Query(False),
    limit: int = Query(50, le=500),
    offset: int = Query(0, ge=0),
) -> list[dict]:
    docs = _doc_names()
    return [
        _relation_view(r, docs)
        for r in store.relations(verdict=verdict, cross_doc_only=cross_doc,
                                 limit=limit, offset=offset)
    ]


@app.get("/api/relations/{relation_id}")
async def relation(relation_id: str) -> dict:
    rel = store.relation(relation_id)
    if rel is None:
        raise HTTPException(404, "no such relation")
    return _relation_view(rel, _doc_names())


@app.get("/api/claims/{claim_id}")
async def claim(claim_id: str) -> dict:
    found = store.claims_by_id([claim_id])
    if claim_id not in found:
        raise HTTPException(404, "no such claim")
    view = _claim_view(found[claim_id])
    view["document"] = _doc_names().get(view["doc_id"], view["doc_id"])
    return view


@app.get("/api/quarantined")
async def quarantined(limit: int = Query(50, le=500)) -> list[dict]:
    docs = _doc_names()
    out = []
    for c in store.quarantined(limit=limit):
        view = _claim_view(c)
        view["document"] = docs.get(c.doc_id, c.doc_id)
        view["reason"] = c.quarantine_reason
        out.append(view)
    return out


@app.get("/api/registry")
async def registry() -> dict:
    reg = MeasureRegistry.from_dict(store.load_registry() or {})
    return {
        "stats": reg.stats(),
        "measures": sorted(
            (
                {
                    "key": k,
                    "canonical_name": e.canonical_name,
                    "surface_forms": sorted(e.surface_forms),
                }
                for k, e in reg.entries.items()
            ),
            key=lambda m: -len(m["surface_forms"]),
        ),
    }


def _pick(verdict: str, *, cross_doc: bool = False, needs: str | None = None,
          outcome: tuple[str, ...] = ("pass", "fail", "info")) -> Relation | None:
    """First relation of a verdict, preferring one whose trace shows `needs`."""
    pool = store.relations(verdict=verdict, cross_doc_only=cross_doc, limit=300)
    if needs:
        for r in pool:
            if any(s.check == needs and s.outcome in outcome for s in r.trace):
                return r
        return None
    return pool[0] if pool else None


@app.get("/api/highlights")
async def highlights() -> dict:
    """The four cases the assignment asks for, found by searching the results.

    Nothing here names a document, a figure or a page. Each case is located by
    the *shape* of its derivation -- a cross-document corroboration, a conflict
    with no qualifier to explain it, a period-containment or basis difference --
    so this keeps working on a corpus the system has never seen.
    """
    docs = _doc_names()

    def view(rel: Relation | None) -> dict | None:
        return _relation_view(rel, docs) if rel else None

    corroboration = _pick("corroborates", cross_doc=True) or _pick("corroborates")
    contradiction = _pick("contradicts", cross_doc=True) or _pick("contradicts")
    by_period = _pick("complementary", needs="period-containment", outcome=("pass",))
    by_basis = _pick("complementary", needs="basis", outcome=("info",))
    by_scope = _pick("complementary", needs="scope", outcome=("info",))

    dropped: dict[str, int] = {}
    for c in store.all_claims(active_only=False):
        for k, v in c.qualifiers.extra.items():
            if k.startswith("dropped_"):
                key = f"{k.replace('dropped_', '')} = {v}"
                dropped[key] = dropped.get(key, 0) + 1

    s = store.summary()
    g = s["grounding"]
    total = sum(g.values()) or 1
    return {
        "case1": view(corroboration),
        "case2": view(contradiction),
        "case3": [view(r) for r in (by_period, by_basis, by_scope) if r],
        "case4": {
            "grounding_rate": round(
                (total - g.get("not_found", 0) - g.get("unverifiable", 0)) / total, 4),
            "not_found": g.get("not_found", 0),
            "unverifiable": g.get("unverifiable", 0),
            "grounding": g,
            "quarantined": [
                {**_claim_view(c), "reason": c.quarantine_reason,
                 "document": docs.get(c.doc_id, c.doc_id)}
                for c in store.quarantined(limit=6)
            ],
            "invented_qualifiers": sorted(
                ({"qualifier": k, "count": n} for k, n in dropped.items()),
                key=lambda d: -d["count"],
            )[:8],
        },
    }


@app.post("/api/documents")
async def upload(file: UploadFile = File(...), budget: int | None = None) -> JSONResponse:
    """Accept a PDF, run it through the pipeline, and report what changed."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "only PDF uploads are supported")

    dest = settings.upload_dir / file.filename
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        load_pdf(dest)  # fail fast on an unreadable file
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"could not read PDF: {exc}") from exc

    # One document at a time: ingestion mutates the shared measure registry.
    async with _ingest_lock:
        result = await ingest_pdf(dest, store, llm, budget=budget)
    return JSONResponse(result.as_dict())
