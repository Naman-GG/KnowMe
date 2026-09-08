"""End-to-end ingestion: a PDF goes in, grounded claims and verdicts come out.

The stages are deliberately separable -- ingest, triage, profile, extract,
ground, canonicalise, reconcile -- because each one is worth inspecting on its
own, and because only the extract stage costs money.

Incremental ingestion falls out of two decisions made earlier. Document ids are
content hashes, so re-adding a file is a no-op rather than a duplicate. And
reconciliation is blocked on (subject, measure), so a new document only needs
comparing against the claims that share those keys -- not against everything
already stored. Adding the sixth document does not re-examine the first five.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .db import Store
from .extract import DocProfile, extract_document
from .ground import GroundingReport, ground_claims
from .ingest import Document, load_pdf
from .llm import LLMClient
from .models import Claim, ClaimStatus, Relation
from .reconcile import negative_capable, reconcile_pair
from .registry import BatchResolver, MeasureRegistry, normalise_subject
from .triage import coverage, triage

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    doc_id: str
    filename: str
    page_count: int
    pages_selected: int
    char_coverage: float
    claims_extracted: int
    claims_active: int
    claims_quarantined: int
    relations_new: int
    verdicts: dict[str, int] = field(default_factory=dict)
    grounding: dict = field(default_factory=dict)
    measures_added: int = 0
    seconds: float = 0.0
    skipped: bool = False
    llm_usage: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "pages": self.page_count,
            "pages_selected": self.pages_selected,
            "char_coverage": self.char_coverage,
            "claims_extracted": self.claims_extracted,
            "claims_active": self.claims_active,
            "claims_quarantined": self.claims_quarantined,
            "relations_new": self.relations_new,
            "verdicts": self.verdicts,
            "grounding": self.grounding,
            "measures_added": self.measures_added,
            "seconds": round(self.seconds, 1),
            "skipped": self.skipped,
            "llm_usage": self.llm_usage,
        }


async def canonicalise_claims(
    claims: list[Claim], registry: MeasureRegistry, llm: LLMClient
) -> dict[str, str]:
    """Assign every claim a canonical measure key; return subject keys.

    Resolution runs once per *distinct* measure name, not once per claim. A
    document yields many claims but far fewer distinct measures -- the same
    "revenue from services" appears on a dozen pages -- and resolving a new name
    can cost an LLM adjudication. An earlier version looped over every claim and
    spent minutes on serial round-trips re-deciding questions it had already
    answered.

    The loop stays sequential on purpose: each resolution can extend the
    registry, and later names must be matched against the entries earlier ones
    created.
    """
    resolved = await BatchResolver(registry).resolve_many(
        [c.measure_raw for c in claims], llm
    )
    subject_keys: dict[str, str] = {}
    for claim in claims:
        claim.measure_key = resolved.get(claim.measure_raw) or claim.measure_raw.lower()
        subject_keys[claim.id] = normalise_subject(claim.subject_raw)
        claim.subject_key = subject_keys[claim.id]
    return subject_keys


def reconcile_incremental(
    new_claims: list[Claim], existing: list[Claim], seen: set[tuple[str, str]]
) -> list[Relation]:
    """Compare new claims against each other and against what is already stored.

    Blocking on (subject, measure) is what makes this incremental rather than a
    full rebuild: only the buckets a new claim lands in are ever touched.
    """
    negatives = negative_capable(new_claims + existing)
    buckets: dict[tuple[str, str], list[Claim]] = {}
    for c in existing:
        if c.status is ClaimStatus.ACTIVE:
            buckets.setdefault((c.subject_key or "", c.measure_key or ""), []).append(c)

    relations: list[Relation] = []
    for i, claim in enumerate(new_claims):
        if claim.status is not ClaimStatus.ACTIVE:
            continue
        key = (claim.subject_key or "", claim.measure_key or "")
        # ...against claims already in the store
        for other in buckets.get(key, ()):
            if (claim.id, other.id) in seen or (other.id, claim.id) in seen:
                continue
            if (rel := reconcile_pair(claim, other, negatives)) is not None:
                relations.append(rel)
                seen.add((claim.id, other.id))
        # ...and against the rest of this document
        for other in new_claims[i + 1 :]:
            if other.status is not ClaimStatus.ACTIVE:
                continue
            if (other.subject_key or "", other.measure_key or "") != key:
                continue
            if (rel := reconcile_pair(claim, other, negatives)) is not None:
                relations.append(rel)
                seen.add((claim.id, other.id))
        buckets.setdefault(key, []).append(claim)
    return relations


async def ingest_pdf(
    path: str | Path,
    store: Store,
    llm: LLMClient,
    *,
    budget: int | None = None,
    force: bool = False,
    progress=None,
) -> IngestResult:
    """Run one PDF all the way through and persist the results."""
    started = time.monotonic()
    doc: Document = load_pdf(path)

    if store.has_document(doc.doc_id) and not force:
        return IngestResult(
            doc_id=doc.doc_id, filename=doc.filename, page_count=doc.page_count,
            pages_selected=0, char_coverage=0.0, claims_extracted=0, claims_active=0,
            claims_quarantined=0, relations_new=0, skipped=True,
            seconds=time.monotonic() - started,
        )

    selected = triage(doc, budget=budget)
    cov = coverage(doc, selected)
    if progress:
        progress(f"{doc.filename}: {cov['pages_selected']}/{cov['pages_total']} pages selected")

    profile, claims = await extract_document(doc, llm, budget=budget)

    pages = {p.page_no: p for p in doc.pages}
    report: GroundingReport = ground_claims(claims, pages)

    registry = MeasureRegistry.from_dict(store.load_registry() or {})
    before = len(registry.entries)
    subject_keys = await canonicalise_claims(claims, registry, llm)
    store.save_registry(registry.to_dict())

    store.upsert_document(
        doc_id=doc.doc_id, filename=doc.filename, path=str(path), sha256=doc.sha256,
        title=doc.title, page_count=doc.page_count,
        profile=profile.as_dict(), stats={**cov, **report.as_dict()},
    )
    store.save_claims(claims, subject_keys)

    existing = [c for c in store.all_claims() if c.doc_id != doc.doc_id]
    relations = reconcile_incremental(claims, existing, store.existing_pairs())
    by_id = {c.id: c for c in claims + existing}
    cross = {
        r.id: by_id[r.claim_a_id].doc_id != by_id[r.claim_b_id].doc_id
        for r in relations
        if r.claim_a_id in by_id and r.claim_b_id in by_id
    }
    store.save_relations(relations, cross)

    verdicts: dict[str, int] = {}
    for r in relations:
        verdicts[r.verdict.value] = verdicts.get(r.verdict.value, 0) + 1

    return IngestResult(
        doc_id=doc.doc_id,
        filename=doc.filename,
        page_count=doc.page_count,
        pages_selected=int(cov["pages_selected"]),
        char_coverage=float(cov["char_coverage"]),
        claims_extracted=len(claims),
        claims_active=sum(c.status is ClaimStatus.ACTIVE for c in claims),
        claims_quarantined=sum(c.status is ClaimStatus.QUARANTINED for c in claims),
        relations_new=len(relations),
        verdicts=verdicts,
        grounding=report.as_dict(),
        measures_added=len(registry.entries) - before,
        seconds=time.monotonic() - started,
        llm_usage=llm.usage.as_dict(),
    )
