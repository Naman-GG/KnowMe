"""SQLite persistence for documents, claims and relations.

SQLite rather than a graph database or a vector store. The access patterns here
are "all claims for this document", "all claims sharing a canonical measure" and
"all relations touching this claim" -- ordinary indexed lookups over a few
thousand rows. A graph database would add an operational dependency without
answering a question that a table cannot.

Claims and relations are stored as JSON alongside a handful of extracted,
indexed columns. The JSON is the record of truth, so the schema never has to be
migrated when a new qualifier appears; the columns exist purely so the common
queries can use an index. That combination is what lets the fact schema evolve
with the corpus while the storage layer stays fixed.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .models import Claim, Relation

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    path         TEXT,
    sha256       TEXT NOT NULL,
    title        TEXT,
    page_count   INTEGER NOT NULL,
    profile_json TEXT,
    stats_json   TEXT,
    ingested_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claims (
    id          TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    subject_key TEXT,
    measure_key TEXT,
    period_start TEXT,
    period_end   TEXT,
    status      TEXT NOT NULL,
    grounding   TEXT,
    claim_json  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_claims_doc     ON claims(doc_id);
CREATE INDEX IF NOT EXISTS idx_claims_measure ON claims(subject_key, measure_key);
CREATE INDEX IF NOT EXISTS idx_claims_status  ON claims(status);

CREATE TABLE IF NOT EXISTS relations (
    id            TEXT PRIMARY KEY,
    claim_a_id    TEXT NOT NULL,
    claim_b_id    TEXT NOT NULL,
    verdict       TEXT NOT NULL,
    confidence    REAL NOT NULL,
    cross_doc     INTEGER NOT NULL DEFAULT 0,
    relation_json TEXT NOT NULL,
    UNIQUE(claim_a_id, claim_b_id)
);
CREATE INDEX IF NOT EXISTS idx_rel_verdict ON relations(verdict);
CREATE INDEX IF NOT EXISTS idx_rel_a       ON relations(claim_a_id);
CREATE INDEX IF NOT EXISTS idx_rel_b       ON relations(claim_b_id);

CREATE TABLE IF NOT EXISTS registry (
    id       INTEGER PRIMARY KEY CHECK (id = 1),
    data_json TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # -- documents --------------------------------------------------------

    def has_document(self, doc_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return row is not None

    def upsert_document(
        self, doc_id: str, filename: str, path: str, sha256: str, title: str | None,
        page_count: int, profile: dict, stats: dict,
    ) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT INTO documents
                   (doc_id, filename, path, sha256, title, page_count,
                    profile_json, stats_json, ingested_at)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(doc_id) DO UPDATE SET
                     profile_json=excluded.profile_json,
                     stats_json=excluded.stats_json,
                     ingested_at=excluded.ingested_at""",
                (doc_id, filename, path, sha256, title, page_count,
                 json.dumps(profile), json.dumps(stats),
                 datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )

    def documents(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM documents ORDER BY ingested_at DESC"
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["profile"] = json.loads(d.pop("profile_json") or "{}")
            d["stats"] = json.loads(d.pop("stats_json") or "{}")
            out.append(d)
        return out

    def delete_document(self, doc_id: str) -> None:
        with self.tx() as c:
            ids = [r["id"] for r in c.execute(
                "SELECT id FROM claims WHERE doc_id = ?", (doc_id,))]
            if ids:
                marks = ",".join("?" * len(ids))
                c.execute(
                    f"DELETE FROM relations WHERE claim_a_id IN ({marks}) "
                    f"OR claim_b_id IN ({marks})", ids + ids)
            c.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    # -- claims -----------------------------------------------------------

    def save_claims(self, claims: list[Claim], subject_keys: dict[str, str]) -> None:
        rows = []
        for c in claims:
            p = c.qualifiers.period
            rows.append((
                c.id, c.doc_id, subject_keys.get(c.id), c.measure_key,
                p.start.isoformat() if p and p.start else None,
                p.end.isoformat() if p and p.end else None,
                c.status.value, c.evidence.grounding.value,
                c.model_dump_json(),
            ))
        with self.tx() as conn:
            conn.executemany(
                """INSERT INTO claims
                   (id, doc_id, subject_key, measure_key, period_start, period_end,
                    status, grounding, claim_json)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     claim_json=excluded.claim_json, status=excluded.status,
                     grounding=excluded.grounding, measure_key=excluded.measure_key""",
                rows,
            )

    def _claims_from_rows(self, rows) -> list[Claim]:
        return [Claim.model_validate_json(r["claim_json"]) for r in rows]

    def all_claims(self, active_only: bool = True) -> list[Claim]:
        sql = "SELECT claim_json FROM claims"
        if active_only:
            sql += " WHERE status = 'active'"
        return self._claims_from_rows(self._conn.execute(sql).fetchall())

    def claims_for_document(self, doc_id: str) -> list[Claim]:
        return self._claims_from_rows(self._conn.execute(
            "SELECT claim_json FROM claims WHERE doc_id = ?", (doc_id,)).fetchall())

    def claims_by_id(self, ids: list[str]) -> dict[str, Claim]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT claim_json FROM claims WHERE id IN ({marks})", ids).fetchall()
        return {c.id: c for c in self._claims_from_rows(rows)}

    def quarantined(self, limit: int = 100) -> list[Claim]:
        return self._claims_from_rows(self._conn.execute(
            "SELECT claim_json FROM claims WHERE status='quarantined' LIMIT ?",
            (limit,)).fetchall())

    # -- relations --------------------------------------------------------

    def save_relations(self, relations: list[Relation], cross_doc: dict[str, bool]) -> None:
        rows = [
            (r.id, r.claim_a_id, r.claim_b_id, r.verdict.value, r.confidence,
             int(cross_doc.get(r.id, False)), r.model_dump_json())
            for r in relations
        ]
        with self.tx() as conn:
            conn.executemany(
                """INSERT INTO relations
                   (id, claim_a_id, claim_b_id, verdict, confidence, cross_doc, relation_json)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(claim_a_id, claim_b_id) DO UPDATE SET
                     verdict=excluded.verdict, confidence=excluded.confidence,
                     relation_json=excluded.relation_json""",
                rows,
            )

    def relations(
        self, verdict: str | None = None, cross_doc_only: bool = False,
        limit: int = 200, offset: int = 0,
    ) -> list[Relation]:
        sql = "SELECT relation_json FROM relations WHERE 1=1"
        args: list = []
        if verdict:
            sql += " AND verdict = ?"
            args.append(verdict)
        if cross_doc_only:
            sql += " AND cross_doc = 1"
        sql += " ORDER BY confidence DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
        rows = self._conn.execute(sql, args).fetchall()
        return [Relation.model_validate_json(r["relation_json"]) for r in rows]

    def relation(self, relation_id: str) -> Relation | None:
        row = self._conn.execute(
            "SELECT relation_json FROM relations WHERE id = ?", (relation_id,)).fetchone()
        return Relation.model_validate_json(row["relation_json"]) if row else None

    def existing_pairs(self) -> set[tuple[str, str]]:
        return {
            (r["claim_a_id"], r["claim_b_id"])
            for r in self._conn.execute("SELECT claim_a_id, claim_b_id FROM relations")
        }

    # -- registry ---------------------------------------------------------

    def save_registry(self, data: dict) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO registry (id, data_json) VALUES (1, ?) "
                "ON CONFLICT(id) DO UPDATE SET data_json=excluded.data_json",
                (json.dumps(data),),
            )

    def load_registry(self) -> dict | None:
        row = self._conn.execute("SELECT data_json FROM registry WHERE id = 1").fetchone()
        return json.loads(row["data_json"]) if row else None

    # -- summary ----------------------------------------------------------

    def summary(self) -> dict:
        c = self._conn
        verdicts = {
            r["verdict"]: r["n"]
            for r in c.execute("SELECT verdict, COUNT(*) n FROM relations GROUP BY verdict")
        }
        grounding = {
            r["grounding"]: r["n"]
            for r in c.execute("SELECT grounding, COUNT(*) n FROM claims GROUP BY grounding")
        }
        return {
            "documents": c.execute("SELECT COUNT(*) n FROM documents").fetchone()["n"],
            "claims": c.execute("SELECT COUNT(*) n FROM claims").fetchone()["n"],
            "claims_active": c.execute(
                "SELECT COUNT(*) n FROM claims WHERE status='active'").fetchone()["n"],
            "claims_quarantined": c.execute(
                "SELECT COUNT(*) n FROM claims WHERE status='quarantined'").fetchone()["n"],
            "relations": c.execute("SELECT COUNT(*) n FROM relations").fetchone()["n"],
            "relations_cross_doc": c.execute(
                "SELECT COUNT(*) n FROM relations WHERE cross_doc=1").fetchone()["n"],
            "verdicts": verdicts,
            "grounding": grounding,
        }

    def close(self) -> None:
        self._conn.close()
