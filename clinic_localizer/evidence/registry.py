from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from clinic_localizer.cohorts.protocol import Decision, DomainRecord, ContractError, canonical_json, stable_hash, utc_now
from clinic_localizer.governance.use_scope import prepare_scopes, register_scope, link_cohort, issue_scope, restrict_scope, scope_history
from clinic_localizer.governance import two_stage
from clinic_localizer.statistics.two_stage_metrics import CohortObservation


SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS records(
    record_id TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    state TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    source TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS record_versions(
    record_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    state TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    stored_at TEXT NOT NULL,
    PRIMARY KEY(record_id, version)
);
CREATE TABLE IF NOT EXISTS decisions(
    decision_id TEXT PRIMARY KEY,
    record_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    score REAL NOT NULL,
    decision_json TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    FOREIGN KEY(record_id) REFERENCES records(record_id)
);
CREATE TABLE IF NOT EXISTS events(
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    aggregate_id TEXT NOT NULL,
    aggregate_version INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    previous_hash TEXT NOT NULL,
    event_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    UNIQUE(aggregate_id, aggregate_version, event_type)
);
CREATE TABLE IF NOT EXISTS runs(
    run_id TEXT PRIMARY KEY,
    fixture_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    record_count INTEGER NOT NULL,
    result_json TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_records_state ON records(state, updated_at);
CREATE INDEX IF NOT EXISTS idx_events_aggregate ON events(aggregate_id, event_id);
CREATE INDEX IF NOT EXISTS idx_decisions_record ON decisions(record_id, decided_at);
"""


class Repository:
    def __init__(self, database: str | Path):
        self.database = str(database)
        self._lock = threading.RLock()
        Path(self.database).parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            prepare_scopes(connection)
            two_stage.prepare_two_stage(connection)

    def register_scope(self, scope_id: str, group: str, purpose: str) -> None:
        with self.transaction() as connection:
            register_scope(connection, scope_id, group, purpose)

    def link_cohort(self, scope_id: str, cohort_id: str) -> None:
        with self.transaction() as connection:
            link_cohort(connection, scope_id, cohort_id)

    def issue_scope(self, issue_id: str, scope_id: str) -> dict[str, Any]:
        with self.transaction() as connection:
            return issue_scope(connection, issue_id, scope_id)

    def restrict_scope(self, restriction_id: str, issue_id: str, reason: str) -> None:
        with self.transaction() as connection:
            restrict_scope(connection, restriction_id, issue_id, reason)

    def scope_history(self, scope_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            return scope_history(connection, scope_id)

    # ------------------------------------------------------------------ #
    # 两阶段验证：候选选择与独立确认
    # ------------------------------------------------------------------ #
    def create_cycle(self, cycle_id: str, product_code: str, intended_use: str,
                     context: Mapping[str, Any] | None = None) -> None:
        with self.transaction() as connection:
            two_stage.create_cycle(connection, cycle_id, product_code, intended_use, context)

    def assign_split(self, cycle_id: str, split_role: str,
                     observations: Sequence[CohortObservation]) -> None:
        with self.transaction() as connection:
            two_stage.assign_split(connection, cycle_id, split_role, observations)

    def record_candidate(self, cycle_id: str, model_id: str, operating_threshold: float,
                         observations: Sequence[CohortObservation] | None = None,
                         context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        with self.transaction() as connection:
            result = two_stage.record_candidate(
                connection, cycle_id, model_id, operating_threshold, observations, context)
            self.append_event(connection, "candidate.recorded", cycle_id,
                              self.event_count(cycle_id, connection) + 1,
                              {"candidate_key": result["candidate_key"]})
            return result

    def freeze_selection(self, cycle_id: str, expected_model_id: str | None = None,
                         expected_operating_threshold: float | None = None,
                         context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        with self.transaction() as connection:
            decision = two_stage.freeze_selection(
                connection, cycle_id, expected_model_id, expected_operating_threshold, context)
            self.append_event(connection, "selection.frozen", cycle_id,
                              self.event_count(cycle_id, connection) + 1,
                              {"model_id": decision["selected_model_id"],
                               "operating_threshold": decision["selected_operating_threshold"]})
            return decision

    def submit_confirmation(self, review_id: str, cycle_id: str,
                            observations: Sequence[CohortObservation],
                            context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        with self.transaction() as connection:
            review = two_stage.submit_confirmation(
                connection, review_id, cycle_id, observations, context)
            self.append_event(connection, "confirmation.submitted", cycle_id,
                              self.event_count(cycle_id, connection) + 1,
                              {"review_id": review_id, "tuning_allowed": False})
            return review

    def issue_acceptance(self, acceptance_id: str, cycle_id: str,
                         context: Mapping[str, Any] | None = None,
                         allow_limited: bool = False) -> dict[str, Any]:
        with self.transaction() as connection:
            result = two_stage.issue_acceptance(
                connection, acceptance_id, cycle_id, context, allow_limited)
            self.append_event(connection, "acceptance.issued", cycle_id,
                              self.event_count(cycle_id, connection) + 1,
                              {"acceptance_id": acceptance_id, "outcome": result["outcome"],
                               "basis_hash": result["basis_hash"]})
            return result

    def restrict_acceptance(self, restriction_id: str, acceptance_id: str, trigger_type: str,
                            reason: str, detail: Mapping[str, Any] | Sequence[Any] | None = None) -> None:
        with self.transaction() as connection:
            two_stage.restrict_acceptance(
                connection, restriction_id, acceptance_id, trigger_type, reason, detail)
            self.append_event(connection, "acceptance.restricted", acceptance_id,
                              self.event_count(acceptance_id, connection) + 1,
                              {"restriction_id": restriction_id, "trigger_type": trigger_type})

    def basis_drift_report(self, cycle_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            return two_stage.basis_drift_report(connection, cycle_id)

    def cycle_history(self, cycle_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            return two_stage.cycle_history(connection, cycle_id)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            connection = self.connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    def _last_hash(self, connection: sqlite3.Connection) -> str:
        row = connection.execute("SELECT event_hash FROM events ORDER BY event_id DESC LIMIT 1").fetchone()
        return row[0] if row else "0" * 64

    def append_event(self, connection: sqlite3.Connection, event_type: str,
                     aggregate_id: str, aggregate_version: int, payload: Mapping[str, Any]) -> str:
        previous = self._last_hash(connection)
        created_at = utc_now()
        event_hash = stable_hash({"event_type": event_type, "aggregate_id": aggregate_id,
                                  "aggregate_version": aggregate_version, "payload": payload,
                                  "previous_hash": previous, "created_at": created_at})
        try:
            connection.execute("INSERT INTO events(event_type,aggregate_id,aggregate_version,payload_json,previous_hash,event_hash,created_at) VALUES(?,?,?,?,?,?,?)",
                               (event_type, aggregate_id, aggregate_version, canonical_json(payload),
                                previous, event_hash, created_at))
        except sqlite3.IntegrityError:
            row = connection.execute("SELECT event_hash,payload_json FROM events WHERE aggregate_id=? AND aggregate_version=? AND event_type=?",
                                     (aggregate_id, aggregate_version, event_type)).fetchone()
            if row and row[1] == canonical_json(payload):
                return row[0]
            raise ContractError("conflicting event replay") from None
        return event_hash

    def save_record(self, record: DomainRecord, expected_version: int | None = None) -> str:
        with self.transaction() as connection:
            current = connection.execute("SELECT version,fingerprint FROM records WHERE record_id=?", (record.record_id,)).fetchone()
            if current and expected_version is not None and current[0] != expected_version:
                raise ContractError("optimistic version conflict")
            if current and current[1] == record.fingerprint:
                return record.fingerprint
            if current and record.version <= current[0]:
                raise ContractError("record version must advance")
            payload = canonical_json(record.payload)
            connection.execute("INSERT INTO records(record_id,version,state,payload_json,fingerprint,occurred_at,source,updated_at) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(record_id) DO UPDATE SET version=excluded.version,state=excluded.state,payload_json=excluded.payload_json,fingerprint=excluded.fingerprint,occurred_at=excluded.occurred_at,source=excluded.source,updated_at=excluded.updated_at",
                               (record.record_id, record.version, record.state, payload, record.fingerprint,
                                record.occurred_at, record.source, utc_now()))
            connection.execute("INSERT INTO record_versions(record_id,version,state,payload_json,fingerprint,stored_at) VALUES(?,?,?,?,?,?)",
                               (record.record_id, record.version, record.state, payload, record.fingerprint, utc_now()))
            self.append_event(connection, "record.saved", record.record_id, record.version,
                              {"state": record.state, "fingerprint": record.fingerprint})
            return record.fingerprint

    def get_record(self, record_id: str) -> DomainRecord | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM records WHERE record_id=?", (record_id,)).fetchone()
        if not row:
            return None
        return DomainRecord(record_id=row["record_id"], payload=json.loads(row["payload_json"]),
                            state=row["state"], version=row["version"], occurred_at=row["occurred_at"],
                            source=row["source"])

    def list_records(self, state: str | None = None) -> list[DomainRecord]:
        query = "SELECT * FROM records"
        params: tuple[Any, ...] = ()
        if state:
            query += " WHERE state=?"
            params = (state,)
        query += " ORDER BY record_id"
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [DomainRecord(record_id=row["record_id"], payload=json.loads(row["payload_json"]),
                             state=row["state"], version=row["version"], occurred_at=row["occurred_at"],
                             source=row["source"]) for row in rows]

    def save_decision(self, decision: Decision) -> str:
        decision_id = decision.fingerprint
        with self.transaction() as connection:
            if not connection.execute("SELECT 1 FROM records WHERE record_id=?", (decision.record_id,)).fetchone():
                raise ContractError("decision record does not exist")
            body = canonical_json(decision.as_dict())
            connection.execute("INSERT OR IGNORE INTO decisions(decision_id,record_id,outcome,score,decision_json,decided_at) VALUES(?,?,?,?,?,?)",
                               (decision_id, decision.record_id, decision.outcome, decision.score, body, decision.decided_at))
            self.append_event(connection, "decision.saved", decision.record_id,
                              self.event_count(decision.record_id, connection) + 1,
                              {"decision_id": decision_id, "outcome": decision.outcome, "score": decision.score})
        return decision_id

    def event_count(self, aggregate_id: str, connection: sqlite3.Connection | None = None) -> int:
        owns = connection is None
        connection = connection or self.connect()
        try:
            row = connection.execute("SELECT COUNT(*) FROM events WHERE aggregate_id=?", (aggregate_id,)).fetchone()
            return int(row[0])
        finally:
            if owns:
                connection.close()

    def begin_run(self, fixture_hash: str, record_count: int) -> str:
        run_id = stable_hash({"fixture_hash": fixture_hash, "record_count": record_count})[:24]
        with self.transaction() as connection:
            row = connection.execute("SELECT fixture_hash,record_count FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row and (row[0] != fixture_hash or row[1] != record_count):
                raise ContractError("run identifier collision")
            connection.execute("INSERT OR IGNORE INTO runs(run_id,fixture_hash,status,record_count,result_json,started_at) VALUES(?,?,?,?,?,?)",
                               (run_id, fixture_hash, "running", record_count, "{}", utc_now()))
        return run_id

    def completed_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT status,result_json FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not row or row["status"] != "finished":
            return None
        result = json.loads(row["result_json"])
        if not isinstance(result, dict):
            raise ContractError("stored run result is invalid")
        return result

    def finish_run(self, run_id: str, result: Mapping[str, Any]) -> None:
        with self.transaction() as connection:
            changed = connection.execute("UPDATE runs SET status='finished',result_json=?,finished_at=? WHERE run_id=?",
                                         (canonical_json(result), utc_now(), run_id)).rowcount
            if changed != 1:
                raise ContractError("unknown run")
            self.append_event(connection, "run.finished", run_id, 1, dict(result))

    def verify_chain(self) -> list[str]:
        errors: list[str] = []
        previous = "0" * 64
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM events ORDER BY event_id").fetchall()
        for row in rows:
            payload = json.loads(row["payload_json"])
            expected = stable_hash({"event_type": row["event_type"], "aggregate_id": row["aggregate_id"],
                                    "aggregate_version": row["aggregate_version"], "payload": payload,
                                    "previous_hash": row["previous_hash"], "created_at": row["created_at"]})
            if row["previous_hash"] != previous:
                errors.append(f"event {row['event_id']} previous hash mismatch")
            if row["event_hash"] != expected:
                errors.append(f"event {row['event_id']} content hash mismatch")
            previous = row["event_hash"]
        return errors

    def snapshot(self) -> dict[str, Any]:
        with self.connect() as connection:
            tables = ("records", "record_versions", "decisions", "events", "runs",
                      "validation_cycles", "cycle_splits", "candidate_evaluations",
                      "selection_decisions", "confirmation_reviews",
                      "safety_acceptances", "acceptance_restrictions")
            counts = {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                      for table in tables}
            states = {row[0]: row[1] for row in connection.execute("SELECT state,COUNT(*) FROM records GROUP BY state")}
        return {"counts": counts, "states": states, "chain_errors": self.verify_chain()}

    def decision_rows(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT decision_json FROM decisions ORDER BY record_id,decided_at").fetchall()
        return [json.loads(row[0]) for row in rows]

    def event_rows(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM events ORDER BY event_id").fetchall()
        return [dict(row) | {"payload": json.loads(row["payload_json"])} for row in rows]
