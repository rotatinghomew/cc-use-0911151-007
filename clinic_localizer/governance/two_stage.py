"""两阶段验证审批的持久化编排。

阶段划分：
    created -> split_assigned -> candidates_recorded -> selection_frozen
            -> confirmation_submitted -> acceptance_issued

不可变边界：
- selection_frozen 之后确认集上只允许评估已冻结的模型/阈值，没有任何调参入口；
- acceptance_issued 之后证据与结论按签发时点冻结，后续变化只追加 restriction，
  不覆盖原结论，历史可逐条追溯；
- 确认集补报、同意撤回会进入 basis_drift_report，尚未签发的审批在签发时强制复核，
  依据不通过则拒绝签发。
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Mapping, Sequence

from clinic_localizer.cohorts.protocol import (
    ContractError,
    DomainRecord,
    canonical_json,
    finite_number,
    identifier,
    stable_hash,
    utc_now,
)
from clinic_localizer.statistics.cohort_guard import consent_allows
from clinic_localizer.statistics.two_stage_metrics import (
    CohortObservation,
    DEFAULT_MAX_SUBGROUP_GAP,
    DEFAULT_SENSITIVITY_LOW_FLOOR,
    DEFAULT_SPECIFICITY_LOW_FLOOR,
    assert_disjoint,
    evidence_bundle,
    select_candidate,
)

CYCLE_STATES = (
    "created",
    "split_assigned",
    "candidates_recorded",
    "selection_frozen",
    "confirmation_submitted",
    "acceptance_issued",
)
PRE_ISSUE_STATES = frozenset(CYCLE_STATES[:-1])


def prepare_two_stage(connection: sqlite3.Connection) -> None:
    connection.execute("""CREATE TABLE IF NOT EXISTS validation_cycles(
        cycle_id TEXT PRIMARY KEY,
        product_code TEXT NOT NULL,
        intended_use TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        context_json TEXT NOT NULL
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS cycle_splits(
        cycle_id TEXT NOT NULL,
        split_role TEXT NOT NULL,
        cohort_id TEXT NOT NULL,
        record_version INTEGER NOT NULL,
        record_fingerprint TEXT NOT NULL,
        observation_json TEXT NOT NULL,
        assigned_at TEXT NOT NULL,
        PRIMARY KEY(cycle_id, split_role, cohort_id),
        FOREIGN KEY(cycle_id) REFERENCES validation_cycles(cycle_id)
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS candidate_evaluations(
        cycle_id TEXT NOT NULL,
        model_id TEXT NOT NULL,
        operating_threshold REAL NOT NULL,
        candidate_key TEXT NOT NULL,
        evidence_json TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        PRIMARY KEY(cycle_id, candidate_key),
        FOREIGN KEY(cycle_id) REFERENCES validation_cycles(cycle_id)
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS selection_decisions(
        cycle_id TEXT PRIMARY KEY,
        model_id TEXT NOT NULL,
        operating_threshold REAL NOT NULL,
        selection_json TEXT NOT NULL,
        frozen_at TEXT NOT NULL,
        FOREIGN KEY(cycle_id) REFERENCES validation_cycles(cycle_id)
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS confirmation_reviews(
        review_id TEXT PRIMARY KEY,
        cycle_id TEXT NOT NULL,
        model_id TEXT NOT NULL,
        operating_threshold REAL NOT NULL,
        review_json TEXT NOT NULL,
        superseded INTEGER NOT NULL DEFAULT 0,
        submitted_at TEXT NOT NULL,
        UNIQUE(cycle_id, review_id),
        FOREIGN KEY(cycle_id) REFERENCES validation_cycles(cycle_id)
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS safety_acceptances(
        acceptance_id TEXT PRIMARY KEY,
        cycle_id TEXT NOT NULL UNIQUE,
        model_id TEXT NOT NULL,
        operating_threshold REAL NOT NULL,
        outcome TEXT NOT NULL,
        basis_hash TEXT NOT NULL,
        basis_json TEXT NOT NULL,
        evidence_scope_json TEXT NOT NULL,
        supported_uses_json TEXT NOT NULL,
        issued_at TEXT NOT NULL,
        FOREIGN KEY(cycle_id) REFERENCES validation_cycles(cycle_id)
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS acceptance_restrictions(
        restriction_id TEXT PRIMARY KEY,
        acceptance_id TEXT NOT NULL,
        trigger_type TEXT NOT NULL,
        reason TEXT NOT NULL,
        detail_json TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        FOREIGN KEY(acceptance_id) REFERENCES safety_acceptances(acceptance_id)
    )""")


# --------------------------------------------------------------------------- #
# 内部读取工具
# --------------------------------------------------------------------------- #
def _cycle(connection: sqlite3.Connection, cycle_id: str) -> sqlite3.Row:
    prepare_two_stage(connection)
    row = connection.execute("SELECT * FROM validation_cycles WHERE cycle_id=?", (cycle_id,)).fetchone()
    if row is None:
        raise ContractError("unknown validation cycle")
    return row


def _require_status(connection: sqlite3.Connection, cycle_id: str, allowed: Sequence[str]) -> sqlite3.Row:
    row = _cycle(connection, cycle_id)
    if row["status"] not in allowed:
        raise ContractError(
            f"cycle status '{row['status']}' does not permit this action (allowed: {', '.join(allowed)})"
        )
    return row


def _set_status(connection: sqlite3.Connection, cycle_id: str, status: str) -> None:
    connection.execute("UPDATE validation_cycles SET status=? WHERE cycle_id=?", (status, cycle_id))


def _split_observations(
    connection: sqlite3.Connection, cycle_id: str, split_role: str
) -> list[CohortObservation]:
    rows = connection.execute(
        "SELECT observation_json FROM cycle_splits WHERE cycle_id=? AND split_role=? ORDER BY cohort_id",
        (cycle_id, split_role),
    ).fetchall()
    return [CohortObservation.from_dict(json.loads(row["observation_json"])) for row in rows]


def _context(connection: sqlite3.Connection, cycle_id: str) -> dict[str, Any]:
    return dict(json.loads(_cycle(connection, cycle_id)["context_json"]))


def _basis_drift(
    connection: sqlite3.Connection,
    cycle_id: str,
    observations: Sequence[CohortObservation],
) -> list[dict[str, Any]]:
    """重新核对观察值背后的记录版本/指纹，并检查确认同意是否仍覆盖 validation 用途。"""
    drift: list[dict[str, Any]] = []
    for observation in observations:
        current = connection.execute(
            "SELECT record_id,version,state,payload_json,fingerprint,occurred_at,source "
            "FROM records WHERE record_id=?",
            (observation.cohort_id,),
        ).fetchone()
        if current is None:
            drift.append({"type": "record_missing", "cohort_id": observation.cohort_id,
                          "detail": "underlying cohort record no longer exists"})
            continue
        if current["fingerprint"] != observation.record_fingerprint:
            drift.append({"type": "record_changed", "cohort_id": observation.cohort_id,
                          "observed_fingerprint": observation.record_fingerprint,
                          "current_fingerprint": current["fingerprint"],
                          "detail": "cohort record changed after the observation was frozen"})
        payload = json.loads(current["payload_json"])
        current_record = DomainRecord(
            record_id=observation.cohort_id,
            payload=payload,
            state=current["state"],
            version=current["version"],
            occurred_at=current["occurred_at"],
            source=current["source"],
        )
        if not consent_allows(current_record, "validation"):
            drift.append({"type": "consent_withdrawn", "cohort_id": observation.cohort_id,
                          "detail": "validation consent no longer covers this cohort"})
    # 同意撤回是独立的阻断项，保证调用方能按类型检索，不被 record_changed 顺序掩盖。
    drift.sort(key=lambda item: (item["cohort_id"], item["type"]))
    return drift


# --------------------------------------------------------------------------- #
# 阶段 1：建周期、分群、候选记录与选择冻结
# --------------------------------------------------------------------------- #
def create_cycle(
    connection: sqlite3.Connection,
    cycle_id: str,
    product_code: str,
    intended_use: str,
    context: Mapping[str, Any] | None = None,
) -> None:
    prepare_two_stage(connection)
    cycle_id = identifier(cycle_id)
    if not all(isinstance(value, str) and value.strip() for value in (product_code, intended_use)):
        raise ContractError("cycle requires product code and intended use")
    if connection.execute("SELECT 1 FROM validation_cycles WHERE cycle_id=?", (cycle_id,)).fetchone():
        return
    connection.execute(
        "INSERT INTO validation_cycles VALUES(?,?,?,?,?,?)",
        (cycle_id, product_code, intended_use, "created", utc_now(),
         canonical_json(dict(context or {}))),
    )


def assign_split(
    connection: sqlite3.Connection,
    cycle_id: str,
    split_role: str,
    observations: Sequence[CohortObservation],
) -> None:
    row = _require_status(connection, cycle_id, ("created", "split_assigned"))
    if split_role not in ("selection", "confirmation"):
        raise ContractError("split role must be selection or confirmation")
    if not observations:
        raise ContractError("a split needs at least one observation")
    if any(item.split_role != split_role for item in observations):
        raise ContractError("observation split role does not match the assigned split")
    # 同一阶段内部不允许重复受试者（含此前已登记的批次）；确认阶段的机构也不能与已登记机构重复。
    seen_subjects: set[str] = set()
    for existing in _split_observations(connection, cycle_id, split_role):
        seen_subjects.update(existing.subject_ids)
    for item in observations:
        overlap = seen_subjects.intersection(item.subject_ids)
        if overlap:
            raise ContractError(f"duplicate subject within {split_role}: {','.join(sorted(overlap))}")
        seen_subjects.update(item.subject_ids)
    confirmation_existing = _split_observations(connection, cycle_id, "confirmation")
    if split_role == "confirmation" and confirmation_existing:
        raise ContractError("confirmation split is fixed once assigned; report new data through the review flow")
    for item in observations:
        record = connection.execute(
            "SELECT version,fingerprint FROM records WHERE record_id=?", (item.cohort_id,)
        ).fetchone()
        if record is None:
            raise ContractError(f"cohort {item.cohort_id} must be registered before split assignment")
        if record["fingerprint"] != item.record_fingerprint:
            raise ContractError(f"cohort {item.cohort_id} changed before split assignment")
        connection.execute(
            "INSERT INTO cycle_splits VALUES(?,?,?,?,?,?,?)",
            (cycle_id, split_role, item.cohort_id, record["version"], item.record_fingerprint,
             canonical_json(item.as_dict()), utc_now()),
        )
    # 两阶段都到位时立即做受试者/机构不相交检查。
    selection = _split_observations(connection, cycle_id, "selection")
    confirmation = _split_observations(connection, cycle_id, "confirmation")
    if selection and confirmation:
        assert_disjoint(selection, confirmation)
    if row["status"] == "created":
        _set_status(connection, cycle_id, "split_assigned")


def record_candidate(
    connection: sqlite3.Connection,
    cycle_id: str,
    model_id: str,
    operating_threshold: float,
    observations: Sequence[CohortObservation] | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require_status(connection, cycle_id, ("split_assigned", "candidates_recorded"))
    model_id = identifier(model_id)
    threshold = float(operating_threshold)
    observations = observations if observations is not None else _split_observations(connection, cycle_id, "selection")
    if not observations or any(item.split_role != "selection" for item in observations):
        raise ContractError("candidate evaluation must use selection-split observations only")
    known = {item.cohort_id for item in _split_observations(connection, cycle_id, "selection")}
    unknown = [item.cohort_id for item in observations if item.cohort_id not in known]
    if unknown:
        raise ContractError("candidate references cohorts not assigned to the selection split: " + ",".join(unknown))
    merged_context = {**_context(connection, cycle_id), **dict(context or {})}
    evidence = evidence_bundle(observations, merged_context)
    candidate_key = f"{model_id}@{threshold:.6g}"
    connection.execute(
        "INSERT OR IGNORE INTO candidate_evaluations VALUES(?,?,?,?,?,?)",
        (cycle_id, model_id, threshold, candidate_key, canonical_json(evidence), utc_now()),
    )
    _set_status(connection, cycle_id, "candidates_recorded")
    return {"candidate_key": candidate_key, "model_id": model_id,
            "operating_threshold": threshold, "evidence": evidence}


def freeze_selection(
    connection: sqlite3.Connection,
    cycle_id: str,
    expected_model_id: str | None = None,
    expected_operating_threshold: float | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require_status(connection, cycle_id, ("candidates_recorded",))
    rows = connection.execute(
        "SELECT model_id,operating_threshold,evidence_json FROM candidate_evaluations WHERE cycle_id=? ORDER BY candidate_key",
        (cycle_id,),
    ).fetchall()
    if not rows:
        raise ContractError("no candidate evaluation recorded; nothing to select")
    from clinic_localizer.statistics.two_stage_metrics import Candidate

    candidates = [
        Candidate(
            model_id=row["model_id"],
            operating_threshold=row["operating_threshold"],
            observations=(),
            evidence=json.loads(row["evidence_json"]),
        )
        for row in rows
    ]
    merged_context = {**_context(connection, cycle_id), **dict(context or {})}
    decision = select_candidate(candidates, merged_context)
    if expected_model_id is not None and identifier(expected_model_id) != decision["selected_model_id"]:
        raise ContractError("selection winner does not match the expected model")
    if expected_operating_threshold is not None and \
            float(expected_operating_threshold) != decision["selected_operating_threshold"]:
        raise ContractError("selection winner does not match the expected threshold")
    connection.execute(
        "INSERT INTO selection_decisions VALUES(?,?,?,?,?)",
        (cycle_id, decision["selected_model_id"], decision["selected_operating_threshold"],
         canonical_json(decision), decision["frozen_at"]),
    )
    _set_status(connection, cycle_id, "selection_frozen")
    return decision


# --------------------------------------------------------------------------- #
# 阶段 2：独立确认（不允许调参）与安全签发
# --------------------------------------------------------------------------- #
def submit_confirmation(
    connection: sqlite3.Connection,
    review_id: str,
    cycle_id: str,
    observations: Sequence[CohortObservation],
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require_status(connection, cycle_id, ("selection_frozen", "confirmation_submitted"))
    review_id = identifier(review_id)
    selection_row = connection.execute(
        "SELECT model_id,operating_threshold FROM selection_decisions WHERE cycle_id=?", (cycle_id,)
    ).fetchone()
    if selection_row is None:
        raise ContractError("selection must be frozen before independent confirmation")
    if not observations or any(item.split_role != "confirmation" for item in observations):
        raise ContractError("confirmation review must use confirmation-split observations only")
    registered = {item.cohort_id: item
                  for item in _split_observations(connection, cycle_id, "confirmation")}
    for item in observations:
        if item.cohort_id not in registered:
            raise ContractError(
                f"cohort {item.cohort_id} is not part of the frozen confirmation split"
            )
    selection_observations = _split_observations(connection, cycle_id, "selection")
    assert_disjoint(selection_observations, observations)
    merged_context = {**_context(connection, cycle_id), **dict(context or {})}
    evidence = evidence_bundle(observations, merged_context)
    review = {
        "review_id": review_id,
        "cycle_id": cycle_id,
        "model_id": selection_row["model_id"],
        "operating_threshold": selection_row["operating_threshold"],
        "tuning_allowed": False,
        "evidence": evidence,
        "leakage": {
            "selection_subject_count": len({s for item in selection_observations for s in item.subject_ids}),
            "confirmation_subject_count": len({s for item in observations for s in item.subject_ids}),
            "subject_overlap": [],
            "hospital_overlap": [],
        },
        "submitted_at": utc_now(),
    }
    connection.execute(
        "UPDATE confirmation_reviews SET superseded=1 WHERE cycle_id=? AND superseded=0", (cycle_id,)
    )
    connection.execute(
        "INSERT OR IGNORE INTO confirmation_reviews VALUES(?,?,?,?,?,?,?)",
        (review_id, cycle_id, review["model_id"], review["operating_threshold"],
         canonical_json(review), 0, review["submitted_at"]),
    )
    _set_status(connection, cycle_id, "confirmation_submitted")
    return review


def _latest_review(connection: sqlite3.Connection, cycle_id: str) -> sqlite3.Row:
    row = connection.execute(
        "SELECT * FROM confirmation_reviews WHERE cycle_id=? AND superseded=0 ORDER BY submitted_at DESC LIMIT 1",
        (cycle_id,),
    ).fetchone()
    if row is None:
        raise ContractError("no confirmation review available")
    return row


def issue_acceptance(
    connection: sqlite3.Connection,
    acceptance_id: str,
    cycle_id: str,
    context: Mapping[str, Any] | None = None,
    allow_limited: bool = False,
) -> dict[str, Any]:
    prepare_two_stage(connection)
    acceptance_id = identifier(acceptance_id)
    existing = connection.execute(
        "SELECT * FROM safety_acceptances WHERE acceptance_id=? OR cycle_id=?",
        (acceptance_id, cycle_id),
    ).fetchone()
    if existing is not None:
        # 幂等重放：同一签发请求返回原时点结论，不允许借重发改写结论。
        if existing["acceptance_id"] != acceptance_id:
            raise ContractError("cycle already has an issued acceptance with a different id")
        return {
            "acceptance_id": existing["acceptance_id"],
            "cycle_id": existing["cycle_id"],
            "outcome": existing["outcome"],
            "model_id": existing["model_id"],
            "operating_threshold": existing["operating_threshold"],
            "basis_hash": existing["basis_hash"],
            "evidence_scope": json.loads(existing["evidence_scope_json"]),
            "supported_uses": json.loads(existing["supported_uses_json"]),
            "drift": json.loads(existing["basis_json"])["basis_recheck_at_issuance"]["drift"],
        }
    _require_status(connection, cycle_id, ("confirmation_submitted",))
    selection_row = connection.execute(
        "SELECT * FROM selection_decisions WHERE cycle_id=?", (cycle_id,)
    ).fetchone()
    review_row = _latest_review(connection, cycle_id)
    review = json.loads(review_row["review_json"])
    # 确认评估的模型/阈值必须与选择冻结的完全一致，确认数据上不允许重新选参。
    if (review["model_id"], review["operating_threshold"]) != \
            (selection_row["model_id"], selection_row["operating_threshold"]):
        raise ContractError("confirmation model/threshold drift; tuning on confirmation data is forbidden")

    confirmation_observations = _split_observations(connection, cycle_id, "confirmation")
    drift = _basis_drift(connection, cycle_id, confirmation_observations)
    blocking = [item for item in drift if item["type"] in {"record_changed", "record_missing", "consent_withdrawn"}]
    if blocking:
        raise ContractError("basis recheck failed at issuance: " + canonical_json(blocking))

    merged_context = {**_context(connection, cycle_id), **dict(context or {})}
    evidence = review["evidence"]
    pooled = evidence["pooled"]
    subgroups = evidence["subgroups"]
    sens_floor = finite_number(merged_context.get("sensitivity_low_floor", DEFAULT_SENSITIVITY_LOW_FLOOR))
    spec_floor = finite_number(merged_context.get("specificity_low_floor", DEFAULT_SPECIFICITY_LOW_FLOOR))
    max_gap = finite_number(merged_context.get("max_subgroup_gap", DEFAULT_MAX_SUBGROUP_GAP))

    rejection_reasons: list[str] = []
    if not pooled["estimable"]:
        rejection_reasons.append("confirmation outcomes are incomplete; evidence supports screening use only")
    else:
        if pooled["sensitivity_low"] < sens_floor:
            rejection_reasons.append(
                f"sensitivity low bound {pooled['sensitivity_low']:.4f} below floor {sens_floor:.4f}")
        if pooled["specificity_low"] < spec_floor:
            rejection_reasons.append(
                f"specificity low bound {pooled['specificity_low']:.4f} below floor {spec_floor:.4f}")
        if subgroups["by_population_group"]["sensitivity_gap"] > max_gap:
            rejection_reasons.append("population subgroup sensitivity gap exceeds the allowed maximum")
        if subgroups["by_hospital"]["sensitivity_gap"] > max_gap:
            rejection_reasons.append("institution subgroup sensitivity gap exceeds the allowed maximum")

    flags = tuple(evidence["adequacy_flags"])
    if rejection_reasons:
        # 不签发被拒结论，明确告知证据仍支持的用途，防止失败审批被当作接受材料引用。
        raise ContractError(
            "acceptance rejected: " + "; ".join(rejection_reasons)
            + f"; supported uses remain {evidence['supported_uses']['allowed']}"
        )
    outcome = "accepted_full" if not flags else "accepted_limited"
    if outcome == "accepted_limited" and not allow_limited:
        raise ContractError(
            "evidence only supports a limited acceptance; re-issue with allow_limited=True to record it"
        )

    basis = {
        "cycle_id": cycle_id,
        "selection_decision": json.loads(selection_row["selection_json"]),
        "confirmation_review_id": review_row["review_id"],
        "confirmation_evidence": evidence,
        "basis_recheck_at_issuance": {"checked_at": utc_now(), "drift": drift, "passed": not blocking},
        "safety_thresholds": {
            "sensitivity_low_floor": sens_floor,
            "specificity_low_floor": spec_floor,
            "max_subgroup_gap": max_gap,
        },
    }
    basis_hash = stable_hash(basis)
    evidence_scope = {
        "model_id": review["model_id"],
        "operating_threshold": review["operating_threshold"],
        "split": {
            "selection_cohorts": [item.cohort_id for item in
                                  _split_observations(connection, cycle_id, "selection")],
            "confirmation_cohorts": [item.cohort_id for item in confirmation_observations],
            "selection_record_fingerprints": sorted(
                item.record_fingerprint for item in _split_observations(connection, cycle_id, "selection")
            ),
            "confirmation_record_fingerprints": sorted(
                item.record_fingerprint for item in confirmation_observations
            ),
            "institutions": sorted({item.hospital_code for item in confirmation_observations}),
            "population_groups": sorted({item.population_group for item in confirmation_observations}),
        },
        "metrics_window": {
            "sensitivity_low": pooled["sensitivity_low"],
            "specificity_low": pooled["specificity_low"],
            "missed_diagnosis_rate_high": pooled["missed_diagnosis_rate_high"],
            "false_positive_rate_high": pooled["false_positive_rate_high"],
            "population_sensitivity_gap": subgroups["by_population_group"]["sensitivity_gap"],
            "hospital_sensitivity_gap": subgroups["by_hospital"]["sensitivity_gap"],
        },
    }
    supported = evidence["supported_uses"]
    connection.execute(
        "INSERT INTO safety_acceptances VALUES(?,?,?,?,?,?,?,?,?,?)",
        (acceptance_id, cycle_id, review["model_id"], review["operating_threshold"], outcome,
         basis_hash, canonical_json(basis), canonical_json(evidence_scope),
         canonical_json(supported), utc_now()),
    )
    _set_status(connection, cycle_id, "acceptance_issued")
    return {
        "acceptance_id": acceptance_id,
        "cycle_id": cycle_id,
        "outcome": outcome,
        "model_id": review["model_id"],
        "operating_threshold": review["operating_threshold"],
        "basis_hash": basis_hash,
        "evidence_scope": evidence_scope,
        "supported_uses": supported,
        "drift": drift,
    }


def restrict_acceptance(
    connection: sqlite3.Connection,
    restriction_id: str,
    acceptance_id: str,
    trigger_type: str,
    reason: str,
    detail: Mapping[str, Any] | None = None,
) -> None:
    """已签发材料只允许追加限制，原时点结论保持不变。"""
    prepare_two_stage(connection)
    restriction_id = identifier(restriction_id)
    if not reason.strip():
        raise ContractError("restriction reason is required")
    if not connection.execute(
        "SELECT 1 FROM safety_acceptances WHERE acceptance_id=?", (acceptance_id,)
    ).fetchone():
        raise ContractError("restriction requires an issued acceptance")
    trigger_type = identifier(trigger_type)
    detail_payload: Any = dict(detail) if isinstance(detail, Mapping) else list(detail or [])
    connection.execute(
        "INSERT OR IGNORE INTO acceptance_restrictions VALUES(?,?,?,?,?,?)",
        (restriction_id, acceptance_id, trigger_type, reason,
         canonical_json(detail_payload), utc_now()),
    )


def basis_drift_report(connection: sqlite3.Connection, cycle_id: str) -> dict[str, Any]:
    """供尚未完成的审批在签发前核对，也供已签发周期持续监测后续限制来源。"""
    _cycle(connection, cycle_id)
    confirmation = _split_observations(connection, cycle_id, "confirmation")
    drift = _basis_drift(connection, cycle_id, confirmation)
    acceptance_row = connection.execute(
        "SELECT acceptance_id FROM safety_acceptances WHERE cycle_id=?", (cycle_id,)
    ).fetchone()
    restrictions: list[dict[str, Any]] = []
    if acceptance_row is not None:
        restrictions = [
            dict(row) for row in connection.execute(
                "SELECT * FROM acceptance_restrictions WHERE acceptance_id=? ORDER BY recorded_at,restriction_id",
                (acceptance_row["acceptance_id"],),
            ).fetchall()
        ]
    return {
        "cycle_id": cycle_id,
        "status": _cycle(connection, cycle_id)["status"],
        "drift": drift,
        "restrictions": restrictions,
        "blocks_pending_issuance": bool(drift),
    }


def cycle_history(connection: sqlite3.Connection, cycle_id: str) -> dict[str, Any]:
    row = _cycle(connection, cycle_id)
    selection = connection.execute("SELECT * FROM selection_decisions WHERE cycle_id=?", (cycle_id,)).fetchone()
    selection_dict: dict[str, Any] | None = None
    if selection is not None:
        selection_dict = dict(selection)
        selection_dict["selection"] = json.loads(selection_dict.pop("selection_json"))
        selection_dict["ranking"] = selection_dict["selection"]["ranking"]
    reviews = [
        dict(item) for item in connection.execute(
            "SELECT * FROM confirmation_reviews WHERE cycle_id=? ORDER BY submitted_at,review_id", (cycle_id,)
        ).fetchall()
    ]
    for item in reviews:
        item["review"] = json.loads(item.pop("review_json"))
    acceptance_row = connection.execute(
        "SELECT * FROM safety_acceptances WHERE cycle_id=?", (cycle_id,)
    ).fetchone()
    acceptance = None
    if acceptance_row is not None:
        acceptance = dict(acceptance_row)
        for key in ("basis_json", "evidence_scope_json", "supported_uses_json"):
            acceptance[key.removesuffix("_json")] = json.loads(acceptance.pop(key))
        acceptance["restrictions"] = [
            dict(item) for item in connection.execute(
                "SELECT * FROM acceptance_restrictions WHERE acceptance_id=? ORDER BY recorded_at,restriction_id",
                (acceptance_row["acceptance_id"],),
            ).fetchall()
        ]
    splits = {}
    for role in ("selection", "confirmation"):
        splits[role] = [
            {"cohort_id": item.cohort_id, "record_fingerprint": item.record_fingerprint,
             "hospital_code": item.hospital_code, "population_group": item.population_group,
             "subject_count": len(item.subject_ids)}
            for item in _split_observations(connection, cycle_id, role)
        ]
    return {
        "cycle": dict(row),
        "splits": splits,
        "selection": selection_dict,
        "confirmation_reviews": reviews,
        "acceptance": acceptance,
        "drift": basis_drift_report(connection, cycle_id),
    }
