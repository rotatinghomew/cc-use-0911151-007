"""两阶段验证的纯领域逻辑：候选选择与独立确认。

设计原则：
- 选择（selection）阶段允许在同一批选择队列上反复比较模型与阈值；
- 确认（confirmation）阶段只评估已冻结的模型/阈值，模块不提供任何调参入口；
- 受试者标识与机构标识在两个阶段之间必须完全不相交，从数据结构上杜绝样本混用；
- 漏诊（1-敏感度）、误报（1-特异度）均给出 Wilson 区间，并按人群/机构分层比较；
- 小样本或结果缺失时不伪造结论，只输出证据仍支持的用途清单。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from clinic_localizer.cohorts.protocol import (
    ContractError,
    DomainRecord,
    bounded,
    finite_number,
    identifier,
    normalized_text,
    stable_hash,
    utc_now,
)
from clinic_localizer.statistics.cohort_guard import wilson_interval

SPLIT_SELECTION = "selection"
SPLIT_CONFIRMATION = "confirmation"
SPLIT_ROLES = (SPLIT_SELECTION, SPLIT_CONFIRMATION)

# 充分性默认门槛，可由调用方在编排层通过 context 覆盖。
MIN_TOTAL_SUBJECTS = 50.0
MIN_POSITIVE_CASES = 10.0
MIN_NEGATIVE_CASES = 10.0
MAX_MISSINGNESS = 0.40
# 独立确认签发的最低安全底线（Wilson 下界）。
DEFAULT_SENSITIVITY_LOW_FLOOR = 0.70
DEFAULT_SPECIFICITY_LOW_FLOOR = 0.70
DEFAULT_MAX_SUBGROUP_GAP = 0.15

FULL_USES = (
    "independent_confirmation",
    "deployment_with_post_market_monitoring",
)
LIMITED_USES = (
    "limited_internal_qa",
    "prospective_monitoring",
)
SCREENING_USES = (
    "candidate_screening",
    "protocol_refinement",
)
WITHHELD_DEPLOYMENT = (
    "standalone_diagnosis",
    "new_institution_deployment",
)


def proportion(value: Any) -> float:
    """把 0..1 的比例或 0..100 的百分数统一归一化到 [0,1]。"""
    number = finite_number(value)
    if number > 1.0:
        number = number / 100.0
    return bounded(number)


def extract_subject_ids(record: DomainRecord, explicit: Sequence[str] | None = None) -> tuple[str, ...]:
    """提取受试者标识；缺失时直接拒绝，避免在无标识情况下假装能防混用。"""
    if explicit is not None:
        raw: Any = list(explicit)
    else:
        raw = record.payload.get("subject_ids", record.payload.get("subject_id"))
    if raw is None:
        raise ContractError("subject identifiers are required to keep selection and confirmation disjoint")
    if isinstance(raw, (str, int, float)):
        raw = [raw]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ContractError("subject identifiers must be a non-empty list")
    ids = tuple(dict.fromkeys(identifier(item) for item in raw if str(item).strip()))
    if not ids:
        raise ContractError("subject identifiers must not be blank")
    return ids


def extract_consent_scopes(record: DomainRecord) -> tuple[str, ...]:
    text = normalized_text(record.payload.get("consent_scope", ""))
    return tuple(part.strip() for part in text.replace(";", ",").split(",") if part.strip())


@dataclass(frozen=True)
class CohortObservation:
    """单个队列在某个阶段上的冻结观察值，是后续所有统计的唯一证据载体。"""

    cohort_id: str
    split_role: str
    hospital_code: str
    population_group: str
    subject_ids: tuple[str, ...]
    sample_size: float
    positive_rate: float
    sensitivity: float | None
    specificity: float | None
    missingness: float
    missing_results: tuple[str, ...]
    consent_scopes: tuple[str, ...]
    record_fingerprint: str
    observed_at: str
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "cohort_id", identifier(self.cohort_id))
        object.__setattr__(self, "split_role", normalized_text(self.split_role))
        if self.split_role not in SPLIT_ROLES:
            raise ContractError("split role must be selection or confirmation")
        object.__setattr__(self, "hospital_code", identifier(self.hospital_code))
        object.__setattr__(self, "population_group", identifier(self.population_group))
        if not self.subject_ids:
            raise ContractError("observation requires subject identifiers")
        if self.sample_size <= 0:
            raise ContractError("sample size must be positive")
        if not 0.0 <= self.positive_rate <= 1.0:
            raise ContractError("positive rate must stay within [0,1]")

    @classmethod
    def from_record(
        cls,
        record: DomainRecord,
        split_role: str,
        explicit_subject_ids: Sequence[str] | None = None,
    ) -> "CohortObservation":
        payload = record.payload
        subject_ids = extract_subject_ids(record, explicit_subject_ids)
        missing_results: list[str] = []
        sensitivity: float | None
        specificity: float | None
        if payload.get("sensitivity") in (None, ""):
            sensitivity = None
            missing_results.append("sensitivity")
        else:
            sensitivity = proportion(payload.get("sensitivity"))
        if payload.get("specificity") in (None, ""):
            specificity = None
            missing_results.append("specificity")
        else:
            specificity = proportion(payload.get("specificity"))
        if payload.get("positive_rate") in (None, "") or payload.get("sample_size") in (None, ""):
            raise ContractError("sample size and positive rate are required for an observation")
        return cls(
            cohort_id=record.record_id,
            split_role=split_role,
            hospital_code=payload.get("hospital_code", "unknown-hospital"),
            population_group=payload.get("population_group", "unknown-group"),
            subject_ids=subject_ids,
            sample_size=max(1.0, finite_number(payload.get("sample_size"))),
            positive_rate=proportion(payload.get("positive_rate")),
            sensitivity=sensitivity,
            specificity=specificity,
            missingness=proportion(payload.get("missingness", 0.0)),
            missing_results=tuple(missing_results),
            consent_scopes=extract_consent_scopes(record),
            record_fingerprint=record.fingerprint,
            observed_at=record.occurred_at,
            source=record.source,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CohortObservation":
        return cls(
            cohort_id=value["cohort_id"],
            split_role=value["split_role"],
            hospital_code=value["hospital_code"],
            population_group=value["population_group"],
            subject_ids=tuple(value["subject_ids"]),
            sample_size=float(value["sample_size"]),
            positive_rate=float(value["positive_rate"]),
            sensitivity=None if value.get("sensitivity") is None else float(value["sensitivity"]),
            specificity=None if value.get("specificity") is None else float(value["specificity"]),
            missingness=float(value["missingness"]),
            missing_results=tuple(value.get("missing_results", ())),
            consent_scopes=tuple(value.get("consent_scopes", ())),
            record_fingerprint=value["record_fingerprint"],
            observed_at=value["observed_at"],
            source=value["source"],
        )

    @property
    def estimable(self) -> bool:
        return self.sensitivity is not None and self.specificity is not None

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "cohort_id": self.cohort_id,
            "split_role": self.split_role,
            "hospital_code": self.hospital_code,
            "population_group": self.population_group,
            "subject_ids": list(self.subject_ids),
            "sample_size": self.sample_size,
            "positive_rate": self.positive_rate,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "missingness": self.missingness,
            "missing_results": list(self.missing_results),
            "consent_scopes": list(self.consent_scopes),
            "record_fingerprint": self.record_fingerprint,
            "observed_at": self.observed_at,
            "source": self.source,
        }


def confusion_counts(observation: CohortObservation) -> dict[str, float | None]:
    """返回 tp/fn/tn/fp；关键结局缺失时返回 None 而不是补零。"""
    positives = observation.sample_size * observation.positive_rate
    negatives = observation.sample_size - positives
    if not observation.estimable:
        return {"tp": None, "fn": None, "tn": None, "fp": None,
                "positives": positives, "negatives": negatives, "estimable": False}
    assert observation.sensitivity is not None and observation.specificity is not None
    tp = positives * observation.sensitivity
    fn = positives * (1.0 - observation.sensitivity)
    tn = negatives * observation.specificity
    fp = negatives * (1.0 - observation.specificity)
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "positives": positives, "negatives": negatives, "estimable": True}


def performance_profile(observation: CohortObservation) -> dict[str, Any]:
    """单队列漏诊/误报画像，区间采用 Wilson 95% 双侧界。"""
    counts = confusion_counts(observation)
    profile: dict[str, Any] = {
        "cohort_id": observation.cohort_id,
        "hospital_code": observation.hospital_code,
        "population_group": observation.population_group,
        "split_role": observation.split_role,
        "sample_size": observation.sample_size,
        "positive_rate": observation.positive_rate,
        "missingness": observation.missingness,
        "missing_results": list(observation.missing_results),
        "estimable": counts["estimable"],
        "subject_count": len(observation.subject_ids),
    }
    if not counts["estimable"]:
        profile.update({
            "sensitivity": None, "sensitivity_low": None, "sensitivity_high": None,
            "specificity": None, "specificity_low": None, "specificity_high": None,
            "missed_diagnosis_rate": None, "missed_diagnosis_rate_high": None,
            "false_positive_rate": None, "false_positive_rate_high": None,
        })
        return profile
    sens = observation.sensitivity
    spec = observation.specificity
    assert sens is not None and spec is not None
    sens_low, sens_high = wilson_interval(counts["tp"], counts["tp"] + counts["fn"])
    spec_low, spec_high = wilson_interval(counts["tn"], counts["tn"] + counts["fp"])
    profile.update({
        "sensitivity": sens, "sensitivity_low": sens_low, "sensitivity_high": sens_high,
        "specificity": spec, "specificity_low": spec_low, "specificity_high": spec_high,
        # 漏诊率上界 = 1 - 敏感度下界，是面向安全的保守口径。
        "missed_diagnosis_rate": 1.0 - sens,
        "missed_diagnosis_rate_high": 1.0 - sens_low,
        "false_positive_rate": 1.0 - spec,
        "false_positive_rate_high": 1.0 - spec_low,
    })
    return profile


def pool_observations(observations: Sequence[CohortObservation]) -> dict[str, Any]:
    """跨队列合并计数后重新计算区间，禁止对比例做简单平均。"""
    if not observations:
        raise ContractError("cannot pool an empty observation set")
    totals = {"tp": 0.0, "fn": 0.0, "tn": 0.0, "fp": 0.0,
              "positives": 0.0, "negatives": 0.0, "sample_size": 0.0}
    missing: set[str] = set()
    estimable = True
    for observation in observations:
        counts = confusion_counts(observation)
        totals["sample_size"] += observation.sample_size
        totals["positives"] += counts["positives"]
        totals["negatives"] += counts["negatives"]
        missing.update(observation.missing_results)
        if not counts["estimable"]:
            estimable = False
            continue
        for key in ("tp", "fn", "tn", "fp"):
            totals[key] += counts[key]
    weighted_missingness = (
        sum(item.sample_size * item.missingness for item in observations) / totals["sample_size"]
    )
    pooled: dict[str, Any] = {
        "sample_size": totals["sample_size"],
        "positive_cases": totals["positives"],
        "negative_cases": totals["negatives"],
        "missingness": bounded(weighted_missingness),
        "missing_results": sorted(missing),
        "estimable": estimable,
        "institution_count": len({item.hospital_code for item in observations}),
        "population_count": len({item.population_group for item in observations}),
        "subject_count": len({subject for item in observations for subject in item.subject_ids}),
    }
    if estimable:
        sens_low, sens_high = wilson_interval(totals["tp"], totals["tp"] + totals["fn"])
        spec_low, spec_high = wilson_interval(totals["tn"], totals["tn"] + totals["fp"])
        pooled.update({
            "sensitivity": totals["tp"] / max(totals["tp"] + totals["fn"], 1e-12),
            "sensitivity_low": sens_low, "sensitivity_high": sens_high,
            "specificity": totals["tn"] / max(totals["tn"] + totals["fp"], 1e-12),
            "specificity_low": spec_low, "specificity_high": spec_high,
            "missed_diagnosis_rate": totals["fn"] / max(totals["tp"] + totals["fn"], 1e-12),
            "missed_diagnosis_rate_high": 1.0 - sens_low,
            "false_positive_rate": totals["fp"] / max(totals["tn"] + totals["fp"], 1e-12),
            "false_positive_rate_high": 1.0 - spec_low,
        })
    else:
        for key in ("sensitivity", "sensitivity_low", "sensitivity_high", "specificity",
                    "specificity_low", "specificity_high", "missed_diagnosis_rate",
                    "missed_diagnosis_rate_high", "false_positive_rate",
                    "false_positive_rate_high"):
            pooled[key] = None
    return pooled


def subgroup_profiles(observations: Sequence[CohortObservation]) -> dict[str, Any]:
    """按人群和机构分别给出漏诊/误报，并计算最大组间差距。"""
    def _group(key: str) -> dict[str, Any]:
        buckets: dict[str, list[CohortObservation]] = {}
        for observation in observations:
            buckets.setdefault(getattr(observation, key), []).append(observation)
        rows = {name: pool_observations(items) for name, items in sorted(buckets.items())}
        sens_values = [row["sensitivity_low"] for row in rows.values() if row["estimable"]]
        fpr_values = [row["false_positive_rate_high"] for row in rows.values() if row["estimable"]]
        return {
            "groups": rows,
            "sensitivity_gap": (max(sens_values) - min(sens_values)) if len(sens_values) > 1 else 0.0,
            "false_positive_rate_gap": (max(fpr_values) - min(fpr_values)) if len(fpr_values) > 1 else 0.0,
        }

    return {"by_population_group": _group("population_group"),
            "by_hospital": _group("hospital_code")}


def leakage_report(
    selection_observations: Sequence[CohortObservation],
    confirmation_observations: Sequence[CohortObservation],
) -> dict[str, Any]:
    """按受试者和机构两个维度检查选择集与确认集是否混用。"""
    selection_subjects = {s for item in selection_observations for s in item.subject_ids}
    confirmation_subjects = {s for item in confirmation_observations for s in item.subject_ids}
    selection_hospitals = {item.hospital_code for item in selection_observations}
    confirmation_hospitals = {item.hospital_code for item in confirmation_observations}
    return {
        "subject_overlap": sorted(selection_subjects & confirmation_subjects),
        "hospital_overlap": sorted(selection_hospitals & confirmation_hospitals),
        "selection_subject_count": len(selection_subjects),
        "confirmation_subject_count": len(confirmation_subjects),
        "selection_hospitals": sorted(selection_hospitals),
        "confirmation_hospitals": sorted(confirmation_hospitals),
        "disjoint": not (selection_subjects & confirmation_subjects)
        and not (selection_hospitals & confirmation_hospitals),
    }


def assert_disjoint(
    selection_observations: Sequence[CohortObservation],
    confirmation_observations: Sequence[CohortObservation],
) -> dict[str, Any]:
    report = leakage_report(selection_observations, confirmation_observations)
    if report["subject_overlap"]:
        raise ContractError(
            "subject overlap between selection and confirmation: " + ",".join(report["subject_overlap"])
        )
    if report["hospital_overlap"]:
        raise ContractError(
            "hospital overlap between selection and confirmation: " + ",".join(report["hospital_overlap"])
        )
    return report


def adequacy_flags(pooled: Mapping[str, Any], context: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    context = context or {}
    min_total = finite_number(context.get("min_total_subjects", MIN_TOTAL_SUBJECTS))
    min_positive = finite_number(context.get("min_positive_cases", MIN_POSITIVE_CASES))
    min_negative = finite_number(context.get("min_negative_cases", MIN_NEGATIVE_CASES))
    max_missingness = finite_number(context.get("max_missingness", MAX_MISSINGNESS))
    flags: list[str] = []
    if pooled.get("missing_results"):
        flags.append("missing_outcomes")
    if pooled["sample_size"] < min_total:
        flags.append("small_sample")
    if pooled["positive_cases"] < min_positive or pooled["negative_cases"] < min_negative:
        flags.append("insufficient_case_counts")
    if pooled["missingness"] > max_missingness:
        flags.append("high_missingness")
    if pooled["institution_count"] < 2:
        flags.append("single_institution")
    return tuple(flags)


def supported_uses(flags: Sequence[str]) -> dict[str, Any]:
    """小样本/缺失时明确证据仍支持哪些用途、不能支持哪些用途。"""
    reasons: dict[str, str] = {}
    if "missing_outcomes" in flags:
        reasons["missing_outcomes"] = "关键结局缺失，只能用于候选筛选与方案完善，不能作独立确认或临床部署"
        return {
            "allowed": list(SCREENING_USES),
            "withheld": list(WITHHELD_DEPLOYMENT) + ["independent_confirmation"],
            "reasons": reasons,
        }
    limited = {"small_sample", "insufficient_case_counts", "high_missingness", "single_institution"}
    if limited.intersection(flags):
        for flag in limited.intersection(flags):
            reasons[flag] = {
                "small_sample": "总样本量低于充分性门槛，区间较宽",
                "insufficient_case_counts": "阳/阴性病例数不足，漏诊与误报估计不稳定",
                "high_missingness": "缺失率超过门槛，结论可能存在选择偏倚",
                "single_institution": "仅覆盖单一机构，不能支撑新机构泛化声明",
            }[flag]
        return {
            "allowed": list(LIMITED_USES),
            "withheld": list(WITHHELD_DEPLOYMENT),
            "reasons": reasons,
        }
    reasons["full_evidence"] = "样本量、病例数、缺失率与机构覆盖均满足充分性门槛"
    return {"allowed": list(FULL_USES), "withheld": [], "reasons": reasons}


def evidence_bundle(
    observations: Sequence[CohortObservation],
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """一次评估的完整证据包：总体指标 + 分层差距 + 充分性 + 可用用途。"""
    pooled = pool_observations(observations)
    flags = adequacy_flags(pooled, context)
    return {
        "pooled": pooled,
        "subgroups": subgroup_profiles(observations),
        "adequacy_flags": list(flags),
        "supported_uses": supported_uses(flags),
        "cohort_fingerprints": sorted({item.fingerprint for item in observations}),
        "record_fingerprints": sorted({item.record_fingerprint for item in observations}),
    }


@dataclass(frozen=True)
class Candidate:
    model_id: str
    operating_threshold: float
    observations: tuple[CohortObservation, ...]
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.model_id}@{self.operating_threshold:.6g}"


def select_candidate(
    candidates: Sequence[Candidate],
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """只使用选择集证据做候选排序，输出确定性的获胜模型/阈值与完整排名。

    排序口径（安全优先）：可估计者优先；敏感度 Wilson 下界高者优先（漏诊更少）；
    其次特异度下界（误报更少）；再次人群差距小者；最后按标识确定性破平。
    """
    if not candidates:
        raise ContractError("at least one candidate is required for selection")
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        evidence = candidate.evidence or evidence_bundle(candidate.observations, context)
        pooled = evidence["pooled"]
        rows.append({
            "candidate_key": candidate.key,
            "model_id": candidate.model_id,
            "operating_threshold": candidate.operating_threshold,
            "eligible": pooled["estimable"],
            "sensitivity_low": pooled["sensitivity_low"],
            "specificity_low": pooled["specificity_low"],
            "subgroup_sensitivity_gap": evidence["subgroups"]["by_population_group"]["sensitivity_gap"],
            "adequacy_flags": evidence["adequacy_flags"],
            "supported_uses": evidence["supported_uses"],
            "evidence_fingerprint": stable_hash(evidence),
        })

    def rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            0 if row["eligible"] else 1,
            -(row["sensitivity_low"] or -1.0),
            -(row["specificity_low"] or -1.0),
            row["subgroup_sensitivity_gap"],
            row["model_id"],
            row["operating_threshold"],
        )

    ranking = sorted(rows, key=rank_key)
    winner = ranking[0]
    if not winner["eligible"]:
        raise ContractError("no candidate has complete sensitivity/specificity outcomes; selection cannot proceed")
    return {
        "selected_model_id": winner["model_id"],
        "selected_operating_threshold": winner["operating_threshold"],
        "winner_key": winner["candidate_key"],
        "ranking": ranking,
        "frozen_at": utc_now(),
        "rule": "sensitivity_low desc, specificity_low desc, subgroup gap asc, deterministic id tie-break",
    }
