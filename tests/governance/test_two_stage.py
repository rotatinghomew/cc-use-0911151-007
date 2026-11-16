from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clinic_localizer.cohorts.protocol import ContractError, DomainRecord
from clinic_localizer.evidence.registry import Repository
from clinic_localizer.statistics.two_stage_metrics import CohortObservation


def make_record(record_id: str, hospital: str, group: str, subjects, *,
                sample_size=120, positive_rate=0.25, sensitivity=0.92,
                specificity=0.9, missingness=0.05, consent_scope="validation") -> DomainRecord:
    return DomainRecord(record_id, {
        "hospital_code": hospital,
        "population_group": group,
        "condition_code": "condition-1",
        "sample_size": sample_size,
        "positive_rate": positive_rate,
        "consent_scope": consent_scope,
        "terminology_version": "v1",
        "missingness": missingness,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "adverse_events": 0,
        "review_board": "board-1",
        "subject_ids": list(subjects),
    })


def with_metrics(record: DomainRecord, **changes) -> DomainRecord:
    payload = dict(record.payload)
    payload.update(changes)
    return DomainRecord(record.record_id, payload)


class TwoStageCycleTestBase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repository = Repository(Path(self.temp.name) / "two-stage.sqlite3")
        self.selection_records = [
            make_record("sel-a", "hospital-1", "group-1", [f"s-{idx}" for idx in range(60)]),
            make_record("sel-b", "hospital-2", "group-2", [f"s-{idx}" for idx in range(60, 120)]),
        ]
        self.confirmation_records = [
            make_record("con-a", "hospital-3", "group-1", [f"c-{idx}" for idx in range(60)]),
            make_record("con-b", "hospital-4", "group-2", [f"c-{idx}" for idx in range(60, 120)]),
        ]
        for record in self.selection_records + self.confirmation_records:
            self.repository.save_record(record)
        self.repository.create_cycle("cycle-1", "product-1", "screening")
        self.repository.assign_split(
            "cycle-1", "selection",
            [CohortObservation.from_record(item, "selection") for item in self.selection_records],
        )
        self.repository.assign_split(
            "cycle-1", "confirmation",
            [CohortObservation.from_record(item, "confirmation") for item in self.confirmation_records],
        )

    def selection_observations(self, sensitivity=0.92, specificity=0.9):
        observations = []
        for record in self.selection_records:
            changed = with_metrics(record, sensitivity=sensitivity, specificity=specificity)
            observations.append(CohortObservation.from_record(
                changed, "selection", explicit_subject_ids=changed.payload["subject_ids"]))
        return observations

    def confirmation_observations(self, **changes):
        observations = []
        for record in self.confirmation_records:
            changed = with_metrics(record, **changes) if changes else record
            observations.append(CohortObservation.from_record(
                changed, "confirmation", explicit_subject_ids=changed.payload["subject_ids"]))
        return observations

    def freeze_with_two_candidates(self):
        self.repository.record_candidate("cycle-1", "model-a", 0.4,
                                         self.selection_observations(0.86, 0.84))
        self.repository.record_candidate("cycle-1", "model-b", 0.5,
                                         self.selection_observations(0.93, 0.91))
        return self.repository.freeze_selection("cycle-1")


class SplitLeakageTests(TwoStageCycleTestBase):
    def test_subject_overlap_between_splits_is_rejected(self):
        self.repository.create_cycle("cycle-leak", "product-1", "screening")
        leaked = make_record("leak-a", "hospital-9", "group-1", ["s-1", "s-2"])
        self.repository.save_record(leaked)
        self.repository.assign_split(
            "cycle-leak", "selection",
            [CohortObservation.from_record(leaked, "selection")],
        )
        with self.assertRaisesRegex(ContractError, "subject overlap"):
            self.repository.assign_split(
                "cycle-leak", "confirmation",
                [CohortObservation.from_record(leaked, "confirmation",
                                               explicit_subject_ids=leaked.payload["subject_ids"])],
            )

    def test_hospital_overlap_between_splits_is_rejected(self):
        self.repository.create_cycle("cycle-hosp", "product-1", "screening")
        shared = make_record("hosp-a", "hospital-1", "group-9", ["z-1"])
        other = make_record("hosp-b", "hospital-1", "group-8", ["z-2"])
        self.repository.save_record(shared)
        self.repository.save_record(other)
        self.repository.assign_split(
            "cycle-hosp", "selection",
            [CohortObservation.from_record(shared, "selection")],
        )
        with self.assertRaisesRegex(ContractError, "hospital overlap"):
            self.repository.assign_split(
                "cycle-hosp", "confirmation",
                [CohortObservation.from_record(other, "confirmation",
                                               explicit_subject_ids=other.payload["subject_ids"])],
            )

    def test_duplicate_subject_within_split_is_rejected(self):
        self.repository.create_cycle("cycle-dup", "product-1", "screening")
        first = make_record("dup-a", "hospital-7", "group-1", ["u-1"])
        second = make_record("dup-b", "hospital-8", "group-2", ["u-1", "u-2"])
        self.repository.save_record(first)
        self.repository.save_record(second)
        self.repository.assign_split("cycle-dup", "selection",
                                     [CohortObservation.from_record(first, "selection")])
        with self.assertRaisesRegex(ContractError, "duplicate subject"):
            self.repository.assign_split("cycle-dup", "selection",
                                         [CohortObservation.from_record(second, "selection")])

    def test_split_assignment_requires_registered_unchanged_record(self):
        self.repository.create_cycle("cycle-stale", "product-1", "screening")
        stale = make_record("stale-a", "hospital-6", "group-1", ["w-1"])
        self.repository.save_record(stale)
        observation = CohortObservation.from_record(stale, "selection")
        self.repository.save_record(stale.with_payload({"missingness": 0.2}), expected_version=1)
        with self.assertRaisesRegex(ContractError, "changed before split assignment"):
            self.repository.assign_split("cycle-stale", "selection", [observation])


class SelectionFreezeTests(TwoStageCycleTestBase):
    def test_winner_and_full_ranking_are_persisted(self):
        decision = self.freeze_with_two_candidates()
        self.assertEqual(decision["selected_model_id"], "model-b")
        self.assertEqual(decision["selected_operating_threshold"], 0.5)
        history = self.repository.cycle_history("cycle-1")
        self.assertEqual(history["cycle"]["status"], "selection_frozen")
        self.assertEqual(history["selection"]["model_id"], "model-b")
        self.assertGreaterEqual(len(history["selection"]["ranking"]), 2)

    def test_expected_winner_mismatch_blocks_freeze(self):
        self.repository.record_candidate("cycle-1", "model-a", 0.4,
                                         self.selection_observations(0.93, 0.91))
        with self.assertRaisesRegex(ContractError, "expected model"):
            self.repository.freeze_selection("cycle-1", expected_model_id="model-other")

    def test_candidate_cannot_use_confirmation_data(self):
        with self.assertRaisesRegex(ContractError, "selection-split observations only"):
            self.repository.record_candidate(
                "cycle-1", "model-x", 0.9,
                [CohortObservation.from_record(self.confirmation_records[0], "confirmation")],
            )

    def test_recording_after_freeze_is_blocked(self):
        self.freeze_with_two_candidates()
        with self.assertRaisesRegex(ContractError, "does not permit this action"):
            self.repository.record_candidate("cycle-1", "model-late", 0.7,
                                             self.selection_observations())


class ConfirmationLockTests(TwoStageCycleTestBase):
    def test_confirmation_uses_frozen_model_and_threshold_without_tuning(self):
        self.freeze_with_two_candidates()
        review = self.repository.submit_confirmation(
            "review-1", "cycle-1", self.confirmation_observations())
        self.assertFalse(review["tuning_allowed"])
        self.assertEqual(review["model_id"], "model-b")
        self.assertEqual(review["operating_threshold"], 0.5)
        self.assertEqual(review["leakage"]["subject_overlap"], [])

    def test_confirmation_rejects_unknown_cohort(self):
        self.freeze_with_two_candidates()
        extra = make_record("con-x", "hospital-5", "group-3", ["x-1", "x-2"])
        self.repository.save_record(extra)
        with self.assertRaisesRegex(ContractError, "not part of the frozen confirmation split"):
            self.repository.submit_confirmation(
                "review-x", "cycle-1",
                [CohortObservation.from_record(extra, "confirmation")],
            )

    def test_late_supplemental_confirmation_supersedes_previous_review(self):
        self.freeze_with_two_candidates()
        first = self.repository.submit_confirmation(
            "review-1", "cycle-1", self.confirmation_observations(sensitivity=0.8))
        second = self.repository.submit_confirmation(
            "review-2", "cycle-1", self.confirmation_observations(sensitivity=0.93))
        history = self.repository.cycle_history("cycle-1")
        reviews = history["confirmation_reviews"]
        self.assertEqual(len(reviews), 2)
        self.assertEqual([item["superseded"] for item in reviews], [1, 0])
        self.assertNotEqual(first["submitted_at"], second["submitted_at"])

    def test_supplement_after_issuance_is_rejected_but_restriction_can_be_appended(self):
        self.freeze_with_two_candidates()
        self.repository.submit_confirmation("review-1", "cycle-1", self.confirmation_observations())
        acceptance = self.repository.issue_acceptance("acceptance-1", "cycle-1")
        with self.assertRaisesRegex(ContractError, "does not permit this action"):
            self.repository.submit_confirmation(
                "review-late", "cycle-1", self.confirmation_observations(sensitivity=0.7))
        self.repository.restrict_acceptance(
            "restriction-1", acceptance["acceptance_id"], "supplemental_confirmation",
            "supplemental confirmation received after issuance; pending review",
            {"review": "review-late"},
        )
        history = self.repository.cycle_history("cycle-1")
        self.assertEqual(history["acceptance"]["outcome"], "accepted_full")
        self.assertEqual(history["acceptance"]["restrictions"][0]["trigger_type"],
                         "supplemental_confirmation")


class IssuanceRecheckTests(TwoStageCycleTestBase):
    def test_full_acceptance_names_model_threshold_and_evidence_scope(self):
        self.freeze_with_two_candidates()
        self.repository.submit_confirmation("review-1", "cycle-1", self.confirmation_observations())
        acceptance = self.repository.issue_acceptance("acceptance-1", "cycle-1")
        self.assertEqual(acceptance["outcome"], "accepted_full")
        self.assertEqual(acceptance["model_id"], "model-b")
        self.assertEqual(acceptance["operating_threshold"], 0.5)
        scope = acceptance["evidence_scope"]
        self.assertEqual(set(scope["split"]["institutions"]), {"hospital-3", "hospital-4"})
        self.assertEqual(len(scope["split"]["confirmation_record_fingerprints"]), 2)
        self.assertIsNotNone(scope["metrics_window"]["missed_diagnosis_rate_high"])
        history = self.repository.cycle_history("cycle-1")
        self.assertTrue(history["acceptance"]["basis"]["basis_recheck_at_issuance"]["passed"])

    def test_consent_withdrawal_blocks_pending_issuance(self):
        self.freeze_with_two_candidates()
        self.repository.submit_confirmation("review-1", "cycle-1", self.confirmation_observations())
        withdrawn = self.confirmation_records[0].with_payload({"consent_scope": "research-only"})
        self.repository.save_record(withdrawn, expected_version=1)
        report = self.repository.basis_drift_report("cycle-1")
        self.assertTrue(report["blocks_pending_issuance"])
        self.assertIn("consent_withdrawn", {item["type"] for item in report["drift"]})
        with self.assertRaisesRegex(ContractError, "basis recheck failed"):
            self.repository.issue_acceptance("acceptance-blocked", "cycle-1")

    def test_confirmation_record_change_blocks_issuance(self):
        self.freeze_with_two_candidates()
        self.repository.submit_confirmation("review-1", "cycle-1", self.confirmation_observations())
        changed = self.confirmation_records[1].with_payload({"sample_size": 130})
        self.repository.save_record(changed, expected_version=1)
        with self.assertRaisesRegex(ContractError, "record_changed"):
            self.repository.issue_acceptance("acceptance-stale", "cycle-1")

    def test_issuance_is_idempotent_and_keeps_original_conclusion(self):
        self.freeze_with_two_candidates()
        self.repository.submit_confirmation("review-1", "cycle-1", self.confirmation_observations())
        first = self.repository.issue_acceptance("acceptance-1", "cycle-1")
        second = self.repository.issue_acceptance("acceptance-1", "cycle-1")
        self.assertEqual(first["basis_hash"], second["basis_hash"])

    def test_safety_floor_failure_rejects_acceptance(self):
        self.freeze_with_two_candidates()
        self.repository.submit_confirmation(
            "review-1", "cycle-1",
            self.confirmation_observations(sensitivity=0.72, specificity=0.7),
        )
        with self.assertRaisesRegex(ContractError, "acceptance rejected"):
            self.repository.issue_acceptance(
                "acceptance-1", "cycle-1",
                context={"sensitivity_low_floor": 0.85},
            )
        # 被拒后不存在签发记录，避免失败审批被当作接受材料引用。
        self.assertIsNone(self.repository.cycle_history("cycle-1")["acceptance"])

    def test_limited_acceptance_requires_explicit_flag_and_lists_uses(self):
        self.freeze_with_two_candidates()
        small = [
            CohortObservation.from_record(
                with_metrics(record, sample_size=20, positive_rate=0.5),
                "confirmation", explicit_subject_ids=record.payload["subject_ids"])
            for record in self.confirmation_records
        ]
        self.repository.submit_confirmation("review-1", "cycle-1", small)
        # 安全底线通过、但充分性标志存在：未显式允许限制接受时拒绝签发。
        limited_context = {"sensitivity_low_floor": 0.6, "specificity_low_floor": 0.6}
        with self.assertRaisesRegex(ContractError, "limited acceptance"):
            self.repository.issue_acceptance("acceptance-1", "cycle-1", context=limited_context)
        acceptance = self.repository.issue_acceptance(
            "acceptance-1", "cycle-1", allow_limited=True, context=limited_context)
        self.assertEqual(acceptance["outcome"], "accepted_limited")
        self.assertIn("limited_internal_qa", acceptance["supported_uses"]["allowed"])
        self.assertIn("new_institution_deployment", acceptance["supported_uses"]["withheld"])

    def test_missing_outcomes_are_rejected_without_pretending_support(self):
        self.freeze_with_two_candidates()
        observations = self.confirmation_observations()
        broken_record = self.confirmation_records[0].with_payload({"sensitivity": None})
        observations[0] = CohortObservation.from_record(
            broken_record, "confirmation", explicit_subject_ids=broken_record.payload["subject_ids"])
        self.repository.submit_confirmation("review-1", "cycle-1", observations)
        with self.assertRaisesRegex(ContractError, "acceptance rejected") as context:
            self.repository.issue_acceptance("acceptance-1", "cycle-1")
        # 拒绝信息中仍写明证据当前支持的用途，而不是伪造接受结论。
        self.assertIn("candidate_screening", str(context.exception))
        self.assertIsNone(self.repository.cycle_history("cycle-1")["acceptance"])


class PostIssuanceHistoryTests(TwoStageCycleTestBase):
    def test_issued_material_keeps_point_in_time_conclusion_and_records_restriction(self):
        self.freeze_with_two_candidates()
        self.repository.submit_confirmation("review-1", "cycle-1", self.confirmation_observations())
        acceptance = self.repository.issue_acceptance("acceptance-1", "cycle-1")
        original_hash = acceptance["basis_hash"]
        # 签发后撤回同意：原结论不变，漂移可查，限制意见追加。
        withdrawn = self.confirmation_records[0].with_payload({"consent_scope": "research-only"})
        self.repository.save_record(withdrawn, expected_version=1)
        report = self.repository.basis_drift_report("cycle-1")
        self.assertIn("consent_withdrawn", {item["type"] for item in report["drift"]})
        self.repository.restrict_acceptance(
            "restriction-1", acceptance["acceptance_id"], "consent_withdrawal",
            "validation consent withdrawn after issuance",
            report["drift"],
        )
        history = self.repository.cycle_history("cycle-1")
        self.assertEqual(history["acceptance"]["basis_hash"], original_hash)
        self.assertEqual(history["acceptance"]["outcome"], "accepted_full")
        self.assertEqual(len(history["acceptance"]["restrictions"]), 1)
        self.assertEqual(history["drift"]["restrictions"][0]["reason"],
                         "validation consent withdrawn after issuance")

    def test_restriction_requires_issued_acceptance_and_reason(self):
        with self.assertRaisesRegex(ContractError, "issued acceptance"):
            self.repository.restrict_acceptance("r1", "missing", "consent_withdrawal", "reason")

    def test_audit_chain_still_verifies_after_full_cycle(self):
        self.freeze_with_two_candidates()
        self.repository.submit_confirmation("review-1", "cycle-1", self.confirmation_observations())
        acceptance = self.repository.issue_acceptance("acceptance-1", "cycle-1")
        self.repository.restrict_acceptance("r1", acceptance["acceptance_id"],
                                             "monitoring", "post-market signal")
        self.assertEqual(self.repository.verify_chain(), [])
        counts = self.repository.snapshot()["counts"]
        self.assertEqual(counts["validation_cycles"], 1)
        self.assertEqual(counts["safety_acceptances"], 1)
        self.assertEqual(counts["acceptance_restrictions"], 1)


if __name__ == "__main__":
    unittest.main()
