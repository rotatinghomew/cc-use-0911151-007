from __future__ import annotations

import math
import unittest

from clinic_localizer.cohorts.protocol import ContractError, DomainRecord
from clinic_localizer.statistics.two_stage_metrics import (
    Candidate,
    CohortObservation,
    adequacy_flags,
    assert_disjoint,
    confusion_counts,
    evidence_bundle,
    leakage_report,
    performance_profile,
    pool_observations,
    proportion,
    select_candidate,
    subgroup_profiles,
    supported_uses,
)


def make_record(record_id: str, hospital: str = "h1", group: str = "g1",
                subjects=None, sensitivity=0.9, specificity=0.85,
                sample_size=100, positive_rate=0.2, missingness=0.05,
                consent_scope="validation") -> DomainRecord:
    return DomainRecord(record_id, {
        "hospital_code": hospital,
        "population_group": group,
        "condition_code": "c1",
        "sample_size": sample_size,
        "positive_rate": positive_rate,
        "consent_scope": consent_scope,
        "terminology_version": "v1",
        "missingness": missingness,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "adverse_events": 0,
        "review_board": "rb1",
        "subject_ids": subjects if subjects is not None else [f"{record_id}-s{idx}" for idx in range(5)],
    })


def make_observation(record_id: str, split_role: str = "selection", **kwargs) -> CohortObservation:
    return CohortObservation.from_record(make_record(record_id, **kwargs), split_role)


class ProportionAndObservationTests(unittest.TestCase):
    def test_percentage_values_are_normalized(self):
        self.assertEqual(proportion(90), 0.9)
        self.assertEqual(proportion(0.9), 0.9)
        self.assertEqual(proportion(1.2), 0.012)

    def test_observation_requires_subject_identifiers(self):
        record = DomainRecord("no-subjects", {"sample_size": 10, "positive_rate": 0.2,
                                              "sensitivity": 0.9, "specificity": 0.8})
        with self.assertRaisesRegex(ContractError, "subject identifiers"):
            CohortObservation.from_record(record, "selection")
        explicit = CohortObservation.from_record(record, "selection", explicit_subject_ids=["x-1"])
        self.assertEqual(explicit.subject_ids, ("x-1",))

    def test_observation_round_trips_through_dict(self):
        observation = make_observation("o1")
        self.assertEqual(CohortObservation.from_dict(observation.as_dict()), observation)

    def test_missing_outcomes_are_kept_not_filled(self):
        record = make_record("o2")
        record = record.with_payload({"sensitivity": None})
        observation = CohortObservation.from_record(record, "confirmation",
                                                    explicit_subject_ids=record.payload["subject_ids"])
        self.assertIn("sensitivity", observation.missing_results)
        self.assertFalse(observation.estimable)
        counts = confusion_counts(observation)
        self.assertFalse(counts["estimable"])
        self.assertIsNone(counts["tp"])


class PerformanceProfileTests(unittest.TestCase):
    def test_wilson_bounds_are_conservative(self):
        profile = performance_profile(make_observation("p1", sample_size=200, positive_rate=0.5))
        self.assertLess(profile["sensitivity_low"], profile["sensitivity"])
        self.assertLess(profile["sensitivity"], profile["sensitivity_high"])
        # 漏诊率上界 = 1 - 敏感度下界
        self.assertAlmostEqual(profile["missed_diagnosis_rate_high"], 1.0 - profile["sensitivity_low"])
        self.assertAlmostEqual(profile["false_positive_rate_high"], 1.0 - profile["specificity_low"])

    def test_missing_outcome_profile_uses_none(self):
        record = make_record("p2").with_payload({"specificity": None})
        observation = CohortObservation.from_record(record, "selection",
                                                    explicit_subject_ids=record.payload["subject_ids"])
        profile = performance_profile(observation)
        self.assertFalse(profile["estimable"])
        self.assertIsNone(profile["false_positive_rate"])

    def test_pooling_recomputes_intervals_from_counts(self):
        left = make_observation("pool-1", hospital="h1", group="g1", sample_size=100)
        right = make_observation("pool-2", hospital="h2", group="g2", sample_size=300)
        pooled = pool_observations([left, right])
        self.assertEqual(pooled["sample_size"], 400)
        self.assertEqual(pooled["institution_count"], 2)
        self.assertTrue(math.isfinite(pooled["sensitivity_low"]))
        self.assertLessEqual(pooled["sensitivity_low"], pooled["sensitivity"])

    def test_pooling_empty_is_rejected(self):
        with self.assertRaisesRegex(ContractError, "empty observation set"):
            pool_observations([])


class LeakageGuardTests(unittest.TestCase):
    def test_disjoint_splits_pass(self):
        selection = [make_observation("s1", hospital="h1", subjects=["a", "b"])]
        confirmation = [make_observation("c1", hospital="h9", subjects=["x", "y"],
                                         split_role="confirmation")]
        report = leakage_report(selection, confirmation)
        self.assertTrue(report["disjoint"])
        self.assertEqual(assert_disjoint(selection, confirmation), report)

    def test_subject_overlap_is_blocked(self):
        selection = [make_observation("s2", hospital="h1", subjects=["a", "b"])]
        confirmation = [make_observation("c2", hospital="h9", subjects=["b", "c"],
                                         split_role="confirmation")]
        with self.assertRaisesRegex(ContractError, "subject overlap"):
            assert_disjoint(selection, confirmation)

    def test_hospital_overlap_is_blocked_even_with_unique_subjects(self):
        selection = [make_observation("s3", hospital="h1", subjects=["a"])]
        confirmation = [make_observation("c3", hospital="h1", subjects=["z"],
                                         split_role="confirmation")]
        with self.assertRaisesRegex(ContractError, "hospital overlap"):
            assert_disjoint(selection, confirmation)

    def test_within_split_duplicate_subjects_are_visible(self):
        selection = [
            make_observation("s4", hospital="h1", subjects=["a"]),
            make_observation("s5", hospital="h2", group="g2", subjects=["a", "b"]),
        ]
        report = leakage_report(selection, [make_observation("c4", hospital="h9",
                                                             subjects=["z"], split_role="confirmation")])
        self.assertEqual(report["selection_subject_count"], 2)


class SubgroupAndAdequacyTests(unittest.TestCase):
    def _two_groups(self):
        return [
            make_observation("g-a", hospital="h1", group="group-a", sensitivity=0.95, specificity=0.9),
            make_observation("g-b", hospital="h2", group="group-b", sensitivity=0.75, specificity=0.8),
        ]

    def test_subgroup_gap_quantifies_population_difference(self):
        profiles = subgroup_profiles(self._two_groups())
        self.assertIn("group-a", profiles["by_population_group"]["groups"])
        self.assertGreater(profiles["by_population_group"]["sensitivity_gap"], 0.0)
        self.assertEqual(profiles["by_hospital"]["sensitivity_gap"],
                         profiles["by_population_group"]["sensitivity_gap"])

    def test_small_sample_limits_supported_uses(self):
        pooled = pool_observations([make_observation("small", sample_size=8, positive_rate=0.5)])
        flags = adequacy_flags(pooled)
        self.assertIn("small_sample", flags)
        uses = supported_uses(flags)
        self.assertIn("limited_internal_qa", uses["allowed"])
        self.assertIn("standalone_diagnosis", uses["withheld"])

    def test_missing_outcomes_restrict_to_screening(self):
        record = make_record("missing").with_payload({"sensitivity": None})
        observation = CohortObservation.from_record(record, "selection",
                                                    explicit_subject_ids=record.payload["subject_ids"])
        bundle = evidence_bundle([observation])
        self.assertIn("missing_outcomes", bundle["adequacy_flags"])
        self.assertEqual(bundle["supported_uses"]["allowed"],
                         ["candidate_screening", "protocol_refinement"])

    def test_full_evidence_supports_deployment(self):
        observations = [
            make_observation("f1", hospital="h1", group="g1", sample_size=120),
            make_observation("f2", hospital="h2", group="g2", sample_size=120),
        ]
        bundle = evidence_bundle(observations)
        self.assertEqual(bundle["adequacy_flags"], [])
        self.assertIn("deployment_with_post_market_monitoring", bundle["supported_uses"]["allowed"])
        self.assertEqual(bundle["supported_uses"]["withheld"], [])

    def test_single_institution_blocks_new_institution_claim(self):
        pooled = pool_observations([make_observation("one", hospital="h1")])
        self.assertIn("single_institution", adequacy_flags(pooled))

    def test_high_missingness_flag(self):
        pooled = pool_observations([make_observation("hm", missingness=0.8)])
        self.assertIn("high_missingness", adequacy_flags(pooled))


class CandidateSelectionTests(unittest.TestCase):
    def _candidate(self, model_id: str, threshold: float, sensitivity: float,
                   specificity: float) -> Candidate:
        observations = [
            make_observation(f"{model_id}-a", hospital="h1", group="g1",
                             sensitivity=sensitivity, specificity=specificity),
            make_observation(f"{model_id}-b", hospital="h2", group="g2",
                             sensitivity=sensitivity - 0.02, specificity=specificity),
        ]
        return Candidate(model_id, threshold, tuple(observations), evidence_bundle(observations))

    def test_higher_sensitivity_lower_bound_wins(self):
        weaker = self._candidate("model-a", 0.4, 0.82, 0.9)
        stronger = self._candidate("model-b", 0.5, 0.92, 0.9)
        decision = select_candidate([weaker, stronger])
        self.assertEqual(decision["selected_model_id"], "model-b")
        self.assertEqual(decision["selected_operating_threshold"], 0.5)
        self.assertEqual(len(decision["ranking"]), 2)

    def test_tie_break_is_deterministic(self):
        left = self._candidate("model-a", 0.4, 0.9, 0.9)
        right = self._candidate("model-a", 0.5, 0.9, 0.9)
        first = select_candidate([left, right])
        second = select_candidate([right, left])
        self.assertEqual(first["winner_key"], second["winner_key"])

    def test_candidate_with_missing_outcomes_cannot_win(self):
        broken = Candidate(
            "model-broken", 0.3,
            (CohortObservation.from_record(
                make_record("broken").with_payload({"sensitivity": None}),
                "selection", explicit_subject_ids=["broken-1"]),),
        )
        healthy = self._candidate("model-ok", 0.5, 0.9, 0.9)
        decision = select_candidate([broken, healthy])
        self.assertEqual(decision["selected_model_id"], "model-ok")

    def test_all_broken_candidates_block_selection(self):
        record = make_record("all-broken").with_payload({"sensitivity": None})
        broken = Candidate("model-broken", 0.3,
                           (CohortObservation.from_record(
                               record, "selection",
                               explicit_subject_ids=record.payload["subject_ids"]),))
        with self.assertRaisesRegex(ContractError, "no candidate has complete"):
            select_candidate([broken])

    def test_empty_candidate_list_is_rejected(self):
        with self.assertRaisesRegex(ContractError, "at least one candidate"):
            select_candidate([])


if __name__ == "__main__":
    unittest.main()
