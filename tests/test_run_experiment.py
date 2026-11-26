import json
import subprocess
import sys
import unittest
from pathlib import Path

from projects.run_experiment import make_plan, run_smoke


LAUNCHER = Path(__file__).resolve().parents[1] / "projects" / "run_experiment.py"


class ExperimentTests(unittest.TestCase):
    def test_matrix_has_one_shared_kd_baseline(self):
        result = subprocess.run(
            [sys.executable, str(LAUNCHER), "--matrix"],
            capture_output=True, text=True, check=True,
        )
        matrix = json.loads(result.stdout)
        kd_settings = {
            (plan["teacher_top_k"], plan["teacher_temperature"])
            for plan in matrix if plan["preset"] == "kd"
        }
        self.assertEqual(len(matrix), 11)
        self.assertEqual(kd_settings, {
            (1, 1.0), (4, 1.0), (16, 1.0), (64, 1.0), (256, 1.0),
            (256, 0.5), (256, 0.25),
        })

    def test_paper_matrix_includes_both_token_routing_thresholds(self):
        result = subprocess.run(
            [sys.executable, str(LAUNCHER), "--paper-matrix", "--smoke"],
            capture_output=True, text=True, check=True,
        )
        matrix = json.loads(result.stdout)
        self.assertEqual(len(matrix), 13)
        self.assertEqual(
            {plan["preset"] for plan in matrix},
            {"lm", "kd", "domain", "otmass", "entropy"},
        )
        token_routes = [plan for plan in matrix if "routing_threshold" in plan]
        self.assertEqual({
            (plan["routing"], plan["routing_threshold"], plan["kd_when"])
            for plan in token_routes
        }, {
            ("otmass", 0.3678, "below_threshold"),
            ("otmass", 0.9876, "below_threshold"),
            ("entropy", 0.1, "above_threshold"),
            ("entropy", 0.5, "above_threshold"),
        })
        actual_weights = {}
        for plan in token_routes:
            self.assertEqual((plan["teacher_top_k"], plan["teacher_temperature"]), (256, 1.0))
            self.assertIsNone(plan["planned_kd_weight"])
            weight = plan["smoke"]["effective_kd_weight"]
            self.assertTrue(0 < weight < 1)
            actual_weights[plan["preset"], plan["routing_threshold"]] = weight
        self.assertLess(actual_weights["otmass", 0.3678], actual_weights["otmass", 0.9876])
        self.assertGreater(actual_weights["entropy", 0.1], actual_weights["entropy", 0.5])

    def test_token_routing_uses_correct_threshold_direction(self):
        for preset, low, high, low_weight, high_weight in (
            ("otmass", 0.0, 1.0, 0.0, 1.0),
            ("entropy", 0.0, 10.0, 1.0, 0.0),
        ):
            with self.subTest(preset=preset):
                for threshold, expected in ((low, low_weight), (high, high_weight)):
                    plan = make_plan(preset, 256, 1.0, 42, threshold)
                    self.assertEqual(run_smoke(plan)["effective_kd_weight"], expected)
        tied = run_smoke(make_plan("entropy", 1, 1.0, 42, 0.0))
        self.assertEqual(tied["effective_kd_weight"], 0.0)

    def test_matched_controls_keep_kd_budget_and_change_routing(self):
        results = {
            preset: run_smoke(make_plan(preset, 256, 1.0, 42))
            for preset in ("lm", "kd", "domain", "matched-mix", "matched-random")
        }
        self.assertEqual(results["lm"]["effective_kd_weight"], 0)
        self.assertEqual(results["kd"]["effective_kd_weight"], 1)
        for preset in ("domain", "matched-mix", "matched-random"):
            self.assertAlmostEqual(results[preset]["effective_kd_weight"], 0.33)
        self.assertEqual(results["domain"]["effective_kd_weight_by_domain"], {
            "general": 1.0, "math": 0.0, "code": 0.0,
        })
        for weight in results["matched-mix"]["effective_kd_weight_by_domain"].values():
            self.assertAlmostEqual(weight, 0.33)
        self.assertGreater(results["matched-random"]["effective_kd_weight_by_domain"]["math"], 0)
        self.assertNotEqual(results["lm"]["validation_ntp_ce"], results["kd"]["validation_ntp_ce"])

    def test_temperature_preserves_coverage(self):
        base = run_smoke(make_plan("kd", 256, 1.0, 42))
        colder = run_smoke(make_plan("kd", 256, 0.25, 42))
        narrower = run_smoke(make_plan("kd", 1, 1.0, 42))
        self.assertEqual(base["teacher_coverage_at_k"], colder["teacher_coverage_at_k"])
        self.assertNotEqual(base["teacher_observed_token_mass"], colder["teacher_observed_token_mass"])
        self.assertLess(narrower["teacher_coverage_at_k"], base["teacher_coverage_at_k"])


if __name__ == "__main__":
    unittest.main()
