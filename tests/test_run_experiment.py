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
