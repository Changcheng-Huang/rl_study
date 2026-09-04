from __future__ import annotations

import unittest
from pathlib import Path

from web.algorithm_registry import (
    AlgorithmManifest,
    ExperimentReporter,
    InstalledAlgorithm,
)
from web.algorithm_registry.runtime import (
    load_isolated_spec,
    run_isolated_experiment,
)


class QLearningGridworldExampleTests(unittest.TestCase):
    def test_reference_experiment_matches_the_grid_presentation_contract(self) -> None:
        root = (
            Path(__file__).resolve().parents[1]
            / "algorithm_packages"
            / "examples"
            / "q_learning_gridworld"
        )
        manifest = AlgorithmManifest(
            schema_version=2,
            algorithm_id="q-learning-gridworld-reference",
            name="Q-Learning GridWorld Reference",
            version="1.0.0",
            summary="Engineering acceptance reference.",
            category="model-free-control",
            theory_file="theory.md",
            experiment={"module": "experiment.py", "requirements": []},
        )
        algorithm = InstalledAlgorithm(manifest=manifest, path=root)

        spec = load_isolated_spec(algorithm, timeout_seconds=10)
        defaults = {
            name: definition["default"]
            for name, definition in spec["parameters"].items()
        }
        defaults["episodes"] = 1200
        reporter = ExperimentReporter()
        result = run_isolated_experiment(
            algorithm,
            defaults,
            reporter,
            spec=spec,
            timeout_seconds=10,
        )

        self.assertEqual(
            spec["presentation"]["environment_map"]["layout"],
            ["SFFF", "FHFH", "FFFH", "HFFG"],
        )
        self.assertIn("Goal", spec["presentation"]["task"]["mission"])
        self.assertEqual(len(result["views"]["policy_grid"]["state_values"]), 16)
        self.assertEqual(len(result["views"]["policy_grid"]["best_actions"]), 16)
        self.assertGreater(result["summary"]["final_success_rate"], 0.5)
        self.assertTrue(reporter.progress_events)
        self.assertTrue(reporter.metric_events)


if __name__ == "__main__":
    unittest.main()
