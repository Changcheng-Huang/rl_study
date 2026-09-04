from __future__ import annotations

import unittest

from web.algorithm_registry import (
    get_experiment_design_preset,
    recommend_experiment_design_preset,
    validate_experiment_design,
)


class ExperimentDesignTests(unittest.TestCase):
    def test_frozen_lake_and_cliff_walking_are_matched(self) -> None:
        self.assertEqual(
            recommend_experiment_design_preset(
                name="Q-Learning",
                supported_environments=["FrozenLake-style tabular environment"],
            ),
            "frozen-lake-4x4-v1",
        )
        self.assertEqual(
            recommend_experiment_design_preset(
                name="Temporal-difference control",
                supported_environments=["CliffWalking"],
            ),
            "cliff-walking-4x12-v1",
        )

    def test_unrelated_algorithm_has_no_automatic_grid_recommendation(self) -> None:
        self.assertIsNone(
            recommend_experiment_design_preset(
                name="Policy Gradient",
                supported_environments=["Continuous control"],
            )
        )

    def test_preset_is_validated_and_returned_as_an_independent_copy(self) -> None:
        first = get_experiment_design_preset("frozen-lake-4x4-v1")
        second = get_experiment_design_preset("frozen-lake-4x4-v1")
        first["environment_map"]["layout"][0] = "XXXX"

        validated = validate_experiment_design(second)
        self.assertEqual(validated["environment_map"]["layout"][0], "SFFF")

    def test_unknown_platform_preset_is_rejected(self) -> None:
        design = get_experiment_design_preset("frozen-lake-4x4-v1")
        design["provenance"]["preset_id"] = "unknown"
        with self.assertRaisesRegex(ValueError, "preset is unknown"):
            validate_experiment_design(design)


if __name__ == "__main__":
    unittest.main()
