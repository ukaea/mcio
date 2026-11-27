"""unittest: decomposition parameters with one value per attempt, and the retry of a
solid which fails to decompose (DecompositionSolid.run).

The decomposition itself is replaced with a function which fails on request, so no
CAD models are involved.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

from mcio.decomposeparams import DecompositionParameters
from mcio.decomposesolid import DecompositionSolid

PER_ATTEMPT = [
    "bool_common_fuzzy_values",
    "bool_cutter_fuzzy_values",
    "bool_splitter_fuzzy_values",
    "bool_degenerate_min_area_to_vol_ratio_values",
    "bool_degenerate_min_vol_fraction_of_original_values",
    "bool_degenerate_min_ab_vol_values",
]


class MCIOUnitTestPerAttemptParameters(unittest.TestCase):
    def test_defaults(self):
        params = DecompositionParameters()
        self.assertEqual(params.bool_common_fuzzy_values, [1e-6, 1e-5, 1e-4])
        self.assertEqual(
            params.bool_degenerate_min_area_to_vol_ratio_values, [1e-4, 1e-6, 1e-8]
        )
        self.assertEqual(params.n_attempts, 3)
        self.assertEqual(params.attempt, 0)

    def test_defaults_are_not_shared_between_instances(self):
        first, second = DecompositionParameters(), DecompositionParameters()
        first.bool_common_fuzzy_values.append(1.0)
        self.assertEqual(second.bool_common_fuzzy_values, [1e-6, 1e-5, 1e-4])

    def test_single_value_names_follow_the_attempt(self):
        params = DecompositionParameters()
        self.assertEqual(params.bool_common_fuzzy_value, 1e-6)
        self.assertEqual(params.for_attempt(1).bool_common_fuzzy_value, 1e-5)
        self.assertEqual(params.for_attempt(2).bool_cutter_fuzzy_value, 1e-4)
        self.assertEqual(params.for_attempt(2).bool_degenerate_min_ab_vol, 1e-14)

    def test_for_attempt_does_not_change_the_original(self):
        params = DecompositionParameters()
        copy = params.for_attempt(2)
        self.assertEqual(params.attempt, 0)
        self.assertEqual(params.bool_common_fuzzy_value, 1e-6)
        self.assertEqual(copy.attempt, 2)
        self.assertIsNot(copy.bool_common_fuzzy_values, params.bool_common_fuzzy_values)

    def test_short_list_repeats_its_last_value(self):
        params = DecompositionParameters(bool_common_fuzzy_values=[1e-6, 1e-3])
        self.assertEqual(params.n_attempts, 3)  # set by the longest list
        self.assertEqual(params.for_attempt(2).bool_common_fuzzy_value, 1e-3)
        self.assertEqual(params.for_attempt(9).bool_common_fuzzy_value, 1e-3)

    def test_single_attempt_when_every_list_has_one_value(self):
        params = DecompositionParameters()
        for name in PER_ATTEMPT:
            setattr(params, name, [1e-6])
        self.assertEqual(params.n_attempts, 1)

    def test_attempt_values(self):
        values = DecompositionParameters().for_attempt(1).attempt_values()
        self.assertEqual(values["bool_common_fuzzy_value"], 1e-5)
        self.assertEqual(values["bool_degenerate_min_ab_vol"], 1e-12)
        self.assertEqual(len(values), 6)

    def test_a_single_number_is_a_list_of_one(self):
        params = DecompositionParameters(bool_cutter_fuzzy_values=1e-5)
        self.assertEqual(params.bool_cutter_fuzzy_values, [1e-5])

    def test_assigning_old_single_value_name_sets_one_attempt(self):
        params = DecompositionParameters()
        params.bool_common_fuzzy_value = 1e-5
        self.assertEqual(params.bool_common_fuzzy_values, [1e-5])

    def test_invalid_values_raise(self):
        for bad in ([], [-1e-6], [float("nan")], ["a"], [True]):
            with self.assertRaises(ValueError, msg=str(bad)):
                DecompositionParameters(bool_common_fuzzy_values=bad)

    def test_zero_is_allowed(self):
        params = DecompositionParameters(bool_degenerate_min_ab_vol_values=[1e-10, 0.0])
        self.assertEqual(params.bool_degenerate_min_ab_vol_values, [1e-10, 0.0])

    def test_json_round_trip(self):
        params = DecompositionParameters(bool_common_fuzzy_values=[1e-7, 1e-3])
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "params.json")
            params.write(path)
            read = DecompositionParameters.read(path)
        self.assertEqual(read.bool_common_fuzzy_values, [1e-7, 1e-3])
        self.assertEqual(read.bool_degenerate_min_ab_vol_values, [1e-10, 1e-12, 1e-14])

    def test_reading_old_single_value_names(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "old.json")
            with open(path, "w") as f:
                json.dump(
                    {
                        "bool_common_fuzzy_value": 1e-5,
                        "bool_degenerate_min_ab_vol": 1e-12,
                    },
                    f,
                )
            read = DecompositionParameters.read(path)
        self.assertEqual(read.bool_common_fuzzy_values, [1e-5])
        self.assertEqual(read.bool_degenerate_min_ab_vol_values, [1e-12])


def _box_solid() -> DecompositionSolid:
    solid = BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Solid()
    return DecompositionSolid(solid, DecompositionParameters(), name="box")


class MCIOUnitTestDecompositionAttempts(unittest.TestCase):
    def run_with(self, solid, outcomes, seen=None):
        """Run the solid, with each attempt of ``_decompose`` giving the next outcome.

        An outcome which is an exception is raised, anything else is a success.
        """
        outcomes = list(outcomes)

        def fake_decompose(self):
            if seen is not None:
                seen.append(self.decom_params.bool_common_fuzzy_value)
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                # leave results behind, as a failure part way through would
                self._split_solids = ["left over"]
                self._volume_original_solid = 1.0
                raise outcome
            self._split_solids = ["decomposed"]
            self._volume_original_solid = 1.0
            self._volume_decomposed_solids = 1.0
            return self

        with mock.patch.object(DecompositionSolid, "_decompose", fake_decompose):
            solid.run()

    def test_first_attempt_succeeds(self):
        solid = _box_solid()
        original_params = solid.decom_params
        seen = []
        self.run_with(solid, [None], seen)
        self.assertEqual(solid.attempts_used, 1)
        self.assertEqual(seen, [1e-6])
        self.assertIs(solid.decom_params, original_params)
        self.assertIsNone(solid.attempts[0]["error"])

    def test_retries_with_the_next_values(self):
        solid = _box_solid()
        original_params = solid.decom_params
        seen = []
        with self.assertLogs("mcio", "WARNING") as logs:
            self.run_with(
                solid, [ValueError("first"), RuntimeError("second"), None], seen
            )
        self.assertEqual(seen, [1e-6, 1e-5, 1e-4])
        self.assertEqual(solid.attempts_used, 3)
        self.assertEqual(
            [a["error"] for a in solid.attempts], ["first", "second", None]
        )
        self.assertEqual(solid.attempts[1]["values"]["bool_common_fuzzy_value"], 1e-5)
        self.assertIs(solid.decom_params, original_params)
        self.assertEqual(original_params.attempt, 0)
        self.assertTrue(any("decomposed on attempt 3 of 3" in m for m in logs.output))

    def test_results_of_a_failed_attempt_are_cleared(self):
        solid = _box_solid()
        self.run_with(solid, [ValueError("first"), None])
        self.assertEqual(solid._split_solids, ["decomposed"])

    def test_all_attempts_fail(self):
        solid = _box_solid()
        original_params = solid.decom_params
        with self.assertRaises(Exception) as context:
            with self.assertLogs("mcio", "WARNING"):
                self.run_with(
                    solid,
                    [ValueError("one"), ValueError("two"), KeyError("three")],
                )
        message = str(context.exception)
        self.assertIn("failed after 3 decomposition attempts", message)
        for text in ("one", "two", "three", "common_fuzzy_value=1e-05"):
            self.assertIn(text, message)
        self.assertIsInstance(context.exception.__cause__, KeyError)
        self.assertIs(solid.decom_params, original_params)
        self.assertEqual(solid._split_solids, [])

    def test_single_attempt_raises_the_original_exception(self):
        solid = _box_solid()
        for name in PER_ATTEMPT:
            setattr(solid.decom_params, name, [1e-6])
        with self.assertRaises(ValueError) as context:
            self.run_with(solid, [ValueError("only")])
        self.assertEqual(str(context.exception), "only")

    def test_warns_when_volume_changes(self):
        solid = _box_solid()

        def fake_decompose(self):
            self._volume_original_solid = 100.0
            self._volume_decomposed_solids = 100.5
            return self

        with mock.patch.object(DecompositionSolid, "_decompose", fake_decompose):
            with self.assertLogs("mcio", "WARNING") as logs:
                solid.run()
        self.assertAlmostEqual(solid.volume_mismatch, 5e-3)
        self.assertTrue(any("differs from the original" in m for m in logs.output))

    def test_no_warning_within_tolerance(self):
        solid = _box_solid()

        def fake_decompose(self):
            self._volume_original_solid = 100.0
            self._volume_decomposed_solids = 100.00001
            return self

        with mock.patch.object(DecompositionSolid, "_decompose", fake_decompose):
            with self.assertNoLogs("mcio", "WARNING"):
                solid.run()

    def test_volume_mismatch_none_before_decomposition(self):
        self.assertIsNone(_box_solid().volume_mismatch)


if __name__ == "__main__":
    unittest.main()
