"""unittest: the parsers and checks of the model suite (scripts/model_suite).

These use hand written MCTAL and MCNP output text, laid out as described in the
MCNP manual, not real MCNP output.
"""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "model_suite"),
)

import model_suite  # noqa: E402

MCNP_INPUT = """title
C ====== TALLIES
F14:N    1 2 3
FC14     stochastic_volume_calculation
NPS      1e+06
"""

MCTAL = """mcnp   6   03/01/2026 10:00:00        1
title
ntal       1
    14
dump no.  1 nps 1000000 ctm 12.5
tally       14    1    0
f           3
         1         2         3
d           1
u           1
s           1
m           1
c           1
e           1
t           1
vals
 1.00020E+00 1.0000E-03 9.99500E-01 2.0000E-03 1.10000E+00 1.0000E-03
tfc 1 1 1 1 1 1 1 1
"""


class MCIOUnitTestStochasticVolume(unittest.TestCase):
    def test_find_tally_number(self):
        self.assertEqual(model_suite.find_stochastic_tally_number(MCNP_INPUT), 14)
        self.assertIsNone(model_suite.find_stochastic_tally_number("F4:N 1"))

    def test_parse_mctal(self):
        cells, values, errors, nps = model_suite.parse_mctal_tally(MCTAL, 14)
        self.assertEqual(cells, [1, 2, 3])
        self.assertEqual(values, [1.0002, 0.9995, 1.1])
        self.assertEqual(errors, [0.001, 0.002, 0.001])
        self.assertEqual(nps, 1000000)

    def test_parse_mctal_uses_last_bin_per_cell(self):
        text = MCTAL.replace(
            " 1.00020E+00 1.0000E-03 9.99500E-01 2.0000E-03 1.10000E+00 1.0000E-03",
            " 0.5 0.1 1.0 0.001 0.5 0.1 0.9 0.002 0.5 0.1 1.2 0.003",
        )
        cells, values, errors, _ = model_suite.parse_mctal_tally(text, 14)
        self.assertEqual(values, [1.0, 0.9, 1.2])
        self.assertEqual(errors, [0.001, 0.002, 0.003])

    def test_parse_mctal_errors(self):
        with self.assertRaises(ValueError):
            model_suite.parse_mctal_tally(MCTAL, 24)
        with self.assertRaises(ValueError):
            model_suite.parse_mctal_tally(MCTAL.replace("1.0000E-03\ntfc", "\ntfc"), 14)

    def test_check_flags_deviating_cell_only(self):
        cells, values, errors, nps = model_suite.parse_mctal_tally(MCTAL, 14)
        result = model_suite.check_stochastic_volumes(cells, values, errors, nps)
        self.assertEqual(result.n_cells, 3)
        self.assertEqual(result.n_flagged, 1)
        self.assertEqual(result.worst[0]["cell"], 3)
        self.assertEqual(result.n_unconverged, 0)

    def test_small_deviation_not_flagged_and_unconverged_counted(self):
        result = model_suite.check_stochastic_volumes(
            [1, 2, 3], [1.001, 1.0, 0.0], [1e-6, 0.5, 0.0]
        )
        # cell 1 is 1000 sigma out but only 0.1% off, below min_deviation
        self.assertEqual(result.n_flagged, 0)
        # cell 2 has a large error and cell 3 no score
        self.assertEqual(result.n_unconverged, 2)


class MCIOUnitTestLostParticles(unittest.TestCase):
    def test_counts_records(self):
        text = (
            " x,y,z coordinates:  1.0E+00  2.0E+00  3.0E+00\n"
            " u,v,w direction cosines:  0.0E+00  0.0E+00  1.0E+00\n" * 2
        )
        result = model_suite.count_lost_particles(text)
        self.assertEqual(result["records"], 2)
        self.assertFalse(result["terminated_on_lost"])

    def test_terminated(self):
        text = " run terminated when   10 particle histories were lost.\n"
        self.assertTrue(model_suite.count_lost_particles(text)["terminated_on_lost"])

    def test_none_lost(self):
        self.assertEqual(model_suite.count_lost_particles("all good")["records"], 0)


class MCIOUnitTestStatsAndVerdict(unittest.TestCase):
    def test_solid_time_statistics(self):
        stats = model_suite.solid_time_statistics([1.0, 2.0, 3.0, 4.0, 100.0])
        self.assertEqual(stats["n"], 5)
        self.assertEqual(stats["median"], 3.0)
        self.assertEqual(stats["max"], 100.0)
        self.assertEqual(stats["p95"], 100.0)
        self.assertEqual(model_suite.solid_time_statistics([]), {})

    def test_verdicts(self):
        ok = {
            "decompose_ok": True,
            "n_failed_solids": 0,
            "stochastic_volume": {"status": "ok", "n_flagged": 0},
            "lost_particles": {
                "status": "ok",
                "records": 0,
                "terminated_on_lost": False,
            },
        }
        self.assertEqual(model_suite.model_verdict(ok), "PASS")
        self.assertEqual(
            model_suite.model_verdict({**ok, "decompose_ok": False}), "FAIL"
        )
        self.assertEqual(
            model_suite.model_verdict({**ok, "n_failed_solids": 2}), "FAIL"
        )
        self.assertEqual(
            model_suite.model_verdict(
                {**ok, "stochastic_volume": {"status": "ok", "n_flagged": 3}}
            ),
            "WARN",
        )
        self.assertEqual(
            model_suite.model_verdict(
                {
                    **ok,
                    "lost_particles": {
                        "status": "ok",
                        "records": 1,
                        "terminated_on_lost": False,
                    },
                }
            ),
            "FAIL",
        )
        self.assertEqual(
            model_suite.model_verdict(
                {**ok, "stochastic_volume": {"status": "no mctal file"}}
            ),
            "NOT RUN",
        )

    def test_report_builds(self):
        record = {
            "name": "m",
            "input": os.path.join("x", "m", "model_mcnp.i"),
            "decompose_ok": True,
            "wall_time": 10.0,
            "n_solids": 3,
            "n_failed_solids": 0,
            "stage_times": {"decompose": 5.0},
            "solid_times": model_suite.solid_time_statistics([1.0, 2.0]),
        }
        text = model_suite.build_report([record], "x")
        self.assertIn("| m |", text)
        self.assertIn("Run time statistics", text)


if __name__ == "__main__":
    unittest.main()
