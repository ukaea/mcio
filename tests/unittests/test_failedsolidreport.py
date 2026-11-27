"""unittest: the report on a solid which fails to decompose (mcio.failedsolidreport)."""

import os
import tempfile
import unittest
from unittest import mock

from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCone
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

from mcio import failedsolidreport as report
from mcio.decomposeparams import DecompositionParameters
from mcio.decomposesolid import DecompositionSolid


def _solid(shape) -> DecompositionSolid:
    return DecompositionSolid(
        shape,
        DecompositionParameters(),
        structure_original=["assembly", "part"],
        name="cell_7",
    )


def _box() -> DecompositionSolid:
    return _solid(BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Solid())


def _flat_cone() -> DecompositionSolid:
    # a cone 1e-5 high with a base radius of 1: a semi-angle of 89.9994 degrees
    axes = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    return _solid(BRepPrimAPI_MakeCone(axes, 0.0, 1.0, 1e-5).Solid())


ATTEMPT = {
    "attempt": 1,
    "values": {"bool_common_fuzzy_value": 1e-6},
    "error": "boom | with a bar\nand a new line",
    "time": 1.5,
}


class MCIOUnitTestFailureText(unittest.TestCase):
    def test_failure_type(self):
        self.assertEqual(
            report.failure_type(
                "stopped after exceeding the time limit of 600 seconds"
            ),
            "timeout",
        )
        self.assertEqual(
            report.failure_type("worker process stopped unexpectedly"), "worker stopped"
        )
        self.assertEqual(report.failure_type("something else"), "error")

    def test_possible_causes(self):
        causes = report.possible_causes(
            "auxiliary halfspace cut and common failed (volume not conserved)"
        )
        self.assertEqual(len(causes), 1)
        self.assertIn("do not add up", causes[0])
        self.assertEqual(report.possible_causes("an unknown error"), [])

    def test_debug_files_in(self):
        text = "failed - see analysis/a/one.stp\n and see analysis\\b\\two.step. Again see analysis/a/one.stp"
        self.assertEqual(
            report.debug_files_in(text),
            ["analysis/a/one.stp", "analysis\\b\\two.step"],
        )


class MCIOUnitTestGeometrySummary(unittest.TestCase):
    def test_box(self):
        summary = report.geometry_summary(_box()._solid_original)
        self.assertAlmostEqual(summary["volume"], 6.0)
        self.assertAlmostEqual(summary["surface_area"], 22.0)
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["n_faces"], 6)
        self.assertEqual(summary["face_counts"], {"Geom_Plane": 6})
        self.assertEqual(summary["near_planar_cones"], [])
        self.assertEqual(summary["unsupported_surfaces"], {})
        self.assertEqual(summary["bounding_box"]["max"][2] >= 3.0, True)

    def test_near_planar_cone_is_found(self):
        summary = report.geometry_summary(_flat_cone()._solid_original)
        self.assertEqual(len(summary["near_planar_cones"]), 1)
        self.assertGreater(summary["near_planar_cones"][0]["semi_angle_degrees"], 89.99)

    def test_one_failing_item_does_not_stop_the_others(self):
        with mock.patch.object(
            report, "_surface_area_of_ocp_solids", side_effect=RuntimeError("no area")
        ):
            summary = report.geometry_summary(_box()._solid_original)
        self.assertIsNone(summary["surface_area"])
        self.assertAlmostEqual(summary["volume"], 6.0)
        self.assertIn("surface_area: no area", summary["errors"])


class MCIOUnitTestWriteReport(unittest.TestCase):
    def test_report_has_every_section(self):
        solid = _box()
        solid._attempts = [ATTEMPT]
        with tempfile.TemporaryDirectory() as folder:
            path = report.write_failed_solid_report(
                7,
                solid,
                "volume not conserved - see analysis/x/y.stp",
                "Traceback text",
                files={"traceback": "analysis/trackbacks/t.txt"},
                directory=folder,
            )
            self.assertEqual(path, os.path.join(folder, "solid_7_report.md"))
            with open(path, encoding="utf-8") as f:
                text = f.read()
        for heading in (
            "# Failed solid 7",
            "## Solid",
            "## Failure",
            "## Decomposition attempts",
            "## Geometry of the original solid",
            "## Suspicious features",
            "## Files",
            "## Versions",
            "## Traceback",
        ):
            self.assertIn(heading, text)
        self.assertIn("assembly / part", text)
        self.assertIn("valid (BRepCheck): True", text)
        self.assertIn("analysis/x/y.stp", text)
        self.assertIn("None found.", text)
        self.assertIn("Traceback text", text)
        # a | in an error does not break the table
        self.assertIn("boom / with a bar and a new line", text)

    def test_near_planar_cone_is_listed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = report.write_failed_solid_report(
                1, _flat_cone(), "x", directory=folder
            )
            with open(path, encoding="utf-8") as f:
                text = f.read()
        self.assertIn("within 0.1 degrees of flat", text)
        self.assertNotIn("None found.", text)

    def test_no_attempt_information(self):
        with tempfile.TemporaryDirectory() as folder:
            path = report.write_failed_solid_report(
                1,
                _box(),
                "stopped after exceeding the time limit of 600 seconds",
                directory=folder,
            )
            with open(path, encoding="utf-8") as f:
                text = f.read()
        self.assertIn("type: timeout", text)
        self.assertIn("No attempt information", text)

    def test_default_directory_comes_from_the_parameters(self):
        solid = _box()
        with tempfile.TemporaryDirectory() as folder:
            solid.decom_params.debug_failed_solid_report_directory = os.path.join(
                folder, "reports"
            )
            path = report.write_failed_solid_report(3, solid, "x")
            self.assertTrue(os.path.isfile(path))

    def test_writing_never_raises(self):
        with mock.patch.object(
            report, "failed_solid_report_data", side_effect=RuntimeError("broken")
        ):
            with self.assertLogs("mcio", "WARNING") as logs:
                result = report.write_failed_solid_report(
                    1, _box(), "x", directory=tempfile.gettempdir()
                )
        self.assertIsNone(result)
        self.assertTrue(any("could not write the report" in m for m in logs.output))


if __name__ == "__main__":
    unittest.main()
