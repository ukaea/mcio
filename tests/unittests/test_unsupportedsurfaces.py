"""unittest: a solid with faces which cannot be converted fails on its own, and does
not stop the analysis or the decomposition of other solids."""

import unittest

from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeWire,
)
from OCP.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeRevol,
    BRepPrimAPI_MakeTorus,
)
from OCP.GeomAPI import GeomAPI_PointsToBSpline
from OCP.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt
from OCP.TColgp import TColgp_Array1OfPnt

from mcio.decompose import Analysis
from mcio.decomposeparams import DecompositionParameters
from mcio.decomposesolid import (
    DEGENERATE_TORUS,
    DecompositionSolid,
    UnsupportedSurfaceError,
    _unsupported_surface_types,
)


def _revolved_spline_solid():
    """A solid made by revolving a face with a curved (spline) side about the z axis."""
    points = TColgp_Array1OfPnt(1, 3)
    points.SetValue(1, gp_Pnt(1.0, 0.0, 0.0))
    points.SetValue(2, gp_Pnt(1.6, 0.0, 1.0))
    points.SetValue(3, gp_Pnt(1.0, 0.0, 2.0))
    curve = GeomAPI_PointsToBSpline(points).Curve()
    wire = BRepBuilderAPI_MakeWire()
    wire.Add(BRepBuilderAPI_MakeEdge(curve).Edge())
    wire.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(1, 0, 2), gp_Pnt(0, 0, 2)).Edge())
    wire.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(0, 0, 2), gp_Pnt(0, 0, 0)).Edge())
    wire.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(0, 0, 0), gp_Pnt(1, 0, 0)).Edge())
    face = BRepBuilderAPI_MakeFace(wire.Wire()).Face()
    return BRepPrimAPI_MakeRevol(face, gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))).Shape()


def _box_solid() -> DecompositionSolid:
    return DecompositionSolid(
        BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Solid(), DecompositionParameters()
    )


def _degenerate_torus_solid() -> DecompositionSolid:
    # a minor radius larger than the major radius
    axes = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    return DecompositionSolid(
        BRepPrimAPI_MakeTorus(axes, 1.0, 2.0).Shape(), DecompositionParameters()
    )


def _ordinary_torus_solid() -> DecompositionSolid:
    axes = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    return DecompositionSolid(
        BRepPrimAPI_MakeTorus(axes, 3.0, 1.0).Shape(), DecompositionParameters()
    )


def _unsupported_solid() -> DecompositionSolid:
    return DecompositionSolid(_revolved_spline_solid(), DecompositionParameters())


class MCIOUnitTestUnsupportedSurfaces(unittest.TestCase):
    def test_box_has_none(self):
        self.assertEqual(_unsupported_surface_types(_box_solid()._solid_original), {})

    def test_revolved_spline_is_unsupported(self):
        found = _unsupported_surface_types(_unsupported_solid()._solid_original)
        self.assertEqual(len(found), 1)
        self.assertEqual(sum(found.values()), 1)

    def test_degenerate_torus_is_counted_by_name(self):
        found = _unsupported_surface_types(_degenerate_torus_solid()._solid_original)
        self.assertEqual(list(found), [DEGENERATE_TORUS])

    def test_ordinary_torus_is_supported(self):
        self.assertEqual(
            _unsupported_surface_types(_ordinary_torus_solid()._solid_original), {}
        )

    def test_degenerate_torus_solid_fails_without_retries(self):
        solid = _degenerate_torus_solid()
        with self.assertRaises(UnsupportedSurfaceError) as context:
            solid.run()
        message = str(context.exception)
        self.assertIn("degenerate tori", message)
        self.assertNotIn("unsupported surface types", message)
        self.assertEqual(solid.attempts_used, 1)

    def test_solid_fails_without_retries(self):
        solid = _unsupported_solid()
        self.assertGreater(solid.decom_params.n_attempts, 1)
        with self.assertRaises(UnsupportedSurfaceError) as context:
            solid.run()
        self.assertIn("unsupported surface types", str(context.exception))
        self.assertNotIn("degenerate tori", str(context.exception))
        self.assertIn("simplified in the CAD model", str(context.exception))
        # one attempt, not one for each of the values of the parameters
        self.assertEqual(solid.attempts_used, 1)
        self.assertIs(solid.decom_params.attempt, 0)

    def test_supported_solid_is_not_stopped(self):
        solid = _box_solid()
        solid.run()
        self.assertEqual(solid.attempts_used, 1)
        self.assertIsNotNone(solid.mcio_cell)

    def test_analysis_records_instead_of_raising(self):
        good, bad = _box_solid(), _unsupported_solid()
        analysis = Analysis([good, bad])
        self.assertEqual(list(analysis.UNSUPPORTED_SURFACES), [bad])
        self.assertEqual(sum(analysis.UNSUPPORTED_SURFACES[bad].values()), 1)
        self.assertEqual(analysis.NUMBER_OF_BODIES, 2)
        self.assertIn("bodies with unsupported surfaces", analysis.summary)

    def test_analysis_counts_degenerate_tori_per_body(self):
        good, bad = _ordinary_torus_solid(), _degenerate_torus_solid()
        analysis = Analysis([good, bad])
        self.assertEqual(analysis.DEGENERATE_TORI_COUNT, 1)
        self.assertEqual(analysis.DEGENERATE_TORI_PER_BODY, {bad: 1})


if __name__ == "__main__":
    unittest.main()
