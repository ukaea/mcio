"""
unittest: recursive splitting of solids - skipping planar faces that cannot
split a solid must not change the split solids produced.
"""

import unittest

from OCP.gp import gp_Pnt, gp_Ax2, gp_Dir, gp_Trsf, gp_Ax1
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCP.Geom import Geom_Plane

from mcio.decomposeparams import DecompositionParameters
from mcio.ocpbooleans import _recursively_split_ocp_solid, _solid_behind_planar_face
from mcio.ocpfilereadwrite import extract_ocp_shape_type_from_shape, _get_ocp_surface_from_ocp_face
from mcio.ocpproperties import _volume_of_ocp_solids


def _solid(shape):
    """Return the single solid of a shape."""
    solids = extract_ocp_shape_type_from_shape(shape, TopAbs_SOLID, cast_from_shape=True)
    assert len(solids) == 1
    return solids[0]


def _faces(solid):
    return extract_ocp_shape_type_from_shape(solid, TopAbs_FACE, cast_from_shape=True)


def _box(x, y, z, dx, dy, dz):
    return BRepPrimAPI_MakeBox(gp_Pnt(x, y, z), dx, dy, dz).Shape()


def _rotated(shape):
    trsf = gp_Trsf()
    trsf.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 2, 3)), 0.7)
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


def _l_shape():
    """L-shaped (non-convex) solid: a 20x20x10 block minus a 10x10x10 corner."""
    return _solid(BRepAlgoAPI_Cut(_box(0, 0, 0, 20, 20, 10), _box(10, 10, 0, 10, 10, 10)).Shape())


class MCIOUnitTestSolidBehindPlanarFace(unittest.TestCase):
    """Detection of planar faces whose plane does not cut the solid."""

    def test_box_all_faces_behind(self):
        box = _solid(_rotated(_box(0, 0, 0, 10, 20, 30)))
        self.assertTrue(all(_solid_behind_planar_face(box, f) for f in _faces(box)))

    def test_l_shape_reentrant_faces_cut(self):
        solid = _solid(_rotated(_l_shape()))
        results = [_solid_behind_planar_face(solid, f) for f in _faces(solid)]
        # the two re-entrant faces' planes cut the solid; the other six do not
        self.assertEqual(results.count(False), 2)
        self.assertEqual(results.count(True), 6)

    def test_cylinder_caps_behind_side_not_planar(self):
        cyl = _solid(BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(1, 2, 3), gp_Dir(1, 1, 0)), 5.0, 30.0).Shape())
        for face in _faces(cyl):
            planar = isinstance(_get_ocp_surface_from_ocp_face(face), Geom_Plane)
            self.assertEqual(_solid_behind_planar_face(cyl, face), planar)

    def test_cavity_faces_cut(self):
        # box with an internal cavity: cavity walls (reversed faces) cut the solid
        solid = _solid(BRepAlgoAPI_Cut(_box(0, 0, 0, 30, 30, 30), _box(10, 10, 10, 10, 10, 10)).Shape())
        results = [_solid_behind_planar_face(solid, f) for f in _faces(solid)]
        self.assertEqual(results.count(True), 6)
        self.assertEqual(results.count(False), 6)


class MCIOUnitTestRecursiveSplit(unittest.TestCase):
    """Skipping non-cutting planes must give the same split solids."""

    def _split_volumes(self, solid, skip):
        params = DecompositionParameters()
        params.split_skip_non_cutting_planes = skip
        return sorted(round(_volume_of_ocp_solids([s]), 6) for s in _recursively_split_ocp_solid(solid, params))

    def _check(self, solid, n_expected):
        with_skip = self._split_volumes(solid, True)
        without_skip = self._split_volumes(solid, False)
        self.assertEqual(len(with_skip), n_expected)
        self.assertEqual(with_skip, without_skip)
        self.assertAlmostEqual(sum(with_skip), _volume_of_ocp_solids([solid]), places=4)

    def test_convex_box_not_split(self):
        self._check(_solid(_rotated(_box(0, 0, 0, 10, 20, 30))), 1)

    def test_l_shape(self):
        self._check(_solid(_rotated(_l_shape())), 2)

    def test_box_with_cavity(self):
        self._check(_solid(BRepAlgoAPI_Cut(_box(0, 0, 0, 30, 30, 30), _box(10, 10, 10, 10, 10, 10)).Shape()), 6)

    def test_plate_with_hole(self):
        # the hole is handled by auxiliary surfaces later; splitting leaves one solid
        plate = _box(0, 0, 0, 40, 40, 5)
        hole = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(20, 20, -1), gp_Dir(0, 0, 1)), 6.0, 7.0).Shape()
        solid = _solid(BRepAlgoAPI_Cut(plate, hole).Shape())
        with_skip = self._split_volumes(solid, True)
        self.assertEqual(with_skip, self._split_volumes(solid, False))
        self.assertEqual(len(with_skip), 1)
