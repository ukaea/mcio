"""
regression: side of the surface chosen when converting OCP faces to mcio half
spaces, for solids built directly in OCP / cadquery (not read from STEP).

Mirrored solids, and some cadquery operations such as revolve, produce faces
whose surfaces have left handed axes, which reverses the surface normal. For
every face, a point just inside the solid must lie in the half space and a
point just outside must not. Decomposed cells must also match the solid at
random points.
"""

import collections
import unittest

import numpy as np
import cadquery as cq

from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeCone,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakeSphere,
    BRepPrimAPI_MakeTorus
)
from OCP.BRepTools import BRepTools
from OCP.Bnd import Bnd_Box
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID, TopAbs_IN, TopAbs_OUT, TopAbs_REVERSED
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf

from mcio.decompose import Decomposer
from mcio.decomposeparams import DecompositionParameters
from mcio.ocpfilereadwrite import extract_ocp_shape_type_from_shape
from mcio.ocptomcio import get_mcio_halfspace_from_ocp_face


from unitutils import TemporaryWorkingDirectory


# run in a temporary working directory, as the decomposer writes files to the working directory
_working_directory = TemporaryWorkingDirectory()


def setUpModule():
    _working_directory.enter()


def tearDownModule():
    _working_directory.exit()


def _mirror(shape, normal=(1, 0, 0)):
    trsf = gp_Trsf()
    trsf.SetMirror(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(*normal)))
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


def _solid(shape):
    return extract_ocp_shape_type_from_shape(shape, TopAbs_SOLID, cast_from_shape=True)[0]


def _faces(shape):
    return extract_ocp_shape_type_from_shape(shape, TopAbs_FACE, cast_from_shape=True)


def _box_with_hole():
    box = BRepPrimAPI_MakeBox(gp_Pnt(-10, -10, 0), 20, 20, 10).Shape()
    hole = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -1), gp_Dir(0, 0, 1)), 4, 12).Shape()
    return _solid(BRepAlgoAPI_Cut(box, hole).Shape())


def _test_solids():
    return {
        'box': BRepPrimAPI_MakeBox(gp_Pnt(1, 2, 3), 10, 20, 30).Shape(),
        'box with hole': _box_with_hole(),
        'sphere': BRepPrimAPI_MakeSphere(gp_Pnt(1, 2, 3), 7).Shape(),
        'cone': BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(1, 2, 3), gp_Dir(0, 0, 1)), 6, 2, 10).Shape(),
        'tilted cone': BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(1, 2, 3), gp_Dir(1, 1, 1)), 6, 2, 10).Shape(),
        'torus': BRepPrimAPI_MakeTorus(gp_Ax2(gp_Pnt(1, 2, 3), gp_Dir(0, 0, 1)), 20, 4).Shape(),
        'cq spherical shell': cq.Workplane('XY').sphere(10).cut(cq.Workplane('XY').sphere(5)).translate((3, 4, 5)).val().wrapped,
    }


class MCIORegressionTestFaceOrientation(unittest.TestCase):
    """Half spaces from faces must contain the material side of the face."""

    @classmethod
    def setUpClass(cls):
        solids = _test_solids()
        cls.solids = dict(solids)
        for name, solid in solids.items():
            cls.solids[f'mirrored {name}'] = _mirror(solid)
        cls.solids['mirrored in z cone'] = _mirror(solids['cone'], (0, 0, 1))
        # the flat end of a revolved cone has a left handed plane, without any mirroring
        cls.solids['cq revolved cone'] = (
            cq.Workplane('XZ').polyline([(0, 0), (6, 0), (2, 10), (0, 10)]).close()
            .revolve(360, (0, 0, 0), (0, 1, 0)).val().wrapped
        )

    def _probe_points(self, solid, face, eps=1e-3):
        """Points just inside and outside the solid either side of the face."""
        adaptor = BRepAdaptor_Surface(face)
        u_min, u_max, v_min, v_max = BRepTools.UVBounds_s(face)
        points = []
        for fu in (0.3, 0.5, 0.7):
            for fv in (0.3, 0.5, 0.7):
                props = BRepLProp_SLProps(adaptor, u_min + fu * (u_max - u_min), v_min + fv * (v_max - v_min), 1, 1e-7)
                if not props.IsNormalDefined():
                    continue
                p, n = props.Value(), props.Normal()
                if face.Orientation() == TopAbs_REVERSED:
                    n.Reverse()
                inside = (p.X() - eps * n.X(), p.Y() - eps * n.Y(), p.Z() - eps * n.Z())
                outside = (p.X() + eps * n.X(), p.Y() + eps * n.Y(), p.Z() + eps * n.Z())
                # keep points the solid itself agrees with (e.g. not past the face boundary)
                if (BRepClass3d_SolidClassifier(solid, gp_Pnt(*inside), 1e-7).State() == TopAbs_IN and
                        BRepClass3d_SolidClassifier(solid, gp_Pnt(*outside), 1e-7).State() == TopAbs_OUT):
                    points.append((inside, outside))
        return points

    def test_face_half_spaces(self):
        params = DecompositionParameters()
        for name, solid in self.solids.items():
            for i, face in enumerate(_faces(solid)):
                with self.subTest(solid=name, face=i):
                    half_space = get_mcio_halfspace_from_ocp_face(face, params, surfaces_generated=collections.defaultdict(dict))
                    region = half_space.openmc_region
                    points = self._probe_points(solid, face)
                    self.assertTrue(points, 'no probe points found')
                    for inside, outside in points:
                        self.assertIn(inside, region)
                        self.assertNotIn(outside, region)

    def test_decomposed_cells_match_solid(self):
        rng = np.random.default_rng(0)
        for name in ('mirrored box with hole', 'mirrored cq spherical shell', 'mirrored torus',
                     'mirrored in z cone', 'mirrored tilted cone', 'cq revolved cone'):
            with self.subTest(solid=name):
                solid = self.solids[name]
                decomposer = Decomposer()
                decomposer.add_solid(solid, 'solid')
                decomposer.run(cores=1)
                region = decomposer.decomposition_solids[0].mcio_cell.volume.openmc_region

                bounds = Bnd_Box()
                BRepBndLib.Add_s(solid, bounds)
                x0, y0, z0, x1, y1, z1 = bounds.Get()
                wrong = 0
                for point in rng.uniform([x0, y0, z0], [x1, y1, z1], (500, 3)):
                    in_solid = BRepClass3d_SolidClassifier(solid, gp_Pnt(*point), 1e-7).State() == TopAbs_IN
                    if in_solid != (tuple(point) in region):
                        wrong += 1
                self.assertEqual(wrong, 0)
