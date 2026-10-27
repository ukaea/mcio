"""
regression: inversion of CSG half spaces into OCP solids.

Each half space is converted into an OCP solid clipped to a bounding box. The
volume of the inside (-) solid is checked against the analytic value, the two
sides must add up to the bounding box, and points are classified against the
solid and compared with the MCNP surface function.
"""

import math

import numpy as np
import unittest

from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_IN, TopAbs_OUT

import mcio.surface as S
from mcio.geometric import BoundingBox
from mcio.decomposeparams import DecompositionParameters
from mcio.ocpproperties import _volume_of_ocp_solids


from unitutils import CustomAssertions


def _torus_function(rho, axial, a, b, c):
    """MCNP torus function: a major radius, b axial and c radial semi-axis."""
    return axial ** 2 / b ** 2 + (rho - a) ** 2 / c ** 2 - 1.0


class MCIORegressionTestInversionHalfSpace(unittest.TestCase, CustomAssertions):
    def setUp(self):
        self._extent = 50.0
        self._box_volume = (2 * self._extent) ** 3
        self._bounding_box = BoundingBox(-self._extent, self._extent, -self._extent, self._extent, -self._extent, self._extent)
        self._decom_params = DecompositionParameters()
        self._decom_params.bounding_box_multiplier_half_space = 1.0
        self._tol = 1e-6

    def _solids(self, surface):
        neg = (-surface).ocp_solid(self._bounding_box, self._decom_params)
        pos = (+surface).ocp_solid(self._bounding_box, self._decom_params)
        return neg, pos

    def _check_volumes(self, surface, expected_inside):
        neg, pos = self._solids(surface)
        volume_neg = _volume_of_ocp_solids([neg])
        volume_pos = _volume_of_ocp_solids([pos])
        self.assertRelativeToleranceAlmostEqual(expected_inside, volume_neg, self._tol)
        self.assertRelativeToleranceAlmostEqual(self._box_volume, volume_neg + volume_pos, self._tol)
        return neg, pos

    def _check_points(self, solid, function, points, inside):
        """Classify points against the solid and compare with the sign of the surface function."""
        for point in points:
            value = function(*point)
            if abs(value) < 1e-3:
                continue
            state = BRepClass3d_SolidClassifier(solid, gp_Pnt(*point), 1e-7).State()
            expected = TopAbs_IN if ((value < 0) == inside) else TopAbs_OUT
            self.assertEqual(state, expected, f'point {point} (f = {value:.3e})')

    def _grid(self, n=9):
        values = [-self._extent + (i + 0.5) * 2 * self._extent / n for i in range(n)]
        return [(x, y, z) for x in values for y in values for z in values]

    # ---------------------------------------------------------------- planes

    def test_plane_halfspace(self):
        # plane through the box centre, normal (1, 1, 0)
        surface = S.Plane(a=1.0, b=1.0, c=0.0, d=0.0)
        neg, pos = self._check_volumes(surface, self._box_volume / 2.0)
        f = lambda x, y, z: x + y
        self._check_points(neg, f, self._grid(), inside=True)
        self._check_points(pos, f, self._grid(), inside=False)

    def test_plane_offset_halfspace(self):
        # x + y + z = 15 cuts a corner tetrahedron-like region, check by points only
        surface = S.Plane(a=1.0, b=1.0, c=1.0, d=15.0)
        neg, pos = self._solids(surface)
        f = lambda x, y, z: x + y + z - 15.0
        self._check_points(neg, f, self._grid(), inside=True)
        self._check_points(pos, f, self._grid(), inside=False)

    def test_axis_planes_halfspace(self):
        for surface, f in (
            (S.XPlane(x0=10.0), lambda x, y, z: x - 10.0),
            (S.YPlane(y0=-20.0), lambda x, y, z: y + 20.0),
            (S.ZPlane(z0=30.0), lambda x, y, z: z - 30.0),
        ):
            with self.subTest(surface=type(surface).__name__):
                offset = abs(f(0.0, 0.0, 0.0))
                fraction = (self._extent + (offset if f(0.0, 0.0, 0.0) < 0 else -offset)) / (2 * self._extent)
                neg, _ = self._check_volumes(surface, self._box_volume * fraction)
                self._check_points(neg, f, self._grid(), inside=True)

    # ---------------------------------------------------------------- cylinders

    def test_cylinder_halfspace(self):
        radius = 10.0
        for surface, f in (
            (S.CylinderParallelToXAxis(y0=5.0, z0=-3.0, radius=radius), lambda x, y, z: (y - 5) ** 2 + (z + 3) ** 2 - radius ** 2),
            (S.CylinderParallelToYAxis(x0=5.0, z0=-3.0, radius=radius), lambda x, y, z: (x - 5) ** 2 + (z + 3) ** 2 - radius ** 2),
            (S.CylinderParallelToZAxis(x0=5.0, y0=-3.0, radius=radius), lambda x, y, z: (x - 5) ** 2 + (y + 3) ** 2 - radius ** 2),
            (S.CylinderOnZAxis(radius=radius), lambda x, y, z: x ** 2 + y ** 2 - radius ** 2),
        ):
            with self.subTest(surface=type(surface).__name__):
                neg, pos = self._check_volumes(surface, math.pi * radius ** 2 * 2 * self._extent)
                self._check_points(neg, f, self._grid(11), inside=True)
                self._check_points(pos, f, self._grid(11), inside=False)

    def test_cylinder_on_x_axis_halfspace(self):
        self._check_volumes(S.CylinderOnXAxis(radius=10.0), math.pi * 100.0 * 2 * self._extent)

    def test_cylinder_on_y_axis_halfspace(self):
        self._check_volumes(S.CylinderOnYAxis(radius=10.0), math.pi * 100.0 * 2 * self._extent)

    # ---------------------------------------------------------------- spheres

    def test_sphere_halfspace(self):
        radius = 20.0
        for surface, centre in (
            (S.Sphere(x0=1.0, y0=2.0, z0=3.0, radius=radius), (1.0, 2.0, 3.0)),
            (S.SphereAtOrigin(radius=radius), (0.0, 0.0, 0.0)),
        ):
            with self.subTest(surface=type(surface).__name__):
                f = lambda x, y, z: (x - centre[0]) ** 2 + (y - centre[1]) ** 2 + (z - centre[2]) ** 2 - radius ** 2
                neg, pos = self._check_volumes(surface, 4.0 / 3.0 * math.pi * radius ** 3)
                self._check_points(neg, f, self._grid(11), inside=True)
                self._check_points(pos, f, self._grid(11), inside=False)

    # ---------------------------------------------------------------- cones

    def _cone_function(self, axis, apex, t_sq, sheet):
        """MCNP cone function, +inf on the side of the apex excluded by the sheet."""
        i = 'xyz'.index(axis)
        def f(*p):
            d = [p[k] - apex[k] for k in range(3)]
            axial = d[i]
            if sheet and axial * sheet < 0:
                return 1.0
            return sum(d[k] ** 2 for k in range(3) if k != i) - t_sq * axial ** 2
        return f

    def test_cone_halfspace(self):
        # double cone (sheet 0) on z through the box centre, radius 10 at the box faces
        t_sq = 0.04
        surface = S.ConeParallelToZAxis(x0=0.0, y0=0.0, z0=0.0, t_sq=t_sq, sheet=0)
        radius_at_box = math.sqrt(t_sq) * self._extent
        expected = 2.0 * math.pi / 3.0 * radius_at_box ** 2 * self._extent
        neg, pos = self._check_volumes(surface, expected)
        f = self._cone_function('z', (0.0, 0.0, 0.0), t_sq, 0)
        self._check_points(neg, f, self._grid(11), inside=True)
        self._check_points(pos, f, self._grid(11), inside=False)

    def test_cone_sheets_halfspace(self):
        # single sheets hold half the double cone volume, on their own side of the apex
        t_sq = 0.04
        expected = math.pi / 3.0 * (math.sqrt(t_sq) * self._extent) ** 2 * self._extent
        for sheet in (1, -1):
            with self.subTest(sheet=sheet):
                surface = S.ConeParallelToZAxis(x0=0.0, y0=0.0, z0=0.0, t_sq=t_sq, sheet=sheet)
                neg, pos = self._check_volumes(surface, expected)
                f = self._cone_function('z', (0.0, 0.0, 0.0), t_sq, sheet)
                self._check_points(neg, f, self._grid(11), inside=True)
                self._check_points(pos, f, self._grid(11), inside=False)

    def test_cone_types_halfspace(self):
        # every cone class, off centre apex, all sheets, checked by points
        apex = (5.0, -3.0, 2.0)
        t_sq = 0.3
        cases = []
        for sheet in (0, 1, -1):
            cases += [
                (S.ConeParallelToXAxis(x0=apex[0], y0=apex[1], z0=apex[2], t_sq=t_sq, sheet=sheet), 'x', apex),
                (S.ConeParallelToYAxis(x0=apex[0], y0=apex[1], z0=apex[2], t_sq=t_sq, sheet=sheet), 'y', apex),
                (S.ConeParallelToZAxis(x0=apex[0], y0=apex[1], z0=apex[2], t_sq=t_sq, sheet=sheet), 'z', apex),
                (S.ConeOnXAxis(x0=apex[0], t_sq=t_sq, sheet=sheet), 'x', (apex[0], 0.0, 0.0)),
                (S.ConeOnYAxis(y0=apex[1], t_sq=t_sq, sheet=sheet), 'y', (0.0, apex[1], 0.0)),
                (S.ConeOnZAxis(z0=apex[2], t_sq=t_sq, sheet=sheet), 'z', (0.0, 0.0, apex[2])),
            ]
        for surface, axis, cone_apex in cases:
            with self.subTest(surface=type(surface).__name__, sheet=surface.sheet):
                neg, pos = self._solids(surface)
                self.assertRelativeToleranceAlmostEqual(self._box_volume, _volume_of_ocp_solids([neg]) + _volume_of_ocp_solids([pos]), self._tol)
                f = self._cone_function(axis, cone_apex, t_sq, surface.sheet)
                self._check_points(neg, f, self._grid(7), inside=True)
                self._check_points(pos, f, self._grid(7), inside=False)

    def test_cone_in_volume(self):
        # inside of a double cone cut by a plane through the apex leaves one nappe
        t_sq = 0.04
        cone = S.ConeParallelToZAxis(x0=0.0, y0=0.0, z0=0.0, t_sq=t_sq, sheet=0)
        solid = ((-cone) & (+S.ZPlane(z0=0.0))).ocp_solid(self._bounding_box, self._decom_params)
        expected = math.pi / 3.0 * (math.sqrt(t_sq) * self._extent) ** 2 * self._extent
        self.assertRelativeToleranceAlmostEqual(expected, _volume_of_ocp_solids([solid]), self._tol)

    # ---------------------------------------------------------------- tori

    def test_torus_halfspace(self):
        # circular tori, volume 2 pi^2 R r^2
        a, r = 20.0, 5.0
        for surface in (
            S.ToriParralleltoXAxis(x0=1.0, y0=2.0, z0=3.0, a=a, b=r, c=r),
            S.ToriParralleltoYAxis(x0=1.0, y0=2.0, z0=3.0, a=a, b=r, c=r),
            S.ToriParralleltoZAxis(x0=1.0, y0=2.0, z0=3.0, a=a, b=r, c=r),
        ):
            with self.subTest(surface=type(surface).__name__):
                self._check_volumes(surface, 2.0 * math.pi ** 2 * a * r ** 2)

    def test_torus_centre_of_tube_inside(self):
        surface = S.ToriParralleltoZAxis(x0=0.0, y0=0.0, z0=0.0, a=20.0, b=5.0, c=5.0)
        neg, pos = self._solids(surface)
        f = lambda x, y, z: _torus_function(math.hypot(x, y), z, 20.0, 5.0, 5.0)
        self._check_points(neg, f, [(20.0, 0.0, 0.0), (0.0, -20.0, 0.0), (0.0, 0.0, 0.0), (30.0, 0.0, 0.0)], inside=True)

    def test_torus_points(self):
        surface = S.ToriParralleltoZAxis(x0=0.0, y0=0.0, z0=0.0, a=20.0, b=5.0, c=5.0)
        neg, _ = self._solids(surface)
        f = lambda x, y, z: _torus_function(math.hypot(x, y), z, 20.0, 5.0, 5.0)
        self._check_points(neg, f, [(20.0, 0.0, 4.0), (24.0, 0.0, 0.0), (16.0, 0.0, 0.0)], inside=True)

    def test_elliptical_torus_points(self):
        a, b, c = 20.0, 3.0, 6.0
        surface = S.ToriParralleltoZAxis(x0=0.0, y0=0.0, z0=0.0, a=a, b=b, c=c)
        neg, _ = self._solids(surface)
        f = lambda x, y, z: _torus_function(math.hypot(x, y), z, a, b, c)
        self._check_points(neg, f, [(20.0, 0.0, 2.5), (25.0, 0.0, 0.0), (20.0, 0.0, 3.5), (27.0, 0.0, 0.0)], inside=True)


    def test_torus_types_points(self):
        # circular, elliptical (both ways) and degenerate (c >= a) tori on each axis, checked by points
        centre = (2.0, -3.0, 1.0)
        for a, b, c in ((20.0, 5.0, 5.0), (20.0, 3.0, 6.0), (20.0, 8.0, 4.0), (6.0, 5.0, 9.0)):
            for surface_type, axis in ((S.ToriParralleltoXAxis, 0), (S.ToriParralleltoYAxis, 1), (S.ToriParralleltoZAxis, 2)):
                surface = surface_type(x0=centre[0], y0=centre[1], z0=centre[2], a=a, b=b, c=c)
                with self.subTest(surface=surface_type.__name__, a=a, b=b, c=c):
                    neg, pos = self._solids(surface)
                    self.assertRelativeToleranceAlmostEqual(self._box_volume, _volume_of_ocp_solids([neg]) + _volume_of_ocp_solids([pos]), self._tol)

                    def f(*p, axis=axis, a=a, b=b, c=c):
                        d = [p[k] - centre[k] for k in range(3)]
                        rho = math.sqrt(sum(d[k] ** 2 for k in range(3) if k != axis))
                        return _torus_function(rho, d[axis], a, b, c)

                    self._check_points(neg, f, self._grid(11), inside=True)
                    self._check_points(pos, f, self._grid(11), inside=False)

    def test_elliptical_torus_volume(self):
        # volume of an elliptical torus is 2 pi^2 a b c
        a, b, c = 20.0, 3.0, 6.0
        self._check_volumes(S.ToriParralleltoZAxis(x0=0.0, y0=0.0, z0=0.0, a=a, b=b, c=c), 2.0 * math.pi ** 2 * a * b * c)

    # ---------------------------------------------------------------- spheres on axes

    def test_sphere_on_axis_halfspace(self):
        radius = 15.0
        for surface, centre in (
            (S.SphereOnXAxis(x0=10.0, radius=radius), (10.0, 0.0, 0.0)),
            (S.SphereOnYAxis(y0=-10.0, radius=radius), (0.0, -10.0, 0.0)),
            (S.SphereOnZAxis(z0=5.0, radius=radius), (0.0, 0.0, 5.0)),
        ):
            with self.subTest(surface=type(surface).__name__):
                f = lambda x, y, z: (x - centre[0]) ** 2 + (y - centre[1]) ** 2 + (z - centre[2]) ** 2 - radius ** 2
                neg, pos = self._check_volumes(surface, 4.0 / 3.0 * math.pi * radius ** 3)
                self._check_points(neg, f, self._grid(11), inside=True)


    # ---------------------------------------------------------------- general quadratics

    def _general_quadratic(self, local_matrix, rotation, centre, local_constant, sign=1.0):
        """GQ for (R^T (x - c))^T L (R^T (x - c)) + k, scaled by sign."""
        m = rotation @ np.diag(local_matrix) @ rotation.T
        c = np.array(centre)
        b = -m @ c
        k = c @ m @ c + local_constant
        m, b, k = sign * m, sign * b, sign * k
        return S.GeneralQuadratic(a=m[0, 0], b=m[1, 1], c=m[2, 2], d=2 * m[0, 1], e=2 * m[1, 2], f=2 * m[0, 2],
                                  g=2 * b[0], h=2 * b[1], j=2 * b[2], k=k)

    def test_general_quadratic_halfspace(self):
        # rotated circular and elliptic cylinders and cones and an ellipsoid, each with both signs of f
        rotation = np.array([[0.36, 0.48, -0.8], [-0.8, 0.6, 0.0], [0.48, 0.64, 0.6]])
        centre = (3.0, -4.0, 2.0)
        cases = {
            'cylinder': ([1, 1, 0], -100.0),
            'elliptic cylinder': ([1 / 36, 1 / 144, 0], -1.0),
            'cone': ([1, 1, -0.09], 0.0),
            'elliptic cone': ([1, 4, -0.25], 0.0),
            'ellipsoid': ([1 / 100, 1 / 225, 1 / 400], -1.0),
        }
        for name, (diagonal, constant) in cases.items():
            for sign in (1.0, -1.0):
                with self.subTest(surface=name, sign=sign):
                    s = self._general_quadratic(diagonal, rotation, centre, constant, sign)
                    f = lambda x, y, z, s=s: (s.a * x * x + s.b * y * y + s.c * z * z + s.d * x * y + s.e * y * z
                                              + s.f * z * x + s.g * x + s.h * y + s.j * z + s.k)
                    neg, pos = self._solids(s)
                    # elliptical sections are scaled B-splines, so allow a small volume error
                    self.assertRelativeToleranceAlmostEqual(self._box_volume, _volume_of_ocp_solids([neg]) + _volume_of_ocp_solids([pos]), 1e-3)
                    self._check_points(neg, f, self._grid(9), inside=True)
                    self._check_points(pos, f, self._grid(9), inside=False)

    def test_general_quadratic_rounded_coefficients(self):
        # a rotated cylinder with coefficients rounded to 7 significant figures, as decks give them
        rotation = np.array([[0.36, 0.48, -0.8], [-0.8, 0.6, 0.0], [0.48, 0.64, 0.6]])
        s = self._general_quadratic([1, 1, 0], rotation, (3.0, -4.0, 2.0), -100.0)
        for name in 'abcdefghjk':
            setattr(s, name, float(f'{getattr(s, name):.7g}'))
        f = lambda x, y, z: (s.a * x * x + s.b * y * y + s.c * z * z + s.d * x * y + s.e * y * z
                             + s.f * z * x + s.g * x + s.h * y + s.j * z + s.k)
        neg, pos = self._solids(s)
        self._check_points(neg, f, self._grid(9), inside=True)

    def test_general_quadratic_cylinder_volume(self):
        # cylinder of radius 10 along z as a GQ: exact primitive, so the volume is exact
        s = S.GeneralQuadratic(a=1.0, b=1.0, c=0.0, d=0.0, e=0.0, f=0.0, g=0.0, h=0.0, j=0.0, k=-100.0)
        self._check_volumes(s, math.pi * 100.0 * 2 * self._extent)

    def test_general_quadratic_hyperboloid_raises(self):
        s = S.GeneralQuadratic(a=1.0, b=1.0, c=-1.0, k=-1.0)
        with self.assertRaises(ValueError):
            (-s).ocp_solid(self._bounding_box, self._decom_params)
