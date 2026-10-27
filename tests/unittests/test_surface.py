import unittest

import mcio
from mcio.formatting import formatting
from mcio.mcnp import MCNPCard
from mcio.surface import (
    Plane,
    PlaneFromThreePoints,
    XPlane,
    YPlane,
    ZPlane,
    Sphere,
    SphereAtOrigin,
    SphereOnXAxis,
    SphereOnYAxis,
    SphereOnZAxis,
    CylinderParallelToXAxis,
    CylinderParallelToYAxis,
    CylinderParallelToZAxis,
    CylinderOnXAxis,
    CylinderOnYAxis,
    CylinderOnZAxis,
    ConeParallelToXAxis,
    ConeParallelToYAxis,
    ConeParallelToZAxis,
    ConeOnXAxis,
    ConeOnYAxis,
    ConeOnZAxis,
    Ellipsoid,
    GeneralQuadratic,
    ToriParralleltoXAxis,
    ToriParralleltoYAxis,
    ToriParralleltoZAxis,
    mcnp_surface,
)


class TestPlaneSurfaces(unittest.TestCase):

    def setUp(self):
        self.plane = Plane(a=1.0, b=0.0, c=0.0, d=5.0)
        self.plane.number = 1

        self.plane_3p = PlaneFromThreePoints(
            x0=0.0, y0=0.0, z0=0.0, x1=1.0, y1=0.0, z1=0.0, x2=0.0, y2=1.0, z2=0.0
        )
        self.plane_3p.number = 2

        self.x_plane = XPlane(x0=10.0)
        self.x_plane.number = 3

        self.y_plane = YPlane(y0=20.0)
        self.y_plane.number = 4

        self.z_plane = ZPlane(z0=30.0)
        self.z_plane.number = 5

    def test_surface_params_plane(self):
        self.assertEqual(self.plane.a, 1.0)
        self.assertEqual(self.plane.b, 0.0)
        self.assertEqual(self.plane.c, 0.0)
        self.assertEqual(self.plane.d, 5.0)

    def test_surface_params_plane_from_three_points(self):
        self.assertEqual(
            (self.plane_3p.x0, self.plane_3p.y0, self.plane_3p.z0),
            (0.0, 0.0, 0.0),
        )
        self.assertEqual(
            (self.plane_3p.x1, self.plane_3p.y1, self.plane_3p.z1),
            (1.0, 0.0, 0.0),
        )
        self.assertEqual(
            (self.plane_3p.x2, self.plane_3p.y2, self.plane_3p.z2),
            (0.0, 1.0, 0.0),
        )

    def test_surface_params_xplane(self):
        self.assertEqual(self.x_plane.x0, 10.0)

    def test_surface_params_yplane(self):
        self.assertEqual(self.y_plane.y0, 20.0)

    def test_surface_params_zplane(self):
        self.assertEqual(self.z_plane.z0, 30.0)

    def test_surface_mcnp_card_plane(self):
        params = [
            f"{1.0:{formatting.MCNP_FMT_SURFACE_PLANE_NORMAL}}",
            f"{0.0:{formatting.MCNP_FMT_SURFACE_PLANE_NORMAL}}",
            f"{0.0:{formatting.MCNP_FMT_SURFACE_PLANE_NORMAL}}",
            f"{5.0:{formatting.MCNP_FMT_SURFACE_PLANE_DISPLACEMENT}}",
        ]
        expected_card = mcnp_surface("P", params, self.plane).as_string
        self.assertEqual(self.plane.mcnp_card.as_string, expected_card)

    def test_surface_mcnp_card_plane_from_three_points(self):
        params = [
            f"{0.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{0.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{0.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{1.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{0.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{0.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{0.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{1.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{0.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
        ]
        expected_card = mcnp_surface("P", params, self.plane_3p).as_string
        self.assertEqual(self.plane_3p.mcnp_card.as_string, expected_card)

    def test_surface_mcnp_card_xplane(self):
        expected_card = (
            f"{self.x_plane.number:<{formatting.MCNP_INDENT}}"
            f" {'PX':<{formatting.MCNP_INDENT}}"
            f" {self.x_plane.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}\n"
        )
        self.assertEqual(self.x_plane.mcnp_card.as_string, expected_card)

    def test_surface_mcnp_card_yplane(self):
        expected_card = (
            f"{self.y_plane.number:<{formatting.MCNP_INDENT}}"
            f" {'PY':<{formatting.MCNP_INDENT}}"
            f" {self.y_plane.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}\n"
        )
        self.assertEqual(self.y_plane.mcnp_card.as_string, expected_card)

    def test_surface_mcnp_card_zplane(self):
        expected_card = (
            f"{self.z_plane.number:<{formatting.MCNP_INDENT}}"
            f" {'PZ':<{formatting.MCNP_INDENT}}"
            f" {self.z_plane.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}\n"
        )
        self.assertEqual(self.z_plane.mcnp_card.as_string, expected_card)


class TestSphereSurfaces(unittest.TestCase):

    def setUp(self):
        self.sphere = Sphere(x0=1.0, y0=2.0, z0=3.0, radius=5.0)
        self.sphere.number = 1

        self.sphere_origin = SphereAtOrigin(radius=10.0)
        self.sphere_origin.number = 2

        self.sphere_x = SphereOnXAxis(x0=2.0, radius=4.0)
        self.sphere_x.number = 3

        self.sphere_y = SphereOnYAxis(y0=3.0, radius=6.0)
        self.sphere_y.number = 4

        self.sphere_z = SphereOnZAxis(z0=4.0, radius=8.0)
        self.sphere_z.number = 5

    def test_surface_params_sphere(self):
        self.assertEqual(
            (
                self.sphere.x0,
                self.sphere.y0,
                self.sphere.z0,
                self.sphere.radius,
            ),
            (1.0, 2.0, 3.0, 5.0),
        )

    def test_surface_params_sphere_at_origin(self):
        self.assertEqual(self.sphere_origin.radius, 10.0)

    def test_surface_params_sphere_on_xaxis(self):
        self.assertEqual((self.sphere_x.x0, self.sphere_x.radius), (2.0, 4.0))

    def test_surface_params_sphere_on_yaxis(self):
        self.assertEqual((self.sphere_y.y0, self.sphere_y.radius), (3.0, 6.0))

    def test_surface_params_sphere_on_zaxis(self):
        self.assertEqual((self.sphere_z.z0, self.sphere_z.radius), (4.0, 8.0))

    def test_surface_mcnp_card_sphere(self):
        params = [
            f"{1.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{2.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{3.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{5.0:{formatting.MCNP_FMT_SURFACE_RADIUS}}",
        ]
        expected_card = mcnp_surface("S", params, self.sphere).as_string
        self.assertEqual(self.sphere.mcnp_card.as_string, expected_card)

    def test_surface_mcnp_card_sphere_at_origin(self):
        expected_card = (
            f"{self.sphere_origin.number:<{formatting.MCNP_INDENT}}"
            f" {'SO':<{formatting.MCNP_INDENT}}"
            f" {self.sphere_origin.radius:{formatting.MCNP_FMT_SURFACE_RADIUS}}\n"
        )
        self.assertEqual(self.sphere_origin.mcnp_card.as_string, expected_card)

    def test_surface_mcnp_card_sphere_on_xaxis(self):
        params = [
            f"{2.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{4.0:{formatting.MCNP_FMT_SURFACE_RADIUS}}",
        ]
        expected_card = mcnp_surface("SX", params, self.sphere_x).as_string
        self.assertEqual(self.sphere_x.mcnp_card.as_string, expected_card)

    def test_surface_mcnp_card_sphere_on_yaxis(self):
        params = [
            f"{3.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{6.0:{formatting.MCNP_FMT_SURFACE_RADIUS}}",
        ]
        expected_card = mcnp_surface("SY", params, self.sphere_y).as_string
        self.assertEqual(self.sphere_y.mcnp_card.as_string, expected_card)

    def test_surface_mcnp_card_sphere_on_zaxis(self):
        params = [
            f"{4.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{8.0:{formatting.MCNP_FMT_SURFACE_RADIUS}}",
        ]
        expected_card = mcnp_surface("SZ", params, self.sphere_z).as_string
        self.assertEqual(self.sphere_z.mcnp_card.as_string, expected_card)


class TestCylinderSurfaces(unittest.TestCase):

    def setUp(self):
        self.cyl_x = CylinderParallelToXAxis(y0=1.0, z0=2.0, radius=3.0)
        self.cyl_x.number = 1

        self.cyl_y = CylinderParallelToYAxis(x0=2.0, z0=3.0, radius=4.0)
        self.cyl_y.number = 2

        self.cyl_z = CylinderParallelToZAxis(x0=3.0, y0=4.0, radius=5.0)
        self.cyl_z.number = 3

        self.cyl_on_x = CylinderOnXAxis(radius=6.0)
        self.cyl_on_x.number = 4

        self.cyl_on_y = CylinderOnYAxis(radius=7.0)
        self.cyl_on_y.number = 5

        self.cyl_on_z = CylinderOnZAxis(radius=8.0)
        self.cyl_on_z.number = 6

    def test_surface_params_cylinder_parallel_to_xaxis(self):
        self.assertEqual(
            (self.cyl_x.y0, self.cyl_x.z0, self.cyl_x.radius), (1.0, 2.0, 3.0)
        )

    def test_surface_params_cylinder_parallel_to_yaxis(self):
        self.assertEqual(
            (self.cyl_y.x0, self.cyl_y.z0, self.cyl_y.radius), (2.0, 3.0, 4.0)
        )

    def test_surface_params_cylinder_parallel_to_zaxis(self):
        self.assertEqual(
            (self.cyl_z.x0, self.cyl_z.y0, self.cyl_z.radius), (3.0, 4.0, 5.0)
        )

    def test_surface_params_cylinder_on_xaxis(self):
        self.assertEqual(self.cyl_on_x.radius, 6.0)

    def test_surface_params_cylinder_on_yaxis(self):
        self.assertEqual(self.cyl_on_y.radius, 7.0)

    def test_surface_params_cylinder_on_zaxis(self):
        self.assertEqual(self.cyl_on_z.radius, 8.0)

    def test_surface_mcnp_card_cylinder_parallel_to_xaxis(self):
        params = [
            f"{1.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{2.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{3.0:{formatting.MCNP_FMT_SURFACE_RADIUS}}",
        ]
        expected_card = mcnp_surface("C/X", params, self.cyl_x).as_string
        self.assertEqual(self.cyl_x.mcnp_card.as_string, expected_card)

    def test_surface_mcnp_card_cylinder_parallel_to_yaxis(self):
        params = [
            f"{2.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{3.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{4.0:{formatting.MCNP_FMT_SURFACE_RADIUS}}",
        ]
        expected_card = mcnp_surface("C/Y", params, self.cyl_y).as_string
        self.assertEqual(self.cyl_y.mcnp_card.as_string, expected_card)

    def test_surface_mcnp_card_cylinder_parallel_to_zaxis(self):
        params = [
            f"{3.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{4.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{5.0:{formatting.MCNP_FMT_SURFACE_RADIUS}}",
        ]
        expected_card = mcnp_surface("C/Z", params, self.cyl_z).as_string
        self.assertEqual(self.cyl_z.mcnp_card.as_string, expected_card)

    def test_surface_mcnp_card_cylinder_on_xaxis(self):
        expected_card = (
            f"{self.cyl_on_x.number:<{formatting.MCNP_INDENT}}"
            f" {'CX':<{formatting.MCNP_INDENT}}"
            f" {self.cyl_on_x.radius:{formatting.MCNP_FMT_SURFACE_RADIUS}}\n"
        )
        self.assertEqual(self.cyl_on_x.mcnp_card.as_string, expected_card)

    def test_surface_mcnp_card_cylinder_on_yaxis(self):
        expected_card = (
            f"{self.cyl_on_y.number:<{formatting.MCNP_INDENT}}"
            f" {'CY':<{formatting.MCNP_INDENT}}"
            f" {self.cyl_on_y.radius:{formatting.MCNP_FMT_SURFACE_RADIUS}}\n"
        )
        self.assertEqual(self.cyl_on_y.mcnp_card.as_string, expected_card)

    def test_surface_mcnp_card_cylinder_on_zaxis(self):
        expected_card = (
            f"{self.cyl_on_z.number:<{formatting.MCNP_INDENT}}"
            f" {'CZ':<{formatting.MCNP_INDENT}}"
            f" {self.cyl_on_z.radius:{formatting.MCNP_FMT_SURFACE_RADIUS}}\n"
        )
        self.assertEqual(self.cyl_on_z.mcnp_card.as_string, expected_card)


class TestConeSurfaces(unittest.TestCase):

    def setUp(self):
        self.cone_x = ConeParallelToXAxis(
            x0=1.0, y0=2.0, z0=3.0, t_sq=0.5, sheet=1
        )
        self.cone_x.number = 1

        self.cone_y = ConeParallelToYAxis(
            x0=1.0, y0=2.0, z0=3.0, t_sq=0.5, sheet=-1
        )
        self.cone_y.number = 2

        self.cone_z = ConeParallelToZAxis(
            x0=1.0, y0=2.0, z0=3.0, t_sq=0.5, sheet=0
        )
        self.cone_z.number = 3

        self.cone_on_x = ConeOnXAxis(x0=1.0, t_sq=0.25, sheet=1)
        self.cone_on_x.number = 4

        self.cone_on_y = ConeOnYAxis(y0=2.0, t_sq=0.25, sheet=-1)
        self.cone_on_y.number = 5

        self.cone_on_z = ConeOnZAxis(z0=3.0, t_sq=0.25, sheet=0)
        self.cone_on_z.number = 6

    def test_surface_params_cone_parallel_to_xaxis(self):
        self.assertEqual(
            (
                self.cone_x.x0,
                self.cone_x.y0,
                self.cone_x.z0,
                self.cone_x.t_sq,
                self.cone_x.sheet,
            ),
            (1.0, 2.0, 3.0, 0.5, 1),
        )

    def test_surface_params_cone_parallel_to_yaxis(self):
        self.assertEqual(
            (
                self.cone_y.x0,
                self.cone_y.y0,
                self.cone_y.z0,
                self.cone_y.t_sq,
                self.cone_y.sheet,
            ),
            (1.0, 2.0, 3.0, 0.5, -1),
        )

    def test_surface_params_cone_parallel_to_zaxis(self):
        self.assertEqual(
            (
                self.cone_z.x0,
                self.cone_z.y0,
                self.cone_z.z0,
                self.cone_z.t_sq,
                self.cone_z.sheet,
            ),
            (1.0, 2.0, 3.0, 0.5, 0),
        )

    def test_surface_params_cone_on_xaxis(self):
        self.assertEqual(
            (self.cone_on_x.x0, self.cone_on_x.t_sq, self.cone_on_x.sheet),
            (1.0, 0.25, 1),
        )

    def test_surface_params_cone_on_yaxis(self):
        self.assertEqual(
            (self.cone_on_y.y0, self.cone_on_y.t_sq, self.cone_on_y.sheet),
            (2.0, 0.25, -1),
        )

    def test_surface_params_cone_on_zaxis(self):
        self.assertEqual(
            (self.cone_on_z.z0, self.cone_on_z.t_sq, self.cone_on_z.sheet),
            (3.0, 0.25, 0),
        )

    def test_surface_mcnp_card_cone_parallel_to_xaxis(self):
        params = [
            f"{1.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{2.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{3.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{0.5:{formatting.MCNP_FMT_SURFACE_T_SQ}}",
            "1",
        ]
        expected_card = mcnp_surface("K/X", params, self.cone_x).as_string
        self.assertEqual(self.cone_x.mcnp_card.as_string, expected_card)

    def test_surface_mcnp_card_cone_parallel_to_yaxis(self):
        params = [
            f"{1.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{2.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{3.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{0.5:{formatting.MCNP_FMT_SURFACE_T_SQ}}",
            "-1",
        ]
        expected_card = mcnp_surface("K/Y", params, self.cone_y).as_string
        self.assertEqual(self.cone_y.mcnp_card.as_string, expected_card)

    def test_surface_mcnp_card_cone_parallel_to_zaxis(self):
        params = [
            f"{1.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{2.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{3.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{0.5:{formatting.MCNP_FMT_SURFACE_T_SQ}}",
            "0",
        ]
        expected_card = mcnp_surface("K/Z", params, self.cone_z).as_string
        self.assertEqual(self.cone_z.mcnp_card.as_string, expected_card)

    def test_surface_mcnp_card_cone_on_xaxis(self):
        params = [
            f"{1.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{0.25:{formatting.MCNP_FMT_SURFACE_T_SQ}}",
            "1",
        ]
        expected_card = mcnp_surface("KX", params, self.cone_on_x).as_string
        self.assertEqual(self.cone_on_x.mcnp_card.as_string, expected_card)

    def test_surface_mcnp_card_cone_on_yaxis(self):
        params = [
            f"{2.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{0.25:{formatting.MCNP_FMT_SURFACE_T_SQ}}",
            "-1",
        ]
        expected_card = mcnp_surface("KY", params, self.cone_on_y).as_string
        self.assertEqual(self.cone_on_y.mcnp_card.as_string, expected_card)

    def test_surface_mcnp_card_cone_on_zaxis(self):
        params = [
            f"{3.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{0.25:{formatting.MCNP_FMT_SURFACE_T_SQ}}",
            "0",
        ]
        expected_card = mcnp_surface("KZ", params, self.cone_on_z).as_string
        self.assertEqual(self.cone_on_z.mcnp_card.as_string, expected_card)


class TestQuadraticSurfaces(unittest.TestCase):

    def setUp(self):
        self.ellipsoid = Ellipsoid(
            a=1.0,
            b=2.0,
            c=3.0,
            d=4.0,
            e=5.0,
            f=6.0,
            g=7.0,
            x0=0.1,
            y0=0.2,
            z0=0.3,
        )
        self.ellipsoid.number = 1

        self.gq = GeneralQuadratic(
            a=1.0,
            b=2.0,
            c=3.0,
            d=4.0,
            e=5.0,
            f=6.0,
            g=7.0,
            h=8.0,
            j=9.0,
            k=10.0,
        )
        self.gq.number = 2

    def test_surface_params_ellipsoid(self):
        self.assertEqual(
            (
                self.ellipsoid.a,
                self.ellipsoid.b,
                self.ellipsoid.c,
                self.ellipsoid.d,
                self.ellipsoid.e,
                self.ellipsoid.f,
                self.ellipsoid.g,
            ),
            (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
        )
        self.assertEqual(
            (self.ellipsoid.x0, self.ellipsoid.y0, self.ellipsoid.z0),
            (0.1, 0.2, 0.3),
        )

    def test_surface_params_general_quadratic(self):
        self.assertEqual(
            (
                self.gq.a,
                self.gq.b,
                self.gq.c,
                self.gq.d,
                self.gq.e,
                self.gq.f,
                self.gq.g,
                self.gq.h,
                self.gq.j,
                self.gq.k,
            ),
            (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0),
        )

    def test_surface_mcnp_card_ellipsoid(self):
        params = [
            f"{1.0:{formatting.MCNP_FMT_SURFACE_ELIPSOID}}",
            f"{2.0:{formatting.MCNP_FMT_SURFACE_ELIPSOID}}",
            f"{3.0:{formatting.MCNP_FMT_SURFACE_ELIPSOID}}",
            f"{4.0:{formatting.MCNP_FMT_SURFACE_ELIPSOID}}",
            f"{5.0:{formatting.MCNP_FMT_SURFACE_ELIPSOID}}",
            f"{6.0:{formatting.MCNP_FMT_SURFACE_ELIPSOID}}",
            f"{7.0:{formatting.MCNP_FMT_SURFACE_ELIPSOID}}",
            f"{0.1:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{0.2:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{0.3:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
        ]
        expected_card = mcnp_surface("SQ", params, self.ellipsoid).as_string
        self.assertEqual(self.ellipsoid.mcnp_card.as_string, expected_card)

    def test_surface_mcnp_card_general_quadratic(self):
        params = [
            f"{1.0:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}",
            f"{2.0:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}",
            f"{3.0:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}",
            f"{4.0:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}",
            f"{5.0:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}",
            f"{6.0:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}",
            f"{7.0:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}",
            f"{8.0:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}",
            f"{9.0:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}",
            f"{10.0:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}",
        ]
        expected_card = mcnp_surface("GQ", params, self.gq).as_string
        self.assertEqual(self.gq.mcnp_card.as_string, expected_card)


class TestToriSurfaces(unittest.TestCase):

    def setUp(self):
        self.tori_x = ToriParralleltoXAxis(
            x0=1.0, y0=2.0, z0=3.0, a=10.0, b=5.0, c=2.0
        )
        self.tori_x.number = 1

        self.tori_y = ToriParralleltoYAxis(
            x0=1.0, y0=2.0, z0=3.0, a=10.0, b=5.0, c=2.0
        )
        self.tori_y.number = 2

        self.tori_z = ToriParralleltoZAxis(
            x0=1.0, y0=2.0, z0=3.0, a=10.0, b=5.0, c=2.0
        )
        self.tori_z.number = 3

    def test_surface_params_tori_parallel_to_xaxis(self):
        self.assertEqual(
            (
                self.tori_x.x0,
                self.tori_x.y0,
                self.tori_x.z0,
                self.tori_x.a,
                self.tori_x.b,
                self.tori_x.c,
            ),
            (1.0, 2.0, 3.0, 10.0, 5.0, 2.0),
        )

    def test_surface_params_tori_parallel_to_yaxis(self):
        self.assertEqual(
            (
                self.tori_y.x0,
                self.tori_y.y0,
                self.tori_y.z0,
                self.tori_y.a,
                self.tori_y.b,
                self.tori_y.c,
            ),
            (1.0, 2.0, 3.0, 10.0, 5.0, 2.0),
        )

    def test_surface_params_tori_parallel_to_zaxis(self):
        self.assertEqual(
            (
                self.tori_z.x0,
                self.tori_z.y0,
                self.tori_z.z0,
                self.tori_z.a,
                self.tori_z.b,
                self.tori_z.c,
            ),
            (1.0, 2.0, 3.0, 10.0, 5.0, 2.0),
        )

    def test_surface_mcnp_card_tori_parallel_to_xaxis(self):
        params = [
            f"{1.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{2.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{3.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{10.0:{formatting.MCNP_FMT_SURFACE_TORI}}",
            f"{5.0:{formatting.MCNP_FMT_SURFACE_TORI}}",
            f"{2.0:{formatting.MCNP_FMT_SURFACE_TORI}}",
        ]
        expected_card = mcnp_surface("TX", params, self.tori_x).as_string
        self.assertEqual(self.tori_x.mcnp_card.as_string, expected_card)

    def test_surface_mcnp_card_tori_parallel_to_yaxis(self):
        params = [
            f"{1.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{2.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{3.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{10.0:{formatting.MCNP_FMT_SURFACE_TORI}}",
            f"{5.0:{formatting.MCNP_FMT_SURFACE_TORI}}",
            f"{2.0:{formatting.MCNP_FMT_SURFACE_TORI}}",
        ]
        expected_card = mcnp_surface("TY", params, self.tori_y).as_string
        self.assertEqual(self.tori_y.mcnp_card.as_string, expected_card)

    def test_surface_mcnp_card_tori_parallel_to_zaxis(self):
        params = [
            f"{1.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{2.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{3.0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}",
            f"{10.0:{formatting.MCNP_FMT_SURFACE_TORI}}",
            f"{5.0:{formatting.MCNP_FMT_SURFACE_TORI}}",
            f"{2.0:{formatting.MCNP_FMT_SURFACE_TORI}}",
        ]
        expected_card = mcnp_surface("TZ", params, self.tori_z).as_string
        self.assertEqual(self.tori_z.mcnp_card.as_string, expected_card)
