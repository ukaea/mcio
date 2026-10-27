"""
unittest: macrobodies (RPP, RCC) and transforms - MCNP cards, conversion to
volumes of primitive surfaces and inversion into OCP solids.
"""

import math
import unittest

from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.gp import gp_Pnt

import mcio.surface as S
from mcio.macrobody import RectangularParallelepiped, RightCircularCylinder
from mcio.transform import Transform
from mcio.volume import And, HalfVolume
from mcio.geometric import BoundingBox
from mcio.decomposeparams import DecompositionParameters
from mcio.ocpproperties import _volume_of_ocp_solids


from unitutils import CustomAssertions


def _centre_of_mass(shape):
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    centre = props.CentreOfMass()
    return centre.X(), centre.Y(), centre.Z()


class MCIOUnitTestMacrobody(unittest.TestCase, CustomAssertions):
    def setUp(self):
        self._bounding_box = BoundingBox(-50, 50, -50, 50, -50, 50)
        self._decom_params = DecompositionParameters()
        self._decom_params.bounding_box_multiplier_half_space = 1.0

    # ---------------------------------------------------------------- RPP

    def test_rpp_mcnp_card(self):
        rpp = RectangularParallelepiped(-1, 2, -3, 4, -5, 6)
        rpp.number = 3
        self.assertEqual(rpp.mcnp_card.as_string.split(),
                         ['3', 'RPP', '-1.000000', '2.000000', '-3.000000', '4.000000', '-5.000000', '6.000000'])

    def test_rpp_half_volumes(self):
        rpp = RectangularParallelepiped(-1, 2, -3, 4, -5, 6)
        rpp.number = 3
        self.assertEqual((-rpp).mcnp_string, '-3')
        self.assertEqual((+rpp).mcnp_string, '3')
        self.assertIs((-rpp).surface, rpp)

    def test_rpp_as_volume_of_surfaces(self):
        rpp = RectangularParallelepiped(-1, 2, -3, 4, -5, 6)
        volume = rpp.as_volume_of_surfaces()
        self.assertIsInstance(volume.boolean, And)
        expected = [('+', S.XPlane, -1), ('-', S.XPlane, 2), ('+', S.YPlane, -3),
                    ('-', S.YPlane, 4), ('+', S.ZPlane, -5), ('-', S.ZPlane, 6)]
        for half_volume, (direction, surface_type, value) in zip(volume.boolean.sub_booleans, expected):
            self.assertEqual(half_volume.direction, direction)
            self.assertIsInstance(half_volume.surface, surface_type)
        self.assertEqual([hv.surface.x0 for hv in volume.boolean.sub_booleans[:2]], [-1, 2])
        self.assertEqual(len(rpp.as_half_volumes()), 6)

    def test_rpp_inside_ocp_solid(self):
        rpp = RectangularParallelepiped(-1, 2, -3, 4, -5, 6)
        solid = (-rpp).ocp_solid(self._bounding_box, self._decom_params)
        self.assertRelativeToleranceAlmostEqual(3 * 7 * 11, _volume_of_ocp_solids([solid]), 1e-6)
        for value, expected in zip(_centre_of_mass(solid), (0.5, 0.5, 0.5)):
            self.assertAlmostEqual(value, expected, places=6)

    def test_rpp_outside_ocp_solid(self):
        rpp = RectangularParallelepiped(-1, 2, -3, 4, -5, 6)
        solid = (+rpp).ocp_solid(self._bounding_box, self._decom_params)
        self.assertRelativeToleranceAlmostEqual(100.0 ** 3 - 3 * 7 * 11, _volume_of_ocp_solids([solid]), 1e-6)

    # ---------------------------------------------------------------- RCC

    def test_rcc_mcnp_card(self):
        rcc = RightCircularCylinder(1, 2, 3, 0, 0, 10, 4)
        rcc.number = 4
        self.assertEqual(rcc.mcnp_card.as_string.split(),
                         ['4', 'RCC', '1.000000', '2.000000', '3.000000', '0.000000', '0.000000', '10.000000', '4.000000'])
        for line in rcc.mcnp_card.as_string.splitlines():
            self.assertLess(len(line), 80)

    def test_rcc_zero_height_raises(self):
        with self.assertRaises(ValueError):
            RightCircularCylinder(0, 0, 0, 0, 0, 0, 1).as_volume_of_surfaces()

    def test_rcc_inside_ocp_solid_volume(self):
        for args in ((1, 2, 3, 0, 0, 10, 4), (1, 2, 3, 3, 4, 0, 4), (5, -2, 1, 1, 1, 1, 2)):
            with self.subTest(rcc=args):
                x0, y0, z0, hx, hy, hz, radius = args
                solid = (-RightCircularCylinder(*args)).ocp_solid(self._bounding_box, self._decom_params)
                expected = math.pi * radius ** 2 * math.sqrt(hx ** 2 + hy ** 2 + hz ** 2)
                self.assertRelativeToleranceAlmostEqual(expected, _volume_of_ocp_solids([solid]), 1e-6)

    def test_rcc_inside_ocp_solid_position(self):
        for args in ((1, 2, 3, 0, 0, 10, 4), (1, 2, 3, 3, 4, 0, 4), (5, -2, 1, 1, 1, 1, 2)):
            with self.subTest(rcc=args):
                x0, y0, z0, hx, hy, hz, radius = args
                solid = (-RightCircularCylinder(*args)).ocp_solid(self._bounding_box, self._decom_params)
                for value, expected in zip(_centre_of_mass(solid), (x0 + hx / 2, y0 + hy / 2, z0 + hz / 2)):
                    self.assertAlmostEqual(value, expected, places=6)


class MCIOUnitTestTransform(unittest.TestCase):
    def _rows(self, transform):
        """Card lines without the header comments."""
        return [l.split() for l in transform.mcnp_card.as_string.splitlines() if l and not l.startswith('C')]

    def test_degrees_card(self):
        transform = Transform(lateral=(1, 2, 3))
        transform.number = 5
        rows = self._rows(transform)
        self.assertEqual(rows[0][0], '*TR5')
        self.assertEqual([float(v) for v in rows[0][1:]], [1.0, 2.0, 3.0])
        self.assertEqual([float(v) for v in rows[1]], [0.0, 90.0, 90.0])
        self.assertEqual([float(v) for v in rows[3]], [90.0, 90.0, 0.0])
        self.assertEqual(rows[4], ['1'])

    def test_cosine_card(self):
        transform = Transform(lateral=(0, 0, 0), rotation_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), angle_type='cosine')
        transform.number = 6
        self.assertEqual(self._rows(transform)[0][0], 'TR6')

    def test_unknown_angle_type_raises(self):
        transform = Transform(angle_type='radians')
        transform.number = 1
        with self.assertRaises(Exception):
            transform.mcnp_card
        with self.assertRaises(ValueError):
            transform.ocp_transform

    def test_m_flag_minus_one(self):
        transform = Transform(displacement_vector_at_origin=True)
        transform.number = 7
        self.assertEqual(self._rows(transform)[-1], ['-1'])

    def test_ocp_transform_translation(self):
        transform = Transform(lateral=(1, 2, 3))
        point = gp_Pnt(0, 0, 0).Transformed(transform.ocp_transform)
        self.assertAlmostEqual(point.X(), 1.0)
        self.assertAlmostEqual(point.Y(), 2.0)
        self.assertAlmostEqual(point.Z(), 3.0)

    def test_ocp_transform_rotation(self):
        # auxiliary x' axis along global y, y' along -x, z' along z (90 degrees about z)
        transform = Transform(rotation_matrix=((90, 0, 90), (180, 90, 90), (90, 90, 0)))
        point = gp_Pnt(1, 0, 0).Transformed(transform.ocp_transform)
        self.assertAlmostEqual(point.X(), 0.0)
        self.assertAlmostEqual(point.Y(), 1.0)
        self.assertAlmostEqual(point.Z(), 0.0)

    def test_ocp_transform_rotation_and_translation(self):
        transform = Transform(lateral=(1, 2, 3), rotation_matrix=((90, 0, 90), (180, 90, 90), (90, 90, 0)))
        point = gp_Pnt(0, 0, 0).Transformed(transform.ocp_transform)
        self.assertAlmostEqual(point.X(), 1.0)
        self.assertAlmostEqual(point.Y(), 2.0)
        self.assertAlmostEqual(point.Z(), 3.0)


    def test_ocp_transform_m_minus_one(self):
        # m = -1: the displacement is the origin of the main system in the auxiliary system,
        # x' = R x + o, so the auxiliary origin is at x = -R^T o
        transform = Transform(lateral=(1, 2, 3), rotation_matrix=((90, 0, 90), (180, 90, 90), (90, 90, 0)),
                              displacement_vector_at_origin=True)
        point = gp_Pnt(0, 0, 0).Transformed(transform.ocp_transform)
        # R^T o = 1 * x' + 2 * y' + 3 * z' with x' = (0, 1, 0), y' = (-1, 0, 0), z' = (0, 0, 1)
        self.assertAlmostEqual(point.X(), 2.0)
        self.assertAlmostEqual(point.Y(), -1.0)
        self.assertAlmostEqual(point.Z(), -3.0)

    def test_ocp_transform_m_minus_one_translation_only(self):
        transform = Transform(lateral=(1, 2, 3), displacement_vector_at_origin=True)
        point = gp_Pnt(5, 5, 5).Transformed(transform.ocp_transform)
        self.assertAlmostEqual(point.X(), 4.0)
        self.assertAlmostEqual(point.Y(), 3.0)
        self.assertAlmostEqual(point.Z(), 2.0)

    def test_ocp_transform_rounded_cosines(self):
        # cosines rounded to 4 figures are orthonormalised rather than rejected
        c, s = round(math.cos(0.3), 4), round(math.sin(0.3), 4)
        transform = Transform(rotation_matrix=((c, s, 0), (-s, c, 0), (0, 0, 1)), angle_type='cosine')
        point = gp_Pnt(1, 0, 0).Transformed(transform.ocp_transform)
        self.assertAlmostEqual(point.X(), math.cos(0.3), places=4)
        self.assertAlmostEqual(point.Y(), math.sin(0.3), places=4)

    def test_transformed_surface_far_from_origin(self):
        # a cube moved well away from the origin is still built whole within the bounding box
        from mcio.surface import XPlane, YPlane, ZPlane
        transform = Transform(lateral=(300, -200, 150), rotation_matrix=((90, 0, 90), (180, 90, 90), (90, 90, 0)))
        cube = (+XPlane(x0=0.0, transform=transform) & -XPlane(x0=10.0, transform=transform)
                & +YPlane(y0=0.0, transform=transform) & -YPlane(y0=10.0, transform=transform)
                & +ZPlane(z0=0.0, transform=transform) & -ZPlane(z0=10.0, transform=transform))
        params = DecompositionParameters()
        params.bounding_box_multiplier_half_space = 1.0
        solid = cube.ocp_solid(BoundingBox(250, 350, -250, -150, 100, 200), params)
        self.assertAlmostEqual(_volume_of_ocp_solids([solid]), 1000.0, places=6)
        for value, expected in zip(_centre_of_mass(solid), (295.0, -195.0, 155.0)):
            self.assertAlmostEqual(value, expected, places=6)

    def test_m_flag_one(self):
        transform = Transform(displacement_vector_at_origin=False)
        transform.number = 8
        self.assertEqual(self._rows(transform)[-1], ['1'])
