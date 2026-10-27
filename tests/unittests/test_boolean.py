"""
unittest: CSG Boolean expressions - HalfVolume, And, Or and Volume.

Covers construction with the & | ~ operators, flattening and de-duplication of
operands, MCNP geometry strings, De Morgan inversion, OpenMC regions (checked
by point membership) and OCP solids (checked by volume).
"""

import unittest

import mcio.surface as S
from mcio.volume import And, Or, HalfVolume, Volume
from mcio.geometric import BoundingBox
from mcio.decomposeparams import DecompositionParameters
from mcio.ocpproperties import _volume_of_ocp_solids


from unitutils import CustomAssertions


def _numbered(surface, number):
    surface.number = number
    return surface


class MCIOUnitTestBoolean(unittest.TestCase, CustomAssertions):
    def setUp(self):
        # unit cube 0 < x, y, z < 10 from six numbered planes
        self.px0 = _numbered(S.XPlane(x0=0.0), 1)
        self.px1 = _numbered(S.XPlane(x0=10.0), 2)
        self.py0 = _numbered(S.YPlane(y0=0.0), 3)
        self.py1 = _numbered(S.YPlane(y0=10.0), 4)
        self.pz0 = _numbered(S.ZPlane(z0=0.0), 5)
        self.pz1 = _numbered(S.ZPlane(z0=10.0), 6)
        self.sphere = _numbered(S.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=5.0), 7)

        self.cube = ((+self.px0) & (-self.px1) & (+self.py0) & (-self.py1) & (+self.pz0) & (-self.pz1))

        # bounds and parameters for OCP solids
        self._extent = 50.0
        self._bounding_box = BoundingBox(-self._extent, self._extent, -self._extent, self._extent, -self._extent, self._extent)
        self._decom_params = DecompositionParameters()
        self._decom_params.bounding_box_multiplier_half_space = 1.0

    def tearDown(self):
        pass

    # ---------------------------------------------------------------- HalfVolume

    def test_half_volume_direction_and_surface(self):
        pos, neg = +self.sphere, -self.sphere
        self.assertIsInstance(pos, HalfVolume)
        self.assertEqual(pos.direction, '+')
        self.assertEqual(neg.direction, '-')
        self.assertIs(pos.surface, self.sphere)
        self.assertEqual(pos.surfaces, [self.sphere])

    def test_half_volume_mcnp_string(self):
        self.assertEqual((+self.sphere).mcnp_string, '7')
        self.assertEqual((-self.sphere).mcnp_string, '-7')

    def test_half_volume_invert(self):
        inverted = ~(-self.sphere)
        self.assertEqual(inverted.direction, '+')
        self.assertIs(inverted.surface, self.sphere)
        self.assertEqual((~~(-self.sphere)).direction, '-')

    def test_half_volume_hash_independent_of_number(self):
        other = _numbered(S.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=5.0), 99)
        self.assertEqual((-self.sphere).__hash_string__(), (-other).__hash_string__())
        self.assertNotEqual((-self.sphere).__hash_string__(), (+other).__hash_string__())

    # ---------------------------------------------------------------- And / Or

    def test_run_boolean_and(self):
        volume = (+self.px0) & (-self.px1)
        self.assertIsInstance(volume, Volume)
        self.assertIsInstance(volume.boolean, And)
        self.assertEqual(volume.mcnp_string, ['1', '-2'])

    def test_run_boolean_or(self):
        volume = (-self.px0) | (+self.px1)
        self.assertIsInstance(volume, Volume)
        self.assertIsInstance(volume.boolean, Or)
        self.assertEqual(volume.mcnp_string, ['-1', ':', '2'])

    def test_and_chain_is_flattened(self):
        self.assertIsInstance(self.cube.boolean, And)
        self.assertEqual(len(self.cube.boolean.sub_booleans), 6)
        self.assertTrue(all(isinstance(sb, HalfVolume) for sb in self.cube.boolean.sub_booleans))
        self.assertEqual(self.cube.mcnp_string, ['1', '-2', '3', '-4', '5', '-6'])

    def test_or_chain_is_flattened(self):
        volume = (-self.px0) | (+self.px1) | (-self.py0)
        self.assertEqual(len(volume.boolean.sub_booleans), 3)
        self.assertEqual(volume.mcnp_string, ['-1', ':', '2', ':', '-3'])

    def test_duplicate_half_volumes_removed(self):
        self.assertEqual(And([+self.px0, +self.px0, -self.px1]).mcnp_string, ['1', '-2'])
        self.assertEqual(Or([+self.px0, +self.px0]).mcnp_string, ['1'])

    def test_duplicate_half_volumes_removed_across_surface_instances(self):
        # identical surfaces with different numbers are duplicates
        same_plane = _numbered(S.XPlane(x0=0.0), 11)
        self.assertEqual(len(And([+self.px0, +same_plane]).sub_booleans), 1)

    def test_opposite_half_volumes_kept(self):
        self.assertEqual(And([+self.px0, -self.px0]).mcnp_string, ['1', '-1'])

    def test_duplicate_half_volumes_removed_from_nested_and(self):
        nested = And([+self.px0, And([+self.px0, -self.px1])])
        self.assertEqual(nested.mcnp_string, ['1', '-2'])

    def test_duplicate_half_volumes_removed_after_nested_and(self):
        nested = And([And([+self.px0, -self.px1]), +self.px0, -self.px1, +self.py0])
        self.assertEqual(nested.mcnp_string, ['1', '-2', '3'])

    def test_duplicate_half_volumes_removed_between_nested_ands(self):
        nested = And([And([+self.px0, -self.px1]), And([-self.px1, +self.py0])])
        self.assertEqual(nested.mcnp_string, ['1', '-2', '3'])

    def test_duplicate_half_volumes_removed_from_nested_or(self):
        nested = Or([-self.px0, Or([-self.px0, +self.px1])])
        self.assertEqual(nested.mcnp_string, ['-1', ':', '2'])
        nested = Or([Or([-self.px0, +self.px1]), Or([+self.px1, -self.py0])])
        self.assertEqual(nested.mcnp_string, ['-1', ':', '2', ':', '-3'])

    def test_chained_operators_have_no_repeated_terms(self):
        cube_again = (+self.px0) & (-self.px1) & (+self.py0) & (-self.py1) & (+self.pz0) & (-self.pz1)
        combined = self.cube & cube_again & (-self.sphere)
        self.assertEqual(combined.mcnp_string, ['1', '-2', '3', '-4', '5', '-6', '-7'])

    def test_nested_unions_inside_intersection_kept(self):
        # unions are kept as operands, only half volumes are de-duplicated
        union = (-self.px0) | (+self.px1)
        volume = And([union.boolean, -self.sphere, union.boolean])
        self.assertEqual(len(volume.sub_booleans), 3)

    def test_deduplicated_region_unchanged(self):
        nested = Volume(And([+self.px0, And([+self.px0, -self.px1]), -self.sphere]))
        region = nested.openmc_region
        for point, inside in (((2, 0, 0), True), ((-1, 0, 0), False), ((4.9, 0, 0), True), ((6, 0, 0), False)):
            self.assertEqual(point in region, inside, point)

    def test_mixed_boolean_parentheses(self):
        volume = ((+self.px0) & (-self.px1)) | (+self.pz1)
        self.assertEqual(volume.mcnp_string, ['(', '1', '-2', ')', ':', '6'])
        volume = ((-self.px0) | (+self.px1)) & (-self.pz1)
        self.assertEqual(volume.mcnp_string, ['(', '-1', ':', '2', ')', '-6'])

    def test_surfaces(self):
        volume = ((+self.px0) & (-self.px1)) | (-self.sphere)
        self.assertEqual(set(id(s) for s in volume.surfaces), {id(self.px0), id(self.px1), id(self.sphere)})

    # ---------------------------------------------------------------- inversion

    def test_invert_and_is_or(self):
        inverted = ~((+self.px0) & (-self.px1))
        self.assertIsInstance(inverted, Volume)
        self.assertIsInstance(inverted.boolean, Or)
        self.assertEqual(inverted.mcnp_string, ['-1', ':', '2'])

    def test_invert_or_is_and(self):
        inverted = ~((-self.px0) | (+self.px1))
        self.assertIsInstance(inverted.boolean, And)
        self.assertEqual(inverted.mcnp_string, ['1', '-2'])

    def test_invert_nested(self):
        volume = ((+self.px0) & (-self.px1)) | (+self.pz1)
        self.assertEqual((~volume).mcnp_string, ['(', '-1', ':', '2', ')', '-6'])

    def test_double_invert_restores_expression(self):
        volume = ((+self.px0) & (-self.px1)) | (+self.pz1)
        self.assertEqual((~~volume).mcnp_string, volume.mcnp_string)

    def test_volume_of_half_volume(self):
        volume = Volume(-self.sphere)
        self.assertEqual(volume.mcnp_string, '-7')
        self.assertEqual((~volume).mcnp_string, '7')

    def test_volume_value(self):
        volume = Volume(-self.sphere, volume=12.5)
        self.assertEqual(volume.volume, 12.5)
        volume.volume = 3.0
        self.assertEqual(volume.volume, 3.0)

    def test_replace_duplicate_surfaces(self):
        same_sphere = _numbered(S.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=5.0), 20)
        volume = (-self.sphere) | (+self.px1)
        other = (+same_sphere) & (-self.px0)
        surfaces = {}
        volume.replace_duplicate_surfaces(surfaces)
        other.replace_duplicate_surfaces(surfaces)
        # the second sphere is replaced by the first instance seen
        self.assertIs(other.boolean.sub_booleans[0].surface, self.sphere)
        self.assertEqual(len(surfaces), 3)

    # ---------------------------------------------------------------- OpenMC regions

    def _assert_points(self, region, inside, outside):
        for point in inside:
            self.assertIn(point, region, f'{point} should be inside')
        for point in outside:
            self.assertNotIn(point, region, f'{point} should be outside')

    def test_openmc_region_and(self):
        self._assert_points(
            self.cube.openmc_region,
            inside=[(5, 5, 5), (0.1, 9.9, 0.1)],
            outside=[(-1, 5, 5), (5, 11, 5), (5, 5, 10.5)]
        )

    def test_openmc_region_or(self):
        region = ((-self.sphere) | ((+self.px1) & (-self.pz1))).openmc_region
        self._assert_points(region, inside=[(0, 0, 0), (20, 0, 0)], outside=[(6, 0, 0), (20, 0, 20)])

    def test_openmc_region_inverted(self):
        self._assert_points(
            (~self.cube).openmc_region,
            inside=[(-1, 5, 5), (5, 11, 5)],
            outside=[(5, 5, 5)]
        )

    # ---------------------------------------------------------------- OCP solids

    def test_ocp_solid_and_box(self):
        solid = self.cube.ocp_solid(self._bounding_box, self._decom_params)
        self.assertRelativeToleranceAlmostEqual(1000.0, _volume_of_ocp_solids([solid]), 1e-6)

    def test_ocp_solid_or_two_boxes(self):
        # cube union with half space x > 40 (box 10 x 100 x 100 in the bounds)
        far = _numbered(S.XPlane(x0=40.0), 8)
        solid = (self.cube | (+far)).ocp_solid(self._bounding_box, self._decom_params)
        self.assertRelativeToleranceAlmostEqual(1000.0 + 10.0 * 100.0 * 100.0, _volume_of_ocp_solids([solid]), 1e-6)

    def test_ocp_solid_and_sphere_octant(self):
        # positive octant of the sphere
        octant = (-self.sphere) & (+self.px0) & (+self.py0) & (+self.pz0)
        solid = octant.ocp_solid(self._bounding_box, self._decom_params)
        expected = 4.0 / 3.0 * 3.141592653589793 * 5.0 ** 3 / 8.0
        self.assertRelativeToleranceAlmostEqual(expected, _volume_of_ocp_solids([solid]), 1e-6)

    def test_ocp_solid_inverted_box(self):
        solid = (~self.cube).ocp_solid(self._bounding_box, self._decom_params)
        expected = (2 * self._extent) ** 3 - 1000.0
        self.assertRelativeToleranceAlmostEqual(expected, _volume_of_ocp_solids([solid]), 1e-6)
