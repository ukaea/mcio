"""
unittest: axis aligned BoundingBox - derived values, arithmetic, union,
overlap and splitting.
"""

import math
import unittest

from mcio.geometric import BoundingBox


class MCIOUnitTestBoundingBox(unittest.TestCase):
    def setUp(self):
        self.box = BoundingBox(x_min=-1.0, x_max=3.0, y_min=0.0, y_max=2.0, z_min=-5.0, z_max=-4.0)

    def test_derived_values(self):
        self.assertEqual(self.box.minimums, (-1.0, 0.0, -5.0))
        self.assertEqual(self.box.maximums, (3.0, 2.0, -4.0))
        self.assertEqual((self.box.x_length, self.box.y_length, self.box.z_length), (4.0, 2.0, 1.0))
        self.assertEqual(self.box.centre, (1.0, 1.0, -4.5))
        self.assertEqual(self.box.extent, 2.0)
        self.assertEqual(self.box.volume, 8.0)

    def test_radius_enclosure(self):
        # largest half diagonal of the three projections (x-y here)
        self.assertAlmostEqual(self.box.radius_enclosure, math.sqrt(2.0 ** 2 + 1.0 ** 2))

    def test_length_setter_keeps_centre(self):
        self.box.x_length = 10.0
        self.assertEqual((self.box.x_min, self.box.x_max), (-4.0, 6.0))
        self.assertEqual(self.box.x_mid, 1.0)
        self.assertEqual(self.box.volume, 20.0)

    def test_add_grows_every_side(self):
        grown = self.box + 1.0
        self.assertEqual(grown.minimums, (-2.0, -1.0, -6.0))
        self.assertEqual(grown.maximums, (4.0, 3.0, -3.0))
        self.assertEqual((1.0 + self.box).minimums, grown.minimums)
        # the original is unchanged
        self.assertEqual(self.box.minimums, (-1.0, 0.0, -5.0))

    def test_multiply_scales_about_centre(self):
        scaled = self.box * 2.0
        self.assertEqual(scaled.centre, self.box.centre)
        self.assertEqual((scaled.x_length, scaled.y_length, scaled.z_length), (8.0, 4.0, 2.0))

    def test_make_cubic(self):
        cube = self.box.make_cubic()
        self.assertEqual(cube.centre, self.box.centre)
        self.assertEqual((cube.x_length, cube.y_length, cube.z_length), (4.0, 4.0, 4.0))

    def test_union(self):
        other = BoundingBox(10.0, 11.0, -3.0, 1.0, -4.5, 0.0)
        union = self.box | other
        self.assertEqual(union.minimums, (-1.0, -3.0, -5.0))
        self.assertEqual(union.maximums, (11.0, 2.0, 0.0))

    def test_overlap(self):
        overlapping = BoundingBox(2.0, 5.0, 1.0, 5.0, -4.5, 0.0)
        separate = BoundingBox(3.5, 5.0, 0.0, 2.0, -5.0, -4.0)
        self.assertTrue(self.box ^ overlapping)
        self.assertTrue(overlapping ^ self.box)
        self.assertFalse(self.box ^ separate)

    def test_touching_boxes_do_not_overlap(self):
        touching = BoundingBox(3.0, 5.0, 0.0, 2.0, -5.0, -4.0)
        self.assertFalse(self.box ^ touching)

    def test_contained_box_overlaps(self):
        inner = BoundingBox(0.0, 1.0, 0.5, 1.5, -4.8, -4.2)
        self.assertTrue(self.box ^ inner)
        self.assertTrue(inner ^ self.box)

    def test_split_to_halves_longest_axis(self):
        halves = self.box.split_to_halves()
        self.assertEqual(sorted(halves), ['x_max', 'x_min'])
        self.assertEqual(halves['x_min'].maximums[0], 1.0)
        self.assertEqual(halves['x_max'].minimums[0], 1.0)
        tall = BoundingBox(0, 1, 0, 1, 0, 5)
        self.assertEqual(sorted(tall.split_to_halves()), ['z_max', 'z_min'])
        wide_y = BoundingBox(0, 1, 0, 5, 0, 1)
        self.assertEqual(sorted(wide_y.split_to_halves()), ['y_max', 'y_min'])

    def test_split_to_octants(self):
        octants = self.box.split_to_octants()
        self.assertEqual(len(octants), 8)
        self.assertAlmostEqual(sum(o.volume for o in octants.values()), self.box.volume)
        for octant in octants.values():
            self.assertEqual((octant.x_length, octant.y_length, octant.z_length), (2.0, 1.0, 0.5))

    def test_split_recursive(self):
        for n in range(4):
            boxes = self.box.split(n)
            self.assertEqual(len(boxes), 2 ** n)
            self.assertAlmostEqual(sum(b.volume for b in boxes), self.box.volume)
            # children tile the parent without overlapping
            for i, a in enumerate(boxes):
                for b in boxes[i + 1:]:
                    self.assertFalse(a ^ b)
