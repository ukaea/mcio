"""unittest: the volume conservation check of the auxiliary surface generation."""

import math
import unittest

from mcio.decomposeauxiliaries import _volume_is_conserved


class MCIOUnitTestVolumeIsConserved(unittest.TestCase):
    def test_equal_volumes(self):
        self.assertTrue(_volume_is_conserved(10.0, 10.0, 1e-2))

    def test_within_tolerance(self):
        self.assertTrue(_volume_is_conserved(10.0, 10.05, 1e-2))
        self.assertTrue(_volume_is_conserved(10.0, 9.95, 1e-2))

    def test_outside_tolerance(self):
        self.assertFalse(_volume_is_conserved(10.0, 9.0, 1e-2))
        self.assertFalse(_volume_is_conserved(10.0, 11.0, 1e-2))

    def test_no_volume_in_parts_does_not_divide_by_zero(self):
        self.assertFalse(_volume_is_conserved(10.0, 0.0, 1e-2))
        self.assertFalse(_volume_is_conserved(0.0, 0.0, 1e-2))

    def test_tiny_but_positive_volumes_are_compared(self):
        # the previous check compared against exactly 1e-20, so only that value
        # was caught; small volumes are compared like any other
        self.assertTrue(_volume_is_conserved(1e-25, 1e-25, 1e-2))
        self.assertFalse(_volume_is_conserved(1e-25, 2e-25, 1e-2))

    def test_nan_is_not_conserved(self):
        self.assertFalse(_volume_is_conserved(10.0, math.nan, 1e-2))


if __name__ == "__main__":
    unittest.main()
