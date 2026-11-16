"""
unittest: MCNP reader surface lookup - each MCNP mnemonic maps to the MCIO
surface class that writes that same mnemonic.
"""

import unittest

from mcio.reader import MCNP_SURFACE_TYPES


class MCIOUnitTestReaderSurfaceTypes(unittest.TestCase):
    def test_mnemonics_round_trip_through_mcnp_card(self):
        for mnemonic, surface_class in MCNP_SURFACE_TYPES.items():
            with self.subTest(mnemonic=mnemonic, surface_class=surface_class.__name__):
                surface = surface_class()
                surface.number = 1
                written = surface.mcnp_card.as_string.split()[1]
                # planes defined by three points are written as a general plane
                expected = "P" if mnemonic == "P3P" else mnemonic
                self.assertEqual(written, expected)


if __name__ == "__main__":
    unittest.main()
