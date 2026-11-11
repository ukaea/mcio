"""
unittest: DecompositionParameters - json write/read round trip, deprecated upper
case names and the printed parameter tables.
"""

import json
import os
import tempfile
import unittest

from mcio.decomposeparams import DecompositionParameters


class MCIOUnitTestDecompositionParameters(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, 'decompose_parameters.json')

    def tearDown(self):
        self.tmp.cleanup()

    def test_write_read_round_trip(self):
        params = DecompositionParameters(void_min_width=20.0)
        params.void_use_obb_check = False
        params.write(self.path)

        read = DecompositionParameters.read(self.path)
        self.assertEqual(read.void_min_width, 20.0)
        self.assertFalse(read.void_use_obb_check)
        self.assertEqual(read.doc_unit_in_metres, params.doc_unit_in_metres)

    def test_write_contains_parameters_not_definitions(self):
        DecompositionParameters().write(self.path)
        with open(self.path) as f:
            data = json.load(f)
        self.assertIn('void_min_width', data)
        self.assertFalse(any(key.lower().startswith('_def_') for key in data))

    def test_read_upper_case_keys_is_deprecated_but_works(self):
        with open(self.path, 'w') as f:
            json.dump({'VOID_MIN_WIDTH': 12.5}, f)
        with self.assertWarns(DeprecationWarning):
            read = DecompositionParameters.read(self.path)
        self.assertEqual(read.void_min_width, 12.5)

    def test_read_unknown_key_raises(self):
        with open(self.path, 'w') as f:
            json.dump({'not_a_parameter': 1}, f)
        with self.assertRaises(KeyError):
            DecompositionParameters.read(self.path)

    def test_deprecated_upper_case_attributes(self):
        params = DecompositionParameters()
        with self.assertWarns(DeprecationWarning):
            params.VOID_MIN_WIDTH = 7.0
        self.assertEqual(params.void_min_width, 7.0)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(params.VOID_MIN_WIDTH, 7.0)

    def test_unknown_attribute_raises(self):
        with self.assertRaises(AttributeError):
            DecompositionParameters().NOT_A_PARAMETER

    def test_str_lists_parameters_with_descriptions(self):
        text = str(DecompositionParameters())
        self.assertIn('void_min_width', text)
        self.assertIn('void generation', text)  # category table title
        self.assertIn(DecompositionParameters()._def_void_min_width, text)


if __name__ == '__main__':
    unittest.main()
