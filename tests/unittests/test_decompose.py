"""
unittest: Decomposer.add_solid - debug STEP output of added solids is opt in.
"""

import os
import tempfile
import unittest

from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt

from mcio.decompose import Decomposer
from mcio.decomposeparams import DecompositionParameters


class MCIOUnitTestAddSolid(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        self.temp_dir = tempfile.TemporaryDirectory()
        os.chdir(self.temp_dir.name)
        self.box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1, 2, 3).Shape()

    def tearDown(self):
        os.chdir(self.cwd)
        self.temp_dir.cleanup()

    def test_added_solids_not_written_by_default(self):
        decomposer = Decomposer()
        decomposer.add_solid(self.box, 'box')
        self.assertEqual(len(decomposer.decomposition_solids), 1)
        self.assertFalse(os.path.exists('added_solids'))

    def test_added_solids_written_when_requested(self):
        params = DecompositionParameters()
        params.debug_write_added_solids = True
        decomposer = Decomposer(decompose_parameters=params)
        decomposer.add_solid(self.box, 'box')
        self.assertTrue(os.path.isdir('added_solids'))
        self.assertEqual(len(os.listdir('added_solids')), 1)
