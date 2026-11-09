"""
unittest: lost particles - extraction of lost particle positions and directions
from an MCNP output file into a STEP file.
"""

import contextlib
import io
import os
import tempfile
import unittest

from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer

from mcio.diagnostic import lost_particles_from_mcnp_run


def lost_particle_record(x, y, z, u, v, w):
    """Lines of an MCNP lost particle record (position and direction)."""
    return (
        ' 1lost particle no.    1 on history         7\n'
        f'       x,y,z coordinates: {x: .5E} {y: .5E} {z: .5E}\n'
        f'       u,v,w direction cosines: {u: .5E} {v: .5E} {w: .5E}\n'
        '\n'
    )


class MCIOUnitTestLostParticles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.input_path = os.path.join(self.tmp.name, 'run.o')
        self.output_path = os.path.join(self.tmp.name, 'lost_particles.stp')

    def tearDown(self):
        self.tmp.cleanup()

    def write_output(self, records):
        with open(self.input_path, 'w') as f:
            f.write(' mcnp     version 6     ld=...\n\n')
            for record in records:
                f.write(lost_particle_record(*record))
            f.write(' run terminated when 100 particles were lost.\n')

    def run_extraction(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            lost_particles_from_mcnp_run(self.input_path, output_file_path=self.output_path, **kwargs)
        return output.getvalue()

    def spheres_in_output(self):
        # count with the plain OCP reader: the spheres are written as one compound,
        # which mcio's STEPFileReader does not currently extract
        reader = STEPControl_Reader()
        reader.ReadFile(self.output_path)
        reader.TransferRoots()
        explorer = TopExp_Explorer(reader.OneShape(), TopAbs_SOLID)
        count = 0
        while explorer.More():
            count += 1
            explorer.Next()
        return count

    def test_lost_particles_written_as_spheres(self):
        self.write_output([(1.0, 2.0, 3.0, 1.0, 0.0, 0.0),
                           (-4.5, 0.0, 12.25, 0.0, 0.6, 0.8),
                           (0.0, -7.0, 0.0, 0.0, 0.0, -1.0)])
        message = self.run_extraction(radius=0.5, vector_length=10.0)
        self.assertIn('3 lost particles extracted', message)
        self.assertTrue(os.path.isfile(self.output_path))
        self.assertEqual(self.spheres_in_output(), 3)

    def test_max_particles_limits_extraction(self):
        self.write_output([(float(i), 0.0, 0.0, 1.0, 0.0, 0.0) for i in range(5)])
        message = self.run_extraction(max_particles=2)
        self.assertIn('2 lost particles extracted', message)
        self.assertEqual(self.spheres_in_output(), 2)

    def test_no_lost_particles_writes_no_file(self):
        self.write_output([])
        message = self.run_extraction()
        self.assertIn('0 lost particles extracted', message)
        self.assertFalse(os.path.exists(self.output_path))

    def test_zero_direction_raises(self):
        self.write_output([(1.0, 2.0, 3.0, 0.0, 0.0, 0.0)])
        with self.assertRaises(Exception):
            self.run_extraction()


if __name__ == '__main__':
    unittest.main()
