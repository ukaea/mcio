"""
unittest: particle sources - MCNP SDEF cards and their distribution sub-cards.
"""

import math
import unittest

import mcio.surface as S
from mcio.source import (
    SourceEnergyDistribution,
    SourcePositionDistribution,
    Source,
    MCNPSourceFromFile,
    Point,
    SphericalStochasticVolumeSource,
    RectangularParallelepipedVolume,
    IsotropicCylindricalVolume,
    CircularPlaneParrallel
)


from unitutils import CustomAssertions


def _lines(card):
    """Return the card text as stripped, non-empty lines."""
    return [line.rstrip() for line in card.as_string.splitlines() if line.strip()]


def _keywords(card):
    """Return the KEY=VALUE tokens of an SDEF card (continuation lines included)."""
    tokens = []
    for line in _lines(card):
        if line.startswith(('SI', 'SP')):
            break
        tokens.extend(t for t in line.split() if '=' in t)
    return tokens


class MCIOUnitTestSource(unittest.TestCase, CustomAssertions):
    def setUp(self):
        self.sphere = S.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=10.0)
        self.sphere.number = 5

    # ---------------------------------------------------------------- distributions

    def test_energy_distribution_discrete(self):
        si, sp = SourceEnergyDistribution([1.0, 2.0], [0.25, 0.75]).mcnp_subcards
        self.assertEqual(si.as_string.split(), ['SI1', 'L', '1.0', '2.0'])
        self.assertEqual(sp.as_string.split(), ['SP1', '0.25', '0.75'])

    def test_energy_distribution_histogram(self):
        si, sp = SourceEnergyDistribution([1.0, 2.0, 3.0], [0.0, 0.5, 0.5], 'histogram').mcnp_subcards
        self.assertEqual(si.as_string.split()[:2], ['SI1', 'H'])
        self.assertEqual(sp.as_string.split(), ['SP1', '0.0', '0.5', '0.5'])

    def test_energy_distribution_unknown_type_raises(self):
        with self.assertRaises(KeyError):
            SourceEnergyDistribution([1.0], [1.0], 'gaussian').mcnp_subcards

    def test_position_distribution(self):
        si, sp = SourcePositionDistribution([(1, 2, 3), (4, 5, 6)], [0.5, 0.5]).mcnp_subcards
        self.assertEqual(si.as_string.split(), ['SI2', 'L', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0'])
        self.assertEqual(sp.as_string.split(), ['SP2', '0.5', '0.5'])

    # ---------------------------------------------------------------- base class

    def test_particle(self):
        self.assertEqual(Source(particle='photon').particle, 'photon')
        self.assertEqual(Source().particle, 'neutron')

    def test_energy(self):
        self.assertEqual(Source(energy=14.1).energy, 14.1)

    # ---------------------------------------------------------------- read from file

    def test_source_from_file(self):
        source = MCNPSourceFromFile('source.txt')
        self.assertEqual(source.file_path, 'source.txt')
        self.assertEqual(source.mcnp_card.as_string.split(), ['READ', 'file=', 'source.txt'])

    # ---------------------------------------------------------------- point

    def test_point_card(self):
        card = Point(position=(1, 2, 3), energy=14.1).mcnp_card
        lines = _lines(card)
        self.assertEqual(lines[0].split()[0], 'SDEF')
        self.assertIn('ERG=D1', lines[0])
        self.assertIn('POS=D2', lines[0])
        self.assertEqual(lines[1].split(), ['SI1', 'L', '14.1'])
        self.assertEqual(lines[2].split(), ['SP1', '1.0'])
        self.assertEqual(lines[3].split(), ['SI2', 'L', '1.0', '2.0', '3.0'])
        self.assertEqual(lines[4].split(), ['SP2', '1.0'])

    def test_point_default_position(self):
        source = Point()
        self.assertEqual(source.position, (0, 0, 0))
        self.assertEqual(_lines(source.mcnp_card)[3].split(), ['SI2', 'L', '0.0', '0.0', '0.0'])

    def test_point_with_distributions(self):
        energy = SourceEnergyDistribution([1.0, 2.0], [0.5, 0.5])
        position = SourcePositionDistribution([(0, 0, 0), (10, 0, 0)], [0.9, 0.1])
        lines = _lines(Point(position=position, energy=energy).mcnp_card)
        self.assertEqual(lines[1].split(), ['SI1', 'L', '1.0', '2.0'])
        self.assertEqual(lines[3].split(), ['SI2', 'L', '0.0', '0.0', '0.0', '10.0', '0.0', '0.0'])
        self.assertEqual(lines[4].split(), ['SP2', '0.9', '0.1'])

    def test_point_particle_designator(self):
        self.assertIn('PAR=N', _keywords(Point().mcnp_card))

    # ---------------------------------------------------------------- spherical stochastic volume

    def test_spherical_stochastic_volume_card(self):
        card = SphericalStochasticVolumeSource(self.sphere, energy=2.0).mcnp_card
        keywords = _keywords(card)
        self.assertIn('PAR=N', keywords)
        self.assertIn('ERG=D1', keywords)
        self.assertIn('NRM=-1', keywords)
        self.assertIn('SUR=5', keywords)
        self.assertIn('DIR=D2', keywords)
        weight = float([k for k in keywords if k.startswith('WGT=')][0][4:])
        self.assertRelativeToleranceAlmostEqual(math.pi * 10.0 ** 2, weight, 1e-12)
        lines = _lines(card)
        self.assertEqual(lines[-2].split(), ['SI2', '0', '1'])
        self.assertEqual(lines[-1].split(), ['SP2', '-21', '1'])

    def test_spherical_stochastic_volume_photon(self):
        card = SphericalStochasticVolumeSource(self.sphere, particle='photon').mcnp_card
        self.assertIn('PAR=P', _keywords(card))

    # ---------------------------------------------------------------- rectangular volume

    def test_rectangular_volume_card(self):
        source = RectangularParallelepipedVolume(x_min=-1, x_max=1, y_min=-2, y_max=2, z_min=-3, z_max=3,
                                                 direction=(0, 0, 1), energy=1.0)
        card = source.mcnp_card
        keywords = _keywords(card)
        for keyword in ('ERG=D1', 'X=D3', 'Y=D4', 'Z=D5', 'DIR=1'):
            self.assertIn(keyword, keywords)
        lines = _lines(card)
        self.assertIn('VEC=0.000000e+00 0.000000e+00 1.000000e+00', ' '.join(lines))
        self.assertIn(['SI3', '-1.000000e+00', '1.000000e+00'], [l.split() for l in lines])
        self.assertIn(['SI4', '-2.000000e+00', '2.000000e+00'], [l.split() for l in lines])
        self.assertIn(['SI5', '-3.000000e+00', '3.000000e+00'], [l.split() for l in lines])
        for designator in ('SP3', 'SP4', 'SP5'):
            self.assertIn([designator, '0', '1'], [l.split() for l in lines])

    def test_rectangular_volume_degenerate_x(self):
        source = RectangularParallelepipedVolume(x_min=2, x_max=2, y_min=0, y_max=1, z_min=0, z_max=1)
        keywords = _keywords(source.mcnp_card)
        self.assertIn('X=2', keywords)
        self.assertNotIn('SI3', source.mcnp_card.as_string)

    def test_rectangular_volume_degenerate_y(self):
        source = RectangularParallelepipedVolume(x_min=0, x_max=1, y_min=2, y_max=2, z_min=0, z_max=1)
        self.assertIn('Y=2', _keywords(source.mcnp_card))

    def test_rectangular_volume_degenerate_z(self):
        source = RectangularParallelepipedVolume(x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=0)
        keywords = _keywords(source.mcnp_card)
        self.assertIn('Z=0', keywords)
        self.assertEqual(len([k for k in keywords if k.startswith('X=')]), 1)

    # ---------------------------------------------------------------- cylindrical volume

    def test_cylindrical_volume_card(self):
        source = IsotropicCylindricalVolume(x0=1, y0=2, z0=3, u0=0, v0=0, w0=1, radius=4, height=5, energy=2.0)
        card = source.mcnp_card
        keywords = _keywords(card)
        for keyword in ('ERG=D1', 'EXT=d3', 'RAD=D2'):
            self.assertIn(keyword, keywords)
        text = ' '.join(_lines(card))
        self.assertIn('POS=1.000000e+00 2.000000e+00 3.000000e+00', text)
        self.assertIn('AXS=0.000000e+00 0.000000e+00 1.000000e+00', text)
        split_lines = [l.split() for l in _lines(card)]
        self.assertIn(['SI1', 'L', '2.0'], split_lines)
        self.assertIn(['SI2', '0.000000e+00', '4.000000e+00'], split_lines)
        self.assertIn(['SP2', '-21', '1'], split_lines)
        self.assertIn(['SI3', '0.000000e+00', '5.000000e+00'], split_lines)

    # ---------------------------------------------------------------- circular plane

    def test_circular_plane_card(self):
        source = CircularPlaneParrallel(x0=0, y0=0, z0=-10, u0=0, v0=0, w0=1, radius=3, energy=1.0)
        card = source.mcnp_card
        keywords = _keywords(card)
        for keyword in ('ERG=D1', 'EXT=0', 'DIR=1', 'RAD=D2'):
            self.assertIn(keyword, keywords)
        text = ' '.join(_lines(card))
        self.assertIn('POS=0.000000e+00 0.000000e+00 -1.000000e+01', text)
        self.assertIn('AXS=0.000000e+00 0.000000e+00 1.000000e+00', text)
        self.assertIn('VEC=0.000000e+00 0.000000e+00 1.000000e+00', text)
        split_lines = [l.split() for l in _lines(card)]
        self.assertIn(['SI2', '0.000000e+00', '3.000000e+00'], split_lines)
        self.assertIn(['SP2', '-21', '1'], split_lines)


    def test_energy_distribution_returned(self):
        energy = SourceEnergyDistribution([1.0, 2.0], [0.5, 0.5])
        self.assertIs(Source(energy=energy).energy, energy)

    def test_particle_designators(self):
        sources = [Point(particle='photon'),
                   RectangularParallelepipedVolume(x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1, particle='photon'),
                   IsotropicCylindricalVolume(radius=1, height=1, w0=1, particle='photon'),
                   CircularPlaneParrallel(radius=1, w0=1, particle='photon')]
        for source in sources:
            with self.subTest(source=type(source).__name__):
                self.assertIn('PAR=P', _keywords(source.mcnp_card))
        # a designator given directly is passed through
        self.assertIn('PAR=E', _keywords(Point(particle='e').mcnp_card))

    def test_rectangular_volume_degenerate_z_not_at_zero(self):
        source = RectangularParallelepipedVolume(x_min=0, x_max=1, y_min=0, y_max=1, z_min=5, z_max=5)
        self.assertIn('Z=5', _keywords(source.mcnp_card))
        self.assertNotIn('SI5', source.mcnp_card.as_string)
