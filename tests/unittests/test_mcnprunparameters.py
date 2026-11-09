"""
unittest: MCNPRunParameters - NPS, MODE, PHYS, CUT, STOP, WWP, IDUM/RDUM, SPABI,
FCL and XSDIR message cards.
"""

import unittest

from mcio.cell import Cell
from mcio.checkvals import TypeException
from mcio.mcnp import MCNPRunParameters
from mcio.tally import VolumeFluenceTally


def card_lines(run_parameters):
    """Return the run parameter cards as a list of stripped lines."""
    return [line.rstrip() for line in run_parameters.mcnp_card.as_string.splitlines() if line.strip()]


class MCIOUnitTestMCNPRunParameters(unittest.TestCase):
    def test_default_cards(self):
        lines = card_lines(MCNPRunParameters())
        self.assertEqual(lines[0], 'NPS      1e+09')
        self.assertIn('MODE N', lines)
        self.assertIn('LOST     100', lines)
        # optional cards are not written by default
        for designator in ['VOID', 'PHYS', 'CUT', 'STOP', 'WWP', 'IDUM', 'RDUM', 'SPABI', 'FCL']:
            self.assertFalse(any(line.startswith(designator) for line in lines), designator)

    def test_prdmp_from_nps_and_intervals(self):
        lines = card_lines(MCNPRunParameters(nps=1e6, prdmp_intervals=10, write_mctal=True))
        self.assertIn('PRDMP    1.000000e+05 1.000000e+05 1 1 1.000000e+05', lines)

    def test_prdmp_without_mctal(self):
        lines = card_lines(MCNPRunParameters(nps=1e6, prdmp_intervals=10, write_mctal=False))
        self.assertIn('PRDMP    1.000000e+05 1.000000e+05 0 1 1.000000e+05', lines)

    def test_mode_uses_mcnp_particle_designators(self):
        lines = card_lines(MCNPRunParameters(modes=['neutron', 'photon', 'electron']))
        self.assertIn('MODE N P E', lines)

    def test_void_card(self):
        self.assertIn('VOID', card_lines(MCNPRunParameters(void=True)))

    def test_physics_and_cut_cards(self):
        lines = card_lines(MCNPRunParameters(physics_cards=['neutron', 'photon'],
                                             cut_parameters={'neutron': {'t': 1e33, 'e': 1e-3}}))
        self.assertIn('PHYS:N', lines)
        self.assertIn('PHYS:P', lines)
        self.assertIn('CUT:N    1e+33 0.001', lines)

    def test_stop_card_uses_tally_number(self):
        cell = Cell(name='cell')
        cell.number = 1
        tally = VolumeFluenceTally(cells=[cell])
        tally.number = 1
        lines = card_lines(MCNPRunParameters(stop_tallies={tally: 0.02}))
        self.assertIn('STOP     F14 0.02', lines)

    def test_stop_tallies_must_be_dict(self):
        with self.assertRaises(TypeException):
            MCNPRunParameters(stop_tallies=[0.02])

    def test_weight_window_card(self):
        lines = card_lines(MCNPRunParameters(weight_window={'n': [0.5, 2.0, 5]}))
        self.assertIn('WWP:N    0.500000 2.000000 5', lines)

    def test_idum_card(self):
        self.assertIn('IDUM     1 2', card_lines(MCNPRunParameters(idums=[1, 2])))

    def test_rdum_card_uses_rdums(self):
        # regression: RDUM was written as 'RDUMS' using the idums values
        lines = card_lines(MCNPRunParameters(idums=[1, 2], rdums=[0.5, 1.5]))
        self.assertIn('RDUM     0.500000 1.500000', lines)
        self.assertFalse(any(line.startswith('RDUMS') for line in lines))

    def test_spabi_card(self):
        spabi = {'n': {'primary_particles': ['n', 'p'], 'energies': [1.0, 2.0], 'multipliers': [0.5, 0.25]}}
        lines = card_lines(MCNPRunParameters(spabi=spabi))
        self.assertIn('SPABI:n  np 1.000000 0.500000 2.000000 0.250000', lines)

    def test_forced_collision_card_fills_missing_cells_with_zero(self):
        cell_1, cell_3 = Cell(name='cell_1'), Cell(name='cell_3')
        cell_1.number, cell_3.number = 1, 3
        lines = card_lines(MCNPRunParameters(forced_collisions={'n': {cell_1: 0.5, cell_3: 1.0}}))
        self.assertIn('FCL:n    0.500000 0.000000 1.000000', lines)

    def test_forced_collisions_must_be_dict(self):
        with self.assertRaises(Exception):
            MCNPRunParameters(forced_collisions=[0.5])

    def test_xsdir_message(self):
        self.assertEqual(MCNPRunParameters().messages, [])
        messages = MCNPRunParameters(xsdir='/path/to/xsdir').messages
        self.assertEqual(len(messages), 1)
        self.assertIn('XSDIR=/path/to/xsdir', messages[0].as_string)


if __name__ == '__main__':
    unittest.main()
