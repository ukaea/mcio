"""
unittest: MCNPCard formatting - line wrapping, rows, headers and sub-cards.
"""

import unittest

from mcio.mcnp import MCNPCard
from mcio.formatting import formatting


class MCIOUnitTestMCNPCard(unittest.TestCase):
    def test_single_row(self):
        card = MCNPCard('NPS', [['1000']])
        self.assertEqual(card.as_string, f"{'NPS':<{formatting.MCNP_INDENT}} 1000\n")

    def test_integer_designator(self):
        card = MCNPCard(12, [['0', '-1', '2']])
        self.assertEqual(card.as_string.split(), ['12', '0', '-1', '2'])

    def test_long_row_wrapped_below_max_chars(self):
        entries = [f'{i:.6e}' for i in range(40)]
        card = MCNPCard('SI1', [entries])
        lines = card.as_string.splitlines()
        self.assertGreater(len(lines), 1)
        for line in lines:
            self.assertLess(len(line), formatting.MCNP_MAX_CHARS)
        for line in lines[1:]:
            # continuation lines are indented by at least five spaces
            self.assertTrue(line.startswith('     '))

        self.assertEqual(card.as_string.split()[1:], entries)

    def test_each_row_on_new_line(self):
        card = MCNPCard('TR1', [['1', '2', '3'], ['4', '5', '6']])
        lines = card.as_string.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[1].split(), ['4', '5', '6'])
        self.assertTrue(lines[1].startswith(' ' * formatting.MCNP_INDENT))

    def test_add_row(self):
        card = MCNPCard('M1', [['1001.80c', '2.0']])
        card.add_row(['8016.80c', '1.0'])
        self.assertEqual([l.split() for l in card.as_string.splitlines()], [['M1', '1001.80c', '2.0'], ['8016.80c', '1.0']])

    def test_sub_cards_follow_card(self):
        card = MCNPCard('SDEF', [['ERG=D1']])
        card.add_subcard(MCNPCard('SI1', [['14.1']]))
        card.add_subcard(MCNPCard('SP1', [['1']]))
        self.assertEqual([l.split()[0] for l in card.as_string.splitlines()], ['SDEF', 'SI1', 'SP1'])

    def test_header_comments(self):
        card = MCNPCard('M1', [['1001.80c', '1.0']], header_data={'name': 'hydrogen', 'density': 0.1})
        lines = card.as_string.splitlines()
        self.assertEqual(lines[0], 'C')
        self.assertIn('C name: hydrogen', lines)
        self.assertIn('C density: 0.1', lines)
        self.assertEqual(lines[-1].split()[0], 'M1')
        for line in lines[:-1]:
            self.assertTrue(line.startswith('C'))
