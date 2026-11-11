"""
unittest: Serpent2 text building blocks - section headers, card comment headers
and cards with wrapped rows and sub cards. (Writing a full Serpent2 model is not
yet implemented.)
"""

import unittest

from mcio.serpent2 import (
    SERPENT_INDENT,
    SERPENT_MAX_CHARS,
    SerpentCard,
    SerpentCardHeader,
    SerpentHeader,
)


class MCIOUnitTestSerpent2(unittest.TestCase):
    def test_header_banner(self):
        lines = SerpentHeader('SURFACES').as_string.splitlines()
        self.assertEqual(len(lines), 3)
        for line in lines:
            self.assertTrue(line.startswith('%'))
            self.assertEqual(len(line), SERPENT_MAX_CHARS)
        self.assertIn(' SURFACES ', lines[1])

    def test_header_data_pairs(self):
        text = SerpentHeader('SUMMARY', {'cells': 3, 'small': 1e-6, 'width': 2.5}).as_string
        self.assertIn('% === cells : 3', text)
        self.assertIn('% === small : 1.0000e-06', text)
        self.assertIn('% === width : 2.5000', text)

    def test_card_header_uses_serpent_comments(self):
        # regression: comments were written with the MCNP comment card 'C'
        text = SerpentCardHeader({'name': 'sphere', 'cell': 4}).as_string
        self.assertEqual(text, '%\n% name: sphere\n% cell: 4\n%\n')

    def test_card_rows_and_indent(self):
        card = SerpentCard('surf 1 px', [['1.0']])
        # entries start one space after the designator, padded to SERPENT_INDENT
        self.assertEqual(card.as_string, f"{'surf 1 px':<{SERPENT_INDENT}} 1.0\n")

    def test_card_add_row_starts_new_line(self):
        card = SerpentCard('cell 1', [['0', 'fill']])
        card.add_row(['-1', '2'])
        lines = card.as_string.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[1].strip(), '-1 2')
        self.assertTrue(lines[1].startswith(' ' * SERPENT_INDENT))

    def test_long_row_wrapped_below_max_chars(self):
        card = SerpentCard('cell 1', [['12345'] * 60])
        lines = card.as_string.splitlines()
        self.assertGreater(len(lines), 1)
        for line in lines:
            self.assertLessEqual(len(line), SERPENT_MAX_CHARS)
        self.assertEqual(card.as_string.split()[2:].count('12345'), 60)

    def test_header_data_and_sub_cards(self):
        card = SerpentCard('surf 1 px', [['1.0']], header_data={'name': 's1'})
        card.add_subcard(SerpentCard('surf 2 py', [['2.0']]))
        text = card.as_string
        self.assertIn('% name: s1', text)
        self.assertLess(text.index('surf 1 px'), text.index('surf 2 py'))


if __name__ == '__main__':
    unittest.main()
