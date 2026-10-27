"""
unittest: ProgressRegister - counting of tracked items and completion.
"""

import unittest

from mcio.register import ProgressRegister


from unitutils import CustomAssertions


class MCIOUnitTestRegister(unittest.TestCase, CustomAssertions):
    def setUp(self):
        self.items = ['solid_a', 'solid_b', 'solid_c']
        self.register = ProgressRegister(self.items, show_progress=False)

    def test_initial_state(self):
        self.assertEqual(self.register.n_items_to_process, 3)
        self.assertEqual(self.register._register, {'solid_a': 1, 'solid_b': 1, 'solid_c': 1})
        self.assertEqual(self.register._n_complete, 0)
        self.assertIsNone(self.register._progress)

    def test_remove_completes_item(self):
        self.register.remove(['solid_a'])
        self.assertEqual(self.register._register['solid_a'], 0)
        self.assertEqual(self.register._n_complete, 1)

    def test_add_then_remove(self):
        # an item split into two parts needs both parts removed to complete
        self.register.add(['solid_a'])
        self.register.remove(['solid_a'])
        self.assertEqual(self.register._n_complete, 0)
        self.register.remove(['solid_a'])
        self.assertEqual(self.register._n_complete, 1)

    def test_add_to_completed_item_reopens_it(self):
        self.register.remove(['solid_b'])
        self.assertEqual(self.register._n_complete, 1)
        self.register.add(['solid_b'])
        self.assertEqual(self.register._n_complete, 0)
        self.assertEqual(self.register._register['solid_b'], 1)

    def test_add_and_remove_multiple_items(self):
        self.register.add(['solid_a', 'solid_a', 'solid_c'])
        self.assertEqual(self.register._register, {'solid_a': 3, 'solid_b': 1, 'solid_c': 2})
        self.register.remove(['solid_a', 'solid_a', 'solid_a', 'solid_b'])
        self.assertEqual(self.register._n_complete, 2)

    def test_all_items_complete(self):
        self.register.remove(self.items)
        self.register.update()
        self.assertEqual(self.register._n_complete, self.register.n_items_to_process)
        self.assertIsNone(self.register._progress)

    def test_unknown_item_raises(self):
        with self.assertRaises(KeyError):
            self.register.remove(['not_tracked'])
        with self.assertRaises(KeyError):
            self.register.add(['not_tracked'])

    def test_hashable_items(self):
        register = ProgressRegister([1, (2, 3), 'four'], show_progress=False)
        register.remove([(2, 3)])
        self.assertEqual(register._register[(2, 3)], 0)
        self.assertEqual(register._n_complete, 1)

    def test_progress_bar_stops_when_complete(self):
        register = ProgressRegister(['a', 'b'], show_progress=True, title='unittest')
        try:
            self.assertIsNotNone(register._progress)
            register.remove(['a'])
            register.update()
            self.assertIsNotNone(register._progress)
            register.remove(['b'])
            register.update()
            self.assertIsNone(register._progress)
        finally:
            if register._progress is not None and not register._is_notebook:
                register._progress.stop()

    def test_str_lists_items_and_counts(self):
        self.register.add(['solid_c'])
        string = str(self.register)
        for item in self.items:
            self.assertIn(item, string)
        self.assertIn('count', string)
        self.assertIn('2', string)
