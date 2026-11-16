"""
unittest: checkvals - numeric conversion, option and type validation, and the
argument type decorators.
"""

import pathlib
import unittest
from decimal import Decimal

from mcio.checkvals import (
    OptionException,
    TypeException,
    check_module_loaded,
    check_option,
    check_types,
    isnumeric,
    number_to_string,
    numbers_to_strings,
    require_int,
    require_numeric,
    require_path_like,
)


class Setter:
    @require_int
    def set_int(self, value):
        """set an int"""
        return value

    @require_numeric
    def set_numeric(self, value):
        return value

    @require_path_like
    def set_path(self, value):
        return value


class MCIOUnitTestCheckVals(unittest.TestCase):
    def test_isnumeric(self):
        for value in [1, 1.5, Decimal("1.5"), True]:
            self.assertTrue(isnumeric(value), value)
        for value in ["1", None, [1]]:
            self.assertFalse(isnumeric(value), value)

    def test_number_to_string(self):
        self.assertEqual(number_to_string(3), "3")
        self.assertEqual(number_to_string(1.5), "1.500000")
        self.assertEqual(number_to_string(1.5, dp=2), "1.50")
        # non numeric values are passed through unchanged
        self.assertEqual(number_to_string("j"), "j")

    def test_number_to_string_rejects_other_numeric_types(self):
        with self.assertRaises(TypeError):
            number_to_string(Decimal("1.5"))

    def test_numbers_to_strings(self):
        self.assertEqual(numbers_to_strings([1, 2.0, "j"], dp=1), ["1", "2.0", "j"])
        self.assertEqual(numbers_to_strings((1,)), ["1"])
        with self.assertRaises(TypeError):
            numbers_to_strings({1, 2})

    def test_check_option(self):
        self.assertTrue(check_option("ao", ["ao", "wo"]))
        with self.assertRaises(OptionException):
            check_option("xx", ["ao", "wo"])

    def test_check_types(self):
        check_types({}, [dict])
        check_types(1, [int, float])
        with self.assertRaises(TypeException):
            check_types([], [dict])

    def test_check_types_message_names_the_types(self):
        with self.assertRaises(TypeException) as context:
            check_types("a", [int, float])
        self.assertEqual(str(context.exception), "type error: str not in (int, float)")

    def test_check_types_custom_message(self):
        with self.assertRaises(TypeException) as context:
            check_types("a", [int], error_message="needs an int")
        self.assertEqual(str(context.exception), "needs an int")

    def test_check_types_iterable_checks_each_element(self):
        check_types([1, 2.0], [int, float], iterable=True)
        with self.assertRaises(TypeException):
            check_types([1, "a"], [int, float], iterable=True)

    def test_check_module_loaded(self):
        global_vars = {}
        module = check_module_loaded("json", global_vars)
        self.assertEqual(module.__name__, "json")

    def test_require_int(self):
        setter = Setter()
        self.assertEqual(setter.set_int(2), 2)
        with self.assertRaises(TypeError):
            setter.set_int(2.0)

    def test_require_numeric(self):
        setter = Setter()
        self.assertEqual(setter.set_numeric(2.5), 2.5)
        with self.assertRaises(TypeError):
            setter.set_numeric("2.5")

    def test_require_path_like(self):
        setter = Setter()
        self.assertEqual(setter.set_path("model.stp"), "model.stp")
        self.assertEqual(
            setter.set_path(pathlib.Path("model.stp")), pathlib.Path("model.stp")
        )
        with self.assertRaises(TypeError):
            setter.set_path(3)

    def test_decorators_keep_method_name_and_docstring(self):
        self.assertEqual(Setter.set_int.__name__, "set_int")
        self.assertEqual(Setter.set_int.__doc__, "set an int")


if __name__ == "__main__":
    unittest.main()
