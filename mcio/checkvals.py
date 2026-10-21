"""Value and type checking helpers.

Contains custom exceptions, numeric-to-string conversion helpers, option and
type validators, and decorators enforcing argument types on setter methods.
"""
from __future__ import annotations

import sys
import importlib
from numbers import Number
from decimal import Decimal
from fractions import Fraction
from typing import Any, Callable, Dict, List, Sequence, Type, Union


# define custom exceptions
class OptionException(Exception):
    """Exception raised when a provided option is not in the allowed options list."""


class TypeException(Exception):
    """Exception raised when a variable is not of an expected type."""


# define functions for checks
def isnumeric(var: Any) -> bool:
    """Check if a variable is of a numeric type.

    Parameters
    ----------
    var : Any
        Variable to check for numeric type

    Returns
    -------
    bool
        True if variable is an instance of Number, Decimal, or Fraction, False otherwise
    """
    return isinstance(var, (Number, Decimal, Fraction))


def number_to_string(var: Any, dp: int = 6) -> Union[str, Any]:
    """Convert a numeric value to a string with specified decimal precision.

    Parameters
    ----------
    var : Any
        Variable to convert to string
    dp : int, optional
        Number of decimal places for floating point formatting, by default 6

    Returns
    -------
    str or Any
        Formatted string representation of integer or float, or original variable if non-numeric

    Raises
    ------
    TypeError
        If var is numeric but not an instance of int or float
    """
    if not isnumeric(var):
        return var

    if isinstance(var, int):
        return f"{var}"

    if isinstance(var, float):
        return f"{var:.{dp}f}"

    # if is not numeric, the raise exception
    msg = (
        f"variable in list is neither str, int or float" f" and its type is {type(var)}"
    )
    raise TypeError(msg)


def numbers_to_strings(
    list_of_vars: Sequence[Any], dp: int = 6
) -> List[Union[str, Any]]:
    """Convert a sequence of numeric values to formatted strings.

    Parameters
    ----------
    list_of_vars : sequence of Any
        List or tuple of variables to convert
    dp : int, optional
        Number of decimal places for float formatting, by default 6

    Returns
    -------
    list of str or Any
        List of formatted string values or non-numeric pass-through items

    Raises
    ------
    TypeError
        If list_of_vars is not a list or tuple
    """
    if isinstance(list_of_vars, (list, tuple)):
        return [number_to_string(var, dp=dp) for var in list_of_vars]
    else:
        msg = (
            f"numbers_to_strings accepts only lists and tuples,"
            f" not {type(list_of_vars)}"
        )
        raise TypeError(msg)


def check_option(option: Any, options: Sequence[Any]) -> bool:
    """Validate that an option exists within a set of allowed options.

    Parameters
    ----------
    option : Any
        The option to validate
    options : sequence of Any
        Sequence of valid options

    Returns
    -------
    bool
        True if option is found in options

    Raises
    ------
    OptionException
        If option is not present in options
    """
    if option in options:
        return True
    else:
        msg = f"option: {option} not in options. Option must be one of {options}"
        raise OptionException(msg)


def check_types(
    variable: Any,
    variable_types: Sequence[Type],
    error_message: str = None,
    iterable: bool = False,
) -> None:
    """Validate that a variable matches one of the specified types.

    Parameters
    ----------
    variable : Any
        Variable to check
    variable_types : sequence of type
        Allowed types for the variable
    error_message : str, optional
        Custom error message if type check fails, by default None
    iterable : bool, optional
        Flag indicating whether elements of an iterable should be checked
        individually, by default False

    Raises
    ------
    TypeException
        If variable is not an instance of any type in variable_types
    """
    if error_message:
        msg = error_message
    else:
        types_available = "".join([f"{type(v)}" for v in variable_types])
        msg = f"type error: {str(type(variable))} not in {types_available}"

    if not isinstance(variable, tuple(variable_types)):
        raise TypeException(msg)


def check_module_loaded(module_name: str, global_vars: Dict[str, Any]) -> Any:
    """Ensure a module is imported and available in global variables.

    Parameters
    ----------
    module_name : str
        Name of the module to import/check
    global_vars : dict
        Global namespace dictionary (e.g., globals()) to populate

    Returns
    -------
    module
        Imported module object from sys.modules
    """
    if module_name not in sys.modules:
        module = importlib.import_module(module_name)
        global_vars[module_name] = module

    return sys.modules[module_name]


def require_int(func: Callable) -> Callable:
    """Decorator to enforce integer type on the first argument of a method.

    Parameters
    ----------
    func : callable
        Method to wrap

    Returns
    -------
    callable
        Wrapped method enforcing int parameter constraint

    Raises
    ------
    TypeError
        If decorated method receives a non-int argument
    """

    def wrapper(self, value: Any) -> Any:
        """Call ``func`` if ``value`` is an int, otherwise raise.

        Parameters
        ----------
        value : Any
            Value passed to the wrapped method.

        Returns
        -------
        Any
            Result of the wrapped method.

        Raises
        ------
        TypeError
            If ``value`` is not an int.
        """
        if isinstance(value, int):
            return func(self, value)
        else:
            msg = f"function requires int type, not {type(value)}"
            raise TypeError(msg)

    return wrapper


def require_numeric(func: Callable) -> Callable:
    """Decorator to enforce numeric type on the first argument of a method.

    Parameters
    ----------
    func : callable
        Method to wrap

    Returns
    -------
    callable
        Wrapped method enforcing numeric parameter constraint

    Raises
    ------
    TypeError
        If decorated method receives a non-numeric argument
    """

    def wrapper(self, value: Any) -> Any:
        """Call ``func`` if ``value`` is a Number, otherwise raise.

        Parameters
        ----------
        value : Any
            Value passed to the wrapped method.

        Returns
        -------
        Any
            Result of the wrapped method.

        Raises
        ------
        TypeError
            If ``value`` is not a Number.
        """
        if isinstance(value, Number):
            return func(self, value)
        else:
            msg = f"function requires numeric type, not {type(value)}"
            raise TypeError(msg)

    return wrapper


def require_path_like(func: Callable) -> Callable:
    """Decorator to enforce path-like/integer type on the first argument of a method.

    Parameters
    ----------
    func : callable
        Method to wrap

    Returns
    -------
    callable
        Wrapped method enforcing path-like parameter constraint

    Raises
    ------
    TypeError
        If decorated method receives an invalid argument type
    """

    def wrapper(self, value: Any) -> Any:
        """Call ``func`` if ``value`` is an int, otherwise raise.

        Parameters
        ----------
        value : Any
            Value passed to the wrapped method.

        Returns
        -------
        Any
            Result of the wrapped method.

        Raises
        ------
        TypeError
            If ``value`` is not an int.
        """
        if isinstance(value, int):
            return func(self, value)
        else:
            msg = f"function requires int type, not {type(value)}"
            raise TypeError(msg)

    return wrapper
