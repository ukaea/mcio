"""Numerical utility functions for MCIO.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Union


def custom_round(val: Union[float, int], digits: int) -> float:
    """Round a numerical value to a specified number of decimal places using half-up rounding.

    This function formats the input number to a high-precision decimal representation
    before rounding with ``ROUND_HALF_UP`` (ties round away from zero), avoiding
    common binary floating-point representation artifacts.

    Parameters
    ----------
    val : float or int
        Numerical value to round.
    digits : int
        Number of decimal places to round to.

    Returns
    -------
    float
        The rounded floating-point value.

    Examples
    --------
    >>> custom_round(2.5, 0)
    3.0
    >>> custom_round(2.55, 1)
    2.6
    """
    # Format to high precision float string first to avoid binary representation artifacts
    d = Decimal(f"{val:.16f}")
    return float(d.quantize(Decimal(10) ** -digits, rounding=ROUND_HALF_UP))
1