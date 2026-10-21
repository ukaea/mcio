"""Universe and universe fill definitions used to group and place cells."""

from __future__ import annotations

from .transform import Transform


class Universe:
    """A universe: a numbered collection of cells that can fill other cells.

    Parameters
    ----------
    number : int, optional
        Universe identification number, by default None
    transform : Transform, optional
        Transform applied to the universe, by default None
    """

    def __init__(self, number: int | None = None, transform: Transform | None = None) -> None:
        """Initialise the Universe."""
        self.number: int | None = number
        self.transform: Transform | None = transform


class UniverseFill():
    """Specification of a universe used to fill a cell, with an optional transform.

    Parameters
    ----------
    universe : Universe
        Universe used as the fill.
    transform : Transform, optional
        Transform applied to the filling universe, by default None
    """

    def __init__(self, universe: Universe, transform: Transform | None = None) -> None:
        """Initialise the UniverseFill."""
        self.universe: Universe = universe
        self.transform: Transform | None = transform

