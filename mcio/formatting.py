"""Formatting module contains classes which control the formatting of text
based output writers.

:class:`Formatting` holds the numerical formatting parameters for MCNP and
ADVANTG outputs (a module-level default instance is provided as ``formatting``),
and module-level ``SERPENT_*`` constants hold the Serpent2 equivalents.
"""

from __future__ import annotations

import dataclasses

from .paramlookup import ParameterLookup


# formatting for mcnp outputs
class MCNPFormat:
    """Basic MCNP text formatting settings (line length, indent, decimal places)."""

    def __init__(self) -> None:
        """Initialise the default MCNP formatting settings."""
        self._max_chars: int = 80
        self._indent: int = 8
        self._decimal_places: int = 6

    @property
    def max_chars(self) -> int:
        """Maximum number of characters per line.

        Returns
        -------
        int
            Maximum line length.
        """
        return self.max_chars
    
    @max_chars.setter
    def max_chars(self, max_chars: int) -> None:
        """Set the maximum number of characters per line.

        Parameters
        ----------
        max_chars : int
            Maximum line length.
        """
        self._max_chars = max_chars

    @property
    def indent(self) -> int:
        """Number of spaces used to indent continuation lines.

        Returns
        -------
        int
            Indent width.
        """
        return self._indent

    @indent.setter
    def indent(self, indent: int) -> None:
        """Set the indent width.

        Parameters
        ----------
        indent : int
            Indent width.
        """
        self._indent = indent

    @property
    def dp(self) -> int:
        """Number of decimal places for numerical values.

        Returns
        -------
        int
            Number of decimal places.
        """
        return self._decimal_places

    @dp.setter
    def dp(self, dp: int) -> None:
        """Set the number of decimal places.

        Parameters
        ----------
        dp : int
            Number of decimal places.
        """
        self._decimal_places = dp

@dataclasses.dataclass(slots=True)
class Formatting(ParameterLookup):
    """Numerical formatting parameters for text based output writers.

    Each parameter ``NAME`` has a companion ``_DEF_NAME`` field holding its
    description, used when tabulating the parameters (see
    :class:`mcio.paramlookup.ParameterLookup`). ``MCNP_FMT_*`` / ``*_FORMAT_*``
    values are Python format specifiers (e.g. ``'.15e'``).
    """
    MCNP_DP: int = 6
    _DEF_MCNP_DP: str = 'number of decimal places'
    MCNP_EXP = 8
    _DEF_MCNP_EXP: str = 'exponent form'
    MCNP_MAX_CHARS: int = 80
    _DEF_MCNP_MAX_CHARS: str = 'mcnp input file line length'
    MCNP_INDENT: int = 8
    _DEF_MCNP_INDENT: str = 'mcnp card indent'
    MCNP_FORMAT_TRANSFORM_LATTERAL = '.12e'
    _DEF_MCNP_FORMAT_TRANSFORM_LATTERAL: str = ''
    MCNP_FORMAT_TRANSFORM_ROTATION = '.12e'
    _DEF_MCNP_FORMAT_TRANSFORM_ROTATION: str = ''
    MCNP_FMT_SURFACE_CARTESIAN = '.15e'
    _DEF_MCNP_FMT_SURFACE_CARTESIAN: str = 'mcnp surface cartesian numerical formatting'
    MCNP_FMT_SURFACE_RADIUS = '.15e'
    _DEF_MCNP_FMT_SURFACE_RADIUS: str = ''
    MCNP_FMT_SURFACE_PLANE_NORMAL = '.15e'
    _DEF_MCNP_FMT_SURFACE_PLANE_NORMAL: str = ''
    MCNP_FMT_SURFACE_PLANE_DISPLACEMENT = '.15e'
    _DEF_MCNP_FMT_SURFACE_PLANE_DISPLACEMENT: str = ''
    MCNP_FMT_SURFACE_T_SQ = '.10e'
    _DEF_MCNP_FMT_SURFACE_T_SQ: str = ''
    MCNP_FMT_SURFACE_CONE_SHEET = ''
    _DEF_MCNP_FMT_SURFACE_CONE_SHEET: str = ''
    MCNP_FMT_SURFACE_ELIPSOID = '.15e'
    _DEF_MCNP_FMT_SURFACE_ELIPSOID: str = ''
    MCNP_FMT_SURFACE_GENERAL_QUADRATIC = '.15e'
    _DEF_MCNP_FMT_SURFACE_GENERAL_QUADRATIC: str = ''
    MCNP_FMT_SURFACE_TORI = '.15e'
    _DEF_MCNP_FMT_SURFACE_TORI: str = ''
    MCNP_FMT_MATERIAL_DENSITY = '.8e'
    _DEF_MCNP_FMT_MATERIAL_DENSITY: str = ''
    MCNP_FMT_FRACTION = '.8e'
    _DEF_MCNP_FMT_FRACTION: str = ''

    ADVANTG_DP = 6
    _DEF_ADVANTG_DP: str = 'advantg numerical decimal places'
    ADVANTG_EXP = 4
    _DEF_ADVANTG_EXP: str = 'advantg numerical exponent formatting'
    ADVANTG_MAX_CHARS = 80
    _DEF_ADVANTG_MAX_CHARS: str = 'advantg max character line width'
    ADVANTG_INDENT = 40
    _DEF_ADVANTG_INDENT: str = 'advantg indent for parameters and tabulated values'

    def __post_init__(self) -> None:
        """Set up the table row lookup and titles for each parameter category."""
        # define sub category string (first capitalised word in parameter + definition)
        self._rows_lookup = {'MCNP': [],
                             'SERPENT': [],
                             'ADVANTG': []}
        
        # table titles
        self._titles = {'MCNP': 'mcnp formatting',
                        'SERPENT': 'serpent2 formatting',
                        'ADVANTG': 'advantg formatting'}


# set up default values
formatting: Formatting = Formatting()

SERPENT_DP: int = 6
SERPENT_EXP: int = 4
SERPENT_MAX_CHARS: int = 120
SERPENT_INDENT: int = 20
SERPENT_FORMAT_TRANSFORM_LATTERAL: str = '.12e'
SERPENT_FORMAT_TRANSFORM_ROTATION: str = '.12f'

SERPENT_FMT_SURFACE_CARTESIAN: str = '.10e'
SERPENT_FMT_SURFACE_RADIUS: str = '.10e'
SERPENT_FMT_SURFACE_PLANE_NORMAL: str = '.10e'
SERPENT_FMT_SURFACE_PLANE_DISPLACEMENT: str = '.8e'
SERPENT_FMT_SURFACE_T_SQ: str = '.10e'
SERPENT_FMT_SURFACE_CONE_SHEET: str = ''
SERPENT_FMT_SURFACE_ELIPSOID: str = '.15e'
SERPENT_FMT_SURFACE_GENERAL_QUADRATIC: str = '.15e'
SERPENT_FMT_SURFACE_TORI: str = '.15e'

SERPENT_FMT_MATERIAL_DENSITY: str = '.8e'
SERPENT_FMT_FRACTION: str = '.8e'
