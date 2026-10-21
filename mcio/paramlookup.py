"""Base class for parameter collections that can be tabulated and saved to json.

Subclasses (e.g. :class:`mcio.formatting.Formatting` and
``DecompositionParameters``) are slotted dataclasses where each parameter
``name`` has a companion ``_def_name`` (or, for older classes, ``_DEF_NAME``)
slot holding its description.
"""

from __future__ import annotations

from typing import Any

import dataclasses
import json
import warnings


from .table import Table


class ParameterLookup:
    """Mixin providing tabulated printing and json read/write of parameters.

    Subclasses are expected to define ``__slots__`` containing parameter names
    and their ``_DEF_<name>`` descriptions, and to populate ``_titles`` and
    ``_rows_lookup`` keyed by the first word of the parameter names.

    Attributes
    ----------
    _titles : dict[str, str] or None
        Table title for each parameter category.
    _rows_lookup : dict[str, list[list[Any]]] or None
        Table rows collected for each parameter category.
    """

    def __init__(self) -> None:
        """Initialise the empty title and row lookups."""
        self._titles: dict[str, str] | None = None
        self._rows_lookup: dict[str, list[list[Any]]] | None = None

    def _definitions(self) -> list[tuple[str, str]]:
        """Return each documented parameter and the slot holding its definition.

        Returns
        -------
        list[tuple[str, str]]
            ``(parameter name, definition slot name)`` pairs, for every
            ``_def_<name>`` / ``_DEF_<NAME>`` slot whose parameter exists.
        """
        pairs: list[tuple[str, str]] = []
        for slot in self.__slots__:
            if slot[:5].lower() == '_def_':
                parameter = slot[5:]
                if parameter in self.__slots__:
                    pairs.append((parameter, slot))
        return pairs

    def __str__(self) -> str:
        """Return the parameters, values and descriptions as text tables.

        Returns
        -------
        str
            One table per parameter category, concatenated.
        """
        headings = ['variable', 'value', 'description']

        # collect rows per category (first word of the name, upper cased)
        rows_lookup: dict[str, list[list[Any]]] = {key: [] for key in self._rows_lookup}
        for parameter, definition in self._definitions():
            category = parameter.split('_')[0].upper()
            row = [parameter, getattr(self, parameter), getattr(self, definition)]
            rows_lookup[category if category in rows_lookup else 'OTHERS'].append(row)

        tables = [
            Table(rows, headers=headings, title=self._titles[category])
            for category, rows in rows_lookup.items()
            if rows
        ]
        return ''.join([table.as_string for table in tables])

    def write(self, output_path: str) -> None:
        """Write the parameter values to a json file.

        Parameters
        ----------
        output_path : str
            Path of the json file to write.
        """
        data: dict[str, Any] = {
            parameter: getattr(self, parameter) for parameter, _ in self._definitions()
        }
        with open(output_path, 'w') as output_file:
            json.dump(data, output_file, indent=4)

    @classmethod
    def read(cls, input_path: str) -> ParameterLookup:
        """Read parameter values from a json file into a new instance.

        Keys are matched case-insensitively against the parameter names, so
        files written before the lower case renaming can still be read.

        Parameters
        ----------
        input_path : str
            Path of the json file to read.

        Returns
        -------
        ParameterLookup
            New instance with the values from the file.

        Raises
        ------
        KeyError
            If the file contains an unknown parameter.
        """
        params = cls()
        names = {field.name.lower(): field.name for field in dataclasses.fields(cls)}

        with open(input_path, 'r') as input_file:
            data = json.load(input_file)

        for key, value in data.items():
            if key.lower() not in names:
                raise KeyError(f"unknown parameter '{key}' in {input_path}")
            name = names[key.lower()]
            if name != key:
                warnings.warn(
                    f"parameter '{key}' in {input_path} is deprecated, use '{name}'",
                    DeprecationWarning,
                    stacklevel=2,
                )
            setattr(params, name, value)

        return params
