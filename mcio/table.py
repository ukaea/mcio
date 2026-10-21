"""Simple text table generation.

:class:`Table` renders rows of values as an aligned plain-text table, with
per-type number formatting controlled by :class:`TableFormat`.
"""

from __future__ import annotations

from typing import Any

import dataclasses
from dataclasses import dataclass
import numpy as np


@dataclass
class TableFormat:
    """Formatting options for :class:`Table`.

    Note the class-level values are not annotated, so they are plain class
    attributes rather than dataclass fields.

    Attributes
    ----------
    integer : str
        Format specifier for integer values.
    floating : str
        Format specifier for floating point values.
    string : str
        Format specifier for string values.
    string_wrap_length : int
        String wrap length (currently unused).
    header_padding : int
        Extra padding added to each column width.
    column_speration : int
        Number of separator characters between columns.
    column_seperator : str
        Character used to separate columns.
    """

    integer = ''
    floating = '.6e'
    string = ''
    string_wrap_length = 20
    header_padding = 2
    column_speration = 2
    column_seperator = ' '


class Table:
    """Plain-text table of values.

    Parameters
    ----------
    rows : list[list[Any]]
        Table rows; all rows must have equal length. Values may be str, int,
        float, bool or numpy scalars.
    headers : list[str] | None, optional
        Column headers, by default None (dummy ``value_i`` headers are used)
    format : TableFormat | None, optional
        Formatting options, by default None (a default :class:`TableFormat`)
    title : str | None, optional
        Title printed above the table, by default None

    Raises
    ------
    Exception
        If the rows, or the headers and rows, differ in length.
    """

    def __init__(self,
                 rows: list[list[Any]],
                 headers: list[str] | None = None,
                 format: TableFormat | None = None,
                 title: str | None = None) -> None:
        """Initialise the Table and check the shape of the inputs."""
        # store user data
        self._rows: list[list[Any]] = rows
        self._headers: list[str] | None = headers

        # store format
        self.format: TableFormat = TableFormat() if isinstance(format, type(None)) else format

        # store title
        self.title: str | None = title

        # check shape of rows and headers
        self._n_rows: int = len(rows)
        self._row_length: int = self._check_inputs()

    def _check_inputs(self) -> int:
        """Check the shape of the rows and headers provided.

        Returns
        -------
        int
            Length of each row.

        Raises
        ------
        Exception
            If the rows differ in length, or the headers differ from the row
            length.
        """
        self._row_length = len(self._rows[0])
        for row in self._rows[1:]:
            if len(row) != self._row_length:
                msg = f'length of rows in table must be equal'
                raise Exception(msg)
            
        if self._headers:
            if len(self._headers) != self._row_length:
                msg = f'length of headers in table must equal row length'
                raise Exception(msg)
            
        return self._row_length

    def _convert_rows_to_strings(self) -> list[list[str]]:
        """Convert the headers and row values to formatted strings.

        Returns
        -------
        list[list[str]]
            Headers (if any) followed by each row, with values formatted by type.
        """
        rows_output: list[list[str]] = [self._headers] if self._headers else []

        TYPE_FORMATS = self._type_formats()
        
        # convert raw values to strings
        for row in self._rows:
            # convert bools to strings of bools
            row = ['true' if (isinstance(v, bool) and v) else v for v in row]
            row = ['false' if (isinstance(v, bool) and not v) else v for v in row]

            # set up table row
            table_row = []

            for value in row:          
                value_as_string = f'{value:{TYPE_FORMATS[type(value)]}}'
                table_row.append(value_as_string)
            rows_output.append(table_row)

        return rows_output

    def _column_widths(self) -> list[int]:
        """Calculate the maximum string width of each column.

        Returns
        -------
        list[int]
            Width of each column, including headers.
        """
        column_widths = [0 for _ in range(self._row_length)]
        
        for i, row in enumerate(self._rows_as_strings):
            for j, col_value in enumerate(row):
                col_value_length = len(col_value)
                if col_value_length > column_widths[j]:
                    column_widths[j] = col_value_length

        if self._headers:
            for j, header_value in enumerate(self._headers):
                header_value_length = len(header_value)
                if header_value_length > column_widths[j]:
                    column_widths[j] = header_value_length

        return column_widths

    def _type_formats(self) -> dict[type, str]:
        """Return the format specifier lookup for each value type.

        Returns
        -------
        dict[type, str]
            Format specifier keyed by Python / numpy type.
        """
        type_formats: dict[type, str] = {str: self.format.string,
                        np.str_: self.format.string,
                        int: self.format.integer,
                        np.integer: self.format.integer,
                        float: self.format.floating,
                        np.floating: self.format.floating}

        return type_formats

    @property
    def as_string(self) -> str:
        """Table rendered as a string.

        Dummy headers are generated if none were provided.

        Returns
        -------
        str
            Formatted table, preceded by the title if set.
        """
        # add dummy headers if non provided by user
        if not self._headers:
            self._headers = [f'value_{i}' for i in range(self._row_length)]

        # convert all values and headers to strings applying formatting for each type
        self._rows_as_strings = self._convert_rows_to_strings()

        # calculate the required maximum colum width
        self._column_widths = self._column_widths()

        # set formatting types lookup
        TYPE_FORMATS = self._type_formats()

        # set up list to store output rows
        table_rows: list[list[str]] = []
        
        # add headers with column widths as list of strings
        table_row_header = [f'{value: <{self._column_widths[i] + self.format.header_padding}{TYPE_FORMATS[type(value)]}}' for i, value in enumerate(self._headers)]
        table_row_underscore = ['-' * (i + self.format.header_padding) for i in self._column_widths]
        table_rows.append(table_row_header)
        table_rows.append(table_row_underscore)

        # add all values with column widths as list of strings
        for row in self._rows:
            # check if bool value and convert bool to string
            row = ['true' if (isinstance(v, bool) and v) else v for v in row]
            row = ['false' if (isinstance(v, bool) and not v) else v for v in row]
            
            table_row_values = [f'{value: <{self._column_widths[i] + self.format.header_padding}{TYPE_FORMATS[type(value)]}}' for i, value in enumerate(row)]
            table_rows.append(table_row_values)
            # table_rows.append(table_row_underscore)

        # add column seperators
        column_sperators = self.format.column_seperator * self.format.column_speration
        rows_as_strings = [column_sperators.join(row) for row in table_rows]

        # return table as string
        table_string =  '\n'.join(rows_as_strings)

        # generate title
        title = '+' * len(self.title) + '\n' + self.title + '\n' + '+' * len(self.title) + '\n' + '\n' if self.title else ''

        return f'\n{title}{table_string}\n'
        
    def __str__(self) -> str:
        """Return the table as a string.

        Returns
        -------
        str
            Table string.
        """
        return self.string