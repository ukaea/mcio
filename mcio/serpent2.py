"""Serpent 2 input deck card and header builders.

Contains small helper classes used to format Serpent 2 input: section header
banners, comment headers and generic cards with automatic line wrapping.
"""
from __future__ import annotations

from typing import Any

from .formatting import (
    SERPENT_INDENT,
    SERPENT_FMT_SURFACE_CARTESIAN,
    SERPENT_FMT_SURFACE_RADIUS,
    SERPENT_FMT_SURFACE_PLANE_NORMAL,
    SERPENT_FMT_SURFACE_PLANE_DISPLACEMENT,
    SERPENT_FMT_SURFACE_T_SQ,
    SERPENT_FMT_SURFACE_ELIPSOID,
    SERPENT_FMT_SURFACE_GENERAL_QUADRATIC,
    SERPENT_FMT_SURFACE_TORI,
    SERPENT_MAX_CHARS
)

class SerpentHeader:
    """Section header banner for a Serpent 2 input file.

    Parameters
    ----------
    header : str
        Title text placed in the banner line.
    data_pairs : dict, optional
        Label/value pairs listed as a summary beneath the banner, by default None
    banner : list[str], optional
        Additional banner lines (stored but currently unused), by default None
    """

    def __init__(self, header: str, data_pairs: dict[str, Any] | None=None,
                 banner: list[str] | None=None) -> None:
        """Initialise the SerpentHeader."""
        # store user provided parameters
        self._header: str = header
        self._data_pairs: dict[str, Any] | None = data_pairs
        self._banner: list[str] | None = banner

    @property
    def as_string(self) -> str:
        """Return the header formatted as Serpent comment lines.

        Floats are written with 4 decimal places, or in scientific notation
        when smaller than 1e-4.

        Returns
        -------
        str
            The formatted header block, newline terminated.
        """
        line_break = '% ' + '-' * (SERPENT_MAX_CHARS - 2) + '\n'

        header = line_break[:SERPENT_INDENT] + f' {self._header} '
        header += line_break[SERPENT_INDENT + len(self._header) + 2:]

        output = ''.join([line_break, header, line_break])

        if self._data_pairs:
            LINE_INDENT = '% ' + ('=' * 3)
            LINE_SUMMARY = LINE_INDENT + '\n'
            output += LINE_SUMMARY
            max_label_length = max([len(label) for label in self._data_pairs])
            for data_label, data in self._data_pairs.items():
                if type(data) == float and data < 1e-4:
                    line_data = LINE_INDENT + f' {data_label:<{max_label_length}} : {data:.4e} \n'
                elif type(data) == float:
                    line_data = LINE_INDENT + f' {data_label:<{max_label_length}} : {data:.4f} \n'
                else:
                    line_data = LINE_INDENT + f' {data_label:<{max_label_length}} : {data} \n'
                output += line_data

            output += LINE_SUMMARY
            output += line_break

        return output

class SerpentCardHeader:
    """Comment block written above a Serpent card.

    Parameters
    ----------
    data : dict
        Key/value pairs written as one comment line each.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialise the SerpentCardHeader."""
        self._data: dict[str, Any] = data

    @property
    def as_string(self) -> str:
        """Return the comment block as a string.

        Returns
        -------
        str
            Comment lines of the form ``C key: value`` framed by ``C`` lines.
        """
        comments = ''.join([f'C {k}: {v}\n' for k, v in self._data.items()])
        return 'C\n' + comments + 'C\n'


class SerpentCard:
    """Generic Serpent 2 input card with automatic line wrapping.

    Parameters
    ----------
    designator : str
        Leading card text (e.g. ``surf 1 px``), padded to ``SERPENT_INDENT``.
    rows : list[list[str]]
        Rows of pre-formatted entries; each row starts on a new line.
    subcards : list[SerpentCard], optional
        Cards appended after this card's output, by default None
    header_data : dict, optional
        Data written as a comment header above the card, by default None
    """

    def __init__(self, designator: str, rows: list[list[str]],
                 subcards: list[SerpentCard] | None=None,
                 header_data: dict[str, Any] | None=None) -> None:
        """Initialise the SerpentCard."""
        self._designator: str = designator
        self._rows: list[list[str]] = rows
        self._subcards: list[SerpentCard] = subcards if subcards else []
        self._header_data: dict[str, Any] | None = header_data

    def add_row(self, row: list[str]) -> None:
        """Append a row of entries to the card.

        Parameters
        ----------
        row : list[str]
            Pre-formatted entries for the new row.
        """
        self._rows.append(row)

    def add_subcard(self, subcard: SerpentCard) -> None:
        """Append a sub-card written after this card.

        Parameters
        ----------
        subcard : SerpentCard
            Card to append.
        """
        self._subcards.append(subcard)

    @property
    def as_string(self) -> str:
        """Return the card formatted as Serpent input text.

        Entries are wrapped onto continuation lines (indented by
        ``SERPENT_INDENT``) so lines stay below ``SERPENT_MAX_CHARS``.

        Returns
        -------
        str
            The card text including any header comments and sub-cards.
        """
        rows = []
        string = f'{self._designator:<{SERPENT_INDENT}}'

        for i, row in enumerate(self._rows):
            for j, entry in enumerate(row):
                if (len(string) + len(entry) + 1) < SERPENT_MAX_CHARS:
                    string += f' {entry}'
                else:
                    rows.append(f'{string}\n')
                    string =  ' ' * (SERPENT_INDENT) + f' {entry}'
            rows.append(string + '\n')
            string = ' ' * (SERPENT_INDENT)

        output_string = ''.join(rows)

        # add header comments
        if self._header_data is not None:
            mcnp_card_header = SerpentCardHeader(self._header_data)
            output_string = mcnp_card_header.as_string + output_string

        # add sub cards
        for subcard in self._subcards:
            output_string += subcard.as_string

        return output_string
