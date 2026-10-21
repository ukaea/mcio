"""Data sets for elements, isotopes and their natural abundances.

On import, NIST atomic weight and isotopic composition data and FISPACT group
structures are parsed from the ``data`` directory into module-level lookup
dictionaries.
"""

from __future__ import annotations

from typing import Any

import os
import re
import json

# set
ROOT_DIR: str = os.path.dirname(os.path.abspath(__file__))

# load NIST data
with open(os.path.join(ROOT_DIR, "data", "nist_data.txt")) as input_file:
    rows: list[str] = list(input_file)
    indices: list[int] = [-1, *[i for i, row in enumerate(rows) if row == "\n"]]

# set up storage for data
ELEMENT: dict[int, str] = {}
ATOMIC_NUMBER: dict[str, int] = {}
RELATIVE_ATOMIC_MASS: dict[str, float] = {}
ISOTOPIC_COMPOSITION_BY_ATOM_FRACTION: dict[str, dict[str, float]] = {}
ISOTOPIC_COMPOSITION_BY_WEIGHT_FRACTION: dict[str, dict[str, float]] = {}
ZAID_FROM_ISOTOPE: dict[str, str] = {}
ELEMENT_FROM_ISOTOPE: dict[str, str] = {}
RANK_ISOTOPES: dict[str, int] = {}

isotope_rank_counter: int = 1

# parse data into storage
for i_start, i_end in zip(indices[:-1], indices[1:]):
    data = [datum[:-1].split(" = ") for datum in rows[i_start + 1 : i_end]]

    # remove floating numbers in NIST data in form 1.2345(6#)
    strip_nist_uncertainty = re.compile(r"(\([0-9]+#*\))")

    # extract data
    atomic_number = int(data[0][1])
    element = data[1][1]
    mass_number = int(data[2][1])
    relative_atomic_mass = float(re.sub(strip_nist_uncertainty, "", data[3][1]))

    # assign isotopic compositions to elements
    if data[4][-1] == "":
        # case for not a naturally occuring isotope
        isotopic_composition: float | None = None
    else:
        # case for isotope being part of elements natural abundance
        isotopic_composition = float(re.sub(strip_nist_uncertainty, "", data[4][-1]))

    # define isotope
    isotope = f"{element}{mass_number}"

    # store extracted data
    ELEMENT[int(atomic_number)] = element
    ATOMIC_NUMBER[element] = atomic_number
    RELATIVE_ATOMIC_MASS[isotope] = relative_atomic_mass

    # store natural abundances by isoptic composition
    if (
        element not in ISOTOPIC_COMPOSITION_BY_ATOM_FRACTION
        and isotopic_composition is not None
    ):
        ISOTOPIC_COMPOSITION_BY_ATOM_FRACTION[element] = {isotope: isotopic_composition}
    elif isotopic_composition is not None:
        ISOTOPIC_COMPOSITION_BY_ATOM_FRACTION[element][isotope] = float(
            isotopic_composition
        )

    # store atomic and mass number as ZAID
    ZAID_FROM_ISOTOPE[isotope] = f"{atomic_number:>03}{mass_number:>03}"

    # store
    RANK_ISOTOPES[isotope] = isotope_rank_counter
    isotope_rank_counter += 1

ELEMENT_FROM_ISOTOPE = {
    isotope: "".join(c for c in isotope if not c.isnumeric())
    for isotope in RELATIVE_ATOMIC_MASS
}
ISOTOPE_FROM_ZAID: dict[str, str] = {zaid: isotope for isotope, zaid in ZAID_FROM_ISOTOPE.items()}

# calculate raltive atomic masses for each element
for element in ISOTOPIC_COMPOSITION_BY_ATOM_FRACTION:
    ram = 0.0
    for isotope, isotope_atom_fraction in ISOTOPIC_COMPOSITION_BY_ATOM_FRACTION[
        element
    ].items():
        ram += RELATIVE_ATOMIC_MASS[isotope] * isotope_atom_fraction
    RELATIVE_ATOMIC_MASS[element] = ram

# calculate the isotopic composition by weight fraction
for element, isotopic_composition in ISOTOPIC_COMPOSITION_BY_ATOM_FRACTION.items():
    ISOTOPIC_COMPOSITION_BY_WEIGHT_FRACTION[element] = {}
    for isotope, atom_fraction in isotopic_composition.items():
        ISOTOPIC_COMPOSITION_BY_WEIGHT_FRACTION[element][isotope] = (
            RELATIVE_ATOMIC_MASS[isotope]
            * atom_fraction
            / RELATIVE_ATOMIC_MASS[element]
        )

# load fispact group structures
with open(
    os.path.join(ROOT_DIR, "data", "fispact_group_structures.json")
) as input_file:
    fispact_data = json.load(input_file)
    GROUP_STRUCTURES: dict[int, Any] = {
        int(group_structure): group_data
        for group_structure, group_data in fispact_data.items()
    }
