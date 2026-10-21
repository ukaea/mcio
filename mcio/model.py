"""The mcio Model: a collection of cells with tallies, source and run settings.

Contains the :class:`Model` class, which numbers cells, surfaces, materials,
transforms, universes and tallies, and writes the geometry as MCNP, Serpent2,
OpenMC or GDML input.
"""
from __future__ import annotations

import datetime
import os
import time
from typing import TYPE_CHECKING, Any

try:
    import openmc
except:
    pass

from mcio import logger
from .cell import Cell
from .formatting import formatting
from .macrobody import Macrobody
from .material import Materials
from .mcnp import MCNPHeader, MCNPRunParameters
from .serpent2 import SerpentHeader
from .source import Point
from .surface import Surface, _sort_surfaces_by_type
from .tally import Tally
from .utils import check_folder_exists_for_file_path
from .xml import XMLTag

if TYPE_CHECKING:
    import openmc
    from .material import Material
    from .source import Source
    from .transform import Transform
    from .universe import Universe


class Model:
    """A model made of cells, with tallies, a source and run parameters.

    On creation, unique surfaces, macrobodies, materials, transforms and
    universes are collected from the cells and all items are numbered.

    Parameters
    ----------
    cells : list[Cell], optional
        Cells in the model, by default None
    tallies : list[Tally], optional
        Tallies in the model, by default None (no tallies)
    source : Source, optional
        Particle source, by default None (a default :class:`Point` source)
    mcnp_run_parameters : MCNPRunParameters, optional
        MCNP run parameters, by default None (default
        :class:`MCNPRunParameters`)
    cell_number_min : int, optional
        Lowest cell number to assign, by default None (1)
    surface_number_min : int, optional
        Lowest surface number to assign, by default None (1)
    tally_number_min : int, optional
        Lowest tally number to assign, by default None (1)
    keep_surface_numbers : bool, optional
        Stored flag for keeping existing surface numbers (currently unused),
        by default True
    keep_cell_numbers : bool, optional
        Stored flag for keeping existing cell numbers (currently unused), by
        default True
    summary_data : dict[str, Any], optional
        Extra label/value pairs for the MCNP summary header, by default None
    """

    def __init__(self, cells: list[Cell] | None = None, tallies: list[Tally] | None = None, source: Source | None = None, 
                 mcnp_run_parameters: MCNPRunParameters | None = None, cell_number_min: int | None = None, surface_number_min: int | None = None, 
                 tally_number_min: int | None = None, keep_surface_numbers: bool = True,
                 keep_cell_numbers: bool = True, summary_data: dict[str, Any] | None = None) -> None:
        """Initialise the Model and assign numbers."""
        self.cells: list[Cell] = cells
        self._tallies: list[Tally] = tallies if tallies else []
        self.source: Source = source if source else Point()
        self._surfaces: dict[str, Surface] = {}
        self._macrobodies: set[Macrobody] = set()
        self._materials: dict[str, Material] = {}
        self._transforms: set[Transform] = set()
        self._universes: set[Universe] = set()
        self.mcnp: MCNPRunParameters = mcnp_run_parameters if mcnp_run_parameters else MCNPRunParameters()
        self._summary_data: dict[str, Any] = summary_data if summary_data else {}
        self._cell_number_min: int = cell_number_min if cell_number_min else 1
        self._surface_number_min: int = surface_number_min if surface_number_min else 1
        self._tally_number_min: int = tally_number_min if tally_number_min else 1
        self._keep_surface_numbers: bool = keep_surface_numbers
        self._keep_cell_numbers: bool = keep_cell_numbers

        # set up default formatting
        self.formatting = formatting

        # assign material numbers
        self._asign_numbers()

        self._openmc_model: openmc.model.Model | None = None

    @property
    def cell_number_min(self) -> int:
        """Lowest cell number assigned.

        Returns
        -------
        int
            Minimum cell number.
        """
        return self._cell_number_min
    
    @cell_number_min.setter
    def cell_number_min(self, cell_number_min: int) -> None:
        """Set the lowest cell number and renumber if the model has cells.

        Parameters
        ----------
        cell_number_min : int
            Minimum cell number.
        """
        self._cell_number_min = cell_number_min
        if self.cells:
            self._asign_numbers()

    @property
    def surface_number_min(self) -> int:
        """Lowest surface number assigned.

        Returns
        -------
        int
            Minimum surface number.
        """
        return self._surface_number_min
    
    @surface_number_min.setter
    def surface_number_min(self, surface_number_min: int) -> None:
        """Set the lowest surface number and renumber if the model has cells.

        Parameters
        ----------
        surface_number_min : int
            Minimum surface number.
        """
        self._surface_number_min = surface_number_min
        if self.cells:
            self._asign_numbers()

    @property
    def tallies(self) -> list[Tally]:
        """Tallies in the model.

        Returns
        -------
        list[Tally]
            The tallies.
        """
        return self._tallies

    @tallies.setter
    def tallies(self, tallies: list[Tally]) -> None:
        """Set the tallies and renumber them.

        Parameters
        ----------
        tallies : list[Tally]
            The tallies.
        """
        self._tallies = tallies
        self._asign_numbers_tallies()

    @property
    def materials(self) -> Materials:
        """Materials collected from the model cells.

        Returns
        -------
        Materials
            Collection wrapping the internal material dictionary.
        """
        return Materials(self._materials)
    
    def add_materials(self, materials: Any) -> None:
        """Add materials to the model and renumber.

        Parameters
        ----------
        materials : Any
            Materials to add with ``+=`` to the internal material dictionary.
        """
        self._materials += materials
        self._asign_numbers()

    def add_cells(self, cells: list[Cell]) -> None:
        """Add cells to the model and renumber everything.

        Parameters
        ----------
        cells : list[Cell]
            Cells to add.
        """
        # add cells to model
        self.cells.extend(cells)

        # re-assign numbers to cells, surfaces and tallies
        self._asign_numbers()
    
    def _asign_numbers_tallies(self) -> None:
        """Number tallies sequentially from the minimum tally number."""
        # assign tally numbers
        next_tally_number = self._tally_number_min
        for tally in self._tallies:
            tally.number = next_tally_number
            next_tally_number += 1

    def _asign_numbers(self) -> None:
        """Collect unique items from the cells and assign numbers.

        Cells keep valid unique numbers, otherwise they are numbered from the
        minimum cell number. Duplicate surfaces are replaced, and unique
        surfaces, macrobodies, materials, transforms and universes are
        collected and numbered. Surfaces are numbered (sorted by type) before
        macrobodies; materials keep existing numbers. Tallies are renumbered.
        """
        # iterate over Cell instances and collect all instances cells hold
        next_cell_number = self._cell_number_min

        cell_numbers_assigned: set[int] = set()
        for cell in self.cells:
            # assign cell numbers, keep if already assigned
            if cell.number and cell.number < self._cell_number_min:
                cell.number = next_cell_number
                cell_numbers_assigned.add(cell.number)
            elif cell.number and cell.number not in cell_numbers_assigned:
                # case where cell has number assigned and not already used
                cell_numbers_assigned.add(cell.number)
            elif cell.number and cell.number in cell_numbers_assigned:
                # case where cell number already assigned, not need to re-asign
                cell.number = next_cell_number
                next_cell_number += 1
                cell_numbers_assigned.add(cell.number)
            else:
                cell.number = next_cell_number
                next_cell_number += 1
                cell_numbers_assigned.add(cell.number)

            # check that next number to assign is not already in use
            while next_cell_number in cell_numbers_assigned:
                next_cell_number +=1

            # surfaces, replace duplicate surfaces and then collect instances with unique parameters
            cell.volume.replace_duplicate_surfaces(self._surfaces)

            self._surfaces |= {surface.__hash_string__(): surface for surface in cell.volume.surfaces if isinstance(surface, Surface)}
            self._macrobodies |= {macrobody for macrobody in cell.volume.surfaces if isinstance(macrobody, Macrobody)}

            # collect Material instances
            if not cell.material:
                # case 1: where no materials assigned to cell
                pass
            elif cell.material.__hash_string__() in self._materials:
                # case 2: where material assgined but already exists
                cell.material = self._materials[cell.material.__hash_string__()]
            else:
                self._materials[cell.material.__hash_string__()] = cell.material

            # collect transform
            if cell.transform:
                self._transforms.add(cell.transform)

            # collect universes
            if cell.universe:
                self._universes.add(cell.universe)

            # colect transforms from universe fills
            if cell.universe_fill and cell.universe_fill.transform:
                self._transforms.add(cell.universe_fill.transform)

        # assign surface numbers
        surfaces_sorted = _sort_surfaces_by_type(self._surfaces.values())
        next_surface_macrobody_number = self._surface_number_min
        for surface in surfaces_sorted:
            surface.number = next_surface_macrobody_number
            if surface.transform and surface.transform not in self._transforms:
                self._transforms.add(surface.transform)
            next_surface_macrobody_number += 1

        # assign macrobody numbers
        for macrobody in self._macrobodies:
            macrobody.number = next_surface_macrobody_number
            if macrobody.transform and macrobody.transform not in self._transforms:
                self._transforms.add(macrobody.transform)
            next_surface_macrobody_number += 1

        # assign material numbers
        material_numbers_assigned: set[int] = set()
        next_material_number = 1
        for material in self._materials.values():
            # case where material already has number assigned
            if material.number is not None:
                material_numbers_assigned.add(material.number)
                while next_material_number in material_numbers_assigned:
                    next_material_number += 1
            else:
                material.number = next_material_number
                next_material_number += 1

        # assign transform numbers
        next_transform_number = 1
        for transform in self._transforms:
            transform.number = next_transform_number
            next_transform_number += 1

        # assign universe numbers
        next_universe_number = 1
        for universe in self._universes:
            universe.number = next_universe_number
            next_universe_number += 1

        self._asign_numbers_tallies()

    @property
    def mcnp_string(self) -> str:
        """Model as MCNP input text.

        Also updates the summary data with creation date/time and cell and
        surface counts.

        Returns
        -------
        str
            The full MCNP input.
        """
        # set up string for output
        output = ''

        # write messages
        for mcnp_card in self.mcnp.messages:
            output += mcnp_card.as_string + '\n'

        # write summary box
        n_cells = len(self.cells)
        n_cells_with_mats = len([cell for cell in self.cells if cell.material])
        n_cells_with_voids = n_cells - n_cells_with_mats

        self._summary_data['date created'] = datetime.datetime.today().strftime('%d/%m/%Y')
        self._summary_data['time created'] = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-4]
        self._summary_data['material cells'] = n_cells_with_mats
        self._summary_data['void_cells'] = n_cells_with_voids
        self._summary_data['total cells'] = n_cells
        self._summary_data['total surfaces'] = len(self._surfaces)
        output += MCNPHeader('SUMMARY', data_pairs=self._summary_data).as_string

        # write cell definitions
        output += MCNPHeader('CELLS').as_string
        for cell in self.cells:
            # assign mcnp run parameters
            output += f'{cell.mcnp_card.as_string}'
        output += '\n'

        # surfaces and macrobodies section
        output += MCNPHeader('SURFACES').as_string
        surfaces_sorted = _sort_surfaces_by_type(self._surfaces.values())
        for surface in surfaces_sorted:
            output += f'{surface.mcnp_card.as_string}'
        for macrobody in self._macrobodies:
            output += f'{macrobody.mcnp_card.as_string}'
        output += '\n'

        # materials
        if self._materials: output += MCNPHeader('MATERIALS').as_string
        materials_lookup_by_number = {m.number: m for m in self._materials.values()}
        for key in sorted(materials_lookup_by_number):
            output += f'{materials_lookup_by_number[key].mcnp_card.as_string}'

        # transforms
        if self._transforms: output += MCNPHeader('TRANSFORMS').as_string
        for transform in self._transforms:
            output += f'{transform.mcnp_card.as_string}'
        
        # source
        output += MCNPHeader('SOURCE').as_string
        if self.source is not None:
            output += self.source.mcnp_card.as_string

        # tallies
        if self._tallies: output += MCNPHeader('TALLIES').as_string
        tallies = []
        for tally in self._tallies:
            tallies.append(tally.mcnp_card.as_string)
        output += ('C ' + '-' * formatting.MCNP_MAX_CHARS + '\n').join(tallies)

        # run parameters
        output += MCNPHeader('RUN PARAMETERS').as_string
        output += self.mcnp.mcnp_card.as_string

        return output

    @check_folder_exists_for_file_path('filepath')
    def write_mcnp(self, filepath: str | os.PathLike=os.path.join('output', 'model.i')) -> None:
        """Write the model as an MCNP input file.

        The parent folder of ``filepath`` is created if it does not exist.

        Parameters
        ----------
        filepath : str | os.PathLike, optional
            Output file path, by default 'output/model.i'
        """
        logger.debug(f'model - writing mcnp model in write_mcnp')
        
        time_start = time.time()
        with open(filepath, 'w') as output_file:
            output_file.write(self.mcnp_string)
        time_total = time.time() - time_start

        logger.info(f'mcnp model written to: {filepath} - time taken = {time_total:.2e}')

    @property
    def serpent_string(self) -> str:
        """Model surfaces and cells as Serpent2 input text.

        Returns
        -------
        str
            Serpent2 surface and cell definitions.
        """
        # set up string for output
        output = ''

        # write surfaces
        output += SerpentHeader('SURFACE DEFINITIONS').as_string
        for surface in self._surfaces:
            output += f'{surface.serpent_card.as_string}'

        # write cells
        output += SerpentHeader('CELL DEFINITIONS').as_string
        for cell in self.cells:
            output += f'{cell.serpent_card.as_string}'

        return output

    def write_serpent(self, output_filename: str | os.PathLike) -> None:
        """Write the model as a Serpent2 input file.

        Parameters
        ----------
        output_filename : str | os.PathLike
            Output file path.
        """
        with open(output_filename, 'w') as output_file:
            output_file.write(self.serpent_string)

    @property
    def openmc_model(self) -> openmc.model.Model:
        """OpenMC model of the geometry, created on first use and cached.

        Uses default settings (fixed source, 10 batches, 1e7 particles, a 1 MeV
        point source near the origin) and the model tallies.

        Returns
        -------
        openmc.model.Model
            The OpenMC model.
        """
        # return model if already generated.
        if self._openmc_model:
            return self._openmc_model

        # get openmc cells
        openmc_cells = [cell.openmc_cell for cell in self.cells]

        # set default settings
        settings = openmc.settings.Settings()
        settings.batches = 10
        settings.particles = int(1e7)
        settings.run_mode = 'fixed source'
        settings.max_lost_particles = 100

        # add default source
        dist_point = openmc.stats.Point(xyz=(1e-8, 1e-8, 1e-8))
        monoenergetic = openmc.Discrete(x=[1.0], p=[1.0])
        settings.source = openmc.IndependentSource(space=dist_point, energy=monoenergetic)

        # geometry
        geometry = openmc.geometry.Geometry(openmc_cells)
        geometry.remove_redundant_surfaces()

        # empty tally list
        tallies = openmc.Tallies([tally.openmc_tally for tally in self._tallies])
        
        # model
        self._openmc_model = openmc.model.Model(geometry=geometry, settings=settings, tallies=tallies)

        # return model
        return self._openmc_model

    def write_openmc(self, output_folder: str | os.PathLike=os.path.join('output', 'openmc')) -> None:
        """Export the OpenMC model as xml files.

        Parameters
        ----------
        output_folder : str | os.PathLike, optional
            Output folder, by default 'output/openmc'
        """
        logger.debug(f'model - writing openmc file in write_openmc')
        time_start = time.time()
        self.openmc_model.export_to_xml(output_folder)
        time_total = time.time() - time_start
        logger.info(f'openmc model written to: {output_folder} - time taken = {time_total:.2e}')

    @property
    def gdml_string(self) -> str:
        """Model as GDML text (currently materials only).

        Returns
        -------
        str
            GDML xml string.
        """
        # create material XML tags
        root_attributes = {'xmls:gdml': 'http://cern.ch/2001/Schemas/GDML',
                           'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="/afs/cern.ch/sw/lcg/app/releases/GDML/pro/schema/gdml_2.0.xsd'}
        root_tag = XMLTag('gdml', attributes=root_attributes)

        # material tag
        materials_tag = XMLTag('materials')
        root_tag.add_child(materials_tag)

        # write isotopes and 
        root_tag.add_child(self.materials.gdml_xml_tag)

        # return root tag as string
        return root_tag.get_string

    def write_gdml(self, output_filename: str | os.PathLike) -> None:
        """Write the model as a GDML file.

        Parameters
        ----------
        output_filename : str | os.PathLike
            Output file path.
        """
        with open(output_filename, 'w') as output_file:
            output_file.write(self.gdml_string)

    def __and__(self, other_model: Model) -> None:
        """Add the cells of another model to this model in place and renumber.

        Parameters
        ----------
        other_model : Model
            Model whose cells are added.
        """
        self.cells.extend(other_model.cells)
        self._asign_numbers()
