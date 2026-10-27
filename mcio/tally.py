"""Tally definitions for MCNP and OpenMC output.

Contains the :class:`Tally` base class, cell/surface tallies (current, surface
and volume fluence, energy deposition, pulse height) and a regular mesh
fluence tally, each able to produce an MCNP card and an OpenMC tally.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import math

try:
    import openmc
except:
    pass

from mcio import logger, LOGGER_LEVELS
from .formatting import formatting
from .mcnp import MCNPCard, MCNPTallyParameters, MCNP_PARTICLE_DESIGNATORS
from .checkvals import isnumeric
from .openmcparams import OpenMCTallyParameters
from .geometric import BoundingBox

if TYPE_CHECKING:
    from .cell import Cell
    from .surface import Surface


class Tally:
    """Base class for all tallies.

    Parameters
    ----------
    energies : list[float], optional
        Energy bin boundaries in MeV, by default None
    dose_conversion_factors : list[list[float]], optional
        Pair ``[energies, factors]`` defining an energy-dependent response
        function (energies in MeV), by default None
    comment : str, optional
        Tally comment, also used as the OpenMC tally name, by default None
    particles : list[str], optional
        Particle names to tally, by default ['neutron']
    """

    def __init__(self, energies: list[float] | None=None,
                 dose_conversion_factors: list[list[float]] | np.ndarray | None=None,
                 comment: str | None=None, particles: list[str]=['neutron']) -> None:
        """Initialise the Tally base class."""
        self.number: int | None = None
        self.comment: str | None = comment
        self.energies: list[float] | None = energies
        self.dose_conversion_factors: list[list[float]] | np.ndarray | None = dose_conversion_factors
        self.particles: list[str] = particles
        self.mcnp_tally_designator: int | None = None
        self.openmc_tally_score: list[str] | None = None
        self.mcnp: MCNPTallyParameters = MCNPTallyParameters()
        self.openmc_params: OpenMCTallyParameters = OpenMCTallyParameters()

    @classmethod
    def from_dict(cls, tally_type_label: str, tally_data: dict[str, Any]) -> Tally:
        """Construct a tally of the named type from a dictionary of arguments.

        Parameters
        ----------
        tally_type_label : str
            Tally type key, e.g. ``'volumefluencetally'``.
        tally_data : dict[str, Any]
            Keyword arguments passed to the tally constructor.

        Returns
        -------
        Tally
            The constructed tally.

        Raises
        ------
        KeyError
            If ``tally_type_label`` is not a known tally type.
        """
        # available tally type lookup
        TALLY_TYPES = {
                    'fluencemeshtally': FluenceMeshTally,
                    'surfacecurrent': SurfaceCurrent,
                    'surfacefluencetally': SurfaceFluenceTally,
                    'volumefluencetally': VolumeFluenceTally,
                    'energydepositiontally': EnergyDepositionTally,
                    'pulseheighttally': PulseHeightTally
                }

        if tally_type_label not in TALLY_TYPES:
            msg = f'tally type from json {TALLY_TYPES} not available, should be ' + ', '.join(TALLY_TYPES)
            logger.data(msg)
            raise KeyError(msg)

        # special case for mcnp only variables
        if 'multiplier' in tally_data:
            mcnp_multiplier = tally_data['multiplier']
            del tally_data['multiplier']
        else:
            mcnp_multiplier = None

        # set up tally
        tally = TALLY_TYPES[tally_type_label](**tally_data)
        tally.mcnp.multiplier = mcnp_multiplier

        # return tally
        return tally


class CellOrSurfaceTally(Tally):
    """Base class for tallies over a list of cells or surfaces.

    Parameters
    ----------
    cells : list[Cell], optional
        Cells to tally over, by default None
    surfaces : list[Surface], optional
        Surfaces to tally over (used only if no cells are given), by default
        None
    comment : str, optional
        Tally comment, by default None
    **kwargs : Any
        Passed to :class:`Tally`.
    """

    def __init__(self, cells: list[Cell] | None=None, surfaces: list[Surface] | None=None,
                 comment: str | None=None, 
                 **kwargs: Any) -> None:
        """Initialise the CellOrSurfaceTally."""
        super().__init__(**kwargs)
        
        self.cells: list[Cell] | None = cells
        self.surfaces: list[Surface] | None = surfaces
        self.comment: str | None = comment

    @property
    def target_numbers(self) -> list[str]:
        """Cell (or, if no cells, surface) numbers as strings.

        Raises
        ------
        Exception
            If neither cells nor surfaces are set.
        """
        if self.cells:
            numbers = [f'{cell.number}' for cell in self.cells]
        elif self.surfaces:
            numbers = [f'{surface.number}' for surface in self.surfaces]
        else:
            msg = 'no surfaces or cells passed to tally...'
            raise Exception(msg)
        return numbers

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP tally card with its sub-cards.

        Includes cell/surface flag, comment, energy, multiplier, divisor and
        dose function (DE/DF) cards where set.

        Raises
        ------
        Exception
            If ``dose_conversion_factors`` is not a two-item list, tuple or
            ndarray.
        """

        particles = ','.join([MCNP_PARTICLE_DESIGNATORS[p].upper() for p in self.particles])
        mcnp_card = MCNPCard(f'F{self.number}{self.mcnp_tally_designator}:{particles}', 
                             [self.target_numbers])
        
        # case for cell or surface flags
        if self.mcnp.cell_flags:
            cell_numbers = [cell.number for cell in self.mcnp.cell_flags]
            mcnp_card.add_subcard(MCNPCard(f'CF{self.number}{self.mcnp_tally_designator}',
                                           [[f'{cell_number}' for cell_number in cell_numbers]]))
        if self.mcnp.surface_flags:
            surface_numbers = [surface.number for surface in self.mcnp.surface_flags]
            mcnp_card.add_subcard(MCNPCard(f'SF{self.number}{self.mcnp_tally_designator}',
                                           [[f'{surface_number}' for surface_number in surface_numbers]]))

        # # comment
        if self.comment:
            mcnp_card.add_subcard(MCNPCard(f'FC{self.number}{self.mcnp_tally_designator}', 
                                            [[self.comment]]))

        # energy
        if self.energies:
            mcnp_card.add_subcard(MCNPCard(f'E{self.number}{self.mcnp_tally_designator}',
                                        [[f'{e:.{formatting.MCNP_DP}}' for e in self.energies]]))
        
        # multiplier
        if self.mcnp.multiplier:
            if isinstance(self.mcnp.multiplier, (list, tuple)):
                multiplier_as_string = [f'{v:.{formatting.MCNP_DP}}' if isinstance(v, float) else f'{v}' for v in self.mcnp.multiplier]
            else:
                multiplier_as_string = [f'{self.mcnp.multiplier:.{formatting.MCNP_DP}e}']
            mcnp_card.add_subcard(MCNPCard(f'FM{self.number}{self.mcnp_tally_designator}', 
                                         [multiplier_as_string]))
            
        # tally divisor
        if self.mcnp.tally_divisor:
            if isinstance(self.mcnp.tally_divisor, (list, tuple)):
                divisor_as_string = [f'{v:.{formatting.MCNP_DP}}' if isinstance(v, float) else f'{v}' for v in self.mcnp.tally_divisor]
                if len(self.mcnp.tally_divisor) !=len(self.target_numbers):
                    logger.warning(f'Number of tally divisors does not match number of cells/surface for tally {self.number}.')
            else:
                divisor_as_string = [f'{self.mcnp.tally_divisor:.{formatting.MCNP_DP}e} '*len(self.target_numbers)]
            mcnp_card.add_subcard(MCNPCard(f'SD{self.number}{self.mcnp_tally_designator}', 
                                           [divisor_as_string]))
        
        if self.dose_conversion_factors:
            # check format of dose conversion factors
            if isinstance(self.dose_conversion_factors, (list, tuple, np.ndarray)) and len(self.dose_conversion_factors) == 2:
                # check dose interpolation method
                if self.mcnp.dose_interpolation.lower() not in {'log-log', 'log-linear', 'linear-log', 'linear-linear'}:
                    logger.warning(f'dose_interpolations for MCNP should be one of: \'log-log\', \'log-linear\', \'linear-log\', \'linear-linear\'. Not \'{self.mcnp.dose_interpolation}\'. Defaulted to \'log-log\'')
                    mcnp_interpolation = 'log-log'
                else:
                    mcnp_interpolation = self.mcnp.dose_interpolation
                a_interpolation = mcnp_interpolation[:3].upper()
                b_interpolation = mcnp_interpolation.split('-')[1][:3].upper()
                rows_de_df = [[f'de{self.number}{self.mcnp_tally_designator: <{formatting.MCNP_EXP+3}}', f'df{self.number}{self.mcnp_tally_designator}']]
                rows_de_df.append([f'{a_interpolation: <{formatting.MCNP_EXP+6}}', f'{b_interpolation}'])
                for energy, dose_factor in zip(*self.dose_conversion_factors):
                        rows_de_df.append([f'{energy:.{formatting.MCNP_EXP}e}', f'{dose_factor:.{formatting.MCNP_EXP}e}'])

                mcnp_card.add_subcard(MCNPCard(f'#', rows_de_df))
            else:
                msg = f'dose_conversion_factors should be a 2d list, tuple of numpy ndarray type, not {type(self.dose_conversion_factors)}'
                raise Exception(msg)

        return mcnp_card

    @property
    def openmc_tally(self) -> openmc.Tally:
        """OpenMC tally with particle, cell/surface, energy and dose filters.

        Energies are converted from MeV to eV.

        Raises
        ------
        Exception
            If neither cells nor surfaces are set, or if
            ``dose_conversion_factors`` is badly formatted.
        """
        filters=[]

        particle_filter = openmc.ParticleFilter(bins=self.particles)
        filters.append(particle_filter)

        openmc_tally = openmc.Tally()
        openmc_tally.scores = self.openmc_tally_score

        if self.cells:
            cell_or_surface_filter = openmc.CellFilter([cell.openmc_cell for cell in self.cells])
            filters.append(cell_or_surface_filter)
        elif self.surfaces:
            cell_or_surface_filter = openmc.SurfaceFilter([surface.openmc_surface for surface in self.surfaces])
            filters.append(cell_or_surface_filter)
        else:
            msg = f'no surfaces or cells passed to tally {self.number}'
            raise Exception(msg)

        # Comment/name
        if self.comment:
            openmc_tally.name = self.comment

        # Energy
        if self.energies:
            energy_filter = openmc.EnergyFilter([e*1e6 for e in self.energies])
            filters.append(energy_filter)

        if self.dose_conversion_factors:
            # check format of dose conversion factors
            if isinstance(self.dose_conversion_factors, (list, tuple, np.ndarray)) and len(self.dose_conversion_factors) == 2:
                dose_filter = openmc.EnergyFunctionFilter([e*1e6 for e in self.dose_conversion_factors[0]],self.dose_conversion_factors[1])
                # check dose interpolation method
                if self.openmc_params.dose_interpolation.lower() not in {'log-log', 'log-linear', 'linear-log', 'linear-linear', 'histogram', 'quadratic', 'cubic'}:
                    logger.warning(f'dose_interpolations for OpenMC should be one of: \'log-log\', \'log-linear\', \'linear-log\', \'linear-linear\', \'histogram\', \'quadratic\', \'cubic\'. Not \'{self.openmc_params.dose_interpolation}\'. Defaulted to \'linear-linear\'')
                    openmc_interpolation = 'linear-linear'
                else:
                    openmc_interpolation = self.openmc_params.dose_interpolation
                dose_filter.interpolation = openmc_interpolation
                filters.append(dose_filter)
            else:
                msg = f'dose_conversion_factors should be a 2d list, tuple of numpy ndarray type, not {type(self.dose_conversion_factors)}'
                raise Exception(msg)

        openmc_tally.filters = filters

        return openmc_tally

class SurfaceCurrent(CellOrSurfaceTally):
    """Surface current tally (MCNP F1, OpenMC ``current``).

    Parameters
    ----------
    **kwargs : Any
        Passed to :class:`CellOrSurfaceTally`.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialise the SurfaceCurrent."""
        super().__init__(**kwargs)
        self.mcnp_tally_designator = 1
        self.openmc_tally_score = ["current"]
        logger.warning(f'The MCNP surface current tally tallies total current while OpenMC tallies net current.')

class SurfaceFluenceTally(CellOrSurfaceTally):
    """Surface fluence tally (MCNP F2, OpenMC ``flux``).

    Parameters
    ----------
    **kwargs : Any
        Passed to :class:`CellOrSurfaceTally`.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialise the SurfaceFluenceTally."""
        super().__init__(**kwargs)
        self.mcnp_tally_designator = 2
        self.openmc_tally_score = ['flux']

class VolumeFluenceTally(CellOrSurfaceTally):
    """Cell volume fluence tally (MCNP F4, OpenMC ``flux``).

    Parameters
    ----------
    **kwargs : Any
        Passed to :class:`CellOrSurfaceTally`.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialise the VolumeFluenceTally."""
        super().__init__(**kwargs)
        self.mcnp_tally_designator = 4
        self.openmc_tally_score = ['flux']

class EnergyDepositionTally(CellOrSurfaceTally):
    """Energy deposition tally (MCNP F6, OpenMC ``heating``).

    Parameters
    ----------
    **kwargs : Any
        Passed to :class:`CellOrSurfaceTally`.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialise the EnergyDepositionTally."""
        super().__init__(**kwargs)
        self.mcnp_tally_designator = 6
        self.openmc_tally_score = ['heating']

class PulseHeightTally(CellOrSurfaceTally):
    """Pulse height tally (MCNP F8, OpenMC ``pulse-height``).

    Parameters
    ----------
    **kwargs : Any
        Passed to :class:`CellOrSurfaceTally`.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialise the PulseHeightTally."""
        super().__init__(**kwargs)
        self.mcnp_tally_designator = 8
        self.openmc_tally_score = ['pulse-height']


class FluenceMeshTally(Tally):
    """Regular Cartesian mesh fluence tally (MCNP FMESH, OpenMC mesh filter).

    Parameters
    ----------
    x_min : float, optional
        Lower x bound of the mesh, by default 0
    x_max : float, optional
        Upper x bound of the mesh, by default 1
    y_min : float, optional
        Lower y bound of the mesh, by default 0
    y_max : float, optional
        Upper y bound of the mesh, by default 1
    z_min : float, optional
        Lower z bound of the mesh, by default 0
    z_max : float, optional
        Upper z bound of the mesh, by default 1
    x_ints : int or list[int], optional
        Number of mesh intervals in x, by default 1
    y_ints : int or list[int], optional
        Number of mesh intervals in y, by default 1
    z_ints : int or list[int], optional
        Number of mesh intervals in z, by default 1
    **kwargs : Any
        Passed to :class:`Tally`.
    """

    def __init__(self, x_min: float=0, x_max: float=1, y_min: float=0, y_max: float=1,
                 z_min: float=0, z_max: float=1,
                 x_ints: int | list[int]=1, y_ints: int | list[int]=1,
                 z_ints: int | list[int]=1, **kwargs: Any) -> None:
        """Initialise the FluenceMeshTally."""
        super().__init__(**kwargs)
        self.mcnp_tally_designator = 4
        self.openmc_tally_score = ['flux']
        
        self.x_min: float = x_min
        self.x_max: float = x_max
        self.y_min: float = y_min
        self.y_max: float = y_max
        self.z_min: float = z_min
        self.z_max: float = z_max

        self.x_ints: list[int] = [x_ints] if isnumeric(x_ints) else x_ints
        self.y_ints: list[int] = [y_ints] if isnumeric(y_ints) else y_ints
        self.z_ints: list[int] = [z_ints] if isnumeric(z_ints) else z_ints

    @classmethod
    def from_bounding_box(cls, bounding_box: BoundingBox, resolution: float, extension: float, **kwargs) -> FluenceMeshTally:
        return cls(
            x_min = bounding_box.x_min - extension,
            x_max = bounding_box.x_max + extension,
            x_ints = math.ceil(bounding_box.x_length / resolution),
            y_min = bounding_box.y_min - extension,
            y_max = bounding_box.y_max + extension,
            y_ints = math.ceil(bounding_box.y_length / resolution),
            z_min = bounding_box.z_min - extension,
            z_max = bounding_box.z_max + extension,
            z_ints = math.ceil(bounding_box.z_length / resolution),
            **kwargs
        )

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP FMESH card with comment, dose function and multiplier sub-cards.

        Raises
        ------
        Exception
            If ``dose_conversion_factors`` is not a two-item list, tuple or
            ndarray.
        """
        rows = [[f'origin = {self.x_min:.{formatting.MCNP_EXP}e} {self.y_min:.{formatting.MCNP_EXP}e} {self.z_min:.{formatting.MCNP_EXP}e}']]

        rows.append(['imesh', f'{self.x_max:.{formatting.MCNP_EXP}e}', 'iints', *[f'{int(x_int)}' for x_int in self.x_ints]])
        rows.append(['jmesh', f'{self.y_max:.{formatting.MCNP_EXP}e}', 'jints', *[f'{int(y_int)}' for y_int in self.y_ints]])
        rows.append(['kmesh', f'{self.z_max:.{formatting.MCNP_EXP}e}', 'kints', *[f'{int(z_int)}' for z_int in self.z_ints]])

        particles = ' '.join([MCNP_PARTICLE_DESIGNATORS[p.lower()].upper() for p in self.particles])

        if self.energies:
            rows.append([f'emesh'])
            rows.append([*[f'{v:.{formatting.MCNP_DP}e}' for v in self.energies]])

        mcnp_card = MCNPCard(f'FMESH{self.number}{self.mcnp_tally_designator}:{particles}', rows)

        # comment
        if self.comment:
            mcnp_card.add_subcard(MCNPCard(f'FC{self.number}{self.mcnp_tally_designator}', [[self.comment]]))

        if self.dose_conversion_factors:
            # check format of dose conversion factors
            if isinstance(self.dose_conversion_factors, (list, tuple, np.ndarray)) and len(self.dose_conversion_factors) == 2:
                # check dose interpolation method
                if self.mcnp.dose_interpolation.lower() not in {'log-log', 'log-linear', 'linear-log', 'linear-linear'}:
                    logger.warning(f'dose_interpolations for MCNP should be one of: \'log-log\', \'log-linear\', \'linear-log\', \'linear-linear\'. Not \'{self.mcnp.dose_interpolation}\'. Defaulted to \'log-log\'')
                    mcnp_interpolation = 'log-log'
                else:
                    mcnp_interpolation = self.mcnp.dose_interpolation
                a_interpolation = mcnp_interpolation[:3].upper()
                b_interpolation = mcnp_interpolation.split('-')[1][:3].upper()
                rows_de_df = [[f'de{self.number}{self.mcnp_tally_designator: <{formatting.MCNP_EXP+3}}', f'df{self.number}{self.mcnp_tally_designator}']]
                rows_de_df.append([f'{a_interpolation: <{formatting.MCNP_EXP+6}}', f'{b_interpolation}'])
                for energy, dose_factor in zip(*self.dose_conversion_factors):
                        rows_de_df.append([f'{energy:.{formatting.MCNP_EXP}e}', f'{dose_factor:.{formatting.MCNP_EXP}e}'])

                mcnp_card.add_subcard(MCNPCard(f'#', rows_de_df))
            else:
                msg = f'dose_conversion_factors should be a 2d list, tuple of numpy ndarray type, not {type(self.dose_conversion_factors)}'
                raise Exception(msg)

        # multiplier
        if self.mcnp.multiplier:
            if isinstance(self.mcnp.multiplier, (list, tuple)):
                multiplier_as_string = [f'{v:.{formatting.MCNP_DP}}' if isinstance(v, float) else f'{v}' for v in self.mcnp.multiplier]
            else:
                multiplier_as_string = [f'{self.mcnp.multiplier:.{formatting.MCNP_DP}e}']
            mcnp_card.add_subcard(MCNPCard(f'FM{self.number}{self.mcnp_tally_designator}', 
                                         [multiplier_as_string]))

        return mcnp_card

    @property
    def openmc_tally(self) -> openmc.Tally:
        """OpenMC tally on a regular mesh with particle, energy and dose filters.

        Only the first entry of each of ``x_ints``, ``y_ints`` and ``z_ints``
        is used for the mesh dimension. Energies are converted from MeV to eV.

        Raises
        ------
        Exception
            If ``dose_conversion_factors`` is badly formatted.
        """
        filters=[]

        mesh = openmc.RegularMesh()
        mesh.dimension = [self.x_ints[0], self.y_ints[0], self.z_ints[0]] 
        mesh.lower_left = [self.x_min, self.y_min, self.z_min]
        mesh.upper_right = [self.x_max, self.y_max, self.z_max]

        mesh_filter = openmc.MeshFilter(mesh)

        particle_filter = openmc.ParticleFilter(self.particles) #doesn't work for all openmc particles

        filters.extend([particle_filter, mesh_filter])

        openmc_tally = openmc.Tally()
        openmc_tally.scores = self.openmc_tally_score

        # Comment/name
        if self.comment:
            openmc_tally.name = self.comment

        # Energy
        if self.energies:
            energy_filter = openmc.EnergyFilter([e*1e6 for e in self.energies])
            filters.append(energy_filter)

        if self.dose_conversion_factors:
            # check format of dose conversion factors
            if isinstance(self.dose_conversion_factors, (list, tuple, np.ndarray)) and len(self.dose_conversion_factors) == 2:
                dose_filter = openmc.EnergyFunctionFilter([e*1e6 for e in self.dose_conversion_factors[0]],self.dose_conversion_factors[1])
                # check dose interpolation method
                if self.openmc_params.dose_interpolation.lower() not in {'log-log', 'log-linear', 'linear-log', 'linear-linear', 'histogram', 'quadratic', 'cubic'}:
                    logger.warning(f'dose_interpolations for OpenMC should be one of: \'log-log\', \'log-linear\', \'linear-log\', \'linear-linear\', \'histogram\', \'quadratic\', \'cubic\'. Not \'{self.openmc_params.dose_interpolation}\'. Defaulted to \'log-log\'')
                    openmc_interpolation = 'log-log'
                else:
                    openmc_interpolation = self.openmc_params.dose_interpolation
                dose_filter.interpolation = openmc_interpolation
                filters.append(dose_filter)
            else:
                msg = f'dose_conversion_factors should be a 2d list, tuple of numpy ndarray type, not {type(self.dose_conversion_factors)}'
                raise Exception(msg)

        openmc_tally.filters = filters

        return openmc_tally
