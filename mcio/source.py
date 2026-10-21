"""Particle source definitions for MCNP output.

Contains energy and position distributions and source classes (point,
spherical stochastic volume, rectangular volume, cylindrical volume, circular
plane and read-from-file) that write MCNP ``SDEF`` / ``READ`` cards.
"""
from __future__ import annotations

import math

from typing import Any, TYPE_CHECKING

from .formatting import formatting
from .mcnp import MCNPCard, MCNP_PARTICLE_DESIGNATORS

if TYPE_CHECKING:
    from .surface import Surface


def _particle_designator(particle: str) -> str:
    """MCNP particle designator (e.g. ``N``) for a particle name (e.g. ``'neutron'``).

    A designator given directly is passed through.
    """
    return MCNP_PARTICLE_DESIGNATORS.get(particle.lower(), particle).upper()


class SourceEnergyDistribution():
    """Source energy distribution written as MCNP SI1/SP1 cards.

    Parameters
    ----------
    energies : list[float], optional
        Energy values in MeV, by default [0.0]
    distribution : list[float], optional
        Probabilities for each energy, by default [1.0]
    distribution_type : str, optional
        ``'discrete'`` (SI ``L``) or ``'histogram'`` (SI ``H``), by default
        'discrete'
    """

    def __init__(self, energies: list[float]=[0.0], distribution: list[float]=[1.0],
                 distribution_type: str='discrete') -> None:
        """Initialise the SourceEnergyDistribution."""
        self._distribution_type: str = distribution_type
        self._energies: list[float] = energies
        self._distribution: list[float] = distribution

    @property
    def mcnp_subcards(self) -> list[MCNPCard]:
        """MCNP ``SI1`` and ``SP1`` cards for the distribution."""
        self._distribution_types = {'discrete': 'L',
                                    'histogram': 'H'}

        energies = [[f'{float(e):.{formatting.MCNP_DP}}' for e in self._energies]]
        distribution = [[f'{float(p):.{formatting.MCNP_DP}}' for p in self._distribution]]

        designator = 'SI1 ' + self._distribution_types[self._distribution_type]
        card_energies = MCNPCard(designator, energies)
        card_distributions = MCNPCard('SP1', distribution)

        return [card_energies, card_distributions]


class SourcePositionDistribution():
    """Source position distribution written as MCNP SI2/SP2 cards.

    Parameters
    ----------
    positions : list[tuple[float, float, float]]
        Source positions (x, y, z).
    distribution : list[float]
        Probabilities for each position.
    distribution_type : str, optional
        ``'discrete'`` (SI ``L``) or ``'histogram'`` (SI ``H``), by default
        'discrete'
    """

    def __init__(self, positions: list[tuple[float, float, float]], distribution: list[float],
                 distribution_type: str='discrete') -> None:
        """Initialise the SourcePositionDistribution."""
        self._distribution_type: str = distribution_type
        self._positions: list[tuple[float, float, float]] = positions
        self._distribution: list[float] = distribution

    @property
    def mcnp_subcards(self) -> list[MCNPCard]:
        """MCNP ``SI2`` and ``SP2`` cards for the distribution."""
        self._distribution_types = {'discrete': 'L',
                                    'histogram': 'H'}

        positions = [[f'{float(p[0]):.{formatting.MCNP_DP}} {float(p[1]):.{formatting.MCNP_DP}} {float(p[2]):.{formatting.MCNP_DP}}' for p in self._positions]]
        distribution = [[f'{float(p):.{formatting.MCNP_DP}}' for p in self._distribution]]

        designator = 'SI2 ' + self._distribution_types[self._distribution_type]
        card_positions = MCNPCard(designator, positions)
        card_distributions = MCNPCard('SP2', distribution)

        return [card_positions, card_distributions]


class Source:
    """Base class for particle sources.

    Parameters
    ----------
    energy : float or SourceEnergyDistribution, optional
        Mono-energetic source energy in MeV or an energy distribution, by
        default 1.0
    particle : str, optional
        Source particle name, by default 'neutron'
    """

    def __init__(self, energy: float | SourceEnergyDistribution=1.0, particle: str='neutron') -> None:
        """Initialise the Source base class."""
        self._energy: float | SourceEnergyDistribution = energy
        self._particle: str = particle
        self._dp: int = 6

    @property
    def energy(self) -> float | SourceEnergyDistribution:
        """Source energy in MeV or energy distribution."""
        return self._energy

    @property
    def particle(self) -> str:
        """Source particle name."""
        return self._particle


class MCNPSourceFromFile:
    """Source read from an external file via the MCNP ``READ`` card.

    Parameters
    ----------
    file_path : str
        Path of the file containing the source definition.
    """

    def __init__(self, file_path: str) -> None:
        """Initialise the MCNPSourceFromFile."""
        self._file_path: str = file_path

    @property
    def file_path(self) -> str:
        """Path of the source file."""
        return self._file_path

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP ``READ file=`` card."""
        return MCNPCard('READ', [[f'file= {self._file_path}']])

class Point(Source):
    """Point source.

    Parameters
    ----------
    position : tuple[float, float, float] or SourcePositionDistribution, optional
        Source position or position distribution, by default (0, 0, 0)
    **kwargs : Any
        Passed to :class:`Source` (``energy``, ``particle``).
    """

    def __init__(self, position: tuple[float, float, float] | SourcePositionDistribution=(0,0,0),
                 **kwargs: Any) -> None:
        """Initialise the Point source."""
        super().__init__(**kwargs)

        self._position: tuple[float, float, float] | SourcePositionDistribution = position

    @property
    def position(self) -> tuple[float, float, float] | SourcePositionDistribution:
        """Source position or position distribution."""
        return self._position

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP ``SDEF`` card with energy (D1) and position (D2) distributions."""
        # get position distributions
        if not isinstance(self._position, SourcePositionDistribution):
            position = SourcePositionDistribution([self._position], [1.0])
        else:
            position = self._position
        pos_subcard_1, pos_subcard_2 = position.mcnp_subcards

        # get energy distributions
        if not isinstance(self._energy, SourceEnergyDistribution):
            energy = SourceEnergyDistribution([self._energy], [1.0])
        else:
            energy = self._energy
        energy_subcard_1, energy_subcard_2 = energy.mcnp_subcards

        # set up mcnp card
        mcnp_card = MCNPCard('SDEF', [[f'PAR={_particle_designator(self._particle)}', 'ERG=D1', 'POS=D2']])

        # add energy distribution cards
        mcnp_card.add_subcard(energy_subcard_1)
        mcnp_card.add_subcard(energy_subcard_2)

        # add position distribution cards
        mcnp_card.add_subcard(pos_subcard_1)
        mcnp_card.add_subcard(pos_subcard_2)
        return mcnp_card

class SphericalStochasticVolumeSource(Source):
    """Inward-directed surface source on a sphere for stochastic volume calculation.

    Particles start on the spherical surface with a cosine-weighted inward
    direction; the weight is set to the sphere's cross-sectional area.

    Parameters
    ----------
    spherical_surface : Surface
        Sphere surface (must have ``number`` and ``radius``).
    particle : str, optional
        Source particle name, by default 'neutron'
    **kwargs : Any
        Passed to :class:`Source` (``energy``).
    """

    def __init__(self, spherical_surface: Surface, particle: str='neutron', **kwargs: Any) -> None:
        """Initialise the SphericalStochasticVolumeSource."""
        super().__init__(**kwargs)

        self.spherical_surface: Surface = spherical_surface
        self._particle = particle

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP ``SDEF`` surface source card with energy and direction sub-cards."""
        # get energy distributions
        if not isinstance(self._energy, SourceEnergyDistribution):
            energy = SourceEnergyDistribution([self._energy], [1.0])
        else:
            energy = self._energy
        energy_subcard_1, energy_subcard_2 = energy.mcnp_subcards

        # set up source distribution
        radial_distribution_intensity = MCNPCard('SI2', [[f'0 1']])
        radial_distribution_probability = MCNPCard('SP2', [['-21 1']])

        # create mcnp card
        particle_str = f'PAR={MCNP_PARTICLE_DESIGNATORS[self._particle].upper()}'
        energy_str = 'ERG=D1'
        normal_str = f'NRM=-1'
        surface_str = f'SUR={self.spherical_surface.number}'
        weight_str = f'WGT={math.pi * self.spherical_surface.radius ** 2:.12f}'
        direction_str = 'DIR=D2'
        mcnp_card = MCNPCard('SDEF', [[particle_str,
                                       energy_str,
                                       normal_str,
                                       surface_str,
                                       weight_str,
                                       direction_str]])

        # add sub cards
        mcnp_card.add_subcard(energy_subcard_1)
        mcnp_card.add_subcard(energy_subcard_2)
        mcnp_card.add_subcard(radial_distribution_intensity)
        mcnp_card.add_subcard(radial_distribution_probability)

        # return mcnp card
        return mcnp_card

class RectangularParallelepipedVolume(Source):
    """Uniform source in an axis-aligned box, mono-directional.

    Parameters
    ----------
    x_min : float, optional
        Lower x bound, by default 0.0
    x_max : float, optional
        Upper x bound, by default 0.0
    y_min : float, optional
        Lower y bound, by default 0.0
    y_max : float, optional
        Upper y bound, by default 0.0
    z_min : float, optional
        Lower z bound, by default 0.0
    z_max : float, optional
        Upper z bound, by default 0.0
    direction : tuple[float, float, float], optional
        Emission direction vector (``VEC``), by default (1, 0, 0)
    **kwargs : Any
        Passed to :class:`Source` (``energy``, ``particle``).
    """

    def __init__(self, x_min: float=0.0, x_max: float=0.0, y_min: float=0.0, y_max: float=0.0,
                 z_min: float=0.0, z_max: float=0.0,
                 direction: tuple[float, float, float]=(1,0,0), **kwargs: Any) -> None:
        """Initialise the RectangularParallelepipedVolume source."""
        super().__init__(**kwargs)

        self._x_min: float = x_min
        self._x_max: float = x_max
        self._y_min: float = y_min
        self._y_max: float = y_max
        self._z_min: float = z_min
        self._z_max: float = z_max
        self._direction: tuple[float, float, float] = direction

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP ``SDEF`` card with x/y/z range distributions (D3-D5).

        Degenerate axes are written as fixed coordinates instead of
        distributions.
        """
        # get enregy distributions
        if not isinstance(self._energy, SourceEnergyDistribution):
            energy = SourceEnergyDistribution([self._energy], [1.0])
        else:
            energy = self._energy
        energy_subcard_1, energy_subcard_2 = energy.mcnp_subcards

        # set up dir
        direction_as_str = ' '.join([f'{dir:.{formatting.MCNP_DP}e}' for dir in self._direction])

        # set up distributions for x, y and z for volume of source positions
        subcard_x_range = MCNPCard('SI3', [[f'{self._x_min:.{formatting.MCNP_DP}e} {self._x_max:.{formatting.MCNP_DP}e}']])
        subcard_x_range_distribution = MCNPCard('SP3', [['0 1']])
        subcard_y_range = MCNPCard('SI4', [[f'{self._y_min:.{formatting.MCNP_DP}e} {self._y_max:.{formatting.MCNP_DP}e}']])
        subcard_y_range_distribution = MCNPCard('SP4', [['0 1']])
        subcard_z_range = MCNPCard('SI5', [[f'{self._z_min:.{formatting.MCNP_DP}e} {self._z_max:.{formatting.MCNP_DP}e}']])
        subcard_z_range_distribution = MCNPCard('SP5', [['0 1']])

        sub_cards_to_add: list[MCNPCard] = []

        # cases where degenerate along ax axis
        if self._x_min == self._x_max:
            x_var = f'X={self._x_min}'
        else:
            x_var = 'X=D3'
            sub_cards_to_add.append(subcard_x_range)
            sub_cards_to_add.append(subcard_x_range_distribution)
        if self._y_min == self._y_max:
            y_var = f'Y={self._y_min}'
        else:
            y_var = 'Y=D4'
            sub_cards_to_add.append(subcard_y_range)
            sub_cards_to_add.append(subcard_y_range_distribution)
        if self._z_min == self._z_max:
            z_var = f'Z={self._z_min}'
        else:
            z_var = 'Z=D5'
            sub_cards_to_add.append(subcard_z_range)
            sub_cards_to_add.append(subcard_z_range_distribution)

        mcnp_card = MCNPCard('SDEF', [[f'PAR={_particle_designator(self._particle)}', 'ERG=D1', x_var, y_var, z_var],
                                      [f'VEC={direction_as_str}'],
                                      ['DIR=1']])

        # add energy distribution subcards
        mcnp_card.add_subcard(energy_subcard_1)
        mcnp_card.add_subcard(energy_subcard_2)

        for sub_card in sub_cards_to_add:
            mcnp_card.add_subcard(sub_card)

        return mcnp_card

class IsotropicCylindricalVolume(Source):
    """Isotropic source uniformly distributed in a cylinder.

    Parameters
    ----------
    x0 : float, optional
        x coordinate of the base centre, by default 0
    y0 : float, optional
        y coordinate of the base centre, by default 0
    z0 : float, optional
        z coordinate of the base centre, by default 0
    u0 : float, optional
        x component of the cylinder axis, by default 0
    v0 : float, optional
        y component of the cylinder axis, by default 0
    w0 : float, optional
        z component of the cylinder axis, by default 0
    radius : float, optional
        Cylinder radius, by default 0.0
    height : float, optional
        Cylinder height along the axis, by default 0.0
    energy : float or SourceEnergyDistribution, optional
        Source energy in MeV or an energy distribution, by default 0.0
    **kwargs : Any
        Passed to :class:`Source` (``particle``).
    """

    def __init__(self, x0: float=0, y0: float=0, z0: float=0, u0: float=0, v0: float=0,
                 w0: float=0, radius: float=0.0, height: float=0.0,
                 energy: float | SourceEnergyDistribution=0.0, **kwargs: Any) -> None:
        """Initialise the IsotropicCylindricalVolume source."""
        super().__init__(**kwargs)

        self._x0: float = x0
        self._y0: float = y0
        self._z0: float = z0
        self._u0: float = u0
        self._v0: float = v0
        self._w0: float = w0

        self._radius: float = radius
        self._height: float = height
        self._energy = energy

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP ``SDEF`` card with radial (D2) and axial extent (D3) sub-cards."""
        # get enregy distributions
        if not isinstance(self._energy, SourceEnergyDistribution):
            energy = SourceEnergyDistribution([self._energy], [1.0])
        else:
            energy = self._energy
        energy_subcard_1, energy_subcard_2 = energy.mcnp_subcards

        position_string = f'POS={self._x0:.{formatting.MCNP_DP}e} {self._y0:.{formatting.MCNP_DP}e} {self._z0:.{formatting.MCNP_DP}e}'
        extend_string = f'EXT=d3'
        axis_string = f'AXS={self._u0:.{formatting.MCNP_DP}e} {self._v0:.{formatting.MCNP_DP}e} {self._w0:.{formatting.MCNP_DP}e}'

        mcnp_card = MCNPCard('SDEF', [[f'PAR={_particle_designator(self._particle)}'],
                                      [f'ERG=D1'],
                                      [position_string],
                                      [extend_string],
                                      [axis_string],
                                      ['RAD=D2']])

        # add energy distributions subcards
        mcnp_card.add_subcard(energy_subcard_1)
        mcnp_card.add_subcard(energy_subcard_2)

        # add radial distribution subcards
        mcnp_card.add_subcard(MCNPCard('SI2', [[f'{0.0:.{formatting.MCNP_DP}e} {self._radius:.{formatting.MCNP_DP}e}']]))
        mcnp_card.add_subcard(MCNPCard('SP2', [['-21 1']]))

        # add extend distribution subcards
        mcnp_card.add_subcard(MCNPCard('SI3', [[f'{0.0:.{formatting.MCNP_DP}e} {self._height:.{formatting.MCNP_DP}e}']]))
        # mcnp_card.add_subcard(MCNPCard('SP3', [['1']]))

        return mcnp_card

class CircularPlaneParrallel(Source):
    """Mono-directional disc source emitting along its axis.

    Parameters
    ----------
    x0 : float, optional
        x coordinate of the disc centre, by default 0
    y0 : float, optional
        y coordinate of the disc centre, by default 0
    z0 : float, optional
        z coordinate of the disc centre, by default 0
    u0 : float, optional
        x component of the disc axis and emission direction, by default 0
    v0 : float, optional
        y component of the disc axis and emission direction, by default 0
    w0 : float, optional
        z component of the disc axis and emission direction, by default 0
    radius : float, optional
        Disc radius, by default 0.0
    **kwargs : Any
        Passed to :class:`Source` (``energy``, ``particle``).
    """

    def __init__(self, x0: float=0, y0: float=0, z0: float=0, u0: float=0, v0: float=0,
                 w0: float=0, radius: float=0.0, **kwargs: Any) -> None:
        """Initialise the CircularPlaneParrallel source."""
        super().__init__(**kwargs)

        self._x0: float = x0
        self._y0: float = y0
        self._z0: float = z0
        self._u0: float = u0
        self._v0: float = v0
        self._w0: float = w0
        self._radius: float = radius

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP ``SDEF`` disc source card with radial (D2) sub-cards."""
        # get enregy distributions
        if not isinstance(self._energy, SourceEnergyDistribution):
            energy = SourceEnergyDistribution([self._energy], [1.0])
        else:
            energy = self._energy
        energy_subcard_1, energy_subcard_2 = energy.mcnp_subcards

        # strings for positoin and axis
        position_string = f'POS={self._x0:.{formatting.MCNP_DP}e} {self._y0:.{formatting.MCNP_DP}e} {self._z0:.{formatting.MCNP_DP}e}'
        axis_string = f'AXS={self._u0:.{formatting.MCNP_DP}e} {self._v0:.{formatting.MCNP_DP}e} {self._w0:.{formatting.MCNP_DP}e}'
        vector_string = f'VEC={self._u0:.{formatting.MCNP_DP}e} {self._v0:.{formatting.MCNP_DP}e} {self._w0:.{formatting.MCNP_DP}e}'

        mcnp_card = MCNPCard('SDEF', [[f'PAR={_particle_designator(self._particle)}'],
                                      [f'ERG=D1'],
                                      [position_string],
                                      [axis_string],
                                      [vector_string],
                                      ['EXT=0'],
                                      ['DIR=1'],
                                      ['RAD=D2']])

        # add energy distributions subcards
        mcnp_card.add_subcard(energy_subcard_1)
        mcnp_card.add_subcard(energy_subcard_2)

        # add radial distribution subcard
        mcnp_card.add_subcard(MCNPCard('SI2', [[f'{0.0:.{formatting.MCNP_DP}e} {self._radius:.{formatting.MCNP_DP}e}']]))
        mcnp_card.add_subcard(MCNPCard('SP2', [['-21 1']]))

        return mcnp_card
