"""Readers for MCNP input and output files.

Contains readers for MCNP input decks (cells, surfaces, transforms and
materials), MCTAL tally output files and PTRAC source tracks, together with
supporting tally and source sample containers. Several classes are partially
implemented work in progress.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .surface import Surface
    from .volume import HalfVolume, Volume

import re
import numpy as np
import os


from .surface import *
from .macrobody import *
from .cell import Cell
from .model import Model
from .transform import Transform


MCNP_SURFACE_TYPES: dict[str, type] = {'PX': XPlane,
                      'PY': YPlane,
                      'PZ': ZPlane,
                      'P': Plane,
                      'P3P': PlaneFromThreePoints,
                      'C/X': CylinderParallelToXAxis,
                      'C/Y': CylinderParallelToYAxis,
                      'C/Z': CylinderParallelToZAxis,
                      'CX': CylinderOnXAxis,
                      'CY': CylinderOnYAxis,
                      'CZ': CylinderOnZAxis,
                      'KX': ConeOnXAxis,
                      'KY': ConeOnYAxis,
                      'KZ': ConeOnZAxis,
                      'K/X': ConeParallelToXAxis,
                      'K/Y': ConeParallelToZAxis,
                      'K/Z': ConeParallelToZAxis,
                      'SO': SphereAtOrigin,
                      'S': Sphere,
                      'SX': SphereOnXAxis,
                      'SY': SphereOnYAxis,
                      'SZ': SphereOnZAxis,
                      'SQ': Ellipsoid,
                      'GQ': GeneralQuadratic,
                      'TX': ToriParralleltoXAxis,
                      'TY': ToriParralleltoYAxis,
                      'TZ': ToriParralleltoZAxis,
                      'BOX': Box,
                      'RPP': RectangularParallelepiped,
                      'RCC': RightCircularCylinder}



class MCNPReader:
    """Early (incomplete) reader for MCNP input decks.

    Splits the input file into cell, surface and run parameter sections on blank
    lines and parses the run parameter cards into :attr:`cards`. Cell and surface
    parsing are not implemented.

    Parameters
    ----------
    input_filename : str
        Path of the MCNP input file.

    Attributes
    ----------
    cards : dict[str, dict[Any, Any]]
        Parsed run parameter cards keyed by card type then card label.
    """

    def __init__(self, input_filename: str) -> None:
        """Initialise the reader and parse the input file."""
        self.cards: dict[str, dict[Any, Any]] = {'materials': {},
                      'tallies': {},
                      'fmesh': {},
                      'sdef': {},
                      'nps': {},
                      'phys': {},
                      'cut': {},
                      'rand': {},
                      'mode': {},
                      'prdmp': {}}

        with open(input_filename, 'r') as input_file:
            input_rows = [row for row in input_file]

        # Parse cells
        section_seperators = [-1] + [i for i, val in enumerate(input_rows) if val == '\n'] + [len(input_rows)]

        cells_section = [row.strip('\n').lstrip().split('$', 1)[0].rstrip() for row in input_rows[section_seperators[-4] + 1: section_seperators[-3]] if row.lstrip()[0] not in ['c', 'C']]
        surfaces_section = [row.strip('\n').lstrip().split('$', 1)[0].rstrip() for row in input_rows[section_seperators[-3] + 1: section_seperators[-2]] if row.lstrip()[0] not in ['c', 'C']]
        run_parameters_section = [row.strip('\n').lstrip().split('$', 1)[0].rstrip() for row in input_rows[section_seperators[-2] + 1: section_seperators[-1]] if row.lstrip()[0] not in ['c', 'C']]

        self._parse_cells(cells_section)
        self._parse_surfaces(surfaces_section)
        self._parse_run_parameters(run_parameters_section)

    def _parse_cells(self, cells_rows: list[str]) -> None:
        """Parse the cell section (not implemented).

        Parameters
        ----------
        cells_rows : list[str]
            Rows of the cell section with comments removed.
        """
        pass

    def _parse_surfaces(self, surface_rows: list[str]) -> None:
        """Parse the surface section (not implemented).

        Parameters
        ----------
        surface_rows : list[str]
            Rows of the surface section with comments removed.
        """
        pass

    def _parse_run_parameters(self, run_parameters_rows: list[str]) -> None:
        """Parse the run parameter section into :attr:`cards`.

        Rows are grouped into cards by matching the start of each row against
        known card types; each card is then parsed by the matching card parser.
        Only material cards are currently parsed, the others are placeholders.

        Parameters
        ----------
        run_parameters_rows : list[str]
            Rows of the run parameter (data) section with comments removed.
        """
        def parse_material(material_rows: list[str]) -> tuple[int, dict[str, dict[str, Any]]]:
            """Parse a material card.

            Parameters
            ----------
            material_rows : list[str]
                Rows making up the material card.

            Returns
            -------
            tuple[int, dict[str, dict[str, Any]]]
                Material number and composition keyed by zaid, each entry holding
                ``'fraction'`` and ``'library'``.
            """
            material_split = ' '.join(material_rows).split()
            material_number = int(material_split[0][1:])
            composition: dict[str, dict[str, Any]] = {}
            composition_library: dict[str, str] = {}
            for isotope, fraction in zip(material_split[1::2], material_split[2::2]):
                if '.' in isotope:
                    zaid, library = isotope.split('.')
                else:
                    zaid, library = isotope, ''
                composition[zaid] = {'fraction': float(fraction), 
                                     'library': library}

            return material_number, composition

        def parse_tally(tally_rows: list[str]) -> tuple[str, dict[Any, Any]]:
            """Parse a tally card (placeholder).

            Parameters
            ----------
            tally_rows : list[str]
                Rows making up the card.

            Returns
            -------
            tuple[str, dict[Any, Any]]
                Placeholder label ``'a'`` and an empty dictionary.
            """
            return 'a', {}
        
        def parse_fmesh(fmesh_rows: list[str]) -> tuple[str, dict[Any, Any]]:
            """Parse a fmesh card (placeholder).

            Parameters
            ----------
            fmesh_rows : list[str]
                Rows making up the card.

            Returns
            -------
            tuple[str, dict[Any, Any]]
                Placeholder label ``'a'`` and an empty dictionary.
            """
            return 'a', {}

        def parse_sdef(sdef_rows: list[str]) -> tuple[str, dict[Any, Any]]:
            """Parse a sdef card (placeholder).

            Parameters
            ----------
            sdef_rows : list[str]
                Rows making up the card.

            Returns
            -------
            tuple[str, dict[Any, Any]]
                Placeholder label ``'a'`` and an empty dictionary.
            """
            return 'a', {}

        def parse_nps(nps_rows: list[str]) -> tuple[str, dict[Any, Any]]:
            """Parse a nps card (placeholder).

            Parameters
            ----------
            nps_rows : list[str]
                Rows making up the card.

            Returns
            -------
            tuple[str, dict[Any, Any]]
                Placeholder label ``'a'`` and an empty dictionary.
            """
            return 'a', {}

        def parse_phys(phys_rows: list[str]) -> tuple[str, dict[Any, Any]]:
            """Parse a phys card (placeholder).

            Parameters
            ----------
            phys_rows : list[str]
                Rows making up the card.

            Returns
            -------
            tuple[str, dict[Any, Any]]
                Placeholder label ``'a'`` and an empty dictionary.
            """
            return 'a', {}

        def parse_cut(cut_rows: list[str]) -> tuple[str, dict[Any, Any]]:
            """Parse a cut card (placeholder).

            Parameters
            ----------
            cut_rows : list[str]
                Rows making up the card.

            Returns
            -------
            tuple[str, dict[Any, Any]]
                Placeholder label ``'a'`` and an empty dictionary.
            """
            return 'a', {}

        def parse_rand(rand_rows: list[str]) -> tuple[str, dict[Any, Any]]:
            """Parse a rand card (placeholder).

            Parameters
            ----------
            rand_rows : list[str]
                Rows making up the card.

            Returns
            -------
            tuple[str, dict[Any, Any]]
                Placeholder label ``'a'`` and an empty dictionary.
            """
            return 'a', {}

        def parse_mode(mode_rows: list[str]) -> tuple[str, dict[Any, Any]]:
            """Parse a mode card (placeholder).

            Parameters
            ----------
            mode_rows : list[str]
                Rows making up the card.

            Returns
            -------
            tuple[str, dict[Any, Any]]
                Placeholder label ``'a'`` and an empty dictionary.
            """
            return 'a', {}

        def parse_prdmp(prdmp_rows: list[str]) -> tuple[str, dict[Any, Any]]:
            """Parse a prdmp card (placeholder).

            Parameters
            ----------
            prdmp_rows : list[str]
                Rows making up the card.

            Returns
            -------
            tuple[str, dict[Any, Any]]
                Placeholder label ``'a'`` and an empty dictionary.
            """
            return 'a', {}

        re_cards = {'materials': re.compile(r'^[mM]+[0-9]+', re.IGNORECASE),
                    'tallies': re.compile(r'^f[0-9]+[:]+[a-z]+', re.IGNORECASE),
                    'fmesh': re.compile(r'^fmesh', re.IGNORECASE),
                    'sdef': re.compile(r'^sdef', re.IGNORECASE),
                    'nps': re.compile(r'^nps', re.IGNORECASE),
                    'phys': re.compile(r'^phys:[1-z]+', re.IGNORECASE),
                    'cut': re.compile(r'^cut', re.IGNORECASE),
                    'rand': re.compile(r'^rand', re.IGNORECASE),
                    'mode': re.compile(r'^mode', re.IGNORECASE),
                    'prdmp': re.compile(r'^prdmp', re.IGNORECASE)}

        parse_card = {'materials': parse_material,
                      'tallies': parse_tally,
                      'fmesh': parse_fmesh,
                      'sdef': parse_sdef,
                      'nps': parse_nps,
                      'phys': parse_phys,
                      'cut': parse_cut,
                      'rand': parse_rand,
                      'mode': parse_mode,
                      'prdmp': parse_prdmp}

        for i, row in enumerate(run_parameters_rows):
            for potential_card_type, re_card in re_cards.items():
                if bool(re_card.search(row)):
                    if 'card' in locals():
                        label, data = parse_card[card_type](card)
                        self.cards[card_type][label] = data
                    card = []
                    card_type = potential_card_type
                    break
            card.append(row)
        label, data = parse_card[card_type](card)
        self.cards[card_type][label] = data

        # card_indexes = [(i) for i, row in enumerate(run_parameters_rows) if any([bool(v.search(row)) for k, v in re_cards.items()])]


class Mesh():
    """Rectangular mesh boundaries with energy and time bins.

    Parameters
    ----------
    x_bounds : list[float]
        Mesh boundaries along x.
    y_bounds : list[float]
        Mesh boundaries along y.
    z_bounds : list[float]
        Mesh boundaries along z.
    energy_bins : list[float], optional
        Energy bin boundaries, by default [0, 1e36]
    time_bins : list[float], optional
        Time bin boundaries, by default [0, 1e36]
    """

    def __init__(self,
                 x_bounds: list[float],
                 y_bounds: list[float],
                 z_bounds: list[float],
                 energy_bins: list[float] = [0,1e36],
                 time_bins: list[float] = [0,1e36]) -> None:
        """Initialise the Mesh, storing the boundaries as float arrays."""
        self._x_bounds: np.ndarray = np.array(x_bounds, dtype=float)
        self._y_bounds: np.ndarray = np.array(y_bounds, dtype=float)
        self._z_bounds: np.ndarray = np.array(z_bounds, dtype=float)
        self._t_bins: np.ndarray = np.array(time_bins, dtype=float)
        self._e_bins: np.ndarray = np.array(energy_bins, dtype=float)


class MCNPMeshtalreader():
    """Reader for MCNP MESHTAL files (not implemented).

    Parameters
    ----------
    input_path : os.PathLike
        Path of the MESHTAL file.
    """

    def __init__(self, input_path: os.PathLike) -> None:
        """Initialise the reader and read the MESHTAL file."""
        self.mesh: None = self._read_meshtal(input_path)

    def _read_meshtal(input_path: os.PathLike) -> None:
        """Read a MESHTAL file (not implemented).

        Note this method lacks a ``self`` parameter, so calling it on an instance
        passes the instance as ``input_path``.

        Parameters
        ----------
        input_path : os.PathLike
            Path of the MESHTAL file.
        """
        with open(input_path, 'r') as input_file:
            pass


# class Mesh():
#     def __init__(self, 
#                  x_bounds, 
#                  y_bounds, 
#                  z_bounds, 
#                  energy_bins, 
#                  np_dtype='float64'):

#         self._number = None
#         self._particle = None
#         self._np_dtype = np_dtype

#         self._x_boundaries = x_bounds
#         self._y_boundaries = y_bounds
#         self._z_boundaries = z_bounds
#         self._energy_bins = energy_bins

#         mesh_shape = (len(self._x_boundaries) - 1,
#                       len(self._y_boundaries) - 1,
#                       len(self._z_boundaries) - 1,
#                       len(self._energy_bins))

#         self.values = np.zeros(shape=mesh_shape, dtype=np_dtype)
#         self.uncertainties = np.zeros(shape=mesh_shape, dtype=self._np_dtype)

#     @property
#     def x_boundaries(self):
#         return self._x_boundaries

#     @x_boundaries.setter
#     def x_boundaries(self, x_boundaries):
#         self._x_boundaries = np.array(x_boundaries, dtype=self._np_dtype)

#     @property
#     def y_boundaries(self):
#         return self._y_boundaries

#     @x_boundaries.setter
#     def y_boundaries(self, y_boundaries):
#         self._y_boundaries = np.array(y_boundaries, dtype=self._np_dtype)
        
#     @property
#     def z_boundaries(self):
#         return self._z_boundaries

#     @z_boundaries.setter
#     def z_boundaries(self, z_boundaries):
#         self._z_boundaries = np.array(z_boundaries, dtype=self._np_dtype)

#     @property
#     def energy_bins(self):
#         return self._energy_bins

#     @energy_bins.setter
#     def energy_bins(self, energy_bins):
#         self._energy_bins = energy_bins

#     @property
#     def number(self):
#         return self._number

#     @number.setter
#     def number(self, number):
#         self._number = number

#     @property
#     def particle(self):
#         return self._particle

#     @particle.setter
#     def particle(self, particle):
#         self._particle = particle

#     def __add__(self, mesh_to_add):
#         # check that x, y, z and e bounds are the same
#         if set(self.energy_bins) == set(mesh_to_add.energy_bins) and \
#            set(self.x_boundaries) == set(mesh_to_add.x_boundaries) and \
#            set(self.y_boundaries) == set(mesh_to_add.y_boundaries) and \
#            set(self.z_boundaries) == set(mesh_to_add.z_boundaries):
#             mesh = Mesh(self.x_boundaries,
#                         self.y_boundaries,
#                         self.z_boundaries,
#                         self.energy_bins)
#             mesh.particle = f'{self.particle} & {mesh_to_add.particle}'
#             mesh.values = self.values + mesh_to_add.values
#             mesh.uncertainties = self.uncertainties + mesh_to_add.uncertainties
#             return mesh
#         else:
#             raise Exception('Boundaries, energy bins or particle types not equal' 
#                             'not the same across meshes')
    
#     def __mul__(self, multiplier):
#         mesh = Mesh(self.x_boundaries,
#                     self.y_boundaries,
#                     self.z_boundaries,
#                     self.energy_bins)
#         mesh.particle = self.particle
#         self.values *= multiplier
#         self.uncertainties *= multiplier
#         return mesh

#     def __rmul__(self, multiplier):
#         return self.__mul__(multiplier)


# class MCNPMeshtalReader:
#     def __init__(self, input_filename):
#         self._meshes = {}
#         self._read_meshtal(input_filename)

#     @property
#     def meshes(self):
#         return self._meshes

#     def _read_meshtal(self, input_filename):
#         read_parsers = {'header': self._parse_header_row}

#         vars = {'tally_number': None,
#                 'x_bounds': None,
#                 'y_bounds': None,
#                 'z_bounds': None,
#                 'energy_bins': None,
#                 'particle': None}

#         with open(input_filename, 'r') as input_file:
#             # read the first 3 comment lines
#             for i, row in enumerate(input_file, start=1):
#                 if i > 3: 
#                     break

#             read_mode = 'header'

#             # read mesh tallies
#             for i, row in enumerate(input_file, start=1):
#                 read_mode, data_label, data = self._parse_header_row(row)
#                 # case fo reading a header line
#                 if read_mode == 'header' and data_label in vars:
#                     vars[data_label] = data

#                 # case for reading a value from a single energy group tally
#                 elif read_mode == 'data_single':
#                     mesh = Mesh(vars['x_bounds'],
#                                 vars['y_bounds'],
#                                 vars['z_bounds'],
#                                 vars['energy_bins'])
#                     mesh.number = vars['tally_number']
#                     mesh.particle = vars['particle']
#                     self._meshes[vars['tally_number']] = mesh

#                     self._parse_single_energy_data(input_file, mesh)

#                     read_mode = 'header'
#                     vars = {v: None for v in vars}

#                 # case for reading a value from a multiple energy group tally
#                 elif read_mode == 'data_multiple':
#                     mesh = Mesh(vars['x_bounds'],
#                                 vars['y_bounds'],
#                                 vars['z_bounds'],
#                                 [*vars['energy_bins'], 'Total'])
#                     mesh.number = vars['tally_number']
#                     mesh.particle = vars['particle']
#                     self._meshes[vars['tally_number']] = mesh

#                     self._parse_multiple_energy_data(input_file, mesh)
                    
#                     read_mode = 'header'
#                     vars = {v: None for v in vars}
#                 else:
#                     continue

#     def _parse_mesh_tally_number(self, row):
#         return 'header', int(row.split()[3])

#     def _parse_boundaries(self, row):
#         return 'header', [float(v) for v in row.split()[2:]]

#     def _parse_energy_bins(self, row):
#         return 'header', [float(v) for v in row.split()[3:]]

#     def _parse_particle(self, row):
#         return 'header', row.split()[0]

#     def _parse_data_header_single_energy(self, row):
#         return 'data_single', None
    
#     def _parse_data_header_multiple_energy(self, row):
#         return 'data_multiple', None

#     def _parse_header_row(self, row):
#         re_expressions = {'tally_number': re.compile(r'^\s*Mesh Tally Number'),
#                           'x_bounds': re.compile(r'^\s*X direction'),
#                           'y_bounds': re.compile(r'^\s*Y direction'),
#                           'z_bounds': re.compile(r'^\s*Z direction'),
#                           'energy_bins': re.compile(r'^\s*Energy bin boundaries'),
#                           'particle': re.compile(r'^\s*[a-z]{1,}\s*mesh tally'),
#                           'data_single': re.compile(r'^\s{1,}X\s*Y\s*Z'),
#                           'data_multiple': re.compile(r'^\s{1,}Energy\s+X\s*Y\s*Z')}
        
#         parse_functions = {'tally_number': self._parse_mesh_tally_number,
#                            'x_bounds': self._parse_boundaries,
#                            'y_bounds': self._parse_boundaries,
#                            'z_bounds': self._parse_boundaries,
#                            'energy_bins': self._parse_energy_bins,
#                            'particle': self._parse_particle,
#                            'data_single': self._parse_data_header_single_energy,
#                            'data_multiple': self._parse_data_header_multiple_energy}

#         for parameter_label, rr_expression in re_expressions.items():
#             if bool(rr_expression.search(row)):
#                 read_mode, row_data = parse_functions[parameter_label](row)
#                 return read_mode, parameter_label, row_data
        
#         # case where no information extracted from line
#         return 'header', None, None

#     def _parse_single_energy_data(self, input_file, mesh):
#         # read data values into mesh
#         x, y, z, e  = 0, 0, 0, 0
#         for x in range(len(mesh.x_boundaries)-1):
#             for y in range(len(mesh.y_boundaries)-1):
#                 for z in range(len(mesh.z_boundaries)-1):
#                     sub_row_split = input_file.readline().split()
#                     _, _, _, value, uncertainty = sub_row_split
#                     mesh.values[x][y][z][-1] = value
#                     mesh.uncertainties[x][y][z][-1] = uncertainty
    
#     def _parse_multiple_energy_data(self, input_file, mesh):

#         e = 0
#         while True:
#             try:
#                 # read data values into mesh
#                 x, y, z  = 0, 0, 0
#                 for x in range(len(mesh.x_boundaries)-1):
#                     for y in range(len(mesh.y_boundaries)-1):
#                         for z in range(len(mesh.z_boundaries)-1):
#                             sub_row_split = input_file.readline().split()
#                             energy, _, _, _, value, uncertainty = sub_row_split
#                             mesh.values[x][y][z][e] = value
#                             mesh.uncertainties[x][y][z][e] = uncertainty
#                 e += 1
#             except:
#                 print(sub_row_split)
#                 break

class MCNPTally:
    """Container for a tally read from an MCNP MCTAL file.

    Parameters
    ----------
    tally_number : int
        MCNP tally number.
    particle_type : list[str]
        Particle types of the tally.
    comment : str | None, optional
        Tally comment, by default None

    Attributes
    ----------
    MCNP_PARTICLES : list[str]
        Valid MCNP particle names.
    particle : list[str]
        Particle types of the tally.
    """

    def __init__(self,
                 tally_number: int,
                 particle_type: list[str],
                 comment: str | None = None) -> None:
        """Initialise the MCNPTally."""
        self.MCNP_PARTICLES: list[str] = ['neutron',
                               'photon',
                               'electron',
                               'negative_muon', 
                               'anti_neutron', 
                               'electron_neutrino',
                               'muon_neutrino',
                               'positron',
                               'proton',
                               'lambda_baryon',
                               'positive_sigma_baryon',
                               'negative_sigma_baryon',
                               'cascade xi baryon',
                               'negative_cascade_nagative_xi_baryon',
                               'omega_baryon',
                               'positive_muon',
                               'anti_electron_neutrino',
                               'anti_muon_neutrino',
                               'anti_proton',
                               'positive_pion',
                               'neutral_pion',
                               'positive_kaon',
                               'kaon_short',
                               'kaon_long',
                               'anti_lambda_baryon',
                               'anti_positive_sigma_baryon',
                               'anti_negative_sigma_baryon',
                               'anti_cascade_anti_neutral_xi_baryon',
                               'positive_cascade_positive_xi_baryon',
                               'anti_omega',
                               'deuteron',
                               'triton',
                               'helion',
                               'alpha',
                               'negative_pion',
                               'negative_kaon',
                               'heavy_ion']

        self.number = tally_number
        self.particle: list[str] = particle_type
        self.comment = comment

    @property
    def number(self) -> int:
        """MCNP tally number.

        Returns
        -------
        int
            Tally number.
        """
        return self._number

    @number.setter
    def number(self, tally_number: int) -> None:
        """Set the tally number.

        Parameters
        ----------
        tally_number : int
            Tally number.

        Raises
        ------
        TypeError
            If ``tally_number`` is not an int.
        """
        if type(tally_number) == int:
            self._number = tally_number
        else:
            raise TypeError('Tally number is not an int')

    @property
    def particles(self) -> list[str]:
        """Particle types of the tally.

        Returns
        -------
        list[str]
            Particle names.
        """
        return self._particles

    @particles.setter
    def particles(self, particles: list[str]) -> None:
        """Set the particle types after validating them.

        Note the value is stored in ``_particle2``, not ``_particles``.

        Parameters
        ----------
        particles : list[str]
            Particle names, each of which must be in :attr:`MCNP_PARTICLES`.

        Raises
        ------
        Exception
            If any particle name is not a valid MCNP particle.
        """
        if len([particle for particle in particles if particle in self.MCNP_PARTICLES]) == len(particles):
            self._particle2 = particles
        else:
            raise Exception(f'Particle type incorrect: {particles}')

    @property
    def cells(self) -> dict[int, dict[str, np.ndarray]]:
        """Tally values and uncertainties for each cell.

        Returns
        -------
        dict[int, dict[str, np.ndarray]]
            ``{'values': ..., 'uncertainties': ...}`` arrays keyed by cell number.
        """
        return self._cells

    @cells.setter
    def cells(self, cells: dict[int, dict[str, np.ndarray]]) -> None:
        """Set the tally values and uncertainties for each cell.

        Parameters
        ----------
        cells : dict[int, dict[str, np.ndarray]]
            ``{'values': ..., 'uncertainties': ...}`` arrays keyed by cell number.
        """
        self._cells = cells

    @property
    def energy_bins(self) -> list[float]:
        """Upper energy bin boundaries of the tally.

        Returns
        -------
        list[float]
            Energy bin boundaries.
        """
        return self._energy_bins

    @energy_bins.setter
    def energy_bins(self, energy_bins: list[float]) -> None:
        """Set the energy bin boundaries.

        Parameters
        ----------
        energy_bins : list[float]
            Energy bin boundaries.
        """
        self._energy_bins = energy_bins
    
    @property
    def comment(self) -> str | None:
        """Tally comment.

        Returns
        -------
        str or None
            Tally comment.
        """
        return self._comment

    @comment.setter
    def comment(self, comment: str | None) -> None:
        """Set the tally comment.

        Parameters
        ----------
        comment : str | None
            Tally comment.
        """
        self._comment = comment


class MCNPMCTALReader:
    """Reader for MCNP MCTAL tally output files.

    Parameters
    ----------
    input_filename : str
        Path of the MCTAL file.

    Attributes
    ----------
    PARTICLE_TYPES : dict[str, str]
        Particle type code to particle letters lookup.
    PARTICLE_ROW_ORDER : list[str]
        Particle names in the order of the MCTAL particle flag row.
    DETECTOR_TYPES : dict[str, str]
        Detector type code to description lookup.
    run_parameters : dict[str, str]
        Run information from the first line of the file.
    comment : str
        Problem title line.
    number_of_tallies : str
        Number of tallies, as read from the file.
    """

    def __init__(self, input_filename: str) -> None:
        """Initialise the reader and read the MCTAL file."""
        self.PARTICLE_TYPES: dict[str, str] = {'1': 'n',
                               '2': 'p',
                               '3': 'np',
                               '4': 'e',
                               '5': 'ne',
                               '6': 'pe',
                               '7': 'npe'}

        self.PARTICLE_ROW_ORDER: list[str] = ['neutron', 
                                   'photon', 
                                   'electron', 
                                   'negative_muon', 
                                   'anti_neutron', 
                                   'electron_neutrino',
                                   'muon_neutrino',
                                   'positron',
                                   'proton',
                                   'lambda_baryon',
                                   'positive_sigma_baryon',
                                   'negative_sigma_baryon',
                                   'cascade_xi_baryon',
                                   'negative_cascade_negative_xi_baryon',
                                   'omega_baryon',
                                   'positive_muon',
                                   'anti_electron_neutrino',
                                   'anti_muon_neutrino',
                                   'anti_proton',
                                   'positive_pion',
                                   'neutral_pion',
                                   'positive_kaon',
                                   'kaon_short',
                                   'kaon_long',
                                   'anti_lambda_baryon',
                                   'anti_positive_sigma_baryon',
                                   'anti_negative_sigma_baryon',
                                   'anti_cascade_anti_neutral_xi_baryon',
                                   'positive_cascade_positive_xi_baryon',
                                   'anti_omega',
                                   'deuteron',
                                   'triton',
                                   'helion',
                                   'alpha',
                                   'negative_pion',
                                   'negative_kaon',
                                   'heavy_ion']

        self.DETECTOR_TYPES: dict[str, str] = {'0': 'None',
                               '1': 'point',
                               '2': 'ring',
                               '3': 'pinhole (rectangular grid, FIC)',
                               '4': 'transmitted image radiograph (cylindrical grid, FIC)'}

        self._tallies: dict[int, MCNPTally | None] = {}

        self._read_mctal_file(input_filename)

    @property
    def tallies(self) -> dict[int, MCNPTally | None]:
        """Tallies keyed by tally number.

        Returns
        -------
        dict[int, MCNPTally | None]
            Parsed tallies.
        """
        return self._tallies
    
    @property
    def tallies_by_comment(self) -> dict[str | None, MCNPTally]:
        """Tallies keyed by tally comment.

        Returns
        -------
        dict[str | None, MCNPTally]
            Parsed tallies keyed by their comment.

        Raises
        ------
        Exception
            If two tallies share the same comment.
        """
        tallies_by_comment: dict[str | None, MCNPTally] = {}
        for tally in self._tallies.values():
            if tally.comment not in tallies_by_comment:
                tallies_by_comment[tally.comment] = tally
            else:
                msg = f'tally comment {tally.comment} duplicate found'
                raise Exception(msg)
        return tallies_by_comment
        
    def _read_mctal_file(self, input_filename: str) -> None:
        """Read the MCTAL file, populating the run information and tallies.

        Parameters
        ----------
        input_filename : str
            Path of the MCTAL file.
        """
        with open(input_filename, 'r') as input_file:
            rows = [row for row in input_file]
        
        # retrieve run parameters
        keys = ['code', 'version', 'date', 'time',
                'dump', 'nps', 'random_seed']
        self.run_parameters: dict[str, str] = {k: v for k, v in zip(keys, rows[0].split())}

        # retrieve comment
        self.comment: str = rows[1]

        # retrieve number of tallies and purtibations
        self.number_of_tallies: str = rows[2].split()[0]

        for row in rows[3:]:
            if 'tally' not in row:
                self._tallies = {**self._tallies, **{int(k): None for k in row.split()}}
            else:
                break

        # find starting position of each tally section

        tally_start_indices = [i+3 for i, row in enumerate(rows[3:]) if \
            ('tally' in row and row[0] == 't' and \
                int(row.split()[1]) in self._tallies)]
        tally_start_indices.append(len(rows))

        for start, stop in zip(tally_start_indices[:-1], tally_start_indices[1:]):
            tally_rows = [row.strip('\n') for row in rows[start:stop]]
            tally = self._parse_tally(tally_rows)
            self._tallies[tally.number] = tally

    def _parse_tally(self, rows: list[str]) -> MCNPTally:
        """Parse the rows of a single tally section.

        Rows are grouped by card type (``tally``, ``f``, ``d``, ``u``, ``s``,
        ``m``, ``c``, ``e``, ``t``, ``vals``, ``tfc``); the tally header, cell
        list, energy bins and values are then parsed.

        Parameters
        ----------
        rows : list[str]
            Rows of the tally section, without newline characters.

        Returns
        -------
        MCNPTally
            Parsed tally.
        """
        def parse_tally(rows: list[str]) -> tuple[int, list[str], str, str, str | None]:
            """Parse the tally header card.

            Parameters
            ----------
            rows : list[str]
                Rows of the ``tally`` card.

            Returns
            -------
            tuple[int, list[str], str, str, str | None]
                Tally number, particle types, detector type, tally modifier and
                tally comment.
            """
            _, tally_number, i, j, k = rows[0].split()

            if int(i) < 0:
                particle_row = [int(v) for v in rows[1].split()]
                particle_types = [q for p, q in zip(particle_row, self.PARTICLE_ROW_ORDER) if p==1]
            else:
                particle_types = [self.PARTICLE_TYPES[i]]

            detector_type = self.DETECTOR_TYPES[j]
            modifier = k

            if len(rows) > 2:
                tally_comment = rows[2].lstrip().rstrip()
            else:
                tally_comment = None

            return int(tally_number), particle_types, detector_type, modifier, tally_comment
        
        def parse_f(rows: list[str]) -> list[int]:
            """Parse the cell / surface list card.

            Parameters
            ----------
            rows : list[str]
                Rows of the ``f`` card.

            Returns
            -------
            list[int]
                Cell or surface numbers.
            """
            cells = [v for row in rows for v in row.split()]
            
            return [int(v) for v in cells[2:]]
        
        def parse_d(rows: list[str]) -> None:
            """Parse the ``d`` card (not implemented).

            Parameters
            ----------
            rows : list[str]
                Rows of the card.
            """
            pass

        def parse_u(rows: list[str]) -> None:
            """Parse the ``u`` card (not implemented).

            Parameters
            ----------
            rows : list[str]
                Rows of the card.
            """
            pass

        def parse_s(rows: list[str]) -> None:
            """Parse the ``s`` card (not implemented).

            Parameters
            ----------
            rows : list[str]
                Rows of the card.
            """
            pass

        def parse_m(rows: list[str]) -> None:
            """Parse the ``m`` card (not implemented).

            Parameters
            ----------
            rows : list[str]
                Rows of the card.
            """
            pass
        
        def parse_c(rows: list[str]) -> None:
            """Parse the ``c`` card (not implemented).

            Parameters
            ----------
            rows : list[str]
                Rows of the card.
            """
            pass

        def parse_e(rows: list[str]) -> list[float]:
            """Parse the energy bin card.

            Parameters
            ----------
            rows : list[str]
                Rows of the ``e`` card.

            Returns
            -------
            list[float]
                Energy bin boundaries, or ``[1e99]`` if there are no bins.
            """
            e_vals = [v for row in rows for v in row.split()]
            n_bins = int(e_vals[1])
            
            if n_bins > 0:
                energy_bins = [float(v) for v in e_vals[2:]]
            else:
                energy_bins = [1e99]

            return energy_bins
        
        def parse_t(rows: list[str]) -> None:
            """Parse the ``t`` (time) card (not implemented).

            Parameters
            ----------
            rows : list[str]
                Rows of the card.
            """
            pass

        def parse_vals(rows: list[str],
                       cells: list[int]) -> dict[int, dict[str, np.ndarray]]:
            """Parse the tally values and relative uncertainties.

            Values are split evenly between the cells in order.

            Parameters
            ----------
            rows : list[str]
                Rows of the ``vals`` card.
            cells : list[int]
                Cell numbers of the tally.

            Returns
            -------
            dict[int, dict[str, np.ndarray]]
                ``{'values': ..., 'uncertainties': ...}`` arrays keyed by cell.
            """
            s = [float(v) for row in rows[1:] for v in row.split()]

            values = s[::2]
            uncertainties = s[1::2]

            n_vals_per_cell = int(len(values) / len(cells))

            values_by_cell = {cell: values[i*n_vals_per_cell:(i+1)*n_vals_per_cell] for i, cell in enumerate(cells)}
            uncertainties_by_cell = {cell: uncertainties[i*n_vals_per_cell:(i+1)*n_vals_per_cell] for i, cell in enumerate(cells)}

            vals_and_uncertainties_by_cell: dict[int, dict[str, np.ndarray]] = {}
            
            for i, cell in enumerate(cells):
                val = np.array(values_by_cell[cell])
                unc = np.array(uncertainties_by_cell[cell])
                vals_and_uncertainties_by_cell[cell] = {'values': val, 'uncertainties': unc}

            # vals_and_uncertainties_by_cell = {cell: {'values': val, 'uncertainties': unc} for cell, val, unc in zip(cells, values_by_cell.values(), uncertainties_by_cell.values())}

            return vals_and_uncertainties_by_cell

        def parse_tfc(rows: list[str]) -> None:
            """Parse the ``tfc`` (tally fluctuation chart) card (not implemented).

            Parameters
            ----------
            rows : list[str]
                Rows of the card.
            """
            pass

        re_cards = {'tally': re.compile(r'^tally\s+[0-9]{1,}'),
                    'f': re.compile(r'^f\s+[0-9]{1,}'),
                    'd': re.compile(r'^d\s+[0-9]{1,}'),
                    'u': re.compile(r'^(?:u|ut|uc)\s+[0-9]{1,}'),
                    's': re.compile(r'^(?:s|st|sc)\s+[0-9]{1,}'),
                    'm': re.compile(r'^(?:m|mt|mc)\s+[0-9]{1,}'),
                    'c': re.compile(r'^(?:c|ct|cc)\s+[0-9]{1,}'),
                    'e': re.compile(r'^(?:e|et|ec)\s+[0-9]{1,}'),
                    't': re.compile(r'^(?:t|tt|tc)\s\s+[0-9]{1,}'),
                    'vals': re.compile(r'^(?:vals|vals pert)'),
                    'tfc': re.compile(r'^tfc\s+[0-9]{1,}')}

        parse_functions = {'tally': parse_tally,
                           'f': parse_f,
                           'd': parse_d,
                           'u': parse_u,
                           's': parse_s,
                           'm': parse_m,
                           'c': parse_c,
                           'e': parse_e,
                           't': parse_t,
                           'vals': parse_vals,
                           'tfc': parse_tfc}
        
        card_rows: dict[str, list[str]] = {v: [] for v in re_cards.keys()}

        for row in rows:
            for potential_card_type, re_card in re_cards.items():
                if bool(re_card.search(row)):
                    card_type = potential_card_type
            card_rows[card_type].append(row)

            

        # parse tally number and type
        number, particle, detector, modifier, comment = parse_tally(card_rows['tally'])
        cells = parse_f(card_rows['f'])
    
        # load data into tally class instance
        tally = MCNPTally(number, particle)
        tally.cells = parse_vals(card_rows['vals'], cells)
        tally.energy_bins = parse_e(card_rows['e'])
        tally.comment = comment
        
        return tally


class SourceDistributionSample:
    """Sampled source particle positions, directions, energies, weights and times.

    Setters validate that values are a list or tuple of real values.

    Parameters
    ----------
    x : Sequence[float]
        x coordinates.
    y : Sequence[float]
        y coordinates.
    z : Sequence[float]
        z coordinates.
    u : Sequence[float]
        x direction cosines.
    v : Sequence[float]
        y direction cosines.
    w : Sequence[float]
        z direction cosines.
    e : Sequence[float]
        Energies.
    wgt : Sequence[float]
        Particle weights.
    t : Sequence[float]
        Times.
    """

    def __init__(self,
                 x: Sequence[float],
                 y: Sequence[float],
                 z: Sequence[float],
                 u: Sequence[float],
                 v: Sequence[float],
                 w: Sequence[float],
                 e: Sequence[float],
                 wgt: Sequence[float],
                 t: Sequence[float]) -> None:
        """Initialise the SourceDistributionSample."""
        self._x: Sequence[float] = x
        self._y: Sequence[float] = y
        self._z: Sequence[float] = z
        self._u: Sequence[float] = u
        self._v: Sequence[float] = v
        self._w: Sequence[float] = w
        self._e: Sequence[float] = e
        self._wgt: Sequence[float] = wgt
        self._t: Sequence[float] = t
    
    def _check_if_input_list_or_tuple_of_real_values(
        function: Callable[[Any, Sequence[float]], None],
    ) -> Callable[[Any, Sequence[float]], None]:
        """Decorate a setter to check its input is a list or tuple of real values.

        Used as a decorator within the class body (not a method).

        Parameters
        ----------
        function : Callable[[Any, Sequence[float]], None]
            Setter to wrap.

        Returns
        -------
        Callable[[Any, Sequence[float]], None]
            Wrapped setter.
        """
        def wrapper(self: Any, input: Sequence[float]) -> None:
            """Call the setter if ``input`` is a list or tuple of real values.

            Parameters
            ----------
            input : Sequence[float]
                Value to set.

            Raises
            ------
            Exception
                If ``input`` is not a list or tuple of real values.
            """
            if isinstance(input, (list, tuple)) and all(np.isreal(n) for n in input):
                function(self, input)
            else:
                msg = f'Source distribution {function.__name__}' \
                       'must be list or tuple of real values'
                raise Exception(msg)
        return wrapper

    @property
    def x(self) -> Sequence[float]:
        """X coordinates of the samples.

        Returns
        -------
        Sequence[float]
            X coordinates.
        """
        return self._x

    @x.setter
    @_check_if_input_list_or_tuple_of_real_values
    def x(self, x: Sequence[float]) -> None:
        """Set the x coordinates of the samples.

        Parameters
        ----------
        x : Sequence[float]
            X coordinates.
        """
        self._x = x

    @property
    def y(self) -> Sequence[float]:
        """Y coordinates of the samples.

        Returns
        -------
        Sequence[float]
            Y coordinates.
        """
        return self._y

    @y.setter
    @_check_if_input_list_or_tuple_of_real_values
    def y(self, y: Sequence[float]) -> None:
        """Set the y coordinates of the samples.

        Parameters
        ----------
        y : Sequence[float]
            Y coordinates.
        """
        self._y = y

    @property
    def z(self) -> Sequence[float]:
        """Z coordinates of the samples.

        Returns
        -------
        Sequence[float]
            Z coordinates.
        """
        return self._z

    @z.setter
    @_check_if_input_list_or_tuple_of_real_values
    def z(self, z: Sequence[float]) -> None:
        """Set the z coordinates of the samples.

        Parameters
        ----------
        z : Sequence[float]
            Z coordinates.
        """
        self._z = z

    @property
    def u(self) -> Sequence[float]:
        """X direction cosines of the samples.

        Returns
        -------
        Sequence[float]
            X direction cosines.
        """
        return self._u

    @u.setter
    @_check_if_input_list_or_tuple_of_real_values
    def u(self, u: Sequence[float]) -> None:
        """Set the x direction cosines of the samples.

        Parameters
        ----------
        u : Sequence[float]
            X direction cosines.
        """
        self._u = u

    @property
    def v(self) -> Sequence[float]:
        """Y direction cosines of the samples.

        Returns
        -------
        Sequence[float]
            Y direction cosines.
        """
        return self._v

    @v.setter
    @_check_if_input_list_or_tuple_of_real_values
    def v(self, v: Sequence[float]) -> None:
        """Set the y direction cosines of the samples.

        Parameters
        ----------
        v : Sequence[float]
            Y direction cosines.
        """
        self._v = v

    @property
    def w(self) -> Sequence[float]:
        """Z direction cosines of the samples.

        Returns
        -------
        Sequence[float]
            Z direction cosines.
        """
        return self._w

    @w.setter
    @_check_if_input_list_or_tuple_of_real_values
    def w(self, w: Sequence[float]) -> None:
        """Set the z direction cosines of the samples.

        Parameters
        ----------
        w : Sequence[float]
            Z direction cosines.
        """
        self._w = w

    @property
    def e(self) -> Sequence[float]:
        """Energies of the samples.

        Returns
        -------
        Sequence[float]
            Energies.
        """
        return self._e

    @e.setter
    @_check_if_input_list_or_tuple_of_real_values
    def e(self, e: Sequence[float]) -> None:
        """Set the energies of the samples.

        Parameters
        ----------
        e : Sequence[float]
            Energies.
        """
        self._e = e

    @property
    def wgt(self) -> Sequence[float]:
        """Particle weights of the samples.

        Returns
        -------
        Sequence[float]
            Particle weights.
        """
        return self._wgt

    @wgt.setter
    @_check_if_input_list_or_tuple_of_real_values
    def wgt(self, wgt: Sequence[float]) -> None:
        """Set the particle weights of the samples.

        Parameters
        ----------
        wgt : Sequence[float]
            Particle weights.
        """
        self._wgt = wgt

    @property
    def t(self) -> Sequence[float]:
        """Times of the samples.

        Returns
        -------
        Sequence[float]
            Times.
        """
        return self._t

    @t.setter
    @_check_if_input_list_or_tuple_of_real_values
    def t(self, t: Sequence[float]) -> None:
        """Set the times of the samples.

        Parameters
        ----------
        t : Sequence[float]
            Times.
        """
        self._t = t

    @property
    def r(self) -> list[float]:
        """Radial distance from the z axis of each sample.

        Calculated from :attr:`x` and :attr:`y` on each access.

        Returns
        -------
        list[float]
            Radial distances.
        """
        self._r: list[float] = [(x ** 2 + y ** 2) ** 0.5 for x, y in zip(self._x, self._y)]
        return self._r

    @r.setter
    @_check_if_input_list_or_tuple_of_real_values
    def r(self, r: Sequence[float]) -> None:
        """Set the radial distances (overwritten on the next access of ``r``).

        Parameters
        ----------
        r : Sequence[float]
            Radial distances.
        """
        self._r = r

    def plot_vectors(self, plotter: str = 'matplotlib') -> None:
        """Plot the sample positions and directions as 3D vectors.

        Parameters
        ----------
        plotter : str, optional
            Plotting library, ``'matplotlib'`` or ``'plotly'``, by default
            'matplotlib'
        """

        def _plot_vectors_matplotlib() -> None:
            """Plot the vectors with a matplotlib 3D quiver plot."""
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.quiver(self.x, self.y, self.z, self.u, self.v, self.w)
            plt.show()

        def _plot_starting_locations_plotly() -> None:
            """Plot the vectors with a plotly cone plot."""
            import plotly.graph_objects as go

            fig = go.Figure(data=go.Cone(x=self.x, 
                                         y=self.y, 
                                         z=self.z, 
                                         u=self.u, 
                                         v=self.v, 
                                         w=self.w,
                                         sizeref=0.01))
            fig.show()

        plotters = {'matplotlib': _plot_vectors_matplotlib,
                    'plotly': _plot_starting_locations_plotly}

        plotters[plotter]()

    def plot_starting_locations(self, plotter: str = 'matplotlib') -> None:
        """Plot the sample starting positions as a 3D scatter plot.

        Parameters
        ----------
        plotter : str, optional
            Plotting library, ``'matplotlib'`` or ``'plotly'``, by default
            'matplotlib'
        """
        def _plot_starting_locations_matplotlib() -> None:
            """Plot the positions with a matplotlib 3D scatter plot."""
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(self.x, self.y, self.z)
            plt.show()

        def _plot_starting_locations_plotly() -> None:
            """Plot the positions with a plotly 3D scatter plot."""
            import plotly.graph_objects as go

            fig = go.Figure(data=[go.Scatter3d(x=self.x, y=self.y, z=self.z, mode='markers')])
            fig.show()

        plotters = {'matplotlib': _plot_starting_locations_matplotlib,
                    'plotly': _plot_starting_locations_plotly}

        plotters[plotter]()


class MCNPPTRACSourceReader(SourceDistributionSample):
    """Reader for source particles from an MCNP PTRAC file.

    Parameters
    ----------
    input_file_path : str
        Path of the PTRAC file.
    """

    def __init__(self, input_file_path: str) -> None:
        """Initialise the reader and parse the PTRAC file."""
        self._parse_track_source(input_file_path)

    def _parse_track_source(self, input_file_path: str) -> None:
        """Read the source particle data from a PTRAC file.

        Skips the 9 header rows, then reads every third row starting from the
        third, storing the columns as x, y, z, u, v, w, e, wgt and t.

        Parameters
        ----------
        input_file_path : str
            Path of the PTRAC file.
        """
        with open(input_file_path, 'r') as input_file:
            rows = [[float(v) for v in row.split()] for row in [row for row in input_file][9:][2::3]]

        cols = [col for col in zip(*rows)]
        self.x, self.y, self.z, self.u, self.v, self.w, self.e, self.wgt, self.t = cols

class weightWindow:
    """Placeholder for a weight window."""

    def __init__(self) -> None:
        """Initialise the weightWindow."""
        pass

class MCNPWeightWindow:
    """Placeholder for an MCNP weight window."""

    def __init__(self) -> None:
        """Initialise the MCNPWeightWindow."""
        pass

class MCNPInputReader2():
    """Reader converting an MCNP input deck into mcio surfaces and cells.

    The file is split into sections on blank lines (optional header, cells,
    surfaces, parameters), surfaces are created with
    :meth:`Surface.from_mcnp_card` and cell booleans are converted to volumes.

    Parameters
    ----------
    input_filepath : str
        Path of the MCNP input file.
    verbosity : int, optional
        Verbosity level, by default 0

    Attributes
    ----------
    surfaces : dict[int, Surface]
        Surfaces keyed by surface number.
    cells : dict[int, Cell]
        Cells keyed by MCNP cell number.

    Raises
    ------
    Exception
        If the file does not contain 3 or 4 sections.
    """

    def __init__(self, input_filepath: str, verbosity: int = 0) -> None:
        """Initialise the reader and parse the input file."""
        # set verbosity 
        self._verbosity: int = verbosity

        # read mcnp input_file and convert to string
        with open(input_filepath, 'r') as input_file:
            # save strings to list
            regex_blank_line = re.compile(r'^\s*\n')
            rows = [r for r in input_file]
            
            # find blank rows to split file intp sections
            split_indices = [i for i, r in enumerate(rows) if re.search(regex_blank_line, r)]

            # split the file rows into sections
            if len(split_indices) == 2:
                self._sections: dict[str, list[str]] = {'cells': self._return_cards([r for r in rows[0: split_indices[0]]]),
                                 'surfaces': self._return_cards([r for r in rows[split_indices[0]+1: split_indices[1]]]),
                                 'parameters': self._return_cards([r for r in rows[split_indices[1]+1:]])}
            elif len(split_indices) == 3:
                self._sections = {'header': self._return_cards([r for r in rows[0: split_indices[0]]]),
                                  'cells': self._return_cards([r for r in rows[split_indices[0]+1: split_indices[1]]]),
                                  'surfaces': self._return_cards([r for r in rows[split_indices[1]+1: split_indices[2]]]),
                                  'parameters': self._return_cards([r for r in rows[split_indices[2]+1:]])}
            else:
                msg = f'number of section expected is 3 or 4: {len(split_indices)-1} read'
                raise Exception(msg)

        # store cards from readout
        self.surfaces: dict[int, Surface] = self._parse_surfaces()

        print('surfaces')
        print(self.surfaces)
        
        self.cells: dict[int, Cell] = self._parse_cells()


    def _parse_surfaces(self) -> dict[int, Surface]:
        """Create surfaces from the surface section cards.

        Returns
        -------
        dict[int, Surface]
            Surfaces keyed by surface number.
        """
        # store surface numbers with their surfaces
        surfaces: dict[int, Surface] = {}

        # define regex searched
        regex_surf = re.compile(r'(?:\s*)(?P<surf_number>([0-9]+))(?:\s+)' + \
                                r'(?P<transform>[0-9]*)(?:\s*)' + \
                                r'(?P<surf_type>[a-zA-Z/]+)' \
                                r'(?P<params>.*)')

        # get surface cards
        for surface_card in self._sections['surfaces']:
            # convert mcnp card as string to surface instance
            surface = Surface.from_mcnp_card(surface_card)

            # store inverted surface
            surfaces[surface.number] = surface

        # return parsed surfaces
        return surfaces

    def _get_volume_from_boolean_string(self, boolean: str) -> HalfVolume:
        """Return the half-space for a signed surface number string.

        Parameters
        ----------
        boolean : str
            Signed surface number, e.g. ``'-3'``.

        Returns
        -------
        HalfVolume
            Negative half-space of the surface for negative numbers, otherwise
            the positive half-space.
        """
        boolean = int(boolean)
        if boolean < 0:
            volume = -self.surfaces[abs(boolean)]
        else:
            volume = +self.surfaces[boolean]

        return volume
    
    def _get_first_volume_in_booleans(
        self, booleans: list[str], level: int = 1
    ) -> tuple[HalfVolume | Volume, list[str]] | None:
        """Return the first volume in a list of boolean terms and the remainder.

        If the list starts with an open bracket, the bracketed terms are parsed
        recursively with ``_parse_booleans``.

        Parameters
        ----------
        booleans : list[str]
            Boolean terms (signed surface numbers, ``'('``, ``')'``, ``':'``).
        level : int, optional
            Bracket nesting level, by default 1

        Returns
        -------
        tuple[HalfVolume | Volume, list[str]] or None
            First volume and the remaining terms, or None if an opening bracket
            is never closed.
        """
        if booleans[0] == '(':
            sub_boolean_strings: list[str] = []
            n_open_brackets = 1
            for boolean_string in booleans[1:]:
                if boolean_string == '(':
                    # case for another sub boolean
                    n_open_brackets += 1
                    sub_boolean_strings.append(boolean_string)
                elif boolean_string == ')' and n_open_brackets > 1:
                    # case for secondard bracket closing
                    sub_boolean_strings.append(boolean_string)
                    n_open_brackets -= 1
                elif boolean_string == ')' and n_open_brackets == 1:
                    # case for primary bracket closing - passing contents of bracket recursively
                    volume_first = self._parse_booleans(sub_boolean_strings, level=level+1)
                    booleans_remaining = booleans[len(sub_boolean_strings)+2:]
                    return volume_first, booleans_remaining
                else:
                    sub_boolean_strings.append(boolean_string)
        else:
            volume_first: HalfVolume = self._get_volume_from_boolean_string(booleans[0])
            return volume_first, booleans[1:]

    def _parse_booleans(self, booleans: list[str]) -> HalfVolume | Volume:
        """Convert a list of boolean terms into a volume.

        Juxtaposed terms are intersected and terms separated by ``':'`` are
        unioned, evaluated left to right.

        Parameters
        ----------
        booleans : list[str]
            Boolean terms (signed surface numbers, ``'('``, ``')'``, ``':'``).

        Returns
        -------
        HalfVolume | Volume
            Resulting volume.

        Raises
        ------
        ValueError
            If the numbers of open and closed brackets differ.
        """
        print('booleans')
        print(booleans)

        # check that the number of open blackers equals closed brackets
        n_open_brackets, n_closed_brackets = booleans.count('('), booleans.count(')')
        if n_open_brackets != n_closed_brackets:
            msg = f'reading cell boolean(s) - number of open ({n_open_brackets}) not equal to closed ({n_closed_brackets})'
            logger.fatal(msg)
            raise ValueError(msg)

        # get the first term in the boolean list
        volume, booleans_remaining = self._get_first_volume_in_booleans(booleans)

        # iterate over the rest of the list and add to volume
        while len(booleans_remaining) > 0:
            if booleans_remaining[0] == ':':
                volume_next, booleans_remaining = self._get_first_volume_in_booleans(booleans_remaining[1:])
                volume = volume | volume_next
            else:
                volume_next, booleans_remaining = self._get_first_volume_in_booleans(booleans_remaining)
                volume = volume & volume_next

        # return results
        return volume

    def _parse_cells(self) -> dict[int, Cell]:
        """Create cells from the cell section cards.

        Returns
        -------
        dict[int, Cell]
            Cells keyed by MCNP cell number.
        """
        # store cell numbers with their cells
        cells: dict[int, Cell] = {}

        # define regex searches
        regex_cell = re.compile(r'(?P<cell_number>^\s*[0-9]+)(?:\s+)(?:(?P<material_number_void>0)|((?P<material_number>[0-9]+)(?:\s+)(?P<density>[0-9\-\+\.eE]+)))(?:\s*)(?P<boolean>[0-9\s\.\-\:\(\)\#]+)(?P<parameters>.*)')
        regex_cell_vol = re.compile(r'(?:vol\s*=\s*)(?P<volume>[0-9\-\+\.eE]+)', re.IGNORECASE)

        for card in self._sections['cells']:
            groups_cell = re.search(regex_cell, card)
            groups_cell_params = re.search(regex_cell_vol, groups_cell['parameters'])

            cell_number = int(groups_cell['cell_number'])
            cell_mat = int(groups_cell['material_number']) if groups_cell['material_number'] else int(groups_cell['material_number_void'])
            cell_density = float(groups_cell['density']) if groups_cell['density'] else 0.0
            cell_volume = float(groups_cell_params['volume']) if groups_cell_params else None
            cell_boolean = groups_cell['boolean']

            # print(f'='*100)
            # print(f'cell_{cell_number}')
            # print(f'mat_{cell_mat}')
            # print(f'density_{cell_density}')
            # print(f'volume_{cell_volume}')
            # print(f'boolean')
            # print(cell_boolean)
            # print(card)

            # clean boolean string, ensuring spacing between operators E.G. -3:4(5 6) converts to -3 : 4 ( 5 6 )
            boolean_terms = [term for term in cell_boolean.replace('(', ' ( ').replace(')', ' )').replace(':', ' : ').split(' ') if term != '']

            # # convert all surface term strings to mcio half spaces
            # boolean_terms = [term if term in ['(', ')', ':'] else +self._surfaces[int(term)] if int(term) > 0 else -self._surfaces[abs(int(term))] for term in boolean_terms]
            
            # print(f'terms')
            # print(boolean_terms)

            vol = self._parse_booleans(boolean_terms)

            # # convert boolean terms into volume
            # vol = _parse_boolean_string(boolean_terms)

            # create cell with boolean volume, cell number and volume
            cell: Cell = Cell(volume = vol)
            cell.number = 1
            cell.mcnp.volume = cell_volume if cell_volume else None
            cells[cell_number] = cell

            # print('cell from parsed')

            # model = Model(cells=[cell])
            # print(model.mcnp_string)

            # if cell_volume:
            #     cell_volumes[cell_number] = cell_volume

        # return parsed cells
        return cells

    def _return_cards(self, rows: list[str]) -> list[str]:
        """Split rows of a section into cards, removing comments.

        Comment lines (starting ``c`` or ``$``) and ``$`` comments are removed,
        rows starting with five spaces are joined to the preceding card and
        newlines are removed.

        Parameters
        ----------
        rows : list[str]
            Rows of a section of the input file.

        Returns
        -------
        list[str]
            One string per card.
        """
        # remove comment lines
        rows_no_comments = [row.split('$', 1)[0] for row in rows if row.lstrip()[0].lower() not in ('c', '$')]

        # store indices of card start positions
        indices =  [i for i, row in enumerate(rows_no_comments) if row[:5] != '     ']

        # split rows into cards 
        cards = [rows_no_comments[start:end] for start, end in zip(indices[:-1], indices[1:])]
        cards.append(rows_no_comments[indices[-1]:])
        
        # remove new lines within each card
        cards_no_newlines = [' '.join(card).replace('\n', '') for card in cards]

        return cards_no_newlines


class MCNPInputReader():
    """Reader for MCNP input decks: surfaces, transforms, materials and cells.

    Parameters
    ----------
    input_filepath : str
        Path of the MCNP input file. Note it is currently unused: the file is
        never read and the ``_sections`` attribute the parsers rely on is never
        set.
    verbosity : int, optional
        Verbosity level, by default 0
    """

    def __init__(self, input_filepath: str, verbosity: int = 0) -> None:
        """Initialise the reader and parse the input sections."""
        # set verbosity
        self._verbosity: int = verbosity

        # store read cards
        self._surfaces: dict[int, Surface] = {}
        self._transforms: dict[int, Transform] = {}
        self._cells: dict[int, Cell] = {}
        self._cell_volumes: dict[int, float] = {}
        self._material_compositions: dict[int, dict[str, Any]] = {}
        self._materials: dict[int, Any] = {}

        # store warnings for incongruous definitions
        self._warnings: list[str] = []

        # parse cells
        self._parse_surface_section()
        self._parse_parameters_section()
        self._parse_cell_section()

        # print warnings
        if len(self._warnings) > 0:
            print('warnings:')
            for warning in self._warnings:
                print(f'[warning] {warning}')

    @property
    def model(self) -> Model:
        """Model built from the parsed cells.

        Returns
        -------
        Model
            Model containing the parsed cells.
        """
        return Model(cells=self._cells.values())

    @property
    def cells(self) -> dict[int, Cell]:
        """Parsed cells keyed by cell number.

        Returns
        -------
        dict[int, Cell]
            Parsed cells.
        """
        return self._cells
    
    @property
    def cell_volumes(self) -> dict[int, float | None]:
        """MCNP volume of each parsed cell.

        Returns
        -------
        dict[int, float | None]
            Cell volumes keyed by cell number.
        """
        return {cell_number: cell.mcnp.volume for cell_number, cell in self._cells.items()}
    
    @property
    def surfaces(self) -> dict[int, Surface]:
        """Parsed surfaces keyed by surface number.

        Returns
        -------
        dict[int, Surface]
            Parsed surfaces.
        """
        return self._surfaces
    
    @property
    def materials(self) -> dict[int, Any]:
        """Parsed materials keyed by material number.

        Returns
        -------
        dict[int, Any]
            Parsed materials (currently never populated).
        """
        return self._materials

    def _return_cards(self, rows: list[str]) -> list[str]:
        """Split rows of a section into cards, removing comments.

        Comment lines (starting ``c`` or ``$``) and ``$`` comments are removed,
        rows starting with five spaces are joined to the preceding card and
        newlines are removed.

        Parameters
        ----------
        rows : list[str]
            Rows of a section of the input file.

        Returns
        -------
        list[str]
            One string per card.
        """
        # remove comment lines
        rows_no_comments = [row.split('$', 1)[0] for row in rows if row.lstrip()[0].lower() not in ('c', '$')]

        # store indices of card start positions
        indices =  [i for i, row in enumerate(rows_no_comments) if row[:5] != '     ']

        # split rows into cards 
        cards = [rows_no_comments[start:end] for start, end in zip(indices[:-1], indices[1:])]
        cards.append(rows_no_comments[indices[-1]:])
        
        # remove new lines within each card
        cards_no_newlines = [' '.join(card).replace('\n', '') for card in cards]

        return cards_no_newlines

    def _get_volume_from_boolean_string(self, boolean: str) -> HalfVolume:
        """Return the half-space for a signed surface number string.

        Parameters
        ----------
        boolean : str
            Signed surface number, e.g. ``'-3'``.

        Returns
        -------
        HalfVolume
            Negative half-space of the surface for negative numbers, otherwise
            the positive half-space.
        """
        boolean = int(boolean)
        if boolean < 0:
            volume = -self._surfaces[abs(boolean)]
        else:
            volume = +self._surfaces[boolean]

        return volume
    
    def _get_first_volume_in_booleans(
        self, booleans: list[str], level: int = 1
    ) -> tuple[HalfVolume | Volume, list[str]] | None:
        """Return the first volume in a list of boolean terms and the remainder.

        If the list starts with an open bracket, the bracketed terms are parsed
        recursively with ``_parse_booleans``.

        Parameters
        ----------
        booleans : list[str]
            Boolean terms (signed surface numbers, ``'('``, ``')'``, ``':'``).
        level : int, optional
            Bracket nesting level, by default 1

        Returns
        -------
        tuple[HalfVolume | Volume, list[str]] or None
            First volume and the remaining terms, or None if an opening bracket
            is never closed.
        """
        if booleans[0] == '(':
            sub_boolean_strings: list[str] = []
            n_open_brackets = 1
            for boolean_string in booleans[1:]:
                if boolean_string == '(':
                    # case for another sub boolean
                    n_open_brackets += 1
                    sub_boolean_strings.append(boolean_string)
                elif boolean_string == ')' and n_open_brackets > 1:
                    # case for secondard bracket closing
                    sub_boolean_strings.append(boolean_string)
                    n_open_brackets -= 1
                elif boolean_string == ')' and n_open_brackets == 1:
                    # case for primary bracket closing - passing contents of bracket recursively
                    volume_first = self._parse_booleans(sub_boolean_strings, level=level+1)
                    booleans_remaining = booleans[len(sub_boolean_strings)+2:]
                    return volume_first, booleans_remaining
                else:
                    sub_boolean_strings.append(boolean_string)
        else:
            volume_first = self._get_volume_from_boolean_string(booleans[0])
            print('first_volume', volume_first)
            return volume_first, booleans[1:]

    def _parse_booleans(self, booleans: list[str], level: int = 1) -> HalfVolume | Volume:
        """Convert a list of boolean terms into a volume.

        Juxtaposed terms are intersected and terms separated by ``':'`` are
        unioned, evaluated left to right.

        Parameters
        ----------
        booleans : list[str]
            Boolean terms (signed surface numbers, ``'('``, ``')'``, ``':'``).
        level : int, optional
            Bracket nesting level (unused), by default 1

        Returns
        -------
        HalfVolume | Volume
            Resulting volume.
        """
        # get the first term in the boolean list
        volume, booleans_remaining = self._get_first_volume_in_booleans(booleans)

        # iterate over the rest of the list and add to volume
        while booleans_remaining:
            if booleans_remaining[0] == ':':
                volume_next, booleans_remaining = self._get_first_volume_in_booleans(booleans_remaining[1:])
                volume = volume | volume_next
            else:
                volume_next, booleans_remaining = self._get_first_volume_in_booleans(booleans_remaining)
                volume = volume & volume_next

        # return results
        return volume

    def _parse_cell_section(self) -> None:
        """Parse the cell section, currently storing only the cell volumes.

        Cell volumes given by ``vol=`` are stored in ``_cell_volumes``.
        """
        # regex for parsing cell
        regex_cell = re.compile(r'(?P<cell_number>^\s*[0-9]+)(?:\s+)(?:(?P<material_number_void>0)|((?P<material_number>[0-9]+)(?:\s+)(?P<density>[0-9\-\+\.eE]+)))(?:\s*)(?P<boolean>[0-9\s\.\-\:\(\)\#]+)(?P<parameters>.*)')
        regex_cell_vol = re.compile(r'(?:vol\s*=\s*)(?P<volume>[0-9\-\+\.eE]+)', re.IGNORECASE)
        regex_booleans_split = re.compile(r'(?:\(){1,1}|(?:\)){1,1}|(?:\-{0,1}[0-9]+)|(?:#[0-9]+)')
        
        # parse the cell card, recursively where #cell_number is used.
        def _parse_ce_parse_boolean_card(card: str) -> Cell:
            """Parse a cell card into a Cell (work in progress, currently unused).

            Parameters
            ----------
            card : str
                Cell card.

            Returns
            -------
            Cell
                Cell with the card's cell number.
            """
            # print('='*200)
            # # cell_number_temp = card
            # # print(f'cell_number: {cell_number}')
            # print(card)

            # regex cell
            groups_cell = re.search(regex_cell, card).groupdict()
            search_vol = re.search(regex_cell_vol, card)

            # card regex
            groups_cell = re.search(regex_cell, card).groupdict()
            search_vol = re.search(regex_cell_vol, card)
            
            # extract key values
            cell_number = int(groups_cell['cell_number'])
            booleans_seperated = re.findall(regex_booleans_split, groups_cell['boolean'])
            
            cell = Cell(cell_number=cell_number)

            return cell

        # get cards from section
        cards = self._return_cards(self._sections['cells'])

        # set up dictionaries to help organise order in which cards are parsed, 
        # as some booleans have hash values which require cards to be parsed
        # out of the order they appear in the mcnp input file
        cards_indexed = {i: card for i, card in enumerate(cards)}

        for i, card in enumerate(cards, start=1):
            groups_cell = re.search(regex_cell, card).groupdict()
            search_vol = re.search(regex_cell_vol, card)

            cell_number = int(groups_cell['cell_number'])
            if search_vol is not None:
                self._cell_volumes[cell_number] = float(search_vol['volume'])

        return

    def _parse_surface_section(self) -> None:
        """Parse the surface section, storing surfaces in ``_surfaces``."""
        # strip comments from rows and return cards
        cards = self._return_cards(self._sections['surfaces'])

        # surface card regex
        regex_surf = re.compile(r'(?:\s*)(?P<surf_number>([0-9]+))(?:\s+)(?P<transform>[0-9]*)(?:\s*)(?P<surf_type>[a-zA-Z/]+)(?P<params>.*)')

        # parse cards
        for i, card in enumerate(cards, start=1):
            groups_surf = re.search(regex_surf, card).groupdict()
         
            surf_number = int(groups_surf['surf_number'])
            surf_type = groups_surf['surf_type']

            surf_params = [float(v) for v in groups_surf['params'].split()]

            # case for plane from three points
            if surf_type.lower() == 'p' and len(surf_params) == 9:
                surf_type = 'P3P'
            
            surface = MCNP_SURFACE_TYPES[surf_type.upper()](*surf_params)
            surface.number = surf_number
            self._surfaces[surf_number] = surface

            if self._verbosity > 0:
                print('='*100)
                print(card)
                print(f' -> number {surf_number}')
                print(f' -> type    {surf_type}')
                print(f' -> params  {surf_params}')
                
    def _parse_parameters_section(self) -> None:
        """Parse transform and material cards from the parameters section.

        Transforms are stored in ``_transforms`` and material compositions in
        ``_material_compositions``; inconsistent libraries or fraction types are
        recorded in ``_warnings``.

        Raises
        ------
        Exception
            If a material card has unequal numbers of zaids and fractions.
        """
        # get cards from section
        cards = self._return_cards(self._sections['parameters'])

        # define material card regex
        regex_mats = re.compile(r'(?:\s*[mM])(?P<material_number>[0-9]+)(?:\s+)(?P<zaids>([0-9]{4,6}\.{0,1}[0-9]{0,2}[a-zA-Z]{0,1}\s+[0-9\-\+\.eE]+\s*)+)')
        regex_mats_zaids_and_lib = re.compile(r'(?P<zaid>[0-9]{4,6})(?P<library>\.[0-9]{2}[a-zA-Z]){0,1}')
        
        # define transform regex
        regex_transforms = re.compile(r'(?:\s*)(?P<star>\*{0,1})(?:\s*)(?:[tT][rR])(?P<number>[0-9]+)')
        regex_transforms_number_and_vals = re.compile(r'(?:\s*)(?P<star>\*{0,1})(?:\s*)(?:[tT][rR])(?P<number>[0-9]+)(:?\s*)(?P<x>[0-9\.\+\-eE]+)(:?\s*)(?P<y>[0-9\.\+\-eE]+)(:?\s*)(?P<z>[0-9\.\+\-eE]+)(:?\s*)(?P<xx>[0-9\.\+\-eE]*)(:?\s*)(?P<yx>[0-9\.\+\-eE]*)(:?\s*)(?P<zx>[0-9\.\+\-eE]*)(:?\s*)(?P<xy>[0-9\.\+\-eE]*)(:?\s*)(?P<yy>[0-9\.\+\-eE]*)(:?\s*)(?P<zy>[0-9\.\+\-eE]*)(:?\s*)(?P<xz>[0-9\.\+\-eE]*)(:?\s*)(?P<yz>[0-9\.\+\-eE]*)(:?\s*)(?P<zz>[0-9\.\+\-eE]*)')

        # filter cards by type
        material_cards = [card for card in cards if re.search(regex_mats, card)]
        transform_cards = [card for card in cards if re.search(regex_transforms, card)]

        # parse transform cards
        for i, card in enumerate(transform_cards):
            
            # regex search transform cards
            groups_transforms = re.search(regex_transforms_number_and_vals, card).groupdict()

            transform_number = int(groups_transforms['number'])
            

            latteral = (
                float(groups_transforms['x']) if groups_transforms['x'] != '' else 0.0,
                float(groups_transforms['y']) if groups_transforms['y'] != '' else 0.0,
                float(groups_transforms['z']) if groups_transforms['z'] != '' else 0.0,
            )
            
            print(latteral)

            matrix = (
                (
                float(groups_transforms['xx']) if groups_transforms['xx'] != '' else 0,
                float(groups_transforms['yx']) if groups_transforms['yx'] != '' else 90,
                float(groups_transforms['zx']) if groups_transforms['zx'] != '' else 90,
                ),
                (
                float(groups_transforms['xy']) if groups_transforms['xy'] != '' else 90,
                float(groups_transforms['yy']) if groups_transforms['yy'] != '' else 0,
                float(groups_transforms['zy']) if groups_transforms['zy'] != '' else 90,
                ),
                (
                float(groups_transforms['xz']) if groups_transforms['xz'] != '' else 90,
                float(groups_transforms['yz']) if groups_transforms['yz'] != '' else 90,
                float(groups_transforms['zz']) if groups_transforms['zz'] != '' else 0,
                ),
            )

            # create transform and store with transform number
            angle = 'cosine' if groups_transforms['star'] == '' else 'degrees'
            transform = Transform(latteral=latteral,
                                                 matrix=matrix,
                                                 angle_type=angle)
            self._transforms[transform_number] = transform

        # parse material cards
        for i, card in enumerate(material_cards, start=1):
            # regex search material data
            groups_mats = re.search(regex_mats, card).groupdict()
            mat_number = int(groups_mats['material_number'])
            mat_zaids = groups_mats['zaids'].split()

            # check to make sure each zaid has an associated fraction loaded
            if len(mat_zaids) % 2 != 0:
                msg = f'number of zaids and fractions in material {mat_number} not equal'
                raise Exception (msg)
            
            # strip z, a and libraries for each zaid fraction pair
            zaids_and_fractions: dict[int, dict[int, float]] = {}
            libraries: list[str | None] = []
            fractions: list[float] = []

            for zaid_and_lib, fraction in zip(mat_zaids[::2], mat_zaids[1::2]):
                # split zaid and density string into z, a, library and fraction
                groups_mat_zaid_and_lib = re.search(regex_mats_zaids_and_lib, zaid_and_lib).groupdict()
                zaid = groups_mat_zaid_and_lib['zaid']
                z = int(zaid[:-3])
                a = int(zaid[-3:])
                fraction = float(fraction)
                fractions.append(fraction)
                
                # add zaids and fractions
                if z not in zaids_and_fractions:
                    # case for a new element
                    zaids_and_fractions[z] = {a: abs(fraction)}
                elif a in zaids_and_fractions[z]:
                    # case for a repetition of a zaid
                    zaids_and_fractions[z][a] += abs(fraction)
                else:
                    # case for a new isotope
                    zaids_and_fractions[z][a] = abs(fraction)

                if 'library' in groups_mat_zaid_and_lib:
                    libraries.append(groups_mat_zaid_and_lib['library'])
                
            # add material zaids
            self._material_compositions[mat_number] = {'zaids_and_fractions': zaids_and_fractions,
                                                       'fraction_type': 'wo' if all([fraction > 0 for fraction in fractions  ]) else 'ao',
                                                       'library': libraries[0] if len(libraries) > 0 else None}

            # check if library is the same for all zaids
            if (len(libraries) > 0) and libraries.count(libraries[0]) < len(libraries):
                msg = f'material {mat_number} contains different libraries for zaids. first library assigned only.'
                self._warnings.append(msg)
            
            # check if fraction types are the same
            if not (all([fraction > 0 for fraction in fractions]) or all([fraction < 0 for fraction in fractions])):
                msg = f'material {mat_number} contains fractions of atom and weight in the same material. First fraction type assigned to all.'
                self._warnings.append(msg)

            if self._verbosity > 0:
                print('='*100)
                print(card)

                print(f'material {mat_number}')
                for z in self._material_compositions[mat_number]['zaids_and_fractions']:
                    for a in self._material_compositions[mat_number]['zaids_and_fractions'][z]:
                        mat_composition = self._material_compositions[mat_number]['zaids_and_fractions'][z][a]
                        print(f' -> [{z}][{a}] {mat_composition}')
