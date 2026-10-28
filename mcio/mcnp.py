"""MCNP card building and MCNP-specific parameter classes.

Contains classes that build MCNP input text (:class:`MCNPHeader`,
:class:`MCNPCardHeader`, :class:`MCNPCard`) and the MCNP parameter containers
attached to models, tallies, cells and materials (:class:`MCNPRunParameters`,
:class:`MCNPTallyParameters`, :class:`MCNPCellParameters`,
:class:`MCNPMaterialParameters`).
"""
from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any

import numpy as np


from .formatting import formatting
from .checkvals import numbers_to_strings, check_types
from .utils import SerialisableFromDict

if TYPE_CHECKING:
    from .cell import Cell
    from .surface import Surface
    from .tally import Tally


# list MCNP particle types
MCNP_PARTICLE_TYPES: dict[str, str] = {'n': 'neutron', 
                       'p': 'photon',
                       'e': 'electron',
                       '|': 'negative_muon',
                       'q': 'anti_neutron',
                       'u': 'electron_neutrino',
                       'v': 'muon_neutrino',
                       'f': 'positron',
                       'h': 'proton',
                       'l': 'lambda_baryon',
                       '+': 'positive_sigma_baryon',
                       '-': 'negative_sigma_baryon',
                       'x': 'cascade_xi_baryon',
                       'y': 'negative_cascade_negative_xi_baryon',
                       'o': 'omega_baryon',
                       '!': 'positive_muon',
                       '<': 'anti_electron_neutrino',
                       '>': 'anti_muon_neutrino',
                       'g': 'anti_proton',
                       '/': 'positive_pion',
                       'z': 'neutral_pion',
                       'k': 'positive_kaon',
                       '%': 'kaon_short',
                       '^': 'kaon_long',
                       'b': 'anti_lambda_baryon',
                       '_': 'anti_positive_sigma_baryon',
                       '~': 'anti_negative_sigma_baryon',
                       'c': 'anti_cascade_anti_neutral_xi_baryon',
                       'w': 'positive_cascade_positive_xi_baryon',
                       '@': 'anti_omega',
                       'd': 'deuteron',
                       't': 'triton',
                       's': 'helion',
                       'a': 'alpha',
                       '*': 'negative_pion',
                       '?': 'negative_kaon',
                       '#': 'heavy_ion'}

# invert dictionary of MCNP particles types
MCNP_PARTICLE_DESIGNATORS: dict[str, str] = {v: k for k, v in MCNP_PARTICLE_TYPES.items()}



class MCNPHeader:
    """MCNP section header written as a block of comment lines.

    Produces a banner of ``C ===`` comment lines containing the header title
    and, optionally, a list of label/value pairs.

    Parameters
    ----------
    header : str
        Title written in the banner.
    data_pairs : dict[str, Any], optional
        Label/value pairs written under the title, by default None
    banner : list[str], optional
        Banner lines (stored but currently unused), by default None
    """
    def __init__(self, header: str, data_pairs: dict[str, Any] | None = None, banner: list[str] | None = None) -> None:
        """Initialise the MCNPHeader."""
        # store user provided parameters
        self._header: str = header
        self._data_pairs: dict[str, Any] | None = data_pairs
        self._banner: list[str] | None = banner

    @property
    def as_string(self) -> str:
        """Header as MCNP comment lines.

        Float values below 1e-4 are written in scientific notation, other
        floats with four decimal places.

        Returns
        -------
        str
            Comment block, terminated with a newline.
        """
        line_break = 'C ' + '=' * (formatting.MCNP_MAX_CHARS - 2) + '\n'

        header = line_break[:formatting.MCNP_INDENT] + f' {self._header} '
        header += line_break[formatting.MCNP_INDENT + len(self._header) + 2:]

        output = ''.join([line_break, header, line_break])

        if self._data_pairs:
            LINE_INDENT = 'C ' + ('=' * 3)
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


class MCNPCardHeader:
    """Comment lines written above an MCNP card.

    Parameters
    ----------
    data : dict[str, Any]
        Key/value pairs, each written as a ``C key: value`` comment line.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialise the MCNPCardHeader."""
        self._data: dict[str, Any] = data

    @property
    def as_string(self) -> str:
        """Header as MCNP comment lines.

        Returns
        -------
        str
            Comment lines bracketed by blank ``C`` lines.
        """
        # long values are wrapped onto further comment lines to respect the line length
        lines = [
            line
            for k, v in self._data.items()
            for line in textwrap.wrap(f'C {k}: {v}', width=formatting.MCNP_MAX_CHARS, subsequent_indent='C   ')
        ]
        comments = ''.join(f'{line}\n' for line in lines)
        return 'C\n' + comments + 'C\n'


class MCNPCard:
    """A generic MCNP card made of a designator and rows of entries.

    Entries are wrapped onto continuation lines (indented by
    ``formatting.MCNP_INDENT``) so lines stay below
    ``formatting.MCNP_MAX_CHARS``. Each row starts on a new line.

    Parameters
    ----------
    designator : str or int
        Card name or number written first (e.g. ``'M1'``, ``'NPS'`` or a
        cell number).
    rows : list[list[str]]
        Rows of string entries.
    subcards : list[MCNPCard], optional
        Cards written directly after this card, by default None
    header_data : dict[str, Any], optional
        Data written as comment lines above the card, by default None
    """

    def __init__(self, designator: str | int, rows: list[list[str]], subcards: list[MCNPCard] | None = None, header_data: dict[str, Any] | None = None) -> None:
        """Initialise the MCNPCard."""
        self._designator: str | int = designator
        self._rows: list[list[str]] = rows
        self._subcards: list[MCNPCard] = subcards if subcards else []
        self._header_data: dict[str, Any] | None = header_data

    def add_row(self, row: list[str]) -> None:
        """Append a row of entries to the card.

        Parameters
        ----------
        row : list[str]
            Entries of the row.
        """
        self._rows.append(row)

    def add_subcard(self, subcard: MCNPCard) -> None:
        """Append a sub-card, written after this card.

        Parameters
        ----------
        subcard : MCNPCard
            Card to append.
        """
        self._subcards.append(subcard)

    @property
    def as_string(self) -> str:
        """Card (with header comments and sub-cards) as MCNP input text.

        Returns
        -------
        str
            The formatted card.
        """
        rows: list[str] = []

        string = f'{self._designator:<{formatting.MCNP_INDENT}}'

        for i, row in enumerate(self._rows):
            for j, entry in enumerate(row):
                if (len(string) + len(entry) + 1) < formatting.MCNP_MAX_CHARS:
                    string += f' {entry}'
                else:
                    rows.append(f'{string}\n')
                    string =  ' ' * (formatting.MCNP_INDENT) + f' {entry}'
            rows.append(string + '\n')
            string = ' ' * (formatting.MCNP_INDENT)

        output_string = ''.join(rows)

        # add header comments
        if self._header_data is not None:
            mcnp_card_header = MCNPCardHeader(self._header_data)
            output_string = mcnp_card_header.as_string + output_string

        # add sub cards
        for subcard in self._subcards:
            output_string += subcard.as_string

        return output_string


class MCNPRunParameters:
    """MCNP run control parameters (NPS, MODE, PHYS, CUT, etc.).

    Parameters
    ----------
    modes : list[str], optional
        Transported particle names (keys of ``MCNP_PARTICLE_DESIGNATORS``),
        by default ['neutron']
    nps : int, optional
        Number of particle histories, by default 1e9
    prdmp_intervals : int, optional
        Number of dump intervals over the run used for the PRDMP card, by
        default 10
    lost : int, optional
        Maximum number of lost particles, by default 100
    write_mctal : bool, optional
        Whether to write an MCTAL file, by default True
    xsdir : str, optional
        XSDIR path written to the MESSAGE block, by default None
    cut_parameters : dict[str, dict[str, float]], optional
        CUT card values per particle name, by default None
    physics_cards : list[str], optional
        Particle names for which PHYS cards are written, by default None
    stop_tallies : dict[Tally, float], optional
        STOP card relative error cutoff per tally, by default None
    weight_window : dict[str, list[float]], optional
        WWP card values per particle designator, by default None
    idums : list[int], optional
        IDUM card values, by default None
    rdums : list[float], optional
        RDUM card values, by default None
    spabi : dict[str, dict[str, Any]], optional
        SPABI card data per secondary particle, with keys
        ``primary_particles``, ``energies`` and ``multipliers``, by default
        None
    forced_collisions : dict[str, dict[Cell, float]], optional
        FCL card weighting per cell for each particle, by default None
    void : bool, optional
        Whether to write a VOID card, by default False
    """

    def __init__(self, 
                 modes: list[str]=['neutron'],
                 nps: int=1e9,
                 prdmp_intervals: int=10, 
                 lost: int=100,
                 write_mctal: bool=True, 
                 xsdir: str | None = None, 
                 cut_parameters: dict[str, dict[str, float]] | None = None,
                 physics_cards: list[str] | None = None, 
                 stop_tallies: dict[Tally, float] | None = None, 
                 weight_window: dict[str, list[float]] | None = None, 
                 idums: list[int] | None = None, 
                 rdums: list[float] | None = None, 
                 spabi: dict[str, dict[str, Any]] | None = None, 
                 forced_collisions: dict[str, dict[Cell, float]] | None = None,
                 void: bool = False) -> None:
        """Initialise the MCNP run parameters."""
        self._nps: int = nps
        self._prdmp_intervals: int = prdmp_intervals
        self._lost: int = lost
        self._write_mctal: bool = write_mctal
        self._modes: list[str] = [p.lower() for p in modes]
        self.cut_parameters: dict[str, dict[str, float]] = cut_parameters if cut_parameters else {}
        self.physics_particles = physics_cards if physics_cards else []
        self._stop_tallies: dict[Tally, float] = stop_tallies if stop_tallies else {}
        self._xsdir: str | None = xsdir
        self._weight_window: dict[str, list[float]] | None = weight_window
        self._idums: list[int] | None = idums
        self._rdums: list[float] | None = rdums
        self._spabi: dict[str, dict[str, Any]] | None = spabi
        self._void: bool = void
        self._forced_collisions: dict[str, dict[Cell, float]] | None = forced_collisions

        # # set up physics parameter dictionaries for the phys particle types
        # self.cut_parameters = {p: {'t': 'j',
        #                            'e': 'j',
        #                            'wc1': 'j',
        #                            'wc2': 'j',
        #                            'swtm': 'j'} for p in self._cut_particles}

    @property
    def physics_particles(self) -> list[str]:
        """Particle names for which PHYS cards are written.

        Returns
        -------
        list[str]
            Particle names.
        """
        return self._physics_particles
    
    @physics_particles.setter
    def physics_particles(self, physics_particles: list[str]) -> None:
        """Set the PHYS particles and reset their parameter lists to empty.

        Parameters
        ----------
        physics_particles : list[str]
            Particle names.
        """
        self._physics_particles: list[str] = physics_particles
        self._physics_parameters: dict[str, list[float]] = {p: [] for p in self._physics_particles}

    @property
    def nps(self) -> int:
        """Number of particle histories.

        Returns
        -------
        int
            Number of histories.
        """
        return self._nps

    @nps.setter
    def nps(self, nps: int) -> None:
        """Set the number of particle histories.

        Parameters
        ----------
        nps : int
            Number of histories.
        """
        self._nps = nps

    @property
    def xsdir(self) -> str | None:
        """XSDIR path written to the MESSAGE block.

        Returns
        -------
        str or None
            XSDIR path.
        """
        return self._xsdir

    @xsdir.setter
    def xsdir(self, xsdir: str | None) -> None:
        """Set the XSDIR path.

        Parameters
        ----------
        xsdir : str or None
            XSDIR path.
        """
        self._xsdir = xsdir

    @property
    def modes(self) -> list[str]:
        """Transported particle names.

        Returns
        -------
        list[str]
            Particle names.
        """
        return self._modes

    @modes.setter
    def modes(self, modes: list[str]) -> None:
        """Set the transported particle names.

        Parameters
        ----------
        modes : list[str]
            Particle names (not lower-cased here).
        """
        self._modes = modes

    @property
    def lost(self) -> int:
        """Maximum number of lost particles.

        Returns
        -------
        int
            Lost particle limit.
        """
        return self._lost

    @lost.setter
    def lost(self, lost: int) -> None:
        """Set the maximum number of lost particles.

        Parameters
        ----------
        lost : int
            Lost particle limit.
        """
        self._lost = lost

    @property
    def messages(self) -> list[MCNPCard]:
        """MCNP MESSAGE block cards.

        Returns
        -------
        list[MCNPCard]
            A ``MESSAGE:`` card with the XSDIR, if set, else empty.
        """
        messages = []
        if self._xsdir:
            messages.append(MCNPCard('MESSAGE:', [[f'XSDIR={self._xsdir}']]))
        return messages
    
    @property
    def nps(self) -> int:
        """Number of particle histories.

        Returns
        -------
        int
            Number of histories.
        """
        return self._nps
    
    @nps.setter
    def nps(self, nps: int) -> None:
        """Set the number of particle histories.

        Parameters
        ----------
        nps : int
            Number of histories.
        """
        self._nps = nps

    @property
    def secondary_particle_biasing(self) -> dict[str, dict[str, Any]] | None:
        """SPABI card data per secondary particle.

        Returns
        -------
        dict[str, dict[str, Any]] or None
            Secondary particle biasing data.
        """
        return self._spabi
    
    @secondary_particle_biasing.setter
    def secondary_particle_biasing(self, parameters: dict[str, dict[str, Any]]) -> None:
        """Set the SPABI card data.

        Parameters
        ----------
        parameters : dict[str, dict[str, Any]]
            Per secondary particle, a dict with ``primary_particles``,
            ``energies`` and ``multipliers``.

        Raises
        ------
        Exception
            If ``parameters`` is not a dict.
        """
        if isinstance(parameters, dict):
            self._spabi = parameters
        else:
            msg = f'SPABI card requires dictionary of secondary particles types' \
                  f'with a subdict of primary_particles, energies and multipliers'
            raise Exception(msg)
        
    @property
    def stop_tallies(self) -> dict[Tally, float]:
        """STOP card relative error cutoff per tally.

        Returns
        -------
        dict[Tally, float]
            Tally to cutoff mapping.
        """
        return self._stop_tallies
    
    @stop_tallies.setter
    def stop_tallies(self, stop_tallies: dict[Tally, float]) -> None: 
        """Set the STOP card tallies.

        Parameters
        ----------
        stop_tallies : dict[Tally, float]
            Tally to relative error cutoff mapping.
        """
        # cehck types
        check_types(stop_tallies, [dict])

        # 
        self._stop_tallies = stop_tallies
        
    @property
    def forced_collisions(self) -> dict[str, dict[Cell, float]] | None:
        """FCL card weighting per cell for each particle.

        Returns
        -------
        dict[str, dict[Cell, float]] or None
            Forced collision data.
        """
        return self._forced_collisions
    
    @forced_collisions.setter
    def forced_collisions(self, parameters: dict[str, dict[Cell, float]]) -> None:
        """Set the FCL card data.

        Parameters
        ----------
        parameters : dict[str, dict[Cell, float]]
            Per particle, the forced collision fraction for each cell.

        Raises
        ------
        Exception
            If ``parameters`` is not a dict.
        """
        if isinstance(parameters, dict):
            self._forced_collisions = parameters
        else:
            msg = f'FCL card requires dictionary of particle type and sub-dict of' \
                  f'cells with the fraction probability of forced collision'
            raise Exception(msg)

    @property
    def mcnp_card(self) -> MCNPCard:
        """Run parameter cards as an NPS card with sub-cards.

        Sub-cards include PRDMP, MODE, LOST and, where set, VOID, PHYS, CUT,
        STOP, WWP, IDUM, RDUMS, SPABI and FCL.

        Returns
        -------
        MCNPCard
            The NPS card with the other run cards as sub-cards.
        """
        # setr up mcnp card
        mcnp_card = MCNPCard('NPS', [[f'{self._nps:.{formatting.MCNP_DP}}']])

        # add sub card for prdmp
        mcnp_card.add_subcard(MCNPCard('PRDMP', [[f'{self._nps/self._prdmp_intervals:.{formatting.MCNP_DP}e}', f'{self._nps/self._prdmp_intervals:.{formatting.MCNP_DP}e}', f'{+(self._write_mctal)}', '1', f'{self._nps/self._prdmp_intervals:.{formatting.MCNP_DP}e}']]))

        # add sub card for modes
        modes = ' '.join([MCNP_PARTICLE_DESIGNATORS[mode.lower()].upper() for mode in self._modes])
        mcnp_card.add_subcard(MCNPCard(f'MODE {modes}', [[]]))

        # set up card for max lost particles
        mcnp_card.add_subcard(MCNPCard('LOST', [[f'{int(self._lost)}']]))

        # set up void card
        if self._void:
            mcnp_card.add_subcard(MCNPCard('VOID', [[]]))

        # add physics cards
        for particle, parameters in self._physics_parameters.items():
            mcnp_card.add_subcard(MCNPCard(f'PHYS:{MCNP_PARTICLE_DESIGNATORS[particle.lower()].upper()}', [[f'{v:.{6}}' for v in parameters]]))

        # add cut cards
        for particle, parameters in self.cut_parameters.items():
            mcnp_card.add_subcard(MCNPCard(f'CUT:{MCNP_PARTICLE_DESIGNATORS[particle.lower()].upper()}', [[f'{v:.{6}}' for v in parameters.values()]]))

        # stop card
        for tally, cutoff in self._stop_tallies.items():
            mcnp_card.add_subcard(MCNPCard('STOP', [[f'F{tally.number}{tally.mcnp_tally_designator}', f'{cutoff:.{formatting.MCNP_DP}}']]))

        # weight window
        if self._weight_window:
            for particle, parameters in self._weight_window.items():
                as_strings = numbers_to_strings(parameters)
                mcnp_card.add_subcard(MCNPCard(f'WWP:{particle.upper()}', [as_strings]))

        # idums and rdums
        if self._idums:
            mcnp_card.add_subcard(MCNPCard('IDUM', [numbers_to_strings(self._idums)]))

        if self._rdums:
            mcnp_card.add_subcard(MCNPCard('RDUMS', [numbers_to_strings(self._idums, dp=formatting.MCNP_DP)]))

        # spabi card
        if self._spabi:
            for particle, params in self._spabi.items():
                spabi_values = []
                spabi_values.append(''.join(params['primary_particles']))
                for energy, multiplier in zip(params['energies'], params['multipliers']):
                    spabi_values.append(energy)
                    spabi_values.append(multiplier)
                mcnp_card.add_subcard(MCNPCard(f'SPABI:{particle}', [numbers_to_strings(spabi_values, dp=formatting.MCNP_DP)]))
    
        # forced collisions
        if self._forced_collisions:
            for particle, cell_weightings in self._forced_collisions.items():
                cell_number_with_weightings = {cell.number: weighting for cell, weighting in cell_weightings.items()}
                cell_weightings = [cell_number_with_weightings[cell_number] if cell_number in cell_number_with_weightings else 0.0 for cell_number  in range(1, max(cell_number_with_weightings)+1)]
                mcnp_card.add_subcard(MCNPCard(f'FCL:{particle}', [numbers_to_strings(cell_weightings, dp=formatting.MCNP_DP)]))

        return mcnp_card

class MCNPTallyParameters:
    """MCNP-specific tally options.

    Parameters
    ----------
    tally_divisor : float or list[float], optional
        Tally divisor (SD card), either a single value or one per target, by
        default None
    multiplier : float, optional
        Tally multiplier, by default None
    cell_flags : list[Cell], optional
        Cells used for cell flagging (CF card), by default None
    surface_flags : list[Surface], optional
        Surfaces used for surface flagging (SF card), by default None
    dose_interpolation : str, optional
        Interpolation for dose functions, by default 'log-log'
    """

    def __init__(self, tally_divisor: float | list[float] | None = None, multiplier: float | None = None, cell_flags: list[Cell] | None = None, surface_flags: list[Surface] | None = None, 
                 dose_interpolation: str='log-log') -> None:
        """Initialise the MCNP tally parameters."""
        self._tally_divisor: float | list[float] | None = tally_divisor
        self._multiplier: float | None = multiplier
        self._cell_flags: list[Cell] | None = cell_flags
        self._surface_flags: list[Surface] | None = surface_flags
        self.dose_interpolation: str = dose_interpolation

    @property
    def tally_divisor(self) -> float | list[float] | None:
        """Tally divisor, a single value or one per target.

        Returns
        -------
        float or list[float] or None
            Tally divisor.
        """
        return self._tally_divisor
    
    @property
    def multiplier(self) -> float | None:
        """Tally multiplier.

        Returns
        -------
        float or None
            Multiplier.
        """
        return self._multiplier
    
    @property
    def cell_flags(self) -> list[Cell] | None:
        """Cells used for cell flagging.

        Returns
        -------
        list[Cell] or None
            Flagged cells.
        """
        return self._cell_flags
    
    @property
    def surface_flags(self) -> list[Surface] | None:
        """Surfaces used for surface flagging.

        Returns
        -------
        list[Surface] or None
            Flagged surfaces.
        """
        return self._surface_flags

    @property
    def dose_interpolation(self) -> str:
        """Interpolation for dose functions.

        Returns
        -------
        str
            Interpolation scheme, e.g. 'log-log'.
        """
        return self._dose_interpolation
    
    @tally_divisor.setter
    def tally_divisor(self, tally_divisor: float | list[float] | None) -> None:
        """Set the tally divisor.

        Parameters
        ----------
        tally_divisor : float or list[float] or None
            Tally divisor.
        """
        self._tally_divisor = tally_divisor

    @multiplier.setter
    def multiplier(self, multiplier: float | None) -> None:
        """Set the tally multiplier.

        Parameters
        ----------
        multiplier : float or None
            Multiplier.
        """
        self._multiplier = multiplier

    @cell_flags.setter
    def cell_flags(self, cell_flags: list[Cell] | None) -> None:
        """Set the cells used for cell flagging.

        Parameters
        ----------
        cell_flags : list[Cell] or None
            Flagged cells.
        """
        self._cell_flags = cell_flags

    @surface_flags.setter
    def surface_flags(self, surface_flags: list[Surface] | None) -> None:
        """Set the surfaces used for surface flagging.

        Parameters
        ----------
        surface_flags : list[Surface] or None
            Flagged surfaces.
        """
        self._surface_flags = surface_flags

    @dose_interpolation.setter
    def dose_interpolation(self, dose_interpolation: str) -> None:
        """Set the interpolation for dose functions.

        Parameters
        ----------
        dose_interpolation : str
            Interpolation scheme.
        """
        self._dose_interpolation: str = dose_interpolation


class MCNPCellParameters:
    """MCNP-specific cell options.

    Parameters
    ----------
    graveyard : bool, optional
        Whether the cell is a graveyard; if True all importances are set to
        zero, by default False
    importances : dict[str, float], optional
        Importance per particle name, by default None (1.0 for neutron,
        photon and electron)
    volume : float, optional
        Manually assigned cell volume written as VOL, by default None
    """

    def __init__(self, graveyard: bool = False, importances: dict[str, float] | None = None, volume: float | None = None) -> None:
        """Initialise the MCNP cell parameters."""
        self._importances: dict[str, float] = importances if importances \
            else {p: 1.0 for p in ['neutron', 'photon', 'electron']}
        self._graveyard: bool = graveyard
        self._volume: float | None = volume

        self._cell_read_number: int | None = None
        self._cell_read_importances: dict[str, float] | None = None

        if graveyard:
            for particle in self.importances:
                self.importances[particle] = 0.0

    @property
    def graveyard(self) -> bool:
        """Whether the cell is a graveyard.

        Returns
        -------
        bool
            Graveyard flag.
        """
        return self._graveyard
    
    @graveyard.setter
    def graveyard(self, graveyard: bool) -> None:
        """Set the graveyard flag (stored as ``_graveyward``, see Notes).

        Parameters
        ----------
        graveyard : bool
            Graveyard flag.

        Notes
        -----
        The value is assigned to ``self._graveyward`` (sic), so the getter
        does not reflect it and importances are not updated.
        """
        self._graveyward: bool = graveyard

    @property
    def importances(self) -> dict[str, float]:
        """Importance per particle name.

        Returns
        -------
        dict[str, float]
            Particle importances.
        """
        return self._importances

    @property
    def volume(self) -> float | None:
        """Manually assigned cell volume.

        Returns
        -------
        float or None
            Cell volume.
        """
        return self._volume
    
    @volume.setter
    def volume(self, volume: float | None) -> None:
        """Set the manually assigned cell volume.

        Parameters
        ----------
        volume : float or None
            Cell volume.
        """
        self._volume = volume


class MCNPMaterialParameters(SerialisableFromDict):
    """MCNP-specific material options, serialisable to/from a dictionary.

    Attributes
    ----------
    sab_cards : list[str]
        S(a,b) table identifiers written to the MT card.
    libs : dict[str, str]
        Particle-dependent library per particle designator (e.g. ``PLIB``).
    """

    def __init__(self) -> None:
        """Initialise the MCNPMaterialParameters with empty values."""
        self.sab_cards: list[str] = []
        self.libs: dict[str, str] = {}
        self._library: str | None = None
        self._library_by_isotope: dict[str, str] | None = None

        # load seralisable from dictionary
        self._dictable_attributes: list[str] = ['sab_cards', 'libs', 'library', 'library_by_isotope']
        self._dictable_sub_classes: dict[str, type] = {}

    @property
    def library(self) -> str:
        """Default library suffix appended to each zaid (e.g. ``.80c``).

        Returns
        -------
        str
            Library suffix, or an empty string if unset.
        """
        if self._library:
            return self._library
        else:
            return ''

    @library.setter
    def library(self, library: str | None) -> None:
        """Set the default library suffix.

        Parameters
        ----------
        library : str or None
            Library suffix.
        """
        self._library = library

    @property
    def library_by_isotope(self) -> dict[str, str] | None:
        """Library suffix overrides per isotope.

        Returns
        -------
        dict[str, str] or None
            Isotope to library suffix mapping.
        """
        return self._library_by_isotope

    @library_by_isotope.setter
    def library_by_isotope(self, library_by_isotope: dict[str, str] | None) -> None:
        """Set the library suffix overrides per isotope.

        Parameters
        ----------
        library_by_isotope : dict[str, str] or None
            Isotope to library suffix mapping.
        """
        self._library_by_isotope = library_by_isotope
