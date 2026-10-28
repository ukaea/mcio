"""Material definitions for mcio.

Contains the :class:`Material` class (a single material composition with
MCNP, OpenMC and GDML output), the :class:`Materials` collection, and helper
functions for loading materials from json files and assigning them to solids
from CAD colour or ``_mN_`` structure labels.
"""
from __future__ import annotations

import json
import re
import copy
from typing import TYPE_CHECKING

try:
    import openmc
except ModuleNotFoundError:
    pass

from .formatting import formatting
from .mcnp import MCNPCard, MCNPMaterialParameters
from .xml import XMLTag
from .data import (
    ATOMIC_NUMBER, 
    ISOTOPIC_COMPOSITION_BY_ATOM_FRACTION,
    ISOTOPIC_COMPOSITION_BY_WEIGHT_FRACTION,
    ZAID_FROM_ISOTOPE,
    ISOTOPE_FROM_ZAID,
    RELATIVE_ATOMIC_MASS, 
    ELEMENT_FROM_ISOTOPE, 
    ELEMENT
    )
from .decomposeparams import DecompositionParameters
from .utils import SerialisableFromDict

# import first to allow logging of optional modules 
from mcio import logger

if TYPE_CHECKING:
    import openmc


def _normalise_label(label: str) -> str:
    """Normalise a material json label key to the form used in the structure tree.

    Underscores are removed and numeric labels are prefixed with ``m``, so
    ``_m1_``, ``m1`` and ``1`` all become ``m1``.

    Parameters
    ----------
    label : str
        Label key from the materials json file.

    Returns
    -------
    str
        The normalised label.
    """
    label = label.replace('_', '')
    return f'm{label}' if label.isdigit() else label

def _materials_from_json(json_path: str) -> tuple[str, dict[str, Material]]:
    """Load materials from a json file and detect the assignment method.

    The json keys must either all be rgb colour strings (e.g. ``015_150_025``)
    or all be material labels (e.g. ``m1`` / ``_m1_``). Rgb keys are zero
    padded to three digits per channel, and label keys are normalised with
    :func:`_normalise_label` (``_m1_``, ``m1`` and ``1`` become ``m1``).

    Parameters
    ----------
    json_path : str
        Path to the materials json file.

    Returns
    -------
    tuple[str, dict[str, Material]]
        The material assignment method (``'rgb'`` or ``'label'``) and a
        dictionary mapping the normalised key to the :class:`Material`.

    Raises
    ------
    KeyError
        If an rgb key is repeated, or keys are not all of a single type.
    """
    RE_CHECK_RGB = re.compile(r'^\d{1,3}_\d{1,3}_\d{1,3}$')
    RE_CHECK_MN = re.compile(r'^_*m?\d+_+$|^_*m?\d+$')

    with open(json_path, 'r') as input_file:
        data = json.load(input_file)

    # check each key type for either rgb or _mn_label
    key_types = ['rgb' if RE_CHECK_RGB.match(k) else ('label' if RE_CHECK_MN.match(k) else None) for k in data ]
    n_keys = len(data)

    # set up materials dictionary to store keys and material objects
    materials: dict[str, Material] = {}

    if key_types.count('rgb') == n_keys:
        assignment_method = 'rgb'
        for colour_label, mat_dict in data.items():
            red, green, blue = colour_label.split('_')
            colour = f'{red:0>3}_{green:0>3}_{blue:0>3}'
            if colour in materials:
                msg = f'rgb key in materials json repeated ({colour})'
                logger.fatal(msg)
                raise KeyError(msg)
            materials[colour] = Material.from_dict(mat_dict)
    elif key_types.count('label') == n_keys:
        assignment_method = 'label'
        # case where all keys are labels
        materials: dict[str, Material] = {_normalise_label(label): Material.from_dict(mat_dict) for label, mat_dict in data.items()}
    else:
        msg = f'material json keys should all be either rgb (e.g. 015_150_025) or label (e.g. 1)'
        logger.fatal(msg)
        raise KeyError(msg)

    return assignment_method, materials

def _materials_by_colour_from_json(json_path: str) -> dict[str, Material]:
    """Load materials keyed by rgb colour from a json file.

    Colour keys are normalised to zero padded three digit channels, e.g.
    ``17_210_1`` becomes ``017_210_001``.

    Parameters
    ----------
    json_path : str
        Path to the materials json file.

    Returns
    -------
    dict[str, Material]
        Dictionary mapping normalised rgb strings to materials.
    """
    # load material data from json file
    with open(json_path, 'r') as input_file:
        data = json.load(input_file)

    # de-serialise dict data to mcio material instances
    materials_raw: dict[str, Material] = {colour: Material.from_dict(mat_dict) for colour, mat_dict in data.items()}

    # parse colour string to ensure rgb has format 001_022_333 (e.g. 17_210_1 -> 017_210_001)
    materials: dict[str, Material] = {}
    for colour_raw, material in materials_raw.items():
        red, green, blue = colour_raw.split('_')
        colour = f'{red:0>3}_{green:0>3}_{blue:0>3}'
        materials[colour] = material

    # return materails to colour lookup dict
    return materials

def _materials_by_label_from_json(json_path: str) -> dict[str, Material]:
    """Load materials keyed by label from a json file.

    Labels are normalised with :func:`_normalise_label` (``_m1_``, ``m1`` and
    ``1`` all become ``m1``).

    Parameters
    ----------
    json_path : str
        Path to the materials json file.

    Returns
    -------
    dict[str, Material]
        Dictionary mapping labels to materials.
    """
    # load material data from json file
    with open(json_path, 'r') as input_file:
        data = json.load(input_file)

    # de-serialise dict data to mcio material instances
    materials: dict[str, Material] = {_normalise_label(label): Material.from_dict(mat_dict) for label, mat_dict in data.items()}

    # return materails to colour lookup dict
    return materials

def _material_lookup(material_assignment_method: str, red: int, green: int, blue: int, structure: list[str], materials: dict[str, Material], decom_params: DecompositionParameters) -> Material | None:
    """Find the material for a solid from its colour or structure labels.

    Solids whose structure tree contains ``_void_`` or ``_m0_`` are treated as
    void and return None. For the ``'rgb'`` method the solid colour is looked
    up; for the ``'label'`` method the single unique ``_mN_`` label found in
    the structure tree is looked up.

    Parameters
    ----------
    material_assignment_method : str
        Either ``'rgb'`` or ``'label'``.
    red : int
        Red channel of the solid colour (0-255).
    green : int
        Green channel of the solid colour (0-255).
    blue : int
        Blue channel of the solid colour (0-255).
    structure : list[str]
        Names of the nodes in the STEP structure tree leading to the solid.
    materials : dict[str, Material]
        Materials keyed by rgb string or label, as returned by
        :func:`_materials_from_json`.
    decom_params : DecompositionParameters
        Decomposition parameters (``model_implicit_void_for_no_mateirals`` is
        used when no label is found).

    Returns
    -------
    Material or None
        The matched material, or None for void / unlabelled solids.

    Raises
    ------
    Exception
        If the rgb colour is not found, or more than one ``_mN_`` label is
        assigned to the solid.
    ValueError
        If the label is not found in ``materials`` or the assignment method is
        not recognised.
    """
    # join sub-structure tree into single string to allow regex
    structure_as_string = '/'.join(structure)

    # define void regex checks in structure tree, and check if void (_void_ or _m0_ in structure string)
    REGEX_COMPILED_VOID_LABEL = re.compile(r'_void_', re.IGNORECASE)
    REGEX_COMPILED_VOID_MATERIAL = re.compile(f'_m0_', re.IGNORECASE)

    if bool(REGEX_COMPILED_VOID_LABEL.search(''.join(structure))) or bool(REGEX_COMPILED_VOID_MATERIAL.search(''.join(structure))):
        return None

    if material_assignment_method == 'rgb':
        rgb_as_string = f'{red:0>3}_{green:0>3}_{blue:0>3}'
        if rgb_as_string in materials:
            return materials[rgb_as_string]
        else:
            
            msg = f'rgb value for {structure_as_string} {red} {green} {blue} ({rgb_as_string}) not found in materials data'
            logger.fatal(msg)
            raise Exception(msg)

    elif material_assignment_method == 'label':
        # get the bottom level m tag
        regex_m_label = re.compile(r"_(m\d+)_")
        m_labels = []
        for sub_structure in structure:
            m_labels.extend(re.findall(regex_m_label, sub_structure))

        # look for unique m labels, as sometimes CAD codes will save a solid 
        # into a component, 
        # e.g. 
        #  -container_m1_
        #  --solid_1
        #  --container_sub
        #  --solid_2
        #
        # solid_1 will contimes be placed in component called container_m1_
        # and hence will repeat application of mn labels in structure tree
        
        # find unique labels in each node and parents of node
        m_labels = list(set(m_labels))
        
        # check only one m label in structure tree for this solid. 
        if len(m_labels) == 1 and m_labels[0] in materials:
            return materials[m_labels[0]]
        elif len(m_labels) == 1 and m_labels[0] not in materials:
            msg = f'material label {m_labels[0]} not found in materials json'
            logger.fatal(msg)
            raise ValueError(msg)
        elif len(m_labels) > 1:
            m_labels_joined = ' '.join(m_labels)
            structure_as_string = '/'.join(structure)
            msg = f'only one _mn_ label should be assigned to a body, not {m_labels_joined} (from solid {structure_as_string})'
            logger.fatal(msg)
            raise Exception(msg)
        elif len(m_labels) == 0 and decom_params.model_implicit_void_for_no_mateirals:
            pass
        elif len(m_labels) == 0 and not decom_params.model_implicit_void_for_no_mateirals:
            structure_as_string = '/'.join(structure)
            msg = f'implicit void for cells is off, and {structure_as_string} not assigned any material...'
            logger.fatal(msg)
            raise ValueError(msg)
    else:
        msg = f'material_assignmnet_method hsould be rgb or label not {material_assignment_method}'
        logger.fatal(msg)
        raise ValueError(msg)

def openmc_material_to_mcio_material(material_openmc: "openmc.Material") -> Material:
    """Convert an OpenMC material into an mcio material.

    Parameters
    ----------
    material_openmc : openmc.Material
        OpenMC material with nuclides all of the same fraction type and
        density in g/cc, g/cm3 or kg/m3.

    Returns
    -------
    Material
        Equivalent mcio material, with density in g/cm3.

    Raises
    ------
    ValueError
        If the material has no nuclides, mixed fraction types, or an
        unsupported density unit.
    """
    # get a list of fraction types
    if not material_openmc.nuclides:
        msg = f'openmc material passed to openmc_material_to_mcio_material()'+\
              f' contains no elements or isotopes'
        logger.fatal(msg)
        raise ValueError(msg)
    elif all([ptype == 'ao' for _, _, ptype in material_openmc.nuclides]):
        fraction_type = 'ao'
    elif all([ptype == 'wo' for _, _, ptype in material_openmc.nuclides]):
        fraction_type = 'wo'
    else:
        msg = f'openmc material contains mixed fraction types. Can\'t convert'
        logger.fatal(msg)
        raise ValueError(msg)

    # check density units
    if material_openmc.density_units in ['g/cc', 'g/cm3']:
        density = material_openmc.density
    elif material_openmc.density_units == 'kg/m3':
        density = material_openmc.density * 1e-3
    else:
        msg = f'density_unit should be either g/cc g/cm3 or kg/m3 not' +\
              f'{material_openmc.density_units}'
        logger.fatal(msg)
        raise ValueError(msg)

    # extract composition from openmc material
    composition = {}
    for nuclide, fraction, ptype in material_openmc.nuclides:
        composition[nuclide] = fraction

    # return the mcio material
    return Material(
        density=density,
        composition=composition,
        name=material_openmc.name,
        fraction_type=fraction_type)


class Material(SerialisableFromDict):
    """A material defined by density and elemental/isotopic composition.

    Composition keys may be element symbols (e.g. ``Li``), isotopes (element
    symbol and mass number, e.g. ``Li6``) or zaids as strings (e.g.
    ``3006``).

    Parameters
    ----------
    density : float, optional
        Material density, by default 0.0
    composition : dict[str, float], optional
        Mapping of element, isotope or zaid to fraction, by default None
        (empty composition)
    fraction_type : str, optional
        ``'ao'`` for atom fractions or ``'wo'`` for weight fractions, by
        default 'ao'
    density_unit : str, optional
        One of ``'g/cc'``, ``'g/cm3'`` or ``'atoms/barn-cm'``, by default
        'g/cm3'
    name : str, optional
        Material name, by default None
    reference : str, optional
        Reference / source of the material data, by default None
    number : int, optional
        Material number, by default None

    Raises
    ------
    Exception
        If ``fraction_type`` or ``density_unit`` is not recognised.
    """

    def __init__(self,
                 density: float = 0.0, 
                 composition: dict[str, float] | None = None,
                 fraction_type: str = 'ao', 
                 density_unit: str = 'g/cm3',
                 name: str | None = None, 
                 reference: str | None = None,
                 number: int | None = None) -> None:
        """Initialise the Material."""
        self.density: float = density
        self.composition: dict[str, float] = composition if composition else {}
        self.number: int | None = number
        self.name: str | None = name
        self.reference: str | None = reference
        self.fraction_type: str = fraction_type
        self.density_unit: str = density_unit
        self.mcnp: MCNPMaterialParameters = MCNPMaterialParameters()

        # load seralisable from dictionary
        self._dictable_attributes: list[str] = ['density', 'composition', 'number', 'reference',
                                     'fraction_type', 'density_unit', 'name']
        self._dictable_sub_classes: dict[str, type] = {'mcnp': MCNPMaterialParameters}

        # run post init
        self.__post_init__()

    def __post_init__(self) -> None:
        """Reset cached data and validate the fraction type and density unit.

        Raises
        ------
        Exception
            If ``fraction_type`` is not ``'ao'``/``'wo'`` or ``density_unit``
            is not recognised.
        """
        # anything needed to be checked other than setting variables in
        # self._dictable_attributes and self._dictable_sub_classes

        self._openmc_material: openmc.Material | None = None
        
        # check fraction type
        if self.fraction_type not in ['ao', 'wo']:
            msg = f'fraction_type for material should be either ao for atom or wo for weight fraction'
            logger.fatal(msg)
            raise Exception(msg)

        # check density units
        if self.density_unit not in ['g/cc', 'g/cm3', 'atoms/barn-cm']:
            msg = f'density_unit should be either g/cc, g/cm3'
            logger.fatal(msg)
            raise Exception(msg)

        pass

    @property
    def isotopes(self) -> list[str]:
        """Composition keys that are isotopes.

        Returns
        -------
        list[str]
            Isotopes (e.g. ``Li6``) present directly in the composition.
        """
        isotopes = []
        for component in self.composition.keys():
            if component in RELATIVE_ATOMIC_MASS:
                isotopes.append(component)

        return isotopes

    @property
    def elements(self) -> list[str]:
        """Composition keys that are elements.

        Returns
        -------
        list[str]
            Element symbols present directly in the composition.
        """
        elements = []
        for component in self.composition.keys():
            if component in ATOMIC_NUMBER:
                elements.append(component)

        return elements

    def __hash__(self) -> int:
        """Hash the material by its MCNP card string.

        Returns
        -------
        int
            Hash of the MCNP card string.
        """
        return hash(f'{self.mcnp_card.as_string}')

    def __hash_string__(self) -> str:
        """Return the MCNP card string of a copy of the material numbered 0.

        Useful for comparing materials independent of their number.

        Returns
        -------
        str
            MCNP card string with material number 0.
        """

        # cache the material number
        material_number_cached = self.number

        # set the material number to zero
        self.number = 0

        # produce the mcnp card of the material with number 0
        material_string = self.mcnp_card.as_string

        # return the number of the original one set via the cached value
        self.number = material_number_cached

        # return the mcnp string of the material
        return material_string

    def _combine_isotopes(self) -> dict[str, float]:
        """Expand the composition into isotopes.

        Zaids are converted to isotopes and elements are split by natural
        abundance (by atom or weight fraction to match ``fraction_type``);
        fractions of repeated isotopes are summed.

        Returns
        -------
        dict[str, float]
            Mapping of isotope (e.g. ``Li6``) to fraction.

        Raises
        ------
        TypeError
            If a component is not an element, isotope or zaid string.
        """
        # isotopic fraction lookup for fraction types
        isotopic_fraction_lookup = {
                'ao': ISOTOPIC_COMPOSITION_BY_ATOM_FRACTION,
                'wo': ISOTOPIC_COMPOSITION_BY_WEIGHT_FRACTION
            }

        # check all components for either being element, isotopes or zaids, else raise
        for component in self.composition:
            if (component not in ATOMIC_NUMBER) and (not re.fullmatch(r"([A-Z][a-z]?)([1-9]\d*)", component.strip())) and ((not isinstance(component, str)) or (not isinstance(component, str))):
                msg = f'component of materials should be either element, isotope or zaid as string, not {component}'
                logger.fatal(msg)
                raise TypeError(msg)

        # split composition into elements and isotopes
        isotopes = {component: fraction for component, fraction in self.composition.items() if component not in ATOMIC_NUMBER and not component.isnumeric()}
        zaids = {component: fraction for component, fraction in self.composition.items() if component not in ATOMIC_NUMBER and component.isnumeric()}
        elements = {component: fraction for component, fraction in self.composition.items() if component in ATOMIC_NUMBER}

        # combine zaids and isotopes
        for zaid, fraction in zaids.items():
            # convert zaid to isotope as string of element symbol and mass number
            atomic_number = int(zaid[:-3])
            mass_number = int(zaid[-3:])
            isotope = f'{ELEMENT[atomic_number]}{mass_number}'

            # check if isotope already defined
            if isotope in isotopes:
                isotopes[isotope] += fraction
            else:
                isotopes[isotope] = fraction

        # split elements into natural abundances and combine with isotopes
        for element, fraction in elements.items():
            for isotope, isotope_fraction_natural in isotopic_fraction_lookup[self.fraction_type][element].items():
                if isotope in isotopes:
                    isotopes[isotope] += fraction * isotope_fraction_natural
                else:
                    isotopes[isotope] = fraction * isotope_fraction_natural

        return isotopes

    @property
    def mcnp_card(self) -> MCNPCard:
        """Generate an MCNP material card from the material composition.

        Returns
        -------
        MCNPCard
            The MCNP material card (with an MT sub-card if S(a,b) cards are
            set).

        Raises
        ------
        TypeError
            TypeError raised if composition contains entrance that is neither
            an element (as one or two letter symbol, e.g. Li), isotopes (as 
            symbol and numeric value as mass number, e.g. Li6) or zaid 
            (4-6 integer long numeric of the atomic number and mass number,
            e.g. 3006, 03006 or 003006).
        """
        # define data
        data = {'name': self.name,
                'density': f'{self.density:{formatting.MCNP_FMT_MATERIAL_DENSITY}} g/cc',
                'reference': self.reference}
        
        # set up mcnp card to store card data
        mcnp_card = MCNPCard(f'M{self.number}', [], header_data=data)

        # fraction modifier for weight fraction
        fraction_modifier_composition = {'ao': 1, 'wo': -1}

        # split and combine composition into isotopes
        isotopes = self._combine_isotopes()

        # sort isotopes by atomic then mass numbers
        zaids_sorted = {int(ZAID_FROM_ISOTOPE[isotope]): isotope for isotope in isotopes}

        # iterate over components within material
        for zaid_sorted in sorted(zaids_sorted):
            # get fraction and zaid as string
            isotope = zaids_sorted[zaid_sorted]
            fraction = isotopes[isotope]
            zaid = ZAID_FROM_ISOTOPE[isotope]
            
            # get the default library
            library = self.mcnp.library if self.mcnp.library else ''

            if self.mcnp.library_by_isotope and isotope in self.mcnp.library_by_isotope:
                library = self.mcnp.library_by_isotope[isotope]
            elif self.mcnp.library:
                library = self.mcnp.library
            else:
                library = ''

            # convert isotopes into zaid
            mcnp_card.add_row([zaid+library, f'{fraction_modifier_composition[self.fraction_type] * fraction:{formatting.MCNP_FMT_FRACTION}}', f'$ {isotope}'])

        # add particle dependent libs
        if self.mcnp.libs:
            for particle, library in self.mcnp.libs.items():
                mcnp_card.add_row([f'{particle.upper()}={library}'])

        # add sab cards
        if self.mcnp.sab_cards:
            mcnp_card.add_subcard(MCNPCard(f'MT{self.number}', [self.mcnp.sab_cards]))

        return mcnp_card

    def from_mcnp_card(self, mcnp_card: str) -> Material:
        """Create a material from an MCNP card (not yet implemented).

        Parameters
        ----------
        mcnp_card : str
            MCNP material card string.

        Returns
        -------
        Material
            Currently returns None, as parsing is not implemented.
        """
        # parse mcnp card
        pass

    @property
    def openmc_material(self) -> openmc.Material:
        """OpenMC material equivalent of this material, created on first use.

        Returns
        -------
        openmc.Material
            Cached OpenMC material with density in g/cc.
        """
        # check if openmc material already generated, else create openmc material and store it
        if not self._openmc_material:
            self._openmc_material = openmc.material.Material()
            self._openmc_material.set_density(units='g/cc', density=self.density)

            for component, fraction in self.composition.items():
                # case where zaid provided, convert to isotope, E.G. 006012 -> C12
                if component.isnumeric():
                    atomic_number = int(component[:-3])
                    mass_number = int(component[-3:])
                    component = f'{ELEMENT[atomic_number]}{mass_number}'

                if component in ZAID_FROM_ISOTOPE:
                    self._openmc_material.add_nuclide(nuclide=component, percent=fraction, percent_type=self.fraction_type)
                elif component in ATOMIC_NUMBER:
                    self._openmc_material.add_element(element=component, percent=fraction, percent_type=self.fraction_type)

        return self._openmc_material

    @property
    def gdml_xml_tag(self) -> XMLTag:
        """GDML ``material`` XML tag (currently with no attributes or children).

        Returns
        -------
        XMLTag
            Empty GDML material tag.
        """
        attributes: dict[str, str] = {}
        material_tag = XMLTag(f'material', attributes=attributes)

        return material_tag

    @property
    def serpent2_card(self) -> None:
        """Serpent2 material card (not yet implemented).

        Returns
        -------
        None
            Always None.
        """
        pass

    def strip_isotopes(self, isotopes_to_strip: list[str]) -> None:
        """Replace the composition with its isotopes, excluding those given.

        Elements and zaids are expanded to isotopes first (see
        :meth:`_combine_isotopes`).

        Parameters
        ----------
        isotopes_to_strip : list[str]
            Isotopes to remove (e.g. those missing from ADVANTG libraries).
        """
        # strip isotopes
        isotopes = self._combine_isotopes()

        # purge advantg isotopes to strip
        self.composition = {isotope: fraction for isotope, fraction in isotopes.items() if isotope not in isotopes_to_strip}




class Materials:
    """Collection of materials.

    Parameters
    ----------
    materials : list[Material]
        Materials in the collection.
    """

    def __init__(self, materials: list[Material]) -> None:
        """Initialise the Materials collection."""
        self._materials: list[Material] = materials
        self._isotopes: list[str] = []
        self._elements: list[str] = []

    @property
    def materials(self) -> list[Material]:
        """Materials in the collection.

        Returns
        -------
        list[Material]
            The materials.
        """
        return self._materials
    
    @property
    def isotopes(self) -> list[str]:
        """Isotopes appearing directly in any material composition.

        Returns
        -------
        list[str]
            Isotopes, in order of appearance (duplicates are not removed).
        """
        # extract all isotopes and order them by z and a
        isotopes = []
        for material in self._materials:
            for component in material.composition.keys():
                if component in RELATIVE_ATOMIC_MASS:
                    isotopes.append(component)

        return isotopes

    @property
    def elements(self) -> list[str]:
        """Elements appearing directly in any material composition.

        Returns
        -------
        list[str]
            Element symbols, in order of appearance (duplicates are not
            removed).
        """
        elements = []
        for material in self._materials:
            for component in material.composition.keys():
                if component in ATOMIC_NUMBER:
                    elements.append(component)

        return elements

    @property
    def gdml_xml_tag(self) -> XMLTag:
        """GDML ``materials`` XML tag containing isotope and element tags.

        Returns
        -------
        XMLTag
            Materials tag with an ``isotope`` child per isotope and an
            ``element`` child per element.
        """
        materials_tag = XMLTag('materials')

        # material_isotope_compositions = {}

        # for material in self._materials:
        #     isotopes = {}
        #     for composition, fraction in material.composition.items():
        #         # if composition in 
        #         pass

        # write isotopes
        for isotope in self.isotopes:
            # get properties of isotope
            element_of_isotope = ELEMENT_FROM_ISOTOPE[isotope]
            rel_atomic_mass = str(RELATIVE_ATOMIC_MASS[isotope])

            # create isotope tag
            isotope_att = {'Z': str(ATOMIC_NUMBER[element_of_isotope]),
                           'N': str(RELATIVE_ATOMIC_MASS[isotope]),
                           'name': isotope}
            isotope_tag = XMLTag('isotope', attributes=isotope_att)
            
            # set relative atomic mass in atom tag
            atom_att = {'type': 'A', 'value': rel_atomic_mass}
            atom_tag = XMLTag('atom', attributes=atom_att)
            isotope_tag.add_child(atom_tag)

            # add isotope tag to materials tag
            materials_tag.add_child(isotope_tag)

        # gather isotopes into elements to be used in materials
        for element in self.elements:
            # create element tags
            attributes = {'Z': str(ATOMIC_NUMBER[element]),
                          'formula': element,
                          'name': element}
            element_tag = XMLTag('element', attributes=attributes)
            
            # add element tag to materials tag
            materials_tag.add_child(element_tag)

        return materials_tag
