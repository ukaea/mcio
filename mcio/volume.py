"""CSG volume and Boolean expression classes.

Contains :class:`HalfVolume` (one side of a surface), the Boolean operators
:class:`And` (intersection) and :class:`Or` (union), and :class:`Volume`, which
wraps a Boolean expression. These are exported to MCNP strings, OpenMC regions
and Open CASCADE solids.
"""
from __future__ import annotations

import copy

from typing import TYPE_CHECKING

from mcio import logger
from .inversion import (
    _convert_mcio_half_volume_into_ocp_solid
)
from .geometric import BoundingBox
try:
    from .ocpbooleans import (
        _common_ocp_solid,
        _fuse_ocp_solid
    )
    from .decomposeparams import DecompositionParameters
except:
    pass

if TYPE_CHECKING:
    import openmc
    from OCP.TopoDS import TopoDS_Shape
    from .surface import Surface
    from .macrobody import Macrobody


class Boolean:
    """Base class for Boolean combinations of half volumes and other Booleans.

    Sub-classes :class:`And` and :class:`Or` populate ``_sub_booleans``.
    """

    def __init__(self) -> None:
        """Initialise the Boolean with no sub-booleans."""
        self._sub_booleans: list[Boolean] = []

    @property
    def sub_booleans(self) -> list[Boolean | HalfVolume]:
        """Sub-booleans and half volumes combined by this Boolean."""
        return self._sub_booleans

    @property
    def surfaces(self) -> list[Surface | Macrobody]:
        """All surfaces referenced by this Boolean, recursively (may repeat)."""
        surfaces = []

        for sub_bool in self._sub_booleans:
            surfaces.extend(sub_bool.surfaces)

        return surfaces

    def replace_duplicate_surfaces(self, surfaces: dict[str, Surface]) -> None:
        """Replace duplicate surfaces in all sub-booleans with shared instances.

        Parameters
        ----------
        surfaces : dict[str, Surface]
            Map of surface hash string to the surface instance to keep; updated
            in place with newly seen surfaces.
        """
        for sub_bool in self._sub_booleans:
            sub_bool.replace_duplicate_surfaces(surfaces)

    def __invert__(self) -> Boolean:
        """Return the complement using De Morgan's laws.

        An :class:`And` becomes an :class:`Or` (and vice versa) of the inverted
        sub-booleans.

        Returns
        -------
        Boolean
            The inverted Boolean.
        """
        self._inv_booleans = {And: Or, Or: And}
        inv_bool = self._inv_booleans[type(self)]
        sub_booleans = [~sb for sb in self._sub_booleans]

        inv_bool = inv_bool(sub_booleans)

        return inv_bool


class And(Boolean):
    """Intersection of half volumes and/or other Booleans.

    Nested :class:`And` operands are flattened into this one, and duplicate
    half volumes are dropped.

    Parameters
    ----------
    sub_booleans : list[Boolean | HalfVolume]
        Operands of the intersection.
    """

    def __init__(self, sub_booleans: list[Boolean | HalfVolume]) -> None:
        """Initialise the And Boolean."""
        super().__init__()

        self._sub_booleans = []

        half_vols_inserted: set[str] = set()

        for sub_bool in sub_booleans:
            if isinstance(sub_bool, And):
                # case where sub boolean is an And, combine And booleans into current,
                # skipping half volumes already added
                for nested in sub_bool.sub_booleans:
                    if isinstance(nested, HalfVolume):
                        if nested.__hash_string__() in half_vols_inserted:
                            continue
                        half_vols_inserted.add(nested.__hash_string__())
                    self._sub_booleans.append(nested)
            elif isinstance(sub_bool, Or):
                # case where sub boolean is an Or, add to the current
                self._sub_booleans.append(sub_bool)
            elif sub_bool.__hash_string__() not in half_vols_inserted:
                # case where a unique half volume is found, add to current
                self._sub_booleans.append(sub_bool)
                half_vols_inserted.add(sub_bool.__hash_string__())

    @property
    def mcnp_string(self) -> list[str]:
        """MCNP geometry tokens for the intersection.

        Sub-expressions that are lists are wrapped in parentheses.
        """
        sb_strings = [sb.mcnp_string for sb in self._sub_booleans]

        result = []

        for val in sb_strings:
            if isinstance(val, list):
                result.append('(')
                for subval in val:
                    result.append(subval)
                result.append(')')
            else:
                result.append(val)

        return result

    @property
    def openmc_region(self) -> openmc.Region:
        """OpenMC region formed by intersecting all sub-boolean regions."""
        vol_openmc = self._sub_booleans[0].openmc_region

        for sub_bool in self._sub_booleans[1:]:
            vol_openmc = vol_openmc & sub_bool.openmc_region

        return vol_openmc

    def ocp_solid(self, bounding_box: BoundingBox | None=None,
                  decom_params: DecompositionParameters | None=None) -> "TopoDS_Shape":
        """Build an Open CASCADE solid of the intersection.

        Parameters
        ----------
        bounding_box : BoundingBox, optional
            Bounds used to make half spaces finite, by default None (a
            +/-1000 box)
        decom_params : DecompositionParameters, optional
            Decomposition parameters for the OCP Boolean operations, by default
            None (default parameters)

        Returns
        -------
        TopoDS_Shape
            Solid representing the intersection.
        """
        # use default bounds
        bounding_box = bounding_box if bounding_box else BoundingBox(-1000, 1000, -1000, 1000, -1000, 1000)
        decom_params = decom_params if decom_params else DecompositionParameters()

        # get first volume in sub bools and ocp solid
        ocp_solid_vol = self._sub_booleans[0].ocp_solid(bounding_box, decom_params)

        # common all other sub bools to initial volume
        for sub_bool in self._sub_booleans[1:]:
            # generate common solids for all sub-booleans
            common_solids = _common_ocp_solid(ocp_solid_vol, sub_bool.ocp_solid(bounding_box, decom_params), decom_params).solids

            # fuse common solids, as _common_ocp_solid always extracts all solids from a common operation
            ocp_solid_vol = common_solids[0]
            for common_solid in common_solids[1:]:
                ocp_solid_vol = _fuse_ocp_solid(ocp_solid_vol, common_solid, decom_params, min_solids_required=1, max_solids_required=1)

        # return volume
        return ocp_solid_vol

    def __str__(self) -> str:
        """Return an indented tree representation of the intersection."""
        string = 'and:\n'
        for sub_boolean in self._sub_booleans:
            string += sub_boolean.__str__()
        return f'{string}\n'.replace('\n', '\n ')


class Or(Boolean):
    """Union of half volumes and/or other Booleans.

    Nested :class:`Or` operands are flattened into this one, and duplicate
    half volumes are dropped.

    Parameters
    ----------
    sub_booleans : list[Boolean | HalfVolume]
        Operands of the union.
    """

    def __init__(self, sub_booleans: list[Boolean | HalfVolume]) -> None:
        """Initialise the Or Boolean."""
        super().__init__()

        self._sub_booleans = []

        half_vols_inseted: set[str] = set()

        for sub_bool in sub_booleans:
            if isinstance(sub_bool, Or):
                # case where sub boolean is an Or, combine Or booleans into current,
                # skipping half volumes already added
                for nested in sub_bool.sub_booleans:
                    if isinstance(nested, HalfVolume):
                        if nested.__hash_string__() in half_vols_inseted:
                            continue
                        half_vols_inseted.add(nested.__hash_string__())
                    self._sub_booleans.append(nested)
            elif isinstance(sub_bool, And):
                # case where sub boolean is an And, add to the current
                self._sub_booleans.append(sub_bool)
            elif sub_bool.__hash_string__() not in half_vols_inseted:
                # cahse where a unique half volume is found, add to current
                self._sub_booleans.append(sub_bool)
                half_vols_inseted.add(sub_bool.__hash_string__())

    @property
    def mcnp_string(self) -> list[str]:
        """MCNP geometry tokens for the union, operands separated by ``:``.

        Sub-expressions that are lists are wrapped in parentheses.
        """
        sb_strings = [sb.mcnp_string for sb in self._sub_booleans]

        or_string = [':'] * (len(sb_strings) * 2 - 1)
        or_string[0::2] = sb_strings

        result = []

        for val in or_string:
            if isinstance(val, list):
                result.append('(')
                for subval in val:
                    result.append(subval)
                result.append(')')
            else:
                result.append(val)

        return result

    @property
    def openmc_region(self) -> openmc.Region:
        """OpenMC region formed by the union of all sub-boolean regions."""
        vol_openmc = self._sub_booleans[0].openmc_region

        for sub_bool in self._sub_booleans[1:]:
            vol_openmc = vol_openmc | sub_bool.openmc_region

        return vol_openmc

    def ocp_solid(self, bounding_box: BoundingBox | None=None,
                  decom_params: DecompositionParameters | None=None) -> TopoDS_Shape:
        """Build an Open CASCADE solid of the union.

        Parameters
        ----------
        bounding_box : BoundingBox, optional
            Bounds used to make half spaces finite, by default None (a
            +/-1000 box)
        decom_params : DecompositionParameters, optional
            Decomposition parameters for the OCP Boolean operations, by default
            None (default parameters)

        Returns
        -------
        TopoDS_Shape
            Solid representing the union.
        """
        # use default bounds
        bounding_box = bounding_box if bounding_box else BoundingBox(-1000, 1000, -1000, 1000, -1000, 1000)
        decom_params = decom_params if decom_params else DecompositionParameters()

        # get first volume in sub bools and ocp solid
        ocp_solid_vol = self._sub_booleans[0].ocp_solid(bounding_box, decom_params)

        # common all other sub bools to initial volume
        for sub_bool in self._sub_booleans[1:]:
            ocp_solid_vol = _fuse_ocp_solid(ocp_solid_vol, sub_bool.ocp_solid(bounding_box, decom_params),decom_params)

        # return volume
        return ocp_solid_vol

    def __str__(self) -> str:
        """Return an indented tree representation of the union."""
        string = 'or:\n'
        for sub_boolean in self._sub_booleans:
            string += sub_boolean.__str__()
        return f'{string}\n'.replace('\n', '\n ')

class HalfVolume():
    """One side (half space) of a surface.

    Usually created with the unary ``+``/``-`` operators on a surface.

    Parameters
    ----------
    direction : str
        ``'+'`` for the positive side of the surface, ``'-'`` for the negative.
    surface : Surface or Macrobody
        The bounding surface.
    """

    def __init__(self, direction: str, surface: Surface | Macrobody) -> None:
        """Initialise the HalfVolume."""
        self._direction: str = direction
        self._surface: Surface | Macrobody = surface
        self._volume: float | None = None

    def replace_duplicate_surfaces(self, surfaces: dict[str, Surface]) -> None:
        """Swap this half volume's surface for an identical stored one.

        Macrobodies are skipped. If no matching surface is stored, this
        surface is added to ``surfaces``.

        Parameters
        ----------
        surfaces : dict[str, Surface]
            Map of surface hash string to the surface instance to keep; updated
            in place.
        """
        # skip if a macrobody is found.
        if self._surface.__class__.__mro__[-2].__name__ == 'Macrobody':
            return

        surface_hash = self._surface.__hash_string__()
        if surface_hash in surfaces:
            logger.debug(f'surface already found...')
            logger.debug(f'{surface_hash}')
            logger.debug(f'{self._surface.type}')
            logger.debug(f'{self._surface}')
            logger.debug(f'{surfaces[surface_hash]}')
            self._surface = surfaces[surface_hash]
        else:
            surfaces[surface_hash] = self._surface

    @property
    def volume(self) -> float | None:
        """Volume value stored on the half volume."""
        return self._volume

    @volume.setter
    def volume(self, volume: float | None) -> None:
        """Set the stored volume value.

        Parameters
        ----------
        volume : float or None
            Volume value to store.
        """
        self._volume = volume

    @property
    def volume(self) -> float | None:
        """Volume value stored on the half volume (read only redefinition)."""
        return self._volume

    @property
    def surface(self) -> Surface | Macrobody:
        """The bounding surface."""
        return self._surface

    @property
    def surfaces(self) -> list[Surface | Macrobody]:
        """List containing the single bounding surface."""
        return [self._surface]

    @property
    def direction(self) -> str:
        """Side of the surface, ``'+'`` or ``'-'``."""
        return self._direction

    @property
    def openmc_region(self) -> openmc.Halfspace:
        """OpenMC half space for this side of the surface."""
        if self._direction == '+':
            return +self._surface.openmc_surface
        else:
            return -self._surface.openmc_surface

    def ocp_solid(self, bounding_box: BoundingBox | None=None,
                  decom_params: DecompositionParameters | None=None) -> TopoDS_Shape:
        """Build an Open CASCADE solid of the half space clipped to a box.

        Parameters
        ----------
        bounding_box : BoundingBox, optional
            Bounds used to make the half space finite, by default None (a
            +/-1000 box)
        decom_params : DecompositionParameters, optional
            Decomposition parameters for the OCP Boolean operations, by default
            None (default parameters)

        Returns
        -------
        TopoDS_Shape
            Solid representing the bounded half space.
        """
        # use default bounds
        bounding_box = bounding_box if bounding_box else BoundingBox(-1000, 1000, -1000, 1000, -1000, 1000)
        decom_params = decom_params if decom_params else DecompositionParameters()

        half_space_ocp_solid = _convert_mcio_half_volume_into_ocp_solid(self, bounding_box, decom_params)

        return half_space_ocp_solid

    def __hash_string__(self) -> str:
        """Return a string identifying the half volume independent of numbering.

        Built from the direction and the MCNP card of a copy of the surface
        with its number set to 0.

        Returns
        -------
        str
            Hash string for duplicate detection.
        """
        directions = {'+': '', '-': '-'}

        surface = copy.deepcopy(self._surface)
        surface.number = 0

        return f'{directions[self._direction]}_{surface.mcnp_card.as_string}'

    def __and__(self, other: HalfVolume | Volume) -> Volume:
        """Return the intersection of this half volume with another operand.

        Parameters
        ----------
        other : HalfVolume or Volume
            Second operand.

        Returns
        -------
        Volume
            Volume wrapping an :class:`And` Boolean.
        """
        boolean = And([self, other if isinstance(other, HalfVolume) else other.boolean])
        volume = Volume(boolean)
        return volume

    def __or__(self, other: HalfVolume | Volume) -> Volume:
        """Return the union of this half volume with another operand.

        Parameters
        ----------
        other : HalfVolume or Volume
            Second operand.

        Returns
        -------
        Volume
            Volume wrapping an :class:`Or` Boolean.
        """
        boolean = Or([self, other if isinstance(other, HalfVolume) else other.boolean])
        volume = Volume(boolean)
        return volume

    def __invert__(self) -> HalfVolume:
        """Return the half volume on the opposite side of the same surface.

        Returns
        -------
        HalfVolume
            Half volume with the direction flipped.
        """
        inverse_map = {'+': '-', '-': '+'}
        return HalfVolume(inverse_map[self._direction], self._surface)

    @property
    def mcnp_string(self) -> str:
        """MCNP token for the half volume (signed surface number)."""
        directions = {'+': '', '-': '-'}
        return f'{directions[self._direction]}{self._surface.number}'

    def __str__(self) -> str:
        """Return a short text representation of the half volume."""
        return f'half_vol: {self._direction}{self._surface}\n'


class Volume:
    """CSG volume defined by a Boolean expression.

    Parameters
    ----------
    boolean : Boolean
        Boolean expression (or half volume) defining the region.
    volume : float, optional
        Volume value associated with the region, by default None
    """

    def __init__(self, boolean: Boolean, volume: float | None=None) -> None:
        """Initialise the Volume."""
        self._boolean: Boolean = boolean
        self._volume: float | None = volume

    def replace_duplicate_surfaces(self, surfaces: dict[str, Surface]) -> None:
        """Replace duplicate surfaces in the Boolean with shared instances.

        Parameters
        ----------
        surfaces : dict[str, Surface]
            Map of surface hash string to the surface instance to keep; updated
            in place.
        """
        self._boolean.replace_duplicate_surfaces(surfaces)

    @property
    def volume(self) -> float | None:
        """Volume value associated with the region."""
        return self._volume

    @volume.setter
    def volume(self, volume: float | None) -> None:
        """Set the volume value.

        Parameters
        ----------
        volume : float or None
            Volume value to store.
        """
        self._volume = volume

    @property
    def boolean(self) -> Boolean:
        """Boolean expression defining the region."""
        return self._boolean

    def __and__(self, other: HalfVolume | Volume) -> Volume:
        """Return the intersection of this volume with another operand.

        Parameters
        ----------
        other : HalfVolume or Volume
            Second operand.

        Returns
        -------
        Volume
            Volume wrapping an :class:`And` Boolean.
        """
        boolean = And([self.boolean, other if isinstance(other, HalfVolume) else other.boolean])
        volume = Volume(boolean)
        return volume

    def __or__(self, other: HalfVolume | Volume) -> Volume:
        """Return the union of this volume with another operand.

        Parameters
        ----------
        other : HalfVolume or Volume
            Second operand.

        Returns
        -------
        Volume
            Volume wrapping an :class:`Or` Boolean.
        """
        boolean = Or([self.boolean, other if isinstance(other, HalfVolume) else other.boolean])
        volume = Volume(boolean)
        return volume

    def __invert__(self) -> Volume:
        """Return the complement of this volume.

        Returns
        -------
        Volume
            Volume wrapping the inverted Boolean.
        """
        return Volume(~self._boolean)

    @property
    def mcnp_string(self) -> list[str] | str:
        """MCNP geometry tokens for the volume."""
        return self._boolean.mcnp_string

    @property
    def openmc_region(self) -> openmc.Region:
        """OpenMC region for the volume."""
        return self._boolean.openmc_region

    def ocp_solid(self, bounding_box: BoundingBox | None=None,
                  decom_params: DecompositionParameters | None=None) -> TopoDS_Shape:
        """Build an Open CASCADE solid of the volume.

        Parameters
        ----------
        bounding_box : BoundingBox, optional
            Bounds used to make half spaces finite, by default None (a
            +/-1000 box)
        decom_params : DecompositionParameters, optional
            Decomposition parameters for the OCP Boolean operations, by default
            None (default parameters)

        Returns
        -------
        TopoDS_Shape
            Solid representing the volume.
        """
        # use default bounds
        bounding_box = bounding_box if bounding_box else BoundingBox(-1000, 1000, -1000, 1000, -1000, 1000)
        decom_params = decom_params if decom_params else DecompositionParameters()

        return self._boolean.ocp_solid(bounding_box, decom_params)

    @property
    def surfaces(self) -> list[Surface | Macrobody]:
        """All surfaces referenced by the volume (may repeat)."""
        return self._boolean.surfaces

    def __str__(self) -> str:
        """Return an indented tree representation of the volume."""
        return f'vol:\n{self._boolean}'.replace('\n', '\n ')
