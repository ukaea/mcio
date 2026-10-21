"""Cell definition for MCIO geometry models.

Contains the :class:`Cell` class, which combines a CSG volume with a material
and optional universe/transform information, and exports to MCNP and OpenMC.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import openmc
except ModuleNotFoundError:
    pass

try:
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
except:
    pass

from .formatting import formatting
from .geometric import BoundingBox
from .decomposeparams import DecompositionParameters
from .material import Material
from .mcnp import MCNPCard, MCNPCellParameters, MCNP_PARTICLE_DESIGNATORS
from .transform import Transform
from .universe import Universe, UniverseFill
from .volume import Volume

if TYPE_CHECKING:
    from OCP.TopoDS import TopoDS_Shape
    from .surface import Surface


class Cell:
    """Cell representation for geometry modeling and export to MCNP and OpenMC."""

    def __init__(
        self,
        volume: Volume = None,
        material: Material = None,
        name: str = None,
        comment: str = None,
        transform: Transform = None,
        universe: Universe = None,
        universe_fill: UniverseFill = None,
        graveyard: bool = False,
        mcnp_cell_parameters: MCNPCellParameters = None,
    ) -> None:
        """Cell object containing geometric and material definition.

        Parameters
        ----------
        volume : Volume, optional
            Geometric volume of the cell, by default None
        material : Material, optional
            Material assigned to the cell, by default None
        name : str, optional
            Name of the cell, by default None
        comment : str, optional
            Comment associated with the cell, by default None
        transform : Transform, optional
            Geometric transformation applied to the cell, by default None
        universe : Universe, optional
            Universe in which the cell exists, by default None
        universe_fill : UniverseFill, optional
            Universe filling this cell, by default None
        graveyard : bool, optional
            Flag indicating if the cell is a graveyard cell, by default False
        mcnp_cell_parameters : MCNPCellParameters, optional
            Specific MCNP cell parameters, by default None
        """
        self.volume: Volume = volume
        self.material: Material = material
        self.number: int = None
        self.comment: str = comment
        self.name: str = name
        self.transform: Transform = transform
        self.universe: Universe = universe
        self.universe_fill: UniverseFill = universe_fill
        self.graveyard: bool = graveyard
        self._run_parameters: dict = {"mcnp": None}
        self.mcnp: MCNPCellParameters = (
            MCNPCellParameters(graveyard=graveyard)
            if not mcnp_cell_parameters
            else mcnp_cell_parameters
        )

        # store generated properties
        self._mcnp_card: MCNPCard = None
        self._openmc_cell: "openmc.cell.Cell" = None

    @property
    def material(self) -> Material:
        """Material assigned to the cell."""
        return self._material

    @material.setter
    def material(self, material: Material) -> None:
        """Set the material assigned to the cell.

        Parameters
        ----------
        material : Material
            Material to assign, or None for a void cell.

        Raises
        ------
        TypeError
            If ``material`` is a class (type) rather than an instance.
        """
        if isinstance(material, type(Material)):
            msg = f"cell.material() requires {type(Material)} not {type(material)}"
            raise TypeError(msg)
        else:
            self._material = material

    @property
    def surfaces(self) -> list[Surface] | None:
        """Surfaces bounding the cell volume, or None if no volume is set."""
        return self.volume.surfaces if self.volume else None

    def ocp_solid(
        self, bounding_box: BoundingBox, decom_params: DecompositionParameters
    ) -> TopoDS_Shape:
        """Generate Open CASCADE CAD solid for the cell volume.

        Parameters
        ----------
        bounding_box : BoundingBox
            Bounding box constraint for the solid generation
        decom_params : DecompositionParameters
            Decomposition parameters for CAD construction

        Returns
        -------
        TopoDS_Shape
            Open CASCADE shape representing the transformed cell solid
        """
        solid_from_vol = self.volume.ocp_solid(bounding_box, decom_params)

        if self.transform:
            solid_from_vol = BRepBuilderAPI_Transform(
                solid_from_vol, self.transform.ocp_transform, True
            ).Shape()

        return solid_from_vol

    @property
    def mcnp_card(self) -> MCNPCard:
        """Generate the corresponding MCNP cell card definition.

        Returns
        -------
        MCNPCard
            MCNP cell card object
        """
        # set up string for boolean expression
        boolean_strings = self.volume.mcnp_string

        density_fraction_modifier = {"g/cc": -1, "g/cm3": -1, "atoms/barn-cm": 1}

        # case where single surface used (only half volume)
        if not isinstance(boolean_strings, list):
            boolean_strings = [boolean_strings]

        if self._material is not None:
            density = [
                f"{self.material.number}",
                f"{density_fraction_modifier[self._material.density_unit]*self._material.density:.4e}",
            ]
        else:
            density = ["0"]

        header_data = {"name": self.name, "comment": self.comment}

        # write cell importances
        importances = [
            f"IMP:{MCNP_PARTICLE_DESIGNATORS[p.lower()].upper()}={i}"
            for p, i in self.mcnp.importances.items()
        ]

        # set up rows for MCNPCard
        rows = [density, boolean_strings, importances]

        # write universe
        if self.universe:
            rows.append([f"U={self.universe.number}"])

        # write universe fill
        if self.universe_fill:
            transforms_as_bracketed_string = (
                f"({self.universe_fill.transform.number})"
                if self.universe_fill.transform
                else ""
            )
            universe_fill_str = f"FILL={self.universe_fill.universe.number} {transforms_as_bracketed_string}"
            rows.append([universe_fill_str])

        # check if volume manually assigned
        if self.mcnp.volume:
            rows.append([f"VOL={self.mcnp.volume:.{formatting.MCNP_DP}e}"])

        # assign transform
        if self.transform:
            rows.append([f"TRCL={self.transform.number}"])

        mcnp_card = MCNPCard(
            self.number,
            rows,
            header_data=header_data if (self.comment or self.name) else None,
        )

        return mcnp_card

    @property
    def openmc_cell(self) -> "openmc.cell.Cell":
        """Generate or retrieve the corresponding OpenMC cell object.

        Returns
        -------
        openmc.cell.Cell
            Constructed OpenMC cell
        """
        if self._openmc_cell:
            return self._openmc_cell

        cell_openmc = openmc.cell.Cell(
            region=self.volume.openmc_region if self.volume else None,
            fill=self.material.openmc_material if self.material else None,
        )

        # check if this cell is a graveyard, if so set all surfaces within it
        # to 'vacuum' and kill particles entering.
        if self.graveyard:
            if cell_openmc.region:
                surfaces_dict = cell_openmc.region.get_surfaces()

                for surface in surfaces_dict.values():
                    surface.boundary_type = "vacuum"

        self._openmc_cell = cell_openmc

        return self._openmc_cell
