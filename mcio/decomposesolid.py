"""Module contains the decomposition solid objects which encapsulate a single
solid during the decomposition of a full cad model.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os

from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm.notebook import tqdm as tqdm_notebook

try:
    import casadi
    import OCP
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.TopAbs import TopAbs_REVERSED, TopAbs_FACE
    from OCP.TopoDS import TopoDS_Solid, TopoDS_Shape
except Exception as e:
    from mcio import logger

    logger.warning("failed to load OCP")
    logger.warning(f"{e}")

from mcio import logger
from .colour import ColourRGB
from .decomposeauxiliaries import generate_auxiliary_surfaces_from_solids
from .decomposeparams import DecompositionParameters
from .decomposesplitsolids import SplitSolid
from .decomposevoid import VoidOctreeFaces, VoidOctreeSplitSolids
from .surface import _map_of_duplicate_surfaces
from .source import SphericalStochasticVolumeSource
from .geometric import BoundingBox
from .tally import VolumeFluenceTally
from .profile import function_timer
from .cell import Cell
from .model import Model

from .ocp import (
    convert_ocp_shape_to_binary,
    convert_shape_binary_to_ocp_shape,
    _split_solids_on_axis_by_circular_edges,
    _split_solid_on_connection_between_equal_radii_cylinders,
)
from .ocpboundingbox import ocp_box_from_bounding_box, bounding_box_from_shape
from .ocpbooleans import _common_ocp_solid, _recursively_split_ocp_solid
from .ocpfilereadwrite import (
    extract_ocp_shape_type_from_shape,
    _get_ocp_surface_from_ocp_face,
    STEPFileWriter,
)
from .ocprepair import _unify_same_domains, _shape_fix_solid, _fix_inexact_edges
from .ocpproperties import (
    ocp_face_params_as_string,
    _volume_of_ocp_solids,
)
from .utils import in_notebook
from .universe import Universe, UniverseFill


@function_timer
def generate_mcio_void_octree(
    decomposition_solids: List["DecompositionSolid"],
    surfaces_generated: Dict[Any, Any],
    decom_params: DecompositionParameters,
    universe: Optional[Universe] = None,
    run: bool = True,
    show_progress: bool = True,
) -> Union[VoidOctreeFaces, VoidOctreeSplitSolids]:
    """Generate a void octree structure for MCIO geometry decomposition.

    Parameters
    ----------
    decomposition_solids : list of DecompositionSolid
        List of target decomposition solids to generate void space around.
    surfaces_generated : dict
        Mapping of existing generated surfaces to prevent duplicate creation.
    decom_params : DecompositionParameters
        Configuration parameters controlling decomposition and void behavior.
    universe : Universe, optional
        Target MCIO Universe assignment for the generated void space, by default None.
    run : bool, optional
        Whether to execute the octree generation immediately upon instantiation,
        by default True.
    show_progress : bool, optional
        Whether to render execution progress bars during generation, by default True.

    Returns
    -------
    VoidOctreeFaces or VoidOctreeSplitSolids
        Instantiated void octree object filtered by face or split solid depending
        on parameters.
    """
    # select either filter by face or filter by solids method
    if decom_params.void_filter_by_face:
        # collect all faces from each split solid within each
        # decomposition_solid, grouping by solid
        split_solids_with_faces: Dict[SplitSolid, List[Any]] = {}

        # iterate over all split solids and create dictionaries which can be
        # filtered by intersecting solids/faces
        for decomposition_solid in decomposition_solids:
            for split_solid in decomposition_solid.split_solids:
                split_solids_with_faces[split_solid] = list(
                    split_solid.face_to_mcio_halfspaces
                )

        void_octree = VoidOctreeFaces(
            split_solids_with_faces,
            decom_params,
            surfaces_generated,
            universe=universe,
            show_progress=show_progress,
        )

    else:
        # run alternative method 1 : sort by split solids
        split_solids: List[SplitSolid] = []
        for decomposition_solid in decomposition_solids:
            split_solids.extend(decomposition_solid.split_solids)

        void_octree = VoidOctreeSplitSolids(
            split_solids,
            decom_params,
            surfaces_generated,
            universe=universe,
            show_progress=show_progress,
        )

    # run the void octree generation (single core)
    if run:
        void_octree.run()

    # return void octree
    return void_octree


@function_timer
def _generate_mcio_cells(
    decomposition_solids: List["DecompositionSolid"],
    decom_params: DecompositionParameters,
) -> List[Cell]:
    """Convert a collection of decomposition solids into MCIO Cell objects.

    Parameters
    ----------
    decomposition_solids : list of DecompositionSolid
        List of target decomposition solid instances.
    decom_params : DecompositionParameters
        Decomposition settings specifying metadata and cell output flags.

    Returns
    -------
    list of Cell
        List of constructed MCIO Cell objects containing geometry and metadata.
    """
    # generate mcio cells from each decomposition solids
    cells: List[Cell] = []
    n_solids: int = len(decomposition_solids)

    desc_stage: str = "[Generating mcio cells]"
    is_notebook_env: bool = in_notebook()

    if is_notebook_env:
        pbar = tqdm_notebook(total=n_solids, desc=desc_stage, unit="cell")

        def update_progress() -> None:
            """Advance the notebook progress bar by one cell."""
            pbar.update(1)

        def close_progress() -> None:
            """Close the notebook progress bar."""
            pbar.close()

    else:
        desc: str = f"[cyan]{desc_stage: <30}[/cyan]"
        progress = Progress(
            TextColumn(desc),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        progress.start()
        task_id = progress.add_task("generate_cells", total=n_solids)

        def update_progress() -> None:
            """Advance the rich progress bar by one cell."""
            progress.advance(task_id)

        def close_progress() -> None:
            """Stop the rich progress bar."""
            progress.stop()

    for i, decomposition_solid in enumerate(decomposition_solids, start=1):
        # set up cell and assign name & comment for cell number and structure
        cell: Cell = decomposition_solid.mcio_cell

        # add name and comments to cell
        bbox: BoundingBox = decomposition_solid._solid_original_bounding_box
        location: str = (
            f"location {bbox.x_mid:.2f} {bbox.y_mid:.2f} "
            + f"{bbox.z_mid:.2f} ext "
            + f"{max(bbox.x_length, bbox.y_length, bbox.z_length):.2f}"
        )
        cell.name = f"cell number {i} of {n_solids} : {location}"
        cell.comment = "/".join(decomposition_solid._structure_original or [])

        # if user requires, set volume on each cell
        if decom_params.model_set_output_cell_vols:
            cell.mcnp.volume = decomposition_solid._volume_original_solid

        # add cell to list
        cells.append(cell)

        update_progress()

    close_progress()

    # return list of cells
    return cells


class DecompositionSolid:
    """Manage geometric decomposition, auxiliary surface generation, and cell synthesis
    for an individual OpenCASCADE TopoDS_Solid.

    Parameters
    ----------
    solid_original : TopoDS_Solid or bytes
        Original OpenCASCADE shape solid or its serialized binary byte representation.
    decom_params : DecompositionParameters
        Configuration options controlling splitting routines and tolerances.
    structure_original : list of str, optional
        Original CAD hierarchy structural path metadata strings, by default None.
    colour_rgb : ColourRGB or tuple of int, optional
        RGB color tuple assigned to the original shape, by default None.
    material : Any, optional
        Assigned material definition for cell synthesis, by default None.
    universe : Universe, optional
        Parent universe assignment, by default None.
    universe_fill : UniverseFill, optional
        Universe fill specification, by default None.
    serialise : bool, optional
        Whether to immediately convert the CAD shape to a binary string, by
        default False.
    surfaces_generated : dict, optional
        Mapping tracking unique generated surfaces to prevent redundancy, by
        default None.

    Attributes
    ----------
    decom_params : DecompositionParameters
        Stored parameters used throughout execution.
    material : Any or None
        Assigned material object for the generated cell.
    reserialise_required : bool
        Flag indicating if the object needs to be re-serialized after execution.
    """

    def __init__(
        self,
        solid_original: Union["TopoDS_Solid", bytes],
        decom_params: DecompositionParameters,
        structure_original: Optional[List[str]] = None,
        colour_rgb: Optional[Union[ColourRGB, Tuple[int, int, int]]] = None,
        material: Optional[Any] = None,
        universe: Optional[Universe] = None,
        universe_fill: Optional[UniverseFill] = None,
        serialise: bool = False,
        surfaces_generated: Optional[Dict[Any, Any]] = None,
    ) -> None:
        """Initialise the DecompositionSolid, applying any requested shape cleanup."""
        # set input variables
        self._solid_original: Union["TopoDS_Solid", bytes] = solid_original
        self._solid_original_bounding_box: BoundingBox = bounding_box_from_shape(
            self._solid_original
        )
        self._structure_original: Optional[List[str]] = structure_original
        self._colour_rgb: Optional[ColourRGB] = (
            ColourRGB(*colour_rgb) if colour_rgb else None
        )
        self.material: Optional[Any] = material
        self._universe: Optional[Universe] = universe
        self._universe_fill: Optional[UniverseFill] = universe_fill
        self.decom_params: DecompositionParameters = decom_params

        # unify duplicated faces
        if self.decom_params.cleanup_unify_same_domains:
            self._solid_original = _unify_same_domains(self._solid_original)

        if self.decom_params.cleanup_shape_fix:
            self._solid_original = _shape_fix_solid(self._solid_original)

        if self.decom_params.cleanup_inexact_edges:
            step_file_writer = STEPFileWriter(self.decom_params)
            step_file_writer.add_solids([self._solid_original], "origingal_solid")
            self._solid_original = _fix_inexact_edges(
                self._solid_original, self.decom_params
            )
            step_file_writer.add_solids([self._solid_original], "origingal_solid_fixed")
            step_file_writer.write("trials.stp")

        # run and formatting parameters
        self._params: DecompositionParameters = DecompositionParameters()

        # profiling information
        self._profile_time_decomposition: Optional[float] = None
        self._profile_time_void_octree: Optional[float] = None
        self._profile_time_generate_mcio_cells: Optional[float] = None

        # set up serialisation variables
        self._is_serialised: bool = False
        self.reserialise_required: bool = False

        # set up variables to store decomposed solids and auxiliary surfaces
        self._split_solids: List[SplitSolid] = []

        # to store mcio cell and void octree
        self._cell: Optional[Cell] = None
        self._void_octree: Optional[Union[VoidOctreeFaces, VoidOctreeSplitSolids]] = (
            None
        )

        # set up variables for quality assurance
        self._volume_original_solid: Optional[float] = None
        self._volume_decomposed_solids: Optional[float] = None

        # store OCP surface to MCIO surface to reduce repetition of surfaces
        self._surfaces_generated: Dict[Any, Any] = (
            surfaces_generated if surfaces_generated else _map_of_duplicate_surfaces()
        )

        # model
        self._profile_time_model: Optional[float] = None
        self._model: Optional[Model] = None

        # serialise if required
        if serialise:
            self.serialise()

    @property
    def volume(self) -> Optional[float]:
        """float or None: Total volume of the original shape solid."""
        return self._volume_original_solid

    @property
    def is_serialised(self) -> bool:
        """bool: True if the OpenCASCADE shape is currently serialised to binary."""
        return self._is_serialised

    @property
    def profile_time_decomposition(self) -> Optional[float]:
        """float or None: Total time spent in seconds performing solid decomposition."""
        return self._profile_time_decomposition

    @property
    def volume_original_solid(self) -> Optional[float]:
        """float or None: Volume of the target solid prior to decomposition."""
        return self._volume_original_solid

    @property
    def volume_decomposed_solids(self) -> Optional[float]:
        """float or None: Sum of volumes across all generated decomposed sub-solids."""
        return self._volume_decomposed_solids

    @property
    def universe(self) -> Optional[Universe]:
        """Universe or None: Assigned MCIO Universe."""
        return self._universe

    @universe.setter
    def universe(self, universe: Optional[Universe]) -> None:
        """Set the universe and propagate it to the generated cell, if any.

        Parameters
        ----------
        universe : Universe or None
            Universe to assign.
        """
        self._universe = universe
        if self._cell:
            self._cell.universe = universe

    @property
    def universe_fill(self) -> Optional[UniverseFill]:
        """UniverseFill or None: Assigned MCIO UniverseFill."""
        return self._universe_fill

    @universe_fill.setter
    def universe_fill(self, universe_fill: Optional[UniverseFill]) -> None:
        """Set the universe fill and propagate it to the generated cell, if any.

        Parameters
        ----------
        universe_fill : UniverseFill or None
            Universe fill to assign.
        """
        self._universe_fill = universe_fill
        if self._cell:
            self._cell.universe_fill = self.universe_fill

    @property
    def void_octree(self) -> Union[VoidOctreeFaces, VoidOctreeSplitSolids]:
        """VoidOctreeFaces or VoidOctreeSplitSolids: Void octree for the solid.

        Lazily built (and run) on first access.
        """
        # if not generated, set up parent at top level
        if not self._void_octree:
            # generate parent at top level to allow user to manually control
            # void run via multiprocess
            self._void_octree, self._profile_time_void_octree = (
                generate_mcio_void_octree(
                    [self], self._surfaces_generated, self.decom_params, run=True
                )
            )

        return self._void_octree

    def serialise(self) -> None:
        """Convert the OpenCASCADE TopoDS_Solid and sub-solids into binary byte strings.

        Notes
        -----
        This enables serialization for multiprocessing or pickling across processes.
        """
        # case where already serialised
        if self._is_serialised:
            return

        # convert the original solid to binary
        self._solid_original = convert_ocp_shape_to_binary(self._solid_original)

        # serialise split solids
        for split_solid in self._split_solids:
            split_solid.serialise()

        # set serialised flag
        self._is_serialised = True

    def deserialise(self) -> None:
        """Convert binary byte strings back to native OpenCASCADE shape objects.

        Notes
        -----
        Reverses the operations performed by `serialise()`.
        """
        # case where already deserialised
        if not self._is_serialised:
            return

        # convert the original solid to binary
        self._solid_original = convert_shape_binary_to_ocp_shape(self._solid_original)

        for split_solid in self._split_solids:
            split_solid.deserialise()

        # set serialised flag
        self._is_serialised = False

    @property
    def original_solid(self) -> Union["TopoDS_Solid", bytes]:
        """TopoDS_Solid or bytes: Original shape object or raw binary string."""
        return self._solid_original

    @property
    def bounding_box(self) -> BoundingBox:
        """BoundingBox: Bounding box surrounding the original shape solid."""
        return self._solid_original_bounding_box

    @property
    def split_solids(self) -> List[SplitSolid]:
        """list of SplitSolid: Decomposed sub-solids created during decomposition."""
        return self._split_solids

    @property
    def mcio_cell(self) -> Optional[Cell]:
        """Cell or None: MCIO Cell representation corresponding to this solid."""
        return self._cell

    @property
    def structure_in_input_file(self) -> Optional[List[str]]:
        """list of str or None: CAD structure path metadata from the STEP file."""
        return self._structure_original

    @property
    def rgb(self) -> Optional[ColourRGB]:
        """ColourRGB or None: RGB color assigned to this solid."""
        return self._colour_rgb

    def save(
        self, output_path: Union[str, os.PathLike] = "analysis/solids/solid_cell_1.stp"
    ) -> str:
        """Save the original solid, decomposed sub-solids and auxiliary surfaces to STEP.

        Parameters
        ----------
        output_path : str or os.PathLike, optional
            Target output file path, by default "analysis/solids/solid_cell_1.stp".

        Returns
        -------
        str
            Resolved path to the exported STEP file.
        """
        # set up step file writer
        step_file_writer = STEPFileWriter(decom_params=self.decom_params)

        # add the original solid
        step_file_writer.add_solid(self._solid_original, "original_solid")

        # add the split solid(s)
        step_file_writer.add_solids(
            [ss.split_solid for ss in self._split_solids], "split_solids"
        )

        # add the split solid(s) and their auxiliary surface(s)
        for split_solid in self._split_solids:
            auxiliary_surfaces_to_write: List[Any] = []
            for face, auxiliary_surfaces in split_solid.faces_with_auxiliaries.items():
                if auxiliary_surfaces:
                    # add face to list for writing to step file
                    auxiliary_surfaces_to_write.append(face)

                    # create faces for aux surfaces, using nominal uv params
                    faces = [
                        [
                            BRepBuilderAPI_MakeFace(
                                auxiliary_surface,
                                0.0,
                                1.0,
                                0.0,
                                1.0,
                                self.decom_params.make_face_tol,
                            ).Face(),
                            orientation,
                        ]
                        for [auxiliary_surface, orientation] in auxiliary_surfaces
                    ]

                    # create a bounding box the the solid
                    bounding_boxes = [
                        bounding_box_from_shape(split_solid.split_solid)
                        * self.decom_params.aux_bounding_box_analysis
                        for _ in auxiliary_surfaces
                    ]

                    # create an rpp solid of the bounding box
                    boxes = [
                        ocp_box_from_bounding_box(bounding_box)
                        for bounding_box in bounding_boxes
                    ]

                    # take common space between box and auxiliary surface using
                    # it's infinite half space - require a single solid to
                    # be produced
                    boxes_common_with_auxiliary = [
                        _common_ocp_solid(
                            box,
                            face,
                            self.decom_params,
                            min_solids_required=1,
                            reverse_face_orientation=orientation == TopAbs_REVERSED,
                        ).solids[0]
                        for box, (face, orientation) in zip(boxes, faces)
                    ]

                    # add compound to return dictionary
                    auxiliary_surfaces_to_write.extend(boxes_common_with_auxiliary)

            step_file_writer.add_solids(
                auxiliary_surfaces_to_write, "auxiliary_surfaces"
            )

        # write step file
        written_path: str = step_file_writer.write(output_path)

        # return path to saved step file
        return written_path

    @function_timer
    def _decompose(self) -> "DecompositionSolid":
        """Execute geometric splitting algorithms to decompose non-convex solids.

        Returns
        -------
        DecompositionSolid
            Self reference for pipeline chaining or multiprocessing return contexts.
        """
        # output to logger for debugging
        logger.debug("running _decompose")

        # check if serialised
        if self._is_serialised:
            self.deserialise()

        # debugging
        logger.debug("decompose() - starting with surfaces...")
        if logger.getEffectiveLevel() <= 10:
            faces = extract_ocp_shape_type_from_shape(
                self._solid_original, TopAbs_FACE, cast_from_shape=True
            )
            for i, face in enumerate(faces):
                surface = _get_ocp_surface_from_ocp_face(face)
                logger.debug(f"face {i} type is {type(surface)}")
                logger.debug(ocp_face_params_as_string(face))

        # pre-split
        if self.decom_params.presplit_concentril_2nd_order:
            logger.debug("pre-splitting solids - concentric (2nd order faces on axis)")
            pre_split_solids_concentrics = _split_solids_on_axis_by_circular_edges(
                self._solid_original, self.decom_params
            )

            # debugging - save pre-splitting on connected concentric 2nd orders
            if logger.getEffectiveLevel() <= 10:
                step_file_writer = STEPFileWriter(self.decom_params)
                step_file_writer.add_solids([self._solid_original], "original_solid")
                step_file_writer.add_solids(
                    pre_split_solids_concentrics, "split_solids"
                )
                output_path = step_file_writer.write(
                    self.decom_params.debug_concentric_splitter, enumerate_files=True
                )
                logger.debug(f"pre-splitter - concentric : see output {output_path}")

            solids_created = len(pre_split_solids_concentrics) - 1
            logger.debug(
                f"pre-splitting solids - concentril - created {solids_created} pre-split solids"
            )
        else:
            pre_split_solids_concentrics = [self._solid_original]

        # pre-splitting - split pipes (connected cylinders with equal radii
        # by elipse or cicle)
        if self.decom_params.presplit_pipes:
            logger.debug("pre-splitting solids - pipe splitter")
            pre_split_solids_pipes: List[TopoDS_Solid] = []
            for pre_split_solid_pipes in pre_split_solids_concentrics:
                pre_split_solids_pipes.extend(
                    _split_solid_on_connection_between_equal_radii_cylinders(
                        pre_split_solid_pipes, self.decom_params
                    )
                )

            logger.debug("pre-splitting solids - pipe splitter -  complete")
        else:
            pre_split_solids_pipes = pre_split_solids_concentrics

        # debugging - save pre-splitting on pipes
        if logger.getEffectiveLevel() <= 10:
            step_file_writer = STEPFileWriter(self.decom_params)
            step_file_writer.add_solids([self._solid_original], "original_solid")
            step_file_writer.add_solids(pre_split_solids_pipes, "split_solids")
            output_path = step_file_writer.write(
                self.decom_params.debug_pipe_splitter, enumerate_files=True
            )
            logger.debug(f"pre-splitter - pipe : see output {output_path}")

        # decompose split solids recursively
        logger.debug("splitting recursively")
        split_solids: List[TopoDS_Solid] = []
        for pre_split_solid in pre_split_solids_pipes:
            split_solids.extend(
                _recursively_split_ocp_solid(pre_split_solid, self.decom_params)
            )

        logger.debug("splitting recursively complete")

        # if no new solids are created, use original solid - avoiding thin degenerate slithers being removed reducing volume loss and possible degenreate solids
        if len(split_solids) == 1:
            split_solids = [self._solid_original]

        # generate auxiliary surfacess, and attach surfaces to each split solid in dictionary
        logger.debug("generating auxiliary surface")
        split_solids_with_auxiliaries = generate_auxiliary_surfaces_from_solids(
            split_solids, self.decom_params
        )

        # convert the raw ocp solids to SplitSolid instances to group together
        # required data
        self._split_solids = [
            SplitSolid(
                split_solid,
                faces_with_auxiliries,
                self.decom_params,
                surfaces_generated=self._surfaces_generated,
            )
            for split_solid, faces_with_auxiliries in split_solids_with_auxiliaries.items()
        ]

        # calculate volumes of original vs split
        self._volume_original_solid = _volume_of_ocp_solids([self._solid_original])
        self._volume_decomposed_solids = _volume_of_ocp_solids(
            [split_solid.split_solid for split_solid in self._split_solids]
        )

        # combine volumes into single volumes and create cell
        volume_combined = self._split_solids[0].volume
        for split_solid in self._split_solids[1:]:
            volume_combined = volume_combined | split_solid.volume

        # create mcio cell
        self._cell = Cell(
            volume=volume_combined,
            material=self.material,
            universe=self.universe,
            universe_fill=self.universe_fill,
        )

        # return self - this is required when using multiprocessing to pass
        # back to master
        return self

    @function_timer
    def _generate_mcio_model(self) -> Model:
        """Construct an integrated MCIO Model combining geometry, sources, and tallies.

        Returns
        -------
        Model
            Completed MCIO model instance.
        """
        # gather cells from all decomposition_solids
        cells, self._profile_time_generate_mcio_cells = _generate_mcio_cells(
            [self], self.decom_params
        )

        # generate void cells
        cells_voids = self._void_octree.mcio_cells

        # add stochastic volume source
        source = SphericalStochasticVolumeSource(
            spherical_surface=self._void_octree.surf_stochastic_volume_sphere
        )

        # add tallies for stochastic volume check
        tallies = [
            VolumeFluenceTally(cells=cells, comment="stochastic volume calculation")
        ]

        # summary data
        summary_data = {"void octants": len(cells_voids)}

        # generate model
        return Model(
            cells=[*cells, *cells_voids],
            source=source,
            tallies=tallies,
            summary_data=summary_data,
        )

    @property
    def model(self) -> Model:
        """Model: MCIO simulation model associated with this solid."""
        if not self._void_octree:
            _ = self.void_octree

        if not self._model:
            self._model, self._profile_time_model = self._generate_mcio_model()

        return self._model

    def run(self) -> None:
        """Execute the full geometric decomposition pipeline for this solid."""
        # check if serialised
        if self._is_serialised:
            self.deserialise()
            self.reserialise_required = True
        else:
            self.reserialise_required = False

        # decompose original solid
        _, self._profile_time_decomposition = self._decompose()

        # re-serialise if required
        if self.reserialise_required:
            self.serialise()
