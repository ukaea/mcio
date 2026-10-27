"""Top level decomposition of STEP CAD models into MCIO CSG models.

Contains the :class:`Decomposer`, which loads solids from STEP files, decomposes
each into CSG cells (optionally in parallel), generates the void octree and
assembles an MCIO :class:`~mcio.model.Model`. Also provides the :class:`Analysis`
class for summarising the bodies and surface types of a model, and some small
OCP helper functions.
"""

from __future__ import annotations

import math
import json
import os
import time
import pickle
import pathlib
from concurrent.futures import ProcessPoolExecutor
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    track,
)
from tqdm.notebook import tqdm as tqdm_notebook
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)


from mcio import logger, LOGGER_LEVELS

try:
    import casadi
    import OCP
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.Geom import (
        Geom_Surface,
        Geom_Plane,
        Geom_CylindricalSurface,
        Geom_SphericalSurface,
        Geom_ConicalSurface,
        Geom_ToroidalSurface,
    )
    from OCP.Quantity import Quantity_ColorRGBA, Quantity_TOC_sRGB
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import (
        TopoDS_Shape,
        TopoDS_Solid,
    )
    from OCP.XCAFDoc import XCAFDoc_ColorSurf
except Exception as e:
    from mcio import logger

    logger.warning(f"failed to load OCP")
    logger.warning(f"{e}")


from .colour import ColourRGB
from .decomposesolid import (
    DecompositionSolid,
    _generate_mcio_cells,
    generate_mcio_void_octree,
)
from .decomposeparams import DecompositionParameters
from .decomposemultiprocess import _run_decomposer_solid
from .decomposevoid import VoidOctreeFaces
from .surface import _map_of_duplicate_surfaces
from .source import SphericalStochasticVolumeSource
from .geometric import BoundingBox
from .material import Material, _materials_from_json, _material_lookup
from .tally import VolumeFluenceTally, Tally
from .tally import VolumeFluenceTally
from .table import Table
from .parallellogging import (
    start_multiprocess_logging,
    configure_worker_logger,
    restore_main_process_logging,
)
from .profile import function_timer
from .model import Model
from .ocp import (
    _get_ocp_surface_from_ocp_face,
    extract_ocp_shape_type_from_shape,
    count_surface_by_type,
)
from .ocpfilereadwrite import STEPFileReader, STEPFileWriter
from .utils import in_notebook, check_folder_exists_for_file_path
from .universe import Universe

if TYPE_CHECKING:
    from OCP.TDF import TDF_Label
    from OCP.XCAFDoc import XCAFDoc_ColorTool


class Analysis:
    """Analysis class to measure key properties of a model, such as the number
    of bodies, faces and surface types, and the validity of the CAD.

    Parameters
    ----------
    decomposition_solids : Sequence[DecompositionSolid]
        The decomposition solids containing the CAD solids to analyse.

    Raises
    ------
    ValueError
        If a face has a surface type other than plane, cylinder, cone, sphere
        or torus.

    Attributes
    ----------
    NUMBER_OF_BODIES : int
        Number of decomposition solids analysed.
    NUMBER_OF_FACES : int
        Total number of faces across all solids.
    FACES_PER_BODY : dict[DecompositionSolid, int]
        Number of faces of each decomposition solid.
    FACE_TYPE_COUNT_PER_BODY : dict[DecompositionSolid, dict[type, int]]
        Count of faces by OCP surface type for each decomposition solid.
    FACE_TYPE_COUNT : dict[type, int]
        Count of faces by OCP surface type across all solids.
    DEGENERATE_TORI_COUNT : int
        Number of tori whose minor radius exceeds their major radius.
    CAD_VALID : dict[DecompositionSolid, bool]
        Result of the OCP ``BRepCheck_Analyzer`` validity check for each solid.
    """
    def __init__(self, decomposition_solids: Sequence[DecompositionSolid]) -> None:
        """Initialise the Analysis and analyse the solids and surface types."""
        self.NUMBER_OF_BODIES: int = len(decomposition_solids)
        self.NUMBER_OF_FACES: int = 0
        self.FACES_PER_BODY: dict[DecompositionSolid, int] = {
            decomposition_solid: 0 for decomposition_solid in decomposition_solids
        }

        self.FACE_TYPE_COUNT_PER_BODY: dict[DecompositionSolid, dict[Type, int]] = {
            decomposition_solid: self.gen_face_type_dict()
            for decomposition_solid in decomposition_solids
        }
        self.FACE_TYPE_COUNT: dict[Type, int] = self.gen_face_type_dict()

        self.DEGENERATE_TORI_COUNT: int = 0

        self.CAD_VALID: dict[DecompositionSolid, bool] = {}

        for decomposition_solid in decomposition_solids:
            faces = extract_ocp_shape_type_from_shape(
                decomposition_solid.original_solid, TopAbs_FACE, cast_from_shape=True
            )
            n_faces = len(faces)

            # count the faces per body
            self.NUMBER_OF_FACES += n_faces

            for face in faces:
                # get the surface and its type from the face
                surface = _get_ocp_surface_from_ocp_face(face)
                surface_type = type(surface)

                if surface_type not in [
                    Geom_Plane,
                    Geom_CylindricalSurface,
                    Geom_ConicalSurface,
                    Geom_SphericalSurface,
                    Geom_ToroidalSurface,
                ]:
                    msg = f"face containes unsupported surface type {surface_type}"
                    logger.fatal(msg)
                    raise ValueError(msg)

                # count faces and surfaces by type
                self.FACES_PER_BODY[decomposition_solid] += 1
                self.FACE_TYPE_COUNT[surface_type] += 1
                self.FACE_TYPE_COUNT_PER_BODY[decomposition_solid][surface_type] += 1

                if (
                    isinstance(surface, Geom_ToroidalSurface)
                    and surface.MinorRadius() > surface.MajorRadius()
                ):
                    self.DEGENERATE_TORI_COUNT += 1

            # run analysis to check if cad is valid
            analyser = BRepCheck_Analyzer(decomposition_solid.original_solid, True)
            self.CAD_VALID[decomposition_solid] = analyser.IsValid()

    def __str__(self) -> str:
        """Return the summary and per body tables as a single string.

        Returns
        -------
        str
            The summary and per body tables.
        """
        return self.summary.as_string + "\n" + self.per_body.as_string

    @property
    def summary(self) -> str:
        """Table summarising the number of bodies, faces and surface types.

        Returns
        -------
        str
            The summary table formatted as a string.
        """
        # headers and rows
        headers_summary = ["quantity", "value"]
        rows_summary = [
            ["bodies", self.NUMBER_OF_BODIES],
            ["faces", self.NUMBER_OF_FACES],
            ["planes", self.FACE_TYPE_COUNT[Geom_Plane]],
            ["cylinders", self.FACE_TYPE_COUNT[Geom_CylindricalSurface]],
            ["cones", self.FACE_TYPE_COUNT[Geom_ConicalSurface]],
            ["spheres", self.FACE_TYPE_COUNT[Geom_SphericalSurface]],
            ["tori", self.FACE_TYPE_COUNT[Geom_ToroidalSurface]],
            ["tori (degenerate)", self.DEGENERATE_TORI_COUNT],
        ]

        # generate table
        table_summary = Table(rows_summary, headers=headers_summary)

        # return table
        return table_summary.as_string

    @property
    def per_body(self) -> str:
        """Table of validity and surface type counts for each body.

        Returns
        -------
        str
            The per body table formatted as a string.
        """
        # table - per body breakdown
        headers_per_body = [
            "cell_number",
            "is_valid",
            "surfaces",
            "planes",
            "cylinders",
            "cones",
            "spheres",
            "tori",
            "tori (degenerate)",
        ]
        rows_per_body = []

        for i, decomposition_solid in enumerate(self.FACES_PER_BODY):
            row = [
                i,
                self.CAD_VALID[decomposition_solid],
                self.FACES_PER_BODY[decomposition_solid],
                self.FACE_TYPE_COUNT_PER_BODY[decomposition_solid][Geom_Plane],
                self.FACE_TYPE_COUNT_PER_BODY[decomposition_solid][
                    Geom_CylindricalSurface
                ],
                self.FACE_TYPE_COUNT_PER_BODY[decomposition_solid][Geom_ConicalSurface],
                self.FACE_TYPE_COUNT_PER_BODY[decomposition_solid][
                    Geom_SphericalSurface
                ],
                self.FACE_TYPE_COUNT_PER_BODY[decomposition_solid][
                    Geom_ToroidalSurface
                ],
                self.DEGENERATE_TORI_COUNT,
            ]
            rows_per_body.append(row)

        # generate table
        table_per_body = Table(rows_per_body, headers=headers_per_body)

        # return table
        return table_per_body.as_string

    @staticmethod
    def gen_face_type_dict() -> dict[Type, int]:
        """Create a zeroed counter for each supported OCP surface type.

        Returns
        -------
        dict[Type, int]
            Dictionary mapping each supported ``Geom_*`` surface class to 0.
        """
        face_types = [
            Geom_Plane,
            Geom_CylindricalSurface,
            Geom_ConicalSurface,
            Geom_SphericalSurface,
            Geom_ToroidalSurface,
        ]

        face_dict = {face_type: 0 for face_type in face_types}

        return face_dict


def ocp_shape_intersect_ocp_shape(
    shape_1: "TopoDS_Shape", shape_2: "TopoDS_Shape", tolerance: float = 1e-4
) -> bool:
    """Check if two shapes intersect within a tolerance.

    The minimum distance between the shapes is calculated, and the shapes are
    considered to intersect if this is within the tolerance.

    Parameters
    ----------
    shape_1 : TopoDS_Shape
        First shape to check intersection.
    shape_2 : TopoDS_Shape
        Second shape to check intersecrtion
    tolerance : float, optional
        Tolerance for distance between shapes, by default 1e-4

    Returns
    -------
    bool
        Boolean value for cehck if shapes intersect.
    """
    dist_calc = BRepExtrema_DistShapeShape(shape_1, shape_2)
    dist_calc.Perform()
    return dist_calc.Value() <= tolerance


def print_plane_parameters(surface: Geom_Plane) -> None:
    """Log the parameters of a plane surface at debug level.

    Parameters
    ----------
    surface : Geom_Plane
        The plane surface whose origin, normal, x and y directions and bounds
        are logged.
    """
    gp_plane = surface.Pln()

    # Extract parameters
    location = gp_plane.Location()  # gp_Pnt
    normal = gp_plane.Axis().Direction()  # gp_Dir
    x_dir = gp_plane.XAxis().Direction()
    y_dir = gp_plane.YAxis().Direction()
    bounds = surface.Bounds()

    logger.debug("Plane Parameters:")
    logger.debug(
        f"  Origin     : ({location.X():.3f}, {location.Y():.3f}, {location.Z():.3f})"
    )
    logger.debug(
        f"  Normal     : ({normal.X():.3f}, {normal.Y():.3f}, {normal.Z():.3f})"
    )
    logger.debug(f"  X Direction: ({x_dir.X():.3f}, {x_dir.Y():.3f}, {x_dir.Z():.3f})")
    logger.debug(f"  Y Direction: ({y_dir.X():.3f}, {y_dir.Y():.3f}, {y_dir.Z():.3f})")
    logger.debug(f"  BOUNDS     : {bounds}")


class Decomposer:
    """Convert STEP CAD models into MCIO CSG models.

    On construction the solids are extracted from the STEP files and wrapped as
    :class:`DecompositionSolid` instances. :meth:`run` then decomposes each
    solid into CSG cells and generates the void, and :attr:`model` assembles
    the MCIO model.

    Parameters
    ----------
    step_file_paths : list[str], optional
        Paths of the STEP files to load, by default None
    materials_file_path : str | os.PathLike, optional
        Path to a JSON file defining materials and their assignment, by default
        None
    tallies_file_path : str | os.PathLike, optional
        Path to a JSON file defining tallies to apply after decomposition, by
        default None
    show_file_structure : bool, optional
        Whether to show the file structure of the STEP files, by default False
    decompose_parameters : DecompositionParameters or str or os.PathLike, optional
        Parameters controlling the decomposition, or the path of a json file
        written by :meth:`DecompositionParameters.write`, by default None
        (defaults are used)
    universe : Universe, optional
        Universe applied to all decomposition solids, by default None
    surface_number_min : int, optional
        Minimum surface number used when generating the model, by default None
    cell_number_min : int, optional
        Minimum cell number used when generating the model, by default None
    serialise : bool, optional
        Whether to serialise the OCP objects to binary on construction (needed
        for pickling and multiprocessing), by default False
    """
    def __init__(
        self,
        step_file_paths: Optional[list[str]] = None,
        materials_file_path: Optional[Union[str, os.PathLike]] = None,
        tallies_file_path: Optional[Union[str, os.PathLike]] = None,
        show_file_structure: bool = False,
        decompose_parameters: Optional[Union[DecompositionParameters, str, os.PathLike]] = None,
        universe: Optional[Universe] = None,
        surface_number_min: Optional[int] = None,
        cell_number_min: Optional[int] = None,
        serialise: bool = False,
    ) -> None:
        """Initialise the Decomposer and extract solids from the STEP files."""

        # set decomposition parameters (a json file path is read)
        if isinstance(decompose_parameters, (str, os.PathLike)):
            decompose_parameters = DecompositionParameters.read(decompose_parameters)
        elif decompose_parameters is not None and not isinstance(
            decompose_parameters, DecompositionParameters
        ):
            raise TypeError(
                "decompose_parameters must be a DecompositionParameters instance "
                f"or a json file path, not {type(decompose_parameters).__name__}"
            )
        self.decom_params: DecompositionParameters = (
            decompose_parameters if decompose_parameters else DecompositionParameters()
        )

        # user defines variables
        self._show_files_structure: bool = show_file_structure
        self._step_file_paths: list[str] = step_file_paths if step_file_paths else []

        # load materials form json
        mat_assignment_method, materials = (
            _materials_from_json(materials_file_path)
            if materials_file_path
            else (None, None)
        )
        self._material_assignment_method: Optional[str] = mat_assignment_method
        self._materials: Optional[dict[str, Material]] = materials

        # tallies file path
        self._tallies_file_path: Optional[Union[str, os.PathLike]] = tallies_file_path

        # set minimum surface and cell numbers
        self.surface_number_min: Optional[int] = surface_number_min
        self.cell_number_min: Optional[int] = cell_number_min

        # universe to apply to all decomposition_solids
        self.universe: Optional[Universe] = universe

        # store decomposed solids
        self._decomposition_solids: list[DecompositionSolid] = []

        # store OCP surface to MCIO surface to reduce repetition of surfaces
        self._surfaces_generated: dict[Any, dict[Any, Any]] = (
            _map_of_duplicate_surfaces()
        )

        # extract solids from document
        self._profile_time_load: Optional[float] = None
        _, self._profile_time_extract = self._extract_solids()

        # store profiling times
        self._profile_time_decomposition: Optional[float] = None
        self._profile_time_void_octree: Optional[float] = None
        self._profile_time_generate_mcio_cells: Optional[float] = None
        self._profile_time_model: Optional[float] = None

        # store monte carlo models and void octree
        self._void_octree: Optional[VoidOctreeFaces] = None
        self._model: Optional[Model] = None

        # serailise OCP objcets to binary, required for multiprocessing
        self._serialised: bool = False
        if serialise:
            self._serialised = True
            self.serialise()

    def serialise(self) -> None:
        """Serialise the OCP objects of all decomposition solids to binary.

        Required so that the decomposer can be pickled and multiprocessing used.
        """
        for decomposition_solid in self._decomposition_solids:
            decomposition_solid.serialise()

    def deserialise(self) -> None:
        """Deserialise binary streams back into OCP objects for all solids.

        Reverses :meth:`serialise` following pickling or multiprocessing.
        """
        for decomposition_solid in self._decomposition_solids:
            decomposition_solid.deserialise()

    @check_folder_exists_for_file_path("output_path")
    def pickle(self, output_path: Union[str, pathlib.Path, os.PathLike]) -> None:
        """Serialise and pickle the decomposer to a file.

        Parameters
        ----------
        output_path : str | pathlib.Path | os.PathLike
            Path of the pickle file to write.
        """
        # serialise to convert ocp solids to binary streams
        if not self._serialised:
            self.serialise()

        # pickle the decomposer instance
        with open(output_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def unpickle(cls, input_path: Union[str, pathlib.Path, os.PathLike]) -> "Decomposer":
        """Load a pickled decomposer from a file and deserialise it.

        Parameters
        ----------
        input_path : str | pathlib.Path | os.PathLike
            Path of the pickle file to read.

        Returns
        -------
        Decomposer
            The loaded decomposer instance.

        Raises
        ------
        TypeError
            If the unpickled object is not an instance of this class.
        """
        # load pickle from binary
        with open(input_path, "rb") as input_file:
            instance = pickle.load(input_file)

        # de-serialise to convert binary streams to ocp solids
        instance.deserialise()

        # check if class matches this one
        if not isinstance(instance, cls):
            msg = f" class instance loaded from pickle is not Decomposer()"
            logger.fatal(msg)
            raise TypeError(msg)

        return instance

    def filter_by_label(self, label: str) -> list[DecompositionSolid]:
        """Return decomposition solids whose structure tree contains a label.

        Parameters
        ----------
        label : str
            Substring searched for in the joined structure tree of each solid.

        Returns
        -------
        list[DecompositionSolid]
            The decomposition solids containing the label.
        """
        # filter decomposition solids for those with label in their structure tree
        decomposition_solids: list[DecompositionSolid] = [
            dc
            for dc in self._decomposition_solids
            if label in "".join(dc.structure_in_input_file)
        ]

        # return the list of decomposition_solids_filtered
        return decomposition_solids

    @property
    def analysis(self) -> Analysis:
        """Analysis of the bodies and surface types of the decomposition solids.

        Returns
        -------
        Analysis
            A freshly generated analysis of the decomposition solids.
        """
        # analyse decomposition_solids and produce analysis data
        analysis_data = Analysis(self._decomposition_solids)

        return analysis_data

    def add_solid(
        self,
        solid: "TopoDS_Solid",
        name: str = "added_solid",
        material: Optional[Material] = None,
        structure: Optional[list[str]] = None,
        colour: Optional[ColourRGB] = None,
    ) -> DecompositionSolid:
        """Add an OCP solid to the decomposer as a new decomposition solid.

        If ``decom_params.debug_write_added_solids`` is set, the solid is also
        written to a STEP file in ``added_solids/``.

        Parameters
        ----------
        solid : TopoDS_Solid
            The solid to add.
        name : str, optional
            Name used as the structure tree if none given, by default
            "added_solid"
        material : Material, optional
            Material assigned to the solid, by default None
        structure : list[str], optional
            Structure tree of the solid, by default None
        colour : ColourRGB, optional
            Colour of the solid (currently unused), by default None

        Returns
        -------
        DecompositionSolid
            The decomposition solid created.
        """

        decomposition_solid = DecompositionSolid(
            solid,
            self.decom_params,
            material=material,
            universe=self.universe,
            structure_original=structure if structure else [name],
        )

        if self.decom_params.debug_write_added_solids:
            step_file_writer = STEPFileWriter()
            step_file_writer.add_solid(solid, "added_solid")
            step_file_writer.write("added_solids/added_solid", enumerate_files=True)

        self._decomposition_solids.append(decomposition_solid)

        return decomposition_solid

    def profile(self, all_decomosition_solids: bool = False) -> None:
        """Print a table of the time taken by each stage of the conversion.

        Parameters
        ----------
        all_decomosition_solids : bool, optional
            Whether to also list the decomposition time of each solid, by
            default False
        """
        # calculate the number of solids decomposed
        solids_decomposed = len(
            [
                decomposition_solid
                for decomposition_solid in self._decomposition_solids
                if decomposition_solid._profile_time_decomposition
            ]
        )

        # define headers
        headers = ["action", "time taken (s)"]

        # define rows
        rows = [
            [f"load_step_files", self._profile_time_load],
            [f"extract_solids", self._profile_time_extract],
            [f"decompose_{solids_decomposed}_cells", self._profile_time_decomposition],
            [f"mcio_cell_generation", self._profile_time_generate_mcio_cells],
            [f"recursive_octree_generation", self._profile_time_void_octree],
            [f"model_generatoin", self._profile_time_model],
        ]

        # calculate total time taken
        total_time = sum(
            [
                self._profile_time_load if self._profile_time_load else 0.0,
                self._profile_time_extract if self._profile_time_extract else 0.0,
                (
                    self._profile_time_decomposition
                    if self._profile_time_decomposition
                    else 0.0
                ),
                (
                    self._profile_time_generate_mcio_cells
                    if self._profile_time_generate_mcio_cells
                    else 0.0
                ),
                (
                    self._profile_time_void_octree
                    if self._profile_time_void_octree
                    else 0.0
                ),
            ]
        )

        if all_decomosition_solids:
            for i, decomposition_solid in enumerate(
                self._decomposition_solids, start=1
            ):
                rows.append(
                    [f"cell_{i}", decomposition_solid.profile_time_decomposition]
                )

        rows.append(
            [f"total_time (bodies decomposed = {solids_decomposed})", total_time]
        )

        table = Table(rows=rows, headers=headers)

        print(table)

    @property
    def decomposition_solids(self) -> list[DecompositionSolid]:
        """list[DecompositionSolid]: The decomposition solids of the model."""
        return self._decomposition_solids

    @decomposition_solids.setter
    def decomposition_solids(
        self, decomposition_solids: list[DecompositionSolid]
    ) -> None:
        """Set the decomposition solids of the model.

        Parameters
        ----------
        decomposition_solids : list[DecompositionSolid]
            The decomposition solids to set.
        """
        self._decomposition_solids = decomposition_solids

    @property
    def void_octree(self) -> Optional[VoidOctreeFaces]:
        """The void octree, generated on first access if not already present.

        Returns
        -------
        VoidOctreeFaces or None
            The void octree of the model.
        """
        # if not generated, set up parent at top level
        if not self._void_octree:
            # generate parent at top level to allow user to manually control void run via multiprocess
            self._void_octree, self._profile_time_void_octree = (
                generate_mcio_void_octree(
                    self._decomposition_solids,
                    self._surfaces_generated,
                    self.decom_params,
                    run=True,
                    show_progress=True,
                    universe=self.universe,
                )
            )

        return self._void_octree

    @property
    def model(self) -> Model:
        """The MCIO model, generated on first access if not already present.

        Returns
        -------
        Model
            The MCIO model containing the decomposed and void cells.
        """
        if not self._model:
            logger.debug(f"Decomposer() - generating model")
            self._model, self._profile_time_model = self._generate_mcio_model(
                surface_number_min=self.surface_number_min,
                cell_number_min=self.cell_number_min,
            )
            logger.debug(
                f"Decomposer() - model generated - time taken = {self._profile_time_model:.2e}"
            )
        else:
            logger.debug(f"Decomposer() - model already generated")

        return self._model

    def tallies_from_json(
        self, input_path: Union[str, pathlib.Path, os.PathLike]
    ) -> None:
        """Create tallies from a JSON file and add them to the model.

        Each key of the JSON is a label matched against the structure tree of
        the decomposition solids (see :meth:`filter_by_label`); its value is
        the tally data, including a ``tally_type`` entry.

        Parameters
        ----------
        input_path : str | pathlib.Path | os.PathLike
            Path of the JSON file defining the tallies.
        """
        with open(input_path) as input_file:
            tallies = json.load(input_file)

        tallies_to_apply: list[Tally] = []

        for tally_label, tally_data in tallies.items():
            # get decomposition solids with tally_label in their structure tree
            decomp_solids_to_tally: list[DecompositionSolid] = self.filter_by_label(
                tally_label
            )

            # skip if there are no decomposition solids to apply tally to
            if len(decomp_solids_to_tally) == 0:
                continue

            # get tally type as string, and remove from tally data to pass
            tally_type_label: str = tally_data["tally_type"].lower()
            del tally_data["tally_type"]

            # get parameters to translate for tally types
            if tally_type_label == "fluencemeshtally" and "resolution" in tally_data:
                # case where voxel size calculated from the solids
                resolution: float = tally_data["resolution"]
                del tally_data["resolution"]

                # create bounding box for all solids
                bb: BoundingBox = decomp_solids_to_tally[0].bounding_box

                for decomposition_solid_to_tally in decomp_solids_to_tally[1:]:
                    bb = bb | decomposition_solid_to_tally.bounding_box

                tally_data["x_min"] = bb.x_min
                tally_data["x_max"] = bb.x_max
                tally_data["y_min"] = bb.y_min
                tally_data["y_max"] = bb.y_max
                tally_data["z_min"] = bb.z_min
                tally_data["z_max"] = bb.z_max
                tally_data["x_ints"] = math.ceil((bb.x_max - bb.x_min) / resolution)
                tally_data["y_ints"] = math.ceil((bb.y_max - bb.y_min) / resolution)
                tally_data["z_ints"] = math.ceil((bb.z_max - bb.z_min) / resolution)

            elif tally_type_label in [
                "volumefluencetally",
                "pulseheighttally",
                "energydepositiontally",
            ]:
                tally_data["cells"] = [
                    decomposition_solid.mcio_cell
                    for decomposition_solid in decomp_solids_to_tally
                ]

            # convert tally data to tally, and add to tallies to apply in model
            tallies_to_apply.append(Tally.from_dict(tally_type_label, tally_data))

        # add tallies to model
        self.model.tallies = (
            tallies_to_apply
            if not self.model.tallies
            else [*self.model.tallies, *tallies_to_apply]
        )

    @function_timer
    def run(
        self,
        cores: int = 20,
        analyse_model: bool = True,
        generate_void: bool = True,
        debug: bool = False,
    ) -> None:
        """Decompose all solids into CSG cells and generate the void.

        Parameters
        ----------
        cores : int, optional
            Number of processes used for decomposition, by default 20
        analyse_model : bool, optional
            Whether to analyse the model first and abort on degenerate tori, by
            default True
        generate_void : bool, optional
            Whether to generate the void octree, by default True
        debug : bool, optional
            Whether to run on a single core with debug logging and write each
            original solid to a STEP file, by default False

        Raises
        ------
        Exception
            If degenerate tori are detected, or any solid fails to decompose.
        """

        # run analysis to check model
        if analyse_model:
            analysis = self.analysis

            if analysis.DEGENERATE_TORI_COUNT > 0:
                msg = f"[ERROR] - {analysis.DEGENERATE_TORI_COUNT} degenerate tori detected, and will not convert"
                logger.fatal(msg)
                raise Exception(msg)

        # calculate for formatting output logging strings
        n_decomposition_solids = len(self._decomposition_solids)
        padding_index = math.ceil(math.log10(n_decomposition_solids))

        # set logging level if debugging required
        if debug:
            logger.setLevel(LOGGER_LEVELS.DEBUG)

        # set description for progress bar
        desc_stage = (
            f"[Decomposition (1 core debug)]"
            if debug
            else (
                f"[Decomposition (1 core)]"
                if cores == 1
                else f"[Decomposition ({cores} cores)]"
            )
        )

        # check if running notebook
        is_notebook_env = in_notebook()

        # setup progress reporting mechanism depending on environment
        if is_notebook_env:
            pbar = tqdm_notebook(
                total=n_decomposition_solids, desc=desc_stage, unit="solid"
            )

            def update_progress() -> None:
                """Advance the notebook progress bar by one."""
                pbar.update(1)

            def close_progress() -> None:
                """Close the notebook progress bar."""
                pbar.close()

        else:
            desc = f"[cyan]{desc_stage: <30}[/cyan]"
            progress = Progress(
                TextColumn(desc),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            progress.start()
            task_id = progress.add_task("decomposing", total=n_decomposition_solids)

            def update_progress() -> None:
                """Advance the progress bar by one."""
                progress.advance(task_id)

            def close_progress() -> None:
                """Stop the progress bar."""
                progress.stop()

        # case for single core and/or debugging mode
        if cores == 1 or debug:
            # set up to collect results
            results = []

            try:
                for i, decomposition_solid in list(
                    enumerate(self._decomposition_solids, start=1)
                ):
                    # write to logger
                    logger.info(f"decomposing cell {i}")

                    # pass decomposition solid to _run_decomposer_solid run method
                    result = _run_decomposer_solid(
                        [
                            i,
                            n_decomposition_solids,
                            padding_index,
                            decomposition_solid,
                            debug,
                        ]
                    )
                    # collect result of _run_decomposer_solid run method
                    results.append(result)

                    # for debugging, write original solid to a step file
                    if debug:
                        # write solid to analysis folder
                        step_file_writer = STEPFileWriter(
                            decomposition_solid.decom_params
                        )
                        step_file_writer.add_solids(
                            [decomposition_solid.original_solid],
                            f"original_solid_{i}",
                        )
                        output_path = step_file_writer.write(
                            os.path.join(
                                decomposition_solid.decom_params.debug_save_solid,
                                f"original_solid_{i}.stp",
                            )
                        )
                        logger.debug(f"converting solid {i} - see {output_path}")

                    # advance the progress bar
                    update_progress()
            finally:
                close_progress()

        else:
            # allow parallel logging
            log_queue, listener = start_multiprocess_logging()

            # serialise
            self.serialise()

            enumerated_decomposition_solids = [
                (
                    i,
                    n_decomposition_solids,
                    padding_index,
                    decomposition_solid,
                    debug,
                )
                for i, decomposition_solid in enumerate(self.decomposition_solids)
            ]

            try:
                with ProcessPoolExecutor(
                    max_workers=cores,
                    initializer=configure_worker_logger,
                    initargs=(log_queue,),
                ) as executor:
                    results_generator = executor.map(
                        _run_decomposer_solid,
                        enumerated_decomposition_solids,
                    )

                    results = []
                    for result in results_generator:
                        results.append(result)
                        update_progress()
            finally:
                close_progress()

            # restore single core logging
            listener.stop()
            restore_main_process_logging()

            self.decomposition_solids = [
                decomposition_solid for _, decomposition_solid, _ in results
            ]

            # deserialise
            self.deserialise()

        # collect decompositions solids which failed and collect their exceptions
        results_with_exceptions = [
            (i, decomposition_solid, exception)
            for i, decomposition_solid, exception in results
            if exception
        ]

        # for single core with no debugging, save all failed solids to step
        if not debug and results_with_exceptions:

            msgs = ""
            for i, decomposition_solid, exception in results_with_exceptions:
                msg = f" - solid_{i} failed - see logger file for exception\n"
                msgs += msg
                logger.fatal(msg)
                logger.debug(f"{exception}", exc_info=True)

            msg = (
                f"conversion failed - total of {len(results_with_exceptions)}"
                + "failed to convert - see directory"
                + f"{self.decom_params.debug_failed_to_convert_directory}"
                + "\n\n"
            )
            msg += (
                f"the following {len(results_with_exceptions)}"
                + "step files were produced:\n"
            )
            msg += msgs
            logger.fatal(msgs)
            raise Exception(msg)

        # skip void generation if required
        if generate_void:
            # generate void octree
            self._void_octree, self._profile_time_void_octree = (
                generate_mcio_void_octree(
                    self._decomposition_solids,
                    self._surfaces_generated,
                    self.decom_params,
                    universe=self.universe,
                )
            )

        # if json of tallies provided, apply these
        if self._tallies_file_path:
            self.tallies_from_json(self._tallies_file_path)

    def __add__(self, other: "Decomposer") -> None:
        """Add a decomposer to another (not yet implemented).

        Intended to assume void generation is not required, only adding the
        cells in the models together.

        Parameters
        ----------
        other : Decomposer
            The decomposer to add.
        """

    def __and__(self, other: "Decomposer") -> "Decomposer":
        """Combine two decomposers into a new decomposer.

        The void is reset, and all decomposition solids are assumed to be in
        the same universe.

        Parameters
        ----------
        other : Decomposer
            The decomposer to combine with this one.

        Returns
        -------
        Decomposer
            A new decomposer containing the decomposition solids of both.
        """
        # set up empty decomposer
        decomposer = Decomposer()

        # fill empty decomposer with decomposition solids
        decomposer._decomposition_solids = [
            *self._decomposition_solids,
            *other._decomposition_solids,
        ]

        # return filled decomposer
        return decomposer

    def __str__(self) -> str:
        """Return a table of surface, split solid and auxiliary counts per body.

        Returns
        -------
        str
            The table formatted as a string.
        """
        headers = [
            "body",
            "planes",
            "cylinders",
            "spheres",
            "cones",
            "tori",
            "decomposed_solids",
            "auxiliary_surfaces",
        ]

        rows = []

        for index_decomposer_solid, decomposer_solid in enumerate(
            self._decomposition_solids, start=1
        ):
            face_count = count_surface_by_type(decomposer_solid.original_solid)
            n_split_solids = len(decomposer_solid.split_solids.keys())
            n_auxiliary_surfaces = sum(
                [
                    len(auxiliary_surfaces)
                    for auxiliary_surfaces in decomposer_solid.split_solids.values()
                ]
            )
            row = [
                index_decomposer_solid,
                face_count[Geom_Plane],
                face_count[Geom_CylindricalSurface],
                face_count[Geom_SphericalSurface],
                face_count[Geom_ConicalSurface],
                face_count[Geom_ToroidalSurface],
                n_split_solids,
                n_auxiliary_surfaces,
            ]
            rows.append(row)

        return Table(rows, headers=headers).as_string

    @function_timer
    def _generate_mcio_model(
        self,
        surface_number_min: Optional[int] = None,
        cell_number_min: Optional[int] = None
    ) -> Model:
        """Generate the MCIO model from the decomposed and void cells.

        A spherical stochastic volume source and a volume fluence tally over
        the decomposed cells are added for stochastic volume calculation.

        Parameters
        ----------
        surface_number_min : int, optional
            Minimum surface number, by default None
        cell_number_min : int, optional
            Minimum cell number, by default None

        Returns
        -------
        Model
            The generated model (wrapped by ``function_timer`` so the caller
            receives ``(Model, time_taken)``).
        """
        # gather cells from all decomposition_solids
        logger.debug(f"Decomposer() - getting mcio cells of solids")
        cells, self._profile_time_generate_mcio_cells = _generate_mcio_cells(
            self._decomposition_solids, self.decom_params
        )
        logger.debug(
            f"DEcomposer() - generated mcio cells - time taken = {self._profile_time_generate_mcio_cells:.2e}"
        )

        # set up dictionary of surfaces previously generated
        logger.debug(f"DEcomposer() - getting map of duplicated")
        time_start = time.time()
        self._surfaces_generated = _map_of_duplicate_surfaces()
        time_total = time.time()
        logger.debug(f"Decomposer() - map of duplicated - time take = {time_total:.2e}")

        # generate void cells
        logger.debug(f"VoidOctree() - getting .mcio_cells")
        time_start = time.time()
        cells_voids = self.void_octree.mcio_cells
        time_total = time.time() - time_start
        logger.debug(
            f"VoidOctree() - .mcio_cells generated - time taken = {time_total:.2e}"
        )

        # add stochastic volume source
        source = SphericalStochasticVolumeSource(
            spherical_surface=self.void_octree.surf_stochastic_volume_sphere
        )

        # add tallies for stochastic volume check
        tallies = [
            VolumeFluenceTally(cells=cells, comment="stochastic_volume_calculation")
        ]

        # summary data
        total_time = sum(
            [
                self._profile_time_extract if self._profile_time_extract else 0.0,
                (
                    self._profile_time_decomposition
                    if self._profile_time_decomposition
                    else 0.0
                ),
                (
                    self._profile_time_generate_mcio_cells
                    if self._profile_time_generate_mcio_cells
                    else 0.0
                ),
                (
                    self._profile_time_void_octree
                    if self._profile_time_void_octree
                    else 0.0
                ),
            ]
        )

        summary_data = {
            "mcio extract solids time (s)": self._profile_time_extract,
            "mcio decompose time (s)": self._profile_time_decomposition,
            "mcio cell generation (s)": self._profile_time_generate_mcio_cells,
            "mcio octree generation (s)": self._profile_time_void_octree,
            "mcio total time (s)": total_time,
        }

        # generate model
        model = Model(
            cells=[*cells, *cells_voids],
            source=source,
            tallies=tallies,
            summary_data=summary_data,
            surface_number_min=surface_number_min,
            cell_number_min=cell_number_min
        )

        if self.universe:
            for cell in model.cells:
                cell.universe = self.universe

        return model

    def _get_colour(
        self, ref_label: TDF_Label, label: TDF_Label, colour_tool: XCAFDoc_ColorTool
    ) -> tuple[int, int, int]:
        """Get the surface colour of a label as 0-255 RGB integers.

        The referred label is checked first, then the label itself.

        Parameters
        ----------
        ref_label : TDF_Label
            The referred (shape definition) label.
        label : TDF_Label
            The label of the shape instance.
        colour_tool : XCAFDoc_ColorTool
            The XCAF colour tool of the document.

        Returns
        -------
        tuple[int, int, int]
            The red, green and blue values, or (-1, -1, -1) if no colour is set.
        """
        colour_surf = Quantity_ColorRGBA()
        if colour_tool.GetColor_s(ref_label, XCAFDoc_ColorSurf, colour_surf):
            rgb = colour_surf.GetRGB()
            red, green, blue = rgb.Values(Quantity_TOC_sRGB)
            return round(red * 255), round(green * 255), round(blue * 255)
        elif colour_tool.GetColor_s(label, XCAFDoc_ColorSurf, colour_surf):
            rgb = colour_surf.GetRGB()
            red, green, blue = rgb.Values(Quantity_TOC_sRGB)
            return round(red * 255), round(green * 255), round(blue * 255)
        else:
            red, green, blue = -1, -1, -1
            return red, green, blue

    @function_timer
    def _extract_solids(self) -> None:
        """Load the STEP files and create a decomposition solid for each solid.

        Materials are assigned from the materials file if one was given. The
        method is wrapped by ``function_timer``, so the caller receives
        ``(None, time_taken)``.
        """

        desc_stage = "[Loading STEP files]"
        is_notebook_env = in_notebook()

        if is_notebook_env:
            desc_tqdm = f"{desc_stage: <30}"
            pbar = tqdm_notebook(
                total=len(self._step_file_paths), desc=desc_tqdm, unit="file"
            )

            def update_progress() -> None:
                """Advance the notebook progress bar by one."""
                pbar.update(1)

            def close_progress() -> None:
                """Close the notebook progress bar."""
                pbar.close()

        else:
            desc = f"[cyan]{desc_stage: <30}[/cyan]"
            progress = Progress(
                TextColumn(desc),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )

            progress.start()
            task_id = progress.add_task("loading", total=len(self._step_file_paths))

            def update_progress() -> None:
                """Advance the progress bar by one."""
                progress.advance(task_id)

            def close_progress() -> None:
                """Stop the progress bar."""
                progress.stop()

        # iterate over all step files for input
        for i, step_file_path in enumerate(self._step_file_paths, start=1):
            # log each file loading
            logger.info(f"[LOADING] - loading step file {i} {step_file_path}")

            step_file_reader: list[STEPFileReader] = STEPFileReader(step_file_path, self.decom_params)

            for step_solid in step_file_reader.solids:
                logger.debug(f'COLOUR ASSIGNMENT - {step_solid.structure} r={step_solid.colour.red} r={step_solid.colour.green} r={step_solid.colour.blue}')

                material = (
                    _material_lookup(
                        material_assignment_method=self._material_assignment_method,
                        red=step_solid.colour.red,
                        green=step_solid.colour.green,
                        blue=step_solid.colour.blue,
                        structure=step_solid.structure,
                        materials=self._materials,
                        decom_params=self.decom_params,
                    )
                    if self._materials
                    else None
                )

                decompose_solid = DecompositionSolid(
                    step_solid.solid,
                    self.decom_params,
                    structure_original=[s for s in step_solid.structure],
                    colour_rgb=step_solid.colour.rgb,
                    material=material,
                    universe=self.universe,
                    surfaces_generated=self._surfaces_generated,
                )

                self.decomposition_solids.append(decompose_solid)

            # advance progress bar following loading of each step file to doc
            update_progress()

        close_progress()
