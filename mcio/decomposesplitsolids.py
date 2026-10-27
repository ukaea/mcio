"""Module contains classes and methods to encalsulate a split solid, which
is one part of a cad solid during the decomposition process.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import pickle

try:
    import casadi
    import OCP
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.TopAbs import TopAbs_REVERSED, TopAbs_FACE
    from OCP.TopoDS import TopoDS_Solid, TopoDS_Face
except Exception as e:
    from mcio import logger

    logger.warning("failed to load OCP")
    logger.warning(f"{e}")

from mcio import logger
from .cell import Cell
from .volume import HalfVolume
from .decomposeparams import DecompositionParameters
from .ocp import (
    extract_ocp_shape_type_from_shape,
    bounding_box_from_shape,
    convert_ocp_shape_to_binary,
    convert_ocp_surface_to_binary,
    convert_ocp_orientation_to_binary,
    convert_surface_binary_to_ocp_surface,
    convert_orientation_binary_to_ocp_orientation,
    convert_shape_binary_to_ocp_shape,
)
from .ocptomcio import get_mcio_halfspace_from_ocp_face
from .ocpboundingbox import oriented_bounding_box_from_shape
from .ocpproperties import _volume_of_ocp_solids
from .geometric import BoundingBox, OrientedBoundingBox
from .model import Model
from .surface import Sphere


class SplitSolid:
    """Represent an individual sub-solid formed from geometric decomposition.

    Manages geometric boundaries, mapping OpenCASCADE shapes and faces to MCIO
    half-spaces, auxiliary boundary surfaces, bounding box computations, and 
    CSG model generation.

    Parameters
    ----------
    split_solid : TopoDS_Solid or bytes
        The OpenCASCADE sub-solid shape or its binary serialization.
    faces_with_auxiliaries : dict
        Mapping of OpenCASCADE faces (or face binaries) to lists of auxiliary 
        surfaces and orientations.
    decom_params : DecompositionParameters
        Configuration options controlling tolerance and decomposition behavior.
    surfaces_generated : dict, optional
        Mapping tracking existing generated surfaces to prevent duplication, by
        default None.

    Attributes
    ----------
    split_solid : TopoDS_Solid or bytes
        Underlying shape object or its binary equivalent.
    decom_params : DecompositionParameters
        Decomposition settings used by this solid.
    faces_with_auxiliaries : dict
        Associated auxiliary surfaces for each face.
    n_faces : int
        Number of topological faces making up the split solid.
    is_serialised : bool
        Flag indicating if the shape objects are currently in binary 
        serialization format.
    bounding_box : BoundingBox
        Axis-aligned bounding box of the sub-solid.
    oriented_bounding_box : OrientedBoundingBox
        Oriented bounding box enclosing the sub-solid.
    face_to_mcio_halfspaces : dict
        Map of sub-solid faces to corresponding MCIO half-space primitives.
    face_auxiliary_surfaces_to_mcio_halfspaces : dict
        Nested map of auxiliary surface faces to MCIO half-space primitives.
    face_bounding_boxes : dict
        Map of individual faces to their respective axis-aligned bounding boxes.
    volume : HalfVolume or None
        Constructed CSG volume formed by intersecting all bounding half-spaces.
    """

    def __init__(
        self,
        split_solid: Union["TopoDS_Solid", bytes],
        faces_with_auxiliaries: Dict[Any, List[Tuple[Any, Any]]],
        decom_params: DecompositionParameters,
        surfaces_generated: Optional[Dict[Any, Any]] = None,
    ) -> None:
        """Initialise the SplitSolid and build its MCIO half-spaces and volume."""
        # store input variables
        self.split_solid: Union["TopoDS_Solid", bytes] = split_solid
        self.decom_params: DecompositionParameters = decom_params
        self._surfaces_generated: Dict[Any, Any] = (
            surfaces_generated if surfaces_generated is not None else {}
        )
        self.faces_with_auxiliaries: Dict[Any, List[Tuple[Any, Any]]] = (
            faces_with_auxiliaries
        )

        # calculate number of faces
        self.n_faces: int = len(
            extract_ocp_shape_type_from_shape(
                split_solid, TopAbs_FACE, cast_from_shape=False
            )
        )

        # set serialisation flag
        self.is_serialised: bool = False

        # calculate bounding box for split solid
        self.bounding_box: BoundingBox = bounding_box_from_shape(self.split_solid)
        self.oriented_bounding_box: OrientedBoundingBox = (
            oriented_bounding_box_from_shape(self.split_solid)
        )

        # generate dicts to store mcio halfspaces for faces and auxiliary surfaces
        self.face_to_mcio_halfspaces: Dict[Any, Any] = {}
        self.face_auxiliary_surfaces_to_mcio_halfspaces: Dict[Any, Dict[Any, Any]] = {}

        # generate bounding boxes for each face
        self.face_bounding_boxes: Dict[Any, BoundingBox] = {}

        # collect mcio half spaces for creation of mcio volume, for later passing to mcio cell.
        mcio_half_spaces: List[Any] = []

        # store mcio csg volume
        self.volume: Optional[Any] = None

        # store physical volume
        self._geometric_volume: Optional[float] = None
        self._fraction_of_bounding_box: Optional[float] = None

        # produce mcio half spaces for faces and for auxiliary surfaces of faces
        for (face_ocp, auxiliary_surfaces) in faces_with_auxiliaries.items():
            # get the mcio half space for the ocp face and store for later use
            mcio_halfspace_face = get_mcio_halfspace_from_ocp_face(
                face_ocp,
                self.decom_params,
                surfaces_generated=self._surfaces_generated,
            )
            self.face_to_mcio_halfspaces[face_ocp] = mcio_halfspace_face
            self.face_bounding_boxes[face_ocp] = bounding_box_from_shape(face_ocp)
            mcio_half_spaces.append(mcio_halfspace_face)

            # set up dict to store auxiliary surface half spaces
            self.face_auxiliary_surfaces_to_mcio_halfspaces[face_ocp] = {}

            for (surface, orientation) in auxiliary_surfaces:
                # create face for auxiliary surface
                face_aux = BRepBuilderAPI_MakeFace(
                    surface, 0.0, 1.0, 0.0, 1.0, decom_params.make_face_tol
                ).Face()

                # create mcio half space from auxiliary face
                mcio_halfspace_aux = get_mcio_halfspace_from_ocp_face(
                    face_aux,
                    self.decom_params,
                    surfaces_generated=self._surfaces_generated,
                    reverse_direction=orientation == TopAbs_REVERSED,
                )

                # add mcio half space for auxiliary surface to dictionary
                self.face_auxiliary_surfaces_to_mcio_halfspaces[face_ocp][
                    face_aux
                ] = mcio_halfspace_aux
                mcio_half_spaces.append(mcio_halfspace_aux)

        # create mcio volume from half spaces
        self.volume = mcio_half_spaces[0]
        for mcio_half_space in mcio_half_spaces[1:]:
            self.volume = self.volume & mcio_half_space

    def serialise(self) -> Dict["TopoDS_Face", bytes]:
        """Convert OpenCASCADE shape objects and faces into binary byte strings.

        Returns
        -------
        dict
            Mapping of original face objects to their binary representations.

        Notes
        -----
        This facilitates process communication and multiprocessing by stripping
        non-picklable OpenCASCADE references.
        """
        # convert split solid
        self.split_solid = convert_ocp_shape_to_binary(self.split_solid)

        # face_to_binary_lookup
        face_to_binary_lookup: Dict[Any, bytes] = {}

        # iterate over all faces
        for face in list(self.face_to_mcio_halfspaces):
            # convert face to binary
            face_binary = convert_ocp_shape_to_binary(face)

            # collect face to binary lookup to allow void octree to keep the same faces
            face_to_binary_lookup[face] = face_binary

            # convert auxiliary surfaces for each face to binary
            self.faces_with_auxiliaries[face_binary] = [
                [
                    convert_ocp_surface_to_binary(surface),
                    convert_ocp_orientation_to_binary(orientation),
                ]
                for surface, orientation in self.faces_with_auxiliaries[face]
            ]
            del self.faces_with_auxiliaries[face]

            # replace face with face binary for face mcio half spaces
            self.face_to_mcio_halfspaces[face_binary] = self.face_to_mcio_halfspaces[
                face
            ]
            del self.face_to_mcio_halfspaces[face]

            # replace face with binary for face to bounding boxes
            self.face_bounding_boxes[face_binary] = self.face_bounding_boxes[face]
            del self.face_bounding_boxes[face]

            # replace face for auxiliary surface to mcio half spaces
            self.face_auxiliary_surfaces_to_mcio_halfspaces[face_binary] = {}
            for face_aux in self.face_auxiliary_surfaces_to_mcio_halfspaces[face]:
                self.face_auxiliary_surfaces_to_mcio_halfspaces[face_binary][
                    convert_ocp_shape_to_binary(face_aux)
                ] = self.face_auxiliary_surfaces_to_mcio_halfspaces[face][face_aux]
            del self.face_auxiliary_surfaces_to_mcio_halfspaces[face]

        # set serialisation flag
        self.is_serialised = True

        # return
        return face_to_binary_lookup

    def deserialise(self) -> None:
        """Convert binary byte representations back into OpenCASCADE shape objects.

        Notes
        -----
        Reverses the operations performed by `serialise()`.
        """
        # convert split solid binary to split solid
        self.split_solid = convert_shape_binary_to_ocp_shape(self.split_solid)

        # face_to_binary_lookup
        face_binary_to_face_lookup: Dict[bytes, Any] = {}

        # iterate over face binaries
        for face_binary in list(self.face_to_mcio_halfspaces):
            # convert face binary to face
            face = convert_shape_binary_to_ocp_shape(face_binary)

            # store face and face binary for external use
            face_binary_to_face_lookup[face_binary] = face

            # convert auxiliary surfaces for each face to binary
            self.faces_with_auxiliaries[face] = [
                [
                    convert_surface_binary_to_ocp_surface(surface_binary),
                    convert_orientation_binary_to_ocp_orientation(orientation_binary),
                ]
                for surface_binary, orientation_binary in self.faces_with_auxiliaries[
                    face_binary
                ]
            ]
            del self.faces_with_auxiliaries[face_binary]

            # replace face binary with face for face mcio half spaces
            self.face_to_mcio_halfspaces[face] = self.face_to_mcio_halfspaces[
                face_binary
            ]
            del self.face_to_mcio_halfspaces[face_binary]

            # replace face binary with face for face to bounding boxes
            self.face_bounding_boxes[face] = self.face_bounding_boxes[face_binary]
            del self.face_bounding_boxes[face_binary]

            # replace face for auxiliary surface to mcio half spaces
            self.face_auxiliary_surfaces_to_mcio_halfspaces[face] = {}
            for face_aux_binary in self.face_auxiliary_surfaces_to_mcio_halfspaces[
                face_binary
            ]:
                face_aux = convert_shape_binary_to_ocp_shape(face_aux_binary)
                self.face_auxiliary_surfaces_to_mcio_halfspaces[face][face_aux] = (
                    self.face_auxiliary_surfaces_to_mcio_halfspaces[face_binary][
                        face_aux_binary
                    ]
                )
            del self.face_auxiliary_surfaces_to_mcio_halfspaces[face_binary]

        # set serialisation flag
        self.is_serialised = False

    def pickle(self) -> None:
        """Inspect all instance attributes for pickling capability and print failures.

        Notes
        -----
        Utility diagnostic function used to identify non-serializable objects.
        """
        # --- DIAGNOSTIC CODE ---
        print("--- Inspecting Decomposer Attributes for Pickle Safety ---")
        for attr_name, attr_value in self.__dict__.items():
            try:
                pickle.dumps(attr_value)
            except Exception as e:
                print(f"FAILED PICKLE: '{attr_name}' (type: {type(attr_value)}) -> {e}")
        # -----------------------

    @property
    def model(self) -> Model:
        """Model: MCIO model of the split solid enclosed in a spherical graveyard."""
        # create volumes
        vol_graveyard: HalfVolume = +Sphere(
            x0=self.bounding_box.x_mid,
            y0=self.bounding_box.y_mid,
            z0=self.bounding_box.z_mid,
            radius=self.bounding_box.radius_enclosure,
        )
        vol_void = ~vol_graveyard & ~self.volume

        # cell
        cell_split_solid = Cell(volume=self.volume)
        cell_void = Cell(volume=vol_void, name="void_between_model_and_graveyard")
        cell_graveyard = Cell(volume=vol_graveyard, name="graveyard", graveyard=True)

        # return mcio model
        return Model(cells=[cell_split_solid, cell_void, cell_graveyard])

    @property
    def geometric_volume(self) -> float:
        """float: Calculated physical volume of the split solid."""
        if self._geometric_volume is None:
            self._geometric_volume = _volume_of_ocp_solids([self.split_solid])

        return self._geometric_volume

    @property
    def fraction_of_bounding_box(self) -> float:
        """float: Ratio of the sub-solid volume to its bounding box volume."""
        # if not already stored, calculate the fraction of space the split solid takes of the bbox.
        if self._fraction_of_bounding_box is None:
            self._fraction_of_bounding_box = (
                self.geometric_volume / self.bounding_box.volume
            )

        # return the fraction
        return self._fraction_of_bounding_box
