"""General OpenCASCADE (OCP) helper functions used during decomposition.

Contains pre-splitting routines for solids (e.g. splitting pipes on circular
edges), small geometric utilities operating on OCP points, vectors and axes,
XCAF document/label helpers, intersection checks between shapes, and functions
to serialise OCP shapes, surfaces, orientations and documents to/from binary so
they can be pickled (e.g. for multiprocessing).
"""

from __future__ import annotations

import math
import os
import copy
import io
from io import BytesIO
import numpy as np
from dataclasses import dataclass
from collections import OrderedDict, deque
from typing import BinaryIO
from typing import TYPE_CHECKING
from collections.abc import Callable

if TYPE_CHECKING:
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.TopoDS import TopoDS_CompSolid
    from OCP.XCAFDoc import XCAFDoc_ShapeTool



# import first to allow logging of optional modules 
from mcio import logger

try:
    import cadquery as cq
except:
    pass

try:
    from jupyter_cadquery import show
except:
    pass

try:
    import OCP
    from OCP.BRep import (
        BRep_Tool,
        BRep_Builder
    )
    from OCP.BRepAlgoAPI import (
        BRepAlgoAPI_Common,
        BRepAlgoAPI_Section
    )
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeFace
    )
    from OCP.BRepTools import (
        BRepTools,
    )
    from OCP.BRepPrimAPI import (
        BRepPrimAPI_MakeBox,
        BRepPrimAPI_MakeSphere,
        BRepPrimAPI_MakePrism
    )
    from OCP.Geom import (
        Geom_Surface,
        Geom_Plane, 
        Geom_CylindricalSurface,
        Geom_SphericalSurface,
        Geom_ConicalSurface,
        Geom_ToroidalSurface,
        Geom_RectangularTrimmedSurface,
        Geom_Circle,
        Geom_Ellipse
    )
    from OCP.GeomLProp import GeomLProp_SLProps
    from OCP.gp import (
        gp_Pnt,
        gp_Vec,
        gp_Dir,
        gp_Ax2,
        gp_Ax3,
        gp_Pln
    )
    from OCP.TCollection import (
        TCollection_ExtendedString,
        TCollection_AsciiString
    )
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.TDF import (
        TDF_Label,
        TDF_LabelSequence
    )
    from OCP.TopAbs import (
        TopAbs_FORWARD,
        TopAbs_REVERSED,
        TopAbs_FACE,
        TopAbs_COMPOUND,
        TopAbs_SOLID,
        TopAbs_COMPSOLID,
        TopAbs_SHELL,
        TopAbs_EDGE,
        TopAbs_VERTEX,
        TopAbs_WIRE,
        TopAbs_SHAPE,
        TopAbs_Orientation
    )
    from OCP.TopExp import (
        TopExp_Explorer,
        TopExp
    )
    from OCP.TopLoc import TopLoc_Location

    from OCP.TopoDS import (
        TopoDS,
        TopoDS_Edge,
        TopoDS_Face,
        TopoDS_Solid,
        TopoDS_Shape
    )
    from OCP.TopTools import (
        TopTools_IndexedMapOfShape
    )
    from OCP.XCAFApp import (
        XCAFApp_Application
    )

    TOP_ABS_SHAPES: tuple[TopAbs_ShapeEnum, ...] = (TopAbs_COMPOUND,
                        TopAbs_COMPSOLID,
                        TopAbs_SOLID,
                        TopAbs_SHELL,
                        TopAbs_FACE,
                        TopAbs_WIRE,
                        TopAbs_EDGE,
                        TopAbs_VERTEX,
                        TopAbs_SHAPE)

    TOPODS_CAST_METHODS: dict[TopAbs_ShapeEnum, Callable[[TopoDS_Shape], TopoDS_Shape]] = {TopAbs_COMPOUND: TopoDS.Compound_s,
                                 TopAbs_COMPSOLID: TopoDS.CompSolid_s,
                                 TopAbs_SOLID: TopoDS.Solid_s,
                                 TopAbs_SHELL: TopoDS.Shell_s,
                                 TopAbs_FACE: TopoDS.Face_s,
                                 TopAbs_WIRE: TopoDS.Wire_s,
                                 TopAbs_EDGE: TopoDS.Edge_s,
                                 TopAbs_VERTEX: TopoDS.Vertex_s,
                                 TopAbs_SHAPE: lambda shape: shape}

except Exception as e:
    logger.warning(f'failed to load OCP')
    print(e)
    logger.warning(f'{e}')


from .decomposeparams import DecompositionParameters
from .geometric import (
    check_point_on_axis,
    check_vector_on_axis,
    check_vector_parallel_to_axis,
    calculate_plane_parameters_from_point_and_normal,
    angle_between_vectors,
    distance_between_points,
    values_within_tolerance,
    BoundingBox,
    OrientedBoundingBox
)
from .ocpbooleans import (
    _split_ocp_solid,
    generate_half_spaces_from_solid
)
from .ocpboundingbox import (
    bounding_box_from_shape,
)
from .ocpfilereadwrite import (
    extract_ocp_shape_type_from_shape,
    _get_ocp_surface_from_ocp_face,
    STEPFileWriter
)
from .ocpproperties import (
    _get_uv_bounds_from_ocp_faces,
    _get_uv_bounds_from_ocp_face,
    check_ocp_faces_connected
)

from .profile import function_timer
from .utils import check_folder_exists_for_file_path


def _ocp_faces_are_connected(face_1: TopoDS_Face, face_2: TopoDS_Face) -> bool:
    """Check whether two faces share at least one edge.

    Edges are compared by their Python hash of the OCP edge objects.

    Parameters
    ----------
    face_1 : TopoDS_Face
        First face.
    face_2 : TopoDS_Face
        Second face.

    Returns
    -------
    bool
        True if any edge of ``face_1`` has the same hash as an edge of ``face_2``.
    """
    edges_1 = extract_ocp_shape_type_from_shape(face_1, TopAbs_EDGE)
    edges_2 = extract_ocp_shape_type_from_shape(face_2, TopAbs_EDGE)

    for edge_1 in edges_1:
        for edge_2 in edges_2:
            if edge_1.__hash__() == edge_2.__hash__():
                return True
  
    return False

def _split_solid_on_connection_between_equal_radii_cylinders(solid: "TopoDS_Solid", decom_params: DecompositionParameters) -> list["TopoDS_Solid"]:
    """Split a solid where two connected cylinders of equal radius meet.

    For each pair of connected cylindrical faces with the same radius (relative
    tolerance 1e-6), a plane is built from the circular or elliptical edge they
    share and the solid is split with it. Resulting solids are split
    recursively.

    Parameters
    ----------
    solid : TopoDS_Solid
        The solid to pre-split.
    decom_params : DecompositionParameters
        The decomposition parameters.

    Returns
    -------
    list[TopoDS_Solid]
        The split solids, or ``[solid]`` if no split occurred.
    """

    # debug
    logger.debug(f'in _split_solid_on_connction_between_equal_radii_cylinder')

    TOL = 1e-6

    faces = extract_ocp_shape_type_from_shape(solid, TopAbs_FACE, cast_from_shape=True)
    faces_cylinders = [face for face in faces if isinstance(_get_ocp_surface_from_ocp_face(face), Geom_CylindricalSurface)]

    split_solids: list[TopoDS_Solid] = []

    for i, face in enumerate(faces_cylinders[:-1]):
        logger.info(f'face {i}')

        # get properties of face
        surface = _get_ocp_surface_from_ocp_face(face)
        cyl = surface.Cylinder()
        radius = cyl.Radius()

        for j, face_other in enumerate(faces_cylinders[i+1:]):
            # get properties of other face ramining in list
            surface_other = _get_ocp_surface_from_ocp_face(face_other)
            cyl_other = surface_other.Cylinder()
            radius_other = cyl_other.Radius()

            # check cylinders 
            same_radius = (radius * (1-TOL)) <= radius_other <= (radius * (1+TOL))
            faces_connected = check_ocp_faces_connected(face, face_other)
            
            if same_radius and faces_connected:
                # get teh common edges
                edges_common = get_common_ocp_edges_in_ocp_faces(face, face_other)
                edge = BRep_Tool.Curve_s(edges_common[0], 0.0, 0.0)

                curves = [BRep_Tool.Curve_s(edge, 0.0, 0.0) for edge in edges_common]
                curves_circular = [curve for curve in curves if isinstance(curve, Geom_Circle)]
                curves_ellipse = [curve for curve in curves if isinstance(curve, Geom_Ellipse)]

                # if isinstance(edge, Geom_Circle):
                if curves_circular:
                    edge = curves_circular[0]
                    # calculate plane parameters and generate a plane
                    circle: Geom_Circle = edge.Circ()
                    ax2: gp_Ax2 = circle.Position()
                    ax3: gp_Ax3 = gp_Ax3(ax2)
                    surface_plane = Geom_Plane(gp_Pln(ax3))
                    face_of_plane = BRepBuilderAPI_MakeFace(surface_plane, 0.0, 1.0, 0.0, 1.0, decom_params.make_face_tol).Face()

                    # split the solid using the new plane
                    split_solids_from_plane = _split_ocp_solid(solid, face_of_plane, decom_params)
                    
                    if len(split_solids_from_plane) == 1:
                        continue

                    elif len(split_solids_from_plane) > 1:
                        # recursively split the sub-split solids
                        split_solids = []
                        for split_solid in split_solids_from_plane:
                            sub_split_solids_from_plane = _split_solid_on_connection_between_equal_radii_cylinders(split_solid, decom_params)
                            split_solids.extend(sub_split_solids_from_plane)

                        if logger.level <= 10:
                            step_file_writer = STEPFileWriter(decom_params)
                            step_file_writer.add_solids(split_solids_from_plane, 'split_solids_from_plane')
                            step_file_writer.add_solids([solid], 'solid')
                            step_file_writer.add_solids(edges_common, 'edge_circular')
                            step_file_writer.add_solids([face_of_plane], 'face_of_plane')
                            output_path = step_file_writer.write(os.path.join(decom_params.debug_pipe_splitter), enumerate_files=True)
                            logger.debug(f'pipe split on circular edge - see {output_path}')

                        return split_solids
                
                # elif isinstance(edge, Geom_Ellipse):
                elif curves_ellipse:
                    # get the first elliptical curve
                    edge = curves_ellipse[0] 

                    elipse: Geom_Ellipse = edge.Elips()
                    ax2: gp_Ax2 = elipse.Position()
                    ax3: gp_Ax3 = gp_Ax3(ax2)
                    surface_plane = Geom_Plane(gp_Pln(ax3))
                    face_of_plane = BRepBuilderAPI_MakeFace(surface_plane, 0.0, 1.0, 0.0, 1.0, decom_params.make_face_tol).Face()

                    split_solids_from_plane = _split_ocp_solid(solid, face_of_plane, decom_params)
                    
                    if len(split_solids_from_plane) == 1:
                        continue
                    else:
                        split_solids = []
                        for split_solid in split_solids_from_plane:
                            sub_split_solids_from_plane = _split_solid_on_connection_between_equal_radii_cylinders(split_solid, decom_params)
                            split_solids.extend(sub_split_solids_from_plane)

                        if logger.level <= 10:
                            step_file_writer = STEPFileWriter(decom_params)
                            step_file_writer.add_solids(split_solids_from_plane, 'split_solids_from_plane')
                            step_file_writer.add_solids([solid], 'solid')
                            step_file_writer.add_solids(edges_common, 'edge_elliptical')
                            step_file_writer.add_solids([face_of_plane], 'face_of_plane')
                            output_path = step_file_writer.write(os.path.join(decom_params.debug_pipe_splitter), enumerate_files=True)
                            logger.debug(f'pipe split on elliptical edge - see {output_path}')

                        return split_solids 
                else:
                    logger.debug(f'connected pipes found, but edge is type {type(edge)}')

    return [solid]


def _split_solids_on_axis_by_circular_edges(solid: "TopoDS_Solid", decom_params: DecompositionParameters) -> list["TopoDS_Solid"]:
    """Split a solid on circular edges joining coaxial second order surfaces.

    Pre-splitting method which finds connected second order surfaces
    (cylinders, cones, spheres, tori) aligned along the same axis and connected
    by a circular edge whose normal is aligned with that axis. The solid is
    split by the plane of that edge and the resulting solids are split
    recursively.

    Parameters
    ----------
    solid : TopoDS_Solid
        The TopoDS_Solid on which the pre-splitting is to be applied.
    decom_params : DecompositionParameters
        The decomposition parmeters

    Returns
    -------
    list[TopoDS_Solid]
        A list of the split solids produced
    """

    # debug
    logger.debug(f'in _split_solid_on_connction_between_equal_radii_cylinder')

    # set tolerance
    ON_AXIS_TOL = 1e-2

    # get second order faces
    FACE_TYPES = (Geom_CylindricalSurface, Geom_ConicalSurface, Geom_SphericalSurface, Geom_ToroidalSurface)
    faces = extract_ocp_shape_type_from_shape(solid, TopAbs_FACE, cast_from_shape=True)
    faces_filtered = [face for face in faces if isinstance(_get_ocp_surface_from_ocp_face(face), FACE_TYPES)]

    split_solids: list[TopoDS_Solid] = []

    for i, face in enumerate(faces_filtered[:-1]):
        logger.info(f'face {i}')

        # get properties of face
        surface = _get_ocp_surface_from_ocp_face(face)
        ax = surface.Axis()
        direction = ax.Direction()

        for j, face_other in enumerate(faces_filtered[i+1:]):
            # get properties of other face remaining in list

            surface_other = _get_ocp_surface_from_ocp_face(face_other)
            ax_other = surface_other.Axis()
            direction_other = ax_other.Direction()

            angle_between_face_axis = direction.Angle(direction_other)

            if ((-ON_AXIS_TOL <= angle_between_face_axis < ON_AXIS_TOL) or (math.pi - ON_AXIS_TOL <= angle_between_face_axis <= math.pi + ON_AXIS_TOL)) and check_ocp_faces_connected(face, face_other):
                edges_common = get_common_ocp_edges_in_ocp_faces(face, face_other)
                edge = BRep_Tool.Curve_s(edges_common[0], 0.0, 0.0)

                curves = [BRep_Tool.Curve_s(edge, 0.0, 0.0) for edge in edges_common]
                curves_circular = [curve for curve in curves if isinstance(curve, Geom_Circle)]

                # for edge in curves_circular:

                # if isinstance(edge, Geom_Circle):
                if curves_circular:
                    # get the first circular edge
                    edge = curves_circular[0]

                    # calculate plane parameters and generate a plane
                    circle: Geom_Circle = edge.Circ()
                    ax2: gp_Ax2 = circle.Position()
                    ax3: gp_Ax3 = gp_Ax3(ax2)
                    circle_normal: gp_Dir = ax2.Direction()

                    angle_between_circle_normal_and_face_axis = direction.Angle(circle_normal)

                    if not ((-ON_AXIS_TOL <= angle_between_circle_normal_and_face_axis < ON_AXIS_TOL) or (math.pi - ON_AXIS_TOL <= angle_between_circle_normal_and_face_axis <= math.pi + ON_AXIS_TOL)):
                        # case where circle normal is not aligned with axis of coincident faces
                        continue

                    surface_plane = Geom_Plane(gp_Pln(ax3))
                    face_of_plane = BRepBuilderAPI_MakeFace(surface_plane, 0.0, 1.0, 0.0, 1.0, decom_params.make_face_tol).Face()

                    logger.debug(f'CONCENTRIL SPLITTER - edge found...')

                    # split the solid using the new plane
                    split_solids_from_plane = _split_ocp_solid(solid, face_of_plane, decom_params)

                    if len(split_solids_from_plane) <= 1:
                        # if logger.getEffectiveLevel() <= 10:
                        #     step_file_writer = STEPFileWriter(decom_params)
                        #     step_file_writer.add_solids(split_solids_from_plane, 'split_solids_from_plane')
                        #     step_file_writer.add_solids([solid], 'solid')
                        #     step_file_writer.add_solids(edges_common, 'edge_circular')
                        #     step_file_writer.add_solids([face_of_plane], 'face_of_plane')
                        #     output_path = step_file_writer.write(os.path.join('analysis', 'concentric_failed', 'concnetric_failed.stp'), enumerate_files=True)
                        #     logger.debug(f'pipe split on circular edge failed - see {output_path}')

                        continue

                    elif len(split_solids_from_plane) > 1:
                        # recursively split the sub-split solids
                        split_solids = []

                        for split_solid in split_solids_from_plane:
                            sub_split_solids_from_plane = _split_solids_on_axis_by_circular_edges(split_solid, decom_params)
                            split_solids.extend(sub_split_solids_from_plane)

                        # debugging - write solids to step file if split occured
                        # if logger.level <= 10:
                        #     step_file_writer = STEPFileWriter(decom_params)
                        #     step_file_writer.add_solids(split_solids_from_plane, 'split_solids_from_plane')
                        #     step_file_writer.add_solids([solid], 'solid')
                        #     step_file_writer.add_solids(edges_common, 'edge_circular')
                        #     step_file_writer.add_solids([face_of_plane], 'face_of_plane')
                        #     output_path = step_file_writer.write(os.path.join(decom_params.debug_concentric_splitter), enumerate_files=True)
                        #     logger.debug(f'pipe split on circular edge - see {output_path}')

                        return split_solids
    
    return [solid]

def _split_solids_by_rectilinear_grid(solid: TopoDS_Solid, decom_params: DecompositionParameters) -> None:
    """Split a solid on a rectilinear grid of planes (incomplete).

    Builds x-normal planes at multiples of
    ``decom_params.presplit_rectilinear_voxels_width`` inside the solid's
    bounding box and attempts to split the solid with each. The y and z planes
    are computed but not used, and nothing is returned.

    Parameters
    ----------
    solid : TopoDS_Solid
        The solid to split.
    decom_params : DecompositionParameters
        The decomposition parameters.
    """
    # get the bounding box from the solid
    bbox = bounding_box_from_shape(solid)

    # get parameters
    interval = decom_params.presplit_rectilinear_voxels_width

    # set up x, y and z intervals
    x_intervals = [k * interval for k in range(math.ceil(bbox.x_min / interval), math.floor(bbox.x_max / interval) + 1)]
    y_intervals = [k * interval for k in range(math.ceil(bbox.y_min / interval), math.floor(bbox.y_max / interval) + 1)]
    z_intervals = [k * interval for k in range(math.ceil(bbox.z_min / interval), math.floor(bbox.z_max / interval) + 1)]

    # split on x_planes
    x_planes: list[TopoDS_Face] = []
    for x_interval in x_intervals:
        point = gp_Pnt(x_interval, 0, 0)
        dir = gp_Dir(1, 0, 0)
        surface_plane = Geom_Plane(gp_Pln(point, dir))
        face_plane = BRepBuilderAPI_MakeFace(surface_plane, 0.0, 1.0, 0.0, 1.0, decom_params.make_face_tol).Face()
        x_planes.append(face_plane)

    # split all solids
    for x_plane in x_planes:
        split_solids = _split_ocp_solid(solid, x_plane)

def get_common_ocp_edges_in_ocp_faces(face_1: "TopoDS_Face", face_2: "TopoDS_Face") -> list["TopoDS_Edge"]:
    """Return the edges shared by two faces.

    Edges are mapped with ``TopExp.MapShapes_s`` and compared by TShape
    identity.

    Parameters
    ----------
    face_1 : TopoDS_Face
        First face.
    face_2 : TopoDS_Face
        Second face.

    Returns
    -------
    list[TopoDS_Edge]
        Edges present in both faces.
    """
    # set up explorer 
    exp = TopExp()

    # map edges in first face
    map_1 = TopTools_IndexedMapOfShape()
    exp.MapShapes_s(face_1, TopAbs_EDGE, map_1)

    # map edges in second face
    map_2 = TopTools_IndexedMapOfShape()
    exp.MapShapes_s(face_2, TopAbs_EDGE, map_2)

    common_edges: list[TopoDS_Shape] = []

    # compare by TShape identity
    for i in range(1, map_1.Extent() + 1):
        edge_1 = map_1.FindKey(i)
        if map_2.Contains(edge_1):
            common_edges.append(edge_1)
        
    # cast generic shape to edge
    common_edges_cast = [TopoDS.Edge_s(edge) for edge in common_edges]

    return common_edges_cast

def accumulate_location(label: TDF_Label, shape_tool: XCAFDoc_ShapeTool) -> TopLoc_Location:
    """Accumulate the location of a label and all of its ancestors.

    Walks up the label tree via ``Father()``, multiplying in each label's
    location and, where a label refers to another shape, that shape's
    accumulated location.

    Parameters
    ----------
    label : TDF_Label
        The XCAF label to start from.
    shape_tool : XCAFDoc_ShapeTool
        The shape tool of the XCAF document containing the label.

    Returns
    -------
    TopLoc_Location
        The combined location.
    """
    location = TopLoc_Location()

    while not label.IsNull():
        # Apply this label's location
        loc = shape_tool.GetLocation_s(label)
        if not loc.IsIdentity():
            location = loc.Multiplied(location)

        # If this label refers to another, recurse into that
        referred = TDF_Label()
        if shape_tool.GetReferredShape_s(label, referred) and not referred.IsNull():
            location = accumulate_location(referred, shape_tool).Multiplied(location)

        label = label.Father()

    return location



def get_name_from_ocp_label(label: TDF_Label) -> str:
    """Return the name attribute stored on an XCAF label.

    Parameters
    ----------
    label : TDF_Label
        The label to read the ``TDataStd_Name`` attribute from.

    Returns
    -------
    str
        The label name, or ``'NO NAME FOUND'`` if the label has no name.
    """
    sub_shape_name = TDataStd_Name()
    if (label.FindAttribute(TDataStd_Name.GetID_s(), sub_shape_name)):
        return TCollection_AsciiString(sub_shape_name.Get()).ToCString()
    else:
        return 'NO NAME FOUND'

def count_surface_by_type(solid: TopoDS_Solid) -> dict[type[Geom_Surface], int]:
    """Count the faces of a solid by underlying surface type.

    Parameters
    ----------
    solid : TopoDS_Solid
        The solid whose faces are counted.

    Returns
    -------
    dict[type[Geom_Surface], int]
        Number of faces for each of Geom_Plane, Geom_CylindricalSurface,
        Geom_SphericalSurface, Geom_ConicalSurface and Geom_ToroidalSurface.

    Raises
    ------
    Exception
        If a face has a surface type other than those listed above.
    """
    faces = extract_ocp_shape_type_from_shape(solid, TopAbs_FACE)
    surfaces = [_get_ocp_surface_from_ocp_face(face) for face in faces]

    faces_count = {Geom_Plane: 0,
                   Geom_CylindricalSurface: 0,
                   Geom_SphericalSurface: 0,
                   Geom_ConicalSurface: 0,
                   Geom_ToroidalSurface: 0}

    for surface in surfaces:
        surface_type = type(surface)
        if surface_type in faces_count:
            faces_count[surface_type] += 1
        else:
            msg = f'surface type not plane, cylinder, sphere, cone or tori. {type(surface)}'
            raise Exception(msg)

    return faces_count



# def _common_ocp_solid_alternate_trial(
#         shape_1: "TopoDS_Solid | TopoDS_CompSolid",
#         shape_2: "TopoDS_Solid | TopoDS_CompSolid | TopoDS_Face",
#         decom_params: DecompositionParameters,
#         reverse_face_orientation=False,
#         min_solids_required: int = 0,
#         max_solids_required: int = math.inf,
#         max_distance_from_face: float = 0.0,
#         solids_cleaned=False,
#         use_legacy=False) -> BooleanCommonSolidsResults:
#     """
#     method to find the common shape between a solid (shape_1) and either another solid or an unbound face (shape_2)
#     """

#     # check is shape_2 is a face, and if so build an infinite halfspace 
#     if shape_2.ShapeType() == TopAbs_FACE:
#         # generate half space to apply to split operations
#         tool = generate_ocp_half_space_from_face(shape_2, shape_1, decom_params, reverse_halfspace=reverse_face_orientation)
#     elif shape_2.ShapeType() == TopAbs_SOLID or shape_2.ShapeType() == TopAbs_COMPSOLID:
#         tool = shape_2
#     else:
#         msg = f'common_ocp_solid() method requires a solid and either a solid, compsolid or face, not {type(shape_2)}'
#         raise Exception(msg)
    
#     # set up flag to check failures
#     common_failed = False

#     # apply common boolean operation
#     if not use_legacy:
#         # perform BOP common operations
#         bop = BOPAlgo_BOP()
#         bop.AddArgument(shape_1)
#         bop.AddTool(tool)
#         bop.SetOperation(BOPAlgo_COMMON)
#         bop.SetFuzzyValue(decom_params.bool_common_fuzzy_value)
#         bop.SetNonDestructive(True)
#         bop.Perform()
#         common_solids = extract_ocp_shape_type_from_shape(bop.Shape(), TopAbs_SOLID, cast_from_shape=True)

#         # check erros
#         if bop.HasErrors():
#             # set common solids to input solid
#             common_solids = [shape_1]
#             common_failed = True

#             # get errors
#             buffer_errors = BytesIO()
#             bop.DumpWarnings(buffer_errors)
#             error_strings = buffer_errors.getvalue().decode("utf-8", errors="replace").split()

#             # write error solids to analysis folder for debugging            
#             step_file_writer = STEPFileWriter(decom_params)
#             step_file_writer.add_solids([shape_1], 'shape_1')
#             step_file_writer.add_solids([shape_2], 'shape_2')
#             step_file_writer.add_solids([tool], 'tool')
#             step_file_writer.add_solids(common_solids, 'common_solids')
#             output_path = step_file_writer.write(decom_params.debug_common_error_path, enumerate_files=True)

#             # output error message to logger
#             for i, error_string in enumerate(error_strings, start=1):
#                 logger.warning(f'BOPAlgo_COMMON HasErrors {i} = {error_string} - see {output_path}')
        
#         # check warnings
#         if bop.HasWarnings():
#             # get warnings
#             buffer_warnings = BytesIO()
#             bop.DumpWarnings(buffer_warnings)
#             warning_strings = buffer_warnings.getvalue().decode("utf-8", errors="replace").split()

#             # write warnings solids to analysis folder for debugging            
#             step_file_writer = STEPFileWriter(decom_params)
#             step_file_writer.add_solids([shape_1], 'shape_1')
#             step_file_writer.add_solids([shape_2], 'shape_2')
#             step_file_writer.add_solids([tool], 'tool')
#             step_file_writer.add_solids(common_solids, 'common_solids')
#             output_path = step_file_writer.write(decom_params.debug_common_warning_path, enumerate_files=True)

#             # output warning message to logger
#             for i, warning_string in enumerate(warning_strings, start=1):
#                 logger.error(f'BOPAlgo_COMMON HasWarnings {i} = {warning_string} - see {output_path}')

#     else:
#         # legacy facllback for problem solids
#         brep_common = BRepAlgoAPI_Common(shape_1, tool)
#         brep_common.SetFuzzyValue(1e-4)
#         brep_common.SetNonDestructive(True)
#         brep_common.Build()
#         common_solids = extract_ocp_shape_type_from_shape(brep_common.Shape(), TopAbs_SOLID, cast_from_shape=True)

#         # case where all efforts have failed, most of the time the original solid lies within the solid, so try allowing this to continue
#         if len(common_solids) == 0:
#             common_solids = [shape_1]
#             common_failed = True

#         logger.warning(f'BRepAlgoAPI_Common - legacy used - {len(common_solids)} solids produced')

#     # check that the required number of common solids have been produced
#     if not (min_solids_required <= len(common_solids) <= max_solids_required) and not solids_cleaned:
#         # case 1 - common failed, try to clean up and apply common operation again
#         logger.warning(f'BOPAlgo_COMMON failed to produce required number of solids - {len(common_solids)} produced but between {min_solids_required} and {max_solids_required} required')

#         # clean solids
#         shape_1 = _clean_solid(shape_1, decom_params, operation_label=BOPAlgo_COMMON)
#         tool = _clean_solid(tool, decom_params, operation_label=BOPAlgo_COMMON)

#         # reduce fuzzy value
#         decom_params_REDUCED_FUZZY = copy.deepcopy(decom_params)
#         decom_params_REDUCED_FUZZY.bool_common_fuzzy_value *1e2

#         # re-try common operation following checks and cleanup
#         common_solids_result = _common_ocp_solid(
#             shape_1,
#             tool,
#             decom_params_REDUCED_FUZZY,
#             min_solids_required=min_solids_required,
#             max_solids_required=max_solids_required,
#             solids_cleaned=True)
        
#         common_solids = common_solids_result.solids
#         solids_cleaned = common_solids_result.solids_cleaned
#         use_legacy = common_solids_result.legacy_used

#         logger.warning(f'BOPAlgo_COMMON - re-try using cleaned shape_1 and tool produced {len(common_solids)} common_solids')

#     elif not (min_solids_required <= len(common_solids) <= max_solids_required) and solids_cleaned and not use_legacy:
#         # output to logger
#         logger.warning(f'BOPAlgo_COMMON failed to produce required number of solids following cleanup - {len(common_solids)} produced but between {min_solids_required} and {max_solids_required} required')

#         # set use legacy flag, required for output of common result
#         use_legacy = True

#         # re-try common operation following checks and cleanup
#         common_solids_result = _common_ocp_solid(
#             shape_1, 
#             tool, 
#             decom_params,
#             min_solids_required=min_solids_required,
#             max_solids_required=max_solids_required,
#             solids_cleaned=solids_cleaned,
#             use_legacy=use_legacy)

#         # get common solids from result and set flags used        
#         common_solids = common_solids_result.solids
#         solids_cleaned = common_solids_result.solids_cleaned
#         use_legacy = common_solids_result.legacy_used

#         logger.warning(f'BOPAlgo_COMMON - re-try using cleaned shape_1 and tool and legacy produced {len(common_solids)} common_solids')

#     elif not (min_solids_required <= len(common_solids) <= max_solids_required) and solids_cleaned and use_legacy:
#         # output to logger
#         logger.warning(f'BOPAlgo_COMMON failed to produce required number of solids following cleanup and legacy method - {len(common_solids)} produced but between {min_solids_required} and {max_solids_required} required')

#         step_writer = STEPFileWriter(decom_params)
#         step_writer.add_solids([shape_1], 'shape_1')
#         step_writer.add_solids([shape_2], 'shape_2')
#         step_writer.add_solids([tool], 'tool')
#         step_writer.add_solids(common_solids, 'common_solids')
#         output_path = step_writer.write(decom_params.debug_common_fatal_path, enumerate_files=True)
        
#         msg = f'BOPAlgo_COMMON - failed to produce required number of solids, {len(common_solids)} produced, but {min_solids_required} - {max_solids_required} required - see {output_path}'
#         logger.fatal(msg)

#         raise Exception(msg)
    
#     # check for degenrate solids
#     # common_solids = _filter_degenerate_thin_solids_2(shape_1, common_solids, decom_params, BOPAlgo_COMMON)

#     logger.fatal(f'common_failed = {common_failed}')

#     # return common solids
#     result = BooleanCommonSolidsResults(
#         common_solids,
#         decom_params.bool_common_fuzzy_value, 
#         decom_params.cleanup_shape_fix_tol_min,
#         decom_params.cleanup_shape_fix_tol_max,
#         failed=common_failed)
    
#     return result


# def _filter_degenerate_thin_solids(solids: list["TopoDS_Solid"], decom_params: DecompositionParameters):
#     solids_passed = []
#     solids_failed = []

#     for solid in solids:
#         if _volume_of_ocp_solids([solid]) / _surface_area_of_ocp_solids([solid]) > decom_params.SPLIT_MIN_AREA_TO_VOLUME_RATIO:
#             solids_passed.append(solid)
#         else:
#             solids_failed.append(solid)

#     return solids_passed, solids_failed



def _ocp_face_closed(face: TopoDS_Face, decom_params: DecompositionParameters) -> bool:
    """Check whether any edge of a face is closed.

    Parameters
    ----------
    face : TopoDS_Face
        The face to check.
    decom_params : DecompositionParameters
        The decomposition parameters (unused).

    Returns
    -------
    bool
        True if any edge in any wire of the face is closed.
    """
    for wire in extract_ocp_shape_type_from_shape(face, TopAbs_WIRE, cast_from_shape=True):
        for edge in extract_ocp_shape_type_from_shape(wire, TopAbs_EDGE, cast_from_shape=True):
            if edge.Closed():
                return True

    return False




def _generate_ocp_half_space_from_face_plane(face: "TopoDS_Face", bounding_solid: "TopoDS_Solid", decom_params: DecompositionParameters) -> None:
    """Generate a half space from a plane face (not implemented).

    Parameters
    ----------
    face : TopoDS_Face
        The face to generate the half space from.
    bounding_solid : TopoDS_Solid
        The solid the half space should cover.
    decom_params : DecompositionParameters
        The decomposition parameters.
    """
    pass

def _generate_ocp_half_space_from_face_cylinder(face: "TopoDS_Face", bounding_solid: "TopoDS_Solid", decom_params: DecompositionParameters) -> None:
    """Generate a half space from a cylinder face (not implemented).

    Parameters
    ----------
    face : TopoDS_Face
        The face to generate the half space from.
    bounding_solid : TopoDS_Solid
        The solid the half space should cover.
    decom_params : DecompositionParameters
        The decomposition parameters.
    """
    pass

def _generate_ocp_half_space_from_face_sphere(face: "TopoDS_Face", bounding_solid: "TopoDS_Solid", decom_params: DecompositionParameters) -> None:
    """Generate a half space from a sphere face (not implemented).

    Parameters
    ----------
    face : TopoDS_Face
        The face to generate the half space from.
    bounding_solid : TopoDS_Solid
        The solid the half space should cover.
    decom_params : DecompositionParameters
        The decomposition parameters.
    """
    pass
    
def _generate_ocp_half_space_from_face_cone(face: "TopoDS_Face", bounding_solid: "TopoDS_Solid", decom_params: DecompositionParameters) -> None:
    """Generate a half space from a cone face (not implemented).

    Parameters
    ----------
    face : TopoDS_Face
        The face to generate the half space from.
    bounding_solid : TopoDS_Solid
        The solid the half space should cover.
    decom_params : DecompositionParameters
        The decomposition parameters.
    """
    pass

def _generate_ocp_half_space_from_face_tori(face: "TopoDS_Face", bounding_solid: "TopoDS_Solid", decom_params: DecompositionParameters) -> None:
    """Generate a half space from a torus face (not implemented).

    Parameters
    ----------
    face : TopoDS_Face
        The face to generate the half space from.
    bounding_solid : TopoDS_Solid
        The solid the half space should cover.
    decom_params : DecompositionParameters
        The decomposition parameters.
    """
    pass

def _make_quasi_unbound_face_from_surface(face: "TopoDS_Face", decom_params: DecompositionParameters) -> "TopoDS_Face":
    """Build a quasi-unbound face from the surface underlying a face.

    A cylinder held in a Geom_RectangularTrimmedSurface is rebuilt over
    u in [0, 2*pi] and v in [-1e4, 1e4] with the original face orientation;
    other surfaces are rebuilt with their natural bounds.

    Parameters
    ----------
    face : TopoDS_Face
        The (bounded) face.
    decom_params : DecompositionParameters
        The decomposition parameters, ``make_face_tol`` is used.

    Returns
    -------
    TopoDS_Face
        The quasi-unbound face.

    Raises
    ------
    Exception
        If the untrimmed surface is still a Geom_RectangularTrimmedSurface.
    """
    
    # get surface from face
    surface = _get_ocp_surface_from_ocp_face(face, remove_trimmed_surface=False)
    surface_untrimmed = _get_ocp_surface_from_ocp_face(face, remove_trimmed_surface=True)

    if type(surface) == Geom_RectangularTrimmedSurface and type(surface_untrimmed) == Geom_CylindricalSurface:
        # case where a cylinder is bound to a rectangular trimmed surface
        u_min, u_max = 0.0, math.tau
        v_min, v_max = -1e4, 1e4
        face_unbound = BRepBuilderAPI_MakeFace(surface_untrimmed, u_min, u_max, v_min, v_max, decom_params.make_face_tol).Face()

        face_unbound.Orientation(face.Orientation())

    elif type(surface_untrimmed) == Geom_RectangularTrimmedSurface:
        # case shere
        msg = f'surface type is Geom_RectangularTrimmedSurface but not cylinder, not yet coded...'
        raise Exception(msg)
    else:
        face_unbound = BRepBuilderAPI_MakeFace(surface, decom_params.make_face_tol).Face()

    return face_unbound

def move_point_along_vector(p: "gp_Pnt", v: "gp_Vec", l1: float) -> "gp_Pnt":
    """Translate a point a given distance along a vector direction.

    Parameters
    ----------
    p : gp_Pnt
        The starting point.
    v : gp_Vec
        The direction (normalised internally, the input is not modified).
    l1 : float
        Distance to move along ``v``.

    Returns
    -------
    gp_Pnt
        The translated point.
    """
    v2 = gp_Vec(v)      # copy so we don't modify the original
    v2.Normalize()      # make it unit length
    v2.Scale(l1)        # scale to desired length
    return gp_Pnt(p.XYZ() + v2.XYZ())

def point_at_intersection_of_two_lines(point_1: "gp_Pnt", vec_1: "gp_Vec", point_2: "gp_Pnt", vec_2: "gp_Vec", tol: float=1e-12) -> gp_Pnt:
    """Calculate the intersection point of two lines in 3D.

    Each line is defined by a point and a direction vector.

    Parameters
    ----------
    point_1 : gp_Pnt
        A point on the first line.
    vec_1 : gp_Vec
        Direction of the first line.
    point_2 : gp_Pnt
        A point on the second line.
    vec_2 : gp_Vec
        Direction of the second line.
    tol : float, optional
        Tolerance on squared magnitudes used for the parallel and on-line
        checks, by default 1e-12

    Returns
    -------
    gp_Pnt
        The intersection point.

    Raises
    ------
    ValueError
        If the lines are (nearly) parallel, in which case spheres at both
        points are written to ``analysis/points.stp``, or if the lines do not
        intersect.
    """
    # vector between points
    vec_between_points = gp_Vec(point_1, point_2)

    # cross products to detect parallel vectors
    cross = vec_1.Crossed(vec_2)
    cross_normalised_sq = cross.SquareMagnitude()

    if cross_normalised_sq < tol:
        msg = 'point_at_intersection_of_two_lines'
        sphere_p1 = BRepPrimAPI_MakeSphere(point_1, 1.0).Shape()
        sphere_p2 = BRepPrimAPI_MakeSphere(point_2, 1.0).Shape()
        
        step_file_writer = STEPFileWriter(DecompositionParameters())
        step_file_writer.add_solids([sphere_p1], 'point_1_normal_')
        step_file_writer.add_solids([sphere_p2], 'point_2')
        output_path = step_file_writer.write('analysis/points.stp')

        logger.fatal(f'point_at_intersection_of_two_lines(p1, v1, p2, v2, tol)')
        logger.fatal(f'p1 = {point_1.X():.2e} {point_1.Y():.2e} {point_1.Y():.2e}')
        logger.fatal(f'p2 = {point_2.X():.2e} {point_2.Y():.2e} {point_2.Y():.2e}')
        logger.fatal(f'v1 = {vec_1.X():.2e} {vec_1.Y():.2e} {vec_1.Z():.2e}')
        logger.fatal(f'v2 = {vec_2.X():.2e} {vec_2.Y():.2e} {vec_2.Z():.2e}')

        raise ValueError(f'Lines are parallel or nearly parallel - see {output_path}')
    
    # solve for t using triple product
    t = vec_between_points.Crossed(vec_2).Dot(cross) / cross_normalised_sq

    # calculate intersection point
    point_intersection = gp_Pnt(
        point_1.X() + t * vec_1.X(),
        point_1.Y() + t * vec_1.Y(),
        point_1.Z() + t * vec_1.Z()
    )

    # check if the point lies on line 2 (vec_2 with point_2)
    checK_point_on_line_2 = gp_Vec(point_2, point_intersection)
    if checK_point_on_line_2.Crossed(vec_2).SquareMagnitude() > tol:
        msg = f'point at intersection of lines failed, lines do not intersect'
        raise ValueError(msg)

    return point_intersection

def distance_of_point_from_axis(axis: "gp_Ax3", point: "gp_Pnt") -> float:
    """Calculate the orthogonal distance between a point and a gp_Ax3 axis.

    Parameters
    ----------
    axis : gp_Ax3
        An axis, representing a line in 3d space.
    point : gp_Pnt
        A point at some distance from the axis.

    Returns
    -------
    float
        The orthoginal distance beteen the point and the axis provided.
    """
    
    origin = axis.Location()
    direction = axis.Direction()

    # produce a vector from the axis origin to the point
    vec_ax_or_to_point = gp_Vec(origin, point)

    # calculate the cross product, the magnitude of which gives us the otthogonal distance from the axis
    cross = vec_ax_or_to_point.Crossed(gp_Vec(direction))

    # return the distance 
    return cross.Magnitude()

    
    

def flatten_shapes(shape_tool: XCAFDoc_ShapeTool) -> list[TopoDS_Shape]:
    """Flatten an XCAF assembly into a list of located leaf shapes.

    Parameters
    ----------
    shape_tool : XCAFDoc_ShapeTool
        The shape tool of the XCAF document.

    Returns
    -------
    list[TopoDS_Shape]
        All non-assembly shapes, each moved to its accumulated location.
    """
    free_shape_labels = TDF_LabelSequence()
    shape_tool.GetFreeShapes(free_shape_labels)
    flat_shapes: list[TopoDS_Shape] = []

    def recurse(label: TDF_Label, parent_loc: TopLoc_Location) -> None:
        """Collect leaf shapes below a label, accumulating locations.

        Parameters
        ----------
        label : TDF_Label
            The current label.
        parent_loc : TopLoc_Location
            Location accumulated from the parent labels.
        """
        shape = shape_tool.GetShape_s(label)
        loc = shape_tool.GetLocation_s(label)
        full_loc = loc.Multiplied(parent_loc)
        transformed_shape = shape.Located(full_loc)

        if shape_tool.IsAssembly_s(label):
            children = TDF_LabelSequence()
            shape_tool.GetComponents_s(label, children, False)
            for child in children:
                recurse(child, full_loc)
        else:
            flat_shapes.append(transformed_shape)

    for label in free_shape_labels:
        recurse(label, TopLoc_Location())

    return flat_shapes

def _get_normal_at_uv_param_on_ocp_surface(surface: "Geom_Surface", u: float, v: float, tol: float=1e-6) -> gp_Dir:
    """Return the surface normal at a (u, v) parameter.

    Parameters
    ----------
    surface : Geom_Surface
        The surface to evaluate.
    u : float
        The u parameter.
    v : float
        The v parameter.
    tol : float, optional
        Tolerance (unused, 1e-6 is hard coded), by default 1e-6

    Returns
    -------
    gp_Dir
        The normal of the surface at (u, v).

    Raises
    ------
    Exception
        If no normal is defined at (u, v).
    """
    props = GeomLProp_SLProps(surface, u, v, 1, 1e-6)
    if props.IsNormalDefined():
        return props.Normal()
    else:
        surface_type = type(surface)
        msg = f'no normal defined at u ({u:.2f}) v ({v:.2f}) param on surface {surface_type}'
        raise Exception(msg)

def _get_normal_at_uv_param_on_ocp_face(face: TopoDS_Face, u: float, v: float, tol: float=1e-6) -> gp_Dir:
    """Return the face normal at a (u, v) parameter, accounting for orientation.

    The surface normal is returned unchanged for a TopAbs_REVERSED face and
    reversed otherwise.

    Parameters
    ----------
    face : TopoDS_Face
        The face to evaluate.
    u : float
        The u parameter.
    v : float
        The v parameter.
    tol : float, optional
        Tolerance (unused, 1e-6 is hard coded), by default 1e-6

    Returns
    -------
    gp_Dir
        The normal at (u, v).

    Raises
    ------
    Exception
        If no normal is defined at (u, v).
    """
    # get geom surface from face
    surface = _get_ocp_surface_from_ocp_face(face)

    # get teh properties of the surface at the uv params provided
    props = GeomLProp_SLProps(surface, u, v, 1, 1e-6)

    # return the normal based on orientation of the face
    if not props.IsNormalDefined():
        surface_type = type(surface)
        msg = f'no normal defined at u ({u:.2f}) v ({v:.2f}) param on surface {surface_type}'
        raise Exception(msg)
    elif face.Orientation() == TopAbs_REVERSED:
        return props.Normal()
    else:
        return props.Normal().Reversed()

    # if props.IsNormalDefined() and face.Orientation() == TopAbs_FORWARD:
    #     return props.Normal()
    # elif props.IsNormalDefined() and face.Orientatoin() == TpoAbs_
    #     pass
    # else:


def _extrude_ocp_face(faces: list["TopoDS_Face"], length: float, decom_params: DecompositionParameters) -> list[TopoDS_Shape]:
    """Extrude faces along the normal of the first face's surface.

    The normal is evaluated at the mid (u, v) point of the combined bounds.

    Parameters
    ----------
    faces : list[TopoDS_Face]
        The faces to extrude.
    length : float
        Extrusion length (currently unused).
    decom_params : DecompositionParameters
        The decomposition parameters (unused).

    Returns
    -------
    list[TopoDS_Shape]
        One prism per face.
    """
    u_min, u_max, v_min, v_max = _get_uv_bounds_from_ocp_faces(faces)
    u_mid = (u_min + u_max) / 2.0
    v_mid = (u_min + v_min) / 2.0

    surface = _get_ocp_surface_from_ocp_face(faces[0])

    normal = _get_normal_at_uv_param_on_ocp_surface(surface, u_mid, v_mid)
    
    prisms = [BRepPrimAPI_MakePrism(face, normal).Shape() for face in faces]

    return prisms


def show_ocp_solids_in_cq(solids: list[TopoDS_Solid | TopoDS_CompSolid]) -> None:
    """Show OCP solids in the Jupyter CadQuery viewer.

    This requires the mcio.load_jupyter_cadquery() method to be run to load the
    optional package.

    Parameters
    ----------
    solids : list[TopoDS_Solid | TopoDS_CompSolid]
        The solids/compsolids to display.
    """

    # cast OCP shapes to CQ shapes
    solids_cq: list[cq.Shape] = [cq.Shape.cast(s) for s in solids]

    # add cq shapes to workplane
    work_plane = cq.Workplane(solids_cq[0])
    for solid_cq in solids_cq[1:]:
        work_plane.add(solid_cq)

    # show workplane in jupyterlab
    show(work_plane)

def check_solids_intersect(shape_1: "TopoDS_Shape", shape_2: "TopoDS_Shape") -> bool:
    """Check whether two shapes intersect using a section operation.

    Parameters
    ----------
    shape_1 : TopoDS_Shape
        First shape.
    shape_2 : TopoDS_Shape
        Second shape.

    Returns
    -------
    bool
        False if the section fails, otherwise True if the section shape is
        not null.
    """
    
    section_algo = BRepAlgoAPI_Section(shape_1, shape_2, False)
    section_algo.Build()

    if not section_algo.IsDone():
        return False
    
    # if the generated section returns any edges, then intersection has been found
    intersection_shape = section_algo.Shape()
   
    # return check for generating a valid shape from intersection
    return not intersection_shape.IsNull()

def check_solids_intersect_boolean(shape_1: "TopoDS_Shape", shape_2: "TopoDS_Shape") -> bool:
    """Check whether two shapes intersect using a boolean common operation.

    Checks if a common shape exists and contains faces or, failing that, edges.

    Parameters
    ----------
    shape_1 : TopoDS_Shape
        First shape.
    shape_2 : TopoDS_Shape
        Second shape.

    Returns
    -------
    bool
        True if the common shape contains at least one face or edge.
    """
    # 1. Perform the lightweight Boolean Common operation
    common_algo = BRepAlgoAPI_Common(shape_1, shape_2)
    common_algo.Build()
    
    if not common_algo.IsDone():
        return False
        
    # If the common shape is completely Null, they definitely don't intersect
    res_shape = common_algo.Shape()
    if res_shape.IsNull():
        return False
        
    # 2. FIX: Explore for FACES (volume/surface overlap)
    face_explorer = TopExp_Explorer(res_shape, TopAbs_FACE)
    if face_explorer.More():
        return True
        
    # 3. FALLBACK: Explore for EDGES (in case it perfectly grazes a vertex/line)
    edge_explorer = TopExp_Explorer(res_shape, TopAbs_EDGE)
    return edge_explorer.More()

def convert_ocp_shape_to_binary(shape: "TopoDS_Shape") -> bytes:
    """Serialise an OCP shape to BRep binary data.

    Parameters
    ----------
    shape : TopoDS_Shape
        The shape to serialise.

    Returns
    -------
    bytes
        The shape written with ``BRepTools.Write_s``.
    """
    
    # convert shape to binary
    buffer: io.BytesIO = io.BytesIO()
    BRepTools.Write_s(shape, buffer)
    buffer.seek(0)

    # return buffer
    return buffer.getvalue()

def convert_ocp_surface_to_binary(surface: Geom_Surface) -> bytes:
    """Serialise an OCP surface to binary by storing it in a face.

    A face bounded to u, v in [-10, 10] is built on the surface and serialised.

    Parameters
    ----------
    surface : Geom_Surface
        The surface to serialise.

    Returns
    -------
    bytes
        The binary BRep representation of the face.
    """
    # store surface in generic face
    face: TopoDS_Face = BRepBuilderAPI_MakeFace(surface, -10, 10, -10, 10, 1e-6).Face()

    # return buffer with face as binary
    return convert_ocp_shape_to_binary(face)

def convert_ocp_orientation_to_binary(orientation: "TopAbs_Orientation") -> str:
    """Convert an OCP orientation to a string for serialisation.

    Parameters
    ----------
    orientation : TopAbs_Orientation
        TopAbs_FORWARD or TopAbs_REVERSED.

    Returns
    -------
    str
        ``'forward'`` or ``'revereversed'``.

    Raises
    ------
    TypeError
        If the orientation is neither TopAbs_FORWARD nor TopAbs_REVERSED.
    """
    if orientation == TopAbs_FORWARD:
        return 'forward'
    elif orientation == TopAbs_REVERSED:
        return 'revereversed'
    else:
        msg = f'orientation should only be TopAbs_FORWARD or TopAbs_REVERSED'
        logger.fatal(msg)
        raise TypeError(msg)

def convert_ocp_doc_to_binary(doc: "TDocStd_Document") -> bytes:
    """Serialise an OCP document to binary using the XCAF application.

    Parameters
    ----------
    doc : TDocStd_Document
        The document to serialise.

    Returns
    -------
    bytes
        The saved document.
    """
    app: XCAFApp_Application = XCAFApp_Application.GetApplication_s()
    
    # Set up an in-memory stream
    buffer = io.BytesIO()

    # SaveAs writes directly into the BytesIO buffer
    status = app.SaveAs(doc, buffer)
    buffer.seek(0)

    return buffer.getvalue()

def convert_shape_binary_to_ocp_shape(buffer: bytes) -> "TopoDS_Shape":
    """Deserialise BRep binary data back to an OCP shape.

    Parameters
    ----------
    buffer : bytes
        Binary data produced by :func:`convert_ocp_shape_to_binary`.

    Returns
    -------
    TopoDS_Shape
        The deserialised shape.

    Raises
    ------
    Exception
        If the deserialised shape is null.
    """

    stream = io.BytesIO(buffer)

    # convert binary to shape
    builder = BRep_Builder()
    shape = TopoDS_Shape()
    BRepTools.Read_s(shape, stream, builder)

    if shape.IsNull():
        msg = f'shape from deserialise is null'
        raise Exception(msg)
    else:
        return shape

def convert_surface_binary_to_ocp_surface(buffer: bytes) -> "Geom_Surface":
    """Deserialise binary data back to an OCP surface.

    Parameters
    ----------
    buffer : bytes
        Binary data produced by :func:`convert_ocp_surface_to_binary`.

    Returns
    -------
    Geom_Surface
        The surface of the deserialised face.
    """
    # convert binary to face
    face = convert_shape_binary_to_ocp_shape(buffer)

    # extract surface from face
    surface = _get_ocp_surface_from_ocp_face(face)

    # return surface
    return surface

def convert_orientation_binary_to_ocp_orientation(buffer: str) -> TopAbs_Orientation:
    """Convert a serialised orientation string back to an OCP orientation.

    Parameters
    ----------
    buffer : str
        String produced by :func:`convert_ocp_orientation_to_binary`.

    Returns
    -------
    TopAbs_Orientation
        TopAbs_FORWARD if ``buffer == 'forward'``, otherwise TopAbs_REVERSED.
    """
    if buffer == 'forward':
        return TopAbs_FORWARD
    else:
        return TopAbs_REVERSED

def convert_binary_to_ocp_doc(buffer: bytes) -> "TDocStd_Document":
    """Deserialise binary data back to an OCP document.

    Parameters
    ----------
    buffer : bytes
        Binary data produced by :func:`convert_ocp_doc_to_binary`.

    Returns
    -------
    TDocStd_Document
        The loaded document.
    """
    # set up buffer
    buffer = io.BytesIO(buffer)

    # get application
    app: XCAFApp_Application = XCAFApp_Application.GetApplication_s()
    
    # Initialize an empty document
    doc = TDocStd_Document(TCollection_ExtendedString('temp_doc_name'))
    
    # Ensure buffer is at the beginning
    buffer.seek_s(0)
    
    # load from the binary byte stream
    status = app.Open(buffer, doc)
    
    return doc

# def save_to_cad(instance: Model | Cell, decom_params: DecompositionParameters=None) -> str:
#     # set up decompose parameters 
#     decom_params = decom_params if decom_params else DecompositionParameters()

#     # collect all cells from either Model or Cell objects
#     if isinstance(instance, Model):
#         cells = instance.cells
#     elif isinstance(instance, Cell):
#         cells = [instance]
#     else:
#         msg = f'save_to_cad only accepts Model or Cell objects ,' +\
#               f'not {type(instance)}'
#         logger.fatal(msg)
#         raise TypeError(msg)

#     # store solids extracted into step file writer
#     step_file_writer = STEPFileWriter()

#     for cell in cells:
#         cell._volume


def generate_csg_equivalent_quasi_infinite_solid_from_solid(solid: "TopoDS_Solid", decom_params: DecompositionParameters) -> list["TopoDS_Solid"]:
    """Build the CSG equivalent of a solid from its half spaces.

    The half spaces of the solid are intersected in turn with a large box
    (half width 1e8) approximating infinite space.

    Parameters
    ----------
    solid : TopoDS_Solid
        The solid to convert.
    decom_params : DecompositionParameters
        The decomposition parameters, ``bool_common_fuzzy_value`` is used.

    Returns
    -------
    list[TopoDS_Solid]
        Solids of the common shape, or an empty list if a common fails.
    """
    # generate half spaces from solid
    half_spaces = generate_half_spaces_from_solid(solid, decom_params)

    # generate large volume to approximate an infinite surface
    BOX_EXTENT = 1e8
    BOX_WIDTH = BOX_EXTENT * 2.0
    box_origin = gp_Pnt(-BOX_EXTENT, -BOX_EXTENT, -BOX_EXTENT)
    quasi_infinite_box = BRepPrimAPI_MakeBox(box_origin, BOX_WIDTH, BOX_WIDTH, BOX_WIDTH).Shape()

    # take the common space from the 
    common = BRepAlgoAPI_Common(quasi_infinite_box, half_spaces[0])
    common.SetFuzzyValue(decom_params.bool_common_fuzzy_value)
    common.SetNonDestructive(True)
    common_shape = common.Shape()

    if common_shape.IsNull():
        return []

    for half_space in half_spaces[1:]:
        common = BRepAlgoAPI_Common(common_shape, half_space)
        common.SetFuzzyValue(decom_params.bool_common_fuzzy_value)
        common.SetNonDestructive(True)
        common_shape = common.Shape()

        # case where the common operation has failed
        if common_shape.IsNull():
            return []

    common_solids = extract_ocp_shape_type_from_shape(common_shape, TopAbs_SOLID)

    return common_solids

