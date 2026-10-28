"""Boolean operations and half space generation on OCP solids.

Contains result dataclasses for boolean operations, wrappers around the
OpenCASCADE BOPAlgo cut, common, fuse and splitter algorithms (with error and
warning handling and debug STEP output), generation of half space solids from
faces, recursive splitting of solids by their own half spaces, and helpers for
sorting and grouping faces by surface type, coincidence, connectivity and area.
"""

from __future__ import annotations

import math
from io import BytesIO
from dataclasses import dataclass
import os
from mcio import logger
import copy
from collections import OrderedDict, deque
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

if TYPE_CHECKING:
    from OCP.TopAbs import TopAbs_ShapeEnum


try:
    import casadi
    import OCP
    from OCP.BRepAlgoAPI import (
        BRepAlgoAPI_Common,
        BRepAlgoAPI_Cut
    )
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeVertex,
        BRepBuilderAPI_Transform
    )
    from OCP.BRepGProp import (
        BRepGProp
    )
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepPrimAPI import (
        BRepPrimAPI_MakeHalfSpace,
        BRepPrimAPI_MakeCylinder,
        BRepPrimAPI_MakeCone,
        BRepPrimAPI_MakeTorus
    )
    from OCP.BOPAlgo import (
        BOPAlgo_BOP,
        BOPAlgo_CUT,
        BOPAlgo_COMMON,
        BOPAlgo_FUSE,
        BOPAlgo_Splitter
    )
    from OCP.Geom import (
        Geom_Surface,
        Geom_Plane, 
        Geom_CylindricalSurface,
        Geom_SphericalSurface,
        Geom_ConicalSurface,
        Geom_ToroidalSurface
    )
    from OCP.GeomLProp import GeomLProp_SLProps
    from OCP.gp import (
        gp_Pnt,
        gp_Vec,
        gp_Dir,
        gp_Ax1,
        gp_Ax2,
        gp_Cone,
        gp_Trsf,
        gp_Ax3
    )
    from OCP.GProp import (
        GProp_GProps
    )
    from OCP.TDocStd import TDocStd_Document
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
        TopAbs_SHAPE
    )
    from OCP.TopoDS import (
        TopoDS,
        TopoDS_Face,
        TopoDS_Solid,
        TopoDS_CompSolid,
        TopoDS_Shape
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
    logger.warning(f'{e}')


from .decomposeparams import DecompositionParameters
from .geometric import BoundingBox
from .ocpboundingbox import (
    bounding_box_from_shape,
    ocp_box_from_bounding_box
)
from .ocpfilereadwrite import (
    extract_ocp_shape_type_from_shape,
    _get_ocp_surface_from_ocp_face,
    downcast_ocp_shape
)
from .ocpproperties import (
    _volume_of_ocp_solids,
    _surface_area_of_ocp_solids,
    _distance_between_shapes,
    _midpoint_between_two_points,
    _distance_between_ocp_points,
    _distance_of_point_along_vector,
    _get_uv_bounds_from_ocp_face,
    check_solids_intersect_by_faces,
    check_ocp_faces_connected,
    check_ocp_planes_coincident,
    ocp_cylinders_coincident,
    ocp_sphere_coincident,
    ocp_cone_coincident,
    ocp_tori_coincident
)
from .ocprepair import (
    _clean_solid
)
from .ocpfilereadwrite import (
    STEPFileWriter
)

@dataclass
class BooleanSolution:
    """Result of a boolean operation on OCP solids.

    Attributes
    ----------
    solids : list[TopoDS_Solid]
        The solids produced by the operation.
    fuzzy_value : float
        The fuzzy value used by the boolean operation.
    solid_tol_min : float
        Minimum shape-fix tolerance associated with the operation.
    solid_tol_max : float
        Maximum shape-fix tolerance associated with the operation.
    legacy_used : bool
        Whether the legacy BRepAlgoAPI method was used, by default False
    solids_cleaned : bool
        Whether the solids were cleaned before the operation, by default False
    failed : bool
        Whether the operation failed, by default False
    """
    solids: list["TopoDS_Solid"]
    fuzzy_value: float
    solid_tol_min: float
    solid_tol_max: float
    legacy_used: bool = False
    solids_cleaned: bool = False
    failed: bool = False

@dataclass
class BooleanCommonSolidsResults(BooleanSolution):
    """Result of a boolean common operation on OCP solids."""
    pass

@dataclass
class BooleanCutSolidsResults(BooleanSolution):
    """Result of a boolean cut operation on OCP solids."""
    pass

@dataclass
class BooleanSplitSolidsResults(BooleanSolution):
    """Result of a boolean split operation on OCP solids."""
    pass

@dataclass
class BooleanFuseSolidsResults(BooleanSolution):
    """Result of a boolean fuse operation on OCP solids."""
    pass


def _filter_degenerate_thin_solids_2(solid_original: "TopoDS_Solid", solids: list['TopoDS_Solid'], decom_params: DecompositionParameters, operation_type: Any) -> list[TopoDS_Solid]:
    """Remove degenerate (tiny or thin) solids produced by a boolean operation.

    A solid is kept if its volume is at least
    ``bool_degenerate_min_ab_vol``, its volume / surface area ratio is at least
    ``bool_degenerate_min_area_to_vol_ratio`` and its volume as a fraction of
    the original is at least ``bool_degenerate_min_vol_fraction_of_original``.

    Parameters
    ----------
    solid_original : TopoDS_Solid
        The solid the boolean operation was applied to.
    solids : list[TopoDS_Solid]
        The solids produced by the boolean operation.
    decom_params : DecompositionParameters
        The decomposition parameters holding the filter thresholds.
    operation_type : Any
        Label of the operation (e.g. BOPAlgo_COMMON or BOPAlgo_Splitter),
        only used in log messages.

    Returns
    -------
    list[TopoDS_Solid]
        The solids passing the filter, sorted by descending volume, or
        ``[solid_original]`` if ``solids`` is just the original solid.
    """
    # check if only one solid and identical to original solid
    if len(solids) == 1 and solid_original.IsSame(solids[0]):
        return [solid_original]

    # store list of filtered solids
    solids_filtered: list[TopoDS_Solid] = []

    # calculate the volume of the original solid
    vol_solid_original = _volume_of_ocp_solids([solid_original])

    # sort solids by volume, in descenging order
    solids_sorted_by_vol = _sort_solids_by_volume(solids, ascending=False)

    # check each solid 
    for solid in solids_sorted_by_vol:
        # calculate volume of the current solid, along with total surface area
        vol_solid = _volume_of_ocp_solids([solid])

        # check if below minimum volume size
        if vol_solid < decom_params.bool_degenerate_min_ab_vol:
            continue

        # calculate area and carry out checks
        area_solid = _surface_area_of_ocp_solids([solid])
        vol_to_area_ratio = vol_solid / area_solid
        vol_fraction = vol_solid / vol_solid_original

        # check acceptance criteria
        area_check = vol_to_area_ratio >= decom_params.bool_degenerate_min_area_to_vol_ratio
        vol_fraction_check = vol_fraction >= decom_params.bool_degenerate_min_vol_fraction_of_original

        if area_check and vol_fraction_check:
            # add volume to filtered list
            solids_filtered.append(solid)
        else:
            # output to logger of degenerate volume found
            logger.debug(f'{type(operation_type)} - degenerate thin volume detector - vol / area = {vol_to_area_ratio:.2e} < {decom_params.bool_degenerate_min_vol_fraction_of_original:.2e} - vol / vol_original = {vol_fraction:.2e} < {decom_params.bool_degenerate_min_vol_fraction_of_original:.2e}')

            if logger.level <= 10:
                step_writer = STEPFileWriter(decom_params)
                step_writer.add_solids([solid_original], 'original_solis')
                step_writer.add_solids(solids, 'solids')
                step_writer.add_solids([solid], 'solid_filtered_(removed)')
                output_path = step_writer.write(decom_params.debug_filter_degenerate_solid_path, enumerate_files=True)
                logger.warning(f'degenerate solid saved to step - see {output_path}')

    n_solids_in = len(solids)
    n_solids_out = len(solids_filtered)

    if n_solids_out < n_solids_in:
        logger.debug(f'filtering degenerate solids: solids in = {len(solids)} solids out = {len(solids_filtered)}')

    return solids_filtered

def _cut_ocp_solid(
        shape_1: "TopoDS_Solid", 
        shape_2: "TopoDS_Solid | TopoDS_Face",
        decom_params: DecompositionParameters,
        reverse_face_orientation: bool = False,
        min_solids_required: int = 0,
        max_solids_required: float = math.inf,
        clean_solids: bool = False,
        use_legacy: bool = False
        ) -> BooleanCutSolidsResults:
    """Cut a solid (shape_1) by another solid or the half space of a face.

    If ``shape_2`` is a face, a half space is generated from it and used as
    the tool. The BOPAlgo_BOP cut operation is used unless ``use_legacy``, in
    which case BRepAlgoAPI_Cut is used. Errors, or a self-intersection warning,
    result in no solids.

    Parameters
    ----------
    shape_1 : TopoDS_Solid
        The solid to cut.
    shape_2 : TopoDS_Solid | TopoDS_Face
        The tool solid (solid, compsolid or compound) or a face from which a
        half space is generated.
    decom_params : DecompositionParameters
        The decomposition parameters.
    reverse_face_orientation : bool, optional
        Reverse the side of the half space generated from a face, by default
        False
    min_solids_required : int, optional
        Minimum number of solids the cut must produce, by default 0
    max_solids_required : float, optional
        Maximum number of solids the cut may produce, by default math.inf
    clean_solids : bool, optional
        Clean shape_1 and the tool before cutting, by default False
    use_legacy : bool, optional
        Use the legacy BRepAlgoAPI_Cut method, by default False

    Returns
    -------
    BooleanCutSolidsResults
        The cut solids and tolerances used.

    Raises
    ------
    Exception
        If ``shape_2`` is an unsupported shape type, or the number of solids
        produced is outside the required range.
    """

    # check is shape_2 is a face, and if so build an infinite halfspace 
    if shape_2.ShapeType() == TopAbs_FACE:
        # generate half space to apply to split operations
        tool = generate_ocp_half_space_from_face(shape_2, shape_1, decom_params, reverse_halfspace=reverse_face_orientation)
    elif shape_2.ShapeType() == TopAbs_SOLID or shape_2.ShapeType() == TopAbs_COMPSOLID or shape_2.ShapeType() == TopAbs_COMPOUND:
        tool = shape_2
    else:
        msg = f'common_ocp_solid() method requires a solid and either a solid, compsolid or face, not {type(shape_2)}'
        raise Exception(msg)

    # pre clean
    if clean_solids:
        # attempt to clean the two solids
        shape_1 = _clean_solid(shape_1)
        tool = _clean_solid(tool)

        # write to logger cleaning being applied
        logger.warning('BOPAlgo_CUT cleaning shape_1 and tool before BOPAlgo_CUT')

    if use_legacy:
        # legacy facllback for problem solids
        brep_cut = BRepAlgoAPI_Cut(shape_1, tool)
        brep_cut.SetFuzzyValue(1e-4)
        brep_cut.SetNonDestructive(True)
        brep_cut.Build()
        cut_solids = extract_ocp_shape_type_from_shape(brep_cut.Shape(), TopAbs_SOLID, cast_from_shape=True)
        
        # output to logger for debugging
        logger.warning('BOPAlgo_CUT - using legacy method for cut operation')
    else:
        # perform bop cut operations
        bop_cut = BOPAlgo_BOP()
        bop_cut.AddArgument(shape_1)
        bop_cut.AddTool(tool)
        bop_cut.SetOperation(BOPAlgo_CUT)
        bop_cut.SetFuzzyValue(decom_params.bool_cutter_fuzzy_value)
        bop_cut.SetNonDestructive(True)
        bop_cut.Perform()
        cut_solids = extract_ocp_shape_type_from_shape(bop_cut.Shape(), TopAbs_SOLID, cast_from_shape=True)

        # check erros
        if bop_cut.HasErrors():
            # set common solids to input solid
            cut_solids = []

            # get errors
            buffer_errors = BytesIO()
            bop_cut.DumpWarnings(buffer_errors)
            error_strings = buffer_errors.getvalue().decode("utf-8", errors="replace").split()

            # output error message to logger
            n_errors = len(error_strings)
            for i, error_string in enumerate(error_strings, start=1):
                logger.debug(f'BOPAlgo_CUT HasErrors {i} of {n_errors} = {error_string}')

            if logger.level <= 10:
                # write error solids to analysis folder for debugging            
                step_file_writer = STEPFileWriter(decom_params)
                step_file_writer.add_solids([shape_1], 'shape_1')
                step_file_writer.add_solids([shape_2], 'shape_2')
                step_file_writer.add_solids([tool], 'tool')
                step_file_writer.add_solids(cut_solids, 'cut_solids')
                output_path = step_file_writer.write(decom_params.debug_cut_error_path, enumerate_files=True)
                logger.debug(f'BOPAlgo_CUT HasErrors  - see {output_path}')
        
        # check warnings
        if bop_cut.HasWarnings():
            # get warnings
            buffer_warnings = BytesIO()
            bop_cut.DumpWarnings(buffer_warnings)
            warning_strings = buffer_warnings.getvalue().decode("utf-8", errors="replace").split()

            if "BOPAlgo_AlertAcquiredSelfIntersection" in warning_strings:
                cut_solids = []

            n_warnings = len(warning_strings)
            for i, warning_string in enumerate(warning_strings, start=1):
                logger.debug(f'BOPAlgo_CUT HasWarnings {i} of {n_warnings} = {warning_string}')

            if logger.level <= 10:
                # write warnings solids to analysis folder for debugging            
                step_file_writer = STEPFileWriter(decom_params)
                step_file_writer.add_solids([shape_1], 'shape_1')
                step_file_writer.add_solids([shape_2], 'shape_2')
                step_file_writer.add_solids([tool], 'tool')
                step_file_writer.add_solids(cut_solids, 'cut_solids')
                output_path = step_file_writer.write(decom_params.debug_cut_warning_path, enumerate_files=True)
                
                logger.debug(f'BOPAlgo_CUT HasWarnings - see {output_path}')


    # check that the required number of common solids have been produced
    if not (min_solids_required <= len(cut_solids) <= max_solids_required):
        step_writer = STEPFileWriter(decom_params)
        step_writer.add_solids([shape_1], 'shape_1')
        step_writer.add_solids([shape_2], 'shape_2')
        step_writer.add_solids([tool], 'tool')
        step_writer.add_solids(cut_solids, 'cut_solids')
        output_path = step_writer.write(decom_params.debug_cut_fatal_path, enumerate_files=True)
        
        msg = f'BOPAlso_Cut - failed to produce required number of solids, {len(cut_solids)} produced, but {min_solids_required} - {max_solids_required} required - see {output_path}'
        logger.fatal(msg)

        raise Exception(msg)

    # check for degenrate solids
    # cut_solids = _filter_degenerate_thin_solids_2(shape_1, cut_solids, decom_params, BOPAlgo_COMMON)

    result = BooleanCutSolidsResults(
        cut_solids,
        decom_params.bool_cutter_fuzzy_value,
        decom_params.cleanup_shape_fix_tol_min,
        decom_params.cleanup_shape_fix_tol_max
        )
    
    return result

def _common_ocp_solid(
        shape_1: "TopoDS_Solid | TopoDS_CompSolid",
        shape_2: "TopoDS_Solid | TopoDS_CompSolid | TopoDS_Face",
        decom_params: DecompositionParameters,
        reverse_face_orientation: bool = False,
        min_solids_required: int = 0,
        max_solids_required: float = math.inf,
        max_distance_from_face: float = 0.0,
        use_legacy: bool = False) -> BooleanCommonSolidsResults:
    """Find the common shape between a solid (shape_1) and a solid or face.

    If ``shape_2`` is a face, a half space is generated from it and used as
    the tool. The BOPAlgo_BOP common operation is used unless ``use_legacy``,
    in which case BRepAlgoAPI_Common is used. On errors or a self-intersection
    warning ``shape_1`` is returned and the result is flagged as failed. If the
    number of solids is outside the required range, ``shape_1`` is returned
    (assumed to be fully enclosed in ``shape_2``). Degenerate thin solids are
    then filtered out.

    Parameters
    ----------
    shape_1 : TopoDS_Solid | TopoDS_CompSolid
        The solid to intersect.
    shape_2 : TopoDS_Solid | TopoDS_CompSolid | TopoDS_Face
        The tool solid (solid, compsolid or compound) or a face from which a
        half space is generated.
    decom_params : DecompositionParameters
        The decomposition parameters.
    reverse_face_orientation : bool, optional
        Reverse the side of the half space generated from a face, by default
        False
    min_solids_required : int, optional
        Minimum number of solids required, by default 0
    max_solids_required : float, optional
        Maximum number of solids allowed, by default math.inf
    max_distance_from_face : float, optional
        Unused, by default 0.0
    use_legacy : bool, optional
        Use the legacy BRepAlgoAPI_Common method, by default False

    Returns
    -------
    BooleanCommonSolidsResults
        The common solids, tolerances used and failure flag.

    Raises
    ------
    Exception
        If ``shape_2`` is an unsupported shape type.
    """

    # check is shape_2 is a face, and if so build an infinite halfspace 
    if shape_2.ShapeType() == TopAbs_FACE:
        # generate half space to apply to split operations
        tool = generate_ocp_half_space_from_face(shape_2, shape_1, decom_params, reverse_halfspace=reverse_face_orientation)
    elif shape_2.ShapeType() == TopAbs_SOLID or shape_2.ShapeType() == TopAbs_COMPSOLID or shape_2.ShapeType() == TopAbs_COMPOUND:
        tool = shape_2
    else:
        msg = f'common_ocp_solid() method requires a solid and either a solid, compsolid or face, not {type(shape_2)}'
        raise Exception(msg)

    # set up flag to check failures
    common_failed = False

    # logger.debug(f'_common_ocp_solid()')

    # logger.debug(f'common_ocp_solid() - starting with surfaces...')
    # faces = extract_ocp_shape_type_from_shape(shape_1, TopAbs_FACE, cast_from_shape=True)
    # for i, face in enumerate(faces):
    #     surface =  _get_ocp_surface_from_ocp_face(face)
    #     logger.debug(f'face {i} type is {type(surface)}')
    #     if isinstance(surface, Geom_Plane):
    #         print_ocp_plane_params(surface)
    #         logger.debug(f'{face.Orientation()}')
    #     elif isinstance(surface, Geom_CylindricalSurface):
    #         print_ocp_cylinder_params(surface)

    if not use_legacy:
        # perform BOP common operations
        bop = BOPAlgo_BOP()
        bop.AddArgument(shape_1)
        bop.AddTool(tool)
        bop.SetOperation(BOPAlgo_COMMON)
        bop.SetFuzzyValue(decom_params.bool_common_fuzzy_value)
        bop.SetNonDestructive(True)
        bop.Perform()
        common_solids = extract_ocp_shape_type_from_shape(bop.Shape(), TopAbs_SOLID, cast_from_shape=True)

        # check erros
        if bop.HasErrors():
            # set common solids to input solid
            common_solids = [shape_1]
            common_failed = True

            # get errors
            buffer_errors = BytesIO()
            bop.DumpWarnings(buffer_errors)
            error_strings = buffer_errors.getvalue().decode("utf-8", errors="replace").split()
            
            # output error message to logger
            n_errors = len(error_strings)
            for i, error_string in enumerate(error_strings, start=1):
                logger.warning(f'BOPAlgo_COMMON HasErrors {i} of {n_errors} = {error_string}')

            if logger.level <= 10:
                # write error solids to analysis folder for debugging
                step_file_writer = STEPFileWriter(decom_params)
                step_file_writer.add_solids([shape_1], 'shape_1')
                step_file_writer.add_solids([shape_2], 'shape_2')
                step_file_writer.add_solids([tool], 'tool')
                step_file_writer.add_solids(common_solids, 'common_solids')
                output_path = step_file_writer.write(decom_params.debug_common_error_path, enumerate_files=True)
                logger.warning(f'BOPAlgo_COMMON HasError - see file {output_path}')

        # check warnings
        if bop.HasWarnings():
            # get warnings
            buffer_warnings = BytesIO()
            bop.DumpWarnings(buffer_warnings)
            warning_strings = buffer_warnings.getvalue().decode("utf-8", errors="replace").split()

            # check if self intersection produced
            if "BOPAlgo_AlertAcquiredSelfIntersection" in warning_strings:
                common_solids = [shape_1]
                common_failed = True

            # output warning message to logger
            n_warnings = len(warning_strings)
            for i, warning_string in enumerate(warning_strings, start=1):
                logger.error(f'BOPAlgo_COMMON HasWarnings {i} of {n_warnings} = {warning_string}')

            if logger.level <= 10:
                # write warnings solids to analysis folder for debugging            
                step_file_writer = STEPFileWriter(decom_params)
                step_file_writer.add_solids([shape_1], 'shape_1')
                step_file_writer.add_solids([shape_2], 'shape_2')
                step_file_writer.add_solids([tool], 'tool')
                step_file_writer.add_solids(common_solids, 'common_solids')
                output_path = step_file_writer.write(decom_params.debug_common_warning_path, enumerate_files=True)
                logger.warning(f'BOPAlgo_COMMON HasWarning - see file {output_path}')

    else:
        # try legacy boolean common method
        logger.debug(f'_common_ocp_solid() - using legacy BRepAlgoAPI_Common')
        brep_common = BRepAlgoAPI_Common(shape_1, tool)
        brep_common.SetFuzzyValue(decom_params.bool_common_fuzzy_value)
        brep_common.SetNonDestructive(True)
        brep_common.Build()
        common_solids = extract_ocp_shape_type_from_shape(brep_common.Shape(), TopAbs_SOLID, cast_from_shape=True)
        logger.debug(f'legacy method produced {len(common_solids)}')

    # check that the required number of common solids have been produced
    if not (min_solids_required <= len(common_solids) <= max_solids_required):
        # case where BOPAlgo_COMMON fails, but shape_1 is fully enclosed within shape_2
        common_solids = [shape_1]

    elif not (min_solids_required <= len(common_solids) <= max_solids_required):
        logger.warning(f'BOPAlgo_COMMON - failed to produce - tried legacy BRepAlgoAPI_Common method - return {len(common_solids)} solids')

        if check_solids_intersect_by_faces(shape_1, shape_2):
            logger.fatal(f'BPOAlgo_COMMON - failed to produced required solids, but shape_1 is inside shape_2')
        
        step_writer = STEPFileWriter(decom_params)
        step_writer.add_solids([shape_1], 'shape_1')
        step_writer.add_solids([shape_2], 'shape_2')
        step_writer.add_solids([tool], 'tool')
        step_writer.add_solids(common_solids, 'common_solids')
        output_path = step_writer.write(decom_params.debug_common_fatal_path, enumerate_files=True)
       
        msg = f'BOPAlgo_COMMON - failed to produce required number of solids, {len(common_solids)} produced, but {min_solids_required} - {max_solids_required} required - see {output_path}'
        logger.fatal(msg)

        # # if face used for shape_2 check if plane, and print params for this and all other planes
        # if shape_2.ShapeType() == TopAbs_FACE:
        #     for plane in surfaces_planes:
        #         print_ocp_plane_params(plane)

        raise Exception(msg)

    # check for degenrate solids
    n_solids_before_filter = len(common_solids)
    common_solids = _filter_degenerate_thin_solids_2(shape_1, common_solids, decom_params, BOPAlgo_COMMON)
    n_solids_after_filter = len(common_solids)
    if n_solids_after_filter < n_solids_before_filter:
        logger.debug(f'_common_ocp_solids filtered {n_solids_after_filter - n_solids_before_filter} solids')

    # return common solids
    result = BooleanCommonSolidsResults(
        common_solids,
        decom_params.bool_common_fuzzy_value, 
        decom_params.cleanup_shape_fix_tol_min,
        decom_params.cleanup_shape_fix_tol_max,
        failed=common_failed)
    
    return result



def _fuse_ocp_solid(
        shape_1: "TopoDS_Solid | TopoDS_CompSolid",
        shape_2: "TopoDS_Solid | TopoDS_CompSolid | TopoDS_Face",
        decom_params: DecompositionParameters,
        reverse_face_orientation: bool = False,
        min_solids_required: int = 0,
        max_solids_required: float = math.inf,
        max_distance_from_face: float = math.inf) -> TopoDS_Shape:
    """Fuse a solid (shape_1) with another solid or the half space of a face.

    If ``shape_2`` is a face, a half space is generated from it and used as
    the tool. The BOPAlgo_BOP fuse operation is applied and its result
    downcast.

    Parameters
    ----------
    shape_1 : TopoDS_Solid | TopoDS_CompSolid
        The first solid.
    shape_2 : TopoDS_Solid | TopoDS_CompSolid | TopoDS_Face
        The solid (solid, compsolid or compound) to fuse, or a face from which
        a half space is generated.
    decom_params : DecompositionParameters
        The decomposition parameters, ``bool_common_fuzzy_value`` is used as
        the fuzzy value.
    reverse_face_orientation : bool, optional
        Reverse the side of the half space generated from a face, by default
        False
    min_solids_required : int, optional
        Unused, by default 0
    max_solids_required : float, optional
        Unused, by default math.inf
    max_distance_from_face : float, optional
        Unused, by default math.inf

    Returns
    -------
    TopoDS_Shape
        The fused shape, downcast to its specific TopoDS type.

    Raises
    ------
    Exception
        If ``shape_2`` is an unsupported shape type.
    """

    # check is shape_2 is a face, and if so build an infinite halfspace 
    if shape_2.ShapeType() == TopAbs_FACE:
        # generate half space to apply to split operations
        tool = generate_ocp_half_space_from_face(shape_2, shape_1, decom_params, reverse_halfspace=reverse_face_orientation)
    elif shape_2.ShapeType() == TopAbs_SOLID or shape_2.ShapeType() == TopAbs_COMPSOLID or shape_2.ShapeType() == TopAbs_COMPOUND:
        tool = shape_2
    else:
        msg = f'fuse_ocp_solid() method requires a solid and either a solid, compsolid or face, not {type(shape_2)}'
        raise Exception(msg)

    bop = BOPAlgo_BOP()
    bop.AddArgument(shape_1)
    bop.AddTool(tool)
    bop.SetOperation(BOPAlgo_FUSE)
    bop.SetFuzzyValue(decom_params.bool_common_fuzzy_value)
    bop.SetNonDestructive(True)
    bop.Perform()

    # fused_solids = extract_ocp_shape_type_from_shape(bop.Shape(), TopAbs_SOLID, cast_from_shape=True)

    # # print errors
    # if bop.HasErrors() or bop.HasWarnings():
    #     fused_solids = [shape_1]

    # # check that the required number of common solids have been produced
    # if not (min_solids_required <= len(fused_solids) <= max_solids_required):
    #     logger.fatal(f'fused_ocp_solid()')
    #     logger.fatal(f'shape_1.ShapeType() = {shape_1.ShapeType()}')
    #     logger.fatal(f'shape_2.ShapeType() = {shape_2.ShapeType()}')
    #     vol = _volume_of_ocp_solids([shape_1])
    #     area = _surface_area_of_ocp_solids([shape_1])
    #     vol_over_area = vol / area
    #     logger.fatal(f'vol = {vol:.2e}')
    #     logger.fatal(f'area = {area:.2e}')
    #     logger.fatal(f'vol_over_area = {vol_over_area:.2e}')

    #     output_msg = f'err_{bop.HasErrors()}_war_{bop.HasWarnings()}'

    #     step_file_writer = STEPFileWriter(decom_params)
    #     step_file_writer.add_solids([shape_1], 'shape_1')
    #     step_file_writer.add_solids([shape_2], 'shape_2')
    #     step_file_writer.add_solids([tool], 'tool')
    #     step_file_writer.add_solids(fused_solids, 'fused_solids')
    #     output_filename = step_file_writer.write(os.path.join('analysis', 'fused_debugging', f'fused_{output_msg}.stp'), enumerate_files=True)


    #     msg = f'fuse_ocp_solid() produce {len(fused_solids)} solids but was required to produce {min_solids_required} - {max_solids_required} solids. See output file {output_filename}'
    #     logger.fatal(msg)
    #     raise Exception(msg)

    # return the shape cast to a solid
    # print(type(bop.Shape()))
    # shape_as_solid = TopoDS.Solid_s(bop.Shape())

    return downcast_ocp_shape(bop.Shape())

def _split_ocp_solid(
        shape_1: "TopoDS_Solid | TopoDS_CompSolid",
        shape_2: "TopoDS_Solid | TopoDS_CompSolid | TopoDS_Face",
        decom_params: DecompositionParameters,
        reverse_face_orientation: bool = False,
        min_solids_required: int = 0,
        max_solids_required: float = math.inf,
        use_legacy: bool = False) -> list[TopoDS_Solid]:
    """Split a solid (shape_1) by another solid or the half space of a face.

    If ``shape_2`` is a face, a half space is generated from it and used as
    the tool for BOPAlgo_Splitter. On errors ``shape_1`` is returned.
    Degenerate thin solids are filtered out, and if only one solid remains
    ``[shape_1]`` is returned.

    Parameters
    ----------
    shape_1 : TopoDS_Solid | TopoDS_CompSolid
        The solid to split.
    shape_2 : TopoDS_Solid | TopoDS_CompSolid | TopoDS_Face
        The tool solid or compsolid, or a face from which a half space is
        generated.
    decom_params : DecompositionParameters
        The decomposition parameters.
    reverse_face_orientation : bool, optional
        Reverse the side of the half space generated from a face, by default
        False
    min_solids_required : int, optional
        Unused, by default 0
    max_solids_required : float, optional
        Unused, by default math.inf
    use_legacy : bool, optional
        Unused, by default False

    Returns
    -------
    list[TopoDS_Solid]
        The split solids, or ``[shape_1]`` if no split occurred.

    Raises
    ------
    Exception
        If ``shape_2`` is an unsupported shape type.
    """

    # check is shape_2 is a face, and if so build an infinite halfspace 
    if shape_2.ShapeType() == TopAbs_FACE:
        # generate half space to apply to split operations
        tool = generate_ocp_half_space_from_face(shape_2, shape_1, decom_params, reverse_halfspace=reverse_face_orientation)
    elif shape_2.ShapeType() == TopAbs_SOLID or shape_2.ShapeType() == TopAbs_COMPSOLID:
        tool = shape_2
    else:
        msg = f'common_ocp_solid() method requires a solid and either a solid, compsolid or face, not {type(shape_2)}'
        raise Exception(msg)


    # perform bop splitter operation
    bop_splitter = BOPAlgo_Splitter()
    bop_splitter.AddArgument(shape_1)
    bop_splitter.AddTool(tool)
    bop_splitter.SetFuzzyValue(decom_params.bool_splitter_fuzzy_value)
    bop_splitter.SetNonDestructive(True)
    bop_splitter.Perform()
    split_solids = extract_ocp_shape_type_from_shape(bop_splitter.Shape(), TopAbs_SOLID, cast_from_shape=True)

    # check erros
    if bop_splitter.HasErrors():
        # set common solids to input solid
        split_solids = [shape_1]

        # get errors as binary and convert to string
        buffer_errors = BytesIO()
        bop_splitter.DumpWarnings(buffer_errors)
        error_strings = buffer_errors.getvalue().decode("utf-8", errors="replace").split()

        # output error message to logger
        n_errors = len(error_strings)
        for i, error_string in enumerate(error_strings, start=1):
            logger.debug(f'BOPAlgo_CUT HasErrors {i} of {n_errors} = {error_string}')
    
        if logger.level <= 10:
            # write error solids to analysis folder for debugging            
            step_file_writer = STEPFileWriter(decom_params)
            step_file_writer.add_solids([shape_1], 'shape_1')
            step_file_writer.add_solids([shape_2], 'shape_2')
            step_file_writer.add_solids([tool], 'tool')
            step_file_writer.add_solids(split_solids, 'split_solids')
            output_path = step_file_writer.write(decom_params.debug_splitter_error_path, enumerate_files=True)
            logger.debug(f'BOPAlgo_CUT HasErrors - see {output_path}')

    # check warnings
    if bop_splitter.HasWarnings():
        # get warnings
        buffer_warnings = BytesIO()
        bop_splitter.DumpWarnings(buffer_warnings)
        warning_strings = buffer_warnings.getvalue().decode("utf-8", errors="replace").split()

        # output warning message to logger
        n_warnings = len(warning_strings)
        for i, warning_string in enumerate(warning_strings, start=1):
            logger.debug(f'BOPAlgo_CUT HasWarnings {i} of {n_warnings} = {warning_string}')

        if logger.level <= 10:
            # write warnings solids to analysis folder for debugging            
            step_file_writer = STEPFileWriter(decom_params)
            step_file_writer.add_solids([shape_1], 'shape_1')
            step_file_writer.add_solids([shape_2], 'shape_2')
            step_file_writer.add_solids([tool], 'tool')
            step_file_writer.add_solids(split_solids, 'split_solids')
            output_path = step_file_writer.write(decom_params.debug_splitter_warning_path, enumerate_files=True)

            logger.debug(f'BOPAlgo_CUT HasWarnings - see {output_path}')

    split_solids = _filter_degenerate_thin_solids_2(shape_1, split_solids, decom_params, BOPAlgo_Splitter)

    # check for degenrate solids
    n_solids_before_filter = len(split_solids)
    split_solids = _filter_degenerate_thin_solids_2(shape_1, split_solids, decom_params, BOPAlgo_COMMON)
    n_solids_after_filter = len(split_solids)
    if n_solids_after_filter < n_solids_before_filter:
        logger.debug(f'_split_ocp_solids filtered {n_solids_after_filter - n_solids_before_filter} solids')

    # check if split created two new solids, otherwise set to original solid
    if len(split_solids) == 1:
        split_solids = [shape_1]

    # return result
    return split_solids

def _recursively_split_ocp_solid(solid: "TopoDS_Solid", decom_params: DecompositionParameters) -> list["TopoDS_Solid"]:
    """Recursively split a solid by the half spaces of its own faces.

    Faces are grouped and sorted with :func:`_sort_ocp_faces`, and the half
    space of the first face of each group is tried in order; at the first that
    splits the solid, each resulting solid is split recursively.

    Half spaces are built lazily, only when their face is tried, as building
    one is an OCP boolean operation. With
    ``decom_params.split_skip_non_cutting_planes`` set, a planar face whose
    plane leaves the whole solid on its material side is skipped without
    building its half space, as it cannot split the solid (see
    :func:`_solid_behind_planar_face`). This gives the same split solids as
    trying every face, at a fraction of the cost (every face of a convex solid
    is skipped).

    Parameters
    ----------
    solid : TopoDS_Solid
        The solid to split.
    decom_params : DecompositionParameters
        The decomposition parameters.

    Returns
    -------
    list[TopoDS_Solid]
        The split solids, or ``[solid]`` if no half space splits it.
    """
    # gather faces of the solid, grouped by coincident surface and sorted
    faces: list[TopoDS_Face] = [
        face for face in extract_ocp_shape_type_from_shape(solid, TopAbs_FACE, cast_from_shape=True)
    ]
    faces_grouped_and_sorted = _sort_ocp_faces(faces, decom_params)

    # iterate over groups of faces and trial splits
    for group_of_faces in faces_grouped_and_sorted:
        # a plane leaving the whole solid on its material side cannot split it
        if decom_params.split_skip_non_cutting_planes and _solid_behind_planar_face(
            solid, group_of_faces[0], decom_params.split_skip_non_cutting_planes_rel_tol
        ):
            continue

        # generate the half space only when it is needed
        half_space = generate_ocp_half_space_from_face(
            group_of_faces[0],
            solid,
            decom_params,
            max_distance_from_face=decom_params.split_max_distnace_between_face_and_halfspace,
        )
        split_solids = _split_ocp_solid(solid, half_space, decom_params)

        # save_ocp_shapes_to_step(split_solids, 'sphere_split_error.stp', decom_params)
        if len(split_solids) > 1:
            # recursively split
            recursively_split_solids: list[TopoDS_Solid] = []
            for split_solid in split_solids:
                # rescursively split newly created solids
                split_solids_returned = _recursively_split_ocp_solid(split_solid, decom_params)
                
                # store split solids in the return list, and store filtered 
                recursively_split_solids.extend(split_solids_returned)

            # if a split has occured, return the split solids
            return recursively_split_solids

    # if no splits occur, return the original solid
    return [solid]


def _solid_behind_planar_face(
        solid: "TopoDS_Solid | TopoDS_CompSolid",
        face: "TopoDS_Face",
        rel_tol: float = 1e-7) -> bool:
    """Return whether a solid lies entirely on the material side of a planar face.

    The solid is expressed in a frame whose origin is a point on the face's
    plane and whose z axis is the face's outward normal (the surface normal,
    reversed for a reversed face, as in :func:`generate_ocp_half_space_from_face`).
    Its precise bounding box in that frame (``BRepBndLib.AddOptimal`` on the
    exact geometry) gives the furthest extent of the solid beyond the plane.
    If that is within ``rel_tol`` of the solid size, the plane does not cut the
    solid, so the face's half space cannot split it.

    Parameters
    ----------
    solid : TopoDS_Solid or TopoDS_CompSolid
        The solid.
    face : TopoDS_Face
        A face of the solid.
    rel_tol : float, optional
        Tolerance relative to the largest extent of the solid (with a floor of
        one length unit), by default 1e-7.

    Returns
    -------
    bool
        True if the face is planar and the solid lies behind its plane; False
        otherwise, including for non-planar faces and faces whose normal or
        orientation cannot be determined.
    """
    surface: Geom_Surface = _get_ocp_surface_from_ocp_face(face)
    if not isinstance(surface, Geom_Plane):
        return False

    # outward normal at the face mid point (same evaluation as the half space)
    u_min, u_max, v_min, v_max = _get_uv_bounds_from_ocp_face(face)
    props = GeomLProp_SLProps(surface, (u_min + u_max) / 2.0, (v_min + v_max) / 2.0, 1, 1e-6)
    if not props.IsNormalDefined():
        return False
    normal: gp_Dir = props.Normal()
    if face.Orientation() == TopAbs_REVERSED:
        normal = normal.Reversed()
    elif face.Orientation() != TopAbs_FORWARD:
        return False

    # express the solid in a frame on the plane with z along the outward normal
    frame = gp_Ax3(props.Value(), normal)
    transform = gp_Trsf()
    transform.SetTransformation(frame)
    solid_local: TopoDS_Shape = BRepBuilderAPI_Transform(solid, transform, False).Shape()

    # precise extents of the solid in that frame
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(solid_local, box, False, False)
    x_min, y_min, z_min, x_max, y_max, z_max = box.Get()
    size = max(x_max - x_min, y_max - y_min, z_max - z_min, 1.0)

    return z_max <= rel_tol * size


def generate_half_spaces_from_solid(solid: TopoDS_Solid, decom_params: DecompositionParameters, group_connected_faces: bool=False) -> dict[TopoDS_Shape, tuple[TopoDS_Face, ...]]:
    """Generate a half space for each group of faces in a solid.

    Faces are grouped and sorted with :func:`_sort_ocp_faces`, and a half space
    is generated from the first face of each group.

    Parameters
    ----------
    solid : TopoDS_Solid
        The solid whose faces are used.
    decom_params : DecompositionParameters
        The decomposition parameters.
    group_connected_faces : bool, optional
        Unused, by default False

    Returns
    -------
    dict[TopoDS_Shape, tuple[TopoDS_Face, ...]]
        Mapping of each half space solid to the group of faces it came from.
    """
    # get all faces within solid
    faces = [face for face in extract_ocp_shape_type_from_shape(solid, TopAbs_FACE, cast_from_shape=True)]

    # sort faces before half space generation
    faces_grouped_and_sorted = _sort_ocp_faces(faces, decom_params)

    # gather quasi half spaces for all faces in solid
    half_spaces: dict[TopoDS_Shape, tuple[TopoDS_Face, ...]] = {}
    
    for group_of_faces in faces_grouped_and_sorted:
        half_space = generate_ocp_half_space_from_face(group_of_faces[0], solid, decom_params, max_distance_from_face=decom_params.split_max_distnace_between_face_and_halfspace)
        half_spaces[half_space] = group_of_faces

    # return a dictionary of faces with their half spaces
    return half_spaces

def generate_ocp_half_space_from_face(
        face: "TopoDS_Face",
        solid: "TopoDS_Solid | TopoDS_CompSolid",
        decom_params: DecompositionParameters,
        reverse_halfspace: bool = False,
        halfspace_from_solid: bool = False,
        max_distance_from_face: float = math.inf
    ) -> TopoDS_Shape:

    """Generate a bounded half space solid from a face.

    An infinite half space is made from the unbound surface of the face with
    BRepPrimAPI_MakeHalfSpace, on the side given by the face orientation, and
    intersected with a cubic box around ``solid`` scaled by
    ``bounding_box_multiplier_half_space``. If that fails, cylinders, cones
    and tori are rebuilt manually as primitive solids and combined with the
    box by common or cut.

    Parameters
    ----------
    face : TopoDS_Face
        The face to convert into a half space.
    solid : TopoDS_Solid | TopoDS_CompSolid
        The solid the half space must cover.
    decom_params : DecompositionParameters
        The decomposition parameters.
    reverse_halfspace : bool, optional
        Take the opposite side of the face, by default False
    halfspace_from_solid : bool, optional
        Unused, by default False
    max_distance_from_face : float, optional
        If several solids are produced, those further than this from the face
        are discarded, by default math.inf

    Returns
    -------
    TopoDS_Shape
        The half space solid.

    Raises
    ------
    Exception
        If no normal is defined at the face mid point, the face orientation is
        neither forward nor reversed, the surface type is not supported, no
        half space solids are produced, or the half space does not remove any
        volume from the bounding box.
    """
    # define constants
    OFFSET_ALONG_NORMAL = 1e-6

    # extract geometric surface from face
    surface: Geom_Surface = _get_ocp_surface_from_ocp_face(face)

    # produce unbound face
    face_unbound: TopoDS_Face = BRepBuilderAPI_MakeFace(surface, decom_params.make_face_tol).Face()

    # extract uv parameters and calculat etheir mid points
    u_min, u_max, v_min, v_max = _get_uv_bounds_from_ocp_face(face)
    u_mid = (u_max + u_min) / 2.0
    v_mid = (v_max + v_min) / 2.0

    # extract properties at given u v parameters
    props = GeomLProp_SLProps(surface, u_mid, v_mid, 1, 1e-6)
    if not props.IsNormalDefined():
        # check a normal is defined
        msg = f'no normal defined at this point'
        raise Exception(msg)

    # get normal and point on the surface at the u v mid point
    normal = props.Normal()
    point = surface.Value(u_mid, v_mid)

    # generate point from normal and off set to assign generate_half_spaces_from_solidside of face for half space
    if face.Orientation() == TopAbs_FORWARD and reverse_halfspace == False:
        offset_vec: gp_Vec = gp_Vec(normal.XYZ()) * -1.0 * OFFSET_ALONG_NORMAL
    elif face.Orientation() == TopAbs_REVERSED and reverse_halfspace == False:
        offset_vec: gp_Vec = gp_Vec(normal.XYZ()) * OFFSET_ALONG_NORMAL
    elif face.Orientation() == TopAbs_FORWARD and reverse_halfspace:
        offset_vec: gp_Vec = gp_Vec(normal.XYZ()) * OFFSET_ALONG_NORMAL
    elif face.Orientation() == TopAbs_REVERSED and reverse_halfspace:
        offset_vec: gp_Vec = gp_Vec(normal.XYZ()) * -1.0 * OFFSET_ALONG_NORMAL
    else:
        step_file_writer = STEPFileWriter(decom_params)
        step_file_writer.add_solids([face], 'face')
        step_file_writer.add_solids([solid], 'solid')
        output_path = step_file_writer.write(decom_params.debug_half_space_failed_to_generate_path, enumerate_files=True)

        msg = f'generate_ocp_half_space_from_face - offet_point not set as Orientation == {face.Orientation()} - see {output_path}'
        logger.fatal(msg)

        raise Exception(msg)
    
    # calculate the offset point to signify the side of the half-space.
    offset_point: gp_Pnt = gp_Pnt(point.XYZ() + offset_vec.XYZ())

    # generate half space
    half_space: TopoDS_Solid = BRepPrimAPI_MakeHalfSpace(face_unbound, offset_point).Solid()

    # create a bounding bolx solid around solid provided
    bounding_box: BoundingBox = bounding_box_from_shape(solid).make_cubic() * decom_params.bounding_box_multiplier_half_space
    bounding_box_solid = ocp_box_from_bounding_box(bounding_box)
    
    # common the bounding box solid with the unbound face
    half_space_solids = _common_ocp_solid(
        bounding_box_solid, 
        half_space,
        decom_params,
        min_solids_required=1,
        max_solids_required=math.inf,
        max_distance_from_face=decom_params.split_max_distnace_between_face_and_halfspace
    ).solids

    # print(f'in generate_ocp_half_space_from_face() - produced {len(half_space_solids)}')

    # case where half space creation fails
    vol_bounding_box = _volume_of_ocp_solids([bounding_box_solid])
    vol_half_space_solids = _volume_of_ocp_solids(half_space_solids)
    if vol_half_space_solids == vol_bounding_box or len(half_space_solids) == 0 or vol_half_space_solids == 0:
        logger.debug(f'generate_ocP_half_space_from_face() - failed to produce valid half space vol_bb = {vol_bounding_box:.8e} vol_halfspace  = {vol_half_space_solids:.8e}')
        logger.debug(f'generate_ocp manual creation of solid for surface type {surface}')

        if isinstance(surface, Geom_CylindricalSurface):
            logger.debug(f'generate_ocp manual creation of stable cylinder')
            
            cylinder = surface.Cylinder()
            radius = cylinder.Radius()
            direction = cylinder.Axis().Direction()
            dir_vec = gp_Vec(direction.XYZ())

            # 1. Find the axial midpoint
            p_0 = surface.Value(0.0, v_mid)
            p_pi = surface.Value(math.pi, v_mid)
            axial_midpoint = gp_Pnt((p_0.X() + p_pi.X()) / 2.0, 
                                    (p_0.Y() + p_pi.Y()) / 2.0, 
                                    (p_0.Z() + p_pi.Z()) / 2.0)

            # 2. Place extreme points along the axis
            EXTREME_DIST = 1e5
            p_far_1 = gp_Pnt(axial_midpoint.XYZ() + (dir_vec * EXTREME_DIST).XYZ())
            p_far_2 = gp_Pnt(axial_midpoint.XYZ() - (dir_vec * EXTREME_DIST).XYZ())

            # 3. Calculate how far these points are from the RPP
            v1 = BRepBuilderAPI_MakeVertex(p_far_1).Vertex()
            v2 = BRepBuilderAPI_MakeVertex(p_far_2).Vertex()
            
            dist_1 = _distance_between_shapes(v1, bounding_box_solid)
            dist_2 = _distance_between_shapes(v2, bounding_box_solid)

            # 4. Move the points back toward the midpoint
            # We move back by (Distance - Buffer)
            # Buffer = 2 * diameter (4 * radius) to ensure it clears the RPP edges
            buffer = 4.0 * radius
            
            # New distance from midpoint = EXTREME_DIST - (dist_to_rpp - buffer)
            offset_1 = EXTREME_DIST - (dist_1 - buffer)
            offset_2 = EXTREME_DIST - (dist_2 - buffer)

            p_final_1 = gp_Pnt(axial_midpoint.XYZ() + (dir_vec * offset_1).XYZ())
            p_final_2 = gp_Pnt(axial_midpoint.XYZ() - (dir_vec * offset_2).XYZ())

            # 5. Generate the cylinder between these two points
            # We use p_final_2 as the base and the total length as height
            total_height = offset_1 + offset_2
            ax2_final = gp_Ax2(p_final_2, direction)
            cyl_solid = BRepPrimAPI_MakeCylinder(ax2_final, radius, total_height).Solid()

            if cyl_solid.ShapeType() != TopAbs_SOLID:
                logger.error("Cylinder is not a solid!")
            else:
                logger.error("Cylinder is a solid")

            # 6. Boolean Logic
            is_forward = (face.Orientation() == TopAbs_FORWARD)

            # try common with BOP algo approach
            op_func = _common_ocp_solid if is_forward != reverse_halfspace else _cut_ocp_solid
            
            half_space_solids = op_func(bounding_box_solid, cyl_solid, decom_params, min_solids_required=1).solids

            vol_bounding_box = _volume_of_ocp_solids([bounding_box_solid])
            vol_half_space = _volume_of_ocp_solids(half_space_solids)

            if len(half_space_solids) > 1:
                msg = f'half space common generated 1+ '

            logger.info(f'vol_bounding_box = {vol_bounding_box:.8e} vol_half_space = {vol_half_space:.8e}')

            if vol_bounding_box == vol_half_space:
                # case where no solid cut from bounding box
                logger.info(f'common operation failed in generate half space... using legacy common')
                logger.info(op_func)

                decom_params_LOWER_FUZZY = copy.deepcopy(decom_params)
                decom_params_LOWER_FUZZY.bool_common_fuzzy_value = 1e-12

                half_space_solids = op_func(bounding_box_solid, cyl_solid, decom_params_LOWER_FUZZY, use_legacy=True).solids

                vol_bounding_box = _volume_of_ocp_solids([bounding_box_solid])
                vol_half_space = _volume_of_ocp_solids(half_space_solids)
                logger.info(f'vol_bounding_box = {vol_bounding_box:.8e} vol_half_space = {vol_half_space:.8e}')

                if vol_bounding_box == vol_half_space or vol_half_space == 0:
                    # try rotatin gthe cylinder
                    rotation_angle = math.radians(7.3) # Avoid 0, 45, 90
                    trsf = gp_Trsf()
                    trsf.SetRotation(gp_Ax1(axial_midpoint, direction), rotation_angle)
                    ax2_final.Transform(trsf)
                    cyl_solid = BRepPrimAPI_MakeCylinder(ax2_final, radius, total_height).Solid()

                    half_space_solids = op_func(bounding_box_solid, cyl_solid, decom_params_LOWER_FUZZY, use_legacy=False).solids

                    vol_bounding_box = _volume_of_ocp_solids([bounding_box_solid])
                    vol_half_space = _volume_of_ocp_solids(half_space_solids)
                    logger.info(f'vol_following_rotation_of_cylinder = {vol_bounding_box:.8e} vol_half_space = {vol_half_space:.8e}')

                    if vol_bounding_box == vol_half_space:
                        raise Exception('half space cylinder rotation failed to create half space')

            step_file_writer = STEPFileWriter(decom_params)
            step_file_writer.add_solids([cyl_solid], 'cyl_solid')
            step_file_writer.add_solids(half_space_solids, 'half_space_solids')
            output_path = step_file_writer.write('making_halfspace_manual_cylinder.stp')
            # print(f'step file written to {output_path}')

        elif isinstance(surface, Geom_ConicalSurface):
            # print_ocp_cone_params(surface)

            # get the cone from the original rectangular trimmed surfaces
            surface_original: Geom_Surface = _get_ocp_surface_from_ocp_face(face_unbound)

            # get the surface parameters
            cone: gp_Cone = surface_original.Cone()
            semi_angle: float = cone.SemiAngle()
            axis: gp_Ax1 = cone.Axis()
            direction: gp_Dir = axis.Direction()
            apex: gp_Pnt = cone.Apex()

            # calculate the two radii
            r1_p1 = surface.Value(0.0, v_min)
            r1_p2 = surface.Value(math.pi, v_min)
            r1_p_mid = _midpoint_between_two_points(r1_p1, r1_p2)
            r1_dist = _distance_between_ocp_points(apex, r1_p_mid)
            r1_dist_from_vec = _distance_of_point_along_vector(apex, gp_Vec(direction), r1_p_mid)

            r2_p1 = surface.Value(0.0, v_max)
            r2_p2 = surface.Value(math.pi, v_max)
            r2_p_mid = _midpoint_between_two_points(r2_p1, r2_p2)
            r2_dist = _distance_between_ocp_points(apex, r2_p_mid)
            r2_dist_from_vec = _distance_of_point_along_vector(apex, gp_Vec(direction), r2_p_mid)

            # extend the radii by extension length 
            EXTENSION_LENGTH = 1e3
            r1_extended_dist = r1_dist - EXTENSION_LENGTH
            r1_extended_dist = r1_extended_dist if r1_extended_dist > 0 else 0.0
            r2_extended_dist = r2_dist + EXTENSION_LENGTH

            # calculate radii at extension points
            ax2 = gp_Ax2(apex, direction)
            r1 = r1_extended_dist * math.tan(semi_angle)
            r2 = r2_extended_dist * math.tan(semi_angle)
            height = r2_extended_dist - r1_extended_dist

            # make the cone
            cone_at_axis = BRepPrimAPI_MakeCone(ax2, r1, r2, height).Solid()

            # move the cone along the axis 
            translation_vector = gp_Vec(direction)
            translation_vector.Scale(r1_extended_dist)

            transform = gp_Trsf()
            transform.SetTranslation(translation_vector)
            
            cone_solid = BRepBuilderAPI_Transform(cone_at_axis, transform, True).Shape()

            if face.Orientation() == TopAbs_FORWARD and reverse_halfspace == False:
                half_space_solids = _common_ocp_solid(bounding_box_solid, cone_solid, decom_params, min_solids_required=1).solids
            elif face.Orientation() == TopAbs_REVERSED and reverse_halfspace == False:
                half_space_solids = _cut_ocp_solid(bounding_box_solid, cone_solid, decom_params, min_solids_required=1).solids
            elif face.Orientation() == TopAbs_FORWARD and reverse_halfspace:
                half_space_solids = _cut_ocp_solid(bounding_box_solid, cone_solid, decom_params, min_solids_required=1).solids
            elif face.Orientation() == TopAbs_REVERSED and reverse_halfspace:
                half_space_solids = _common_ocp_solid(bounding_box_solid, cone_solid, decom_params, min_solids_required=1).solids

        elif isinstance(surface, Geom_ToroidalSurface):
            # set up a manual tori
            surface_original: Geom_Surface = _get_ocp_surface_from_ocp_face(face_unbound)

            r_major = surface.MajorRadius()
            r_minor = surface.MinorRadius()
            axis = surface.Position().Ax2()

            torus_solid = BRepPrimAPI_MakeTorus(axis, r_major, r_minor).Shape()
            
            if face.Orientation() == TopAbs_FORWARD and reverse_halfspace == False:
                half_space_solids = [torus_solid]
            elif face.Orientation() == TopAbs_REVERSED and reverse_halfspace == False:
                half_space_solids = _cut_ocp_solid(bounding_box_solid, torus_solid, decom_params, min_solids_required=1).solids
            elif face.Orientation() == TopAbs_FORWARD and reverse_halfspace:
                half_space_solids = _cut_ocp_solid(bounding_box_solid, torus_solid, decom_params, min_solids_required=1).solids
            elif face.Orientation() == TopAbs_REVERSED and reverse_halfspace:
                half_space_solids = [torus_solid]

        else:
            msg = f'half space failed generate_ocp_half_space_from_face()... face type {type(surface)} not yet coded'

            step_file_writer = STEPFileWriter(decom_params=decom_params)
            step_file_writer.add_solids([face], 'face')
            step_file_writer.add_solids([solid], 'solid')
            step_file_writer.add_solids(half_space_solids, 'half_space_solids')
            output_path = step_file_writer.write('failed_half_space_generation.stp')

            logger.fatal(f'{type(surface)}')
            raise Exception(msg)
    
    if len(half_space_solids) == 0: 

        step_file_writer = STEPFileWriter(decom_params)
        step_file_writer.add_solids([face], 'face')
        step_file_writer.add_solids([face_unbound], 'face_unbound (quasi)')
        step_file_writer.add_solids([solid], 'solid')
        step_file_writer.add_solids(half_space_solids, 'half_space_solids')
        step_file_writer.add_solids(half_space_solids[:1], 'half_space')
        output_path = step_file_writer.write('halfspace_as_ocp_solid_failed.stp')

        msg = f'[ERROR] - number of half space solids produced is {len(half_space_solids)} - see {output_path}'
        logger.fatal(msg)
        raise Exception(msg)

    # check for common solids which are not touching the original solids face and filter
    if len(half_space_solids) > 1:
        logger.debug(f'generate_ocp_half_space_from_face len(half_space_solids) > 1 ({len(half_space_solids)})')
        # remove solids which are greater than tolerance distance from face, superfluous common solid produced not required
        half_space_solids = [solid for solid in half_space_solids if _distance_between_shapes(face, solid) <= max_distance_from_face]

    half_space = half_space_solids[0]

    # check that the ocp half space has produced a solid with a new volume less than the maximum bounding box
    vol_bounding_box = _volume_of_ocp_solids([bounding_box_solid]) 
    vol_ocp_half_space = _volume_of_ocp_solids([half_space])

    if vol_ocp_half_space == vol_bounding_box:
        step_file_writer = STEPFileWriter(decom_params)
        step_file_writer.add_solids([face], 'face')
        step_file_writer.add_solids([face_unbound], 'face_unbound')
        step_file_writer.add_solids([solid], 'solid')
        step_file_writer.add_solids(half_space_solids, 'half_space_solids')
        step_file_writer.add_solids([half_space], 'half_space_solids[0]_kept_as_half_space')
        step_file_writer.add_solids(half_space_solids[:1], 'half_space_solids[1:]_remainder')
        output_path = step_file_writer.write(decom_params.debug_half_space_failed_to_generate_path, enumerate_files=True)

        msg = f'Creating halfspace did not cut new volume - see {output_path} (volume the same as input) vol_bounding_box = {vol_bounding_box:.8e} vol_ocp_half_space = {vol_ocp_half_space:.8e}'

        logger.critical(msg)

        raise Exception(msg)
    else:
        pass

    # return the common solid produced
    return half_space_solids[0]

def _sort_solids_by_volume(solids: list['TopoDS_Solid'], ascending: bool = True) -> list['TopoDS_Solid']:
    """Sort solids by volume.

    Parameters
    ----------
    solids : list[TopoDS_Solid]
        The solids to sort.
    ascending : bool, optional
        Sort smallest first if True, largest first if False, by default True

    Returns
    -------
    list[TopoDS_Solid]
        The solids sorted by volume.
    """
    # calculate volumes of solids    
    vols = [_volume_of_ocp_solids([solid]) for solid in solids]

    # sort solids by volume
    solids_sorted = [x for x, _ in sorted(zip(solids, vols), key=lambda x: x[1], reverse=not ascending)]

    # return list of solids sorted by volumes
    return solids_sorted


def _sort_ocp_faces(faces: list["TopoDS_Face"], decom_params: DecompositionParameters) -> list[tuple[TopoDS_Face, ...]]:
    """Group faces by type and coincidence and sort the groups by area.

    Groups are ordered by surface type (plane, cylinder, sphere, cone, torus)
    and, within each type, by total area from largest to smallest.

    Parameters
    ----------
    faces : list[TopoDS_Face]
        The faces to sort.
    decom_params : DecompositionParameters
        The decomposition parameters.

    Returns
    -------
    list[tuple[TopoDS_Face, ...]]
        The sorted groups of faces.
    """
    # group faces of type into coincident 
    faces_by_type_and_coincident = _group_ocp_faces_by_coincident(faces, decom_params)

    # set up list to store order of all faces
    faces_sorted: list[tuple[TopoDS_Face, ...]] = []

    # sort faces by area
    for face_type, groups_of_coincident_faces in faces_by_type_and_coincident.items():
        # sort groups of coincident faces
        faces_sorted_by_area = _sort_groups_of_ocp_faces_by_area(groups_of_coincident_faces)

        # store those sorted areas into an output list
        faces_sorted.extend(faces_sorted_by_area)

    # return list of sorted faces
    return faces_sorted

def _sort_ocp_faces_by_connected(faces: list["TopoDS_Face"], decom_params: DecompositionParameters, ignore_planes: bool=False) -> dict[type[Geom_Surface], list[list[TopoDS_Face]]]:
    """Group faces by surface type, coincidence and connectivity.

    Within each group of coincident faces, faces connected to a reference face
    are gathered into sub-groups.

    Parameters
    ----------
    faces : list[TopoDS_Face]
        The faces to group.
    decom_params : DecompositionParameters
        The decomposition parameters.
    ignore_planes : bool, optional
        Remove planes before grouping (calls ``faces.items()``, so expects a
        dict keyed by surface type), by default False

    Returns
    -------
    dict[type[Geom_Surface], list[list[TopoDS_Face]]]
        Groups of connected faces for each surface type.
    """
    # remove planes if required
    faces_filtered: dict = {k: v for k,v in faces.items() if k != Geom_Plane} if ignore_planes else faces

    # group faces of type into coincident 
    faces_by_type_and_coincident = _group_ocp_faces_by_coincident(faces_filtered, decom_params)

    # collect groups of connected faces
    groups_of_connected_faces: dict[type[Geom_Surface], list[list[TopoDS_Face]]] = {}

    for surface_type, groups_of_coincident_faces in faces_by_type_and_coincident.items():
        # gather groups of type with only one face and no connections
        sub_groups_of_connected_faces = [group for group in groups_of_coincident_faces if len(group) == 1]

        # go over all other groups of faces and check for 
        for faces_to_check in [group for group in groups_of_coincident_faces if len(group) > 1]:
            faces_to_check_dequeued = deque(faces_to_check)

            while faces_to_check_dequeued:
                face_ref = faces_to_check_dequeued.popleft()
                group = [face_ref]
                queue = deque([face_ref])
                while queue:
                    face = queue.popleft()
                    # connected_faces = [face for face in faces_to_check_dequeued if _ocp_faces_are_connected(face, face_ref)]
                    connected_faces = [face for face in faces_to_check_dequeued if check_ocp_faces_connected(face, face_ref)]

                    for face in connected_faces:
                        faces_to_check_dequeued.remove(face)
                        queue.append(face)
                        group.append(face)

                sub_groups_of_connected_faces.append(group)
        
        groups_of_connected_faces[surface_type] = sub_groups_of_connected_faces

    return groups_of_connected_faces

def _group_ocp_faces_by_coincident(faces: list["TopoDS_Face"], decom_params: DecompositionParameters, check_connected: bool=False) -> dict[type[Geom_Surface], list[list[TopoDS_Face]]]:
    """Group faces by surface type and then by coincident surfaces.

    A face joins an existing group if it has the same orientation as the
    group's first face, the surfaces are coincident, ``check_connected`` is
    True and the faces are within 1e-4 of each other; otherwise it starts a new
    group.

    Parameters
    ----------
    faces : list[TopoDS_Face]
        The faces to group.
    decom_params : DecompositionParameters
        The decomposition parameters passed to the coincidence checks.
    check_connected : bool, optional
        Required to be True for any faces to be grouped together, by default
        False

    Returns
    -------
    dict[type[Geom_Surface], list[list[TopoDS_Face]]]
        Groups of faces for each surface type present.
    """
    SURFACE_COINCIDENT_CHECKS: dict[type[Geom_Surface], Callable[..., bool]] = {
        Geom_Plane: check_ocp_planes_coincident,
        Geom_CylindricalSurface: ocp_cylinders_coincident,
        Geom_SphericalSurface: ocp_sphere_coincident,
        Geom_ConicalSurface: ocp_cone_coincident,
        Geom_ToroidalSurface: ocp_tori_coincident
    }

    # group faces by type
    faces_by_type: dict[type[Geom_Surface], list[TopoDS_Face]] = {k: v for k, v in _sort_ocp_faces_by_type(faces).items() if len(v) > 0}

    # set up variable to store faces group by type, then by coincident
    faces_by_types_and_groups: dict[type[Geom_Surface], list[list[TopoDS_Face]]] = {face_type: [] for face_type in faces_by_type}

    # iterate over surface types and group coincident surfaces
    for surface_type, faces_to_check in faces_by_type.items():
        # create a method to check if face in in a gorup, avoiding nested loop break out requirements
        def face_in_groups(groups: list[list[TopoDS_Face]], face: TopoDS_Face) -> bool:
            """Add a face to the first matching group, if any.

            Parameters
            ----------
            groups : list[list[TopoDS_Face]]
                Existing groups of faces, modified in place.
            face : TopoDS_Face
                The face to place.

            Returns
            -------
            bool
                True if the face was added to a group.
            """
            surface = _get_ocp_surface_from_ocp_face(face)
            for group in groups:
                # get the first face in the group and its surface
                face_from_group = group[0]
                surface_from_group = _get_ocp_surface_from_ocp_face(face_from_group)
                
                # check if they are coincident and have the same orientation, and return true if they are added to the group
                if ocp_face_orientations_are_the_same(face, face_from_group) and SURFACE_COINCIDENT_CHECKS[surface_type](surface, surface_from_group, decom_params) and check_connected and _distance_between_shapes(face, face_from_group) < 1e-4:
                    group.append(face)
                    return True

            # case for not matching any face groups
            return False

        # set up first group with a surface of this type
        first_group = [faces_to_check[0]]
        faces_by_types_and_groups[surface_type].append(first_group)

        # check other faces against group, either adding to group or creating new group
        for face in faces_to_check[1:]:
            if face_in_groups(faces_by_types_and_groups[surface_type], face):
                continue
            else:
                faces_by_types_and_groups[surface_type].append([face])

    # print(f'by type and group (after filling)')
    # print(faces_by_types_and_groups)

    return faces_by_types_and_groups

def ocp_surface_types_are_the_same(surface_1: "Geom_Surface", surface_2: "Geom_Surface") -> bool:
    """Check that two surfaces are of the same type.

    Parameters
    ----------
    surface_1 : Geom_Surface
        First surface.
    surface_2 : Geom_Surface
        Second surface.

    Returns
    -------
    bool
        True if both surfaces have the same Python type.
    """
    # check that two surfaces are the same type
    return type(surface_1) == type(surface_2)

def ocp_face_orientations_are_the_same(face_1: "TopoDS_Face", face_2: "TopoDS_Face") -> bool:
    """Check that two faces have the same orientation.

    Parameters
    ----------
    face_1 : TopoDS_Face
        First face.
    face_2 : TopoDS_Face
        Second face.

    Returns
    -------
    bool
        True if the orientations are equal.
    """
    # check that two faces have the same orientation
    return face_1.Orientation() == face_2.Orientation()

def _sort_groups_of_ocp_faces_by_area(groups_of_faces: list[list[TopoDS_Face]]) -> list[tuple[TopoDS_Face, ...]]:
    """Sort groups of faces by their total area, largest first.

    Parameters
    ----------
    groups_of_faces : list[list[TopoDS_Face]]
        The groups of faces.

    Returns
    -------
    list[tuple[TopoDS_Face, ...]]
        The groups, as tuples, sorted by descending total area.
    """
    # set up variable to store 
    faces_by_area: dict[tuple["TopoDS_Face"], float] = {}

    for group_of_faces in groups_of_faces:
        area_total: float = 0.0
        for face in group_of_faces:
            props: GProp_GProps = GProp_GProps()
            BRepGProp.SurfaceProperties_s(face, props)
            area_total += props.Mass()
        faces_by_area[tuple(group_of_faces)] = area_total

    # sort the faces by their area from largest to smallest
    faces_by_area_sorted = sorted(faces_by_area.items(), key=lambda item: item[1], reverse=True)

    # return faces sorted by area
    return [group_of_faces for group_of_faces, area in faces_by_area_sorted]

def _sort_ocp_faces_by_type(faces: list["TopoDS_Face"]) -> OrderedDict[type[Geom_Surface], list[TopoDS_Face]]:
    """Sort faces by their underlying surface type.

    Parameters
    ----------
    faces : list[TopoDS_Face]
        The faces to sort.

    Returns
    -------
    OrderedDict[type[Geom_Surface], list[TopoDS_Face]]
        Faces keyed by Geom_Plane, Geom_CylindricalSurface,
        Geom_SphericalSurface, Geom_ConicalSurface and Geom_ToroidalSurface
        (in that order); keys may map to empty lists.

    Raises
    ------
    Exception
        If a face has any other surface type.
    """
    # set up variables to store faces by type
    faces_by_type: OrderedDict[type[Geom_Surface], list[TopoDS_Face]] = OrderedDict()
    faces_by_type[Geom_Plane] = []
    faces_by_type[Geom_CylindricalSurface] = []
    faces_by_type[Geom_SphericalSurface] = []
    faces_by_type[Geom_ConicalSurface] = []
    faces_by_type[Geom_ToroidalSurface] = []

    # iterate over all faces in solid
    for face in faces:
        # get surface from face
        surface = _get_ocp_surface_from_ocp_face(face)

        # check for valid surface types
        if type(surface) not in faces_by_type:
            msg = f'[ERROR] surface type not available: {type(surface)}'
            raise Exception(msg)
        else:
            # append the face to the dictionary with key by surface type
            faces_by_type[type(surface)].append(face)

    # return dictionary of faces sorted by type
    return faces_by_type