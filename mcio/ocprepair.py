"""Repair and clean-up helpers for OpenCASCADE (OCP) solids.

Wraps OCCT healing tools (``ShapeUpgrade_UnifySameDomain``, ``ShapeFix_Solid``,
``ShapeFix_ShapeTolerance``, ``BRepBuilderAPI_Sewing``) used to tidy solids during
decomposition, including detection and fixing of inexact edges.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from mcio import logger

try:
    import casadi
    import OCP
    from OCP.BRep import (
        BRep_Tool
    )
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_Sewing, 
    )
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.ShapeFix import (
        ShapeFix_ShapeTolerance,
        ShapeFix_Solid
    )
    from OCP.ShapeUpgrade import (
        ShapeUpgrade_UnifySameDomain
    )
    from OCP.TopAbs import (
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
        TopoDS_Edge,
        TopoDS_Solid
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

    TOPODS_CAST_METHODS: dict[TopAbs_ShapeEnum, Callable[[TopoDS_Shape], Any]] = {TopAbs_COMPOUND: TopoDS.Compound_s,
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

if TYPE_CHECKING:
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.TopoDS import TopoDS_Shape

from .decomposeparams import DecompositionParameters
from .ocpfilereadwrite import (
    extract_ocp_shape_type_from_shape,
    STEPFileWriter
)


def _unify_same_domains(solid: TopoDS_Solid) -> TopoDS_Solid:
    """Merge faces and edges lying on the same underlying geometry.

    Applies ``ShapeUpgrade_UnifySameDomain`` (unifying faces and edges, with
    concatenation of B-splines) and returns the first solid of the result.

    Parameters
    ----------
    solid : TopoDS_Solid
        Solid to unify.

    Returns
    -------
    TopoDS_Solid
        The unified solid.
    """
    # apply unification to solid
    unifier: ShapeUpgrade_UnifySameDomain = ShapeUpgrade_UnifySameDomain(solid, True, True, True)
    unifier.Build()

    # cast back to solid
    unified_shape: TopoDS_Shape = unifier.Shape()
    unified_solids: list[TopoDS_Solid] = extract_ocp_shape_type_from_shape(unified_shape, TopAbs_SOLID, cast_from_shape=True)

    # return healed solid
    return unified_solids[0]

def _sew_ocp_shape(solid: TopoDS_Solid) -> TopoDS_Shape:
    """Sew the faces of a shape together with ``BRepBuilderAPI_Sewing``.

    Parameters
    ----------
    solid : TopoDS_Solid
        Shape whose faces are to be sewn.

    Returns
    -------
    TopoDS_Shape
        The sewn shape.
    """
    sew = BRepBuilderAPI_Sewing
    sew.Add(solid)
    sew.Perform()
    return sew.SewedShape()

def _shape_fix_tolerance(solid_1: TopoDS_Solid, decom_params: DecompositionParameters) -> None:
    """Limit the tolerances of a solid's sub-shapes to a configured range.

    Uses ``ShapeFix_ShapeTolerance.LimitTolerance``; the solid is modified in
    place.

    Parameters
    ----------
    solid_1 : TopoDS_Solid
        Solid whose tolerances are limited.
    decom_params : DecompositionParameters
        Decomposition parameters supplying ``cleanup_shape_fix_tol_min`` and
        ``cleanup_shape_fix_tol_max``.
    """
    tol_fix = ShapeFix_ShapeTolerance()
    tol_fix.LimitTolerance(solid_1, decom_params.cleanup_shape_fix_tol_min, decom_params.cleanup_shape_fix_tol_max)

def _shape_fix_solid(solid_1: TopoDS_Solid) -> TopoDS_Solid:
    """Heal a solid with ``ShapeFix_Solid``.

    Parameters
    ----------
    solid_1 : TopoDS_Solid
        Solid to fix.

    Returns
    -------
    TopoDS_Solid
        The first solid of the fixed result.
    """
    sf = ShapeFix_Solid()
    sf.Init(solid_1)
    sf.Perform()
    solid = extract_ocp_shape_type_from_shape(sf.Solid(), TopAbs_SOLID, cast_from_shape=True)[0]
    return solid

def _clean_solid(solid: TopoDS_Solid, decom_params: DecompositionParameters, operation_label: str='clean_solid', item_label: str = 'solid') -> TopoDS_Solid:
    """Validate and clean a solid.

    The solid is checked with ``BRepCheck_Analyzer``; if invalid it is written to
    a debug STEP file and an exception is raised. Otherwise same-domain
    unification, tolerance limiting and ``ShapeFix_Solid`` are applied in turn,
    with each step logged.

    Parameters
    ----------
    solid : TopoDS_Solid
        Solid to clean.
    decom_params : DecompositionParameters
        Decomposition parameters supplying clean-up tolerances and the debug
        output path (``debug_invalid_solid_path``).
    operation_label : str, optional
        Label of the calling operation used in log messages, by default
        'clean_solid'
    item_label : str, optional
        Label of the item being cleaned used in log messages, by default 'solid'

    Returns
    -------
    TopoDS_Solid
        The cleaned solid.

    Raises
    ------
    Exception
        If the solid fails the ``BRepCheck_Analyzer`` validity check.
    """
    # check solids using analyzer
    if not BRepCheck_Analyzer(solid).IsValid():
        # dump invalid solid to file
        step_file_writer = STEPFileWriter(decom_params)
        step_file_writer.add_solids([solid], 'solid_analyser_invalid')
        output_path = step_file_writer.write(decom_params.debug_invalid_solid_path, enumerate_files=True)

        msg = f'{operation_label} - analyser - {item_label} is invalid - see {output_path}'
        logger.fatal(msg)

        raise Exception(msg)
    
    # unify same domains
    _unify_same_domains(solid)
    logger.warning(f'{operation_label} - {item_label} - ShapeUpgrade_UnifySameDomain() applied')

    # fit tolerances within solid
    _shape_fix_tolerance(solid, decom_params)
    logger.warning(f'{operation_label} - {item_label} - ShapeFix_ShapeTolerance() applied - tol_min = {decom_params.cleanup_shape_fix_tol_min:.2e} tol_max = {decom_params.cleanup_shape_fix_tol_max:.2e}')

    # try to fix solid
    solid = _shape_fix_solid(solid)
    logger.warning(f'{operation_label} - {item_label} - ShapeFix_Solid() applied')

    # return solid
    return solid

def _fix_inexact_edges(solid: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Solid | TopoDS_Shape:
    """Detect inexact edges in a solid and attempt to fix them.

    An edge is counted as inexact if its tolerance exceeds
    ``cleanup_inexact_edges_tol`` or it fails ``BRepCheck_Analyzer``. If any are
    found, ``ShapeFix_Solid`` is run with the configured min/max tolerances and
    same-parameter wire fixing enabled.

    Parameters
    ----------
    solid : TopoDS_Solid
        Solid to inspect and fix.
    decom_params : DecompositionParameters
        Decomposition parameters supplying the inexact edge tolerances.

    Returns
    -------
    TopoDS_Solid or TopoDS_Shape
        The original solid if no inexact edges are found, otherwise the shape
        produced by ``ShapeFix_Solid``.
    """

    logger.info(f'attempting to fix inexact edges - type {solid}, {type(solid)}')

    # extract edges from the solid
    edges: list[TopoDS_Edge] = extract_ocp_shape_type_from_shape(solid, TopAbs_EDGE, cast_from_shape=True)

    # count number of inexact edges
    inexact_edge_count: int = 0

    # iterate over edges in solid
    for edge in edges:
        # get edge tolerance
        edge_tol: float = BRep_Tool.Tolerance_s(edge)

        # check if edge consistent
        analyser = BRepCheck_Analyzer(edge)
        is_valid = analyser.IsValid()

        if edge_tol > decom_params.cleanup_inexact_edges_tol or not is_valid:
            
            inexact_edge_count += 1

    # set up fixer
    if inexact_edge_count > 0:
        # output to logger
        logger.warning(f'{inexact_edge_count} inexact edge(s) found... attempting fix to solid')

        # if inexact edges found, then attempt fix via tolerance bounds
        fixer: ShapeFix_Solid = ShapeFix_Solid(solid)
        fixer.SetMinTolerance(decom_params.cleanup_inexact_edge_fix_tol_min)
        fixer.SetMaxTolerance(decom_params.cleanup_inexact_edge_fix_tol_max)

        fix_shell_tool = fixer.FixShellTool()
        fix_face_tool = fix_shell_tool.FixFaceTool()
        fix_wire_tool = fix_face_tool.FixWireTool()

        fix_wire_tool.FixSameParameterMode = 1

        # applt fixer
        fixer.Perform()

        return fixer.Shape()
    else:
        return solid
