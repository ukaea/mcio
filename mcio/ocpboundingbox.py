"""Bounding box utilities for OpenCASCADE (OCP) shapes.

Converts between OCP shapes and the mcio :class:`BoundingBox` (axis aligned) and
:class:`OrientedBoundingBox` classes, and builds OCP box solids from mcio bounding
boxes.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from mcio import logger


try:
    import casadi
    import OCP
    from OCP.Bnd import (
        Bnd_Box, 
        Bnd_OBB
    )
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_Transform
    )
    from OCP.BRepPrimAPI import (
        BRepPrimAPI_MakeBox
    )
    from OCP.gp import (
        gp_Pnt,
        gp_Trsf
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
        TopoDS_Solid,
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
    print(e)
    logger.warning(f'{e}')

if TYPE_CHECKING:
    from OCP.TopAbs import TopAbs_ShapeEnum

from .geometric import (
    BoundingBox,
    OrientedBoundingBox
)


def bounding_box_from_shape(shape: TopoDS_Shape) -> BoundingBox:
    """Compute the axis-aligned bounding box of an OCP shape.

    Uses ``BRepBndLib.Add_s`` (with triangulation if present) and a zero gap.

    Parameters
    ----------
    shape : TopoDS_Shape
        The OpenCASCADE shape to bound.

    Returns
    -------
    BoundingBox
        mcio axis-aligned bounding box enclosing the shape.
    """
    # create mesh to tighten bounding box (optinal, can improve precision of higher order surfaces)
    # mesh = OCP.BRepMesh.BRepMesh_IncrementalMesh(shape, 0.1)
    # mesh.Perform()

    # create ocp bounding box
    ocp_bounding_box: Bnd_Box = Bnd_Box()
    ocp_bounding_box.SetGap(0.0)

    BRepBndLib.Add_s(shape, ocp_bounding_box, True)
    
    # BRepBndLib.AddOptimal_s(shape, ocp_bounding_box, True, True)
    
    # transfer the values of the OCP bounding box to a mcio bounding box
    x_min, y_min, z_min, x_max, y_max, z_max = ocp_bounding_box.Get()
    bounding_box = BoundingBox(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max)

    # return mcio boudning box
    return bounding_box
# Assuming OrientedBoundingBox is the class we designed previously

def oriented_bounding_box_from_shape(
    shape: TopoDS_Shape,
    optimal: bool = True,
    use_triangulation: bool = False,
    relative_margin: float = 1e-6,
) -> OrientedBoundingBox:
    """Compute an Oriented Bounding Box (OBB) for an OpenCASCADE shape.

    The OCCT ``Bnd_OBB`` result is translated to the mcio
    :class:`OrientedBoundingBox` format (centre, local axes, half-lengths).

    Parameters
    ----------
    shape : TopoDS_Shape
        The OpenCASCADE shape geometry.
    optimal : bool, optional
        Use OCCT's optimisation loop for a tighter OBB (True) or the faster
        inertia-axes estimate (False), by default True.
    use_triangulation : bool, optional
        Build the OBB from an existing triangulation, by default False. A
        coarse triangulation of a convex curved face lies inside the true
        surface, so the exact geometry is used by default.
    relative_margin : float, optional
        Half-lengths are grown by this fraction of the box diagonal (plus a
        1e-9 absolute floor) so the box encloses the shape even with round-off,
        by default 1e-6.

    Returns
    -------
    OrientedBoundingBox
        mcio oriented bounding box enclosing the shape.

    Raises
    ------
    ValueError
        If the shape is void or degenerate and no OBB could be created.
    """
    # 1. Initialize OpenCascade's native OBB container
    ocp_obb: Bnd_OBB = Bnd_OBB()

    # 2. Extract the oriented box data from the shape
    # Parameters: (shape, out_obb, useTriangulation, isOptimal, useShapeTolerance)
    BRepBndLib.AddOBB_s(shape, ocp_obb, use_triangulation, optimal, True)

    # 3. Handle edge cases where an OBB cannot be calculated (e.g., empty/void shapes)
    if ocp_obb.IsVoid():
        raise ValueError("The provided shape is void or degenerate; an OBB could not be created.")

    # 4. Extract Center point (returns a gp_XYZ object)
    ocp_center = ocp_obb.Center()
    center: tuple[float, float, float] = (ocp_center.X(), ocp_center.Y(), ocp_center.Z())

    # 5. Extract the 3 orthogonal local unit direction vectors
    ocp_x_dir = ocp_obb.XDirection()
    ocp_y_dir = ocp_obb.YDirection()
    ocp_z_dir = ocp_obb.ZDirection()

    axes: tuple[tuple[float, float, float], ...] = (
        (ocp_x_dir.X(), ocp_x_dir.Y(), ocp_x_dir.Z()),
        (ocp_y_dir.X(), ocp_y_dir.Y(), ocp_y_dir.Z()),
        (ocp_z_dir.X(), ocp_z_dir.Y(), ocp_z_dir.Z())
    )

    # 6. Extract the half-lengths along each local axis, grown by a small margin
    half_lengths_raw: tuple[float, float, float] = (
        ocp_obb.XHSize(),
        ocp_obb.YHSize(),
        ocp_obb.ZHSize()
    )
    margin: float = relative_margin * 2.0 * sum(h * h for h in half_lengths_raw) ** 0.5 + 1e-9
    half_lengths: tuple[float, float, float] = tuple(h + margin for h in half_lengths_raw)

    # 7. Instantiate and return the mcio OrientedBoundingBox object
    return OrientedBoundingBox(center=center, axes=axes, half_lengths=half_lengths)


def bounding_box_from_hollow_of_shape(shape: TopoDS_Shape) -> BoundingBox:
    """Compute the bounding box of the hollow (internal cavity) of a shape.

    Incomplete stub: currently computes the shape's axis-aligned bounding box
    but returns nothing.

    Parameters
    ----------
    shape : TopoDS_Shape
        The OpenCASCADE shape whose hollow is to be bounded.

    Returns
    -------
    BoundingBox
        Intended to be the bounding box of the hollow; currently always None.
    """
    bbox = bounding_box_from_shape(shape)


def ocp_box_from_bounding_box(bounding_box: BoundingBox) -> "TopoDS_Solid":
    """Build an OCP box solid matching an axis-aligned bounding box.

    Parameters
    ----------
    bounding_box : BoundingBox
        mcio axis-aligned bounding box giving the box minimum and maximum corners.

    Returns
    -------
    TopoDS_Solid
        Box solid spanning the bounding box extents.
    """
    point_1: gp_Pnt = gp_Pnt(*bounding_box.minimums)
    point_2: gp_Pnt = gp_Pnt(*bounding_box.maximums)

    box: TopoDS_Solid = BRepPrimAPI_MakeBox(point_1, point_2).Solid()

    return box



# Additional imports for your type checking context
# from OCP.TopoDS import TopoDS_Shape 

def ocp_box_from_oriented_bounding_box(obb: OrientedBoundingBox) -> TopoDS_Solid:
    """Build an OCP box solid representing an oriented bounding box.

    A box centred on the origin with the OBB half-lengths is created and then
    placed with a rigid transform whose local frame is the OBB centre and axes.
    The third axis is rebuilt as the cross product of the first two so the frame
    is right-handed (the box is symmetric, so this does not change its shape).

    Parameters
    ----------
    obb : OrientedBoundingBox
        mcio oriented bounding box to convert.

    Returns
    -------
    TopoDS_Solid
        Box solid aligned with the OBB local axes.
    """
    import numpy as np
    from OCP.gp import gp_Ax3, gp_Dir

    h = obb.half_lengths
    base_box: TopoDS_Solid = BRepPrimAPI_MakeBox(
        gp_Pnt(-h[0], -h[1], -h[2]), gp_Pnt(h[0], h[1], h[2])
    ).Solid()

    # right-handed orthonormal frame from the OBB axes
    x_axis = np.asarray(obb.axes[0], dtype=float)
    z_axis = np.cross(x_axis, np.asarray(obb.axes[1], dtype=float))
    frame = gp_Ax3(gp_Pnt(*obb.center), gp_Dir(*z_axis), gp_Dir(*x_axis))

    # transform from the global frame to the OBB frame
    trsf: gp_Trsf = gp_Trsf()
    trsf.SetTransformation(frame, gp_Ax3())

    transformed_shape: TopoDS_Shape = BRepBuilderAPI_Transform(base_box, trsf, True).Shape()

    return TopoDS.Solid_s(transformed_shape)
