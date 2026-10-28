"""Convert MCIO half-volumes back into OpenCASCADE (OCP) solids.

This module contains the CSG to CAD inversion routines. Each supported MCIO
surface type has a generator which builds an unbounded OCP face for the surface
and converts it into a half-space solid clipped to a bounding box. The
``SURFACE_TYPE_LOOKUP`` table maps surface class names to these generators.
"""

from typing import TYPE_CHECKING

import math

import numpy as np
from collections.abc import Callable

from mcio import logger

try:
    import casadi
    import OCP

    from OCP.BRep import BRep_Builder
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_GTransform,
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_Transform
    )
    from OCP.BRepPrimAPI import (
        BRepPrimAPI_MakeCone,
        BRepPrimAPI_MakeCylinder,
        BRepPrimAPI_MakeRevol,
        BRepPrimAPI_MakeSphere,
        BRepPrimAPI_MakeTorus
    )
    from OCP.Geom import (
        Geom_Plane, 
        Geom_CylindricalSurface,
        Geom_SphericalSurface,
        Geom_ConicalSurface,
        Geom_ToroidalSurface,
        Geom_Ellipse,
        Geom_SurfaceOfRevolution
    )
    from OCP.gp import (
        gp_Pnt,
        gp_Vec,
        gp_Dir,
        gp_Ax1,
        gp_Ax2,
        gp_Ax3,
        gp_Elips,
        gp_GTrsf,
        gp_Pln,
        gp_Trsf,
    )
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopoDS import (
        TopoDS_Compound,
        TopoDS_Face,
        TopoDS_Solid,
        TopoDS_Shape
    )

except:
    pass

if TYPE_CHECKING:
    from .volume import HalfVolume

from .decomposeparams import DecompositionParameters
from .geometric import BoundingBox
# from .volume import HalfVolume

try:
    from .ocpbooleans import (
        generate_ocp_half_space_from_face,
        _common_ocp_solid,
        _cut_ocp_solid,
        _fuse_ocp_solid
    )
    from .ocpboundingbox import ocp_box_from_bounding_box, bounding_box_from_shape
    from .ocpfilereadwrite import extract_ocp_shape_type_from_shape
except:
    pass


def _local_bounding_box(bounding_box: BoundingBox, local_to_main: gp_Trsf) -> BoundingBox:
    """Axis aligned box, in local coordinates, enclosing a main system bounding box.

    Parameters
    ----------
    bounding_box : BoundingBox
        Bounding box in the main system.
    local_to_main : gp_Trsf
        Transformation from local to main coordinates.

    Returns
    -------
    BoundingBox
        Box in local coordinates containing the corners of ``bounding_box``.
    """
    main_to_local = local_to_main.Inverted()
    corners = []
    for x in (bounding_box.x_min, bounding_box.x_max):
        for y in (bounding_box.y_min, bounding_box.y_max):
            for z in (bounding_box.z_min, bounding_box.z_max):
                corner = gp_Pnt(x, y, z).Transformed(main_to_local)
                corners.append((corner.X(), corner.Y(), corner.Z()))
    lo = [min(c[i] for c in corners) for i in range(3)]
    hi = [max(c[i] for c in corners) for i in range(3)]
    return BoundingBox(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])

def _convert_mcio_half_volume_into_ocp_solid(half_volume: HalfVolume, bounding_box: BoundingBox, decom_params: DecompositionParameters) -> TopoDS_Solid | TopoDS_Shape:
    """Convert an MCIO half-volume into an OCP solid clipped to a bounding box.

    The generator is selected from ``SURFACE_TYPE_LOOKUP`` using the surface
    class name; surfaces not in the table are treated as macrobodies. Any
    surface transform is applied to the resulting shape.

    Parameters
    ----------
    half_volume : HalfVolume
        MCIO half-volume (surface and direction) to convert.
    bounding_box : BoundingBox
        Bounding box that the half-space solid is clipped to.
    decom_params : DecompositionParameters
        Decomposition parameters supplying tolerances.

    Returns
    -------
    TopoDS_Solid or TopoDS_Shape
        Half-space solid, or the transformed shape if the surface has a
        transform.
    """

    # for a transformed surface the half space is made in the surface's own coordinates, so
    # use a box there that covers the bounding box once transformed
    if half_volume.surface.transform:
        bounding_box = _local_bounding_box(bounding_box, half_volume.surface.transform.ocp_transform)

    # create an ocp box from the bounding box
    bounding_box_ocp: TopoDS_Solid = ocp_box_from_bounding_box(bounding_box)

    # call function for surface type, and return produced solid
    if half_volume.surface.__class__.__name__ in SURFACE_TYPE_LOOKUP:
        ocp_half_space_solid = SURFACE_TYPE_LOOKUP[half_volume.surface.__class__.__name__](half_volume, bounding_box_ocp, decom_params)
    else:
        ocp_half_space_solid = _generate_ocp_solid_from_mcio_half_volume_macrobody(half_volume, bounding_box, decom_params)

    # apply surface transform
    if half_volume.surface.transform:
        ocp_half_space_solid = BRepBuilderAPI_Transform(
            ocp_half_space_solid, 
            half_volume.surface.transform.ocp_transform, 
            True).Shape()

    return ocp_half_space_solid

def _generate_ocp_solid_from_mcio_half_volume_macrobody(half_volume: HalfVolume, bounding_box_ocp: BoundingBox, decom_params: DecompositionParameters) -> TopoDS_Solid:
    """Generate an OCP solid for a macrobody half-volume.

    The macrobody is expressed as a volume of its individual surfaces; for the
    ``'+'`` direction this volume is complemented (outside of the macrobody),
    otherwise it is used directly (inside), and the result is converted into an
    OCP solid.

    Parameters
    ----------
    half_volume : HalfVolume
        MCIO half-volume whose surface is a macrobody providing
        ``as_volume_of_surfaces()``.
    bounding_box_ocp : BoundingBox
        Bounding box used to bound the generated solid. Despite the name, the
        caller passes an MCIO ``BoundingBox`` which is forwarded to
        ``ocp_solid``.
    decom_params : DecompositionParameters
        Decomposition parameters supplying tolerances.

    Returns
    -------
    TopoDS_Solid
        Solid representing the requested side of the macrobody.
    """

    # check direcdtion, if + then volume outside of macrobody required, else inside
    if half_volume.direction == '+':
        # +macrobody, case where outside of macrobody required, get macrobody as mcio volume of 
        # individual surfaces, invert to get space outside and convert to ocp solid
        ocp_solid = (~(half_volume.surface.as_volume_of_surfaces())).ocp_solid(bounding_box_ocp, decom_params)
    else:
        # -macrobody, case where inside macrobody required, convert to ocp solid
        ocp_solid =  half_volume.surface.as_volume_of_surfaces().ocp_solid(bounding_box_ocp, decom_params)

    # return the solid
    return ocp_solid

def _generate_ocp_solid_from_mcio_half_volume_plane(half_volume: HalfVolume, bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Solid:
    """Generate an OCP half-space solid for a general ``Plane`` half-volume.

    The plane is defined by ``a*x + b*y + c*z = d``; the point on the plane closest
    to the origin and the normalised normal are used to build the face.

    Parameters
    ----------
    half_volume : HalfVolume
        MCIO half-volume whose surface is a general plane and whose direction (``'+'``
        or ``'-'``) selects the side of the surface.
    bounding_box_ocp : TopoDS_Solid
        OCP box solid that the half-space is clipped to.
    decom_params : DecompositionParameters
        Decomposition parameters supplying tolerances.

    Returns
    -------
    TopoDS_Solid
        Half-space solid clipped to the bounding box.

    Raises
    ------
    ValueError
        If the plane coefficients a, b and c are all (near) zero.
    """
    # calculate onrmalised dir for plane
    norm_sq = half_volume.surface.a ** 2 + half_volume.surface.b ** 2 +half_volume.surface.c ** 2
    norm = math.sqrt(norm_sq)
    normal_dir = gp_Dir(half_volume.surface.a / norm, half_volume.surface.b / norm, half_volume.surface.c / norm)

    if norm_sq < 1e-12:
        msg = 'Invalid plane equation: coefficients A, B, and C cannot all be zero.'
        logger.fatal(msg)
        raise ValueError(msg)

    point = gp_Pnt(
        half_volume.surface.d * half_volume.surface.a / norm_sq,
        half_volume.surface.d * half_volume.surface.b / norm_sq,
        half_volume.surface.d * half_volume.surface.c / norm_sq
    )

    surface = Geom_Plane(point, normal_dir)

    # generate unfbound face
    face_unbound: TopoDS_Face = BRepBuilderAPI_MakeFace(surface, decom_params.make_face_tol).Face()

    # generate ocp half space from unbound face
    half_space_ocp = generate_ocp_half_space_from_face(
        face_unbound, 
        bounding_box_ocp, 
        decom_params, 
        reverse_halfspace=half_volume.direction=='+')

    # return the surface
    return half_space_ocp

def _generate_ocp_solid_from_mcio_half_volume_xplane(half_volume: HalfVolume, bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Solid:
    """Generate an OCP half-space solid for an ``XPlane`` half-volume.

    Parameters
    ----------
    half_volume : HalfVolume
        MCIO half-volume whose surface is a plane normal to the x axis at ``x0`` and
        whose direction (``'+'`` or ``'-'``) selects the side of the surface.
    bounding_box_ocp : TopoDS_Solid
        OCP box solid that the half-space is clipped to.
    decom_params : DecompositionParameters
        Decomposition parameters supplying tolerances.

    Returns
    -------
    TopoDS_Solid
        Half-space solid clipped to the bounding box.
    """
    # produce point and dir
    point = gp_Pnt(half_volume.surface.x0, 0, 0)
    dir = gp_Dir(gp_Vec(-1, 0, 0))

    # produce plane and surface
    surface = Geom_Plane(point, dir)

    # generate unfbound face
    face_unbound: TopoDS_Face = BRepBuilderAPI_MakeFace(surface, decom_params.make_face_tol).Face()

    # generate ocp half space from unbound face
    half_space_ocp = generate_ocp_half_space_from_face(
        face_unbound, 
        bounding_box_ocp, 
        decom_params, 
        reverse_halfspace=half_volume.direction=='-')

    # return the surface
    return half_space_ocp

def _generate_ocp_solid_from_mcio_half_volume_yplane(half_volume: HalfVolume, bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Solid:
    """Generate an OCP half-space solid for a ``YPlane`` half-volume.

    Parameters
    ----------
    half_volume : HalfVolume
        MCIO half-volume whose surface is a plane normal to the y axis at ``y0`` and
        whose direction (``'+'`` or ``'-'``) selects the side of the surface.
    bounding_box_ocp : TopoDS_Solid
        OCP box solid that the half-space is clipped to.
    decom_params : DecompositionParameters
        Decomposition parameters supplying tolerances.

    Returns
    -------
    TopoDS_Solid
        Half-space solid clipped to the bounding box.
    """
    # produce point and dir
    point = gp_Pnt(0, half_volume.surface.y0, 0)
    dir = gp_Dir(gp_Vec(0, -1, 0))

    # produce plane and surface
    surface = Geom_Plane(point, dir)

    # generate unfbound face
    face_unbound: TopoDS_Face = BRepBuilderAPI_MakeFace(surface, decom_params.make_face_tol).Face()

    # generate ocp half space from unbound face
    half_space_ocp = generate_ocp_half_space_from_face(
        face_unbound, 
        bounding_box_ocp, 
        decom_params, 
        reverse_halfspace=half_volume.direction=='-')

    # return the surface
    return half_space_ocp

def _generate_ocp_solid_from_mcio_half_volume_zplane(half_volume: HalfVolume, bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Solid:
    """Generate an OCP half-space solid for a ``ZPlane`` half-volume.

    Parameters
    ----------
    half_volume : HalfVolume
        MCIO half-volume whose surface is a plane normal to the z axis at ``z0`` and
        whose direction (``'+'`` or ``'-'``) selects the side of the surface.
    bounding_box_ocp : TopoDS_Solid
        OCP box solid that the half-space is clipped to.
    decom_params : DecompositionParameters
        Decomposition parameters supplying tolerances.

    Returns
    -------
    TopoDS_Solid
        Half-space solid clipped to the bounding box.
    """
    # produce point and dir
    point = gp_Pnt(0, 0, half_volume.surface.z0)
    dir = gp_Dir(gp_Vec(0, 0, -1))

    # produce plane and surface
    surface = Geom_Plane(point, dir)

    # generate unfbound face
    face_unbound: TopoDS_Face = BRepBuilderAPI_MakeFace(surface, decom_params.make_face_tol).Face()

    # generate ocp half space from unbound face
    half_space_ocp = generate_ocp_half_space_from_face(
        face_unbound, 
        bounding_box_ocp, 
        decom_params, 
        reverse_halfspace=half_volume.direction=='-')

    # return the surface
    return half_space_ocp

def _generate_ocp_solid_from_mcio_half_volume_sphere_at_origin(half_volume: HalfVolume, bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Solid:
    """Generate an OCP half-space solid for a ``SphereAtOrigin`` half-volume.

    Parameters
    ----------
    half_volume : HalfVolume
        MCIO half-volume whose surface is a sphere centred on the origin and whose
        direction (``'+'`` or ``'-'``) selects the side of the surface.
    bounding_box_ocp : TopoDS_Solid
        OCP box solid that the half-space is clipped to.
    decom_params : DecompositionParameters
        Decomposition parameters supplying tolerances.

    Returns
    -------
    TopoDS_Solid
        Half-space solid clipped to the bounding box.
    """
    # generate the points, direction and axis for the sphere
    point = gp_Pnt(0, 0, 0)
    dir = gp_Dir(0, 0, 1)
    ax = gp_Ax3(point, dir)

    # generate surface
    surface = Geom_SphericalSurface(ax, half_volume.surface.radius)

    # generate unfbound face
    face_unbound: TopoDS_Face = BRepBuilderAPI_MakeFace(surface, decom_params.make_face_tol).Face()

    # generate ocp half space from unbound face
    half_space_ocp = generate_ocp_half_space_from_face(
        face_unbound, 
        bounding_box_ocp, 
        decom_params, 
        reverse_halfspace=half_volume.direction=='+')

    # return the surface
    return half_space_ocp

def _generate_ocp_solid_from_mcio_half_volume_sphere(half_volume: HalfVolume, bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Solid:
    """Generate an OCP half-space solid for a ``Sphere`` half-volume.

    Parameters
    ----------
    half_volume : HalfVolume
        MCIO half-volume whose surface is a sphere centred at ``(x0, y0, z0)`` and whose
        direction (``'+'`` or ``'-'``) selects the side of the surface.
    bounding_box_ocp : TopoDS_Solid
        OCP box solid that the half-space is clipped to.
    decom_params : DecompositionParameters
        Decomposition parameters supplying tolerances.

    Returns
    -------
    TopoDS_Solid
        Half-space solid clipped to the bounding box.
    """
    # generate the points, direction and axis for the sphere
    point = gp_Pnt(half_volume.surface.x0, half_volume.surface.y0, half_volume.surface.z0)
    dir = gp_Dir(0, 0, 1)
    ax = gp_Ax3(point, dir)

    # generate surface
    surface = Geom_SphericalSurface(ax, half_volume.surface.radius)

    # generate unfbound face
    face_unbound: TopoDS_Face = BRepBuilderAPI_MakeFace(surface, decom_params.make_face_tol).Face()

    # generate ocp half space from unbound face
    half_space_ocp = generate_ocp_half_space_from_face(
        face_unbound, 
        bounding_box_ocp, 
        decom_params, 
        reverse_halfspace=half_volume.direction=='+')

    # return the surface
    return half_space_ocp

def _half_space_box(bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Solid:
    """Return the box half spaces are clipped to.

    This is the same box ``generate_ocp_half_space_from_face`` uses: the cubic
    box around ``bounding_box_ocp`` scaled by
    ``decom_params.bounding_box_multiplier_half_space``.

    Parameters
    ----------
    bounding_box_ocp : TopoDS_Solid
        OCP box solid that the half-space is clipped to.
    decom_params : DecompositionParameters
        Decomposition parameters supplying the bounding box multiplier.

    Returns
    -------
    TopoDS_Solid
        The box solid.
    """
    bounding_box = bounding_box_from_shape(bounding_box_ocp).make_cubic() * decom_params.bounding_box_multiplier_half_space
    return ocp_box_from_bounding_box(bounding_box)

def _ocp_half_space_from_primitive(primitives: list[TopoDS_Shape], direction: str, box: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Shape:
    """Clip the inside (-) or outside (+) of primitive solids to a box.

    Parameters
    ----------
    primitives : list[TopoDS_Shape]
        Solids whose union is the inside of the surface within the box.
    direction : str
        ``'-'`` for the inside of the primitives, ``'+'`` for the outside.
    box : TopoDS_Solid
        Box the half space is clipped to.
    decom_params : DecompositionParameters
        Decomposition parameters supplying the boolean fuzzy value.

    Returns
    -------
    TopoDS_Shape
        A solid, or a compound if the half space is in several pieces.
    """
    if direction == '-':
        solids = []
        for primitive in primitives:
            solids.extend(_common_ocp_solid(box, primitive, decom_params).solids)
    else:
        solids = [box]
        for primitive in primitives:
            solids = [s for solid in solids for s in _cut_ocp_solid(solid, primitive, decom_params).solids]

    if len(solids) == 0:
        msg = f'half space ({direction}) of primitive solids does not intersect the bounding box'
        logger.fatal(msg)
        raise Exception(msg)

    if len(solids) == 1:
        return solids[0]

    # half space in several pieces (e.g. both nappes of a cone), return as a compound
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    for solid in solids:
        builder.Add(compound, solid)

    return compound

def _generate_ocp_solid_from_cone(half_volume: HalfVolume, apex: tuple[float, float, float], axis: tuple[float, float, float], bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Shape:
    """Generate an OCP half-space solid for a cone with its axis along a cartesian axis.

    Each nappe of the cone is built as a primitive cone from the apex, long
    enough to pass through the whole bounding box. A sheet of +1 or -1 keeps
    only the nappe on the positive or negative side of the apex along the
    axis, a sheet of 0 keeps both. The inside of the cone is the negative
    half space, as for MCNP. The half angle is ``atan(sqrt(t_sq))``.

    Parameters
    ----------
    half_volume : HalfVolume
        MCIO half-volume whose surface is a cone (with ``t_sq`` and ``sheet``) and
        whose direction (``'+'`` or ``'-'``) selects the side of the surface.
    apex : tuple[float, float, float]
        Apex of the cone.
    axis : tuple[float, float, float]
        Unit vector along the cone axis (the positive sheet direction).
    bounding_box_ocp : TopoDS_Solid
        OCP box solid that the half-space is clipped to.
    decom_params : DecompositionParameters
        Decomposition parameters supplying tolerances.

    Returns
    -------
    TopoDS_Shape
        Half-space solid (or compound of solids) clipped to the bounding box.
    """
    box = _half_space_box(bounding_box_ocp, decom_params)
    box_bounds = bounding_box_from_shape(box)

    # nappe length reaching beyond the far side of the box from the apex
    centre = box_bounds.centre
    half_diagonal = 0.5 * math.sqrt(box_bounds.x_length ** 2 + box_bounds.y_length ** 2 + box_bounds.z_length ** 2)
    height = 1.01 * (math.dist(apex, centre) + half_diagonal)
    radius = height * math.sqrt(half_volume.surface.t_sq)

    sheet = getattr(half_volume.surface, 'sheet', 0)
    sheet_signs = {1: [1.0], -1: [-1.0]}.get(int(sheet), [1.0, -1.0])

    nappes = []
    for sign in sheet_signs:
        ax2 = gp_Ax2(gp_Pnt(*apex), gp_Dir(sign * axis[0], sign * axis[1], sign * axis[2]))
        nappes.append(BRepPrimAPI_MakeCone(ax2, 0.0, radius, height).Solid())

    return _ocp_half_space_from_primitive(nappes, half_volume.direction, box, decom_params)

def _generate_ocp_solid_from_cylinder(half_volume: HalfVolume, point: tuple[float, float, float], axis: tuple[float, float, float], bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Shape:
    """Generate an OCP half-space solid for a cylinder with its axis along a cartesian axis.

    The cylinder is built as a primitive solid long enough to pass through the
    whole bounding box, and the inside (-) or outside (+) is clipped to the box.

    Parameters
    ----------
    half_volume : HalfVolume
        MCIO half-volume whose surface is a cylinder (with ``radius``) and whose
        direction (``'+'`` or ``'-'``) selects the side of the surface.
    point : tuple[float, float, float]
        A point on the cylinder axis.
    axis : tuple[float, float, float]
        Unit vector along the cylinder axis.
    bounding_box_ocp : TopoDS_Solid
        OCP box solid that the half-space is clipped to.
    decom_params : DecompositionParameters
        Decomposition parameters supplying tolerances.

    Returns
    -------
    TopoDS_Shape
        Half-space solid clipped to the bounding box.
    """
    box = _half_space_box(bounding_box_ocp, decom_params)
    box_bounds = bounding_box_from_shape(box)
    half_diagonal = 0.5 * math.sqrt(box_bounds.x_length ** 2 + box_bounds.y_length ** 2 + box_bounds.z_length ** 2)
    reach = 1.01 * (math.dist(point, box_bounds.centre) + half_diagonal)

    base = gp_Pnt(*(point[i] - reach * axis[i] for i in range(3)))
    cylinder = BRepPrimAPI_MakeCylinder(gp_Ax2(base, gp_Dir(*axis)), half_volume.surface.radius, 2.0 * reach).Solid()

    return _ocp_half_space_from_primitive([cylinder], half_volume.direction, box, decom_params)

def generate_ocp_solid_from_mcio_half_volume_cylinder_on_x(half_volume: HalfVolume, bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Shape:
    """Generate an OCP half-space solid for a cylinder on the x axis (CX)."""
    return _generate_ocp_solid_from_cylinder(half_volume, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), bounding_box_ocp, decom_params)

def generate_ocp_solid_from_mcio_half_volume_cylinder_on_y(half_volume: HalfVolume, bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Shape:
    """Generate an OCP half-space solid for a cylinder on the y axis (CY)."""
    return _generate_ocp_solid_from_cylinder(half_volume, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), bounding_box_ocp, decom_params)

def generate_ocp_solid_from_mcio_half_volume_cylinder_on_z(half_volume: HalfVolume, bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Shape:
    """Generate an OCP half-space solid for a cylinder on the z axis (CZ)."""
    return _generate_ocp_solid_from_cylinder(half_volume, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), bounding_box_ocp, decom_params)

def generate_ocp_solid_from_mcio_half_volume_cylinder_parallel_to_x(half_volume: HalfVolume, bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Shape:
    """Generate an OCP half-space solid for a cylinder parallel to the x axis (C/X)."""
    surface = half_volume.surface
    return _generate_ocp_solid_from_cylinder(half_volume, (0.0, surface.y0, surface.z0), (1.0, 0.0, 0.0), bounding_box_ocp, decom_params)

def generate_ocp_solid_from_mcio_half_volume_cylinder_parallel_to_y(half_volume: HalfVolume, bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Shape:
    """Generate an OCP half-space solid for a cylinder parallel to the y axis (C/Y)."""
    surface = half_volume.surface
    return _generate_ocp_solid_from_cylinder(half_volume, (surface.x0, 0.0, surface.z0), (0.0, 1.0, 0.0), bounding_box_ocp, decom_params)

def generate_ocp_solid_from_mcio_half_volume_cylinder_parallel_to_z(half_volume: HalfVolume, bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Shape:
    """Generate an OCP half-space solid for a cylinder parallel to the z axis (C/Z)."""
    surface = half_volume.surface
    return _generate_ocp_solid_from_cylinder(half_volume, (surface.x0, surface.y0, 0.0), (0.0, 0.0, 1.0), bounding_box_ocp, decom_params)

def _generate_ocp_solid_from_mcio_half_volume_cone_parallel_to_x_axis(half_volume: HalfVolume, bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Shape:
    """Generate an OCP half-space solid for a cone parallel to the x axis (K/X)."""
    surface = half_volume.surface
    return _generate_ocp_solid_from_cone(half_volume, (surface.x0, surface.y0, surface.z0), (1.0, 0.0, 0.0), bounding_box_ocp, decom_params)

def _generate_ocp_solid_from_mcio_half_volume_cone_parallel_to_y_axis(half_volume: HalfVolume, bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Shape:
    """Generate an OCP half-space solid for a cone parallel to the y axis (K/Y)."""
    surface = half_volume.surface
    return _generate_ocp_solid_from_cone(half_volume, (surface.x0, surface.y0, surface.z0), (0.0, 1.0, 0.0), bounding_box_ocp, decom_params)

def _generate_ocp_solid_from_mcio_half_volume_cone_parallel_to_z_axis(half_volume: HalfVolume, bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Shape:
    """Generate an OCP half-space solid for a cone parallel to the z axis (K/Z)."""
    surface = half_volume.surface
    return _generate_ocp_solid_from_cone(half_volume, (surface.x0, surface.y0, surface.z0), (0.0, 0.0, 1.0), bounding_box_ocp, decom_params)

def _generate_ocp_solid_from_mcio_half_volume_cone_on_x_axis(half_volume: HalfVolume, bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Shape:
    """Generate an OCP half-space solid for a cone on the x axis (KX)."""
    return _generate_ocp_solid_from_cone(half_volume, (half_volume.surface.x0, 0.0, 0.0), (1.0, 0.0, 0.0), bounding_box_ocp, decom_params)

def _generate_ocp_solid_from_mcio_half_volume_cone_on_y_axis(half_volume: HalfVolume, bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Shape:
    """Generate an OCP half-space solid for a cone on the y axis (KY)."""
    return _generate_ocp_solid_from_cone(half_volume, (0.0, half_volume.surface.y0, 0.0), (0.0, 1.0, 0.0), bounding_box_ocp, decom_params)

def _generate_ocp_solid_from_mcio_half_volume_cone_on_z_axis(half_volume: HalfVolume, bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Shape:
    """Generate an OCP half-space solid for a cone on the z axis (KZ)."""
    return _generate_ocp_solid_from_cone(half_volume, (0.0, 0.0, half_volume.surface.z0), (0.0, 0.0, 1.0), bounding_box_ocp, decom_params)

def _generate_ocp_solid_from_torus(half_volume: HalfVolume, axis: tuple[float, float, float], radial: tuple[float, float, float], bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Shape:
    """Generate an OCP half-space solid for a torus with its axis along a cartesian axis.

    Uses the MCNP definition: ``a`` is the major radius, ``b`` the semi-axis of
    the elliptical cross section parallel to the axis and ``c`` the semi-axis
    perpendicular to it. The cross section is built in the plane of the axis
    and ``radial``, trimmed to the side of the axis it lies on (so degenerate
    tori with ``c >= a`` are handled), and revolved about the axis. The inside
    of the torus is the negative half space.

    Parameters
    ----------
    half_volume : HalfVolume
        MCIO half-volume whose surface is a torus and whose direction (``'+'`` or
        ``'-'``) selects the side of the surface.
    axis : tuple[float, float, float]
        Unit vector along the torus axis.
    radial : tuple[float, float, float]
        Unit vector perpendicular to the axis.
    bounding_box_ocp : TopoDS_Solid
        OCP box solid that the half-space is clipped to.
    decom_params : DecompositionParameters
        Decomposition parameters supplying tolerances.

    Returns
    -------
    TopoDS_Shape
        Half-space solid clipped to the bounding box.
    """
    surface = half_volume.surface
    a, b, c = surface.a, surface.b, surface.c
    centre = gp_Pnt(surface.x0, surface.y0, surface.z0)
    axis_dir = gp_Dir(*axis)
    radial_dir = gp_Dir(*radial)

    if b == c and a > c:
        # circular ring torus
        torus = BRepPrimAPI_MakeTorus(gp_Ax2(centre, axis_dir, radial_dir), a, c).Solid()
    else:
        # elliptical cross section centred at distance a from the axis
        ellipse_centre = centre.Translated(gp_Vec(radial_dir).Multiplied(a))
        normal = gp_Dir(gp_Vec(axis_dir).Crossed(gp_Vec(radial_dir)))
        if b >= c:
            ellipse = gp_Elips(gp_Ax2(ellipse_centre, normal, axis_dir), b, c)
        else:
            ellipse = gp_Elips(gp_Ax2(ellipse_centre, normal, radial_dir), c, b)
        edge = BRepBuilderAPI_MakeEdge(ellipse).Edge()
        cross_section = BRepBuilderAPI_MakeFace(BRepBuilderAPI_MakeWire(edge).Wire()).Face()

        if c >= a:
            # cross section crosses the axis, keep the part on the radial side of the axis
            size = 4.0 * (a + b + c)
            half_plane = BRepBuilderAPI_MakeFace(
                gp_Pln(gp_Ax3(centre, normal, radial_dir)), 0.0, size, -size, size).Face()
            cross_section = BRepAlgoAPI_Common(cross_section, half_plane).Shape()

        torus = BRepPrimAPI_MakeRevol(cross_section, gp_Ax1(centre, axis_dir), math.tau).Shape()
        torus = extract_ocp_shape_type_from_shape(torus, TopAbs_SOLID, cast_from_shape=True)[0]

    box = _half_space_box(bounding_box_ocp, decom_params)
    return _ocp_half_space_from_primitive([torus], half_volume.direction, box, decom_params)

def _generate_ocp_solid_from_mcio_half_volume_tori_on_x_axis(half_volume: HalfVolume, bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Shape:
    """Generate an OCP half-space solid for a torus with its axis parallel to x (TX)."""
    return _generate_ocp_solid_from_torus(half_volume, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), bounding_box_ocp, decom_params)

def _generate_ocp_solid_from_mcio_half_volume_tori_on_y_axis(half_volume: HalfVolume, bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Shape:
    """Generate an OCP half-space solid for a torus with its axis parallel to y (TY)."""
    return _generate_ocp_solid_from_torus(half_volume, (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), bounding_box_ocp, decom_params)

def _generate_ocp_solid_from_mcio_half_volume_tori_on_z_axis(half_volume: HalfVolume, bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Shape:
    """Generate an OCP half-space solid for a torus with its axis parallel to z (TZ)."""
    return _generate_ocp_solid_from_torus(half_volume, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), bounding_box_ocp, decom_params)

def _generate_ocp_solid_from_mcio_half_volume_sphere_on_axis(half_volume: HalfVolume, bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Solid:
    """Generate an OCP half-space solid for a sphere centred on a cartesian axis (SX, SY, SZ).

    The centre is taken from whichever of ``x0``, ``y0`` and ``z0`` the surface
    defines, the others being zero.

    Parameters
    ----------
    half_volume : HalfVolume
        MCIO half-volume whose surface is a sphere on an axis and whose direction
        (``'+'`` or ``'-'``) selects the side of the surface.
    bounding_box_ocp : TopoDS_Solid
        OCP box solid that the half-space is clipped to.
    decom_params : DecompositionParameters
        Decomposition parameters supplying tolerances.

    Returns
    -------
    TopoDS_Solid
        Half-space solid clipped to the bounding box.
    """
    surface = half_volume.surface
    point = gp_Pnt(getattr(surface, 'x0', 0.0), getattr(surface, 'y0', 0.0), getattr(surface, 'z0', 0.0))

    # generate surface and unbound face
    surface_ocp = Geom_SphericalSurface(gp_Ax3(point, gp_Dir(0, 0, 1)), surface.radius)
    face_unbound: TopoDS_Face = BRepBuilderAPI_MakeFace(surface_ocp, decom_params.make_face_tol).Face()

    # generate ocp half space from unbound face
    return generate_ocp_half_space_from_face(
        face_unbound,
        bounding_box_ocp,
        decom_params,
        reverse_halfspace=half_volume.direction=='+')

def _generate_ocp_solid_from_mcio_half_volume_general_quadratic(half_volume: HalfVolume, bounding_box_ocp: TopoDS_Solid, decom_params: DecompositionParameters) -> TopoDS_Shape:
    """Generate an OCP half-space solid for a general quadratic (GQ) surface.

    The quadric is reduced to its principal axes and built from a primitive
    solid: an ellipsoid, an elliptic cylinder or an elliptic cone (both
    nappes). Circular cylinders and cones are built exactly; elliptical
    sections and ellipsoids are made by scaling a primitive, which gives
    B-spline faces. Hyperboloids and paraboloids are not supported.

    Parameters
    ----------
    half_volume : HalfVolume
        MCIO half-volume whose surface is a general quadratic and whose direction
        (``'+'`` or ``'-'``) selects the side of the surface.
    bounding_box_ocp : TopoDS_Solid
        OCP box solid that the half-space is clipped to.
    decom_params : DecompositionParameters
        Decomposition parameters supplying tolerances.

    Returns
    -------
    TopoDS_Shape
        Half-space solid clipped to the bounding box.

    Raises
    ------
    ValueError
        If the quadric is not an ellipsoid, elliptic cylinder or elliptic cone.
    """
    s = half_volume.surface
    quadratic = np.array([[s.a, s.d / 2, s.f / 2],
                          [s.d / 2, s.b, s.e / 2],
                          [s.f / 2, s.e / 2, s.c]])
    linear_half = np.array([s.g, s.h, s.j]) / 2.0
    constant = s.k

    eigenvalues, eigenvectors = np.linalg.eigh(quadratic)
    scale = np.abs(eigenvalues).max()

    # decks give GQ coefficients rounded to ~7 significant figures, so small eigenvalues are zero
    zero = np.abs(eigenvalues) < 1e-5 * scale
    eigenvalues = np.where(zero, 0.0, eigenvalues)
    quadratic = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    box = _half_space_box(bounding_box_ocp, decom_params)
    box_bounds = bounding_box_from_shape(box)
    box_centre = np.array(box_bounds.centre)
    half_diagonal = 0.5 * math.sqrt(box_bounds.x_length ** 2 + box_bounds.y_length ** 2 + box_bounds.z_length ** 2)

    # centre (or a point on the axis), f = (x - c)^T M (x - c) + k'
    centre = -np.linalg.pinv(quadratic) @ linear_half
    reduced_constant = constant + linear_half @ centre
    if zero.sum() == 1 and abs(linear_half @ eigenvectors[:, zero][:, 0]) > 1e-5 * max(1.0, np.abs(linear_half).max()):
        raise ValueError('general quadratic is a paraboloid, which is not supported')
    if zero.sum() > 1:
        raise ValueError('general quadratic with more than one zero eigenvalue is not supported')

    def frame(axis_index: int, x_index: int) -> gp_Ax2:
        """Axis system at the centre with z and x along the given eigenvectors."""
        return gp_Ax2(gp_Pnt(*centre), gp_Dir(*eigenvectors[:, axis_index]), gp_Dir(*eigenvectors[:, x_index]))

    def scaled(shape: TopoDS_Shape, ax2: gp_Ax2, sx: float, sy: float) -> TopoDS_Shape:
        """Scale a shape built with radius sy about ax2 so its x direction has radius sx."""
        if abs(sx - sy) <= 1e-5 * max(sx, sy):
            return shape
        to_local = gp_Trsf()
        to_local.SetTransformation(gp_Ax3(ax2))
        local = BRepBuilderAPI_Transform(shape, to_local, True).Shape()
        gtrsf = gp_GTrsf()
        gtrsf.SetValue(1, 1, sx / sy)
        stretched = BRepBuilderAPI_GTransform(local, gtrsf, True).Shape()
        return BRepBuilderAPI_Transform(stretched, to_local.Inverted(), True).Shape()

    positive = eigenvalues > 0
    reach = 1.01 * (np.linalg.norm(centre - box_centre) + half_diagonal)

    if zero.sum() == 1:
        # elliptic cylinder along the zero eigenvector
        axis_index = int(np.argmax(zero))
        others = [i for i in range(3) if i != axis_index]
        if positive[others[0]] != positive[others[1]]:
            raise ValueError('general quadratic is a hyperbolic cylinder, which is not supported')
        sign = 1.0 if positive[others[0]] else -1.0
        if reduced_constant * sign >= 0:
            raise ValueError('general quadratic cylinder has no real surface')
        r1 = math.sqrt(-reduced_constant / eigenvalues[others[0]])
        r2 = math.sqrt(-reduced_constant / eigenvalues[others[1]])
        ax2 = frame(axis_index, others[0])
        base = gp_Ax2(ax2.Location().Translated(gp_Vec(ax2.Direction()).Multiplied(-reach)), ax2.Direction(), ax2.XDirection())
        primitives = [scaled(BRepPrimAPI_MakeCylinder(base, r2, 2.0 * reach).Solid(), base, r1, r2)]
        inside_is_negative = sign > 0

    elif positive.all() or (~positive).all():
        # ellipsoid
        sign = 1.0 if positive[0] else -1.0
        if reduced_constant * sign >= 0:
            raise ValueError('general quadratic ellipsoid has no real surface')
        radii = np.sqrt(-reduced_constant / eigenvalues)
        ax2 = frame(2, 0)
        sphere = BRepPrimAPI_MakeSphere(ax2, 1.0).Solid()
        to_local = gp_Trsf()
        to_local.SetTransformation(gp_Ax3(ax2))
        gtrsf = gp_GTrsf()
        # local x, y, z are eigenvectors 0, 1, 2
        gtrsf.SetValue(1, 1, radii[0])
        gtrsf.SetValue(2, 2, radii[1])
        gtrsf.SetValue(3, 3, radii[2])
        local = BRepBuilderAPI_Transform(sphere, to_local, True).Shape()
        stretched = BRepBuilderAPI_GTransform(local, gtrsf, True).Shape()
        primitives = [BRepBuilderAPI_Transform(stretched, to_local.Inverted(), True).Shape()]
        inside_is_negative = sign > 0

    else:
        # two eigenvalues of one sign and one of the other: a cone if k' is zero
        if abs(reduced_constant) > 1e-5 * max(1.0, scale * float(centre @ centre)):
            raise ValueError('general quadratic is a hyperboloid, which is not supported')
        odd = int(np.argmax(positive)) if positive.sum() == 1 else int(np.argmax(~positive))
        others = [i for i in range(3) if i != odd]
        tan_1 = math.sqrt(abs(eigenvalues[odd]) / abs(eigenvalues[others[0]]))
        tan_2 = math.sqrt(abs(eigenvalues[odd]) / abs(eigenvalues[others[1]]))
        ax2 = frame(odd, others[0])
        primitives = []
        for direction_sign in (1.0, -1.0):
            nappe_axis = gp_Ax2(ax2.Location(), gp_Dir(gp_Vec(ax2.Direction()).Multiplied(direction_sign)), ax2.XDirection())
            nappe = BRepPrimAPI_MakeCone(nappe_axis, 0.0, reach * tan_2, reach).Solid()
            primitives.append(scaled(nappe, nappe_axis, reach * tan_1, reach * tan_2))
        # f < 0 inside the nappes when the pair of eigenvalues around the axis is positive
        inside_is_negative = bool(positive[others[0]])

    direction = half_volume.direction
    if not inside_is_negative:
        direction = '+' if direction == '-' else '-'

    return _ocp_half_space_from_primitive(primitives, direction, box, decom_params)

SURFACE_TYPE_LOOKUP: dict[str, Callable[[HalfVolume, TopoDS_Solid, DecompositionParameters], TopoDS_Solid]] = {
    "Plane": _generate_ocp_solid_from_mcio_half_volume_plane,
    "XPlane": _generate_ocp_solid_from_mcio_half_volume_xplane,
    "YPlane": _generate_ocp_solid_from_mcio_half_volume_yplane,
    "ZPlane": _generate_ocp_solid_from_mcio_half_volume_zplane,
    "Sphere": _generate_ocp_solid_from_mcio_half_volume_sphere,
    "SphereAtOrigin": _generate_ocp_solid_from_mcio_half_volume_sphere_at_origin,
    "SphereOnXAxis": _generate_ocp_solid_from_mcio_half_volume_sphere_on_axis,
    "SphereOnYAxis": _generate_ocp_solid_from_mcio_half_volume_sphere_on_axis,
    "SphereOnZAxis": _generate_ocp_solid_from_mcio_half_volume_sphere_on_axis,
    "CylinderOnXAxis": generate_ocp_solid_from_mcio_half_volume_cylinder_on_x,
    "CylinderOnYAxis": generate_ocp_solid_from_mcio_half_volume_cylinder_on_y,
    "CylinderOnZAxis": generate_ocp_solid_from_mcio_half_volume_cylinder_on_z,
    "CylinderParallelToXAxis": generate_ocp_solid_from_mcio_half_volume_cylinder_parallel_to_x,
    "CylinderParallelToYAxis": generate_ocp_solid_from_mcio_half_volume_cylinder_parallel_to_y,
    "CylinderParallelToZAxis": generate_ocp_solid_from_mcio_half_volume_cylinder_parallel_to_z,
    "ConeParallelToXAxis": _generate_ocp_solid_from_mcio_half_volume_cone_parallel_to_x_axis,
    "ConeParallelToYAxis": _generate_ocp_solid_from_mcio_half_volume_cone_parallel_to_y_axis,
    "ConeParallelToZAxis": _generate_ocp_solid_from_mcio_half_volume_cone_parallel_to_z_axis,
    "ConeOnXAxis": _generate_ocp_solid_from_mcio_half_volume_cone_on_x_axis,
    "ConeOnYAxis": _generate_ocp_solid_from_mcio_half_volume_cone_on_y_axis,
    "ConeOnZAxis": _generate_ocp_solid_from_mcio_half_volume_cone_on_z_axis,
    "ToriParralleltoXAxis": _generate_ocp_solid_from_mcio_half_volume_tori_on_x_axis,
    "ToriParralleltoYAxis": _generate_ocp_solid_from_mcio_half_volume_tori_on_y_axis,
    "ToriParralleltoZAxis": _generate_ocp_solid_from_mcio_half_volume_tori_on_z_axis,
    "GeneralQuadratic": _generate_ocp_solid_from_mcio_half_volume_general_quadratic
}
