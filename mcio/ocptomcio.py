"""Conversion of OpenCASCADE (OCP) faces to mcio half-spaces.

Each supported OCCT analytic surface (plane, cylinder, sphere, cone, torus) is
converted to the equivalent mcio surface, de-duplicated against a cache of
previously generated surfaces, and returned as the half-space (or volume, for
off-axis cones) lying on the material side of the face.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from mcio import logger

import numpy as np
import math


try:
    import OCP
    from OCP.BRepAdaptor import (
        BRepAdaptor_Surface
    )
    from OCP.BRepGProp import (
        BRepGProp
    )
    from OCP.BRepLProp import (
        BRepLProp_SLProps
    )
    from OCP.ElSLib import (
        ElSLib
    )
    from OCP.GC import GC_MakePlane
    from OCP.Geom import (
        Geom_Surface,
        Geom_Plane, 
        Geom_CylindricalSurface,
        Geom_SphericalSurface,
        Geom_ConicalSurface,
        Geom_ToroidalSurface,
        Geom_Line
    )
    from OCP.GeomAPI import (
        GeomAPI_ProjectPointOnSurf,
        GeomAPI_IntCS
    )
    from OCP.GeomAdaptor import (
        GeomAdaptor_Surface
    )
    from OCP.gp import (
        gp_Pnt,
        gp_Vec,
        gp_Dir,
        gp_Ax3
    )
    from OCP.GProp import (
        GProp_GProps
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
        TopAbs_SHAPE
    )

    from OCP.TopoDS import (
        TopoDS,
        TopoDS_Face
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
    from OCP.gp import gp_Ax1
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.TopoDS import TopoDS_Shape
    from .surface import Surface
    from .volume import HalfVolume, Volume

from .decomposeparams import DecompositionParameters
from .geometric import (
    calculate_plane_parameters_from_point_and_normal,
    check_vector_on_axis,
    angle_between_vectors,
    check_point_on_axis,
    check_vector_parallel_to_axis
)
from .numerical import custom_round
from .surface import (
    Plane, XPlane, YPlane, ZPlane,
    CylinderOnXAxis, CylinderOnYAxis, CylinderOnZAxis,
    CylinderParallelToXAxis, CylinderParallelToYAxis, CylinderParallelToZAxis,
    SphereAtOrigin, Sphere, SphereOnXAxis, SphereOnYAxis, SphereOnZAxis,
    ConeOnXAxis, ConeOnYAxis, ConeOnZAxis, 
    ConeParallelToXAxis, ConeParallelToYAxis, ConeParallelToZAxis,
    ToriParralleltoXAxis, ToriParralleltoYAxis, ToriParralleltoZAxis,
    GeneralQuadratic
)
from .ocpbooleans import (
    _get_ocp_surface_from_ocp_face
)
from .ocpproperties import (
    rotation_matrix_from_ocp_vectors,
    _get_uv_bounds_from_ocp_face
)



def get_mcio_halfspace_from_ocp_face(face: TopoDS_Face, decom_params: DecompositionParameters, surfaces_generated: dict[type[Surface], dict[str, Surface]]={}, reverse_direction: bool=False) -> HalfVolume | Volume:
    """Convert an OCP face to the mcio half-space bounded by its surface.

    Dispatches on the type of the face's underlying ``Geom_Surface`` (plane,
    cylinder, sphere, cone or torus).

    Parameters
    ----------
    face : TopoDS_Face
        Face to convert.
    decom_params : DecompositionParameters
        Decomposition parameters supplying rounding tolerances.
    surfaces_generated : dict[type[Surface], dict[str, Surface]], optional
        Cache of already generated mcio surfaces, keyed by surface class and then
        by surface hash string; new surfaces are added in place. Must contain
        an entry for every surface class that may be produced, by default {}
    reverse_direction : bool, optional
        If True, return the complement (``~``) of the half-space, by default False

    Returns
    -------
    HalfVolume | Volume
        Half-space for the face; off-axis cones return a compound Volume.

    Raises
    ------
    Exception
        If the face's surface type is not supported.
    """
    # get surface from face
    surface = _get_ocp_surface_from_ocp_face(face)

    # group methods for generating half spaces from face
    HALFSPACE_METHODS: dict[type[Geom_Surface], Callable[..., HalfVolume | Volume]] = {
        Geom_Plane: _get_mcio_halfspace_plane_from_ocp_face,
        Geom_CylindricalSurface: _get_mcio_halfspace_cylinder_from_ocp_face,
        Geom_SphericalSurface: _get_mcio_halfspace_sphere_from_ocp_face,
        Geom_ConicalSurface: _get_mcio_halfspace_cone_from_ocp_face,
        Geom_ToroidalSurface: _get_mcio_halfspace_tori_from_ocp_face
    }

    # call method for generating half space, reversing if required by the user.
    if type(surface) in HALFSPACE_METHODS:
        if reverse_direction == False:
            return HALFSPACE_METHODS[type(surface)](face, decom_params, surfaces_generated=surfaces_generated)
        else:
            return ~HALFSPACE_METHODS[type(surface)](face, decom_params, surfaces_generated=surfaces_generated)

    else:
        msg = f'non-encoded surface detected: {type(surface)}'
        raise Exception(msg)

def _get_face_centroid(face: TopoDS_Face) -> gp_Pnt:
    """Calculate the area-weighted centre of mass of a face.

    Parameters
    ----------
    face : TopoDS_Face
        Face whose centroid is required.

    Returns
    -------
    gp_Pnt
        Centre of mass of the face surface.
    """
    # set up empty props instance
    props: GProp_GProps = GProp_GProps()

    # Calculate surface properties (area-weighted center of mass)
    BRepGProp.SurfaceProperties_s(face, props)

    # extract point at centre of mass of face
    centroid: gp_Pnt = props.CentreOfMass()

    return centroid

def project_point_onto_surface(point_pnt: gp_Pnt, surface: Geom_Surface) -> tuple[gp_Pnt, float]:
    """Project a 3D point orthogonally onto an OCP surface.

    Parameters
    ----------
    point_pnt : gp_Pnt
        Point to project.
    surface : Geom_Surface
        Surface to project onto.

    Returns
    -------
    tuple[gp_Pnt, float]
        Nearest point on the surface and its distance from ``point_pnt``.

    Raises
    ------
    ValueError
        If the projection yields no points.
    """
    projector = GeomAPI_ProjectPointOnSurf(point_pnt, surface)
    
    if projector.NbPoints() > 0:
        nearest_pnt = projector.NearestPoint()
        dist = projector.LowerDistance()
        return nearest_pnt, dist
    else:
        raise ValueError("Could not project center of mass onto surface.")


def intersect_normal_line_with_surface(point_pnt: gp_Pnt, normal_dir: gp_Dir, surface: Geom_Surface) -> tuple[gp_Pnt, float]:
    """Intersect an infinite line through a point along a direction with a surface.

    Parameters
    ----------
    point_pnt : gp_Pnt
        Point the line passes through.
    normal_dir : gp_Dir
        Direction of the line.
    surface : Geom_Surface
        Surface to intersect.

    Returns
    -------
    tuple[gp_Pnt, float]
        First intersection point found and its distance from ``point_pnt``.

    Raises
    ------
    ValueError
        If no intersection is found.
    """
    # 1. Create infinite line passing through center of mass along normal
    line_geom = Geom_Line(point_pnt, normal_dir)
    
    # 2. Intersect curve (line) with surface
    intersector = GeomAPI_IntCS(line_geom, surface)
    
    if intersector.IsDone() and intersector.NbPoints() > 0:
        # Take the first/closest intersection point
        int_pnt = intersector.Point(1)
        dist = point_pnt.Distance(int_pnt)
        return int_pnt, dist
    else:
        raise ValueError("No intersection found between normal line and surface.")

def _get_mcio_halfspace_plane_from_ocp_face(face: TopoDS_Face, decom_params: DecompositionParameters, surfaces_generated: dict[type[Surface], dict[str, Surface]]={}) -> HalfVolume:
    """Convert a planar face to an mcio plane half-space.

    Axis-aligned planes become ``XPlane``/``YPlane``/``ZPlane``; other planes
    become a general ``Plane`` with its normal flipped, if necessary, to lie
    within 90 degrees of +x so that coincident planes are de-duplicated.

    Parameters
    ----------
    face : TopoDS_Face
        Planar face to convert.
    decom_params : DecompositionParameters
        Decomposition parameters supplying plane rounding/on-axis tolerances.
    surfaces_generated : dict[type[Surface], dict[str, Surface]], optional
        Cache of already generated mcio surfaces, keyed by surface class and then
        by surface hash string; new surfaces are added in place. Must contain
        an entry for every surface class that may be produced, by default {}

    Returns
    -------
    HalfVolume
        Plane half-space on the material side of the face.
    """
    # get surface
    surface = _get_ocp_surface_from_ocp_face(face)

    # orientation of the face relative to the surface normal (u x v). The surface normal points away from the
    # axis (or along the plane axis) for a right handed axis system but the opposite way for a left handed
    # one, e.g. faces of mirrored solids built in OCP; STEP files always store right handed axes.
    orientation = face.Orientation()
    if not surface.Position().Direct():
        orientation = TopAbs_REVERSED if orientation == TopAbs_FORWARD else TopAbs_FORWARD

    # location = surface.Location()

    # get location by finding point at centre of face u, v range.
    u_min, u_max, v_min, v_max = _get_uv_bounds_from_ocp_face(face)
    u_mid = (u_min + u_max) / 2.0
    v_mid = (v_min + v_max) / 2.0
    # location_from_mid_uv = surface.Value(u_mid, v_mid)
    location_from_mid_uv = _get_face_centroid(face)

    # snal normal to plane
    # exact_pnt, dist_error = project_point_onto_surface(location_from_mid_uv, surface)

    # Debug reporting if numerical drift exceeds tolerance
    # if dist_error > 1e-5:
    #     print(f"[WARN] Center of mass drifted from surface by {dist_error:.8e} mm.")
    #     print(f"       Raw Centroid: ({location_from_mid_uv.X():.8f}, {location_from_mid_uv.Y():.8f}, {location_from_mid_uv.Z():.8f})")
    #     print(f"       Snapped Point: ({exact_pnt.X():.8f}, {exact_pnt.Y():.8f}, {exact_pnt.Z():.8f})")

    # x_delta = abs(location.X() - location_from_mid_uv.X())
    # y_delta = abs(location.Y() - location_from_mid_uv.Y())
    # z_delta = abs(location.Z() - location_from_mid_uv.Z())

    # dist = math.sqrt(x_delta ** 2 + y_delta ** 2 + z_delta ** 2)

    # DELTA_MAX = 100

    # if x_delta > DELTA_MAX:
    #     print(f'x_delta > {DELTA_MAX}')
    # if y_delta > DELTA_MAX:
    #     print(f'y_delta > {DELTA_MAX}')
    # if z_delta > DELTA_MAX:
    #     print(f'z_delta > {DELTA_MAX}')

    # if dist > DELTA_MAX:
    #     print(f'dist > {DELTA_MAX}'*10)

    # get normal
    normal = surface.Axis().Direction()

    # use underlying analytical surface

    # debugging
    # print(f'plane converting from face to mcio halfspace')
    # print(f' -> location.X() = {location.X():.8e}')
    # print(f' -> location_from_mid.X() = {location_from_mid_uv.X():.8e}')
    # print(f' -> location.Y() = {location.Y():.8e}')
    # print(f' -> location_from_mid.Y() = {location_from_mid_uv.Y():.8e}')
    # print(f' -> location.Z() = {location.Z():.8e}')
    # print(f' -> location_from_mid.Z() = {location_from_mid_uv.Z():.8e}')

    # adaptor alternative method
    adaptor = GeomAdaptor_Surface(surface)
    gp_plane = adaptor.Plane()

    u, v = ElSLib.Parameters_s(gp_plane, location_from_mid_uv)
    location_from_mid_uv = ElSLib.Value_s(u_mid, v_mid, gp_plane)

    # get general parameters for plane
    a, b, c, d = calculate_plane_parameters_from_point_and_normal(location_from_mid_uv.X(), location_from_mid_uv.Y(), location_from_mid_uv.Z(), normal.X(), normal.Y(), normal.Z(), reverse=True)

    # remove small rounding errors
    a = custom_round(a, decom_params.surf_plane_rounding_vector)
    b = custom_round(b, decom_params.surf_plane_rounding_vector)
    c = custom_round(c, decom_params.surf_plane_rounding_vector)
    d = custom_round(d, decom_params.surf_plane_rounding_lateral)

    # check for plane being on axis
    is_on_axis = check_vector_on_axis(a, b, c, decom_params.surf_plane_tol_on_axis)

    # check if surface already sonverted to mcio surfaces, else generate mcio surface
    if is_on_axis in ['on_x_pos', 'on_x_neg']:
        # print(f'rounding to {decom_params.surf_plane_rounding_lateral}')
        # print(f'raw = {location_from_mid_uv.X():.8e} - rounded = {custom_round(location_from_mid_uv.X(), decom_params.surf_plane_rounding_lateral):.8e}' )
        # surface_mcio = XPlane(x0=custom_round(location_from_mid_uv.X(), decom_params.surf_plane_rounding_lateral))

        # alternative
        x0 = custom_round(-d / (1.0 if "neg" in is_on_axis else -1.0), decom_params.surf_plane_rounding_lateral)
        surface_mcio = XPlane(x0=x0)

    elif is_on_axis in ['on_y_pos', 'on_y_neg']:
        # print(f'rounding to {decom_params.surf_plane_rounding_lateral}')
        # print(f'raw = {location_from_mid_uv.Y():.8e} - rounded = {custom_round(location_from_mid_uv.Y(), decom_params.surf_plane_rounding_lateral):.8e}' )
        # surface_mcio = YPlane(y0=custom_round(location_from_mid_uv.Y(), decom_params.surf_plane_rounding_lateral))

        y0 = custom_round(-d / (1.0 if "neg" in is_on_axis else -1.0), decom_params.surf_plane_rounding_lateral)
        surface_mcio = YPlane(y0=y0)
        
    elif is_on_axis in ['on_z_pos', 'on_z_neg']:
        # print(f'rounding to {decom_params.surf_plane_rounding_lateral}')
        # print(f'raw = {location_from_mid_uv.Z():.8e} - rounded = {custom_round(location_from_mid_uv.Z(), decom_params.surf_plane_rounding_lateral):.8e}' )
        # surface_mcio = ZPlane(z0=custom_round(location_from_mid_uv.Z(), decom_params.surf_plane_rounding_lateral))

        z0 = custom_round(-d / (1.0 if "neg" in is_on_axis else -1.0), decom_params.surf_plane_rounding_lateral)
        surface_mcio = ZPlane(z0=z0)
    else:
        # align all surfaces to have normals within 90 degrees of reference vector 1,0,0
        # this is designed to remove duplicate surface definitions for off axis 
        # planes that are coincident but flipped by 2 pi
        if angle_between_vectors([1, 0, 0], [a, b, c]) <= (math.pi / 2.0):
            surface_mcio = Plane(a=a, b=b, c=c, d=d)
            flip_surface = False
        else:
            surface_mcio = Plane(a=-a, b=-b, c=-c, d=-d)
            flip_surface = True

    # check if identical surface already defined
    if surface_mcio.__hash_string__() in surfaces_generated[type(surface_mcio)]:
        # if mcio surface already defined, use the previously definde
        surface_mcio = surfaces_generated[type(surface_mcio)][surface_mcio.__hash_string__()]
    else:
        # if mcio surface not already defined add this to dictionary
        surfaces_generated[type(surface_mcio)][surface_mcio.__hash_string__()] = surface_mcio

    # check the orientation and the on axis directions and return half space
    if 'pos' in is_on_axis:
        if orientation == TopAbs_FORWARD:
            return +surface_mcio
        else:
            return -surface_mcio
    elif 'neg' in is_on_axis:
        if orientation == TopAbs_FORWARD:
            return -surface_mcio
        else:
            return +surface_mcio
    else:
        if orientation == TopAbs_FORWARD and flip_surface == False:
            return +surface_mcio
        elif orientation == TopAbs_FORWARD and flip_surface:
            return -surface_mcio
        elif flip_surface:
            return +surface_mcio
        else:
            return -surface_mcio

# def _get_mcio_halfspace_plane_from_ocp_face(face, decom_params: DecompositionParameters, surfaces_generated: dict = None):
#     if surfaces_generated is None:
#         surfaces_generated = {}

#     surface = _get_ocp_surface_from_ocp_face(face)

#     # 1. Get Midpoint UV coordinates on the face
#     u_min, u_max, v_min, v_max = _get_uv_bounds_from_ocp_face(face)
#     u_mid = (u_min + u_max) / 2.0
#     v_mid = (v_min + v_max) / 2.0

#     # 2. Get underlying plane geometry via Adaptor
#     adaptor = GeomAdaptor_Surface(surface)
#     gp_plane = adaptor.Plane()

#     # Get local midpoint on plane
#     location_from_mid_uv = ElSLib.Value_s(u_mid, v_mid, gp_plane)

#     # 3. Calculate true topological outer normal evaluated directly at the face
#     brep_adaptor = BRepAdaptor_Surface(face)
#     props = BRepLProp_SLProps(brep_adaptor, u_mid, v_mid, 1, 1e-7)
    
#     if props.IsNormalDefined():
#         ocp_normal = props.Normal()
#         # BRepLProp_SLProps automatically accounts for face.Orientation()
#         normal_vec = [ocp_normal.X(), ocp_normal.Y(), ocp_normal.Z()]
#     else:
#         # Fallback manual calculation: Surface Normal x Orientation
#         raw_dir = gp_plane.Axis().Direction()
#         sign = -1.0 if face.Orientation() == TopAbs_REVERSED else 1.0
#         normal_vec = [raw_dir.X() * sign, raw_dir.Y() * sign, raw_dir.Z() * sign]

#     # 4. Get general plane parameters (Ax + By + Cz + D = 0) using true outer normal
#     a, b, c, d = calculate_plane_parameters_from_point_and_normal(
#         location_from_mid_uv.X(), location_from_mid_uv.Y(), location_from_mid_uv.Z(),
#         normal_vec[0], normal_vec[1], normal_vec[2],
#         reverse=False  # Do not reverse, normal_vec is already oriented outward
#     )

#     # Round plane coefficients
#     a = custom_round(a, decom_params.SURF_PLANE_ROUNDING_VECTOR)
#     b = custom_round(b, decom_params.SURF_PLANE_ROUNDING_VECTOR)
#     c = custom_round(c, decom_params.SURF_PLANE_ROUNDING_LATERAL)
#     d = custom_round(d, decom_params.SURF_PLANE_ROUNDING_LATERAL)

#     # 5. Check if plane is aligned with a principal axis
#     is_on_axis = check_vector_on_axis(a, b, c, decom_params.SURF_PLANE_TOL_ON_AXIS)

#     if is_on_axis in ['on_x_pos', 'on_x_neg']:
#         x0 = custom_round(-d / a if a != 0 else location_from_mid_uv.X(), decom_params.SURF_PLANE_ROUNDING_LATERAL)
#         surface_mcio = XPlane(x0=x0)
#     elif is_on_axis in ['on_y_pos', 'on_y_neg']:
#         y0 = custom_round(-d / b if b != 0 else location_from_mid_uv.Y(), decom_params.SURF_PLANE_ROUNDING_LATERAL)
#         surface_mcio = YPlane(y0=y0)
#     elif is_on_axis in ['on_z_pos', 'on_z_neg']:
#         z0 = custom_round(-d / c if c != 0 else location_from_mid_uv.Z(), decom_params.SURF_PLANE_ROUNDING_LATERAL)
#         surface_mcio = ZPlane(z0=z0)
#     else:
#         # Canonicalize normal orientation relative to reference vector [1, 0, 0]
#         if angle_between_vectors([1, 0, 0], [a, b, c]) <= (math.pi / 2.0):
#             surface_mcio = Plane(a=a, b=b, c=c, d=d)
#             flip_surface = False
#         else:
#             surface_mcio = Plane(a=-a, b=-b, c=-c, d=-d)
#             flip_surface = True

#     # 6. Deduplicate surfaces
#     surf_type = type(surface_mcio)
#     if surf_type not in surfaces_generated:
#         surfaces_generated[surf_type] = {}

#     hash_str = surface_mcio.__hash_string__()
#     if hash_str in surfaces_generated[surf_type]:
#         surface_mcio = surfaces_generated[surf_type][hash_str]
#     else:
#         surfaces_generated[surf_type][hash_str] = surface_mcio

#     # 7. Return half-space with consolidated orientation logic
#     if is_on_axis != 'none':
#         if 'pos' in is_on_axis:
#             return +surface_mcio
#         else:
#             return -surface_mcio
#     else:
#         return -surface_mcio if flip_surface else +surface_mcio

def _get_mcio_halfspace_cylinder_from_ocp_face(face: TopoDS_Face, decom_params: DecompositionParameters, surfaces_generated: dict[type[Surface], dict[str, Surface]]={}) -> HalfVolume:
    """Convert a cylindrical face to an mcio cylinder half-space.

    Axis-parallel cylinders become ``CylinderParallelTo{X,Y,Z}Axis``; others
    become a ``GeneralQuadratic``. The sense is set by comparing the oriented
    face normal at the face's mid (u, v) point with the radial direction.

    Parameters
    ----------
    face : TopoDS_Face
        Cylindrical face to convert.
    decom_params : DecompositionParameters
        Decomposition parameters supplying cylinder/quadratic rounding.
    surfaces_generated : dict[type[Surface], dict[str, Surface]], optional
        Cache of already generated mcio surfaces, keyed by surface class and then
        by surface hash string; new surfaces are added in place. Must contain
        an entry for every surface class that may be produced, by default {}

    Returns
    -------
    HalfVolume
        Cylinder half-space on the material side of the face.

    Raises
    ------
    ValueError
        If the surface normal cannot be evaluated at the face mid-point.
    """
    # get surface from face
    surface = _get_ocp_surface_from_ocp_face(face)

    # get u and v parameter range, and calculate the mid points
    u_min, u_max, v_min, v_max = _get_uv_bounds_from_ocp_face(face)
    u_mid = (u_min + u_max) / 2.0
    v_mid = (v_min + v_max) / 2.0

    # constants
    TOL_CYL_AXIS_ON_AXIS: float = 1e-6

    # evalulate the outwards direction of a surface, required for half space direction
    adaptor = BRepAdaptor_Surface(face)
    sl_props = BRepLProp_SLProps(adaptor, u_mid, v_mid, 1, 1e-6)

    # get point and compute the geometric normal
    pnt = adaptor.Value(u_mid, v_mid)

    if sl_props.IsNormalDefined():
        surf_normal = sl_props.Normal()
        # open cascade SLProps returns the raw surface normal. 
        # if the face topology is reversed, then flip the normal vector
        if face.Orientation() == TopAbs_REVERSED:
            surf_normal.Reverse()
    else:
        raise ValueError("Could not evaluate surface normal for face.")

    # geom_adaptor = GeomAdaptor_Surface(surface)
    # gp_cylinder = geom_adaptor.Cylinder()
    # axis = gp_cylinder.Axis()
    # ax_dir = ax_cyl.Direction()
    # ax_loc = ax_cyl.Location()

    # radius = custom_round(gp_cyl.Radius(), decom_params.surf_cylinder_rounding_radius)

    # vx, vy, vz = ax_dir.X(), ax_dir.Y(), ax_dir.Z()
    # px, py, pz = ax_loc.X(), ax_loc.Y(), ax_loc.Z()

    # # Project cylinder axis location onto axis direction
    # dot_prod = px * vx + py * vy + pz * vz
    # proj_x = px - dot_prod * vx
    # proj_y = py - dot_prod * vy
    # proj_z = pz - dot_prod * vz

    # # Round projected coordinates
    # location_x = custom_round(proj_x, decom_params.surf_cylinder_rounding_lateral)
    # location_y = custom_round(proj_y, decom_params.surf_cylinder_rounding_lateral)
    # location_z = custom_round(proj_z, decom_params.surf_cylinder_rounding_lateral)



    # # get axis infomration
    # axis = surface.Axis()
    # location = axis.Location()
    # axis_vec = axis.Direction()
    # radius: float = custom_round(surface.Radius(), decom_params.surf_cylinder_rounding_radius)


    # logger.debug(f'SplitSolid._get_mcio_halfspace_cylinder_from_ocp_face')
    # logger.debug(f' -> cylinder (face) -> mcio half space')
    # logger.debug(f'axis x={axis_vec.X():.2e} y={axis_vec.Y():.2e} z={axis_vec.Z():.2e}')
    # logger.debug(f'loc x={location.X():.2e} y={location.Y():.2e} z={location.Z():.2e}')
    # logger.debug(f'radius = {radius:.2f}')
    # logger.debug(f'orientation = {face.Orientation()}')
    # logger.debug(f'u min = {u_min} u = {u_max}')
    # logger.debug(f'v min = {v_min} v = {v_max}')

    # u_min, u_max, v_min, v_max = _get_uv_bounds_from_ocp_face(face)

    # # get the brep surface adaptor to allow reading of normal
    # adaptor = BRepAdaptor_Surface(face)

    # # Evaluate surface properties at mid-parameters
    # # 0 = no derivatives for point, 1 = first derivatives for tangents/normal
    # sl_props = BRepLProp_SLProps(adaptor, u_mid, v_mid, 1, 1e-6)

    # # directionaliry of half space depends on a normalised normal
    # if sl_props.IsNormalDefined():
    #     if face.Orientation() == TopAbs_REVERSED:
    #         face.Reverse()

    # get location from mid point of face at the centre of the cylinder
    # v_mid = (v_min + v_max) / 2.0
    # location_min = surface.Value(0.0, v_mid)
    # location_max = surface.Value(math.pi, v_mid)

    # location_x = (location_min.X() + location_max.X()) / 2.0
    # location_y = (location_min.Y() + location_max.Y()) / 2.0
    # location_z = (location_min.Z() + location_max.Z()) / 2.0

    # location = gp_Pnt(location_x, location_y, location_z)

    # use geometric surface
    adaptor = GeomAdaptor_Surface(surface)

    gp_cyl = adaptor.Cylinder()
    ax_cyl = gp_cyl.Axis()
    ax_dir = ax_cyl.Direction()
    ax_loc = ax_cyl.Location()
    axis_vec = gp_Vec(ax_dir)

    radius = custom_round(gp_cyl.Radius(), decom_params.surf_cylinder_rounding_radius)

    vx, vy, vz = ax_dir.X(), ax_dir.Y(), ax_dir.Z()
    px, py, pz = ax_loc.X(), ax_loc.Y(), ax_loc.Z()

    dot_prod = px * vx + py * vy + pz * vz
    proj_x = px - dot_prod * vx
    proj_y = py - dot_prod * vy
    proj_z = pz - dot_prod * vz

    # Round projected coordinates
    location_x = custom_round(proj_x, decom_params.surf_cylinder_rounding_lateral)
    location_y = custom_round(proj_y, decom_params.surf_cylinder_rounding_lateral)
    location_z = custom_round(proj_z, decom_params.surf_cylinder_rounding_lateral)

    # check if on axis
    # cyl_axis_on_axis: bool = check_vector_on_axis(axis_vec.X(), axis_vec.Y(), axis_vec.Z(), TOL_CYL_AXIS_ON_AXIS)
    cyl_axis_on_axis: str = check_vector_on_axis(vx, vy, vz, TOL_CYL_AXIS_ON_AXIS)

    # removal of rounding errors
    # location_x = custom_round(location.X(), decom_params.surf_cylinder_rounding_lateral)
    # location_y = custom_round(location.Y(), decom_params.surf_cylinder_rounding_lateral)
    # location_z = custom_round(location.Z(), decom_params.surf_cylinder_rounding_lateral)

    # return planes depending if on/off axis
    if cyl_axis_on_axis == 'on_x_pos' or cyl_axis_on_axis == 'on_x_neg':
        surface_mcio = CylinderParallelToXAxis(y0=location_y, z0=location_z, radius=radius)
        # surfaces_generated[surface] = surface_mcio

    elif cyl_axis_on_axis == 'on_y_pos' or cyl_axis_on_axis == 'on_y_neg':
        surface_mcio = CylinderParallelToYAxis(x0=location_x, z0=location_z, radius=radius)
        # surfaces_generated[surface] = surface_mcio

    elif cyl_axis_on_axis == 'on_z_pos' or cyl_axis_on_axis == 'on_z_neg':
        surface_mcio = CylinderParallelToZAxis(x0=location_x, y0=location_y, radius=radius)
        # surfaces_generated[surface] = surface_mcio

    else:
        # generate the rotation matrix for the off axis cylinder - take the transpose
        r_matrix: np.ndarray = rotation_matrix_from_ocp_vectors(gp_Vec(1, 0, 0), axis_vec)
        r_matrix_transp = r_matrix.T

        # calculate the position using the location of the axis and the rotation matrix
        position = -np.dot(r_matrix_transp, np.array([ax_loc.X(), ax_loc.Y(), ax_loc.Z()]))

        # calculate the mcnp general quadratic parameters
        a = r_matrix_transp[1, 0] ** 2 + r_matrix_transp[2, 0] ** 2
        b = r_matrix_transp[1, 1] ** 2 + r_matrix_transp[2, 1] ** 2
        c = r_matrix_transp[1, 2] ** 2 + r_matrix_transp[2, 2] ** 2

        d = 2.0 * ((r_matrix_transp[1, 0] * r_matrix_transp[1, 1]) + (r_matrix_transp[2, 0] * r_matrix_transp[2, 1]))
        e = 2.0 * ((r_matrix_transp[1, 1] * r_matrix_transp[1, 2]) + (r_matrix_transp[2, 1] * r_matrix_transp[2, 2]))
        f = 2.0 * ((r_matrix_transp[1, 2] * r_matrix_transp[1, 0]) + (r_matrix_transp[2, 2] * r_matrix_transp[2, 0]))

        g = 2.0 * ((position[1] * r_matrix_transp[1, 0]) + (position[2] * r_matrix_transp[2, 0]))
        h = 2.0 * ((position[1] * r_matrix_transp[1, 1]) + (position[2] * r_matrix_transp[2, 1]))
        j = 2.0 * ((position[1] * r_matrix_transp[1, 2]) + (position[2] * r_matrix_transp[2, 2]))

        k = (position[1] ** 2) + (position[2] ** 2) - ((radius) ** 2)

        # round values to remove small rounding errors
        a = custom_round(a, decom_params.surf_general_quadratic_rounding_abc)
        b = custom_round(b, decom_params.surf_general_quadratic_rounding_abc)
        c = custom_round(c, decom_params.surf_general_quadratic_rounding_abc)
        d = custom_round(d, decom_params.surf_general_quadratic_rounding_def)
        e = custom_round(e, decom_params.surf_general_quadratic_rounding_def)
        f = custom_round(f, decom_params.surf_general_quadratic_rounding_def)
        g = custom_round(g, decom_params.surf_general_quadratic_rounding_def)
        h = custom_round(h, decom_params.surf_general_quadratic_rounding_def)
        j = custom_round(j, decom_params.surf_general_quadratic_rounding_def)
        k = custom_round(k, decom_params.surf_general_quadratic_rounding_k)

        surface_mcio = GeneralQuadratic(a=a, b=b, c=c, d=d, e=e, f=f, g=g, h=h, j=j, k=k)
        # surfaces_generated[surface] = surface_mcio
    
    # check if identical surface already defined
    if surface_mcio.__hash_string__() in surfaces_generated[type(surface_mcio)]:
        # if mcio surface already defined, use the previously definde
        surface_mcio = surfaces_generated[type(surface_mcio)][surface_mcio.__hash_string__()]
    else:
        # if mcio surface not already defined add this to dictionary
        surfaces_generated[type(surface_mcio)][surface_mcio.__hash_string__()] = surface_mcio

    v_axis_to_pnt = gp_Vec(ax_loc, pnt)
    axis_dir_vec = gp_Vec(ax_dir)
    
    proj_dist = v_axis_to_pnt.Dot(axis_dir_vec)
    
    closest_axis_pnt = ax_loc.Translated(gp_Vec(ax_dir.X() * proj_dist, ax_dir.Y() * proj_dist, ax_dir.Z() * proj_dist))
    
    radial_vec = gp_Vec(closest_axis_pnt, pnt)

    # FIX: Convert surf_normal (gp_Dir) to a gp_Vec so it can match radial_vec
    normal_vec = gp_Vec(surf_normal.X(), surf_normal.Y(), surf_normal.Z())

    # Check alignment: Outward Normal · Radial Vector
    is_positive_halfspace = normal_vec.Dot(radial_vec) > 0.0

    if is_positive_halfspace:
        return -surface_mcio
    else:
        return +surface_mcio

def _get_mcio_halfspace_sphere_from_ocp_face(face: TopoDS_Face, decom_params: DecompositionParameters, surfaces_generated: dict[type[Surface], dict[str, Surface]]={}) -> HalfVolume:
    """Convert a spherical face to an mcio sphere half-space.

    The most specific sphere class is chosen depending on whether the centre
    lies at the origin, on a Cartesian axis, or elsewhere.

    Parameters
    ----------
    face : TopoDS_Face
        Spherical face to convert.
    decom_params : DecompositionParameters
        Decomposition parameters (currently unused for spheres).
    surfaces_generated : dict[type[Surface], dict[str, Surface]], optional
        Cache of already generated mcio surfaces, keyed by surface class and then
        by surface hash string; new surfaces are added in place. Must contain
        an entry for every surface class that may be produced, by default {}

    Returns
    -------
    HalfVolume
        Inside (-) for forward faces, outside (+) otherwise.
    """
    # get surface from face
    surface = _get_ocp_surface_from_ocp_face(face)

    # orientation of the face relative to the surface normal (u x v). The surface normal points away from the
    # axis (or along the plane axis) for a right handed axis system but the opposite way for a left handed
    # one, e.g. faces of mirrored solids built in OCP; STEP files always store right handed axes.
    orientation = face.Orientation()
    if not surface.Position().Direct():
        orientation = TopAbs_REVERSED if orientation == TopAbs_FORWARD else TopAbs_FORWARD

    # constants
    TOL_ON_AXIS_SPHERE: float = 1e-6

    # get location and radius
    location = surface.Location()
    radius = surface.Radius()

    # check if the sphere is on axis
    is_on_axis = check_point_on_axis(location.X(), location.Y(), location.Z(), TOL_ON_AXIS_SPHERE)

    # check if surface already sonverted to mcio surfaces, else generate mcio surface
    if surface in surfaces_generated:
        surface_mcio = surfaces_generated[surface]

    elif is_on_axis == 'at_origin':
        surface_mcio = SphereAtOrigin(radius=radius)

    elif is_on_axis == 'on_x_axis':
        surface_mcio = SphereOnXAxis(x0=location.X(), radius=radius)

    elif is_on_axis == 'on_y_axis':
        surface_mcio = SphereOnYAxis(y0=location.Y(), radius=radius)

    elif is_on_axis == 'on_z_axis':
        surface_mcio = SphereOnZAxis(z0=location.Z(), radius=radius)

    elif is_on_axis == 'off_axis':
        surface_mcio = Sphere(x0=location.X(), y0=location.Y(), z0=location.Z(), radius=radius)

    # check if identical surface already defined
    if surface_mcio.__hash_string__() in surfaces_generated[type(surface_mcio)]:
        # if mcio surface already defined, use the previously definde
        surface_mcio = surfaces_generated[type(surface_mcio)][surface_mcio.__hash_string__()]
    else:
        # if mcio surface not already defined add this to dictionary
        surfaces_generated[type(surface_mcio)][surface_mcio.__hash_string__()] = surface_mcio

    if orientation == TopAbs_FORWARD:
        return -surface_mcio
    else:
        return +surface_mcio

def _get_mcio_halfspace_cone_from_ocp_face(face: TopoDS_Face, decom_params: DecompositionParameters, surfaces_generated: dict[type[Surface], dict[str, Surface]]={}) -> HalfVolume | Volume:
    """Convert a conical face to an mcio cone half-space or volume.

    The cone sheet used is found from the side of the apex on which the face
    mid-point lies. Axis-parallel cones become ``ConeParallelTo{X,Y,Z}Axis``
    half-spaces; off-axis cones become a ``GeneralQuadratic`` combined with a
    plane through the apex to select the sheet.

    Parameters
    ----------
    face : TopoDS_Face
        Conical face to convert.
    decom_params : DecompositionParameters
        Decomposition parameters supplying cone rounding.
    surfaces_generated : dict[type[Surface], dict[str, Surface]], optional
        Cache of already generated mcio surfaces, keyed by surface class and then
        by surface hash string; new surfaces are added in place. Must contain
        an entry for every surface class that may be produced, by default {}
        Note: not used for cones (no de-duplication is performed).

    Returns
    -------
    HalfVolume | Volume
        Half-space for axis-parallel cones; a Volume (or plane half-space for
        a 90 degree semi-angle) for off-axis cones.

    Raises
    ------
    Exception
        If the off-axis cone's semi-angle and orientation match no case.
    """
    # get surface from face
    surface = _get_ocp_surface_from_ocp_face(face)

    # orientation of the face relative to the surface normal (u x v). The surface normal points away from the
    # axis (or along the plane axis) for a right handed axis system but the opposite way for a left handed
    # one, e.g. faces of mirrored solids built in OCP; STEP files always store right handed axes.
    orientation = face.Orientation()
    if not surface.Position().Direct():
        orientation = TopAbs_REVERSED if orientation == TopAbs_FORWARD else TopAbs_FORWARD

    # constants
    TOL_CONE_AXIS_ON_AXIS_CONE: float = 1e-6
    TOL_CONE_LOC_ON_AXIS: float = 1e-6

    # get the location, axis and semi angle
    axis = surface.Axis()             # axis
    location = surface.Location()     # location of axis origin
    axis_vec = axis.Direction()       # vector of axis
    semi_angle: float = surface.SemiAngle()
    semi_angle_as_t_sq_upper = custom_round(math.tan(semi_angle) ** 2, decom_params.surf_cone_rounding_t_sq)
    semi_angle_as_t_sq_lower = custom_round(math.tan(math.pi - semi_angle) ** 2, decom_params.surf_cone_rounding_t_sq)
    apex = surface.Apex()

    # apex_distance = surface.Radius() / semi_angle
    # centre = (location.X() + direction.X() * apex_distance,
    #           location.Y() + direction.Y() * apex_distance,
    #           location.Z() + direction.Z() * apex_distance)

    # check if on axis
    cone_axis_on_axis = check_vector_on_axis(axis_vec.X(), axis_vec.Y(), axis_vec.Z(), TOL_CONE_AXIS_ON_AXIS_CONE)

    # get the uv params to constraint sheet of cone
    u_min, u_max, v_min, v_max = _get_uv_bounds_from_ocp_face(face)
    v_min_value = surface.Value(0.0, v_min)
    v_max_value = surface.Value(0.0, v_max)


    v_mid = (v_min + v_max) / 2.0
    
    # find the mid point of the face
    p_mid = surface.Value(0.0, v_mid)

    # calculate the distnace from the apex to the points
    # u_min_dist = _distance_of_point_along_vector(apex, gp_Vec(axis_vec), v_min_value)
    # u_max_dist = _distance_of_point_along_vector(apex, gp_Vec(axis_vec), v_max_value)

    


    # # check which side of the cone apex is being used (sheet)
    # TOL = 1e-4
    # if u_min_dist >= (0 - TOL) and u_max_dist > 0:
    #     SHEET_POS = True
    # elif u_min_dist <= (0 + TOL) and u_max_dist < 0:
    #     SHEET_POS = False
    # else:
    #     msg = f'mcio halfspace from ocp cone face has v range which spans apex (surf = {type(surface)}\n' + \
    #           f' v_min = {v_min:.2e} v_max = {v_max:.2e} u_min_dist = {u_min_dist:.2e} v_max_dist = {u_max_dist:.2e}'
        
    #     # extrude face and dump to step file
    #     save_ocp_shapes_to_step([face], 'face_cone_error.stp', decom_params)
    #     print_ocp_face_params(face)
    #     logger.fatal(msg)
    #     raise Exception(msg)
    
    # 1. Get the midpoint of the face in V
    v_mid = (v_min + v_max) / 2.0
    p_mid = surface.Value(0.0, v_mid)

    # 2. Vector from apex to the midpoint
    vec_to_mid = gp_Vec(apex, p_mid)

    # 3. Project that vector onto the cone's axis to see which side it's on
    # This dot product is positive if the face is in the direction of axis_vec
    dist_mid = vec_to_mid.Dot(gp_Vec(axis_vec))

    # 4. Determine sheet: True (1) if in direction of axis_vec, False (-1) otherwise
    SHEET_POS = (dist_mid > 0)

    apex_x = custom_round(apex.X(), decom_params.surf_cone_rounding_lateral)
    apex_y = custom_round(apex.Y(), decom_params.surf_cone_rounding_lateral)
    apex_z = custom_round(apex.Z(), decom_params.surf_cone_rounding_lateral)

    # case for parallel to x axis
    if cone_axis_on_axis == 'on_x_pos':
        if orientation == TopAbs_FORWARD and SHEET_POS:
            return -ConeParallelToXAxis(x0=apex_x, y0=apex_y, z0=apex_z, t_sq=semi_angle_as_t_sq_upper, sheet=1)
        elif orientation == TopAbs_FORWARD and not SHEET_POS:
            return -ConeParallelToXAxis(x0=apex_x, y0=apex_y, z0=apex_z, t_sq=semi_angle_as_t_sq_upper, sheet=-1)
        elif orientation == TopAbs_REVERSED and SHEET_POS:
            return +ConeParallelToXAxis(x0=apex_x, y0=apex_y, z0=apex_z, t_sq=semi_angle_as_t_sq_upper, sheet=1)
        else:
            return +ConeParallelToXAxis(x0=apex_x, y0=apex_y, z0=apex_z, t_sq=semi_angle_as_t_sq_upper, sheet=-1)
        
    elif cone_axis_on_axis == 'on_x_neg':
        if orientation == TopAbs_FORWARD and SHEET_POS:
            return -ConeParallelToXAxis(x0=apex_x, y0=apex_y, z0=apex_z, t_sq=semi_angle_as_t_sq_lower, sheet=-1)
        elif orientation == TopAbs_FORWARD and not SHEET_POS:
            return -ConeParallelToXAxis(x0=apex_x, y0=apex_y, z0=apex_z, t_sq=semi_angle_as_t_sq_lower, sheet=1)
        elif orientation == TopAbs_REVERSED and SHEET_POS:
            return +ConeParallelToXAxis(x0=apex_x, y0=apex_y, z0=apex_z, t_sq=semi_angle_as_t_sq_lower, sheet=-1)
        else:
            return +ConeParallelToXAxis(x0=apex_x, y0=apex_y, z0=apex_z, t_sq=semi_angle_as_t_sq_lower, sheet=1)
        
    elif cone_axis_on_axis == 'on_y_pos':
        if orientation == TopAbs_FORWARD and SHEET_POS:
            return -ConeParallelToYAxis(x0=apex_x, y0=apex_y, z0=apex_z, t_sq=semi_angle_as_t_sq_upper, sheet=1)
        elif orientation == TopAbs_FORWARD and not SHEET_POS:
            return -ConeParallelToYAxis(x0=apex_x, y0=apex_y, z0=apex_z, t_sq=semi_angle_as_t_sq_upper, sheet=-1)
        elif orientation == TopAbs_REVERSED and SHEET_POS:
            return +ConeParallelToYAxis(x0=apex_x, y0=apex_y, z0=apex_z, t_sq=semi_angle_as_t_sq_upper, sheet=1)
        else:
            return +ConeParallelToYAxis(x0=apex_x, y0=apex_y, z0=apex_z, t_sq=semi_angle_as_t_sq_upper, sheet=-1)
        
    elif cone_axis_on_axis == 'on_y_neg':
        if orientation == TopAbs_FORWARD and SHEET_POS:
            return -ConeParallelToYAxis(x0=apex_x, y0=apex_y, z0=apex_z, t_sq=semi_angle_as_t_sq_lower, sheet=-1)
        elif orientation == TopAbs_FORWARD and not SHEET_POS:
            return -ConeParallelToYAxis(x0=apex_x, y0=apex_y, z0=apex_z, t_sq=semi_angle_as_t_sq_lower, sheet=1)
        elif orientation == TopAbs_REVERSED and SHEET_POS:
            return +ConeParallelToYAxis(x0=apex_x, y0=apex_y, z0=apex_z, t_sq=semi_angle_as_t_sq_lower, sheet=-1)
        else:
            return +ConeParallelToYAxis(x0=apex_x, y0=apex_y, z0=apex_z, t_sq=semi_angle_as_t_sq_lower, sheet=1)

    elif cone_axis_on_axis == 'on_z_pos':
        if orientation == TopAbs_FORWARD and SHEET_POS:
            return -ConeParallelToZAxis(x0=apex_x, y0=apex_y, z0=apex_z, t_sq=semi_angle_as_t_sq_upper, sheet=1)
        elif orientation == TopAbs_FORWARD and not SHEET_POS:
            return -ConeParallelToZAxis(x0=apex_x, y0=apex_y, z0=apex_z, t_sq=semi_angle_as_t_sq_upper, sheet=-1)
        if orientation == TopAbs_REVERSED and SHEET_POS:
            return +ConeParallelToZAxis(x0=apex_x, y0=apex_y, z0=apex_z, t_sq=semi_angle_as_t_sq_upper, sheet=1)
        else:
            return +ConeParallelToZAxis(x0=apex_x, y0=apex_y, z0=apex_z, t_sq=semi_angle_as_t_sq_upper, sheet=-1)
        
    elif cone_axis_on_axis == 'on_z_neg':
        if orientation == TopAbs_FORWARD and SHEET_POS:
            return -ConeParallelToZAxis(x0=apex_x, y0=apex_y, z0=apex_z, t_sq=semi_angle_as_t_sq_lower, sheet=-1)
        elif orientation == TopAbs_FORWARD and not SHEET_POS:
            return -ConeParallelToZAxis(x0=apex_x, y0=apex_y, z0=apex_z, t_sq=semi_angle_as_t_sq_lower, sheet=1)
        elif orientation == TopAbs_REVERSED and SHEET_POS:
            return +ConeParallelToZAxis(x0=apex_x, y0=apex_y, z0=apex_z, t_sq=semi_angle_as_t_sq_lower, sheet=-1)
        else:
            return +ConeParallelToZAxis(x0=apex_x, y0=apex_y, z0=apex_z, t_sq=semi_angle_as_t_sq_lower, sheet=1)

    else:
        # generate the rotation matrix for the off axis cylinder - take the transpose
        r_mat: np.ndarray = rotation_matrix_from_ocp_vectors(gp_Vec(1, 0, 0), gp_Vec(axis_vec))
        r_mat_t = r_mat.T

        # calculate the position using the location of the axis and the rotation matrix
        position = -np.dot(r_mat_t, np.array([apex.X(), apex.Y(), apex.Z()]))

        semi_angle_tan = np.tan(semi_angle)

        a = r_mat_t[1,0] ** 2 + r_mat_t[2, 0] ** 2 - (semi_angle_tan * r_mat_t[0, 0]) ** 2
        b = r_mat_t[1,1] ** 2 + r_mat_t[2, 1] ** 2 - (semi_angle_tan * r_mat_t[0, 1]) ** 2
        c = r_mat_t[1,2] ** 2 + r_mat_t[2, 2] ** 2 - (semi_angle_tan * r_mat_t[0, 2]) ** 2

        d = 2.0 * ((r_mat_t[1,0] * r_mat_t[1,1]) + (r_mat_t[2,0] * r_mat_t[2,1]) - (semi_angle_tan ** 2 * r_mat_t[0,0] * r_mat_t[0,1]))
        e = 2.0 * ((r_mat_t[1,1] * r_mat_t[1,2]) + (r_mat_t[2,1] * r_mat_t[2,2]) - (semi_angle_tan ** 2 * r_mat_t[0,1] * r_mat_t[0,2]))
        f = 2.0 * ((r_mat_t[1,2] * r_mat_t[1,0]) + (r_mat_t[2,2] * r_mat_t[2,0]) - (semi_angle_tan ** 2 * r_mat_t[0,2] * r_mat_t[0,0]))

        g = 2.0 * ((position[1] * r_mat_t[1,0]) + (position[2] * r_mat_t[2,0]) - (semi_angle_tan ** 2 * position[0] * r_mat_t[0,0]))
        h = 2.0 * ((position[1] * r_mat_t[1,1]) + (position[2] * r_mat_t[2,1]) - (semi_angle_tan ** 2 * position[0] * r_mat_t[0,1]))
        j = 2.0 * ((position[1] * r_mat_t[1,2]) + (position[2] * r_mat_t[2,2]) - (semi_angle_tan ** 2 * position[0] * r_mat_t[0,2]))

        k = position[1] ** 2 + position[2] ** 2 - (semi_angle_tan * position[0]) ** 2

        plane_a, plane_b, plane_c = axis_vec.X(), axis_vec.Y(), axis_vec.Z()
        plane_d = (plane_a * apex.X() + plane_b * apex.Y() + plane_c * apex.Z())

        plane = Plane(a=plane_a, b=plane_b, c=plane_c, d=plane_d)
        general_quadratic = GeneralQuadratic(a=a, b=b, c=c, d=d, e=e, f=f, g=g, h=h, j=j, k=k)

        # case for semi-angle less than pi and forwards orientation
        if orientation == TopAbs_FORWARD and semi_angle < math.pi/2.0:
            if SHEET_POS:
                return -general_quadratic & +plane
            else:
                return -general_quadratic & -plane
        
        # case for degenerate cone with semi-angle equal to pi        
        elif orientation == TopAbs_FORWARD and semi_angle == math.pi/2.0:
            if SHEET_POS:
                return +plane
            else:
                return -plane
        
        # case for semi-angle greater than pi and forwards orientation
        elif orientation == TopAbs_FORWARD and semi_angle > math.pi/2.0:
            if SHEET_POS:
                return (-general_quadratic & -plane) | +plane
            else:
                return (-general_quadratic & +plane) | -plane
        
        # case for semi-angle less than pi and reveresed orientation
        elif orientation == TopAbs_REVERSED and semi_angle < math.pi/2.0:
            if SHEET_POS:
                return (+general_quadratic & +plane) | -plane
            else:
                return (+general_quadratic & -plane) | +plane
        
        elif orientation == TopAbs_REVERSED and semi_angle == math.pi/2.0:
            if SHEET_POS:
                return -plane
            else:
                return +plane
        
        # case for semi-angle 
        elif orientation == TopAbs_REVERSED and semi_angle > math.pi/2.0:
            if SHEET_POS:
                return (+general_quadratic & -plane) | +plane
            else:
                return (+general_quadratic & +plane) | -plane
        
        else:
            msg = f'[ERROR] cone to general quadratic issues with angle and orientation'
            raise Exception(msg)

def _get_mcio_halfspace_tori_from_ocp_face(face: TopoDS_Face, decom_params: DecompositionParameters, surfaces_generated: dict[type[Surface], dict[str, Surface]]={}) -> HalfVolume:
    """Convert a toroidal face to an mcio torus half-space.

    Only tori with axes parallel to a Cartesian axis are supported.

    Parameters
    ----------
    face : TopoDS_Face
        Toroidal face to convert.
    decom_params : DecompositionParameters
        Decomposition parameters supplying torus rounding.
    surfaces_generated : dict[type[Surface], dict[str, Surface]], optional
        Cache of already generated mcio surfaces, keyed by surface class and then
        by surface hash string; new surfaces are added in place. Must contain
        an entry for every surface class that may be produced, by default {}

    Returns
    -------
    HalfVolume
        Inside (-) for forward faces, outside (+) otherwise.

    Raises
    ------
    Exception
        If the torus axis is not parallel to a Cartesian axis.
    """
    # get surface from face
    surface: Geom_Surface = _get_ocp_surface_from_ocp_face(face)

    # orientation of the face relative to the surface normal (u x v). The surface normal points away from the
    # axis (or along the plane axis) for a right handed axis system but the opposite way for a left handed
    # one, e.g. faces of mirrored solids built in OCP; STEP files always store right handed axes.
    orientation = face.Orientation()
    if not surface.Position().Direct():
        orientation = TopAbs_REVERSED if orientation == TopAbs_FORWARD else TopAbs_FORWARD

    # constants
    TOL_TORI_AXIS_ON_AXIS: float = 1e-5

    # get the location, axis and semi angle
    axis: gp_Ax1 = surface.Axis()
    axis_vec: gp_Dir = axis.Direction()
    location: gp_Pnt = axis.Location()
    
    # get the centre of the tori, applying rounding to reduce rounding errors
    location_x: float = custom_round(location.X(), decom_params.surf_tori_rounding_lateral)
    location_y: float = custom_round(location.Y(), decom_params.surf_tori_rounding_lateral)
    location_z: float = custom_round(location.Z(), decom_params.surf_tori_rounding_lateral)

    # set the major and minor radii, applying rounding to reduce rounding errors
    major_radius: float = custom_round(surface.MajorRadius(), decom_params.surf_tori_rounding_radii)
    minor_radius: float = custom_round(surface.MinorRadius(), decom_params.surf_tori_rounding_radii)

    # check if on axis
    tori_parallel_to_axis: str | bool = check_vector_parallel_to_axis(axis_vec.X(), axis_vec.Y(), axis_vec.Z(), TOL_TORI_AXIS_ON_AXIS)

    # case for parallel to x axis
    if tori_parallel_to_axis == 'parallel_to_x':
        surface_mcio = ToriParralleltoXAxis(
            x0=location_x,
            y0=location_y, 
            z0=location_z,
            a=major_radius,
            b=minor_radius, 
            c=minor_radius)
                
    elif tori_parallel_to_axis == 'parallel_to_y':
        surface_mcio = ToriParralleltoYAxis(
            x0=location_x, 
            y0=location_y, 
            z0=location_z, 
            a=major_radius, 
            b=minor_radius, 
            c=minor_radius)
        
    elif tori_parallel_to_axis == 'parallel_to_z':
        surface_mcio = ToriParralleltoZAxis(
            x0=location_x, 
            y0=location_y, 
            z0=location_z, 
            a=major_radius, 
            b=minor_radius, 
            c=minor_radius)

    else:
        # todo - generate general quadratic term for off axis cylinder
        msg = f'off axis tori can\'t be represented. u0={axis_vec.X():.2e} v0={axis_vec.Y():.2e} w0={axis_vec.Z():.2e}'
        logger.critical(msg)
        raise Exception(msg)

    # check if identical surface already defined
    if surface_mcio.__hash_string__() in surfaces_generated[type(surface_mcio)]:
        # if mcio surface already defined, use the previously definde
        surface_mcio = surfaces_generated[type(surface_mcio)][surface_mcio.__hash_string__()]
    else:
        # if mcio surface not already defined add this to dictionary
        surfaces_generated[type(surface_mcio)][surface_mcio.__hash_string__()] = surface_mcio

    if orientation == TopAbs_FORWARD:
        return -surface_mcio
    else:
        return +surface_mcio    