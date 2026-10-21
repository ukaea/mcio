"""Functions providing geometric properties of OCP (OpenCASCADE) instances.

Includes volume, area and distance calculations for OCP shapes, string
formatting of OCP surface parameters, UV parameter bound extraction for faces,
coincidence checks between OCP planes, cylinders, spheres, cones and tori, and
rotation matrix construction from OCP vectors.
"""

from __future__ import annotations

import math
import numpy as np
import os

# import first to allow logging of optional modules 
from mcio import logger

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from OCP.gp import gp_Torus
    from OCP.TopAbs import TopAbs_ShapeEnum


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
        BRep_Tool
    )
    from OCP.BRepAlgoAPI import (
        BRepAlgoAPI_Section
    )
    from OCP.BRepGProp import (
        BRepGProp
    )
    from OCP.BRepTools import (
        BRepTools
    )
    from OCP.BRepPrimAPI import (
        BRepPrimAPI_MakeSphere
    )
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.Geom import (
        Geom_Surface,
        Geom_Plane, 
        Geom_CylindricalSurface,
        Geom_SphericalSurface,
        Geom_ConicalSurface,
        Geom_ToroidalSurface
    )
    from OCP.gp import (
        gp_Pnt,
        gp_Vec,
        gp_Dir,
        gp_Ax3,
        gp_Pln
    )
    from OCP.GProp import (
        GProp_GProps
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
    from OCP.TopExp import (
        TopExp
    )
    from OCP.TopoDS import (
        TopoDS,
        TopoDS_Face,
        TopoDS_Solid,
        TopoDS_Shape
    )
    from OCP.TopTools import (
        TopTools_IndexedMapOfShape
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

from .geometric import (
    angle_between_vectors,
    values_within_tolerance,
    distance_between_points
)
from .ocpfilereadwrite import (
    extract_ocp_shape_type_from_shape,
    _get_ocp_surface_from_ocp_face
)
from .decomposeparams import (
    DecompositionParameters
)


def _volume_of_ocp_solids(solids: list["TopoDS_Solid"]) -> float:
    """calculate the volume of a list of ocp solids

    Parameters
    ----------
    solids : list[TopoDS_Solid]
        A list of ocp TopoDS_Solids for which to calculate the total volume.

    Returns
    -------
    float
        The total volume of all TopoDS_Solids passed.
    """
    # set up volume to store all solid volumes
    volume: float = 0.0

    # iterate over solids and calculate each solid volume and add to total
    for solid in solids:
        props: GProp_GProps = GProp_GProps()
        BRepGProp.VolumeProperties_s(solid, props)
        volume += props.Mass()

    # return the total volume of all solids
    return volume

def _surface_area_of_ocp_face(face: "TopoDS_Face") -> float:
    """calculate the area of an ocp TopoDS_Face

    Parameters
    ----------
    face : TopoDS_Face
        A TopoDS_Face of which to calculate the area

    Returns
    -------
    float
        The area of the provided TopoDS_Face
    """

    # set up a props
    props: GProp_GProps = GProp_GProps()

    # retrieve the surface properties
    BRepGProp.SurfaceProperties_s(face, props)

    # return the area of the TopoDS_Face
    return props.Mass()


def _surface_area_of_ocp_solids(solids: list["TopoDS_Solid"]) -> float:
    """calculate the total surface area of a list of solids/compsolids

    Parameters
    ----------
    solids : list[TopoDS_Solid | TopoDS_CompSolid]
        Calculates the total surface area of solids/compsolids passed in a list or tuple.

    Returns
    -------
    float
        The total surface area of all solids.
    """
    
    # sum total surface areas of all solids / compsolids
    area: float = 0.0

    # iterate over all solids / compsolids
    for solid in solids:
        # get all faces of solid
        faces: list[TopoDS_Face] = extract_ocp_shape_type_from_shape(solid, TopAbs_FACE)

        # calculate the surface area of each face and add to total sum
        for face in faces:
            props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(face, props)
            area += props.Mass()
    
    # return total surface area
    return area


def _distance_between_shapes(shape_1: "TopoDS_Shape", shape_2: "TopoDS_Shape") -> float:
    """Calculate the minimum distance between two OCP shapes.

    Parameters
    ----------
    shape_1 : TopoDS_Shape
        The first shape.
    shape_2 : TopoDS_Shape
        The second shape.

    Returns
    -------
    float
        The minimum distance between the shapes, from BRepExtrema_DistShapeShape.

    Raises
    ------
    Exception
        If the distance calculation does not complete.
    """
    # calculate the distance
    distance_data: BRepExtrema_DistShapeShape = BRepExtrema_DistShapeShape(shape_1, shape_2)

    # check that a distance was calculated
    if distance_data.IsDone():
        return distance_data.Value()
    else:
        msg = f'_distance_between_shapes() failed to calculate distance'
        raise Exception(msg)

def _distance_between_ocp_points(point_1: "gp_Pnt", point_2: "gp_Pnt") -> float:
    """Calculate the euclidean distance between two gp_Pnt instances.

    Parameters
    ----------
    point_1 : gp_Pnt
        The first gp_Pnt instance.
    point_2 : gp_Pnt
        The second gp_Pnt instance.

    Returns
    -------
    float
        The euclidean distance between the two points.
    """
    dx = point_1.X() - point_2.X()
    dy = point_1.Y() - point_2.Y()
    dz = point_1.Z() - point_2.Z()

    return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


def _midpoint_between_two_points(point_1: "gp_Pnt", point_2: "gp_Pnt") -> "gp_Pnt":
    """Produce a gp_Pnt at the mid point between two gp_Pnt instances.

    Parameters
    ----------
    point_1 : gp_Pnt
        The first gp_Pnt instance.
    point_2 : gp_Pnt
        The second gp_Pnt instance.

    Returns
    -------
    gp_Pnt
        The point midway between ``point_1`` and ``point_2``.
    """
    mid_point = gp_Pnt(
        0.5 * (point_1.X() + point_2.X()),
        0.5 * (point_1.Y() + point_2.Y()),
        0.5 * (point_1.Z() + point_2.Z())
    )

    return mid_point

def _distance_of_point_along_vector(origin: "gp_Pnt", vector: "gp_Vec", point: "gp_Pnt") -> float:
    """Find the projection of a point along a vector.

    Computes ``|vector| * dot(point - origin, vector)``.

    Parameters
    ----------
    origin : gp_Pnt
        The origin of the vector.
    vector : gp_Vec
        The vector along which the point is projected.
    point : gp_Pnt
        The point to project.

    Returns
    -------
    float
        The (signed) length of the point along the vector.
    """

    # create vector from origin to point
    vector_from_origin_to_point: gp_Vec = gp_Vec(origin, point)

    # calculate length along vector whcih point lies orthogonally to
    length: float = vector.Magnitude() * vector_from_origin_to_point.Dot(vector)

    return length


def print_ocp_vec(vec: "gp_Vec") -> None:
    """Print the components of a gp_Vec to stdout.

    Parameters
    ----------
    vec : gp_Vec
        The gp_Vec instance to print.

    Raises
    ------
    TypeError
        If ``vec`` is not a gp_Vec instance.
    """
    if not isinstance(vec, gp_Vec):
        raise TypeError(f"Expected gp_Vec, got {type(vec).__name__}")
    print("gp_Vec:")
    print(f"  X: {vec.X()}")
    print(f"  Y: {vec.Y()}")
    print(f"  Z: {vec.Z()}")
    print(f"  Magnitude: {vec.Magnitude()}")

def print_ocp_dir(dir: "gp_Dir") -> None:
    """Print the components of a gp_Dir to stdout.

    Parameters
    ----------
    dir : gp_Dir
        The gp_Dir instance to print.

    Raises
    ------
    TypeError
        If ``dir`` is not a gp_Dir instance.
    """
    if not isinstance(dir, gp_Dir):
        raise TypeError(f"Expected gp_Dir, got {type(dir).__name__}")
    print("gp_Dir:")
    print(f"  X: {dir.X()}")
    print(f"  Y: {dir.Y()}")
    print(f"  Z: {dir.Z()}")

def print_ocp_pnt(pnt: "gp_Pnt") -> None:
    """Print the components of a gp_Pnt to stdout.

    Parameters
    ----------
    pnt : gp_Pnt
        The gp_Pnt instance to print.

    Raises
    ------
    TypeError
        If ``pnt`` is not a gp_Pnt instance.
    """
    if not isinstance(pnt, gp_Pnt):
        raise TypeError(f"Expected gp_Pnt, got {type(pnt).__name__}")
    print("gp_Pnt:")
    print(f"  X: {pnt.X()}")
    print(f"  Y: {pnt.Y()}")
    print(f"  Z: {pnt.Z()}")

def ocp_surface_params_as_string(ocp_surface: Geom_Surface) -> str:
    """Describe the parameters of an OCP geometric surface as a string.

    Dispatches on the exact type of the surface (plane, cylinder, cone, sphere
    or torus).

    Parameters
    ----------
    ocp_surface : Geom_Surface
        The geometric surface, one of Geom_Plane, Geom_CylindricalSurface,
        Geom_ConicalSurface, Geom_SphericalSurface or Geom_ToroidalSurface.

    Returns
    -------
    str
        A description of the surface parameters.
    """
    # lookup methods for geometry surface types
    string_lookups: dict[type[Geom_Surface], Callable[[Geom_Surface], str]] = {
        Geom_Plane: ocp_surface_params_as_string_plane,
        Geom_CylindricalSurface: ocp_surface_params_as_string_cylinder,
        Geom_ConicalSurface: ocp_surface_params_as_string_cone,
        Geom_SphericalSurface: ocp_surface_params_as_string_sphere,
        Geom_ToroidalSurface: ocp_surface_params_as_string_torus
    }

    as_string: str = string_lookups[type(ocp_surface)](ocp_surface)

    return as_string

def ocp_surface_params_as_string_plane(geom_plane: Geom_Plane) -> str:
    """Describe the parameters of a plane as a string.

    Parameters
    ----------
    geom_plane : Geom_Plane
        The plane geometric surface.

    Returns
    -------
    str
        A description of the plane parameters.
    """
    # get the gp_Pln from geometric surface
    plane: gp_Pln = geom_plane.Pln()

    # get the properties
    position: gp_Ax3 = plane.Position()       # position
    location: gp_Pnt = plane.Location()       # location
    normal: gp_Dir = position.Direction()     # normal

    as_string: str = 'Plane parameters:'
    as_string += f'  Origin: ({location.X():.6e}, {location.Y():.6e}, {location.Z():.6e})'
    as_string += f'  Normal direction: ({normal.X():.6e}, {normal.Y():.6e}, {normal.Z():.6e})'

    return as_string

def ocp_surface_params_as_string_cylinder(geom_cylinder: Geom_CylindricalSurface) -> str:
    """Describe the parameters of a cylinder as a string.

    Parameters
    ----------
    geom_cylinder : Geom_CylindricalSurface
        The cylinder geometric surface.

    Returns
    -------
    str
        A description of the cylinder parameters.
    """
    cyl = geom_cylinder.Cylinder()
    axis = cyl.Axis()
    loc = axis.Location()
    dir = axis.Direction()

    as_string: str = 'Cylinder parameters:'
    as_string += f'  Radius: {cyl.Radius():.6e}'
    as_string += f'  Axis origin: ({loc.X():.6e}, {loc.Y():.6e}, {loc.Z():.6e})'
    as_string += f'  Axis direction: ({dir.X():.6e}, {dir.Y():.6e}, {dir.Z():.6e})'

    return as_string

def ocp_surface_params_as_string_cone(geom_cone: Geom_ConicalSurface) -> str:
    """Describe the parameters of a cone as a string.

    Parameters
    ----------
    geom_cone : Geom_ConicalSurface
        The cone geometric surface.

    Returns
    -------
    str
        A description of the cone parameters.
    """
    apex = geom_cone.Apex()
    axis = geom_cone.Axis()
    loc = axis.Location()
    direction = axis.Direction()
    semi_angle = geom_cone.SemiAngle()
    ref_radius = geom_cone.RefRadius()

    as_string: str = 'Conical surface parameters:'
    as_string += f'  Semi-angle: {semi_angle:.6e} radians ({semi_angle:.6e} degrees)'
    as_string += f'  Reference radius: {ref_radius:.6e}'
    as_string += f'  Apex: ({apex.X():.6e}, {apex.Y():.6e}, {apex.Z():.6e})'
    as_string += f'  Axis origin: ({loc.X():.6e}, {loc.Y():.6e}, {loc.Z():.6e})'
    as_string += f'  Axis direction: ({direction.X():.6e}, {direction.Y():.6e}, {direction.Z():.6e})'

    return as_string

def ocp_surface_params_as_string_sphere(geom_sphere: Geom_SphericalSurface) -> str:
    """Describe the parameters of a sphere as a string.

    Parameters
    ----------
    geom_sphere : Geom_SphericalSurface
        The sphere geometric surface.

    Returns
    -------
    str
        A description of the sphere parameters.
    """
    sph = geom_sphere.Sphere()
    pos = sph.Position()
    axis = pos.Axis()  # Access the axis via Position()
    loc = axis.Location()
    dir = axis.Direction()

    # combine string
    as_string: str = 'Spherical surface parameters:'
    as_string += f'  Radius: {sph.Radius():.6e}'
    as_string += f'  Axis origin: ({loc.X():.6e}, {loc.Y():.6e}, {loc.Z():.6e})'
    as_string += f'  Axis direction: ({dir.X():.6e}, {dir.Y():.6e}, {dir.Z():.6e})'

    return as_string

def ocp_surface_params_as_string_torus(geom_torus: Geom_ToroidalSurface) -> str:
    """Describe the parameters of a torus as a string.

    Parameters
    ----------
    geom_torus : Geom_ToroidalSurface
        The torus geometric surface.

    Returns
    -------
    str
        A description of the torus parameters.
    """
    torus: gp_Torus = geom_torus.Torus()
    axis = torus.Axis()
    loc = axis.Location()
    dir = axis.Direction()

    # combine string
    as_string: str = 'Toroidal surface parameters:'
    as_string += f'  Axis origin: ({loc.X():.6e}, {loc.Y():.6e}, {loc.Z():.6e})'
    as_string += f'  Axis direction: ({dir.X():.6e}, {dir.Y():.6e}, {dir.Z():.6e})'
    as_string += f'  Major radius: ({torus.MajorRadius():.6e})'
    as_string += f'  Minor radius: ({torus.MinorRadius():.6e})'

    return as_string

def ocp_face_params_as_string(face: "TopoDS_Face") -> str:
    """Describe the surface parameters and UV bounds of an OCP face as a string.

    Parameters
    ----------
    face : TopoDS_Face
        The face to describe.

    Returns
    -------
    str
        The underlying surface parameters followed by the u and v bounds and the
        points on the surface at those bounds.
    """
    # get the surface from the face
    surface: Geom_Surface = _get_ocp_surface_from_ocp_face(face)

    # get the uv range from the face
    u_min, u_max, v_min, v_max = _get_uv_bounds_from_ocp_face(face)

    u_min_val = surface.Value(u_min, 0.0)
    u_max_val = surface.Value(u_max, 0.0)
    v_min_val = surface.Value(0.0, v_min)
    v_max_val = surface.Value(0.0, v_max)

    # get the underlying geometric surface properties
    as_string = ocp_surface_params_as_string(surface)

    as_string += f'u_min = {u_min:.6e}'
    as_string += f'u_min_val: x={u_min_val.X():.6e} {u_min_val.Y():.6e} {u_min_val.Z():.6e}'
    as_string += f'u_max = {u_max:.6e}'
    as_string += f'u_max_val: x={u_max_val.X():.6e} {u_max_val.Y():.6e} {u_max_val.Z():.6e}'

    as_string += f'v_min = {v_min:.6e}'
    as_string += f'v_min_val: x={v_min_val.X():.6e} {v_min_val.Y():.6e} {v_min_val.Z():.6e}'
    as_string += f'v_max = {v_max:.6e}'
    as_string += f'v_max_val: x={v_max_val.X():.6e} {v_max_val.Y():.6e} {v_max_val.Z():.6e}'

    return as_string

def _get_uv_bounds_from_ocp_faces(
        faces: list["TopoDS_Face"],
        u_range_max: float = math.inf,
        v_range_max: float = math.inf,
        normalise_periodic_boundaries: bool = True
    ) -> tuple[float, float, float, float]:
    """Get the combined UV parameter bounds across a list of OCP faces.

    Parameters
    ----------
    faces : list[TopoDS_Face]
        The faces from which to collect the UV bounds.
    u_range_max : float, optional
        Maximum u range (currently unused), by default math.inf
    v_range_max : float, optional
        Maximum v range (currently unused), by default math.inf
    normalise_periodic_boundaries : bool, optional
        Normalise periodic parameters of second order surfaces, by default True

    Returns
    -------
    tuple[float, float, float, float]
        The minimum u, maximum u, minimum v and maximum v over all faces.
    """

    TOL = 1e-4

    # set variables to get min and max of all u v parmaeters
    u_min_all = float('inf')
    u_max_all = -float('inf')
    v_min_all = float('inf')
    v_max_all = -float('inf')

    # collect face uv params across all faces
    for i, face in enumerate(faces):
        # get uv params
        u_min, u_max, v_min, v_max = _get_uv_bounds_from_ocp_face(
            face, normalise_periodic_params=normalise_periodic_boundaries
        )

        if u_min < u_min_all:
            u_min_all = u_min

        if u_max > u_max_all:
            u_max_all = u_max

        if v_min < v_min_all:
            v_min_all = v_min

        if v_max > v_max_all:
            v_max_all = v_max

    # return u v min / max values
    return u_min_all, u_max_all, v_min_all, v_max_all

def _get_uv_bounds_from_ocp_face(face: "TopoDS_Face", normalise_periodic_params: bool = True) -> tuple[float, float, float, float]:
    """Get the UV parameter bounds of an OCP face.

    Parameters
    ----------
    face : TopoDS_Face
        The face from which to get the UV bounds.
    normalise_periodic_params : bool, optional
        Normalise the periodic parameters of cylinder, cone, sphere and torus
        surfaces across 2pi, by default True

    Returns
    -------
    tuple[float, float, float, float]
        The minimum u, maximum u, minimum v and maximum v of the face.

    Raises
    ------
    Exception
        If the shape type is not supported.
    """
    
    # extract u v bounds from face
    if isinstance(face.ShapeType(), type(TopAbs_FACE)):
        face = TopoDS.Face_s(face)
        u_min, u_max, v_min, v_max = BRepTools.UVBounds_s(face)
    else:
        msg = f'shape type {type(face)} for uv bounds not yet supported'
        raise Exception(msg)

    surface: Geom_Surface = _get_ocp_surface_from_ocp_face(face)
    
    if normalise_periodic_params:
        if isinstance(surface, Geom_CylindricalSurface):
            u_min, u_max = _normalise_periodic_boundaries_across_2pi(u_min, u_max)
        elif isinstance(surface, Geom_ConicalSurface):
            u_min, u_max = _normalise_periodic_boundaries_across_2pi(u_min, u_max)
        elif isinstance(surface, Geom_SphericalSurface):
            u_min, u_max = _normalise_periodic_boundaries_across_2pi(u_min, u_max)
        elif isinstance(surface, Geom_ToroidalSurface):
            u_min, u_max = _normalise_periodic_boundaries_across_2pi(u_min, u_max)
            v_min, v_max = _normalise_periodic_boundaries_across_2pi(u_min, u_max)

    # check range
    return u_min, u_max, v_min, v_max

def _normalise_periodic_boundaries_across_2pi(p_min: float, p_max: float) -> tuple[float, float]:
    """Normalise a periodic parameter range so that it lies within 2pi.

    Parameters
    ----------
    p_min : float
        The minimum value of the periodic parameter (radians).
    p_max : float
        The maximum value of the periodic parameter (radians).

    Returns
    -------
    tuple[float, float]
        The normalised minimum and maximum parameter values.

    Raises
    ------
    Exception
        If ``p_max`` is less than ``p_min``.
    """
    # tolerance for orunding errors
    TOL = 1e-4

    # 2pi constants
    TWO_PI = 2 * math.pi

    # calculate the range
    r_range = p_max - p_min

    # set up the return variables
    r_min, r_max = p_min, p_max

    # check that the p_range is positive
    if r_range < 0.0:
        msg = f'periodic range across 2pi, max greater than min ()'
        raise Exception(msg)

    # ensure that p_range is within 2pi
    if r_range > TWO_PI + TOL:
        remaineder = r_range % TWO_PI
        if remaineder < TOL:
            r_max = r_min + TWO_PI + remaineder
        else:
            r_max = r_min + remaineder
        # re-calculate p_range
        r_range = r_max - r_min

    # check that param range spans -2pi -> 2pi only
    if r_max > (TWO_PI + TOL):
        down_shift = r_min - (r_min % TWO_PI)
        r_min -= down_shift
        r_max -= down_shift

    if r_min < (TWO_PI - TOL):
        up_shift = r_min - (r_min % -TWO_PI)
        r_min -= up_shift
        r_max -= up_shift

    # return min and max
    return r_min, r_max

def _normalise_periodic_boundaries_across_pi(u_min: float, u_max: float) -> None:
    """Normalise a periodic parameter range across pi (not yet implemented).

    Parameters
    ----------
    u_min : float
        The minimum value of the periodic parameter (radians).
    u_max : float
        The maximum value of the periodic parameter (radians).
    """
    # tolerance for rounding errors
    TOL = 1e-4

def _sample_uv_bounds_from_ocp_shape(shape: "TopoDS_Shape", samples: tuple[int, int]=(5, 5), sample_sphere_size: float=1.0) -> list["TopoDS_Solid"]:
    """Sample a face over its UV bounds, placing a sphere at each sample point.

    Intended as a visual debugging aid.

    Parameters
    ----------
    shape : TopoDS_Shape
        The shape to sample, which must be a face.
    samples : tuple[int, int], optional
        Number of samples in u and v, by default (5, 5)
    sample_sphere_size : float, optional
        Radius of the sphere placed at each sample point, by default 1.0

    Returns
    -------
    list[TopoDS_Solid]
        The spheres placed at each sample point.

    Raises
    ------
    Exception
        If the shape is not a face.
    """
    spheres: list[TopoDS_Solid] = []
    if shape.ShapeType() == TopAbs_FACE:
        face = TopoDS.Face_s(shape)
        u_min, u_max, v_min, v_max = _get_uv_bounds_from_ocp_face(shape)
        surface = BRep_Tool.Surface_s(TopoDS.Face_s(face))
        for u in np.linspace(u_min, u_max, samples[0]):
            for v in np.linspace(v_min, v_max, samples[1]):
                point = surface.Value(u, v)
                sphere = BRepPrimAPI_MakeSphere(point, sample_sphere_size).Shape()
                spheres.append(sphere)
    else:
        msg = f'shape type {shape.ShapeType()} for uv bounds not yet supported'
        logger.fatal(msg)
        raise Exception(msg)

    return spheres

def _sample_uv_bounds_for_faces_in_solid(solid: "TopoDS_Solid", samples: tuple[int, int]=(5, 5), sample_sphere_size: float=1.0) -> dict["TopoDS_Face", list["TopoDS_Solid"]]:
    """Sample every face of a solid over its UV bounds with spheres.

    Parameters
    ----------
    solid : TopoDS_Solid
        The solid whose faces are sampled.
    samples : tuple[int, int], optional
        Number of samples in u and v, by default (5, 5)
    sample_sphere_size : float, optional
        Radius of the sphere placed at each sample point, by default 1.0

    Returns
    -------
    dict[TopoDS_Face, list[TopoDS_Solid]]
        The sample spheres for each face of the solid.
    """
    # get all faces in solid
    faces: list[TopoDS_Face] = extract_ocp_shape_type_from_shape(solid, TopAbs_FACE)

    # set up store of all sphere sets
    sphere_sets: dict[TopoDS_Face, list[TopoDS_Solid]] = {}

    # iterate over faces in solid
    for face in faces:
        sphere_sets[face] = _sample_uv_bounds_from_ocp_shape(face, samples=samples, sample_sphere_size=sample_sphere_size)
    
    # return sphere sets
    return sphere_sets

def check_ocp_planes_coincident(surface_plane_1: "Geom_Plane", surface_plane_2: "Geom_Plane", decom_params: DecompositionParameters) -> bool:
    """Check whether two OCP planes are coincident within tolerance.

    The planes are coincident if the angle between their normals is within
    tolerance, they are not perpendicular, and the distance between their
    locations along the first normal is within tolerance.

    Parameters
    ----------
    surface_plane_1 : Geom_Plane
        The first plane.
    surface_plane_2 : Geom_Plane
        The second plane.
    decom_params : DecompositionParameters
        Decomposition parameters providing the tolerances.

    Returns
    -------
    bool
        True if the planes are coincident within tolerance.
    """
    def normal_angles_within_tol() -> bool:
        """Check the angle between the plane normals is within tolerance."""
        # get the normals 
        norm_1 = surface_plane_1.Axis().Direction()
        norm_2 = surface_plane_2.Axis().Direction()

        # normals as vectors
        vec_1 = [norm_1.X(), norm_1.Y(), norm_1.Z()]
        vec_2 = [norm_2.X(), norm_2.Y(), norm_2.Z()]

        # pass
        angle = angle_between_vectors(vec_1, vec_2)

        return angle < decom_params.surf_plane_tol_normal_angle
        
    def check_planes_not_perpendicular() -> bool:
        """Check the absolute dot product of the normals exceeds the tolerance."""
        norm_1 = surface_plane_1.Axis().Direction()
        norm_2 = surface_plane_2.Axis().Direction()

        dot = abs(norm_1.XYZ().Dot(norm_2.XYZ()))
        
        # If dot ≈ 0 → perpendicular
        return dot > decom_params.surf_plane_tol_normal_angle

    def check_distance_along_normal() -> bool:
        """Check the plane separation along the first normal is within tolerance."""
        norm_1 = surface_plane_1.Axis().Direction()
        point_1 = surface_plane_1.Location()
        point_2 = surface_plane_2.Location()

        # Convert gp_Dir → gp_Vec
        normal_vec = gp_Vec(norm_1.X(), norm_1.Y(), norm_1.Z())

        # Vector between plane origins
        vec = gp_Vec(point_1, point_2)

        # Signed distance along normal
        dist = abs(vec.Dot(normal_vec))

        return dist < decom_params.surf_plane_tol_distance

    # def check_parameters_within_tol():
    #     # get the normals 
    #     norm_1 = surface_plane_1.Axis().Direction()
    #     norm_2 = surface_plane_2.Axis().Direction()

    #     origin_1 = surface_plane_1.Location()
    #     origin_2 = surface_plane_2.Location()

    #     vec_beteween_points = gp_Pnt(origin_1.XYZ() - origin_2.XYZ())
    #     dot_product = norm_1.XYZ().Dot(vec_beteween_points.XYZ())

    #     if abs(dot_product) > 1e-6:
    #         return False
    #     else:
    #         return True

    return normal_angles_within_tol() and check_planes_not_perpendicular() and check_distance_along_normal()

def ocp_cylinders_coincident(surface_cylinder_1: "Geom_CylindricalSurface", surface_cylinder_2: "Geom_CylindricalSurface", decom_params: DecompositionParameters) -> bool:
    """Check whether two OCP cylindrical surfaces are coincident within tolerance.

    Compares radii, the distance between axis locations and the angle between
    axis directions.

    Parameters
    ----------
    surface_cylinder_1 : Geom_CylindricalSurface
        The first cylinder.
    surface_cylinder_2 : Geom_CylindricalSurface
        The second cylinder.
    decom_params : DecompositionParameters
        Decomposition parameters providing the tolerances.

    Returns
    -------
    bool
        True if the cylinders are coincident within tolerance.
    """
    def radii_within_tol() -> bool:
        """Check the cylinder radii are within tolerance."""
        rad_1 = surface_cylinder_1.Radius()
        rad_2 = surface_cylinder_2.Radius()
        return values_within_tolerance(rad_1, rad_2, decom_params.surf_cylinder_tol_radius)
    
    def axis_angles_within_tol() -> bool:
        """Check the angle between the cylinder axes is within tolerance."""
        # directions
        axis_dir_1 = surface_cylinder_1.Axis().Direction()
        axis_dir_2 = surface_cylinder_2.Axis().Direction()
        
        # get vectors 
        vec_1 = [axis_dir_1.X(), axis_dir_1.Y(), axis_dir_1.Z()]
        vec_2 = [axis_dir_2.X(), axis_dir_2.Y(), axis_dir_2.Z()]

        return angle_between_vectors(vec_1, vec_2) < decom_params.surf_cylinder_tol_axis_angle

    def axis_distance_within_tol() -> bool:
        """Check the distance between the cylinder axis locations is within tol."""
        # locations from surfaces
        loc_1 = surface_cylinder_1.Position().Location()
        loc_2 = surface_cylinder_2.Position().Location()
        
        # calculate distance
        distance = loc_1.Distance(loc_2)

        return distance < decom_params.surf_cylinder_tol_axis_distance
    
    # check if cylinders coincident
    return radii_within_tol() and axis_distance_within_tol() and axis_angles_within_tol()

def ocp_sphere_coincident(surface_sphere_1: "Geom_SphericalSurface", surface_sphere_2: "Geom_SphericalSurface", decom_params: DecompositionParameters) -> bool:
    """Check whether two OCP spherical surfaces are coincident within tolerance.

    Compares radii and the distance between centres.

    Parameters
    ----------
    surface_sphere_1 : Geom_SphericalSurface
        The first sphere.
    surface_sphere_2 : Geom_SphericalSurface
        The second sphere.
    decom_params : DecompositionParameters
        Decomposition parameters providing the tolerances.

    Returns
    -------
    bool
        True if the spheres are coincident within tolerance.
    """
    # check radii are within tolerance
    def radii_within_tol() -> bool:
        """Check the sphere radii are within tolerance."""
        rad_1 = surface_sphere_1.Radius()
        rad_2 = surface_sphere_2.Radius()
        return values_within_tolerance(rad_1, rad_2, decom_params.surf_sph_radius_tol)

    # check centres are within tolerance
    def centres_within_tol() -> bool:
        """Check the distance between the sphere centres is within tolerance."""
        # get locations and centres of spheres
        location_1 = surface_sphere_1.Location()
        location_2 = surface_sphere_2.Location()
        centre_1 = [location_1.X(), location_1.Y(), location_1.Z()]
        centre_2 = [location_2.X(), location_2.Y(), location_2.Z()]

        return distance_between_points(*centre_1, *centre_2) < decom_params.surf_sph_position_tol

    # carry out checks for spheres being coincident, and return boolean value
    return radii_within_tol() and centres_within_tol()

def ocp_cone_coincident(surface_cone_1: "Geom_ConicalSurface", surface_cone_2: "Geom_ConicalSurface", decom_params: DecompositionParameters) -> bool:
    """Check whether two OCP conical surfaces are coincident within tolerance.

    Compares semi-angles, the angle between axis directions and the distance
    between axis locations.

    Parameters
    ----------
    surface_cone_1 : Geom_ConicalSurface
        The first cone.
    surface_cone_2 : Geom_ConicalSurface
        The second cone.
    decom_params : DecompositionParameters
        Decomposition parameters providing the tolerances.

    Returns
    -------
    bool
        True if the cones are coincident within tolerance.
    """
    def semi_angle_within_tol() -> bool:
        """Check the cone semi-angles are within tolerance."""
        semi_angle_1 = surface_cone_1.SemiAngle()
        semi_angle_2 = surface_cone_2.SemiAngle()
        return values_within_tolerance(semi_angle_1, semi_angle_2, decom_params.surf_cone_semi_angle_tol)

    def axis_angles_within_tol() -> bool:
        """Check the angle between the cone axes is within tolerance."""
        axis_dir_1 = surface_cone_1.Position().Direction()
        axis_dir_2 = surface_cone_2.Position().Direction()

         # get vectors 
        vec_1 = [axis_dir_1.X(), axis_dir_1.Y(), axis_dir_1.Z()]
        vec_2 = [axis_dir_2.X(), axis_dir_2.Y(), axis_dir_2.Z()]

        return angle_between_vectors(vec_1, vec_2) < decom_params.surf_cone_axis_angle_tol

    def axis_distance_within_tol() -> bool:
        """Check the distance between the cone axis locations is within tolerance."""
        # locations from surfaces
        loc_1 = surface_cone_1.Position().Location()
        loc_2 = surface_cone_2.Position().Location()
        
        # calculate distance
        distance = loc_1.Distance(loc_2)

        return distance < decom_params.surf_cone_axis_distance_tol

    return semi_angle_within_tol() and axis_angles_within_tol() and axis_distance_within_tol()

def ocp_tori_coincident(surface_tori_1: "Geom_ToroidalSurface", surface_tori_2: "Geom_ToroidalSurface", decom_params: DecompositionParameters) -> bool:
    """Check whether two OCP toroidal surfaces are coincident within tolerance.

    Compares minor radii, major radii, the angle between axis directions and
    the distance between centres.

    Parameters
    ----------
    surface_tori_1 : Geom_ToroidalSurface
        The first torus.
    surface_tori_2 : Geom_ToroidalSurface
        The second torus.
    decom_params : DecompositionParameters
        Decomposition parameters providing the tolerances.

    Returns
    -------
    bool
        True if the tori are coincident within tolerance.
    """
    def minor_radii_within_tol() -> bool:
        """Check the tori minor radii are within tolerance."""
        minor_radius_1 = surface_tori_1.MinorRadius()
        minor_radius_2 = surface_tori_2.MinorRadius()
        return values_within_tolerance(minor_radius_1, minor_radius_2, decom_params.surf_tol_tori_minor_radius)

    def major_radii_within_tol() -> bool:
        """Check the tori major radii are within tolerance."""
        major_radius_1 = surface_tori_1.MajorRadius()
        major_radius_2 = surface_tori_2.MajorRadius()
        return values_within_tolerance(major_radius_1, major_radius_2, decom_params.surf_tol_tori_major_radius)

    def axis_angles_within_tol() -> bool:
        """Check the angle between the tori axes is within tolerance."""
        axis_dir_1 = surface_tori_1.Position().Direction()
        axis_dir_2 = surface_tori_2.Position().Direction()

         # get vectors 
        vec_1 = [axis_dir_1.X(), axis_dir_1.Y(), axis_dir_1.Z()]
        vec_2 = [axis_dir_2.X(), axis_dir_2.Y(), axis_dir_2.Z()]

        return angle_between_vectors(vec_1, vec_2) < decom_params.surf_tol_tori_axis_angle

    def centres_within_tol() -> bool:
        """Check the distance between the tori centres is within tolerance."""
        centre_1 = surface_tori_1.Position().Location()
        centre_2 = surface_tori_2.Position().Location()

        distance = centre_1.Distance(centre_2)
        
        return distance < decom_params.surf_tol_tori_centre

    return minor_radii_within_tol() and major_radii_within_tol() and axis_angles_within_tol() and centres_within_tol()

def check_solids_intersect_by_faces(inner_solid: "TopoDS_Shape", outer_solid: "TopoDS_Shape", tol: float=1e-6) -> bool:
    """Check that a section can be computed between every pair of faces of two solids.

    For every face of ``inner_solid`` and every face of ``outer_solid`` a
    BRepAlgoAPI_Section is built; the function returns False as soon as one
    section fails to complete.

    Parameters
    ----------
    inner_solid : TopoDS_Shape
        The first solid.
    outer_solid : TopoDS_Shape
        The second solid.
    tol : float, optional
        Tolerance (currently unused; a fixed fuzzy value of 1e-6 is applied),
        by default 1e-6

    Returns
    -------
    bool
        True if all face-face section operations completed, otherwise False.
    """
    # classifier = BRepClass3d_SolidClassifier()

    faces_inner: list[TopoDS_Face] = extract_ocp_shape_type_from_shape(inner_solid, TopAbs_FACE, cast_from_shape=True)
    faces_outer: list[TopoDS_Face] = extract_ocp_shape_type_from_shape(outer_solid, TopAbs_FACE, cast_from_shape=True)

    for face_inner in faces_inner:
        for face_outer in faces_outer:
            section = BRepAlgoAPI_Section(face_inner, face_outer)
            section.Approximation(True)
            section.SetFuzzyValue(1e-6)
            section.Build()

            if not section.IsDone():
                return False
            
    return True

def check_ocp_faces_connected(face_1: "TopoDS_Face", face_2: "TopoDS_Face") -> bool:
    """Check whether two OCP faces share at least one edge.

    Parameters
    ----------
    face_1 : TopoDS_Face
        The first face.
    face_2 : TopoDS_Face
        The second face.

    Returns
    -------
    bool
        True if any edge of ``face_1`` is also an edge of ``face_2``.
    """
    # set up explorer 
    exp = TopExp()

    # map edges in first face
    map_1 = TopTools_IndexedMapOfShape()
    exp.MapShapes_s(face_1, TopAbs_EDGE, map_1)

    # map edges in second face
    map_2 = TopTools_IndexedMapOfShape()
    exp.MapShapes_s(face_2, TopAbs_EDGE, map_2)

    # compare by TShape identity
    for i in range(1, map_1.Extent() + 1):
        edge_1 = map_1.FindKey(i)
        if map_2.Contains(edge_1):
            return True
        
    return False

def rotation_matrix_from_ocp_vectors(vec_1: gp_Vec, vec_2: gp_Vec) -> np.ndarray:
    """Define the rotation matrix which rotates one OCP vector onto another.

    Uses the axis-angle (Rodrigues) formulation, with the rotation axis being
    the normalised cross product of the two vectors. Note that both input
    vectors are normalised in place.

    Parameters
    ----------
    vec_1 : gp_Vec
        The vector to rotate from.
    vec_2 : gp_Vec
        The vector to rotate to.

    Returns
    -------
    np.ndarray
        The 3x3 rotation matrix.
    """
    
    # the axis of rotation
    axis = vec_1.Crossed(vec_2) 
    axis.Normalize()

    vec_1.Normalize()
    vec_2.Normalize()

    vec_1 = np.array([vec_1.X(), vec_1.Y(), vec_1.Z()])
    vec_2 = np.array([vec_2.X(), vec_2.Y(), vec_2.Z()])

    cos_theta = np.dot(vec_1, vec_2)                  
    sin_theta = np.linalg.norm(np.cross(vec_1, vec_2))
    
    # define repeated mathematical operations 
    one_less_cos_theta = 1.0 - cos_theta

    r_11 = ((axis.X()**2) * one_less_cos_theta) + cos_theta
    r_12 = (axis.X() * axis.Y() * one_less_cos_theta) - (axis.Z() * sin_theta)
    r_13 = (axis.X() * axis.Z() * one_less_cos_theta) + (axis.Y() * sin_theta)

    r_21 = (axis.X() * axis.Y() * one_less_cos_theta) + (axis.Z() * sin_theta)
    r_22 = ((axis.Y() ** 2) * one_less_cos_theta) + cos_theta
    r_23 = (axis.Y() * axis.Z() * one_less_cos_theta) - (axis.X() * sin_theta)
    
    r_31 = (axis.X() * axis.Z() * one_less_cos_theta) - (axis.Y() * sin_theta)
    r_32 = (axis.Y() * axis.Z() * one_less_cos_theta) + (axis.X() * sin_theta)
    r_33 = ((axis.Z() ** 2) * one_less_cos_theta) + cos_theta
    
    matrix_rotation = np.array([[r_11, r_12, r_13], 
                                [r_21, r_22, r_23], 
                                [r_31, r_32, r_33]])
    
    # logger.debug('rotation matrix')
    # logger.debug(f'u = {u}')
    # logger.debug(f'v = {v}')
    # logger.debug(f'axis = {axis.X()} {axis.Y()} {axis.Z()}')
    # logger.debug(f'cos_theta = {cos_theta}')
    # logger.debug(f'sin_theta = {sin_theta}')
    # logger.debug(f'r[0,0] = {matrix_rotation[0, 0]}')
    # logger.debug(f'r[0,1] = {matrix_rotation[0, 1]}')
    # logger.debug(f'r[0,2] = {matrix_rotation[0, 2]}')
    # logger.debug(f'r[0,0] = {matrix_rotation[1, 0]}')
    # logger.debug(f'r[0,1] = {matrix_rotation[1, 1]}')
    # logger.debug(f'r[0,2] = {matrix_rotation[1, 2]}')
    # logger.debug(f'r[0,0] = {matrix_rotation[2, 0]}')
    # logger.debug(f'r[0,1] = {matrix_rotation[2, 1]}')
    # logger.debug(f'r[0,2] = {matrix_rotation[2, 2]}')
    
    return matrix_rotation