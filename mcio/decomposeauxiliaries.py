"""Generation of auxiliary surfaces for concave faces of decomposition solids.

Concave faces of second order surfaces (cylinders, spheres, cones and tori) cannot
be represented directly as a single half-space in CSG. This module provides methods
which create auxiliary surfaces (planes, cylinders and cones) that bound such faces,
and which split solids by these auxiliary surfaces where required.
"""

from __future__ import annotations

import math
from typing import Any

try:
    import casadi
    import OCP
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.Geom import (
        Geom_Plane,
        Geom_CylindricalSurface,
        Geom_SphericalSurface,
        Geom_ConicalSurface,
        Geom_ToroidalSurface,
    )
    from OCP.GeomLProp import GeomLProp_SLProps
    from OCP.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Ax3, gp_Pln
    from OCP.TopAbs import (
        TopAbs_FORWARD,
        TopAbs_REVERSED,
        TopAbs_FACE,
        TopAbs_EDGE,
        TopAbs_WIRE,
    )
    from OCP.TopoDS import TopoDS_Solid, TopoDS_Face
except Exception as e:
    from mcio import logger

    logger.warning('failed to load OCP')
    print(e)
    logger.warning(f"{e}")


from mcio import logger
from .decomposeparams import DecompositionParameters
from .ocp import (
    _get_ocp_surface_from_ocp_face,
    extract_ocp_shape_type_from_shape,
    _get_uv_bounds_from_ocp_faces,
    _get_normal_at_uv_param_on_ocp_surface,
    _get_normal_at_uv_param_on_ocp_face,
    point_at_intersection_of_two_lines,
    distance_of_point_from_axis,
)
from .ocpbooleans import (
    _filter_degenerate_thin_solids_2,
    _sort_ocp_faces_by_connected,
    _common_ocp_solid,
    _cut_ocp_solid,
    generate_ocp_half_space_from_face,
)
from .ocpfilereadwrite import STEPFileWriter
from .ocpproperties import (
    check_ocp_planes_coincident,
    _volume_of_ocp_solids,
    _distance_of_point_along_vector,
)


def generate_auxiliary_surfaces_from_solids(
    solids: list[TopoDS_Solid], decom_params: DecompositionParameters
) -> dict[TopoDS_Solid, dict[TopoDS_Face, list[list[Any]]]]:
    """Generate auxiliary surfaces for all concave second order faces of solids.

    Each solid is checked, and split by any auxiliary surfaces where required.
    Any solids created by the splitting (off cuts) are appended to the list of
    solids to check, so that they too are processed.

    Parameters
    ----------
    solids : list[TopoDS_Solid]
        A list of solids for which auxiliary surface requirements checks and
        generation are to be applied.
    decom_params : DecompositionParameters
        Decompositon parameters passed as user defined parameters which control
        the auxiliary surface generations behaviour.

    Returns
    -------
    dict[TopoDS_Solid, dict[TopoDS_Face, list[list[Any]]]]
        A dictionary keyed by each (possibly split) solid, with values mapping
        each face of the solid to its auxiliary surfaces, each given as a
        ``[Geom_Surface, TopAbs_Orientation]`` pair.
    """

    # set up dictionary to store solids checked for auxiliary surfaces
    solids_and_auxiliary_surfaces: dict[
        TopoDS_Solid, dict[TopoDS_Face, list[list[Any]]]
    ] = {}

    # create a fresh list from the keys of solids, so that any clipped
    # and created solids can be appended
    solids_to_check: list[TopoDS_Solid] = list(solids)

    # assign auxiliary surfaces
    for solid in solids_to_check:
        if False:  # ocp_solid_csg_equivalent_is_finite(solid) == True:
            # case where no auxiliary surfaces are required - coding required.
            pass
        else:
            # produce auxiliary surfaces and return solids which may have been
            # created by aux surface clip
            auxiliary_surfaces, solid_enclosed, solids_created = (
                generate_auxiliary_surfaces_from_solid(solid, decom_params)
            )

            # filter degenerate solids created
            solids_created = _filter_degenerate_thin_solids_2(
                solid, solids_created, decom_params, BRepAlgoAPI_Cut
            )

            # store solid and its auxiliary surfaces
            solids_and_auxiliary_surfaces[solid_enclosed] = auxiliary_surfaces

            # add any solids cretaed due to clipping of auxiliary surfaces
            solids_to_check.extend(solids_created)

    return solids_and_auxiliary_surfaces


def generate_auxiliary_surfaces_from_solid(
    solid: TopoDS_Solid, decom_params: DecompositionParameters, level: int = 0
) -> tuple[dict[TopoDS_Face, list[list[Any]]], TopoDS_Solid, list[TopoDS_Solid]]:
    """Recursively check for and create the auxiliary surfaces of a solid.

    Faces of the solid are grouped by surface type and connectivity, and an
    auxiliary surface generation method is applied to each group. Any auxiliary
    surfaces generated are used to split the solid; if new solids are created
    by the split, the enclosed solid is recursively checked for its own
    auxiliary surfaces, and the off cuts are returned for later processing.

    Parameters
    ----------
    solid : TopoDS_Solid
        A solid for which auxiliary surface requirements checks and generation
        are to be applied.
    decom_params : DecompositionParameters
        Decompositon parameters passed as user defined parameters which control
        the auxiliary surface generations behaviour.
    level : int, optional
        Recursion depth, incremented on each recursive call, by default 0

    Returns
    -------
    faces_with_auxiliaries : dict[TopoDS_Face, list[list[Any]]]
        Faces of the enclosed solid mapped to their auxiliary surfaces, each
        given as a ``[Geom_Surface, TopAbs_Orientation]`` pair.
    solid_enclosed : TopoDS_Solid
        The solid enclosed by the auxiliary surfaces (the input solid if no
        split was required).
    solids_created : list[TopoDS_Solid]
        Off cut solids created by splitting on auxiliary surfaces, which require
        their own auxiliary surface checks.

    Raises
    ------
    Exception
        If the volumes of the common and cut solids do not sum to the volume of
        the solid being split (debug STEP files are written before raising).
    """

    # create variable to store solids created from suxiliary surface splitting
    auxiliary_surfaces: list[list[Any]] = []
    faces_with_auxiliaries: dict[TopoDS_Face, list[list[Any]]] = {}
    solid_enclosed = solid
    solids_created: list[TopoDS_Solid] = []

    # extract all faces from solid and sort by type
    faces = extract_ocp_shape_type_from_shape(solid_enclosed, TopAbs_FACE)
    faces_grouped_by_connected = _sort_ocp_faces_by_connected(faces, decom_params)

    # dictionary of methods for each surface type requiring an auxiliary surface
    AUXILIARY_SURFACE_METHODS: dict[type, Any] = {
        Geom_Plane: generate_auxiliary_surfaces_plane_from_ocp_face,
        Geom_CylindricalSurface: generate_auxiliary_surfaces_cylinder_from_ocp_face,
        Geom_SphericalSurface: generate_auxiliary_surfaces_spheres_from_ocp_face,
        Geom_ConicalSurface: generate_auxiliary_surfaces_cones_from_ocp_faces,
        Geom_ToroidalSurface: generate_auxiliary_surfaces_tori_from_ocp_faces,
    }

    # skip auxiliary generation based user define parameters
    if not decom_params.aux_generate:
        faces_with_auxiliaries = {
            face: []
            for face in extract_ocp_shape_type_from_shape(
                solid, TopAbs_FACE, cast_from_shape=True
            )
        }

        return faces_with_auxiliaries, solid, []

    # iterate over face types
    for face_type, groups_of_connected_faces in faces_grouped_by_connected.items():
        # iterate over faces of given surface type
        for group_of_connected_faces in groups_of_connected_faces:

            # generate auxiliary surface for face - and add to collection for whole solid
            auxiliary_surfaces_for_face = AUXILIARY_SURFACE_METHODS[face_type](
                group_of_connected_faces, decom_params
            )

            # produce faces for the auxiliries created, ensure u and v param
            # range produces face for all surface types
            faces_auxiliary = [
                [
                    BRepBuilderAPI_MakeFace(
                        surface, 0.0, 1.0, 0.0, 1.0, decom_params.make_face_tol
                    ).Face(),
                    orientation,
                ]
                for surface, orientation in auxiliary_surfaces_for_face
            ]

            # add all faces to auxiliary surfaces
            for connected_face in group_of_connected_faces:
                faces_with_auxiliaries[connected_face] = []

            # loop through auxiliary faces and try to common these
            for (face_auxiliary, orientation), surface_auxiliary in zip(
                faces_auxiliary, auxiliary_surfaces_for_face
            ):
                # check if face is planar and already contains the plane
                surface_from_face = _get_ocp_surface_from_ocp_face(face_auxiliary)
                faces_in_solid = extract_ocp_shape_type_from_shape(
                    solid_enclosed, TopAbs_FACE, cast_from_shape=True
                )
                planes_in_solid = [
                    _get_ocp_surface_from_ocp_face(face) for face in faces_in_solid
                ]
                coicident_planes = [
                    plane
                    for plane in planes_in_solid
                    if isinstance(plane, Geom_Plane)
                    and check_ocp_planes_coincident(
                        surface_from_face, plane, decom_params
                    )
                ]

                if len(coicident_planes):
                    continue

                # find the common space betwen auxiliary surface and solid
                common_solids_result = _common_ocp_solid(
                    solid_enclosed,
                    face_auxiliary,
                    decom_params,
                    min_solids_required=1,
                    max_solids_required=math.inf,
                    reverse_face_orientation=orientation == TopAbs_REVERSED,
                )
                common_solids = common_solids_result.solids

                # check if the legacy and leanup fall back methods used
                if common_solids_result.legacy_used:
                    logger.debug("BOPAlgo_COMMON - legacy used")
                if common_solids_result.solids_cleaned:
                    logger.debug("BOPAlgo_COMMON - cleanup used")

                # check to see if cut solids are required, if so, apply cut operation
                if (
                    _volume_of_ocp_solids([solid_enclosed]) * (1 - 1e-6)
                    <= _volume_of_ocp_solids(common_solids)
                    <= _volume_of_ocp_solids([solid_enclosed])
                ):
                    # check to see if the common_solids are the same as the
                    # solid the operation was applied to, if so, no cuts needed
                    logger.debug(
                        "BOPAlgo_CUT - passing as common same volume as solid_enclosed"
                    )
                    cut_solids = []
                elif common_solids_result.failed:
                    logger.debug("COPAlgo_CUT - passing as common failed")
                    cut_solids = []
                else:
                    # find any solids which can be cut by aux surface created
                    cut_solids_result = _cut_ocp_solid(
                        solid_enclosed,
                        face_auxiliary,
                        decom_params,
                        reverse_face_orientation=orientation == TopAbs_REVERSED,
                        clean_solids=common_solids_result.solids_cleaned,
                        use_legacy=common_solids_result.legacy_used,
                    )
                    cut_solids = cut_solids_result.solids

                if common_solids_result.legacy_used:
                    logger.debug(
                        'BOPAlgo_CUT - legacy used follwing commin problem' +\
                       f'{len(cut_solids)} solids produced'
                    )

                # check volumes conserved
                vol_solid_enclosed = _volume_of_ocp_solids([solid_enclosed])
                vol_common = _volume_of_ocp_solids(common_solids)
                vol_cut = _volume_of_ocp_solids(cut_solids)
                vol_common_and_cut_solids = vol_common + vol_cut

                if vol_common_and_cut_solids == 1e-20:
                    logger.fatal('vol_common_and_cut_solids vol is zero, divide by zero error encountered')
                    logger.fatal(f'number of common_solids = {len(common_solids)}')
                    logger.fatal(f'number of cut_solids = {len(cut_solids)}')

                TOL = 1e-2
                if abs(1 - (vol_solid_enclosed / vol_common_and_cut_solids)) > TOL:
                    # save operation solids to step file
                    tool = generate_ocp_half_space_from_face(
                        face_auxiliary,
                        solid_enclosed,
                        decom_params,
                        reverse_halfspace=orientation == TopAbs_REVERSED,
                    )

                    # save shapes to step file for debugging
                    step_writer = STEPFileWriter(decom_params)
                    step_writer.add_solids([solid], "solid")
                    step_writer.add_solids([face_auxiliary], "auxiliary_face")
                    step_writer.add_solids([tool], "tool")
                    step_writer.add_solids(common_solids, "common_solids")
                    step_writer.add_solids(cut_solids, "cut_solids")
                    step_writer.add_solids([solid_enclosed], "solids_enclosed")
                    if len(common_solids) > 0:
                        step_writer.add_solids(
                            [common_solids[0]], "solid_enclosed_created"
                        )
                    step_writer.add_solids(
                        [*cut_solids, *common_solids[1:]], "off_cuts"
                    )
                    output_path = step_writer.write(
                        decom_params.debug_volume_conserved_path, enumerate_files=True
                    )

                    # output message to logger and raise exception
                    msg = f"auxiliary halfspace cut and common failed. Vol of solid = {vol_solid_enclosed:.6e} | vol of cut = {vol_common:.6e} | vol of common = {vol_common:.6e} | cut + common = {vol_common_and_cut_solids:.6e} - see {output_path}"
                    logger.fatal(msg)
                    raise Exception(msg)

                elif common_solids_result.legacy_used and cut_solids_result.legacy_used:
                    logger.warning(
                        "common and cut sucessful following fallback to legacy method"
                    )

                # debugging, possibly suspicous solids... (unlikely to be possible)
                if len(common_solids) > 1:

                    step_writer = STEPFileWriter(decom_params)
                    step_writer.add_solids([solid], "solid")
                    step_writer.add_solids(common_solids, "common_solids")
                    step_writer.add_solids(cut_solids, "cut_solids")
                    step_writer.add_solids([solid_enclosed], "solid_enclosed")
                    step_writer.add_solids(
                        [common_solids[0]], "common_solid_enclosed_created"
                    )
                    step_writer.add_solids(
                        [common_solids[0]], "common_solid_excess_added_to_off_cuts"
                    )
                    step_writer.add_solids(
                        [*cut_solids, *common_solids[1:]],
                        "off_cuts_(cuts_and_excess_commons)",
                    )
                    output_path = step_writer.write(decom_params.debug_cut_common_path)

                    msg = (
                        f"auxiliary halfspace cut and common produced"
                        + f"{len(common_solids)} commons not 1, possible"
                        + f"error - see {output_path}"
                    )

                # take the first solid enclosed by the auxiliary surface, and
                # merge remaining with off cuts
                solid_enclosed = common_solids[0]
                solids_off_cuts = [*cut_solids, *common_solids[1:]]

                # if a new cut solid is produces outside of the auxiliary
                # surface add it to solids_created list for return
                if len(solids_off_cuts) > 0:
                    # assumes that the new cut will introduce a new surface in
                    # the enclosed solid to don't need to add aux to list
                    solids_created.extend(solids_off_cuts)

                    # case where auxiliary surfaces need to be recalculated,
                    # call function recursively
                    (
                        auxiliary_surfaces_recursive,
                        solid_enclosed_recursive,
                        solids_off_cuts_recursive,
                    ) = generate_auxiliary_surfaces_from_solid(
                        solid_enclosed, decom_params, level=level + 1
                    )

                    # add any further off cuts to solids created
                    solids_created.extend(solids_off_cuts_recursive)

                    # return the enclosed solid with freshly generated
                    # auxiliary surfaces, and add previously trimmed solids
                    return (
                        auxiliary_surfaces_recursive,
                        solid_enclosed_recursive,
                        solids_created,
                    )
                else:
                    # assumes that no new cut and that the auxiliary surface
                    # will needed to be added later as a virtual surface

                    # old - before grouping by faces
                    auxiliary_surfaces.append(surface_auxiliary)

                    # new - save all faces with their auxilities attached
                    for connected_face in group_of_connected_faces:
                        faces_with_auxiliaries[connected_face].append(surface_auxiliary)

    return faces_with_auxiliaries, solid_enclosed, solids_created


def generate_auxiliary_surfaces_plane_from_ocp_face(
    faces: list[TopoDS_Face], decom_params: DecompositionParameters
) -> list[list[Any]]:
    """Generate auxiliary surfaces for a list of planar faces.

    Planar faces never require auxiliary surfaces, so an empty list is always
    returned. Provided for consistency with the other surface types.

    Parameters
    ----------
    faces : list[TopoDS_Face]
        List of co-planar faces to apply an auxiliary surface
    decom_params : DecompositionParameters
        Decomposition parameters defined by user to control behaviour of
        auxiliary surface generation.

    Returns
    -------
    list[list[Any]]
        A list of auxiliary surfaces generated, each given as a
        ``[Geom_Surface, TopAbs_Orientation]`` pair.
    """
    # planes don't need auxiliary faces, so always return no auxiliary faces
    return []


def generate_auxiliary_surfaces_cylinder_from_ocp_face(
    faces: list[TopoDS_Face], decom_params: DecompositionParameters
) -> list[list[Any]]:
    """Generate auxiliary planes for a list of coincident cylindrical faces.

    Convex (forward oriented) or fully closed faces need no auxiliary surface.
    Otherwise either a single plane (``aux_surface_cylinder_simple``) or one or
    two planes through the cylinder axis at the u parameter limits are created.

    Parameters
    ----------
    faces : list[TopoDS_Face]
        List of conicident cylindrical faces to apply an auxiliary surface
    decom_params : DecompositionParameters
        Decomposition parameters defined by user to control behaviour of
        auxiliary surface generation.

    Returns
    -------
    list[list[Any]]
        A list of auxiliary surfaces generated, each given as a
        ``[Geom_Surface, TopAbs_Orientation]`` pair.
    """

    # face is convex, no auxiliary surface required
    if faces[0].Orientation() == TopAbs_FORWARD:
        return []

    # get the surface of the first face
    surface = _get_ocp_surface_from_ocp_face(faces[0])

    # get the range of u and v across all coincident faces
    u_min, u_max, v_min, v_max = _get_uv_bounds_from_ocp_faces(
        faces, u_range_max=math.tau
    )
    u_mid = (u_min + u_max) / 2.0

    # check if cylinder is closed, i.e. u min and max span 2 pi
    closed_angle = u_max - u_min
    open_angle = (2 * math.pi) - closed_angle

    if open_angle < decom_params.aux_surface_cylinder_angle_tol:
        return []

    # extract the points for the u,v, min and u,v max
    point_u_min_v_min = surface.Value(u_min, v_min)
    point_u_max_v_min = surface.Value(u_max, v_min)

    # get the axis of the cylinder
    axis_dir = surface.Axis().Direction()
    axis_vec = gp_Vec(axis_dir)

    # set up retun list of auxiliary surfaces
    auxiliary_surfaces: list[list[Any]] = []

    # apply either simple single plane as auxiliary or double plane
    if decom_params.aux_surface_cylinder_simple:
        # get normal of face at u_min v_zero
        normal = _get_normal_at_uv_param_on_ocp_face(faces[0], u_mid, 0.0).Reversed()

        # create geometric plane
        point_u_min_v_zero = surface.Value(u_min, 0.0)
        plane_axis = gp_Ax3(point_u_min_v_zero, normal)
        plane_gp = gp_Pln(plane_axis)
        plane_geom = Geom_Plane(plane_gp)

        # add plane to return list of auxiliary surfaces
        auxiliary_surfaces.append([plane_geom, TopAbs_FORWARD])

    else:
        # set up direction at u min v min at normal to cylinder at this point
        normal_dir_at_u_min_v_min = _get_normal_at_uv_param_on_ocp_surface(
            surface, u_min, v_min
        )
        normal_vec_at_u_min_v_min = gp_Vec(normal_dir_at_u_min_v_min)

        # get normal vector for plane at u min
        try:
            normal_u_min_plane = gp_Dir(normal_vec_at_u_min_v_min.Crossed(axis_vec))
        except:
            msg = (
                f"decompose cylinder surface, u_min = {u_min} u_max = "
                + f"{u_max} v_min = {v_min} v_max ={v_max}"
            )
            raise Exception(msg)

        # create geometric plane
        plane_axis_u_min = gp_Ax3(point_u_min_v_min, normal_u_min_plane)
        plane_gp_u_min = gp_Pln(plane_axis_u_min)
        plane_geom_u_min = Geom_Plane(plane_gp_u_min)

        # add plane to auxiliary surface return list
        auxiliary_surfaces.append([plane_geom_u_min, TopAbs_FORWARD])

        # case where second enclosing plane surface is required
        if closed_angle < math.pi - decom_params.aux_surface_cylinder_angle_tol:
            # set up direction at the u_max v_min at normal to the cylinder
            normal_dir_at_u_max_v_min = _get_normal_at_uv_param_on_ocp_surface(
                surface, u_max, v_min
            )
            normal_vec_at_u_max_v_min = gp_Vec(normal_dir_at_u_max_v_min)

            # get normal vector for plane at u min
            try:
                normal_u_max_plane = gp_Dir(axis_vec.Crossed(normal_vec_at_u_max_v_min))
            except:
                msg = f"decompose cylinder surface, u_min = {u_min} u_max = {u_max} v_min = {v_min} v_max ={v_max}"
                raise Exception(msg)

            # create geometric plane
            plane_axis_u_max = gp_Ax3(point_u_max_v_min, normal_u_max_plane)
            plane_gp_u_max = gp_Pln(plane_axis_u_max)
            plane_geom_u_max = Geom_Plane(plane_gp_u_max)

            # add plane to auxiliary surface return list
            auxiliary_surfaces.append([plane_geom_u_max, TopAbs_FORWARD])

    return auxiliary_surfaces


def generate_auxiliary_surfaces_spheres_from_ocp_face(
    faces: list[TopoDS_Face], decom_params: DecompositionParameters
) -> list[list[Any]]:
    """Generate auxiliary planes for a list of coincident spherical faces.

    Convex (forward oriented) faces need no auxiliary surface. For faces open in
    the u parameter with ``aux_surface_sphere_simple`` set, a single plane is
    created normal to the face centre. Other cases are not yet implemented.

    Parameters
    ----------
    faces : list[TopoDS_Face]
        List of conicident spherical faces to apply an auxiliary surface
    decom_params : DecompositionParameters
        Decomposition parameters defined by user to control behaviour of
        auxiliary surface generation.

    Returns
    -------
    list[list[Any]]
        A list of auxiliary surfaces generated, each given as a
        ``[Geom_Surface, TopAbs_Orientation]`` pair.

    Raises
    ------
    NotImplementedError
        If more than one connected coincident spherical face is given.
    Exception
        If the u parameter range of the face does not match a known case.
    """

    # face is convex, no auxiliary surface required
    if faces[0].Orientation() == TopAbs_FORWARD:
        return []

    # get the range of u and v across all coincident faces
    u_min, u_max, v_min, v_max = _get_uv_bounds_from_ocp_faces(
        faces, u_range_max=math.tau, v_range_max=math.pi
    )
    u_mid = (u_min + u_max) / 2.0
    v_mid = (v_min + v_max) / 2.0

    # calculate open and closed angles of u param (equitorial, azimuthal about z axis)
    u_closed_angle = u_max - u_min
    u_is_closed = (
        math.tau - decom_params.aux_surface_sphere_angle_tol
        < u_closed_angle
        < math.tau + decom_params.aux_surface_sphere_angle_tol
    )

    # calculate open and closed angle of v parameter (elevation, polar angle along z axis)
    v_closed_angle = v_max - v_min
    v_is_closed = (
        (math.pi / 2.0) - decom_params.aux_surface_sphere_angle_tol
        < v_closed_angle
        < (math.pi / 2.0) - decom_params.aux_surface_sphere_angle_tol
    )

    # set up list to store auxiliary surfaces
    auxiliary_surfaces: list[list[Any]] = []

    # get the surface of the first face
    surface = _get_ocp_surface_from_ocp_face(faces[0])

    # get the spheres parameters
    centre = surface.Location()
    radius = surface.Radius()

    # extract the points for the u,v, min and u,v max
    point_u_min_v_min = surface.Value(u_min, v_min)
    point_u_min_v_max = surface.Value(u_min, v_max)
    point_u_max_v_min = surface.Value(u_max, v_min)
    point_u_max_v_max = surface.Value(u_max, v_max)
    point_u_min_v_zero = surface.Value(u_min, 0.0)
    point_u_max_v_zero = surface.Value(u_max, 0.0)

    # count the number of edges and wires within the face
    n_wires = len(extract_ocp_shape_type_from_shape(faces[0], TopAbs_WIRE))

    # sphere_centre = BRepPrimAPI_MakeSphere(centre, 0.1).Shape()
    # sphere_u_mid_v_mid = BRepPrimAPI_MakeSphere(point_u_mid_v_mid, 0.1).Shape()
    normal_dir_u_mid_v_mid = _get_normal_at_uv_param_on_ocp_surface(
        surface, u_mid, v_mid
    )

    # check for case where two coincident and connected spheres are found...
    if len(faces) > 1:
        msg = f"case where two or more coincident spheres are connected - not yet coded... sorry :("
        # need to split and isolate each spherical face using a plane, as the intersection of two spheres
        # should form an edge along a plane
        logger.fatal(msg)
        raise NotImplementedError(msg)
    else:
        face = faces[0]

    # case 1 - face is closed about the u parameter range
    if (u_closed_angle >= (math.tau - decom_params.aux_surface_sphere_angle_tol)) and (
        n_wires <= 1
    ):
        # no auxiliary surface required
        pass

    # case 2 - face is spaning 2pi in u range but has more than one edge and may require aux surface
    elif (
        u_closed_angle >= (math.tau - decom_params.aux_surface_sphere_angle_tol)
    ) and (n_wires > 1):
        msg = f"u_range = {u_closed_angle:.2f} n_wires = {n_wires} - case yet to be coded."
        logger.warning(msg)
        # raise Exception(msg)

    # case 3 - face is open in u param range, simple approach to be taken
    elif (
        u_closed_angle < (math.tau - decom_params.aux_surface_sphere_angle_tol)
    ) and decom_params.aux_surface_sphere_simple:
        # simple case of single plane
        # get the normal at the centre of the face via u_mid and v_mid
        normal_dir_u_mid_v_mid = _get_normal_at_uv_param_on_ocp_surface(
            surface, u_mid, v_mid
        )
        normal_dir_u_mid_v_mid_reversed = normal_dir_u_mid_v_mid.Reversed()
        normal_vec_u_mid_v_mid = gp_Vec(normal_dir_u_mid_v_mid)
        normal_vec_u_mid_v_mid.Normalize()
        normal_vec_u_mid_v_mid_reversed = normal_vec_u_mid_v_mid.Reversed()

        # reference point along vector, used to find which point on uv range is closest.
        point_ref = gp_Pnt(
            centre.XYZ() + (radius * 2.0 * normal_vec_u_mid_v_mid_reversed.XYZ())
        )

        # check distance along vector
        distance_to_u_min_v_min = _distance_of_point_along_vector(
            point_ref, normal_vec_u_mid_v_mid, point_u_min_v_min
        )
        distance_to_u_min_v_max = _distance_of_point_along_vector(
            point_ref, normal_vec_u_mid_v_mid, point_u_min_v_max
        )
        distance_to_u_max_v_min = _distance_of_point_along_vector(
            point_ref, normal_vec_u_mid_v_mid, point_u_max_v_min
        )
        distance_to_u_max_v_max = _distance_of_point_along_vector(
            point_ref, normal_vec_u_mid_v_mid, point_u_max_v_max
        )
        distance_to_u_min_v_zero = _distance_of_point_along_vector(
            point_ref, normal_vec_u_mid_v_mid, point_u_min_v_zero
        )
        distance_to_u_max_v_zero = _distance_of_point_along_vector(
            point_ref, normal_vec_u_mid_v_mid, point_u_max_v_zero
        )

        distances = [
            distance_to_u_min_v_min,
            distance_to_u_min_v_max,
            distance_to_u_max_v_min,
            distance_to_u_max_v_max,
            distance_to_u_min_v_zero,
            distance_to_u_max_v_zero,
        ]

        points = [
            point_u_min_v_min,
            point_u_min_v_max,
            point_u_max_v_min,
            point_u_max_v_max,
            point_u_min_v_zero,
            point_u_max_v_zero,
        ]

        # find the point closes to the reference
        point_closest = points[distances.index(min(distances))]

        # create a plane at this point
        plane_axis = gp_Ax3(point_closest, normal_dir_u_mid_v_mid_reversed)
        plane_gp = gp_Pln(plane_axis)
        plane_geom = Geom_Plane(plane_gp)

        auxiliary_surfaces.append([plane_geom, TopAbs_FORWARD])

    elif (
        u_closed_angle < (math.tau - decom_params.aux_surface_sphere_angle_tol)
    ) and not decom_params.aux_surface_sphere_simple:
        # comprehensive case
        msg = 'sperical non-simple case not yet coded'
        # needs to be coded - if required
        pass
    else:
        msg = (
            f"[ERROR] - spherical aux case, u_min = {u_min:.2f} "
            + f"u_max = {u_max:.2f}"
        )
        logger.fatal(msg)
        raise Exception(msg)

    return auxiliary_surfaces


def generate_auxiliary_surfaces_cones_from_ocp_faces(
    faces: list[TopoDS_Face], decom_params: DecompositionParameters
) -> list[list[Any]]:
    """Generate auxiliary planes for a list of coincident conical faces.

    Convex faces, and faces whose u parameter loop is closed, need no auxiliary
    surface. Otherwise either a single plane through the u/v limit points
    (``aux_surface_cone_simple``) or one or two planes through the cone axis at
    the u parameter limits are created.

    Parameters
    ----------
    faces : list[TopoDS_Face]
        List of conicident conical faces to apply an auxiliary surface
    decom_params : DecompositionParameters
        Decomposition parameters defined by user to control behaviour of
        auxiliary surface generation.

    Returns
    -------
    list[list[Any]]
        A list of auxiliary surfaces generated, each given as a
        ``[Geom_Surface, TopAbs_Orientation]`` pair.

    Raises
    ------
    Exception
        If a plane normal cannot be constructed, or the closed angle of the u
        parameter range is out of range.
    """

    # set rounding tolerance on angles
    TOL = 1e-4

    # get the surface of the first face
    surface = _get_ocp_surface_from_ocp_face(faces[0])

    # get the semi angle
    semi_angle = surface.SemiAngle()

    # check if surface is convex or concave
    if faces[0].Orientation() == TopAbs_REVERSED and (semi_angle > (math.pi / 2.0)):
        return []
    elif faces[0].Orientation() == TopAbs_FORWARD and (semi_angle < (math.pi / 2.0)):
        return []

    # get the range of u and v across all coincident faces
    u_min, u_max, v_min, v_max = _get_uv_bounds_from_ocp_faces(
        faces, u_range_max=math.tau
    )
    u_mid = (u_min + u_max) / 2.0
    v_mid = (v_min + v_max) / 2.0

    # check if cylinder is closed, i.e. u min and max span 2 pi
    closed_angle = u_max - u_min
    open_angle = math.tau - closed_angle

    # check if u param loop closed
    if open_angle < decom_params.aux_surface_cylinder_small_angle:
        return []

    # extract the points for the u,v, min and u,v max
    point_u_min_v_min = surface.Value(u_min, v_min)
    point_u_min_v_max = surface.Value(u_min, v_max)
    point_u_max_v_min = surface.Value(u_max, v_min)
    point_u_max_v_max = surface.Value(u_max, v_max)
    point_u_mid_v_mid = surface.Value(u_mid, v_mid)

    # get the axis along teh centre of the cone
    if faces[0].Orientation() == TopAbs_FORWARD and (semi_angle >= (math.pi / 2.0)):
        axis_dir = surface.Axis().Direction()
        axis_vec = -1.0 * gp_Vec(axis_dir)
    elif faces[0].Orientation() == TopAbs_REVERSED and (semi_angle < (math.pi / 2.0)):
        axis_dir = surface.Axis().Direction()
        axis_vec = gp_Vec(axis_dir)

    axis_dir = surface.Axis().Direction()
    axis_vec = gp_Vec(axis_dir)

    # set up list to store auxiliary planes
    auxiliary_surfaces: list[list[Any]] = []

    if decom_params.aux_surface_cone_simple:
        # get the normal vector using the cross product of the uv min/max points
        vec_u_min_v_min_to_u_max_v_min = gp_Vec(point_u_min_v_min, point_u_max_v_min)
        vec_u_min_v_min_to_u_min_v_max = gp_Vec(point_u_min_v_min, point_u_min_v_max)

        normal_dir = gp_Dir(
            vec_u_min_v_min_to_u_max_v_min.Crossed(vec_u_min_v_min_to_u_min_v_max)
        )
        normal_dir = gp_Dir(
            vec_u_min_v_min_to_u_min_v_max.Crossed(vec_u_min_v_min_to_u_max_v_min)
        )

        plane_axis = gp_Ax3(point_u_min_v_min, normal_dir)
        plane_gp = gp_Pln(plane_axis)

        plane_geom = Geom_Plane(plane_gp)

        auxiliary_surfaces.append([plane_geom, TopAbs_FORWARD])

    else:
        normal_dir_at_u_min_v_min = _get_normal_at_uv_param_on_ocp_surface(
            surface, u_min, v_min
        )
        normal_vec_at_u_min_v_min = gp_Vec(normal_dir_at_u_min_v_min)

        # get normal vector for plane at u min
        try:
            normal_u_min_plane = gp_Dir(normal_vec_at_u_min_v_min.Crossed(axis_vec))
        except:
            msg = f"decompose cylinder surface failed, u_min = {u_min} u_max = {u_max} v_min = {v_min} v_max ={v_max}"
            raise Exception(msg)

        # create geometric plane for u_min
        plane_axis_u_min = gp_Ax3(point_u_min_v_min, normal_u_min_plane)
        plane_gp_u_min = gp_Pln(plane_axis_u_min)
        plane_geom_u_min = Geom_Plane(plane_gp_u_min)

        # add auxiliary plane to return list
        auxiliary_surfaces.append([plane_geom_u_min, TopAbs_FORWARD])

        if math.pi + TOL < closed_angle <= math.tau + TOL:
            # set up direction at the u_max v_min at normal to the cylinder at this point
            normal_dir_at_u_max_v_min = _get_normal_at_uv_param_on_ocp_surface(
                surface, u_max, v_min
            )
            normal_vec_at_u_max_v_min = gp_Vec(normal_dir_at_u_max_v_min)

            # get normal vector for plane at u min
            try:
                normal_u_max_plane = gp_Dir(axis_vec.Crossed(normal_vec_at_u_max_v_min))
            except:
                msg = (
                    f"decompose cylinder surface, u_min = {u_min} "
                    + f"u_max = {u_max} v_min = {v_min} v_max ={v_max}"
                )
                raise Exception(msg)

            # create geometric plane
            plane_axis_u_max = gp_Ax3(point_u_max_v_min, normal_u_max_plane)
            plane_gp_u_max = gp_Pln(plane_axis_u_max)
            plane_geom_u_max = Geom_Plane(plane_gp_u_max)

            auxiliary_surfaces.append([plane_geom_u_max, TopAbs_FORWARD])

        elif -TOL <= closed_angle <= math.pi + TOL:
            pass
        else:
            msg = (
                f"[ERROR] - cone auxiliary surface generation, closed "
                + f"angle {closed_angle:.2f} out of range"
            )
            raise Exception(msg)

    return auxiliary_surfaces


def generate_auxiliary_surfaces_tori_from_ocp_faces(
    faces: list[TopoDS_Face], decom_params: DecompositionParameters
) -> list[list[Any]]:
    """Generate auxiliary surfaces for a list of coincident toroidal faces.

    Convex (forward oriented) faces need no auxiliary surface. Open u parameter
    ranges (about the torus axis) are bounded by planes through the axis. Open
    v parameter ranges (about the minor circle) are bounded by a plane,
    cylinder or cone, chosen from the angles between the torus axis and the
    v_min -> v_max chord and the face normal.

    Parameters
    ----------
    faces : list[TopoDS_Face]
        List of conicident toroidal faces to apply an auxiliary surface
    decom_params : DecompositionParameters
        Decomposition parameters defined by user to control behaviour of
        auxiliary surface generation.

    Returns
    -------
    list[list[Any]]
        A list of auxiliary surfaces generated, each given as a
        ``[Geom_Surface, TopAbs_Orientation]`` pair.

    Raises
    ------
    Exception
        If no surface normal is defined at the sample point, or the u parameter
        range does not match a known case.
    """

    # get the first face
    face = faces[0]

    # get surface from first face
    surface = _get_ocp_surface_from_ocp_face(face)

    # check if convex, and not requiring auxiliary surfaces
    if face.Orientation() == TopAbs_FORWARD:
        return []

    # set up list to store auxiliary surface
    auxiliary_surfaces: list[list[Any]] = []

    # get the range of u and v across all coincident faces
    u_min, u_max, v_min, v_max = _get_uv_bounds_from_ocp_faces(
        faces,
        u_range_max=math.tau,
        v_range_max=math.tau,
        normalise_periodic_boundaries=False,
    )
    u_mid = (u_min + u_max) / 2.0
    v_mid = (v_min + v_max) / 2.0

    # check if cylinder is closed, i.e. u min and max span 2 pi
    closed_angle_u = u_max - u_min

    closed_angle_v = v_max - v_min

    auxiliary_surfaces = []

    # collect tori information
    surface = _get_ocp_surface_from_ocp_face(faces[0])
    axis: gp_Ax3 = surface.Position()
    centre: gp_Pnt = axis.Location()
    axis_direction: gp_Dir = axis.Direction()
    axis_vec = gp_Vec(axis_direction)

    # count the number of edges and wires within the face
    n_edges = len(extract_ocp_shape_type_from_shape(faces[0], TopAbs_EDGE))
    n_wires = len(extract_ocp_shape_type_from_shape(faces[0], TopAbs_WIRE))

    # extract the normal at the point
    props = GeomLProp_SLProps(surface, u_min, v_mid, 1, 1e-6)
    if not props.IsNormalDefined():
        # check a normal is defined
        msg = f"no normal defined at this point"
        raise Exception(msg)
    else:
        # get normal and point on the surface at the u v mid point
        normal_dir_at_u_min_v_mid = props.Normal()
        normal_vec_at_u_min_v_mid = -1.0 * gp_Vec(normal_dir_at_u_min_v_mid)

    # ██  ██   █████▄ ▄████▄ █████▄  ▄████▄ ██▄  ▄██ ▄█████
    # ██  ██   ██▄▄█▀ ██▄▄██ ██▄▄██▄ ██▄▄██ ██ ▀▀ ██ ▀▀▀▄▄▄
    # ▀████▀   ██     ██  ██ ██   ██ ██  ██ ██    ██ █████▀

    # check for u parameters
    TOL_ROUNDING = 1e-4
    if (
        math.tau - TOL_ROUNDING <= closed_angle_u <= math.tau + TOL_ROUNDING
    ) and n_wires <= 1:
        # case 1 : closed u loop, u_range close to 2 pi and only single or no seam.
        pass
    elif (
        math.tau - TOL_ROUNDING <= closed_angle_u <= math.tau + TOL_ROUNDING
    ) and n_wires > 1:
        pass
    elif math.pi - TOL_ROUNDING <= closed_angle_u < math.tau - TOL_ROUNDING:
        # case 2 : more than half tori loop - open u loop, closed angle >= pi and < 2 pi
        NUDGE = 1.0123456789e-4

        point_u_min_v_zero = surface.Value(u_min, 0.0)
        vec_centre_to_u_min_v_zero = gp_Vec(point_u_min_v_zero, centre)
        normal_dir = gp_Dir(axis_vec.Crossed(vec_centre_to_u_min_v_zero))
        plane_axis = gp_Ax3(centre, normal_dir)
        plane_gp = gp_Pln(plane_axis)
        plane_geom = Geom_Plane(plane_gp)
        auxiliary_surfaces.append([plane_geom, TopAbs_FORWARD])

        # print('\n\n here - case 1')

    elif closed_angle_u < math.pi - TOL_ROUNDING:

        if decom_params.aux_surface_tori_aux_surface_simple:
            # carry out simple approach, single plane accross span of major radius

            pass

        # case 3 : less than half tori loop - open u loop, closed angle
        point_u_min_v_zero = surface.Value(u_min, 0.0)
        vec_centre_to_u_min_v_zero = gp_Vec(point_u_min_v_zero, centre)
        normal_dir = gp_Dir(axis_vec.Crossed(vec_centre_to_u_min_v_zero))
        plane_axis = gp_Ax3(centre, normal_dir)
        plane_gp = gp_Pln(plane_axis)
        plane_geom = Geom_Plane(plane_gp)
        auxiliary_surfaces.append([plane_geom, TopAbs_FORWARD])

        # face_1 = BRepBuilderAPI_MakeFace(plane_geom, -1000, 1000, -1000, 1000, decom_params.make_face_tol).Face()

        point_u_max_v_zero = surface.Value(u_max, 0.0)
        vec_centre_to_u_max_v_zero = gp_Vec(point_u_max_v_zero, centre)
        normal_dir = gp_Dir(vec_centre_to_u_max_v_zero.Crossed(axis_vec))
        plane_axis = gp_Ax3(centre, normal_dir)
        plane_gp = gp_Pln(plane_axis)
        plane_geom = Geom_Plane(plane_gp)
        auxiliary_surfaces.append([plane_geom, TopAbs_FORWARD])

        # face_2 = BRepBuilderAPI_MakeFace(plane_geom, -1000, 1000, -1000, 1000, decom_params.make_face_tol).Face()

        # step_file_writer = STEPFileWriter(decom_params=decom_params)
        # step_file_writer.add_solids([face_1], 'face_1')
        # step_file_writer.add_solids([face_2], 'face_2')
        # step_file_writer.write('tori_faces.stp')

        # print('\n\n here - case 2')

        # print(auxiliary_surfaces)
    else:
        msg = f"v_range closed parameter hit unknown value: {closed_angle_v:.2f} pi"
        raise Exception(msg)

    # ██  ██   █████▄ ▄████▄ █████▄  ▄████▄ ██▄  ▄██ ▄█████
    # ██▄▄██   ██▄▄█▀ ██▄▄██ ██▄▄██▄ ██▄▄██ ██ ▀▀ ██ ▀▀▀▄▄▄
    #  ▀██▀    ██     ██  ██ ██   ██ ██  ██ ██    ██ █████▀

    # check if v range closed
    if math.tau - TOL_ROUNDING <= closed_angle_v <= math.tau + TOL_ROUNDING:
        return auxiliary_surfaces

    # get points at edges of v parameter range
    point_u_zero_v_min = surface.Value(0.0, v_min)
    point_u_zero_v_max = surface.Value(0.0, v_max)

    # create vector between points at v parameter range
    vec_u_zero_v_min_u_zero_v_max = gp_Vec(point_u_zero_v_min, point_u_zero_v_max)

    # calculate the angle between the v_min -> v_max vector and the axis of the torus
    theta_1 = axis_vec.Angle(vec_u_zero_v_min_u_zero_v_max)

    # calculate the angle between the normal at v_min
    theta_2 = axis_vec.Angle(normal_vec_at_u_min_v_mid)

    # define the margin each angle can be within a case
    MARGIN = decom_params.aux_surface_tori_angle_tol
    HALF_PI = math.pi / 2.0

    if (HALF_PI - MARGIN) < theta_1 <= (HALF_PI + MARGIN) and theta_2 <= MARGIN:
        # case 1 - tori open top, plane required
        height_from_centre_u_min_v_min = _distance_of_point_along_vector(
            centre, axis_vec, point_u_zero_v_min
        )
        height_from_centre_u_min_v_max = _distance_of_point_along_vector(
            centre, axis_vec, point_u_zero_v_max
        )

        if height_from_centre_u_min_v_min <= height_from_centre_u_min_v_max:
            plane_axis = gp_Ax3(point_u_zero_v_min, axis_direction)
        else:
            plane_axis = gp_Ax3(point_u_zero_v_max, axis_direction)

        plane_gp = gp_Pln(plane_axis)
        plane_geom = Geom_Plane(plane_gp)

        auxiliary_surfaces.append([plane_geom, TopAbs_FORWARD])

    elif (HALF_PI + MARGIN < theta_1 <= math.pi - MARGIN) and (
        MARGIN < theta_2 <= HALF_PI - MARGIN
    ):
        # case 2 - tori open outwards at angle away from axis
        point_cone_apex = point_at_intersection_of_two_lines(
            centre, axis_vec, point_u_zero_v_min, vec_u_zero_v_min_u_zero_v_max
        )
        axis_cone = gp_Ax3(point_cone_apex, gp_Dir(-1.0 * axis_vec))
        cone_geom = Geom_ConicalSurface(axis_cone, math.pi - theta_1, 0.0)

        auxiliary_surfaces.append([cone_geom, TopAbs_FORWARD])

    elif (math.pi - MARGIN < theta_1) and (
        HALF_PI - MARGIN < theta_2 <= HALF_PI + MARGIN
    ):
        # case 3
        radius = distance_of_point_from_axis(axis, point_u_zero_v_min)

        cylinder_geom = Geom_CylindricalSurface(axis, radius)

        auxiliary_surfaces.append([cylinder_geom, TopAbs_REVERSED])

    elif (HALF_PI + MARGIN < theta_1 <= math.pi - MARGIN) and (
        HALF_PI + MARGIN < theta_2 <= math.pi - MARGIN
    ):
        # case 4 - tori open outwards and facing away from axis
        point_cone_apex = point_at_intersection_of_two_lines(
            centre, axis_vec, point_u_zero_v_min, vec_u_zero_v_min_u_zero_v_max
        )
        axis_cone = gp_Ax3(point_cone_apex, axis_direction)
        cone_geom = Geom_ConicalSurface(axis_cone, math.pi - theta_1, 0.0)

        auxiliary_surfaces.append([cone_geom, TopAbs_FORWARD])

    elif (HALF_PI - MARGIN) < theta_1 <= (HALF_PI + MARGIN) and (
        math.pi - MARGIN
    ) < theta_2:
        # case 5 - tori open top, plane required
        height_from_centre_u_min_v_min = _distance_of_point_along_vector(
            centre, axis_vec, point_u_zero_v_min
        )
        height_from_centre_u_min_v_max = _distance_of_point_along_vector(
            centre, axis_vec, point_u_zero_v_max
        )

        if height_from_centre_u_min_v_min >= height_from_centre_u_min_v_max:
            plane_axis = gp_Ax3(point_u_zero_v_min, gp_Dir(-1.0 * axis_vec))
        else:
            plane_axis = gp_Ax3(point_u_zero_v_max, gp_Dir(-1.0 * axis_vec))

        plane_gp = gp_Pln(plane_axis)
        plane_geom = Geom_Plane(plane_gp)

        auxiliary_surfaces.append([plane_geom, TopAbs_FORWARD])

    elif (MARGIN < theta_1 <= HALF_PI - MARGIN) and (
        HALF_PI + MARGIN < theta_2 <= math.pi - MARGIN
    ):
        # case 6
        try:
            point_cone_apex = point_at_intersection_of_two_lines(
                centre, axis_vec, point_u_zero_v_min, vec_u_zero_v_min_u_zero_v_max
            )
        except Exception as e:
            step_file_writer = STEPFileWriter(decom_params)
            step_file_writer.add_solids(faces, "faces")
            step_file_writer.write("analysis/failed_parallel_intersecting_lines.stp")
            raise e

        axis_cone = gp_Ax3(point_cone_apex, gp_Dir(-1.0 * axis_vec))
        cone_geom = Geom_ConicalSurface(axis_cone, theta_1, 0.0)

        auxiliary_surfaces.append([cone_geom, TopAbs_REVERSED])

    elif (theta_1 < MARGIN) and (HALF_PI - MARGIN < theta_2 <= HALF_PI + MARGIN):
        # case 7 - tori open inwards towards axis
        radius = distance_of_point_from_axis(axis, point_u_zero_v_min)

        cylinder_geom = Geom_CylindricalSurface(axis, radius)

        auxiliary_surfaces.append([cylinder_geom, TopAbs_FORWARD])

    elif (MARGIN < theta_1 <= HALF_PI - MARGIN) and (
        MARGIN < theta_2 <= HALF_PI - MARGIN
    ):
        # case 8 - tori open inwards at angle up towards axis
        point_cone_apex = point_at_intersection_of_two_lines(
            centre, axis_vec, point_u_zero_v_min, vec_u_zero_v_min_u_zero_v_max
        )
        axis_cone = gp_Ax3(point_cone_apex, axis_direction)
        cone_geom = Geom_ConicalSurface(axis_cone, theta_1, 0.0)

        auxiliary_surfaces.append([cone_geom, TopAbs_REVERSED])

    else:
        logger.warning("tori case not yet coded... sorry!\n" * 10)

    return auxiliary_surfaces
