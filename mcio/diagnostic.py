"""The diagnostics module provides methods to diagnose issues with CAD
to CSG conversion.
"""

import os
import re
from typing import List, Union

try:
    import casadi
    import OCP
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
    from OCP.gp import gp_Pnt, gp_Vec
    from OCP.TopoDS import TopoDS_Edge, TopoDS_Shape
except ImportError:
    pass

from mcio import logger

try:
    from .ocp import STEPFileWriter
except ImportError:
    pass


def lost_particles_from_mcnp_run(
    input_file_path: Union[str, os.PathLike],
    output_file_path: Union[str, os.PathLike] = "lost_particles.stp",
    max_particles: int = 100,
    vector_length: float = 250.0,
    radius: float = 2.0,
) -> None:
    """Extract lost particle positions and trajectory direction vectors from an MCNP output file
    and save them as CAD shapes in a STEP file.

    Parameters
    ----------
    input_file_path : str or os.PathLike
        Path to the MCNP output or print file containing lost particle records.
    output_file_path : str or os.PathLike, optional
        Destination path for the output STEP file, by default 'lost_particles.stp'.
    max_particles : int, optional
        Maximum number of lost particle events to extract, by default 100.
    vector_length : float, optional
        Length of the direction vector indicator lines, by default 250.0.
    radius : float, optional
        Radius of the sphere markers generated at lost particle locations, by default 2.0.

    Raises
    ------
    Exception
        If the calculated direction cosine magnitude is zero for a particle trajectory.
    """

    with open(input_file_path, "r") as input_file:
        rows: List[str] = [r for r in input_file]

    # define regex sub group expressions
    exp_numeric_exponential = r"[-]{0,1}[0-9]+\.[0-9]+[eE][+-][0-9]+"
    exp_coord = r"(?:x,y,z coordinates\:)"
    exp_dir = r"(?:u,v,w direction cosines\:)"
    exp_spaces = r"(?:\s)+"
    exp_coord_x = f"(?P<x>{exp_numeric_exponential})"
    exp_coord_y = f"(?P<y>{exp_numeric_exponential})"
    exp_coord_z = f"(?P<z>{exp_numeric_exponential})"
    exp_dir_u = f"(?P<u>{exp_numeric_exponential})"
    exp_dir_v = f"(?P<v>{exp_numeric_exponential})"
    exp_dir_w = f"(?P<w>{exp_numeric_exponential})"

    # compile regex expressions
    expression_coord = re.compile(
        f"{exp_coord}{exp_spaces}{exp_coord_x}{exp_spaces}{exp_coord_y}{exp_spaces}{exp_coord_z}{exp_spaces}"
    )
    expression_dir = re.compile(
        f"{exp_dir}{exp_spaces}{exp_dir_u}{exp_spaces}{exp_dir_v}{exp_spaces}{exp_dir_w}"
    )

    # iterate over rows
    spheres: List[TopoDS_Shape] = []
    edges: List[TopoDS_Edge] = []
    sphere_count: int = 0
    for i, row in enumerate(rows):
        if search := re.search(expression_coord, row):
            x = float(search.group("x"))
            y = float(search.group("y"))
            z = float(search.group("z"))

            search_direction = re.search(expression_dir, rows[i + 1])

            u_cosine = float(search_direction.group("u"))
            v_cosine = float(search_direction.group("v"))
            w_cosine = float(search_direction.group("w"))

            magnitude = u_cosine**2 + v_cosine**2 + w_cosine**2

            if magnitude == 0:
                msg = "direction consine magnitude is zero"
                raise Exception(msg)

            u_vec = u_cosine / magnitude
            v_vec = v_cosine / magnitude
            w_vec = w_cosine / magnitude

            vec_uvw = gp_Vec(u_vec, v_vec, w_vec)
            vec_uvw.Normalize()
            vec_uvw.Multiply(vector_length)

            point_xyz = gp_Pnt(x, y, z)
            point_uvw = gp_Pnt(point_xyz.XYZ() + vec_uvw.XYZ())
            sphere = BRepPrimAPI_MakeSphere(point_xyz, radius).Shape()
            spheres.append(sphere)
            sphere_count += 1

            edge = BRepBuilderAPI_MakeEdge(point_xyz, point_uvw).Edge()
            edges.append(edge)

            if sphere_count >= max_particles:
                break

    n_lost_particles = len(spheres)
    if n_lost_particles == 0:
        msg = f"0 lost particles extracted from {input_file_path}"
        logger.info(msg)
        print(msg)
    else:
        # write lost particle spheres and vectors to step file
        step_file_writer = STEPFileWriter()
        step_file_writer.add_solids(spheres, "sphere")
        step_file_writer.add_solids(edges, "vectors")
        output_path = step_file_writer.write(output_file_path)

        # write to logger and terminal
        msg = f"{n_lost_particles} lost particles extracted from {input_file_path} - see {output_path}"
        logger.info(msg)
        print(msg)
