"""Module for CAD based operations on STL (stereolithography) geometry.

Contains readers for ascii and binary STL files (:class:`STL`), the facet and
solid containers (:class:`STLFacet`, :class:`STLSolid`), 2D slicing of STL solids
by axis-aligned planes (:class:`STLSlicePX`, :class:`STLSlicePY`,
:class:`STLSlicePZ`) and helpers to sort slice intercepts into closed loops.
"""

from __future__ import annotations

import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .checkvals import check_module_loaded

from collections.abc import Callable, Sequence


def vector_plane_intersection(
    p0: np.ndarray,
    p1: np.ndarray,
    plane_coordinate: np.ndarray | tuple[float, float, float],
    plane_normal: np.ndarray | tuple[float, float, float],
    tol: float = 1e-4,
) -> np.ndarray | None:
    """Find the intersection of the line through two points with a plane.

    Parameters
    ----------
    p0 : np.ndarray
        First point of the line (vector origin).
    p1 : np.ndarray
        Second point of the line (vector end).
    plane_coordinate : np.ndarray | tuple[float, float, float]
        A coordinate on the plane.
    plane_normal : np.ndarray | tuple[float, float, float]
        Normal vector of the plane.
    tol : float, optional
        Minimum absolute dot product of the plane normal and the line vector
        for an intersection to be calculated, by default 1e-4

    Returns
    -------
    np.ndarray | None
        The intersection point, or None if the line is (nearly) parallel to the
        plane. The point may lie outside the segment between ``p0`` and ``p1``.
    """

    # set vector to origin
    vec: np.ndarray = p1 - p0

    # calculate dot product of plane normal to line vector
    dot_plane_norm_vec: float = np.dot(plane_normal, vec)

    if abs(dot_plane_norm_vec) > tol:
        # case where intercept found
        magnitude: float = (
            -np.dot(plane_normal, p0 - plane_coordinate) / dot_plane_norm_vec
        )
        vec = vec * magnitude
        return p0 + vec

    # case that line does not intercept plane
    return None


class STLFacet:
    """A single triangular STL facet.

    Parameters
    ----------
    normal : np.ndarray
        Normal vector of the facet.
    vertex_1 : np.ndarray
        First vertex of the facet.
    vertex_2 : np.ndarray
        Second vertex of the facet.
    vertex_3 : np.ndarray
        Third vertex of the facet.
    """

    def __init__(
        self,
        normal: np.ndarray,
        vertex_1: np.ndarray,
        vertex_2: np.ndarray,
        vertex_3: np.ndarray,
    ) -> None:
        """Initialise the STLFacet."""

        self.normal: np.ndarray = normal
        self._vertex_1: np.ndarray = np.array(vertex_1, dtype=float)
        self._vertex_2: np.ndarray = np.array(vertex_2, dtype=float)
        self._vertex_3: np.ndarray = np.array(vertex_3, dtype=float)

    @property
    def vertices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the three vertices

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            The three vertices
        """

        return (self._vertex_1, self._vertex_2, self._vertex_3)

    @property
    def x_min(self) -> float:
        """Returns the minimum x value across all vertices

        Returns
        -------
        float
            The minimum x value accross the three vertices
        """
        return min(self._vertex_1[0], self._vertex_2[0], self._vertex_3[0])

    @property
    def x_max(self) -> float:
        """Returns the maximum x value across all vertices

        Returns
        -------
        float
            The maximum x value accross the three vertices
        """
        return max(self._vertex_1[0], self._vertex_2[0], self._vertex_3[0])

    @property
    def y_min(self) -> float:
        """Returns the minimum y value across all vertices

        Returns
        -------
        float
            The minimum y value accross the three vertices
        """

        return min(self._vertex_1[1], self._vertex_2[1], self._vertex_3[1])

    @property
    def y_max(self) -> float:
        """Returns the maximum y value across all vertices

        Returns
        -------
        float
            The maximum y value accross the three vertices
        """
        return max(self._vertex_1[1], self._vertex_2[1], self._vertex_3[1])

    @property
    def z_min(self) -> float:
        """Returns the minimum z value across all vertices

        Returns
        -------
        float
            The minimum z value accross the three vertices
        """
        return min(self._vertex_1[2], self._vertex_2[2], self._vertex_3[2])

    @property
    def z_max(self) -> float:
        """Returns the maximum z value across all vertices

        Returns
        -------
        float
            The maximum z value accross the three vertices
        """
        return max(self._vertex_1[2], self._vertex_2[2], self._vertex_3[2])

    def coordinate_in_bounding_box(
        self, coordinate: np.ndarray | None, tol: float = 1e-6
    ) -> bool | None:
        """Checks if coordinate is in the bounding box of the facet

        Parameters
        ----------
        coordinate : np.ndarray | None
            Cartesian coordinate to check.
        tol : float, optional
            Tolerance in x, y or z direction, by default 1e-6

        Returns
        -------
        bool | None
            True if the point lies within the bounding box, None if
            ``coordinate`` is None.
        """

        if coordinate is None:
            return None
        else:
            return (
                (self.x_min - tol <= coordinate[0] <= self.x_max + tol)
                & (self.y_min - tol <= coordinate[1] <= self.y_max + tol)
                & (self.z_min - tol <= coordinate[2] <= self.z_max + tol)
            )


class STLSolid:
    """STLSolid contains a single body made up of STLFacets

    Parameters
    ----------
    facets : list[STLFacet]
        A list of STLFacets
    name : str, optional
        Name label of the STLSolid, by default None
    """

    def __init__(self, facets: list[STLFacet], name: str | None = None) -> None:
        """Initialise the STLSolid."""

        self.name: str | None = name
        self.facets: list[STLFacet] = facets
        self.vertices_loop: bool | None = None

    def slice(
        self, normal_axis: str = "x",
        intercept: float = 0.0,
        create_loops: bool = False
    ) -> STLSlicePX | STLSlicePY | STLSlicePZ:
        """Returns a 2D slice through the STLSolid geometry

        Parameters
        ----------
        normal_axis : str, optional
            The axis to place the normal of the slice plane, x, y or z, by 
            default "x"
        intercept : float, optional
            intercept along the axis, by default 0.0
        create_loops : bool, optional
            Create loops around the body, not always possible to create clean
            loops, by default False

        Returns
        -------
        STLSlicePX | STLSlicePY | STLSlicePZ
            The slice through the STLSolid geometry

        Raises
        ------
        ValueError
            If available normal_axis is not selected
        """

        xs: list[list[float]] = []
        ys: list[list[float]] = []
        zs: list[list[float]] = []

        if normal_axis.lower() == "x":
            vec_intercept: tuple[float, float, float] = (intercept, 0, 0)
            vec_projects: tuple[float, float, float] = (1, 0, 0)
        elif normal_axis.lower() == "y":
            vec_intercept: tuple[float, float, float] = (0, intercept, 0)
            vec_projects: tuple[float, float, float] = (0, 1, 0)
        elif normal_axis.lower() == "z":
            vec_intercept: tuple[float, float, float] = (0, 0, intercept)
            vec_projects: tuple[float, float, float] = (0, 0, 1)
        else:
            msg = f"plane of slice should be either x, y, or z not {normal_axis}"
            raise ValueError(msg)

        for facet in self.facets:
            # line 1 - v1 v2
            inter_1 = vector_plane_intersection(
                facet.vertices[0], facet.vertices[1], vec_intercept, vec_projects
            )
            inter_1 = inter_1 if facet.coordinate_in_bounding_box(inter_1) else None

            # line 2 - v1 v3
            inter_2 = vector_plane_intersection(
                facet.vertices[0], facet.vertices[2], vec_intercept, vec_projects
            )
            inter_2 = inter_2 if facet.coordinate_in_bounding_box(inter_2) else None

            # line 3 - v2 v3
            inter_3 = vector_plane_intersection(
                facet.vertices[1], facet.vertices[2], vec_intercept, vec_projects
            )
            inter_3 = inter_3 if facet.coordinate_in_bounding_box(inter_3) else None

            # calculate intersections between plotting plane and three endges of facet
            inters: list[np.ndarray] = [
                inter for inter in [inter_1, inter_2, inter_3] if inter is not None
            ]

            # check for case that there are 2 intercepts
            if len(inters) == 2:
                if normal_axis.lower() == "x":
                    ys.append([inters[0][1], inters[1][1]])
                    zs.append([inters[0][2], inters[1][2]])
                elif normal_axis.lower() == "y":
                    xs.append([inters[0][0], inters[1][0]])
                    zs.append([inters[0][2], inters[1][2]])
                elif normal_axis.lower() == "z":
                    xs.append([inters[0][0], inters[1][0]])
                    ys.append([inters[0][1], inters[1][1]])
                else:
                    msg = f"normal_axis of slice should be either x, y, or z not {normal_axis}"
                    raise ValueError(msg)

        if normal_axis.lower() == "x":
            return STLSlicePX(ys, zs, self.name, create_loops=create_loops)
        elif normal_axis.lower() == "y":
            return STLSlicePY(xs, zs, self.name, create_loops=create_loops)
        elif normal_axis.lower() == "z":
            return STLSlicePZ(xs, ys, self.name, create_loops=create_loops)


class STL:
    """An STL model

    Parameters
    ----------
    file_path : str | os.PathLike
        Path to input STL file
    format : str, optional
        format, can be either 'ascii' or 'binary', by default "ascii"

    Raises
    ------
    ValueError
        Value error raised if valid format not provided.
    """

    def __init__(self, file_path: str | os.PathLike, format: str = "ascii") -> None:
        """Initialise the STL model and parse the file."""
        self.format: str = format
        self.file_path: str | os.PathLike = file_path
        self.solids: list[STLSolid] = []

        PARSE_METHODS: dict[str, Callable[[str | os.PathLike], None]] = {
            "ascii": self._parse_stl_ascii,
            "binary": self._parse_stl_binary,
        }

        if format in PARSE_METHODS:
            PARSE_METHODS[format](file_path)
        else:
            msg = f"format provided should be ascii or binary, not {format}"
            raise ValueError(msg)

    def _parse_stl_binary(self, file_path: str | os.PathLike) -> None:
        """Parse binary STL files and converts solids into STLSolids

        Parameters
        ----------
        file_path : str | os.PathLike
            Input filepath to binary STL file
        """
        with open(file_path, "rb") as input_file:
            header = input_file.read(80).decode("ascii")
            solid_name = header.split("<")[1]
            n_facets = np.fromfile(input_file, count=1, dtype=np.uint32)[0]
            facets: list[STLFacet] = []

            for i in range(n_facets):
                facet_normal = np.fromfile(input_file, count=3, dtype=np.float32)
                vertex_1 = np.fromfile(input_file, count=3, dtype=np.float32)
                vertex_2 = np.fromfile(input_file, count=3, dtype=np.float32)
                vertex_3 = np.fromfile(input_file, count=3, dtype=np.float32)
                input_file.read(2)

                facet = STLFacet(facet_normal, vertex_1, vertex_2, vertex_3)
                facets.append(facet)

            solid = STLSolid(facets=facets, name=solid_name)
            self.solids.append(solid)

    def _parse_stl_ascii(self, file_path: str | os.PathLike) -> None:
        """Parses ascii STL files and converts solids into STLSolids

        Parameters
        ----------
        file_path : str | os.PathLike
            Input filepath to ascii STL file

        Raises
        ------
        ValueError
            Exception raised is STF facet lines are incongruous
        """

        # read stl file
        with open(file_path, "r") as input_file:
            rows = [r for r in input_file]

            solid_indices_starts = [
                i for i, row in enumerate(rows) if row[:5] == "solid"
            ]
            solid_indices_ends = [
                i for i, row in enumerate(rows) if row[:8] == "endsolid"
            ]

            # iterate over solids in file
            for i_start, i_end in zip(solid_indices_starts, solid_indices_ends):
                # split rows into solid sections
                section_solid = rows[i_start : i_end + 1]

                if len(section_solid[1:-1]) % 7 == 0:
                    sections_facets = [
                        section_solid[1:-1][i : i + 7]
                        for i in range(0, len(section_solid[1:-1]), 7)
                    ]
                else:
                    msg = f"number of lines in facet subsection not equal to 7"
                    raise ValueError(msg)

                facets: list[STLFacet] = []
                for section_facet in sections_facets:
                    # extract normal and vertices from file
                    normal = np.array(
                        [float(val) for val in section_facet[0].split()[-3:]]
                    )
                    vertex_1 = np.array(
                        [float(val) for val in section_facet[2][:-1].split()[-3:]]
                    )
                    vertex_2 = np.array(
                        [float(val) for val in section_facet[3][:-1].split()[-3:]]
                    )
                    vertex_3 = np.array(
                        [float(val) for val in section_facet[4][:-1].split()[-3:]]
                    )

                    facet = STLFacet(normal, vertex_1, vertex_2, vertex_3)
                    facets.append(facet)

                solid_name = rows[i_start][7:-2].split("<")[0]
                solid = STLSolid(facets, name=solid_name)
                self.solids.append(solid)

    def plot_3d(self) -> None:
        """STL plotting function producing 3D plot using matplotlib"""

        fig = plt.figure()
        ax = plt.axes(projection="3d")

        p1: tuple[float, float, float] = (0, 0, 0)
        p2: tuple[float, float, float] = (0, 1, 0)
        p3: tuple[float, float, float] = (1, 0, 0)

        x_min: float = 0.0
        x_max: float = 0.0
        y_min: float = 0.0
        y_max: float = 0.0
        z_min: float = 0.0
        z_max: float = 0.0

        # ax.add_collection3d(Poly3DCollection((p1, p2, p3)))

        for i, facet in enumerate(self._facets):
            for vertex in facet.vertices:
                if vertex[0] < x_min:
                    x_min = vertex[0]
                if vertex[0] > x_max:
                    x_max = vertex[0]
                if vertex[1] < y_min:
                    y_min = vertex[0]
                if vertex[1] > y_max:
                    y_max = vertex[0]
                if vertex[2] < z_min:
                    z_min = vertex[0]
                if vertex[2] > z_max:
                    z_max = vertex[0]

            ax.add_collection3d(Poly3DCollection(facet.vertices))

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        plt.show()

    def plot_2d(self, axis: str = "x", intercept: float = 0.0, grid: bool = True) -> None:
        """STL plotting function producing 2D plot using matplotlib

        Parameters
        ----------
        axis : str, optional
            Axis normal to the plot plane, used to select the axis labels, either
            x, y or z, by default "x". Note the slice itself is always taken
            normal to the z-axis.
        intercept : float, optional
            Intercept of the slice plane along its normal axis, by default 0.0
        grid : bool, optional
            Show a grid on the plot, by default True
        """

        fig, ax = plt.subplots(1)
        plt.gca().set_aspect("equal", adjustable="box")

        for solid in self.solids:
            slice = solid.slice("z", intercept=intercept)
            for x_pair, y_pair in zip(slice.x_pairs, slice.y_pairs):
                ax.plot(x_pair, y_pair, color="black", linewidth=1)

        LABELS: dict[str, list[str]] = {
            "x": ["y-axis (cm)", "z-axis (cm)"],
            "y": ["x-axis (cm)", "z-axis (cm)"],
            "z": ["x-axis (cm)", "y-axis (cm)"],
        }

        ax.set_xlabel(LABELS[axis][0])
        ax.set_ylabel(LABELS[axis][1])
        if grid:
            ax.grid()
        plt.show()


def check_2d_points_are_equal(
    p0: Sequence[float], p1: Sequence[float], tol: float = 1e-4
) -> bool:
    """Checks if a pair of 2D points are equal within an on axis tolerance

    Only the first two components of each point are compared.

    Parameters
    ----------
    p0 : Sequence[float]
        The first point
    p1 : Sequence[float]
        The second point
    tol : float, optional
        The tolerance, on axis, within which the points can be withing, by default 1e-4

    Returns
    -------
    bool
        Boolean value if points are equal within tolerance
    """

    return (p0[0] - tol <= p1[0] <= p0[0] + tol) & (p0[1] - tol <= p1[1] <= p0[1] + tol)


def sort_intercepts_into_loop(
    pairs_1: list[list[float]], pairs_2: list[list[float]], solid_name: str, axis: str
) -> tuple[
    list[float], list[float], list[STLSliceLoopPX | STLSliceLoopPY | STLSliceLoopPZ]
]:
    """Sorts intercepts into contiguous loops.

    This is unstable as not all STL pairs will form loops when second loops
    intercept or touch the first loop.

    Parameters
    ----------
    pairs_1 : list[list[float]]
        List of pairs of the first in-plane coordinate of each line segment
    pairs_2 : list[list[float]]
        List of pairs of the second in-plane coordinate of each line segment
    solid_name : str
        Label as name of loops
    axis : str
        Axis for normal of plane for loops, either x, y or z

    Returns
    -------
    tuple[list[float], list[float], list]
        The first and second coordinates of the final loop start point, each in
        a single element list, and the list of closed loops (STLSliceLoopPX,
        STLSliceLoopPY or STLSliceLoopPZ depending on ``axis``)

    Raises
    ------
    Exception
        If the next point of a loop cannot be found; a debugging plot is saved
        to plot.png before raising.
    """

    loops: list[STLSliceLoopPX | STLSliceLoopPY | STLSliceLoopPZ] = []

    # set starting points
    p_start = (pairs_1[0][0], pairs_2[0][0])
    p_next = (pairs_1[0][1], pairs_2[0][1])

    pairs_1 = pairs_1[1:]
    pairs_2 = pairs_2[1:]

    # create stores for points in current loop
    axis_1 = [p_start[0]]
    axis_2 = [p_start[1]]

    # STLSliceLoop axis lookup
    STL_SLICE_LOOP_AXIS_LOOKUP: dict[
        str, type[STLSliceLoopPX] | type[STLSliceLoopPY] | type[STLSliceLoopPZ]
    ] = {
        "x": STLSliceLoopPX,
        "y": STLSliceLoopPY,
        "z": STLSliceLoopPZ,
    }

    while len(pairs_1) > 0:
        count = 0

        # loop through all other pairs of points to find matching within tolerance
        for i, (pair_1, pair_2) in enumerate(zip(pairs_1, pairs_2)):
            p0 = (pair_1[0], pair_2[0])
            p1 = (pair_1[1], pair_2[1])

            # check if either of the two points in the next vertex match
            if check_2d_points_are_equal(
                [axis_1[-1], axis_2[-1]], p0
            ) or check_2d_points_are_equal([axis_1[-1], axis_2[-1]], p1):
                # case where two bodies have coincident surfaces and point
                # being checked tries to return to a previous point.
                continue
            elif check_2d_points_are_equal(p_next, p0):
                axis_1.append(p0[0])
                axis_2.append(p0[1])
                p_next = p1
                pairs_1.remove(pair_1)
                pairs_2.remove(pair_2)

                # case for loop being closed
                if check_2d_points_are_equal(p_next, p_start):
                    axis_1.append(p_next[0])
                    axis_2.append(p_next[1])

                    loop = STL_SLICE_LOOP_AXIS_LOOKUP[axis](axis_1, axis_2, solid_name)
                    loops.append(loop)

                    # check for remaining points and set up new loop
                    if len(pairs_1) > 0:
                        # set starting points
                        p_start = (pairs_1[0][0], pairs_2[0][0])
                        p_next = (pairs_1[0][1], pairs_2[0][1])

                        pairs_1 = pairs_1[1:]
                        pairs_2 = pairs_2[1:]

                        # create stores for points in currnet loop
                        axis_1 = [p_start[0]]
                        axis_2 = [p_start[1]]
                    break
            elif check_2d_points_are_equal(p_next, p1):
                axis_1.append(p1[0])
                axis_2.append(p1[1])
                p_next = p0
                pairs_1.remove(pair_1)
                pairs_2.remove(pair_2)

                # case for loop being closed
                if check_2d_points_are_equal(p_next, p_start):
                    axis_1.append(p_next[0])
                    axis_2.append(p_next[1])

                    loop = STL_SLICE_LOOP_AXIS_LOOKUP[axis](axis_1, axis_2, solid_name)
                    loops.append(loop)

                    # check for remaining points and set up new loop
                    if len(pairs_1) > 0:
                        # set starting points
                        p_start = (pairs_1[0][0], pairs_2[0][0])
                        p_next = (pairs_1[0][1], pairs_2[0][1])

                        pairs_1 = pairs_1[1:]
                        pairs_2 = pairs_2[1:]

                        # create stores for points in currnet loop
                        axis_1 = [p_start[0]]
                        axis_2 = [p_start[1]]
                    break

            count += 1

        if count > len(pairs_1):
            fig, axes = plt.subplots(2, 2, figsize=(32, 32))
            for pair_1, pair_2 in zip(pairs_1, pairs_2):
                axes[0][0].plot(pair_1, pair_2)
                axes[0][1].plot(pair_1, pair_2, color="black")
                axes[1][0].plot(pair_1, pair_2)
                axes[1][1].plot(pair_1, pair_2, color="black")

            axes[0][1].plot(axis_1, axis_2, color="red")
            axes[1][1].plot(axis_1, axis_2, color="red")

            x_min = p_next[0] - 50
            x_max = p_next[0] + 50
            y_min = p_next[1] - 50
            y_max = p_next[1] + 50

            for axes_row in axes:

                axes_row[1].plot([p_next[0], p_next[0]], [y_min, y_max])
                axes_row[1].plot([x_min, x_max], [p_next[1], p_next[1]])

            for ax in axes[1]:
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)

            for axes_row in axes:
                for ax in axes_row:
                    ax.set_aspect("equal")

            plt.tight_layout()
            plt.savefig("plot.png")
            plt.show()

            msg = f"sort_intercept_into_loops failed to find next point"
            raise Exception(msg)

    return [p_start[0]], [p_start[1]], loops


class STLSlice:
    """Base class for an STLSlice."""

    def __init__(self) -> None:
        """Initialise the STLSlice base class."""
        pass


class STLSlicePX(STLSlice):
    """STLSlice with normal parallel to the x-axis

    Parameters
    ----------
    y_pairs : list[list[float]]
        List of y pairs, the y coordinates of each slice line segment
    z_pairs : list[list[float]]
        List of z pairs, the z coordinates of each slice line segment
    solid_name : str
        Name of solid
    create_loops : bool, optional
        Boolean to create loops, by default False
    """

    def __init__(
        self,
        y_pairs: list[list[float]],
        z_pairs: list[list[float]],
        solid_name: str,
        create_loops: bool = False,
    ) -> None:
        """Initialise the STLSlicePX and optionally sort it into loops."""

        # object parameters
        self._solid_name: str = solid_name

        # store vertices of intesection
        self._ys: list = []
        self._zs: list = []

        self.y_pairs: list = list(y_pairs)
        self.z_pairs: list = list(z_pairs)

        self.loops: list = []

        if len(y_pairs) > 0 and create_loops:
            self._ys, self._zs, self.loops = sort_intercepts_into_loop(
                y_pairs, z_pairs, solid_name, "x"
            )


class STLSlicePY(STLSlice):
    """STLSlice with normal parallel to the y-axis

    Parameters
    ----------
    x_pairs : list[list[float]]
        List of x pairs, the x coordinates of each slice line segment
    z_pairs : list[list[float]]
        List of z pairs, the z coordinates of each slice line segment
    solid_name : str
        Name of solid
    create_loops : bool, optional
        Boolean to create loops, by default False
    """

    def __init__(
        self,
        x_pairs: list[list[float]],
        z_pairs: list[list[float]],
        solid_name: str,
        create_loops: bool = False,
    ) -> None:
        """Initialise the STLSlicePY and optionally sort it into loops."""

        # object parameters
        self._solid_name: str = solid_name

        # store vertices of intesection
        self._xs: list = []
        self._zs: list = []

        self.x_pairs: list = list(x_pairs)
        self.z_pairs: list = list(z_pairs)

        self.loops: list = []

        if len(x_pairs) > 0 and create_loops:
            self._ys, self._zs, self.loops = sort_intercepts_into_loop(
                x_pairs, z_pairs, solid_name, "y"
            )


class STLSlicePZ(STLSlice):
    """STLSlice with normal parallel to the z-axis

    Parameters
    ----------
    x_pairs : list[list[float]]
        List of x pairs, the x coordinates of each slice line segment
    y_pairs : list[list[float]]
        List of y pairs, the y coordinates of each slice line segment
    solid_name : str
        Name of solid
    create_loops : bool, optional
        Boolean to create loops, by default False
    """

    def __init__(
        self,
        x_pairs: list[list[float]],
        y_pairs: list[list[float]],
        solid_name: str,
        create_loops: bool = False,
    ) -> None:
        """Initialise the STLSlicePZ and optionally sort it into loops."""

        # object parameters
        self._solid_name: str = solid_name

        self.x_pairs: list[list[float]] = list(x_pairs)
        self.y_pairs: list[list[float]] = list(y_pairs)

        self.loops: list = []

        if len(x_pairs) > 0 and create_loops:
            self._xs, self._ys, self.loops = sort_intercepts_into_loop(
                x_pairs, y_pairs, solid_name, "z"
            )


class STLSliceLoopPX:
    """STLSliceLoop with normal of slice parallel to x-axis

    Parameters
    ----------
    ys : list[float]
        List of y points
    zs : list[float]
        List of z points
    solid_name : str
        Name of solid
    """

    def __init__(self, ys: list[float], zs: list[float], solid_name: str) -> None:
        """Initialise the STLSliceLoopPX."""
        self.ys: list[float] = ys
        self.zs: list[float] = zs
        self.solid_name: str = solid_name


class STLSliceLoopPY:
    """STLSliceLoop with normal of slice parallel to y-axis

    Parameters
    ----------
    xs : list[float]
        List of x points
    zs : list[float]
        List of z points
    solid_name : str
        Name of solid
    """

    def __init__(self, xs: list[float], zs: list[float], solid_name: str) -> None:
        """Initialise the STLSliceLoopPY."""
        self.xs: list[float] = xs
        self.zs: list[float] = zs
        self.solid_name: str = solid_name


class STLSliceLoopPZ:
    """STLSliceLoop with normal of slice parallel to z-axis

    Parameters
    ----------
    xs : list[float]
        List of x points
    ys : list[float]
        List of y points
    solid_name : str
        Name of solid
    """

    def __init__(self, xs: list[float], ys: list[float], solid_name: str) -> None:
        """Initialise the STLSliceLoopPZ."""
        self.xs: list[float] = xs
        self.ys: list[float] = ys
        self.solid_name: str = solid_name
