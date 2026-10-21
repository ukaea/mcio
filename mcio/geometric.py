"""Collection of simple geometry methods.

Contains helper functions for points, vectors and planes, the axis-aligned
:class:`BoundingBox` (AABB) used for octree splitting and intersection tests, and
the :class:`OrientedBoundingBox` (OBB) with a Separating Axis Theorem overlap
test.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Tuple, Union

import numpy as np


def check_point_on_axis(x0: float, y0: float, z0: float, tolerance: float) -> str:
    """Check if a Cartesian point is located on either of the Cartesian x, y, or z axes.

    Parameters
    ----------
    x0 : float
        x coordinate of point.
    y0 : float
        y coordinate of point.
    z0 : float
        z coordinate of point.
    tolerance : float
        Tolerance for x, y, or z points to be within an axis.

    Returns
    -------
    str
        String describing location: 'at_origin', 'on_x_axis', 'on_y_axis',
        'on_z_axis', or 'off_axis'.
    """

    a_in_tol: bool = abs(x0) < tolerance
    b_in_tol: bool = abs(y0) < tolerance
    c_in_tol: bool = abs(z0) < tolerance

    if a_in_tol and b_in_tol and c_in_tol:
        return "at_origin"
    elif b_in_tol and c_in_tol:
        return "on_x_axis"
    elif a_in_tol and c_in_tol:
        return "on_y_axis"
    elif a_in_tol and b_in_tol:
        return "on_z_axis"
    else:
        return "off_axis"


def check_vector_on_axis(
    u0: float, v0: float, w0: float, tol_axis: float
) -> str:
    """Check if a direction vector lies along a primary axis within tolerance.

    Parameters
    ----------
    u0 : float
        x direction cosine or component.
    v0 : float
        y direction cosine or component.
    w0 : float
        z direction cosine or component.
    tol_axis : float
        Absolute tolerance for components to be considered zero.

    Returns
    -------
    str
        String indicating axis orientation: 'on_x_pos', 'on_x_neg', 'on_y_pos',
        'on_y_neg', 'on_z_pos', 'on_z_neg', or 'off axis'.
    """
    # check if axis within tolerance of axis vector
    u_in_tol: bool = abs(u0) < tol_axis
    v_in_tol: bool = abs(v0) < tol_axis
    w_in_tol: bool = abs(w0) < tol_axis

    # return string indicating if vector is on an axis and its direction
    if (not u_in_tol) and (v_in_tol) and (w_in_tol) and u0 >= 0:
        return "on_x_pos"
    elif (not u_in_tol) and (v_in_tol) and (w_in_tol) and u0 < 0:
        return "on_x_neg"
    elif (not v_in_tol) and (u_in_tol) and (w_in_tol) and v0 >= 0:
        return "on_y_pos"
    elif (not v_in_tol) and (u_in_tol) and (w_in_tol) and v0 < 0:
        return "on_y_neg"
    elif (not w_in_tol) and (u_in_tol) and (v_in_tol) and w0 >= 0:
        return "on_z_pos"
    elif (not w_in_tol) and (u_in_tol) and (v_in_tol) and w0 < 0:
        return "on_z_neg"
    else:
        return "off axis"


def check_vector_parallel_to_axis(
    u0: float, v0: float, w0: float, tol_axis: float
) -> Union[str, bool]:
    """Determine if a vector is parallel to one of the Cartesian axes.

    Parameters
    ----------
    u0 : float
        x component of vector.
    v0 : float
        y component of vector.
    w0 : float
        z component of vector.
    tol_axis : float
        Tolerance for non-principal axis components.

    Returns
    -------
    str or bool
        String indicating axis alignment ('parallel_to_x', 'parallel_to_y',
        or 'parallel_to_z') if parallel, otherwise False.
    """
    # check if axis within tolerance of axis vector
    u_in_tol: bool = abs(u0) < tol_axis
    v_in_tol: bool = abs(v0) < tol_axis
    w_in_tol: bool = abs(w0) < tol_axis

    if (not u_in_tol) and v_in_tol and w_in_tol:
        return "parallel_to_x"
    elif u_in_tol and (not v_in_tol) and w_in_tol:
        return "parallel_to_y"
    elif u_in_tol and v_in_tol and (not w_in_tol):
        return "parallel_to_z"
    else:
        return False


def calculate_plane_parameters_from_point_and_normal(
    x0: float,
    y0: float,
    z0: float,
    u0: float,
    v0: float,
    w0: float,
    reverse: bool = False,
) -> Tuple[float, float, float, float]:
    """Calculate MCNP plane parameters from a point and a normal vector.

    The returned coefficients describe the plane in the MCNP ``P`` card form
    ``a*x + b*y + c*z = d``, with ``d = a*x0 + b*y0 + c*z0``. The normal is not
    normalised.

    Parameters
    ----------
    x0 : float
        x coordinate of a point on the plane.
    y0 : float
        y coordinate of a point on the plane.
    z0 : float
        z coordinate of a point on the plane.
    u0 : float
        x component of normal vector.
    v0 : float
        y component of normal vector.
    w0 : float
        z component of normal vector.
    reverse : bool, optional
        If True, normal vector orientation is reversed, by default False.

    Returns
    -------
    tuple of float
        Plane coefficients (a, b, c, d) for ``a*x + b*y + c*z = d``.
    """
    # vector for forward or reverse of plane
    sign: float = -1.0 if reverse else 1.0

    # calculate the plane in the forward orientation
    a: float = sign * u0
    b: float = sign * v0
    c: float = sign * w0
    d: float = a * x0 + b * y0 + c * z0

    # return parameters
    return a, b, c, d


def calculate_plane_from_three_points(
    x0: float,
    y0: float,
    z0: float,
    x1: float,
    y1: float,
    z1: float,
    x2: float,
    y2: float,
    z2: float,
) -> Tuple[float, float, float, float]:
    """Calculate plane coefficients (a, b, c, d) passing through three 3D points.

    The normal is ``(p1 - p0) x (p2 - p0)`` (not normalised), and the plane is
    returned in the form ``a*x + b*y + c*z = d``.

    Parameters
    ----------
    x0 : float
        x coordinate of the first point.
    y0 : float
        y coordinate of the first point.
    z0 : float
        z coordinate of the first point.
    x1 : float
        x coordinate of the second point.
    y1 : float
        y coordinate of the second point.
    z1 : float
        z coordinate of the second point.
    x2 : float
        x coordinate of the third point.
    y2 : float
        y coordinate of the third point.
    z2 : float
        z coordinate of the third point.

    Returns
    -------
    tuple of float
        Plane coefficients (a, b, c, d) for ``a*x + b*y + c*z = d``.
    """
    # create two vectors from p1 -> p2 and p1 -> p3
    vec_1 = np.array([x1 - x0, y1 - y0, z1 - z0])
    vec_2 = np.array([x2 - x0, y2 - y0, z2 - z0])

    # calculate the normal vector using the cross product
    normal = np.cross(vec_1, vec_2)

    # calculate the general parameters for a plane
    a, b, c, d = calculate_plane_parameters_from_point_and_normal(
        x0, y0, z0, *normal
    )

    return a, b, c, d


def angle_between_vectors(
    vec_1: Union[List[float], np.ndarray], vec_2: Union[List[float], np.ndarray]
) -> float:
    """Calculate the angle in radians between two 3D vectors.

    Parameters
    ----------
    vec_1 : list of float or numpy.ndarray
        First vector.
    vec_2 : list of float or numpy.ndarray
        Second vector.

    Returns
    -------
    float
        Angle between vectors in radians.
    """
    # calculate dot product
    dot_product = np.dot(vec_1, vec_2)

    # calculate the magnitude of each vector
    magnitude_vec_1 = np.linalg.norm(vec_1)
    magnitude_vec_2 = np.linalg.norm(vec_2)

    # calculate the angle between the vectors in radians, correcting for rounding errors passed to arccos
    angle = np.arccos(
        np.round(
            dot_product / (magnitude_vec_1 * magnitude_vec_2), decimals=6
        )
    )

    # return the angle between vectors in radians
    return float(angle)


def distance_between_points(
    x0: float, y0: float, z0: float, x1: float, y1: float, z1: float
) -> float:
    """Calculate the Euclidean distance between two 3D points.

    Parameters
    ----------
    x0 : float
        x coordinate of first point.
    y0 : float
        y coordinate of first point.
    z0 : float
        z coordinate of first point.
    x1 : float
        x coordinate of second point.
    y1 : float
        y coordinate of second point.
    z1 : float
        z coordinate of second point.

    Returns
    -------
    float
        Euclidean distance between the two points.
    """
    # calculate the distance between two points
    distance = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)

    # return the distance between two points
    return distance


def values_within_tolerance(
    val_1: float, val_2: float, tolerance: float
) -> bool:
    """Check if two values are within a given absolute tolerance of each other.

    Parameters
    ----------
    val_1 : float
        First scalar value.
    val_2 : float
        Second scalar value.
    tolerance : float
        Maximum allowable absolute difference.

    Returns
    -------
    bool
        True if val_2 is within (val_1 - tolerance, val_1 + tolerance), False otherwise.
    """
    return (val_1 - tolerance) < val_2 < (val_1 + tolerance)


def normalise_np_vector(vec: np.ndarray) -> np.ndarray:
    """Normalise a NumPy vector to unit length.

    Parameters
    ----------
    vec : numpy.ndarray
        Vector to normalise.

    Returns
    -------
    numpy.ndarray
        Unit vector in the same direction, or the original vector unchanged if
        it has zero length.
    """
    euclidean_norm = np.sqrt(np.sum(vec**2))
    if euclidean_norm == 0:
        return vec
    else:
        return vec / euclidean_norm


class BoundingBox:
    """Axis-aligned rectilinear bounding box (AABB).

    Parameters
    ----------
    x_min : float
        Minimum x boundary.
    x_max : float
        Maximum x boundary.
    y_min : float
        Minimum y boundary.
    y_max : float
        Maximum y boundary.
    z_min : float
        Minimum z boundary.
    z_max : float
        Maximum z boundary.
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_min: float,
        z_max: float,
    ) -> None:
        """Initialise the BoundingBox and derive its lengths and midpoints."""
        self._x_min: float = x_min
        self._x_max: float = x_max
        self._y_min: float = y_min
        self._y_max: float = y_max
        self._z_min: float = z_min
        self._z_max: float = z_max

        self._x_length: float = self._x_max - self._x_min
        self._y_length: float = self._y_max - self._y_min
        self._z_length: float = self._z_max - self._z_min

        self._x_mid: float = (self._x_min + self._x_max) / 2.0
        self._y_mid: float = (self._y_min + self._y_max) / 2.0
        self._z_mid: float = (self._z_min + self._z_max) / 2.0

    @property
    def minimums(self) -> Tuple[float, float, float]:
        """Tuple[float, float, float]: Minimum (x, y, z) boundaries."""
        return self._x_min, self._y_min, self._z_min

    @property
    def maximums(self) -> Tuple[float, float, float]:
        """Tuple[float, float, float]: Maximum (x, y, z) boundaries."""
        return self._x_max, self._y_max, self._z_max

    @property
    def x_min(self) -> float:
        """float: Minimum x coordinate."""
        return self._x_min

    @property
    def x_max(self) -> float:
        """float: Maximum x coordinate."""
        return self._x_max

    @property
    def y_min(self) -> float:
        """float: Minimum y coordinate."""
        return self._y_min

    @property
    def y_max(self) -> float:
        """float: Maximum y coordinate."""
        return self._y_max

    @property
    def z_min(self) -> float:
        """float: Minimum z coordinate."""
        return self._z_min

    @property
    def z_max(self) -> float:
        """float: Maximum z coordinate."""
        return self._z_max

    @property
    def x_mid(self) -> float:
        """float: Midpoint x coordinate."""
        return self._x_mid

    @property
    def y_mid(self) -> float:
        """float: Midpoint y coordinate."""
        return self._y_mid

    @property
    def z_mid(self) -> float:
        """float: Midpoint z coordinate."""
        return self._z_mid

    @property
    def radius_enclosure(self) -> float:
        """float: Maximum half-diagonal across major 2D projection planes."""
        radius_xy = (
            (self._x_length / 2.0) ** 2 + (self._y_length / 2.0) ** 2
        ) ** 0.5
        radius_xz = (
            (self._x_length / 2.0) ** 2 + (self._z_length / 2.0) ** 2
        ) ** 0.5
        radius_yz = (
            (self._y_length / 2.0) ** 2 + (self._z_length / 2.0) ** 2
        ) ** 0.5

        return max([radius_xy, radius_xz, radius_yz])

    @property
    def x_length(self) -> float:
        """float: Length along the x axis."""
        return self._x_length

    @x_length.setter
    def x_length(self, x_length: float) -> None:
        """Set the x length, keeping the x midpoint fixed.

        Parameters
        ----------
        x_length : float
            New length along the x axis.
        """
        self._x_min = self._x_mid - (x_length / 2.0)
        self._x_max = self._x_mid + (x_length / 2.0)
        self._x_length = x_length

    @property
    def y_length(self) -> float:
        """float: Length along the y axis."""
        return self._y_length

    @y_length.setter
    def y_length(self, y_length: float) -> None:
        """Set the y length, keeping the y midpoint fixed.

        Parameters
        ----------
        y_length : float
            New length along the y axis.
        """
        self._y_min = self._y_mid - (y_length / 2.0)
        self._y_max = self._y_mid + (y_length / 2.0)
        self._y_length = y_length

    @property
    def z_length(self) -> float:
        """float: Length along the z axis."""
        return self._z_length

    @z_length.setter
    def z_length(self, z_length: float) -> None:
        """Set the z length, keeping the z midpoint fixed.

        Parameters
        ----------
        z_length : float
            New length along the z axis.
        """
        self._z_min = self._z_mid - (z_length / 2.0)
        self._z_max = self._z_mid + (z_length / 2.0)
        self._z_length = z_length

    @property
    def centre(self) -> Tuple[float, float, float]:
        """Tuple[float, float, float]: Center coordinates (x, y, z)."""
        return self._x_mid, self._y_mid, self._z_mid

    @property
    def extent(self) -> float:
        """float: Half-length of the longest dimension."""
        return max([self._x_length, self._y_length, self._z_length]) / 2.0

    @property
    def volume(self) -> float:
        """Returns the geometric volume of the rectilinear bounding box.

        Returns
        -------
        float
            Geometric volume of the bounding box.
        """
        return self._x_length * self._y_length * self._z_length

    def make_cubic(self) -> "BoundingBox":
        """Construct a new cubic bounding box centered on this box.

        Returns
        -------
        BoundingBox
            A new cubic BoundingBox instance.
        """
        # Find longest length
        max_length_halved = (
            max(self._x_length, self._y_length, self._z_length) / 2.0
        )

        # create new bounding box with all length equal to the maximum length
        bounding_box_cubic = BoundingBox(
            x_min=self._x_mid - max_length_halved,
            x_max=self._x_mid + max_length_halved,
            y_min=self._y_mid - max_length_halved,
            y_max=self._y_mid + max_length_halved,
            z_min=self._z_mid - max_length_halved,
            z_max=self._z_mid + max_length_halved,
        )

        # return new bounding box
        return bounding_box_cubic

    def __str__(self) -> str:
        """Return a multi-line listing of the box limits and midpoints.

        Returns
        -------
        str
            One ``name = value`` line per limit and midpoint.
        """
        string = ""
        string += f"x_min = {self._x_min:.6e}\n"
        string += f"x_max = {self._x_max:.6e}\n"
        string += f"y_min = {self._y_min:.6e}\n"
        string += f"y_max = {self._y_max:.6e}\n"
        string += f"z_min = {self._z_min:.6e}\n"
        string += f"z_max = {self._z_max:.6e}\n"
        string += f"x_mid = {self.x_mid:.6e}\n"
        string += f"y_mid = {self.y_mid:.6e}\n"
        string += f"z_mid = {self.z_mid:.6e}\n"

        return string

    def __add__(self, value: float) -> "BoundingBox":
        """Return a new box grown by ``value`` on every side.

        Parameters
        ----------
        value : float
            Distance added to each face of the box (negative values shrink it).

        Returns
        -------
        BoundingBox
            New expanded bounding box.

        """
        x_min = self._x_mid - (self._x_length / 2.0) - value
        x_max = self._x_mid + (self._x_length / 2.0) + value

        y_min = self._y_mid - (self._y_length / 2.0) - value
        y_max = self._y_mid + (self._y_length / 2.0) + value

        z_min = self._z_mid - (self._z_length / 2.0) - value
        z_max = self._z_mid + (self._z_length / 2.0) + value

        bounding_box = BoundingBox(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
        )

        return bounding_box

    def __radd__(self, value: float) -> "BoundingBox":
        """Support ``value + bounding_box`` by delegating to :meth:`__add__`.

        Parameters
        ----------
        value : float
            Distance added to each face of the box.

        Returns
        -------
        BoundingBox
            New expanded bounding box.
        """
        return self + value

    def __mul__(self, multiplier: float) -> "BoundingBox":
        """Return a new box scaled about its centre.

        Parameters
        ----------
        multiplier : float
            Scale factor applied to each length.

        Returns
        -------
        BoundingBox
            New scaled bounding box with the same centre.
        """
        # scale bounding box by multiplier
        x_min = self._x_mid - ((self._x_length / 2.0) * multiplier)
        x_max = self._x_mid + ((self._x_length / 2.0) * multiplier)
        y_min = self._y_mid - ((self._y_length / 2.0) * multiplier)
        y_max = self._y_mid + ((self._y_length / 2.0) * multiplier)
        z_min = self._z_mid - ((self._z_length / 2.0) * multiplier)
        z_max = self._z_mid + ((self._z_length / 2.0) * multiplier)

        bounding_box = BoundingBox(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
        )

        return bounding_box

    def __or__(self, other: "BoundingBox") -> "BoundingBox":
        """Return the smallest box enclosing both boxes (``self | other``).

        Parameters
        ----------
        other : BoundingBox
            Bounding box to combine with.

        Returns
        -------
        BoundingBox
            Union bounding box.
        """
        x_min = min(self._x_min, other._x_min)
        x_max = max(self._x_max, other._x_max)
        y_min = min(self._y_min, other._y_min)
        y_max = max(self._y_max, other._y_max)
        z_min = min(self._z_min, other._z_min)
        z_max = max(self._z_max, other._z_max)

        return BoundingBox(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
        )

    def __xor__(self, other: Union["BoundingBox", "OrientedBoundingBox"]) -> bool:
        """Check if two bounding boxes overlap (``self ^ other``).

        For an AABB the test uses strict inequalities, so boxes that only touch
        are not considered intersecting. An OBB ``other`` is delegated to
        :meth:`OrientedBoundingBox.__xor__`.

        Parameters
        ----------
        other : BoundingBox or OrientedBoundingBox
            Target bounding box to intersect with.

        Returns
        -------
        bool
            True if boxes intersect, False otherwise.
        """
        # Route to OBB's SAT algorithm if checking against an Oriented Bounding Box
        if isinstance(other, OrientedBoundingBox):
            return other ^ self

        if (
            self._x_max > other._x_min
            and self._x_min < other._x_max
            and self._y_max > other._y_min
            and self._y_min < other._y_max
            and self._z_max > other._z_min
            and self._z_min < other._z_max
        ):
            return True
        else:
            return False

    def split_to_halves(self) -> Dict[str, "BoundingBox"]:
        """Split the bounding box into two equal halves along its longest dimension.

        Returns
        -------
        dict of {str : BoundingBox}
            Dictionary containing the split child BoundingBox instances.
        """
        tol = 1e-6
        x_y_lenghts_equal = values_within_tolerance(
            self._x_length, self._y_length, tol
        )
        x_z_lengths_equal = values_within_tolerance(
            self._x_length, self._z_length, tol
        )
        y_z_lengths_equal = values_within_tolerance(
            self._y_length, self._z_length, tol
        )

        max_length = max(self._x_length, self._y_length, self._z_length)

        x_full_length = values_within_tolerance(
            self._x_length, max_length, tol
        )
        y_full_length = values_within_tolerance(
            self._y_length, max_length, tol
        )
        z_full_length = values_within_tolerance(
            self._z_length, max_length, tol
        )

        # case where px split required (square or x is longest length)
        if (x_y_lenghts_equal and x_z_lengths_equal) or x_full_length:
            bounding_boxes_split_to_halves = {
                "x_min": BoundingBox(
                    x_min=self._x_min,
                    x_max=self._x_mid,
                    y_min=self._y_min,
                    y_max=self._y_max,
                    z_min=self._z_min,
                    z_max=self._z_max,
                ),
                "x_max": BoundingBox(
                    x_min=self._x_mid,
                    x_max=self._x_max,
                    y_min=self._y_min,
                    y_max=self._y_max,
                    z_min=self._z_min,
                    z_max=self._z_max,
                ),
            }
        # case where py split required (y is longer than x)
        elif y_full_length:
            bounding_boxes_split_to_halves = {
                "y_min": BoundingBox(
                    x_min=self._x_min,
                    x_max=self._x_max,
                    y_min=self._y_min,
                    y_max=self._y_mid,
                    z_min=self._z_min,
                    z_max=self._z_max,
                ),
                "y_max": BoundingBox(
                    x_min=self._x_min,
                    x_max=self._x_max,
                    y_min=self._y_mid,
                    y_max=self._y_max,
                    z_min=self._z_min,
                    z_max=self._z_max,
                ),
            }
        # case where pz split required (z is longer than x and y)
        else:
            bounding_boxes_split_to_halves = {
                "z_min": BoundingBox(
                    x_min=self._x_min,
                    x_max=self._x_max,
                    y_min=self._y_min,
                    y_max=self._y_max,
                    z_min=self._z_min,
                    z_max=self._z_mid,
                ),
                "z_max": BoundingBox(
                    x_min=self._x_min,
                    x_max=self._x_max,
                    y_min=self._y_min,
                    y_max=self._y_max,
                    z_min=self._z_mid,
                    z_max=self._z_max,
                ),
            }

        # return bounding box halves
        return bounding_boxes_split_to_halves

    def split_to_octants(self) -> Dict[str, "BoundingBox"]:
        """Split the bounding box into eight equal octant bounding boxes.

        Returns
        -------
        dict of {str : BoundingBox}
            Dictionary mapping octant labels to BoundingBox instances.
        """
        bounding_boxes_split_to_octants = {
            "x_min_y_min_z_min": BoundingBox(
                x_min=self._x_min,
                x_max=self._x_mid,
                y_min=self._y_min,
                y_max=self._y_mid,
                z_min=self._z_min,
                z_max=self._z_mid,
            ),
            "x_min_y_max_z_min": BoundingBox(
                x_min=self._x_min,
                x_max=self._x_mid,
                y_min=self._y_mid,
                y_max=self._y_max,
                z_min=self._z_min,
                z_max=self._z_mid,
            ),
            "x_max_y_min_z_min": BoundingBox(
                x_min=self._x_mid,
                x_max=self._x_max,
                y_min=self._y_min,
                y_max=self._y_mid,
                z_min=self._z_min,
                z_max=self._z_mid,
            ),
            "x_max_y_max_z_min": BoundingBox(
                x_min=self._x_mid,
                x_max=self._x_max,
                y_min=self._y_mid,
                y_max=self._y_max,
                z_min=self._z_min,
                z_max=self._z_mid,
            ),
            "x_min_y_min_z_max": BoundingBox(
                x_min=self._x_min,
                x_max=self._x_mid,
                y_min=self._y_min,
                y_max=self._y_mid,
                z_min=self._z_mid,
                z_max=self._z_max,
            ),
            "x_min_y_max_z_max": BoundingBox(
                x_min=self._x_min,
                x_max=self._x_mid,
                y_min=self._y_mid,
                y_max=self._y_max,
                z_min=self._z_mid,
                z_max=self._z_max,
            ),
            "x_max_y_min_z_max": BoundingBox(
                x_min=self._x_mid,
                x_max=self._x_max,
                y_min=self._y_min,
                y_max=self._y_mid,
                z_min=self._z_mid,
                z_max=self._z_max,
            ),
            "x_max_y_max_z_max": BoundingBox(
                x_min=self._x_mid,
                x_max=self._x_max,
                y_min=self._y_mid,
                y_max=self._y_max,
                z_min=self._z_mid,
                z_max=self._z_max,
            ),
        }

        return bounding_boxes_split_to_octants

    def split(self, n_splits: int = 1) -> List["BoundingBox"]:
        """Recursively halve the box ``n_splits`` times.

        Each pass divides every rectangular parallelepiped (RPP) in half along its
        longest axis (see :meth:`split_to_halves`), so ``2 ** n_splits`` boxes
        are returned.

        Parameters
        ----------
        n_splits : int, optional
            Number of halving passes to apply, by default 1.

        Returns
        -------
        list of BoundingBox
            The ``2 ** n_splits`` child bounding boxes.
        """
        bounding_boxes: List[BoundingBox] = [self]
        for split in range(n_splits):
            bounding_box_results: List[BoundingBox] = []
            for bounding_box in bounding_boxes:
                bounding_box_results.extend(
                    bounding_box.split_to_halves().values()
                )
            bounding_boxes = bounding_box_results

        return bounding_boxes


def obbs_overlap(
    centers: np.ndarray,
    axes: np.ndarray,
    half_lengths: np.ndarray,
    other_center: np.ndarray,
    other_axes: np.ndarray,
    other_half_lengths: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    """Test many oriented boxes against one oriented box (separating axis test).

    Implements the 15-axis separating axis theorem test for OBBs (Gottschalk et
    al. 1996; Ericson, *Real-Time Collision Detection*, sec. 4.4.1), vectorised
    over the first box set. An axis-aligned box is passed as an OBB with
    identity axes.

    Parameters
    ----------
    centers : numpy.ndarray
        Centres of the N boxes, shape (N, 3).
    axes : numpy.ndarray
        Unit local axes of the N boxes, shape (N, 3, 3); ``axes[n, i]`` is the
        i-th axis of box n.
    half_lengths : numpy.ndarray
        Half extents of the N boxes along their axes, shape (N, 3).
    other_center : numpy.ndarray
        Centre of the single box, shape (3,).
    other_axes : numpy.ndarray
        Unit local axes of the single box, shape (3, 3) (rows are axes).
    other_half_lengths : numpy.ndarray
        Half extents of the single box, shape (3,).
    eps : float, optional
        Added to the absolute rotation matrix to make near-parallel edge cross
        products robust, by default 1e-9. Errs on the side of reporting overlap.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape (N,), True where box n overlaps the single box.

    Notes
    -----
    The signed rotation matrix is used for the projected centre separation and
    its absolute value only for the projected radii. Touching boxes are treated
    as overlapping, so the test never reports separation for boxes that share a
    point - a requirement when it is used to discard candidate solids.
    """
    centers = np.asarray(centers, dtype=float)
    axes = np.asarray(axes, dtype=float)
    a = np.asarray(half_lengths, dtype=float)
    b = np.asarray(other_half_lengths, dtype=float)
    other_axes = np.asarray(other_axes, dtype=float)

    # rotation matrix expressing the other box in each box's frame: R[n,i,j] = A_i . B_j
    R = axes @ other_axes.T
    abs_R = np.abs(R) + eps

    # translation between centres, in each box's frame
    t_world = np.asarray(other_center, dtype=float)[np.newaxis, :] - centers
    t = (axes @ t_world[:, :, np.newaxis])[:, :, 0]

    # face axes of box set A (L = A_i) and of box B (L = B_j)
    overlap = np.all(np.abs(t) <= a + abs_R @ b, axis=1)
    proj = (t[:, np.newaxis, :] @ R)[:, 0, :]
    ra = (a[:, np.newaxis, :] @ abs_R)[:, 0, :]
    overlap &= np.all(np.abs(proj) <= ra + b, axis=1)

    # edge cross product axes L = A_i x B_j (all nine at once), only for boxes
    # not already separated by a face axis
    keep = np.nonzero(overlap)[0]
    if keep.size:
        i1 = np.array([1, 2, 0])
        i2 = np.array([2, 0, 1])
        Rk = R[keep]
        aRk = abs_R[keep]
        ak = a[keep]
        tk = t[keep]
        ra = ak[:, i1, np.newaxis] * aRk[:, i2, :] + ak[:, i2, np.newaxis] * aRk[:, i1, :]
        rb = b[i1][np.newaxis, np.newaxis, :] * aRk[:, :, i2] + b[i2][np.newaxis, np.newaxis, :] * aRk[:, :, i1]
        sep = np.abs(tk[:, i2, np.newaxis] * Rk[:, i1, :] - tk[:, i1, np.newaxis] * Rk[:, i2, :])
        overlap[keep] = np.all(sep <= ra + rb, axis=(1, 2))

    return overlap


class OrientedBoundingBox:
    """Oriented 3D bounding box (OBB) defined by a centre, axes and half-lengths.

    Instances are built from OpenCASCADE ``Bnd_OBB`` objects by
    ``ocpboundingbox.oriented_bounding_box_from_shape``.

    Parameters
    ----------
    center : tuple of float
        The (x, y, z) center coordinates of the OBB.
    axes : tuple of tuple of float
        Three orthogonal vectors representing local direction axes (U0, U1, U2).
    half_lengths : tuple of float
        The half-extents (extents from center to face) along each local axis.
    """

    def __init__(
        self,
        center: Tuple[float, float, float],
        axes: Tuple[
            Tuple[float, float, float],
            Tuple[float, float, float],
            Tuple[float, float, float],
        ],
        half_lengths: Tuple[float, float, float],
    ) -> None:
        """Initialise the OBB, normalising each axis to unit length."""
        self._center: np.ndarray = np.array(center, dtype=float)
        # Normalize axes to ensure mathematical consistency
        self._axes: np.ndarray = np.array(
            [np.array(a) / np.linalg.norm(a) for a in axes], dtype=float
        )
        self._half_lengths: np.ndarray = np.array(half_lengths, dtype=float)

    @property
    def center(self) -> np.ndarray:
        """numpy.ndarray: Center coordinates vector (x, y, z)."""
        return self._center

    @property
    def axes(self) -> np.ndarray:
        """numpy.ndarray: 3x3 array containing normalized local principal axes."""
        return self._axes

    @property
    def half_lengths(self) -> np.ndarray:
        """numpy.ndarray: Vector of half-extents along each local axis."""
        return self._half_lengths

    @property
    def volume(self) -> float:
        """float: Geometric volume of the oriented bounding box."""
        return (
            8.0
            * float(self._half_lengths[0])
            * float(self._half_lengths[1])
            * float(self._half_lengths[2])
        )

    def __mul__(self, multiplier: float) -> "OrientedBoundingBox":
        """Scale the bounding box half-lengths uniformly about its centre.

        Parameters
        ----------
        multiplier : float
            Uniform scaling factor.

        Returns
        -------
        OrientedBoundingBox
            Scaled OrientedBoundingBox instance.
        """
        return OrientedBoundingBox(
            center=tuple(self._center),
            axes=tuple(tuple(a) for a in self._axes),
            half_lengths=tuple(self._half_lengths * multiplier),
        )

    def __xor__(self, other: Union[BoundingBox, "OrientedBoundingBox"]) -> bool:
        """Check if this OBB intersects another OBB or an AABB.

        Uses the Separating Axis Theorem (SAT) with 15 candidate axes (3 from each
        box and 9 edge cross products). An AABB is converted to an unrotated OBB
        first. Syntax: ``obb_1 ^ obb_2`` or ``obb_1 ^ aabb_1``.

        Parameters
        ----------
        other : BoundingBox or OrientedBoundingBox
            Target bounding box object to check for intersection.

        Returns
        -------
        bool
            True if the boxes intersect, False otherwise.

        Raises
        ------
        TypeError
            If other is not a BoundingBox or OrientedBoundingBox instance.

        Notes
        -----
        The test is conservative: boxes that only touch, or are separated by
        less than a small epsilon on an edge-edge axis, are reported as
        intersecting. See :func:`obbs_overlap`.
        """
        # --- 1. UNIFY INPUT TYPES TO OBB GEOMETRY ---
        if isinstance(other, BoundingBox):
            # Convert AABB into an unrotated OBB structure
            other_center = np.array(other.centre, dtype=float)
            other_axes = np.identity(3, dtype=float)
            other_half_lengths = (
                np.array(
                    [other.x_length, other.y_length, other.z_length],
                    dtype=float,
                )
                / 2.0
            )

        elif isinstance(other, OrientedBoundingBox):
            # Pull properties directly from the other OBB
            other_center = other.center
            other_axes = other.axes
            other_half_lengths = other.half_lengths

        else:
            raise TypeError(
                "Can only intersect with BoundingBox or OrientedBoundingBox"
            )

        # --- 2. RUN THE 15 SEPARATING AXIS TESTS (vectorised helper) ---
        return bool(
            obbs_overlap(
                self._center[np.newaxis, :],
                self._axes[np.newaxis, :, :],
                self._half_lengths[np.newaxis, :],
                np.asarray(other_center, dtype=float),
                np.asarray(other_axes, dtype=float),
                np.asarray(other_half_lengths, dtype=float),
            )[0]
        )

    def __str__(self) -> str:
        """Return a multi-line description of the OBB centre, half-lengths and axes.

        Returns
        -------
        str
            Human-readable OBB summary.
        """
        return (
            f"OBB Center: {list(self._center)}\n"
            f"Half-lengths: {list(self._half_lengths)}\n"
            f"Axes:\n {self._axes}"
        )
