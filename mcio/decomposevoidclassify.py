"""Analytic classification of CSG half spaces against axis-aligned boxes.

Used by the void octree to decide, for an octant box ``B`` and a split solid
``S = H_1 & H_2 & ... & H_n`` (an intersection of half spaces), whether

- ``B`` lies entirely inside a half space ``H_i`` (the half space can be dropped
  from the void cell of that octant, as ``B & ~S == B & ~(S without H_i)``),
- ``B`` lies entirely outside a half space ``H_i`` (``S`` cannot intersect ``B``,
  so the solid can be dropped from the octant altogether), or
- neither can be proven (the half space is kept).

Every decision is a proof: a half space is only reported as inside/outside when
the surface function is bounded strictly away from zero over the whole box, so
pruning never changes the region a void cell describes. Where a tight bound is
not available (e.g. transformed surfaces, one-sheet cones, macrobodies) the half
space is reported as crossing, which is always safe.

The surface function ``f`` follows the MCNP/OpenMC sense convention: the
positive half space is ``f > 0`` and the negative half space is ``f < 0``.
"""

from __future__ import annotations

import math

import numpy as np
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from .surface import (
    Plane,
    XPlane,
    YPlane,
    ZPlane,
    Sphere,
    SphereAtOrigin,
    SphereOnXAxis,
    SphereOnYAxis,
    SphereOnZAxis,
    CylinderParallelToXAxis,
    CylinderParallelToYAxis,
    CylinderParallelToZAxis,
    CylinderOnXAxis,
    CylinderOnYAxis,
    CylinderOnZAxis,
    ConeParallelToXAxis,
    ConeParallelToYAxis,
    ConeParallelToZAxis,
    ConeOnXAxis,
    ConeOnYAxis,
    ConeOnZAxis,
    GeneralQuadratic,
    ToriParralleltoXAxis,
    ToriParralleltoYAxis,
    ToriParralleltoZAxis,
)
from .volume import And, Boolean, HalfVolume, Or, Volume

if TYPE_CHECKING:
    from .surface import Surface

#: the box lies entirely inside the region
INSIDE: int = 1
#: the box lies entirely outside the region
OUTSIDE: int = -1
#: the box may cross the region boundary (or no proof was found)
CROSSING: int = 0

#: lower/upper bound of a surface function over a box
Range = tuple[float, float]
#: function returning the range of a surface function over a box (lo, hi)
RangeFunction = Callable[[Sequence[float], Sequence[float]], "Range | None"]

# relative tolerance applied to bounds before a decision is made
_REL_TOL: float = 1e-10


def _sq_range(lo: float, hi: float, c: float) -> Range:
    """Return the exact range of ``(u - c)**2`` for ``u`` in ``[lo, hi]``.

    Parameters
    ----------
    lo : float
        Lower bound of ``u``.
    hi : float
        Upper bound of ``u``.
    c : float
        Centre of the square.

    Returns
    -------
    tuple[float, float]
        Minimum and maximum of the function over the interval.
    """
    a = lo - c
    b = hi - c
    a2 = a * a
    b2 = b * b
    vmax = a2 if a2 > b2 else b2
    if a <= 0.0 <= b:
        return 0.0, vmax
    return (a2 if a2 < b2 else b2), vmax


def _quad_1d_range(lo: float, hi: float, alpha: float, beta: float) -> Range:
    """Return the exact range of ``alpha*u**2 + beta*u`` for ``u`` in ``[lo, hi]``.

    Parameters
    ----------
    lo : float
        Lower bound of ``u``.
    hi : float
        Upper bound of ``u``.
    alpha : float
        Quadratic coefficient.
    beta : float
        Linear coefficient.

    Returns
    -------
    tuple[float, float]
        Minimum and maximum of the function over the interval.
    """
    v_lo = alpha * lo * lo + beta * lo
    v_hi = alpha * hi * hi + beta * hi
    vmin = min(v_lo, v_hi)
    vmax = max(v_lo, v_hi)
    if alpha != 0.0:
        u = -beta / (2.0 * alpha)
        if lo < u < hi:
            v = alpha * u * u + beta * u
            vmin = min(vmin, v)
            vmax = max(vmax, v)
    return vmin, vmax


def _plane_range(a: float, b: float, c: float, d: float) -> RangeFunction:
    """Build the exact range function of ``a*x + b*y + c*z - d``.

    Parameters
    ----------
    a : float
        Coefficient of x.
    b : float
        Coefficient of y.
    c : float
        Coefficient of z.
    d : float
        Constant.

    Returns
    -------
    RangeFunction
        Function of the box returning the exact range.
    """

    def f(lo: Sequence[float], hi: Sequence[float]) -> Range:
        """Return the exact range of the plane function over the box.

        Parameters
        ----------
        lo : Sequence[float]
            Box minimum corner.
        hi : Sequence[float]
            Box maximum corner.

        Returns
        -------
        tuple[float, float]
            Minimum and maximum of the plane function.
        """
        vmin = -d
        vmax = -d
        for coef, l, h in ((a, lo[0], hi[0]), (b, lo[1], hi[1]), (c, lo[2], hi[2])):
            if coef >= 0.0:
                vmin += coef * l
                vmax += coef * h
            else:
                vmin += coef * h
                vmax += coef * l
        return vmin, vmax

    return f


def _separable_range(
    weights: tuple[float, float, float],
    centre: tuple[float, float, float],
    constant: float,
) -> RangeFunction:
    """Build the exact range function of ``sum_i w_i*(u_i - c_i)**2 + constant``.

    Covers spheres, axis-parallel cylinders and axis-parallel two-sheet cones. A
    weight of zero removes that coordinate.

    Parameters
    ----------
    weights : tuple[float, float, float]
        Weight of the squared term for x, y and z.
    centre : tuple[float, float, float]
        Centre coordinate for x, y and z.
    constant : float
        Constant term.

    Returns
    -------
    RangeFunction
        Function of the box returning the exact range.
    """

    def f(lo: Sequence[float], hi: Sequence[float]) -> Range:
        """Return the exact range of the separable quadric over the box.

        Parameters
        ----------
        lo : Sequence[float]
            Box minimum corner.
        hi : Sequence[float]
            Box maximum corner.

        Returns
        -------
        tuple[float, float]
            Minimum and maximum of the function.
        """
        vmin = constant
        vmax = constant
        for i in range(3):
            w = weights[i]
            if w == 0.0:
                continue
            smin, smax = _sq_range(lo[i], hi[i], centre[i])
            if w > 0.0:
                vmin += w * smin
                vmax += w * smax
            else:
                vmin += w * smax
                vmax += w * smin
        return vmin, vmax

    return f


def _torus_range(
    axis: int,
    centre: tuple[float, float, float],
    a: float,
    b: float,
    c: float,
) -> RangeFunction:
    """Build the exact range function of an axis-parallel (elliptical) torus.

    The MCNP torus function is ``(s - s0)**2 / b**2 + (rho - a)**2 / c**2 - 1``
    where ``s`` is the coordinate along the axis and ``rho`` the distance from
    the axis. Over a box, ``s`` and ``rho`` vary independently over intervals, so
    the bound is exact.

    Parameters
    ----------
    axis : int
        Axis index (0 = x, 1 = y, 2 = z).
    centre : tuple[float, float, float]
        Torus centre.
    a : float
        Major radius.
    b : float
        Semi-axis of the tube parallel to the torus axis.
    c : float
        Semi-axis of the tube perpendicular to the torus axis.

    Returns
    -------
    RangeFunction
        Function of the box returning the exact range.
    """
    p, q = [i for i in range(3) if i != axis]

    def f(lo: Sequence[float], hi: Sequence[float]) -> Range:
        """Return the exact range of the torus function over the box.

        Parameters
        ----------
        lo : Sequence[float]
            Box minimum corner.
        hi : Sequence[float]
            Box maximum corner.

        Returns
        -------
        tuple[float, float]
            Minimum and maximum of the function.
        """
        ps_min, ps_max = _sq_range(lo[p], hi[p], centre[p])
        qs_min, qs_max = _sq_range(lo[q], hi[q], centre[q])
        rho_min = math.sqrt(ps_min + qs_min)
        rho_max = math.sqrt(ps_max + qs_max)
        r_min, r_max = _sq_range(rho_min, rho_max, a)
        s_min, s_max = _sq_range(lo[axis], hi[axis], centre[axis])
        inv_b2 = 1.0 / (b * b)
        inv_c2 = 1.0 / (c * c)
        return (
            s_min * inv_b2 + r_min * inv_c2 - 1.0,
            s_max * inv_b2 + r_max * inv_c2 - 1.0,
        )

    return f


def _general_quadric_range(
    a: float, b: float, c: float, d: float, e: float,
    f_: float, g: float, h: float, j: float, k: float,
) -> RangeFunction:
    """Build a range function for ``ax2+by2+cz2+dxy+eyz+fzx+gx+hy+jz+k``.

    The lower bound uses the centred (mean value) form about the box centre,
    which is exact for linear terms and tight to second order in the box size.
    When the quadratic form is positive semi-definite (cylinders, ellipsoids)
    the function is convex, so the exact maximum is the largest corner value;
    otherwise the centred-form upper bound is used.

    Parameters
    ----------
    a, b, c : float
        Coefficients of x**2, y**2 and z**2.
    d, e : float
        Coefficients of xy and yz.
    f_ : float
        Coefficient of zx.
    g, h, j : float
        Coefficients of x, y and z.
    k : float
        Constant.

    Returns
    -------
    RangeFunction
        Function of the box returning a (conservative) range.
    """
    Q = ((a, d / 2.0, f_ / 2.0), (d / 2.0, b, e / 2.0), (f_ / 2.0, e / 2.0, c))
    L = (g, h, j)

    # positive semi-definite quadratic form => convex function => max at a corner
    scale = max(abs(a), abs(b), abs(c), abs(d), abs(e), abs(f_), 1e-300)
    is_psd = bool(np.linalg.eigvalsh(np.array(Q)).min() >= -1e-12 * scale)

    def value(x: float, y: float, z: float) -> float:
        """Return the quadric function value at a point.

        Parameters
        ----------
        x, y, z : float
            Point coordinates.

        Returns
        -------
        float
            Function value.
        """
        return (
            a * x * x + b * y * y + c * z * z + d * x * y + e * y * z + f_ * z * x
            + g * x + h * y + j * z + k
        )

    def f(lo: Sequence[float], hi: Sequence[float]) -> Range:
        """Return a conservative range of the quadric over the box.

        Parameters
        ----------
        lo : Sequence[float]
            Box minimum corner.
        hi : Sequence[float]
            Box maximum corner.

        Returns
        -------
        tuple[float, float]
            Lower and upper bounds of the function.
        """
        m = [(lo[i] + hi[i]) * 0.5 for i in range(3)]
        r = [(hi[i] - lo[i]) * 0.5 for i in range(3)]
        f0 = value(m[0], m[1], m[2])
        # gradient at the centre: 2 Q m + L
        grad = [2.0 * (Q[i][0] * m[0] + Q[i][1] * m[1] + Q[i][2] * m[2]) + L[i] for i in range(3)]
        lin = abs(grad[0]) * r[0] + abs(grad[1]) * r[1] + abs(grad[2]) * r[2]
        # quadratic remainder t^T Q t for |t_i| <= r_i
        q_min = 0.0
        q_max = 0.0
        for i in range(3):
            qii = Q[i][i] * r[i] * r[i]
            if qii > 0.0:
                q_max += qii
            else:
                q_min += qii
            for jj in range(i + 1, 3):
                off = 2.0 * abs(Q[i][jj]) * r[i] * r[jj]
                q_min -= off
                q_max += off
        vmin = f0 - lin + q_min
        vmax = f0 + lin + q_max
        if is_psd:
            corner_max = max(
                value(x, y, z)
                for x in (lo[0], hi[0])
                for y in (lo[1], hi[1])
                for z in (lo[2], hi[2])
            )
            vmax = min(vmax, corner_max)
        return vmin, vmax

    return f


def surface_range_function(surface: Surface) -> RangeFunction | None:
    """Return a function bounding a surface's value over a box, if supported.

    Parameters
    ----------
    surface : Surface
        mcio surface.

    Returns
    -------
    RangeFunction or None
        Function ``(lo, hi) -> (fmin, fmax)``, or None if the surface type (or a
        transform on it) is not supported.
    """
    if getattr(surface, "transform", None) is not None:
        return None

    s = surface
    if isinstance(s, XPlane):
        return _plane_range(1.0, 0.0, 0.0, s.x0)
    if isinstance(s, YPlane):
        return _plane_range(0.0, 1.0, 0.0, s.y0)
    if isinstance(s, ZPlane):
        return _plane_range(0.0, 0.0, 1.0, s.z0)
    if type(s) is Plane:
        return _plane_range(s.a, s.b, s.c, s.d)
    if isinstance(s, Sphere):
        return _separable_range((1.0, 1.0, 1.0), (s.x0, s.y0, s.z0), -s.radius ** 2)
    if isinstance(s, SphereAtOrigin):
        return _separable_range((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), -s.radius ** 2)
    if isinstance(s, SphereOnXAxis):
        return _separable_range((1.0, 1.0, 1.0), (s.x0, 0.0, 0.0), -s.radius ** 2)
    if isinstance(s, SphereOnYAxis):
        return _separable_range((1.0, 1.0, 1.0), (0.0, s.y0, 0.0), -s.radius ** 2)
    if isinstance(s, SphereOnZAxis):
        return _separable_range((1.0, 1.0, 1.0), (0.0, 0.0, s.z0), -s.radius ** 2)
    if isinstance(s, CylinderParallelToXAxis):
        return _separable_range((0.0, 1.0, 1.0), (0.0, s.y0, s.z0), -s.radius ** 2)
    if isinstance(s, CylinderParallelToYAxis):
        return _separable_range((1.0, 0.0, 1.0), (s.x0, 0.0, s.z0), -s.radius ** 2)
    if isinstance(s, CylinderParallelToZAxis):
        return _separable_range((1.0, 1.0, 0.0), (s.x0, s.y0, 0.0), -s.radius ** 2)
    if isinstance(s, CylinderOnXAxis):
        return _separable_range((0.0, 1.0, 1.0), (0.0, 0.0, 0.0), -s.radius ** 2)
    if isinstance(s, CylinderOnYAxis):
        return _separable_range((1.0, 0.0, 1.0), (0.0, 0.0, 0.0), -s.radius ** 2)
    if isinstance(s, CylinderOnZAxis):
        return _separable_range((1.0, 1.0, 0.0), (0.0, 0.0, 0.0), -s.radius ** 2)
    # two-sheet cones only: the sense of one-sheet cones is not a single quadric
    if isinstance(s, (ConeParallelToXAxis, ConeOnXAxis)) and s.sheet == 0:
        x0 = getattr(s, "x0", 0.0); y0 = getattr(s, "y0", 0.0); z0 = getattr(s, "z0", 0.0)
        return _separable_range((-s.t_sq, 1.0, 1.0), (x0, y0, z0), 0.0)
    if isinstance(s, (ConeParallelToYAxis, ConeOnYAxis)) and s.sheet == 0:
        x0 = getattr(s, "x0", 0.0); y0 = getattr(s, "y0", 0.0); z0 = getattr(s, "z0", 0.0)
        return _separable_range((1.0, -s.t_sq, 1.0), (x0, y0, z0), 0.0)
    if isinstance(s, (ConeParallelToZAxis, ConeOnZAxis)) and s.sheet == 0:
        x0 = getattr(s, "x0", 0.0); y0 = getattr(s, "y0", 0.0); z0 = getattr(s, "z0", 0.0)
        return _separable_range((1.0, 1.0, -s.t_sq), (x0, y0, z0), 0.0)
    if isinstance(s, GeneralQuadratic):
        return _general_quadric_range(s.a, s.b, s.c, s.d, s.e, s.f, s.g, s.h, s.j, s.k)
    if isinstance(s, ToriParralleltoXAxis):
        return _torus_range(0, (s.x0, s.y0, s.z0), s.a, s.b, s.c)
    if isinstance(s, ToriParralleltoYAxis):
        return _torus_range(1, (s.x0, s.y0, s.z0), s.a, s.b, s.c)
    if isinstance(s, ToriParralleltoZAxis):
        return _torus_range(2, (s.x0, s.y0, s.z0), s.a, s.b, s.c)
    return None


class BoxClassifier:
    """Classify CSG regions against axis-aligned boxes, caching per surface.

    Notes
    -----
    The surface range functions are cached by ``id(surface)``. The cache also
    holds a reference to each surface, so an id cannot be reused by a new
    surface while its entry exists. The cache is not pickled.
    """

    def __init__(self) -> None:
        """Initialise an empty range function cache."""
        self._cache: dict[int, tuple[Any, RangeFunction | None]] = {}

    def __getstate__(self) -> dict[str, Any]:
        """Return the picklable state (the cache is dropped).

        Returns
        -------
        dict[str, Any]
            Empty state.
        """
        return {}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from a pickled state with an empty cache.

        Parameters
        ----------
        state : dict[str, Any]
            Pickled state (ignored).
        """
        self._cache = {}

    def _range_function(self, surface: Surface) -> RangeFunction | None:
        """Return the cached range function of a surface.

        Parameters
        ----------
        surface : Surface
            mcio surface.

        Returns
        -------
        RangeFunction or None
            Range function, or None if the surface is not supported.
        """
        entry = self._cache.get(id(surface))
        if entry is not None and entry[0] is surface:
            return entry[1]
        fn = surface_range_function(surface)
        self._cache[id(surface)] = (surface, fn)
        return fn

    def classify(
        self, region: HalfVolume | Volume | Boolean, lo: Sequence[float], hi: Sequence[float]
    ) -> int:
        """Classify a region against a box.

        Parameters
        ----------
        region : HalfVolume, Volume or Boolean
            CSG region.
        lo : Sequence[float]
            Box minimum corner.
        hi : Sequence[float]
            Box maximum corner.

        Returns
        -------
        int
            :data:`INSIDE`, :data:`OUTSIDE` or :data:`CROSSING`.
        """
        if isinstance(region, HalfVolume):
            fn = self._range_function(region.surface)
            if fn is None:
                return CROSSING
            vmin, vmax = fn(lo, hi)
            tol = _REL_TOL * (abs(vmin) + abs(vmax) + 1.0)
            if vmin > tol:
                side = 1
            elif vmax < -tol:
                side = -1
            else:
                return CROSSING
            positive = region._direction == "+"
            return INSIDE if (side == 1) == positive else OUTSIDE
        if isinstance(region, Volume):
            return self.classify(region.boolean, lo, hi)
        if isinstance(region, And):
            all_inside = True
            for sub in region.sub_booleans:
                status = self.classify(sub, lo, hi)
                if status == OUTSIDE:
                    return OUTSIDE
                if status != INSIDE:
                    all_inside = False
            return INSIDE if all_inside else CROSSING
        if isinstance(region, Or):
            all_outside = True
            for sub in region.sub_booleans:
                status = self.classify(sub, lo, hi)
                if status == INSIDE:
                    return INSIDE
                if status != OUTSIDE:
                    all_outside = False
            return OUTSIDE if all_outside else CROSSING
        return CROSSING

    def prune_terms(
        self,
        terms: Sequence[HalfVolume | Volume | Boolean],
        active: Sequence[int],
        lo: Sequence[float],
        hi: Sequence[float],
    ) -> tuple[int, tuple[int, ...]]:
        """Classify an intersection of terms against a box and drop proven terms.

        Parameters
        ----------
        terms : Sequence[HalfVolume | Volume | Boolean]
            Terms of the intersection ``terms[0] & terms[1] & ...``.
        active : Sequence[int]
            Indices of the terms still to be considered (terms already proven to
            contain a parent box are not re-tested).
        lo : Sequence[float]
            Box minimum corner.
        hi : Sequence[float]
            Box maximum corner.

        Returns
        -------
        tuple[int, tuple[int, ...]]
            Status of the whole intersection (:data:`OUTSIDE` if any term
            excludes the box, :data:`INSIDE` if every term contains it, else
            :data:`CROSSING`) and the indices of the terms that are still needed
            to describe the intersection within the box.
        """
        remaining: list[int] = []
        for i in active:
            status = self.classify(terms[i], lo, hi)
            if status == OUTSIDE:
                return OUTSIDE, ()
            if status == CROSSING:
                remaining.append(i)
        if not remaining:
            return INSIDE, ()
        return CROSSING, tuple(remaining)


def intersection_terms(region: HalfVolume | Volume | Boolean) -> list[HalfVolume | Volume | Boolean]:
    """Split a region into the terms of its top-level intersection.

    Parameters
    ----------
    region : HalfVolume, Volume or Boolean
        CSG region.

    Returns
    -------
    list
        ``[t_1, ..., t_n]`` such that the region equals ``t_1 & ... & t_n``.
    """
    if isinstance(region, Volume):
        region = region.boolean
    if isinstance(region, And):
        return list(region.sub_booleans)
    return [region]


def as_region(term: HalfVolume | Volume | Boolean) -> HalfVolume | Volume:
    """Wrap a raw :class:`Boolean` in a :class:`Volume` so CSG operators work.

    Parameters
    ----------
    term : HalfVolume, Volume or Boolean
        CSG term.

    Returns
    -------
    HalfVolume or Volume
        The term, usable with ``&``, ``|`` and ``~``.
    """
    if isinstance(term, Boolean):
        return Volume(term)
    return term


def intersect_terms(terms: Sequence[HalfVolume | Volume | Boolean]) -> HalfVolume | Volume:
    """Return the intersection of a non-empty sequence of terms as a region.

    Parameters
    ----------
    terms : Sequence[HalfVolume | Volume | Boolean]
        Terms to intersect.

    Returns
    -------
    HalfVolume or Volume
        ``terms[0] & terms[1] & ...``.
    """
    region = as_region(terms[0])
    for term in terms[1:]:
        region = region & as_region(term)
    return region


def _plane_coefficients(term: Any) -> tuple[float, float, float, float, str] | None:
    """Return ``(a, b, c, d, direction)`` if a term is a plane half space.

    Parameters
    ----------
    term : Any
        CSG term.

    Returns
    -------
    tuple or None
        Plane ``a*x + b*y + c*z = d`` and the half volume direction ('+'/'-'),
        or None if the term is not an untransformed plane half space.
    """
    if not isinstance(term, HalfVolume):
        return None
    s = term.surface
    if getattr(s, "transform", None) is not None:
        return None
    if isinstance(s, XPlane):
        return 1.0, 0.0, 0.0, s.x0, term._direction
    if isinstance(s, YPlane):
        return 0.0, 1.0, 0.0, s.y0, term._direction
    if isinstance(s, ZPlane):
        return 0.0, 0.0, 1.0, s.z0, term._direction
    if type(s) is Plane:
        return s.a, s.b, s.c, s.d, term._direction
    return None


def box_minus_planar_halfspaces_is_empty(
    terms: Sequence[Any], lo: Sequence[float], hi: Sequence[float], rel_tol: float = 1e-9
) -> bool:
    """Return whether ``box & ~H_1 & ~H_2 & ...`` has no interior.

    Applies only when every ``H_k`` is a plane half space; the region is then a
    convex polytope and its emptiness is decided exactly by a linear program
    that maximises the radius of a ball (in the max norm for the box) inside it.

    Parameters
    ----------
    terms : Sequence[Any]
        Half spaces ``H_k`` subtracted from the box.
    lo : Sequence[float]
        Box minimum corner.
    hi : Sequence[float]
        Box maximum corner.
    rel_tol : float, optional
        Inscribed radius, relative to the box size, below which the region is
        treated as empty, by default 1e-9.

    Returns
    -------
    bool
        True if the region is (numerically) empty; False if it is not, or if a
        term is not a plane half space, or the LP could not be solved.
    """
    rows: list[list[float]] = []
    rhs: list[float] = []
    for term in terms:
        plane = _plane_coefficients(term)
        if plane is None:
            return False
        a, b, c, d, direction = plane
        norm = math.sqrt(a * a + b * b + c * c)
        if norm == 0.0:
            return False
        a, b, c, d = a / norm, b / norm, c / norm, d / norm
        # complement of '+' (f > 0) is n.x < d ; complement of '-' is n.x > d
        if direction == "+":
            rows.append([a, b, c, 1.0]); rhs.append(d)
        else:
            rows.append([-a, -b, -c, 1.0]); rhs.append(-d)
    for i in range(3):
        row = [0.0, 0.0, 0.0, 1.0]
        row[i] = -1.0
        rows.append(row); rhs.append(-lo[i])
        row = [0.0, 0.0, 0.0, 1.0]
        row[i] = 1.0
        rows.append(row); rhs.append(hi[i])

    size = max(hi[i] - lo[i] for i in range(3))
    try:
        from scipy.optimize import linprog
        result = linprog(
            c=[0.0, 0.0, 0.0, -1.0],
            A_ub=np.array(rows),
            b_ub=np.array(rhs),
            bounds=[(None, None)] * 3 + [(None, size)],
            method="highs",
        )
    except Exception:
        return False
    if result.status != 0:
        # infeasible means even a zero radius ball does not fit: empty
        return result.status == 2
    return -result.fun <= rel_tol * size
