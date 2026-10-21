"""Invert the cells of an MCNP deck into OCP solids.

Each cell (after expanding fills, see :mod:`mcio.mcnpgeometry`) is converted
cell by cell:

1. a bounding box for the cell is found with the half space box classifier,
   by shrinking each side of a world box while the slab it removes is proven
   to be outside the cell;
2. the CSG tree is built into an OCP solid inside that box. Every node is
   classified against the box first, so only surfaces that cross the box are
   turned into half space solids;
3. the solid is checked against the CSG at random points. If the check fails
   the cell is rebuilt with other boolean strategies and the best is kept.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from OCP.BRep import BRep_Builder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.TopAbs import TopAbs_SOLID, TopAbs_IN
from OCP.TopoDS import TopoDS_Compound
from OCP.gp import gp_Pnt

from mcio import logger

from .decomposeparams import DecompositionParameters
from .decomposevoidclassify import BoxClassifier, OUTSIDE, INSIDE
from .geometric import BoundingBox
from .mcnpgeometry import LeafCell
from .ocpbooleans import _common_ocp_solid, _fuse_ocp_solid
from .ocpboundingbox import ocp_box_from_bounding_box
from .ocpfilereadwrite import extract_ocp_shape_type_from_shape
from .ocpproperties import _volume_of_ocp_solids
from .volume import And, Or, HalfVolume, Volume

FULL = 'full'
EMPTY = 'empty'
STRATEGIES = ('mcio', 'legacy', 'pieces')


@dataclass
class InversionResult:
    """Result of inverting one cell."""
    leaf: LeafCell
    solid: object | None
    bounding_box: BoundingBox | None
    seconds: float
    error: str | None = None
    volume: float | None = None
    wrong: int | None = None
    strategy: str | None = None


def _solids(shape) -> list:
    return extract_ocp_shape_type_from_shape(shape, TopAbs_SOLID, cast_from_shape=True)


def _compound(solids: list):
    if len(solids) == 1:
        return solids[0]
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    for solid in solids:
        builder.Add(compound, solid)
    return compound


def cell_bounding_box(volume, world: BoundingBox, classifier: BoxClassifier | None = None,
                      iterations: int = 3, steps: int = 32, pad: float = 1e-3) -> BoundingBox | None:
    """Shrink the world box to a box bounding the cell, None if the cell is proven empty."""
    classifier = classifier or BoxClassifier()
    lo = np.array(world.minimums, dtype=float)
    hi = np.array(world.maximums, dtype=float)

    if classifier.classify(volume, lo, hi) == OUTSIDE:
        return None

    for _ in range(iterations):
        changed = False
        for axis in range(3):
            # raise the lower side: largest t where the slab [lo, t] is outside the cell
            a, b = lo[axis], hi[axis]
            for _ in range(steps):
                t = 0.5 * (a + b)
                slab_hi = hi.copy()
                slab_hi[axis] = t
                if classifier.classify(volume, lo, slab_hi) == OUTSIDE:
                    a = t
                else:
                    b = t
            if a > lo[axis]:
                lo[axis] = a
                changed = True
            # lower the upper side
            a, b = lo[axis], hi[axis]
            for _ in range(steps):
                t = 0.5 * (a + b)
                slab_lo = lo.copy()
                slab_lo[axis] = t
                if classifier.classify(volume, slab_lo, hi) == OUTSIDE:
                    b = t
                else:
                    a = t
            if b < hi[axis]:
                hi[axis] = b
                changed = True
        if not changed:
            break

    lo -= pad
    hi += pad
    return BoundingBox(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])


def _disjoint_pieces(shapes: list):
    """Union as a compound of disjoint solids: each shape minus all the shapes before it."""
    pieces = []
    for i, shape in enumerate(shapes):
        piece = shape
        for previous in shapes[:i]:
            piece = BRepAlgoAPI_Cut(piece, previous).Shape()
        pieces.extend(_solids(piece))
    return _compound(pieces)


def _union(shapes: list, decom_params: DecompositionParameters, strategy: str):
    """Union of shapes using the boolean strategy given."""
    if strategy == 'pieces':
        return _disjoint_pieces(shapes)
    result = shapes[0]
    for other in shapes[1:]:
        if strategy == 'mcio':
            result = _fuse_ocp_solid(result, other, decom_params)
        else:
            result = BRepAlgoAPI_Fuse(result, other).Shape()
    return result


def _common(a, b, decom_params: DecompositionParameters, strategy: str):
    """Common of two shapes using the boolean strategy given, EMPTY if nothing is left."""
    if strategy == 'mcio':
        solids = [r for sa in _solids(a) for sb in _solids(b) for r in _common_ocp_solid(sa, sb, decom_params).solids]
    else:
        solids = _solids(BRepAlgoAPI_Common(a, b).Shape())
    return _compound(solids) if solids else EMPTY


def build_solid(node, bounding_box: BoundingBox, decom_params: DecompositionParameters,
                classifier: BoxClassifier, strategy: str = 'mcio'):
    """OCP solid of a CSG node within a box, or FULL (the whole box) or EMPTY."""
    lo, hi = np.array(bounding_box.minimums), np.array(bounding_box.maximums)
    if isinstance(node, Volume):
        node = node.boolean
    status = classifier.classify(node, lo, hi)
    if status == INSIDE:
        return FULL
    if status == OUTSIDE:
        return EMPTY

    if isinstance(node, HalfVolume):
        try:
            return node.ocp_solid(bounding_box, decom_params)
        except Exception as error:
            # the surface may not cross the box after all: decide from points in the box,
            # but only if they are all on one side
            rng = np.random.default_rng(0)
            corners = [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
            points = np.vstack([rng.uniform(lo, hi, (2000, 3)), corners])
            region = node.openmc_region
            inside = [tuple(p) in region for p in points]
            if all(inside):
                return FULL
            if not any(inside):
                return EMPTY
            raise RuntimeError(f'{type(node.surface).__name__} half space failed: {error}') from error

    children = [build_solid(sub, bounding_box, decom_params, classifier, strategy) for sub in node.sub_booleans]

    if isinstance(node, And):
        if any(c is EMPTY for c in children):
            return EMPTY
        solids = [c for c in children if c is not FULL]
        if not solids:
            return FULL
        # intersect the smallest first
        solids.sort(key=lambda shape: _volume_of_ocp_solids(_solids(shape)))
        result = solids[0]
        for other in solids[1:]:
            result = _common(result, other, decom_params, strategy)
            if result is EMPTY:
                return EMPTY
        return result

    if isinstance(node, Or):
        if any(c is FULL for c in children):
            return FULL
        solids = [c for c in children if c is not EMPTY]
        if not solids:
            return EMPTY
        return _union(solids, decom_params, strategy)

    raise TypeError(f'unknown CSG node {type(node)}')


def count_wrong_points(solid, volume, bounding_box: BoundingBox, n_points: int = 300,
                       expand: float = 0.0, seed: int = 0) -> tuple[int, int, int]:
    """Compare a solid with a CSG volume at random points in (and around) a box.

    Returns
    -------
    tuple[int, int, int]
        Points checked, points where the solid and the CSG disagree, and points
        inside the CSG volume but outside the box.
    """
    rng = np.random.default_rng(seed)
    lo, hi = np.array(bounding_box.minimums), np.array(bounding_box.maximums)
    size = np.maximum(hi - lo, 1e-6)
    points = rng.uniform(lo - expand * size, hi + expand * size, (n_points, 3))
    region = volume.openmc_region
    classifiers = [BRepClass3d_SolidClassifier(s) for s in (_solids(solid) if solid is not None else [])]
    wrong = outside_box = 0
    for p in points:
        in_region = tuple(p) in region
        if in_region and not (np.all(p >= lo) and np.all(p <= hi)):
            outside_box += 1
        in_solid = False
        for classifier in classifiers:
            classifier.Perform(gp_Pnt(*p), 1e-7)
            if classifier.State() == TopAbs_IN:
                in_solid = True
                break
        wrong += in_solid != in_region
    return len(points), wrong, outside_box


def invert_cell(leaf: LeafCell, world: BoundingBox, decom_params: DecompositionParameters,
                classifier: BoxClassifier | None = None, check_points: int = 300) -> InversionResult:
    """Invert one placed cell into an OCP solid, checking the result at random points."""
    classifier = classifier or BoxClassifier()
    start = time.time()
    try:
        bounding_box = cell_bounding_box(leaf.volume, world, classifier)
        if bounding_box is None:
            return InversionResult(leaf, None, None, time.time() - start, 'empty')
        best = None
        for strategy in STRATEGIES:
            try:
                solid = build_solid(leaf.volume, bounding_box, decom_params, classifier, strategy)
            except Exception as error:  # noqa: BLE001 - try the next strategy
                last_error = f'{type(error).__name__}: {error}'.splitlines()[0][:300]
                continue
            if solid is EMPTY:
                return InversionResult(leaf, None, bounding_box, time.time() - start, 'empty')
            if solid is FULL:
                solid = ocp_box_from_bounding_box(bounding_box)
            wrong = count_wrong_points(solid, leaf.volume, bounding_box, check_points)[1] if check_points else 0
            if best is None or wrong < best[0]:
                best = (wrong, solid, strategy)
            if wrong <= max(1, check_points // 300):
                break
        if best is None:
            return InversionResult(leaf, None, bounding_box, time.time() - start, last_error)
        wrong, solid, strategy = best
        return InversionResult(leaf, solid, bounding_box, time.time() - start, None,
                               _volume_of_ocp_solids(_solids(solid)), wrong, strategy)
    except Exception as error:  # noqa: BLE001 - report every failure
        message = f'{type(error).__name__}: {error}'.splitlines()[0][:300]
        return InversionResult(leaf, None, None, time.time() - start, message)


def invert_deck(deck_path: str, step_path: str, world: BoundingBox | None = None,
                cells: list[int] | None = None, include_void: bool = False,
                decom_params: DecompositionParameters | None = None,
                check_points: int = 300) -> list[InversionResult]:
    """Invert an MCNP deck (or some of its cells) and write the solids to a STEP file.

    Solids are named ``cell_<path>_m<material>`` where the path lists the cell
    numbers through any fills, and coloured by material.

    Parameters
    ----------
    deck_path : str
        Path of the MCNP input deck.
    step_path : str
        Path of the STEP file to write.
    world : BoundingBox, optional
        Box containing the whole model, by default +/- 20000 in each direction.
    cells : list[int], optional
        Only invert placed cells whose path contains one of these cell
        numbers, by default all cells.
    include_void : bool, optional
        Also invert cells with no material, by default False.
    decom_params : DecompositionParameters, optional
        Parameters for the OCP booleans, by default the defaults with
        ``bounding_box_multiplier_half_space`` set to 1.5.
    check_points : int, optional
        Number of random points used to check each solid, 0 to skip, by
        default 300.

    Returns
    -------
    list[InversionResult]
        One result per placed cell inverted, including failures.
    """
    import colorsys
    from .mcnpgeometry import MCNPGeometry
    from .ocpfilereadwrite import STEPFileWriter

    world = world or BoundingBox(-20000, 20000, -20000, 20000, -20000, 20000)
    if decom_params is None:
        decom_params = DecompositionParameters()
        decom_params.bounding_box_multiplier_half_space = 1.5

    leaves, errors = MCNPGeometry(deck_path).leaf_cells()
    for path, message in errors.items():
        logger.warning(f'cell {path}: {message}')

    if cells is not None:
        wanted = set(cells)
        leaves = [leaf for leaf in leaves if wanted.intersection(leaf.path)]
    if not include_void:
        leaves = [leaf for leaf in leaves if leaf.material]

    classifier = BoxClassifier()
    writer = STEPFileWriter()
    results = []
    for n, leaf in enumerate(leaves):
        result = invert_cell(leaf, world, decom_params, classifier, check_points)
        results.append(result)
        name = 'cell_' + '_'.join(str(c) for c in leaf.path) + f'_m{leaf.material}'
        if result.error and result.error != 'empty':
            logger.warning(f'{name}: {result.error}')
        elif result.solid is not None:
            hue = (leaf.material * 0.61803398875) % 1.0
            rgb = tuple(round(255 * c) for c in colorsys.hsv_to_rgb(hue, 0.65, 0.9)) if leaf.material else (200, 200, 200)
            writer.add_solid(result.solid, name, rgb=rgb)
            if check_points and result.wrong:
                logger.warning(f'{name}: {result.wrong} of {check_points} check points disagree with the CSG')
        logger.info(f'[{n + 1}/{len(leaves)}] {name} {result.seconds:.1f}s')

    writer.write(step_path)
    return results
