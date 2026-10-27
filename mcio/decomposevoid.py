"""Void generation during the solid decomposition process.

Builds a binary space-partitioning octree over the model's bounding box. Each
node (octant) keeps only the split solids that may intersect it and is halved
along its longest side until termination criteria are met. Leaf octants become
void cells: the octant RPP with the remaining split solid volumes subtracted.

Candidate split solids are filtered per octant in order of increasing cost:

1. **AABB** - axis-aligned bounding box overlap (vectorised over all candidates).
2. **OBB** - oriented bounding box overlap, separating axis test (vectorised;
   ``DecompositionParameters.void_use_obb_check``).
3. **Half-space pruning** - each half space of a split solid is classified
   analytically against the octant box
   (``DecompositionParameters.void_use_halfspace_pruning``, see
   :mod:`mcio.decomposevoidclassify`). A half space that excludes the box proves the solid
   does not intersect the octant; half spaces that contain the box are dropped
   from the octant's void cell; if every half space contains the box the octant
   lies inside the solid and needs no void cell at all. Pruning is inherited, so
   a child only re-tests the half spaces its parent could not resolve. When
   every remaining solid is reduced to one plane half space (typically where
   solids touch, e.g. wall slabs), a small linear program decides exactly
   whether the solids cover the octant, in which case it needs no void cell.
4. **Exact OCP check** - optional, only for solids still undecided:

   - :class:`VoidOctreeSplitSolids`: solid-to-box distance
     (``void_use_split_solid_distance_check``, ``void_split_solid_distance_tol``).
   - :class:`VoidOctreeFaces`: face-to-box distance for faces whose AABB overlaps
     the octant (``void_face_distance_tol``), plus a point classification of the
     octant centre when no face is near, which detects octants inside a solid.

Every filter is conservative (it never discards a solid that intersects the
octant), so the void cells never overlap the solids.

The two classes are selected by ``DecompositionParameters.void_filter_by_face``
in ``decomposesolid.generate_mcio_void_octree``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import pickle
import os

import numpy as np

from .cell import Cell
from .surface import XPlane, YPlane, ZPlane, Sphere
from .geometric import BoundingBox, obbs_overlap
from .universe import Universe

try:
    import casadi
    import OCP
    from OCP.TopoDS import TopoDS_Solid, TopoDS
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.BRep import BRep_Tool
    from OCP.TopExp import TopExp_Explorer
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_IN, TopAbs_VERTEX
except Exception as e:
    from mcio import logger

    logger.warning("failed to load OCP")
    logger.warning(f"{e}")

from mcio import logger
from .decomposeparams import DecompositionParameters
from .register import ProgressRegister
from .decomposesplitsolids import SplitSolid
from .ocpboundingbox import (
    ocp_box_from_bounding_box,
)
from .ocpfilereadwrite import STEPFileWriter
from .ocpproperties import (
    _distance_between_shapes,
)
from .decomposevoidclassify import (
    BoxClassifier,
    INSIDE,
    OUTSIDE,
    _plane_coefficients,
    box_minus_planar_halfspaces_is_empty,
    intersection_terms,
    intersect_terms,
)

if TYPE_CHECKING:
    from OCP.TopoDS import TopoDS_Face
    from .volume import HalfVolume, Volume


_IDENTITY: np.ndarray = np.identity(3)


class _OctreeData:
    """Read-only data shared by every node of one void octree.

    Built once at the root so that children refer to split solids by integer
    index, which keeps nodes small and trivially picklable.

    Parameters
    ----------
    split_solids : list of SplitSolid
        All split solids the void must avoid.
    use_obb : bool
        Whether to collect oriented bounding boxes for the OBB filter.

    Attributes
    ----------
    split_solids : list of SplitSolid
        All split solids, indexed by the integers stored in the nodes.
    aabb_lo, aabb_hi : numpy.ndarray
        Axis-aligned bounding box corners of each split solid, shape (N, 3).
    obb_centres, obb_axes, obb_half_lengths : numpy.ndarray or None
        Oriented bounding boxes of each split solid, shapes (N, 3), (N, 3, 3)
        and (N, 3); ``obb_centres`` is None when the OBB filter is disabled or
        no OBB is tighter than its AABB.
    obb_useful : numpy.ndarray or None
        True for split solids whose OBB volume is under 90 % of their AABB
        volume; only these are given the OBB test.
    terms : list of list
        Terms of each split solid's top-level intersection (its half spaces).
    classifier : BoxClassifier
        Half space / box classifier with a per-surface cache.
    """

    def __init__(self, split_solids: List[SplitSolid], use_obb: bool) -> None:
        """Collect bounding boxes and half space terms of the split solids."""
        self.split_solids: List[SplitSolid] = list(split_solids)
        n = len(self.split_solids)

        self.aabb_lo: np.ndarray = np.array(
            [ss.bounding_box.minimums for ss in self.split_solids], dtype=float
        ).reshape(n, 3)
        self.aabb_hi: np.ndarray = np.array(
            [ss.bounding_box.maximums for ss in self.split_solids], dtype=float
        ).reshape(n, 3)

        self.obb_centres: Optional[np.ndarray] = None
        self.obb_axes: Optional[np.ndarray] = None
        self.obb_half_lengths: Optional[np.ndarray] = None
        self.obb_useful = None
        if use_obb and n > 0:
            self.obb_centres = np.array(
                [ss.oriented_bounding_box.center for ss in self.split_solids]
            )
            self.obb_axes = np.array(
                [ss.oriented_bounding_box.axes for ss in self.split_solids]
            )
            self.obb_half_lengths = np.array(
                [ss.oriented_bounding_box.half_lengths for ss in self.split_solids]
            )
            # the OBB test is only worth running where the OBB is clearly
            # tighter than the AABB (e.g. not for axis aligned solids)
            obb_volume = 8.0 * np.prod(self.obb_half_lengths, axis=1)
            aabb_volume = np.prod(self.aabb_hi - self.aabb_lo, axis=1)
            self.obb_useful: Optional[np.ndarray] = obb_volume < 0.9 * aabb_volume
            if not self.obb_useful.any():
                self.obb_centres = None

        self.terms: List[List[Any]] = [
            intersection_terms(ss.volume) for ss in self.split_solids
        ]
        self.classifier: BoxClassifier = BoxClassifier()

        # per split solid face bounding boxes, built lazily for the face filter
        self._face_aabbs: Dict[int, tuple[np.ndarray, np.ndarray]] = {}

        # per split solid convex polytope data (vertices, SAT axes), built lazily
        self._polytopes: Dict[int, Optional[tuple[np.ndarray, np.ndarray]]] = {}

    def polytope(self, index: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Return the convex polytope data of a planar split solid.

        A split solid whose CSG volume is an intersection of plane half spaces
        only is a convex polytope. Its separating axis candidates against an
        axis-aligned box are the plane normals, the three coordinate axes and
        the cross products of each pairwise plane intersection direction with
        the coordinate axes (a superset of the true edge directions, which keeps
        the test exact).

        Parameters
        ----------
        index : int
            Split solid index.

        Returns
        -------
        tuple of numpy.ndarray or None
            Vertices of the solid, shape (V, 3), and unit candidate axes, shape
            (K, 3); None if the solid has a non-planar half space.
        """
        if index in self._polytopes:
            return self._polytopes[index]

        normals: List[np.ndarray] = []
        for term in self.terms[index]:
            plane = _plane_coefficients(term)
            if plane is None:
                self._polytopes[index] = None
                return None
            n = np.array(plane[:3], dtype=float)
            normals.append(n / np.linalg.norm(n))

        vertices: List[tuple[float, float, float]] = []
        explorer = TopExp_Explorer(self.split_solids[index].split_solid, TopAbs_VERTEX)
        while explorer.More():
            p = BRep_Tool.Pnt_s(TopoDS.Vertex_s(explorer.Current()))
            vertices.append((p.X(), p.Y(), p.Z()))
            explorer.Next()
        if not vertices:
            self._polytopes[index] = None
            return None

        normal_array = np.array(normals)
        candidate_axes = [normal_array, np.identity(3)]
        n_planes = len(normals)
        if n_planes > 1:
            i_idx, j_idx = np.triu_indices(n_planes, k=1)
            edges = np.cross(normal_array[i_idx], normal_array[j_idx])
            edges = edges[np.linalg.norm(edges, axis=1) > 1e-12]
            for k in range(3):
                candidate_axes.append(np.cross(edges, np.identity(3)[k]))
        axes = np.vstack(candidate_axes)
        lengths = np.linalg.norm(axes, axis=1)
        axes = axes[lengths > 1e-12] / lengths[lengths > 1e-12, np.newaxis]
        # remove (anti)parallel duplicates
        signs = np.sign(axes[np.arange(len(axes)), np.argmax(np.abs(axes), axis=1)])
        axes = np.unique(np.round(axes * signs[:, np.newaxis], 12), axis=0)

        self._polytopes[index] = (np.array(vertices, dtype=float), axes)
        return self._polytopes[index]

    def face_aabbs(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the face bounding box corners of a split solid.

        Parameters
        ----------
        index : int
            Split solid index.

        Returns
        -------
        tuple of numpy.ndarray
            Minimum and maximum corners of each face, each of shape (n_faces, 3),
            in the order of ``SplitSolid.face_to_mcio_halfspaces``.
        """
        if index not in self._face_aabbs:
            boxes = list(self.split_solids[index].face_bounding_boxes.values())
            lo = np.array([b.minimums for b in boxes], dtype=float).reshape(-1, 3)
            hi = np.array([b.maximums for b in boxes], dtype=float).reshape(-1, 3)
            self._face_aabbs[index] = (lo, hi)
        return self._face_aabbs[index]


class _LazyBox:
    """OCP box solid of a bounding box, built on first use.

    Parameters
    ----------
    bounding_box : BoundingBox
        Box to convert.
    """

    def __init__(self, bounding_box: BoundingBox) -> None:
        """Store the bounding box."""
        self._bounding_box: BoundingBox = bounding_box
        self._shape: Optional["TopoDS_Solid"] = None

    @property
    def shape(self) -> "TopoDS_Solid":
        """TopoDS_Solid: The OCP box solid."""
        if self._shape is None:
            self._shape = ocp_box_from_bounding_box(self._bounding_box)
        return self._shape


class VoidOctreeBase:
    """Base class for the void octrees.

    Holds the shared octree machinery: candidate filtering, recursion, cell
    generation and (de)serialisation. Subclasses choose the exact intersection
    check and the termination rule.

    Attributes
    ----------
    bounding_box : BoundingBox
        Spatial domain of the octree node.
    covered : bool
        True when the node's box lies entirely inside one split solid, so no
        void cell is required for it.
    universe : Universe or None
        MCIO Universe assigned to the void cells.
    decom_params : DecompositionParameters
        Decomposition parameters.
    is_serialised : bool
        Whether the split solids are currently serialised to binaries.
    """

    _children: List["VoidOctreeBase"]

    # ------------------------------------------------------------------ set up
    def _init_root(
        self,
        split_solids: List[SplitSolid],
        decom_params: DecompositionParameters,
        surfaces_generated: Dict[Any, Any],
        bounding_box: Optional[BoundingBox],
        make_square: bool,
        universe: Optional[Universe],
        level: int,
        show_progress: bool,
        title: str,
    ) -> None:
        """Initialise the root node holding every split solid.

        Parameters
        ----------
        split_solids : list of SplitSolid
            Split solids the void must avoid.
        decom_params : DecompositionParameters
            Decomposition parameters.
        surfaces_generated : dict
            Map of surface type to ``{hash string: surface}`` used to share
            surfaces between cells.
        bounding_box : BoundingBox or None
            Root box; computed from the split solids when None.
        make_square : bool
            Make the computed root box a cube.
        universe : Universe or None
            Universe of the generated void cells.
        level : int
            Depth of the root (normally 0).
        show_progress : bool
            Show a progress bar.
        title : str
            Progress bar title.
        """
        self.decom_params: DecompositionParameters = decom_params
        self._make_square: bool = make_square
        self._surfaces_generated: Dict[Any, Any] = surfaces_generated
        self._parent: Optional[VoidOctreeBase] = None
        self._level: int = level
        self.universe: Optional[Universe] = universe
        self._show_progress: bool = show_progress
        self._data: _OctreeData = _OctreeData(
            split_solids, bool(getattr(decom_params, "void_use_obb_check", False))
        )
        self.bounding_box: BoundingBox = (
            bounding_box if bounding_box else self._generate_bounding_box(split_solids)
        )
        self._children = []
        self._mcio_cells: Optional[List[Cell]] = None
        self.is_serialised: bool = False
        self.covered: bool = False

        # every split solid is a candidate of the root, with all of its terms
        self._candidates: Dict[int, tuple[int, ...]] = {
            i: tuple(range(len(terms))) for i, terms in enumerate(self._data.terms)
        }
        self._near_faces: Dict[int, tuple[int, ...]] = {}

        self.register: ProgressRegister = ProgressRegister(
            list(self._candidates), show_progress=self._show_progress, title=title
        )

    def _init_child(self, parent: "VoidOctreeBase", bounding_box: BoundingBox) -> None:
        """Initialise a child node of ``parent`` and filter its candidates.

        Parameters
        ----------
        parent : VoidOctreeBase
            Parent node.
        bounding_box : BoundingBox
            Box of this child.
        """
        self.decom_params = parent.decom_params
        self._make_square = parent._make_square
        self._surfaces_generated = parent._surfaces_generated
        self._parent = parent
        self._level = parent._level + 1
        self.universe = parent.universe
        self._show_progress = parent._show_progress
        self._data = parent._data
        self.bounding_box = bounding_box
        self._children = []
        self._mcio_cells = None
        self.is_serialised = False
        self.covered = False
        self._near_faces = {}

        self._candidates = self._filter_candidates(parent._candidates)

        self.register = parent.register
        self.register.add(self._candidates)

    # ------------------------------------------------------------- filtering
    def _filter_candidates(
        self, parent_candidates: Dict[int, tuple[int, ...]]
    ) -> Dict[int, tuple[int, ...]]:
        """Return the parent's candidates that may intersect this node's box.

        Applies the AABB, OBB, half-space and exact filters described in the
        module docstring and sets :attr:`covered` when the box is proven to lie
        inside a split solid (in which case no candidates are returned).

        Parameters
        ----------
        parent_candidates : dict of {int : tuple of int}
            Parent candidates: split solid index to the indices of its terms not
            yet proven to contain the parent box.

        Returns
        -------
        dict of {int : tuple of int}
            Candidates of this node, with their remaining term indices.
        """
        if not parent_candidates:
            return {}

        data = self._data
        params = self.decom_params
        lo = np.array(self.bounding_box.minimums, dtype=float)
        hi = np.array(self.bounding_box.maximums, dtype=float)

        # filter 1: AABB overlap (strict, so touching boxes do not count)
        idx = np.fromiter(parent_candidates.keys(), dtype=int, count=len(parent_candidates))
        mask = np.all(data.aabb_lo[idx] < hi, axis=1) & np.all(data.aabb_hi[idx] > lo, axis=1)
        idx = idx[mask]

        # filter 2: OBB overlap (separating axis test against the octant box),
        # only for solids whose OBB is tighter than their AABB
        if idx.size and data.obb_centres is not None:
            test = idx[data.obb_useful[idx]]
            if test.size:
                mask = obbs_overlap(
                    data.obb_centres[test],
                    data.obb_axes[test],
                    data.obb_half_lengths[test],
                    (lo + hi) * 0.5,
                    _IDENTITY,
                    (hi - lo) * 0.5,
                )
                if not mask.all():
                    rejected = set(test[~mask].tolist())
                    idx = np.array([i for i in idx.tolist() if i not in rejected], dtype=int)

        use_pruning = bool(getattr(params, "void_use_halfspace_pruning", False))
        lo_t = lo.tolist()
        hi_t = hi.tolist()
        box_ocp: Optional[_LazyBox] = None

        candidates: Dict[int, tuple[int, ...]] = {}
        for i in idx.tolist():
            remaining = parent_candidates[i]

            # filter 3: half-space classification
            if use_pruning:
                status, remaining = data.classifier.prune_terms(
                    data.terms[i], remaining, lo_t, hi_t
                )
                if status == OUTSIDE:
                    continue
                if status == INSIDE:
                    self.covered = True
                    self._near_faces = {}
                    return {}

            # filter 4: exact check
            if self._uses_exact_check():
                if box_ocp is None:
                    box_ocp = _LazyBox(self.bounding_box)
                result = self._exact_check(i, box_ocp, lo, hi)
                if result == OUTSIDE:
                    continue
                if result == INSIDE:
                    self.covered = True
                    self._near_faces = {}
                    return {}

            candidates[i] = remaining

        # filter 5: octant covered by several touching solids. When every
        # candidate is reduced to a single plane half space the void is a convex
        # polytope whose emptiness is decided exactly by a small LP.
        if (
            use_pruning
            and len(candidates) > 1
            and all(len(t) == 1 for t in candidates.values())
            and box_minus_planar_halfspaces_is_empty(
                [data.terms[i][t[0]] for i, t in candidates.items()], lo_t, hi_t
            )
        ):
            self.covered = True
            self._near_faces = {}
            return {}

        return candidates

    def _uses_exact_check(self) -> bool:
        """Return whether the exact OCP check is enabled for this octree.

        Returns
        -------
        bool
            True if :meth:`_exact_check` should be applied.
        """
        return False

    def _exact_check(
        self, index: int, box_ocp: _LazyBox, lo: np.ndarray, hi: np.ndarray
    ) -> int:
        """Exactly test a split solid against this node's box.

        Parameters
        ----------
        index : int
            Split solid index.
        box_ocp : _LazyBox
            OCP box solid of this node, built on first use.
        lo : numpy.ndarray
            Box minimum corner.
        hi : numpy.ndarray
            Box maximum corner.

        Returns
        -------
        int
            ``OUTSIDE`` if the solid does not intersect the box, ``INSIDE`` if
            the box lies inside the solid, otherwise ``CROSSING`` (0).
        """
        return 0

    # ---------------------------------------------------------- recursion
    def _is_leaf(self) -> bool:
        """Return whether this node meets the termination criteria.

        Returns
        -------
        bool
            True if the node should not be split further.
        """
        raise NotImplementedError

    def split(self, n_splits: int = 1) -> List["VoidOctreeBase"]:
        """Subdivide the node's box and create the child nodes.

        Children are created from :meth:`BoundingBox.split` (halving along the
        longest axis ``n_splits`` times), filtered, and registered; this node's
        candidates are then removed from the progress register.

        Parameters
        ----------
        n_splits : int, optional
            Number of successive halvings, by default 1 (two children).

        Returns
        -------
        list of VoidOctreeBase
            The child nodes.
        """
        for bounding_box in self.bounding_box.split(n_splits=n_splits):
            child = self.__class__.__new__(self.__class__)
            child._init_child(self, bounding_box)
            self._children.append(child)

        self.register.remove(self._candidates)
        self.register.update()

        return self._children

    def run(self) -> None:
        """Recursively subdivide the octree until the termination criteria are met.

        Leaves (including octants found to lie inside a split solid) are removed
        from the progress register. A serialised node is deserialised first and
        re-serialised at the end.
        """
        requires_reserialisation = self.is_serialised
        if requires_reserialisation:
            self.deserialise()

        # iterative depth first traversal (avoids deep Python recursion)
        stack: List[VoidOctreeBase] = [self]
        while stack:
            node = stack.pop()
            if node.covered or node._is_leaf():
                node.register.remove(node._candidates)
                node.register.update()
                continue
            stack.extend(reversed(node.split()))

        if requires_reserialisation:
            self.serialise()

    # -------------------------------------------------------- properties
    @property
    def children(self) -> List["VoidOctreeBase"]:
        """list of VoidOctreeBase: Child nodes of this node."""
        return self._children

    @children.setter
    def children(self, children: List["VoidOctreeBase"]) -> None:
        """Set the child nodes.

        Parameters
        ----------
        children : list of VoidOctreeBase
            New child nodes.
        """
        self._children = children

    @property
    def show_progress(self) -> bool:
        """bool: Whether the progress bar is shown."""
        return self._show_progress

    @show_progress.setter
    def show_progress(self, show_progress: bool) -> None:
        """Set whether the progress bar is shown.

        Parameters
        ----------
        show_progress : bool
            Show the progress bar.
        """
        self._show_progress = show_progress
        self.register.show_progress = show_progress

    @property
    def _decom_params(self) -> DecompositionParameters:
        """DecompositionParameters: Alias kept for backwards compatibility."""
        return self.decom_params

    @property
    def split_solids(self) -> List[SplitSolid]:
        """list of SplitSolid: Split solids that may intersect this node."""
        return [self._data.split_solids[i] for i in self._candidates]

    @property
    def n_terms(self) -> int:
        """int: Number of half spaces needed to carve the solids from this node.

        With half-space pruning enabled this counts only the half spaces not
        proven to contain the node's box; otherwise it is the total number of
        faces of the candidate split solids.
        """
        if getattr(self.decom_params, "void_use_halfspace_pruning", False):
            return sum(len(t) for t in self._candidates.values())
        return sum(self._data.split_solids[i].n_faces for i in self._candidates)

    @property
    def voids(self) -> List["VoidOctreeBase"]:
        """list of VoidOctreeBase: Leaf nodes that require a void cell.

        Leaves lying entirely inside a split solid (:attr:`covered`) are
        excluded.
        """
        voids: List[VoidOctreeBase] = []
        stack: List[VoidOctreeBase] = [self]
        while stack:
            node = stack.pop()
            if node._children:
                stack.extend(reversed(node._children))
            elif not node.covered:
                voids.append(node)
        return voids

    def void_region(self) -> "Volume":
        """Return the CSG region of this node's void cell.

        The node's RPP with each candidate split solid subtracted. With
        half-space pruning, each solid is represented only by the half spaces
        not proven to contain the box, which describes the same region within
        the box.

        Returns
        -------
        Volume
            Void region of the node.
        """
        void_volume = self.bounding_box_mcio_volume()
        use_pruning = getattr(self.decom_params, "void_use_halfspace_pruning", False)
        for i, remaining in self._candidates.items():
            if use_pruning:
                terms = self._data.terms[i]
                region = intersect_terms([terms[t] for t in remaining])
            else:
                region = self._data.split_solids[i].volume
            void_volume = void_volume & ~region
        return void_volume

    @property
    def mcio_cells(self) -> List[Cell]:
        """list of Cell: Void cells for each leaf octant, plus graveyard cells.

        Each leaf octant that is not inside a split solid gives a cell (see
        :meth:`void_region`). A sphere enclosing the root bounding box is
        added, with a graveyard cell outside it and a void cell between the
        octree RPP and the sphere. The sphere surface is stored in
        ``self.surf_stochastic_volume_sphere``. The result is cached.
        """
        if self._mcio_cells:
            return self._mcio_cells

        cells: List[Cell] = []
        voids = self.voids
        n_voids = len(voids)

        for i, void in enumerate(voids, start=1):
            bb = void.bounding_box
            cell_label = (
                f"void_cell_{i}_of_{n_voids} - location = "
                f"{bb.x_mid:.2f} {bb.y_mid:.2f} {bb.z_mid:.2f} ex {bb.extent:.2f}"
            )
            cell_comment = (
                f"n_split_solids = {len(void._candidates)} "
                f"n_surfaces = {void.n_terms}"
            )
            cells.append(
                Cell(
                    volume=void.void_region(),
                    name=cell_label,
                    comment=cell_comment,
                    universe=self.universe,
                )
            )

        # set up sphere around model
        sphere_outer_radius = (
            (self.bounding_box.x_length / 2.0) ** 2
            + (self.bounding_box.y_length / 2.0) ** 2
            + (self.bounding_box.z_length / 2.0) ** 2
        ) ** 0.5
        surf_sphere = Sphere(
            x0=self.bounding_box.x_mid,
            y0=self.bounding_box.y_mid,
            z0=self.bounding_box.z_mid,
            radius=sphere_outer_radius,
        )

        # volumes outside the sphere and between the octree RPP and the sphere
        vol_graveyard = +surf_sphere
        vol_void_outside_octree = ~self.bounding_box_mcio_volume() & ~vol_graveyard

        cells.extend(
            [
                Cell(
                    volume=vol_graveyard,
                    name="graveyard",
                    graveyard=True,
                    universe=self.universe,
                ),
                Cell(
                    volume=vol_void_outside_octree,
                    name="void_between_model_and_graveyard",
                    universe=self.universe,
                ),
            ]
        )

        # save surface to allow use in stochastic volume calculations
        self.surf_stochastic_volume_sphere: Sphere = surf_sphere

        self._mcio_cells = cells
        return self._mcio_cells

    # ---------------------------------------------------- serialisation
    def serialise(self) -> None:
        """Serialise the OCP shapes of all split solids to binaries.

        Nodes only store integer indices, so serialising the shared split solid
        list is sufficient for the whole octree to be pickled.
        """
        for split_solid in self._data.split_solids:
            if not split_solid.is_serialised:
                split_solid.serialise()
        for node in self._all_nodes():
            node.is_serialised = True

    def deserialise(self) -> None:
        """Rebuild the OCP shapes of all split solids from their binaries."""
        for split_solid in self._data.split_solids:
            if split_solid.is_serialised:
                split_solid.deserialise()
        self._data._face_aabbs = {}
        for node in self._all_nodes():
            node.is_serialised = False

    def _all_nodes(self) -> List["VoidOctreeBase"]:
        """Return this node and all of its descendants.

        Returns
        -------
        list of VoidOctreeBase
            Nodes in depth first order.
        """
        nodes: List[VoidOctreeBase] = []
        stack: List[VoidOctreeBase] = [self]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node._children)
        return nodes

    def pickle(self) -> None:
        """Serialise the octree and report attributes that cannot be pickled."""
        self.serialise()

        # --- DIAGNOSTIC CODE ---
        print("--- Inspecting Decomposer Attributes for Pickle Safety ---")
        for attr_name, attr_value in self.__dict__.items():
            try:
                pickle.dumps(attr_value)
            except Exception as e:
                print(f"FAILED PICKLE: '{attr_name}' (type: {type(attr_value)}) -> {e}")
        # -----------------------

    # ------------------------------------------------------------ geometry
    def bounding_box_mcio_volume(self) -> Volume:
        """Construct an MCIO CSG volume bounded by the node's bounding box.

        The six axis-aligned planes are looked up in, or added to,
        ``self._surfaces_generated`` (keyed by surface type then
        ``__hash_string__()``) so that identical planes are shared between
        octants.

        Returns
        -------
        Volume
            Intersection of the six half-spaces forming the bounding box.
        """
        bb = self.bounding_box
        surfaces = [
            XPlane(x0=bb.x_min),
            XPlane(x0=bb.x_max),
            YPlane(y0=bb.y_min),
            YPlane(y0=bb.y_max),
            ZPlane(z0=bb.z_min),
            ZPlane(z0=bb.z_max),
        ]
        shared = []
        for surface in surfaces:
            store = self._surfaces_generated[type(surface)]
            key = surface.__hash_string__()
            if key in store:
                surface = store[key]
            else:
                store[key] = surface
            shared.append(surface)

        s_x_min, s_x_max, s_y_min, s_y_max, s_z_min, s_z_max = shared
        return +s_x_min & -s_x_max & +s_y_min & -s_y_max & +s_z_min & -s_z_max

    def _generate_bounding_box(self, items_with_bounding_boxes: List[Any]) -> BoundingBox:
        """Compute a box enclosing a collection of items.

        If ``self._make_square`` is set, all three lengths are set to the longest
        length (about the existing centre) before scaling. The items' own boxes
        are never modified.

        Parameters
        ----------
        items_with_bounding_boxes : list
            Objects with a ``bounding_box`` attribute (e.g. SplitSolid).

        Returns
        -------
        BoundingBox
            Enclosing bounding box scaled by 1.05 about its centre.
        """
        # copy (``* 1.0`` returns a new box) so the first item is not modified
        bounding_box: BoundingBox = items_with_bounding_boxes[0].bounding_box * 1.0
        for item_with_bounding_box in items_with_bounding_boxes[1:]:
            bounding_box = bounding_box | item_with_bounding_box.bounding_box

        if self._make_square:
            max_length = max(
                bounding_box.x_length, bounding_box.y_length, bounding_box.z_length
            )
            bounding_box.x_length = max_length
            bounding_box.y_length = max_length
            bounding_box.z_length = max_length

        return bounding_box * 1.05

    def save(
        self, output_path: Union[str, os.PathLike] = "analysis/solids/void_octree_1.stp"
    ) -> None:
        """Export the node's box and its candidate split solids to a STEP file.

        Parameters
        ----------
        output_path : str or os.PathLike, optional
            Target output file path, by default "analysis/solids/void_octree_1.stp".
        """
        n_split_solids = len(self._candidates)
        width_min = min(
            self.bounding_box.x_length,
            self.bounding_box.y_length,
            self.bounding_box.z_length,
        )

        step_file_writer: STEPFileWriter = STEPFileWriter(self.decom_params)
        step_file_writer.add_solids(
            [ocp_box_from_bounding_box(self.bounding_box)],
            f"split_solids_{n_split_solids}_terms_{self.n_terms}_width_{width_min}_level_{self._level}",
        )
        for split_solid in self.split_solids:
            step_file_writer.add_solids(
                [split_solid.split_solid],
                f"fraction_{split_solid.fraction_of_bounding_box:.3f}",
            )
        step_file_writer.write(output_path, enumerate_files=True)


class VoidOctreeSplitSolids(VoidOctreeBase):
    """Void octree that keeps, per octant, the split solids intersecting it.

    The filters are described in the module docstring. The optional exact check
    is the OCP distance between the split solid and the octant box
    (``void_use_split_solid_distance_check``, ``void_split_solid_distance_tol``).

    A node stops splitting when it lies inside a split solid, when its level
    reaches ``void_max_recursion_depth``, when its *longest* side is at most
    ``void_min_width``, or when it has at most
    ``void_max_split_solids_per_octant`` split solids and fewer than
    ``void_max_faces_per_octant`` half spaces (see :attr:`n_terms`).

    Parameters
    ----------
    split_solids : list of SplitSolid
        Split solids to carve from the void.
    decom_params : DecompositionParameters
        Configuration options controlling octree depth and filtering.
    surfaces_generated : dict
        Mapping tracking existing generated surfaces to prevent duplication.
    bounding_box : BoundingBox, optional
        Spatial domain of the root node, by default None (computed).
    parent : VoidOctreeSplitSolids, optional
        Parent node; when given, ``split_solids`` is ignored and the node
        filters the parent's candidates, by default None.
    make_square : bool, optional
        Whether to make the computed root box a cube, by default True.
    universe : Universe, optional
        MCIO Universe assignment for generated void cells, by default None.
    level : int, optional
        Depth of the root node, by default 0.
    show_progress : bool, optional
        Whether to show a progress bar, by default True.
    """

    def __init__(
        self,
        split_solids: List[SplitSolid],
        decom_params: DecompositionParameters,
        surfaces_generated: Dict[Any, Any],
        bounding_box: Optional[BoundingBox] = None,
        parent: Optional["VoidOctreeSplitSolids"] = None,
        make_square: bool = True,
        universe: Optional[Universe] = None,
        level: int = 0,
        show_progress: bool = True,
    ) -> None:
        """Initialise the octree node."""
        if parent is not None:
            self._init_child(parent, bounding_box)
            return
        self._init_root(
            split_solids,
            decom_params,
            surfaces_generated,
            bounding_box,
            make_square,
            universe,
            level,
            show_progress,
            "Generating void",
        )

    def _uses_exact_check(self) -> bool:
        """Return whether the solid-to-box distance check is enabled.

        Returns
        -------
        bool
            Value of ``void_use_split_solid_distance_check``.
        """
        return bool(self.decom_params.void_use_split_solid_distance_check)

    def _exact_check(
        self, index: int, box_ocp: _LazyBox, lo: np.ndarray, hi: np.ndarray
    ) -> int:
        """Test whether a split solid lies within tolerance of the box.

        Split solids bounded only by planes are convex polytopes and are tested
        exactly with the separating axis theorem (see
        :meth:`_OctreeData.polytope`); other solids use the OCP shape distance.

        Parameters
        ----------
        index : int
            Split solid index.
        box_ocp : _LazyBox
            OCP box solid of this node (used for the OCP distance).
        lo : numpy.ndarray
            Box minimum corner.
        hi : numpy.ndarray
            Box maximum corner.

        Returns
        -------
        int
            ``OUTSIDE`` if the solid is separated from the box by at least
            ``void_split_solid_distance_tol``, otherwise ``CROSSING`` (0).
        """
        tol = self.decom_params.void_split_solid_distance_tol
        polytope = self._data.polytope(index)
        if polytope is not None:
            vertices, axes = polytope
            centre = (lo + hi) * 0.5
            half = (hi - lo) * 0.5
            projected = vertices @ axes.T
            box_centre = axes @ centre
            box_radius = np.abs(axes) @ half
            separated = (projected.max(axis=0) < box_centre - box_radius - tol) | (
                projected.min(axis=0) > box_centre + box_radius + tol
            )
            return OUTSIDE if separated.any() else 0

        split_solid = self._data.split_solids[index]
        distance = _distance_between_shapes(split_solid.split_solid, box_ocp.shape)
        if distance < self.decom_params.void_split_solid_distance_tol:
            return 0
        return OUTSIDE

    def _is_leaf(self) -> bool:
        """Return whether this node meets the termination criteria.

        Returns
        -------
        bool
            True if the node should not be split further.
        """
        params = self.decom_params
        if self._level >= params.void_max_recursion_depth:
            return True
        max_length = max(
            self.bounding_box.x_length,
            self.bounding_box.y_length,
            self.bounding_box.z_length,
        )
        if max_length <= params.void_min_width:
            return True
        return (len(self._candidates) <= params.void_max_split_solids_per_octant) and (
            self.n_terms < params.void_max_faces_per_octant
        )


class VoidOctreeFaces(VoidOctreeBase):
    """Void octree whose exact check is based on the faces of the split solids.

    After the AABB, OBB and half-space filters, a split solid is kept only if
    one of its faces whose AABB overlaps the octant lies within
    ``void_face_distance_tol`` of the octant box. When no face is near, the box
    is either entirely inside or entirely outside the solid; the octant centre
    is classified to decide, and an octant inside a solid needs no void cell.
    The faces near each octant are recorded in
    :attr:`split_solid_faces_filterable`.

    A node stops splitting when it lies inside a split solid, when its level
    reaches ``void_max_recursion_depth``, when its *shortest* side is at most
    ``void_min_width``, or when at most ``void_max_faces_per_octant`` faces are
    near it.

    Parameters
    ----------
    split_solids_with_faces_filtered_by_parent : dict
        Mapping of each SplitSolid to its candidate OCP faces (keys of
        ``SplitSolid.face_to_mcio_halfspaces``). Only the keys are used: all
        faces of each split solid are considered.
    decom_params : DecompositionParameters
        Configuration options controlling octree depth and face limits.
    surfaces_generated : dict
        Mapping tracking existing generated surfaces to prevent duplication.
    bounding_box : BoundingBox, optional
        Spatial domain of the root node, by default None (computed).
    parent : VoidOctreeFaces, optional
        Parent node; when given, the first argument is ignored and the node
        filters the parent's candidates, by default None.
    make_square : bool, optional
        Whether to make the computed root box a cube, by default True.
    universe : Universe, optional
        MCIO Universe for the generated void cells, by default None.
    level : int, optional
        Depth of the root node, by default 0.
    show_progress : bool, optional
        Whether to show a progress bar, by default True.
    """

    def __init__(
        self,
        split_solids_with_faces_filtered_by_parent: Dict[Any, List[Any]],
        decom_params: DecompositionParameters,
        surfaces_generated: Dict[Any, Any],
        bounding_box: Optional[BoundingBox] = None,
        parent: Optional["VoidOctreeFaces"] = None,
        make_square: bool = True,
        universe: Optional[Universe] = None,
        level: int = 0,
        show_progress: bool = True,
    ) -> None:
        """Initialise the octree node."""
        if parent is not None:
            self._init_child(parent, bounding_box)
            return
        self._init_root(
            list(split_solids_with_faces_filtered_by_parent),
            decom_params,
            surfaces_generated,
            bounding_box,
            make_square,
            universe,
            level,
            show_progress,
            "Generating void (faces)",
        )
        # at the root every face of every split solid is a candidate
        self._near_faces = {
            i: tuple(range(len(ss.face_to_mcio_halfspaces)))
            for i, ss in enumerate(self._data.split_solids)
        }

    def _uses_exact_check(self) -> bool:
        """Return True: the face distance check is always used.

        Returns
        -------
        bool
            Always True.
        """
        return True

    def _exact_check(
        self, index: int, box_ocp: _LazyBox, lo: np.ndarray, hi: np.ndarray
    ) -> int:
        """Test a split solid against the box using its faces.

        Parameters
        ----------
        index : int
            Split solid index.
        box_ocp : _LazyBox
            OCP box solid of this node, built on first use.
        lo : numpy.ndarray
            Box minimum corner.
        hi : numpy.ndarray
            Box maximum corner.

        Returns
        -------
        int
            ``CROSSING`` (0) if a face lies within ``void_face_distance_tol`` of
            the box, otherwise ``INSIDE`` or ``OUTSIDE`` from classifying the box
            centre against the solid.
        """
        split_solid = self._data.split_solids[index]
        faces = list(split_solid.face_to_mcio_halfspaces)
        face_lo, face_hi = self._data.face_aabbs(index)
        tol = self.decom_params.void_face_distance_tol

        # faces whose (tolerance grown) AABB overlaps the box
        near = np.nonzero(
            np.all(face_lo - tol < hi, axis=1) & np.all(face_hi + tol > lo, axis=1)
        )[0]
        near_faces = tuple(
            int(f) for f in near if _distance_between_shapes(faces[f], box_ocp.shape) < tol
        )
        if near_faces:
            self._near_faces[index] = near_faces
            return 0

        # no face near the box: the box is entirely inside or outside the solid
        centre = (lo + hi) * 0.5
        classifier = BRepClass3d_SolidClassifier(
            split_solid.split_solid, gp_Pnt(*centre.tolist()), tol
        )
        return INSIDE if classifier.State() == TopAbs_IN else OUTSIDE

    def _is_leaf(self) -> bool:
        """Return whether this node meets the termination criteria.

        Returns
        -------
        bool
            True if the node should not be split further.
        """
        params = self.decom_params
        if self._level >= params.void_max_recursion_depth:
            return True
        min_length = min(
            self.bounding_box.x_length,
            self.bounding_box.y_length,
            self.bounding_box.z_length,
        )
        if min_length <= params.void_min_width:
            return True
        return self.n_faces <= params.void_max_faces_per_octant

    @property
    def n_faces(self) -> int:
        """int: Number of split solid faces near this node."""
        if self._parent is None:
            return sum(len(f) for f in self._near_faces.values())
        return sum(len(self._near_faces.get(i, ())) for i in self._candidates)

    @property
    def void_is_required(self) -> bool:
        """bool: False when the node lies inside a split solid."""
        return not self.covered

    @property
    def split_solid_faces_filterable(self) -> Dict[SplitSolid, List["TopoDS_Face"]]:
        """dict: Each candidate split solid mapped to its faces near this node."""
        result: Dict[SplitSolid, List[Any]] = {}
        for i in self._candidates:
            split_solid = self._data.split_solids[i]
            faces = list(split_solid.face_to_mcio_halfspaces)
            result[split_solid] = [faces[f] for f in self._near_faces.get(i, ())]
        return result

    @property
    def bounding_box_as_solid(self) -> "TopoDS_Solid":
        """TopoDS_Solid: OCP box solid of this node's bounding box."""
        return ocp_box_from_bounding_box(self.bounding_box)

    def save(
        self, output_path: Union[str, os.PathLike] = "analysis/solids/void_octree_1.stp"
    ) -> None:
        """Export the node's box, candidate split solids and near faces to STEP.

        Parameters
        ----------
        output_path : str or os.PathLike, optional
            Target output path, by default "analysis/solids/void_octree_1.stp".
        """
        faces: List[Any] = []
        for face_list in self.split_solid_faces_filterable.values():
            faces.extend(face_list)

        step_file_writer = STEPFileWriter(self.decom_params)
        step_file_writer.add_solids([ocp_box_from_bounding_box(self.bounding_box)], "void")
        step_file_writer.add_solids([ss.split_solid for ss in self.split_solids], "split_solids")
        if faces:
            step_file_writer.add_solids(faces, "faces")
        step_file_writer.write(output_path, enumerate_files=True)
