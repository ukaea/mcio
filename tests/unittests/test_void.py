"""
unittest: void generation - OBB separating axis test, OBB enclosure, half space
box classification and end-to-end void octree correctness.
"""

import math
import unittest

import numpy as np

import mcio
import mcio.surface as S
from mcio.geometric import BoundingBox, OrientedBoundingBox, obbs_overlap
from mcio.decomposevoidclassify import BoxClassifier, INSIDE, OUTSIDE, CROSSING
from mcio.decomposeparams import DecompositionParameters


from unitutils import TemporaryWorkingDirectory


# run in a temporary working directory, as the decomposer writes files to the working directory
_working_directory = TemporaryWorkingDirectory()


def setUpModule():
    _working_directory.enter()


def tearDownModule():
    _working_directory.exit()


try:
    import openmc  # only used for independent region evaluation
    HAS_OPENMC = True
except Exception:
    HAS_OPENMC = False


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Return a random proper rotation matrix (rows are the rotated axes)."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.T


def _boxes_overlap_lp(c1, A1, h1, c2, A2, h2) -> float:
    """Return the largest slack of a point inside both boxes (>0 overlap, <0 apart)."""
    from scipy.optimize import linprog
    a_ub = np.vstack([A1, -A1, A2, -A2])
    b_ub = np.concatenate([h1 + A1 @ c1, h1 - A1 @ c1, h2 + A2 @ c2, h2 - A2 @ c2])
    res = linprog(
        c=[0, 0, 0, -1],
        A_ub=np.hstack([a_ub, np.ones((12, 1))]),
        b_ub=b_ub,
        bounds=[(None, None)] * 3 + [(None, 10)],
    )
    return -res.fun


class MCIOUnitTestOBBSeparatingAxis(unittest.TestCase):
    """The OBB separating axis test must agree with an exact LP feasibility test."""

    def test_random_obb_pairs_against_lp(self):
        rng = np.random.default_rng(1)
        wrong = 0
        for k in range(1500):
            A1 = _random_rotation(rng)
            A2 = np.identity(3) if k % 2 else _random_rotation(rng)
            h1, h2 = rng.uniform(0.1, 3, 3), rng.uniform(0.1, 3, 3)
            c1, c2 = rng.uniform(-4, 4, 3), rng.uniform(-4, 4, 3)
            slack = _boxes_overlap_lp(c1, A1, h1, c2, A2, h2)
            if abs(slack) < 1e-6:
                continue
            obb = OrientedBoundingBox(tuple(c1), tuple(map(tuple, A1)), tuple(h1))
            if k % 2:
                other = BoundingBox(c2[0] - h2[0], c2[0] + h2[0], c2[1] - h2[1],
                                    c2[1] + h2[1], c2[2] - h2[2], c2[2] + h2[2])
            else:
                other = OrientedBoundingBox(tuple(c2), tuple(map(tuple, A2)), tuple(h2))
            if (obb ^ other) != (slack > 0):
                wrong += 1
        self.assertEqual(wrong, 0)

    def test_aabb_xor_obb_symmetric(self):
        aabb = BoundingBox(0, 1, 0, 1, 0, 1)
        r = 1.0 / math.sqrt(2.0)
        obb = OrientedBoundingBox((1.6, 0.5, 0.5), ((r, r, 0), (-r, r, 0), (0, 0, 1)), (0.5, 0.5, 0.5))
        # rotated cube corner reaches x = 1.6 - 0.707 = 0.893 < 1 -> overlap
        self.assertTrue(aabb ^ obb)
        self.assertTrue(obb ^ aabb)
        obb_far = OrientedBoundingBox((1.8, 0.5, 0.5), ((r, r, 0), (-r, r, 0), (0, 0, 1)), (0.5, 0.5, 0.5))
        self.assertFalse(aabb ^ obb_far)
        self.assertFalse(obb_far ^ aabb)

    def test_vectorised_matches_scalar(self):
        rng = np.random.default_rng(3)
        n = 200
        centres = rng.uniform(-3, 3, (n, 3))
        axes = np.array([_random_rotation(rng) for _ in range(n)])
        halfs = rng.uniform(0.1, 2, (n, 3))
        mask = obbs_overlap(centres, axes, halfs, np.zeros(3), np.identity(3), np.ones(3))
        aabb = BoundingBox(-1, 1, -1, 1, -1, 1)
        for i in range(n):
            obb = OrientedBoundingBox(tuple(centres[i]), tuple(map(tuple, axes[i])), tuple(halfs[i]))
            self.assertEqual(bool(mask[i]), obb ^ aabb)


class MCIOUnitTestOBBEnclosure(unittest.TestCase):
    """OBBs built from OCP shapes must enclose the shape."""

    def test_rotated_cylinder_enclosed(self):
        from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
        from mcio.ocpboundingbox import oriented_bounding_box_from_shape

        axis = np.array([1.0, 2.0, 0.5]); axis /= np.linalg.norm(axis)
        base = np.array([3.0, -2.0, 1.0]); radius = 4.0; height = 30.0
        cyl = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*base), gp_Dir(*axis)), radius, height).Shape()
        obb = oriented_bounding_box_from_shape(cyl)

        # sample points on the curved surface and the caps
        u = np.cross(axis, [0.0, 0.0, 1.0]); u /= np.linalg.norm(u); v = np.cross(axis, u)
        pts = []
        for t in np.linspace(0, 2 * math.pi, 73):
            for s in np.linspace(0, height, 11):
                pts.append(base + axis * s + radius * (math.cos(t) * u + math.sin(t) * v))
        pts = np.array(pts)
        local = np.abs((pts - obb.center) @ obb.axes.T)
        self.assertTrue(np.all(local <= obb.half_lengths + 1e-9))

    def test_obb_box_solid_matches_obb(self):
        from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
        from mcio.ocpboundingbox import oriented_bounding_box_from_shape, ocp_box_from_oriented_bounding_box
        from mcio.ocpproperties import _volume_of_ocp_solids

        cyl = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 1, 1)), 2.0, 20.0).Shape()
        obb = oriented_bounding_box_from_shape(cyl)
        box = ocp_box_from_oriented_bounding_box(obb)
        self.assertAlmostEqual(_volume_of_ocp_solids([box]) / obb.volume, 1.0, places=6)


def _f(surface, P):
    """Independent evaluation of the MCNP surface function for test surfaces."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    if isinstance(surface, S.Plane):
        return surface.a * x + surface.b * y + surface.c * z - surface.d
    if isinstance(surface, S.Sphere):
        return (x - surface.x0) ** 2 + (y - surface.y0) ** 2 + (z - surface.z0) ** 2 - surface.radius ** 2
    if isinstance(surface, S.CylinderParallelToZAxis):
        return (x - surface.x0) ** 2 + (y - surface.y0) ** 2 - surface.radius ** 2
    if isinstance(surface, S.ConeParallelToXAxis):
        return (y - surface.y0) ** 2 + (z - surface.z0) ** 2 - surface.t_sq * (x - surface.x0) ** 2
    if isinstance(surface, S.ToriParralleltoZAxis):
        rho = np.sqrt((x - surface.x0) ** 2 + (y - surface.y0) ** 2)
        return (z - surface.z0) ** 2 / surface.b ** 2 + (rho - surface.a) ** 2 / surface.c ** 2 - 1
    if isinstance(surface, S.GeneralQuadratic):
        s = surface
        return (s.a * x * x + s.b * y * y + s.c * z * z + s.d * x * y + s.e * y * z
                + s.f * z * x + s.g * x + s.h * y + s.j * z + s.k)
    raise NotImplementedError


class MCIOUnitTestBoxClassifier(unittest.TestCase):
    """Inside/outside decisions must hold at every point of the box."""

    def _random_surface(self, rng, k):
        c = rng.uniform(-5, 5, 3); r = rng.uniform(0.5, 6)
        if k == 0:
            n = rng.normal(size=3)
            return S.Plane(a=n[0], b=n[1], c=n[2], d=rng.uniform(-5, 5))
        if k == 1:
            return S.Sphere(x0=c[0], y0=c[1], z0=c[2], radius=r)
        if k == 2:
            return S.CylinderParallelToZAxis(x0=c[0], y0=c[1], radius=r)
        if k == 3:
            return S.ConeParallelToXAxis(x0=c[0], y0=c[1], z0=c[2], t_sq=rng.uniform(0.1, 3), sheet=0)
        if k == 4:
            return S.ToriParralleltoZAxis(x0=c[0], y0=c[1], z0=c[2], a=r + 2, b=1.0, c=1.5)
        # rotated cylinder as a general quadric
        axis = _random_rotation(rng)[2]
        P = np.identity(3) - np.outer(axis, axis)
        L = -2 * P @ c; K = c @ P @ c - r * r
        return S.GeneralQuadratic(a=P[0, 0], b=P[1, 1], c=P[2, 2], d=2 * P[0, 1], e=2 * P[1, 2],
                                  f=2 * P[0, 2], g=L[0], h=L[1], j=L[2], k=K)

    def test_decisions_are_proofs(self):
        rng = np.random.default_rng(7)
        classifier = BoxClassifier()
        decided = 0
        for trial in range(3000):
            surface = self._random_surface(rng, trial % 6)
            half_volume = +surface if trial % 2 else -surface
            lo = rng.uniform(-8, 6, 3); hi = lo + rng.uniform(0.05, 5, 3)
            status = classifier.classify(half_volume, lo, hi)
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            pts = np.vstack([rng.uniform(lo, hi, (300, 3)), corners])
            values = _f(surface, pts)
            inside = values > 0 if trial % 2 else values < 0
            if status == INSIDE:
                decided += 1
                self.assertTrue(inside.all())
            elif status == OUTSIDE:
                decided += 1
                self.assertFalse(inside.any())
        # the classifier should resolve a good share of the cases
        self.assertGreater(decided, 1500)

    def test_transformed_surface_is_never_decided(self):
        surface = S.Sphere(radius=1.0, transform=object())
        self.assertEqual(BoxClassifier().classify(-surface, [10, 10, 10], [11, 11, 11]), CROSSING)

    def test_prune_terms(self):
        # unit cube solid: 0 < x,y,z < 1
        terms = [+S.XPlane(x0=0), -S.XPlane(x0=1), +S.YPlane(y0=0), -S.YPlane(y0=1), +S.ZPlane(z0=0), -S.ZPlane(z0=1)]
        clf = BoxClassifier()
        # box crossing only the x = 1 face
        status, remaining = clf.prune_terms(terms, range(6), [0.9, 0.2, 0.2], [1.2, 0.8, 0.8])
        self.assertEqual(status, CROSSING)
        self.assertEqual(remaining, (1,))
        # box inside the cube
        status, remaining = clf.prune_terms(terms, range(6), [0.2, 0.2, 0.2], [0.8, 0.8, 0.8])
        self.assertEqual(status, INSIDE)
        # box beyond x = 1
        status, remaining = clf.prune_terms(terms, range(6), [1.1, 0.2, 0.2], [1.5, 0.8, 0.8])
        self.assertEqual(status, OUTSIDE)


def _eval_region(region, P, cache):
    """Vectorised evaluation of an OpenMC region at points."""
    if isinstance(region, openmc.Halfspace):
        key = id(region.surface)
        if key not in cache:
            try:
                a, b, c, d, e, f, g, h, j, k = region.surface._get_base_coeffs()
                x, y, z = P[:, 0], P[:, 1], P[:, 2]
                cache[key] = a*x*x + b*y*y + c*z*z + d*x*y + e*y*z + f*x*z + g*x + h*y + j*z + k
            except Exception:
                cache[key] = np.array([region.surface.evaluate(p) for p in P])
        return cache[key] > 0 if region.side == "+" else cache[key] < 0
    if isinstance(region, openmc.Intersection):
        out = np.ones(len(P), dtype=bool)
        for r in region:
            out &= _eval_region(r, P, cache)
        return out
    if isinstance(region, openmc.Union):
        out = np.zeros(len(P), dtype=bool)
        for r in region:
            out |= _eval_region(r, P, cache)
        return out
    if isinstance(region, openmc.Complement):
        return ~_eval_region(region.node, P, cache)
    return np.array([tuple(p) in region for p in P])


@unittest.skipUnless(HAS_OPENMC, "openmc python package required for region evaluation")
class MCIOUnitTestVoidOctree(unittest.TestCase):
    """Every point of space must lie in exactly one cell, for every void method."""

    @classmethod
    def setUpClass(cls):
        from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
        cls.solids = {
            "slab": BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 100.0, 100.0, 20.0).Shape(),
            "block": BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 20), 30.0, 30.0, 30.0).Shape(),
            "rod": BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(60, 60, 30), gp_Dir(1, 1, 1)), 5.0, 40.0).Shape(),
        }

    def _check(self, **overrides):
        params = DecompositionParameters()
        params.void_max_split_solids_per_octant = 1
        params.void_max_faces_per_octant = 12
        params.void_min_width = 5.0
        for key, value in overrides.items():
            setattr(params, key, value)
        decomposer = mcio.decompose.Decomposer(decompose_parameters=params)
        for name, solid in self.solids.items():
            decomposer.add_solid(solid, name)
        decomposer.run(cores=1)
        model = decomposer.model

        bb = decomposer.void_octree.bounding_box
        rng = np.random.default_rng(0)
        lo, hi = np.array(bb.minimums), np.array(bb.maximums)
        pts = rng.uniform(lo - 5, hi + 5, (20000, 3))
        counts = np.zeros(len(pts), dtype=int)
        cache = {}
        for cell in model.cells:
            counts += _eval_region(cell.volume.openmc_region, pts, cache)
        self.assertEqual(int((counts == 0).sum()), 0, "points in no cell (gaps)")
        self.assertEqual(int((counts > 1).sum()), 0, "points in several cells (overlaps)")
        return decomposer

    def test_split_solids_aabb_only(self):
        self._check(void_use_obb_check=False, void_use_halfspace_pruning=False)

    def test_split_solids_obb(self):
        self._check(void_use_halfspace_pruning=False)

    def test_split_solids_obb_and_pruning(self):
        self._check()

    def test_split_solids_distance_check(self):
        self._check(void_use_split_solid_distance_check=True)

    def test_filter_by_face(self):
        decomposer = self._check(void_filter_by_face=True)
        self.assertIsInstance(decomposer.void_octree, mcio.decomposevoid.VoidOctreeFaces)

    def test_filter_by_face_without_pruning(self):
        self._check(void_filter_by_face=True, void_use_halfspace_pruning=False)

    def test_pruning_reduces_void_complexity(self):
        def total_terms(d):
            return sum(v.n_terms for v in d.void_octree.voids)
        plain = self._check(void_use_obb_check=False, void_use_halfspace_pruning=False)
        pruned = self._check()
        self.assertLess(total_terms(pruned), total_terms(plain))

    def test_polytope_separating_axis_matches_ocp_distance(self):
        from mcio.decomposevoid import _LazyBox, OUTSIDE
        from mcio.ocpproperties import _distance_between_shapes
        decomposer = self._check(void_use_split_solid_distance_check=True)
        octree = decomposer.void_octree
        data = octree._data
        rng = np.random.default_rng(2)
        n_tested = 0
        for k in range(300):
            i = k % len(data.split_solids)
            if data.polytope(i) is None:
                continue
            ss = data.split_solids[i]
            lo = np.array(ss.bounding_box.minimums) - 20
            hi = np.array(ss.bounding_box.maximums) + 20
            c = rng.uniform(lo, hi); w = rng.uniform(1, 20, 3)
            box = BoundingBox(c[0] - w[0], c[0] + w[0], c[1] - w[1], c[1] + w[1], c[2] - w[2], c[2] + w[2])
            lazy = _LazyBox(box)
            result = octree._exact_check(i, lazy, np.array(box.minimums), np.array(box.maximums))
            distance = _distance_between_shapes(ss.split_solid, lazy.shape)
            if result == OUTSIDE:
                self.assertGreaterEqual(distance, params_tol := decomposer.decom_params.void_split_solid_distance_tol)
            elif distance > 1e-3:
                self.fail(f"separated solid (distance {distance}) not detected")
            n_tested += 1
        self.assertGreater(n_tested, 50)

    def test_octree_serialise_round_trip(self):
        import pickle
        decomposer = self._check()
        octree = decomposer.void_octree
        n_voids = len(octree.voids)
        octree.serialise()
        restored = pickle.loads(pickle.dumps(octree))
        restored.deserialise()
        self.assertEqual(len(restored.voids), n_voids)
