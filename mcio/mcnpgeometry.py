"""Read the geometry of an MCNP input deck into mcio cells.

Supports the cell, surface and TR/*TR cards needed to invert a deck into CAD:

* surfaces: P (coefficient and three point forms), PX/PY/PZ, SO/S/SX/SY/SZ,
  C/X C/Y C/Z, CX/CY/CZ, K/X K/Y K/Z, KX/KY/KZ (with sheets), TX/TY/TZ, SQ, GQ
  and the RPP, BOX and RCC macrobodies (including facets ``n.k``)
* surface transforms, TR and *TR (m = 1 or -1, full or two-vector rotations)
* cells: material and density, intersection, union (``:``), parentheses,
  cell complements (``#n``) and expression complements (``#(...)``),
  ``u=`` universes and ``fill=`` with a TR number or an inline transform
* the IMP:N data card, to find the outside (importance zero) cells

Every surface is held as a quadric (a 4 x 4 symmetric matrix) so that
transforms, including those inherited from fills, are applied exactly. When a
cell is built the quadric is converted back to the simplest mcio surface
(plane, axis aligned plane, sphere, axis parallel cylinder or cone) or to a
general quadratic. Tori are kept as tori, and are only supported when their
transformed axis is parallel to a cartesian axis.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

import mcio.surface as S
from mcio.volume import And, Or, HalfVolume, Volume


class MCNPGeometryError(Exception):
    """Raised when part of a deck cannot be represented."""


# ------------------------------------------------------------------------------------------------ cards

_COMMENT = re.compile(r'^\s{0,4}[cC](\s|$)')


def _expand_repeats(tokens: list[str]) -> list[str]:
    """Expand MCNP ``nR`` repeat shortcuts in a list of data tokens."""
    out: list[str] = []
    for token in tokens:
        match = re.fullmatch(r'(\d*)[rR]', token)
        if match and out:
            out.extend([out[-1]] * int(match.group(1) or 1))
        else:
            out.append(token)
    return out


def _join_cards(rows: list[str]) -> list[str]:
    """Join the rows of one block into cards, removing comments."""
    cards: list[str] = []
    continue_next = False
    for row in rows:
        if _COMMENT.match(row):
            continue
        row = row.split('$', 1)[0].replace('\t', ' ').rstrip()
        if not row.strip():
            continue
        if cards and (continue_next or row.startswith('     ')):
            cards[-1] = cards[-1] + ' ' + row.strip()
        else:
            cards.append(row.strip())
        continue_next = cards[-1].endswith('&')
        if continue_next:
            cards[-1] = cards[-1][:-1]
    return cards


def read_blocks(path: str) -> tuple[str, list[str], list[str], list[str]]:
    """Split a deck into title, cell cards, surface cards and data cards."""
    with open(path, 'r', errors='replace') as f:
        rows = f.read().splitlines()

    # skip an optional message block, ended by a blank line
    start = 0
    first = next(i for i, r in enumerate(rows) if r.strip())
    if rows[first].strip().lower().startswith('message:'):
        start = next(i for i in range(first, len(rows)) if not rows[i].strip()) + 1

    title = rows[start]
    blocks: list[list[str]] = [[]]
    for row in rows[start + 1:]:
        if not row.strip():
            if len(blocks) == 3:
                break
            blocks.append([])
        else:
            blocks[-1].append(row)
    while len(blocks) < 3:
        blocks.append([])

    return title, _join_cards(blocks[0]), _join_cards(blocks[1]), _join_cards(blocks[2])


# ------------------------------------------------------------------------------------------------ transforms

@dataclass(frozen=True)
class Affine:
    """Map from global coordinates to local coordinates, p_local = A p + b."""
    A: tuple
    b: tuple

    @staticmethod
    def identity() -> 'Affine':
        return Affine(tuple(map(tuple, np.identity(3))), (0.0, 0.0, 0.0))

    @property
    def matrix(self) -> np.ndarray:
        m = np.identity(4)
        m[:3, :3] = np.array(self.A)
        m[:3, 3] = np.array(self.b)
        return m

    def then(self, inner: 'Affine') -> 'Affine':
        """Compose: apply self (global to parent), then inner (parent to local)."""
        m = inner.matrix @ self.matrix
        return Affine(tuple(map(tuple, m[:3, :3])), tuple(m[:3, 3]))


def _transform_from_values(values: list[float], degrees: bool) -> Affine:
    """Build the global-to-local map of an MCNP transform (TR / *TR or inline fill)."""
    o = np.array(values[:3], dtype=float)
    rest = list(values[3:])
    m = 1
    if len(rest) in (10, 7, 4, 1):
        m = int(rest[-1])
        rest = rest[:-1]
    if len(rest) == 0:
        rotation = np.identity(3)
    elif len(rest) in (9, 6):
        rotation = np.array(rest, dtype=float).reshape(-1, 3)
        if degrees:
            rotation = np.cos(np.radians(rotation))
        if len(rest) == 6:
            # two vectors given, the third is their cross product
            rotation = np.vstack([rotation, np.cross(rotation[0], rotation[1])])
    else:
        raise MCNPGeometryError(f'transform with {len(rest)} rotation entries is not supported')

    # entries are given as (xx', yx', zx'), (xy', yy', zy'), (xz', yz', zz'): the rows are
    # the local axes x', y', z' in the main system. Orthonormalise (decks round the cosines).
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt

    if m == 1:
        # o is the origin of the local system in the main system: p_local = R (p - o)
        b = -rotation @ o
    else:
        # o is the origin of the main system in the local system: p_local = R p + o
        b = o
    return Affine(tuple(map(tuple, rotation)), tuple(b))


# ------------------------------------------------------------------------------------------------ surfaces

def _quadric(a=0., b=0., c=0., d=0., e=0., f=0., g=0., h=0., j=0., k=0.) -> np.ndarray:
    """4 x 4 symmetric matrix of A x2 + B y2 + C z2 + D xy + E yz + F zx + G x + H y + J z + K."""
    return np.array([[a, d / 2, f / 2, g / 2],
                     [d / 2, b, e / 2, h / 2],
                     [f / 2, e / 2, c, j / 2],
                     [g / 2, h / 2, j / 2, k]], dtype=float)


def _coefficients(q: np.ndarray) -> tuple[float, ...]:
    return (q[0, 0], q[1, 1], q[2, 2], 2 * q[0, 1], 2 * q[1, 2], 2 * q[0, 2], 2 * q[0, 3], 2 * q[1, 3], 2 * q[2, 3], q[3, 3])


def _clean(value: float, scale: float = 1.0) -> float:
    return 0.0 if abs(value) < 1e-12 * max(scale, 1.0) else float(value)


@dataclass
class SurfaceDefinition:
    """A surface card: either a quadric, a torus or a macrobody."""
    number: int
    kind: str                      # 'quadric', 'torus' or 'macrobody'
    local: Affine                  # map from the deck's main system to this surface's system
    quadric: np.ndarray | None = None
    cone: tuple | None = None      # (apex, axis, sheet) for one sheet cones
    torus: tuple | None = None     # (centre, axis, a, b, c)
    facets: list | None = None     # macrobody facets as quadrics in local coordinates


def _three_point_plane(p1, p2, p3) -> np.ndarray:
    """Plane through three points, with the MCNP sense rule: the origin is on the negative side."""
    p1, p2, p3 = map(np.array, (p1, p2, p3))
    n = np.cross(p2 - p1, p3 - p1)
    d = n @ p1
    # MCNP: the positive side is the side away from the origin; if the plane passes through
    # the origin, the side containing (0, 0, inf), then (0, inf, 0), then (inf, 0, 0)
    if abs(d) > 1e-12 * max(1.0, np.linalg.norm(n) * np.linalg.norm(p1)):
        if d < 0:
            n, d = -n, -d
    else:
        for axis in (2, 1, 0):
            if abs(n[axis]) > 1e-12:
                if n[axis] < 0:
                    n, d = -n, -d
                break
    return _quadric(g=n[0], h=n[1], j=n[2], k=-d)


def _axis_vector(letter: str) -> np.ndarray:
    return np.identity(3)['xyz'.index(letter)]


def parse_surface(card: str, transforms: dict[int, Affine]) -> SurfaceDefinition:
    """Parse one surface card."""
    tokens = card.split()
    number = int(tokens[0].lstrip('*+'))
    i = 1
    local = Affine.identity()
    if re.fullmatch(r'-?\d+', tokens[1]):
        tr = int(tokens[1])
        if tr < 0:
            raise MCNPGeometryError(f'surface {number}: periodic surfaces are not supported')
        local = transforms[tr]
        i = 2
    mnemonic = tokens[i].lower()
    v = [float(t) for t in tokens[i + 1:]]

    def quad(q, cone=None):
        return SurfaceDefinition(number, 'quadric', local, quadric=q, cone=cone)

    if mnemonic == 'p':
        if len(v) == 4:
            return quad(_quadric(g=v[0], h=v[1], j=v[2], k=-v[3]))
        if len(v) == 9:
            return quad(_three_point_plane(v[0:3], v[3:6], v[6:9]))
    if mnemonic in ('px', 'py', 'pz'):
        n = _axis_vector(mnemonic[1])
        return quad(_quadric(g=n[0], h=n[1], j=n[2], k=-v[0]))
    if mnemonic == 'so':
        return quad(_quadric(1, 1, 1, k=-v[0] ** 2))
    if mnemonic == 's':
        x, y, z, r = v
        return quad(_quadric(1, 1, 1, g=-2 * x, h=-2 * y, j=-2 * z, k=x * x + y * y + z * z - r * r))
    if mnemonic in ('sx', 'sy', 'sz'):
        c = _axis_vector(mnemonic[1]) * v[0]
        r = v[1]
        return quad(_quadric(1, 1, 1, g=-2 * c[0], h=-2 * c[1], j=-2 * c[2], k=c @ c - r * r))
    if mnemonic in ('c/x', 'c/y', 'c/z', 'cx', 'cy', 'cz'):
        axis = 'xyz'.index(mnemonic[-1])
        if mnemonic.startswith('c/'):
            u0, v0, r = v
        else:
            u0, v0, r = 0.0, 0.0, v[0]
        others = [k for k in range(3) if k != axis]
        diag = [1.0, 1.0, 1.0]
        diag[axis] = 0.0
        lin = [0.0, 0.0, 0.0]
        lin[others[0]] = -2 * u0
        lin[others[1]] = -2 * v0
        return quad(_quadric(*diag, g=lin[0], h=lin[1], j=lin[2], k=u0 * u0 + v0 * v0 - r * r))
    if mnemonic in ('k/x', 'k/y', 'k/z', 'kx', 'ky', 'kz'):
        axis = mnemonic[-1]
        ia = 'xyz'.index(axis)
        if mnemonic.startswith('k/'):
            apex = np.array(v[0:3])
            rest = v[3:]
        else:
            apex = _axis_vector(axis) * v[0]
            rest = v[1:]
        t_sq = rest[0]
        sheet = int(rest[1]) if len(rest) > 1 else 0
        diag = np.ones(3)
        diag[ia] = -t_sq
        lin = -2 * diag * apex
        k = float(np.sum(diag * apex * apex))
        q = _quadric(*diag, g=lin[0], h=lin[1], j=lin[2], k=k)
        return quad(q, cone=(apex, _axis_vector(axis), sheet) if sheet else None)
    if mnemonic in ('tx', 'ty', 'tz'):
        x, y, z, a, b, c = v
        return SurfaceDefinition(number, 'torus', local, torus=(np.array([x, y, z]), _axis_vector(mnemonic[1]), a, b, c))
    if mnemonic == 'sq':
        a, b, c, d, e, f, g, x, y, z = v
        # A(x-x0)^2 + B(y-y0)^2 + C(z-z0)^2 + 2D(x-x0) + 2E(y-y0) + 2F(z-z0) + G = 0
        return quad(_quadric(a, b, c,
                             g=-2 * a * x + 2 * d, h=-2 * b * y + 2 * e, j=-2 * c * z + 2 * f,
                             k=a * x * x + b * y * y + c * z * z - 2 * d * x - 2 * e * y - 2 * f * z + g))
    if mnemonic == 'gq':
        return quad(_quadric(*v))
    if mnemonic == 'rpp':
        x0, x1, y0, y1, z0, z1 = v
        facets = [_quadric(g=1, k=-x1), _quadric(g=-1, k=x0), _quadric(h=1, k=-y1),
                  _quadric(h=-1, k=y0), _quadric(j=1, k=-z1), _quadric(j=-1, k=z0)]
        return SurfaceDefinition(number, 'macrobody', local, facets=facets)
    if mnemonic == 'box':
        if len(v) != 12:
            raise MCNPGeometryError(f'surface {number}: infinite BOX is not supported')
        corner = np.array(v[0:3])
        facets = []
        for a_vec in (np.array(v[3:6]), np.array(v[6:9]), np.array(v[9:12])):
            n = a_vec / np.linalg.norm(a_vec)
            facets.append(_quadric(g=n[0], h=n[1], j=n[2], k=-(n @ (corner + a_vec))))
            facets.append(_quadric(g=-n[0], h=-n[1], j=-n[2], k=(n @ corner)))
        return SurfaceDefinition(number, 'macrobody', local, facets=facets)
    if mnemonic == 'rcc':
        base = np.array(v[0:3])
        hvec = np.array(v[3:6])
        r = v[6]
        n = hvec / np.linalg.norm(hvec)
        # cylinder of radius r about the line base + t n
        m = np.identity(3) - np.outer(n, n)
        lin = -2 * m @ base
        k = base @ m @ base - r * r
        cyl = _quadric(m[0, 0], m[1, 1], m[2, 2], 2 * m[0, 1], 2 * m[1, 2], 2 * m[0, 2], lin[0], lin[1], lin[2], k)
        top = _quadric(g=n[0], h=n[1], j=n[2], k=-(n @ (base + hvec)))
        bottom = _quadric(g=-n[0], h=-n[1], j=-n[2], k=(n @ base))
        return SurfaceDefinition(number, 'macrobody', local, facets=[cyl, top, bottom])

    raise MCNPGeometryError(f'surface {number}: {mnemonic.upper()} with {len(v)} entries is not supported')


def _transform_quadric(q: np.ndarray, local: Affine) -> np.ndarray:
    """Quadric in global coordinates from a quadric in local coordinates."""
    m = local.matrix
    return m.T @ q @ m


def _mcio_from_quadric(q: np.ndarray) -> tuple[S.Surface, float]:
    """Return the simplest mcio surface for a quadric, and the sign of f relative to it."""
    quad = q[:3, :3]
    lin = 2 * q[:3, 3]
    k = q[3, 3]
    scale = max(np.abs(q).max(), 1e-300)

    # planes
    if np.abs(quad).max() < 1e-12 * scale:
        norm = np.linalg.norm(lin)
        n = lin / norm
        d = -k / norm
        for axis, cls, name in ((0, S.XPlane, 'x0'), (1, S.YPlane, 'y0'), (2, S.ZPlane, 'z0')):
            if abs(abs(n[axis]) - 1.0) < 1e-12:
                sign = 1.0 if n[axis] > 0 else -1.0
                return cls(**{name: _clean(d * sign)}), sign
        return S.Plane(a=_clean(n[0]), b=_clean(n[1]), c=_clean(n[2]), d=_clean(d, abs(d))), 1.0

    # normalise so the largest eigenvalue magnitude is 1 (keeping the sign of f)
    eig = np.linalg.eigvalsh(quad)
    lead = eig[np.argmax(np.abs(eig))]
    qn = q / abs(lead)
    quad = qn[:3, :3]
    lin = 2 * qn[:3, 3]
    k = qn[3, 3]
    off_diag = max(abs(quad[0, 1]), abs(quad[1, 2]), abs(quad[0, 2]))
    diag = np.diag(quad)

    if off_diag < 1e-10:
        # sphere
        if np.allclose(diag, diag[0], atol=1e-10) and abs(diag[0]) > 1e-10:
            sign = 1.0 if diag[0] > 0 else -1.0
            c = -lin / (2 * diag[0])
            r_sq = c @ c - k / diag[0]
            if r_sq > 0:
                return S.Sphere(x0=_clean(c[0]), y0=_clean(c[1]), z0=_clean(c[2]), radius=math.sqrt(r_sq)), sign
        # cylinders and cones parallel to an axis
        for axis in range(3):
            others = [i for i in range(3) if i != axis]
            if abs(diag[others[0]] - diag[others[1]]) > 1e-10 or abs(diag[others[0]]) < 1e-10:
                continue
            s = diag[others[0]]
            sign = 1.0 if s > 0 else -1.0
            u0 = -lin[others[0]] / (2 * s)
            v0 = -lin[others[1]] / (2 * s)
            if abs(diag[axis]) < 1e-10 and abs(lin[axis]) < 1e-10:
                r_sq = u0 * u0 + v0 * v0 - k / s
                if r_sq > 0:
                    centre = [0.0, 0.0, 0.0]
                    centre[others[0]], centre[others[1]] = u0, v0
                    cls = (S.CylinderParallelToXAxis, S.CylinderParallelToYAxis, S.CylinderParallelToZAxis)[axis]
                    kwargs = {('x0', 'y0', 'z0')[i]: _clean(centre[i]) for i in others}
                    return cls(radius=math.sqrt(r_sq), **kwargs), sign
            elif diag[axis] / s < -1e-12:
                t_sq = -diag[axis] / s
                w0 = -lin[axis] / (2 * diag[axis])
                apex = [0.0, 0.0, 0.0]
                apex[others[0]], apex[others[1]], apex[axis] = u0, v0, w0
                residual = k / s - (u0 * u0 + v0 * v0 - t_sq * w0 * w0)
                if abs(residual) < 1e-8 * max(1.0, u0 * u0 + v0 * v0 + t_sq * w0 * w0):
                    cls = (S.ConeParallelToXAxis, S.ConeParallelToYAxis, S.ConeParallelToZAxis)[axis]
                    return cls(x0=_clean(apex[0]), y0=_clean(apex[1]), z0=_clean(apex[2]), t_sq=t_sq, sheet=0), sign

    a, b, c, d, e, f, g, h, j, kk = _coefficients(qn)
    return S.GeneralQuadratic(a=a, b=b, c=c, d=d, e=e, f=f, g=g, h=h, j=j, k=kk), 1.0


# ------------------------------------------------------------------------------------------------ booleans

def _boolean(x):
    return x.boolean if isinstance(x, Volume) else x


def _and(items: list) -> Volume:
    if len(items) == 1:
        return items[0] if isinstance(items[0], Volume) else Volume(items[0])
    return Volume(And([_boolean(i) for i in items]))


def _or(items: list) -> Volume:
    if len(items) == 1:
        return items[0] if isinstance(items[0], Volume) else Volume(items[0])
    return Volume(Or([_boolean(i) for i in items]))


class SurfaceFactory:
    """Creates (and shares) mcio surfaces for surface definitions under a placement."""

    def __init__(self) -> None:
        self._cache: dict[tuple, Any] = {}

    def _quadric_half_volume(self, q_global: np.ndarray, sense: str) -> HalfVolume:
        key = ('q', tuple(np.round(q_global, 12).ravel()))
        if key not in self._cache:
            self._cache[key] = _mcio_from_quadric(q_global)
        surface, sign = self._cache[key]
        if sign < 0:
            sense = '+' if sense == '-' else '-'
        return +surface if sense == '+' else -surface

    def half_volume(self, definition: SurfaceDefinition, placement: Affine, sense: str, facet: int | None = None):
        """mcio half volume / volume for ``sense`` of a surface, placed by ``placement``."""
        local = placement.then(definition.local)

        if definition.kind == 'macrobody':
            facets = [_transform_quadric(q, local) for q in definition.facets]
            if facet is not None:
                return self._quadric_half_volume(facets[facet - 1], sense)
            inside = _and([self._quadric_half_volume(f, '-') for f in facets])
            return inside if sense == '-' else ~inside

        if facet is not None:
            raise MCNPGeometryError(f'surface {definition.number} is not a macrobody, facet {facet} used')

        if definition.kind == 'torus':
            centre, axis, a, b, c = definition.torus
            inv = np.linalg.inv(local.matrix)
            centre_g = (inv @ np.append(centre, 1.0))[:3]
            axis_g = inv[:3, :3] @ axis
            idx = int(np.argmax(np.abs(axis_g)))
            if abs(abs(axis_g[idx]) - 1.0) > 1e-9:
                raise MCNPGeometryError(f'surface {definition.number}: torus axis is not parallel to a cartesian axis after transforming')
            cls = (S.ToriParralleltoXAxis, S.ToriParralleltoYAxis, S.ToriParralleltoZAxis)[idx]
            key = ('t', idx, *np.round(centre_g, 10), a, b, c)
            if key not in self._cache:
                self._cache[key] = cls(x0=_clean(centre_g[0]), y0=_clean(centre_g[1]), z0=_clean(centre_g[2]), a=a, b=b, c=c)
            surface = self._cache[key]
            return +surface if sense == '+' else -surface

        q_global = _transform_quadric(definition.quadric, local)
        if definition.cone is None:
            return self._quadric_half_volume(q_global, sense)

        # one sheet cone: the inside is the inside of the double cone on the sheet side of the apex
        apex, axis, sheet = definition.cone
        plane = _quadric(g=sheet * axis[0], h=sheet * axis[1], j=sheet * axis[2], k=-sheet * (axis @ apex))
        inside = _and([self._quadric_half_volume(q_global, '-'), self._quadric_half_volume(_transform_quadric(plane, local), '+')])
        return inside if sense == '-' else ~inside


_TOKEN = re.compile(r'\s*(#\s*\(|#\s*\d+|\(|\)|:|[+-]?\d+(?:\.\d+)?)')


def tokenise_geometry(text: str) -> list[str]:
    """Split a cell geometry description into tokens."""
    tokens = []
    pos = 0
    text = text.strip()
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if not match:
            raise MCNPGeometryError(f'cannot parse geometry at "{text[pos:pos + 30]}"')
        tokens.append(match.group(1).replace(' ', ''))
        pos = match.end()
        while pos < len(text) and text[pos] == ' ':
            pos += 1
    return tokens


# ------------------------------------------------------------------------------------------------ cells

_PARAM = re.compile(r'(?:^|\s)(\*?[a-zA-Z][a-zA-Z:,]*)\s*=', re.I)


@dataclass
class CellDefinition:
    number: int
    material: int
    density: float | None
    geometry: str
    universe: int = 0
    fill: int | None = None
    fill_transform: Affine | None = None
    importance: float = 1.0
    index: int = 0


def parse_cell(card: str, transforms: dict[int, Affine]) -> CellDefinition:
    """Parse one cell card."""
    tokens = card.split()
    number = int(tokens[0])
    if tokens[1].lower() == 'like':
        raise MCNPGeometryError(f'cell {number}: LIKE n BUT is not supported')
    header = re.match(r'\s*(\d+)\s+(\d+)\s*', card)
    material = int(header.group(2))
    rest = card[header.end():]
    density = None
    if material:
        dm = re.match(r'([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)', rest)
        density = float(dm.group(1))
        rest = rest[dm.end():]

    match = _PARAM.search(rest)
    geometry = rest[:match.start()] if match else rest
    params_text = rest[match.start():] if match else ''

    cell = CellDefinition(number, material, density, geometry.strip())

    # parameters, keyword=value (fill may have a transform in brackets)
    for m in re.finditer(r'(\*?[a-zA-Z][a-zA-Z:,]*)\s*=\s*([^=]*?)(?=\s+\*?[a-zA-Z][a-zA-Z:,]*\s*=|$)', params_text.strip()):
        key, value = m.group(1).lower(), m.group(2).strip()
        if key == 'u':
            cell.universe = int(value.split()[0])
        elif key in ('fill', '*fill'):
            fm = re.fullmatch(r'(-?\d+)\s*(?:\((.*)\))?', value)
            if not fm:
                raise MCNPGeometryError(f'cell {number}: fill "{value}" (lattice fills) is not supported')
            cell.fill = int(fm.group(1))
            if fm.group(2):
                values = [float(t) for t in fm.group(2).split()]
                if len(values) == 1:
                    cell.fill_transform = transforms[int(values[0])]
                else:
                    cell.fill_transform = _transform_from_values(values, degrees=key.startswith('*'))
        elif key.startswith('imp:n'):
            cell.importance = float(value.split()[0])
        elif key == 'lat':
            raise MCNPGeometryError(f'cell {number}: lattices are not supported')
        elif key in ('trcl', '*trcl'):
            raise MCNPGeometryError(f'cell {number}: TRCL is not supported')
    return cell


@dataclass
class LeafCell:
    """A cell of the deck as placed in the global system (after expanding fills)."""
    path: tuple[int, ...]
    material: int
    density: float | None
    volume: Volume


class MCNPGeometry:
    """Geometry of an MCNP deck.

    Parameters
    ----------
    path : str
        Path of the MCNP input deck.

    Attributes
    ----------
    cells, surfaces, transforms : dict
        Parsed cards keyed by number.
    cell_errors, surface_errors : dict
        Cards that could not be read, with the reason.
    """

    def __init__(self, path: str) -> None:
        self.title, cell_cards, surface_cards, data_cards = read_blocks(path)

        self.transforms: dict[int, Affine] = {}
        importances: list[float] | None = None
        for card in data_cards:
            tokens = card.split()
            name = tokens[0].lower()
            tm = re.fullmatch(r'(\*?)tr(\d+)', name)
            if tm:
                values = [float(t) for t in _expand_repeats(tokens[1:])]
                self.transforms[int(tm.group(2))] = _transform_from_values(values, degrees=bool(tm.group(1)))
            elif name.startswith('imp:n'):
                importances = [float(t) for t in _expand_repeats(tokens[1:])]

        self.surfaces: dict[int, SurfaceDefinition] = {}
        self.surface_errors: dict[int, str] = {}
        for card in surface_cards:
            try:
                definition = parse_surface(card, self.transforms)
                self.surfaces[definition.number] = definition
            except (MCNPGeometryError, KeyError, ValueError) as error:
                self.surface_errors[int(card.split()[0].lstrip('*+'))] = str(error)

        self.cells: dict[int, CellDefinition] = {}
        self.cell_errors: dict[int, str] = {}
        for index, card in enumerate(cell_cards):
            try:
                cell = parse_cell(card, self.transforms)
                cell.index = index
                self.cells[cell.number] = cell
            except (MCNPGeometryError, KeyError, ValueError) as error:
                self.cell_errors[int(card.split()[0])] = str(error)

        if importances is not None:
            for cell in self.cells.values():
                if cell.index < len(importances):
                    cell.importance = importances[cell.index]

        self.universes: dict[int, list[CellDefinition]] = {}
        for cell in self.cells.values():
            self.universes.setdefault(cell.universe, []).append(cell)

        self.factory = SurfaceFactory()

    def cell_volume(self, cell: CellDefinition, placement: Affine) -> Volume:
        """mcio volume of a cell's own geometry, placed by ``placement``."""
        tokens = tokenise_geometry(cell.geometry)
        position = 0

        def parse_union():
            nonlocal position
            terms = [parse_intersection()]
            while position < len(tokens) and tokens[position] == ':':
                position += 1
                terms.append(parse_intersection())
            return _or(terms)

        def parse_intersection():
            nonlocal position
            factors = []
            while position < len(tokens) and tokens[position] not in (':', ')'):
                factors.append(parse_factor())
            if not factors:
                raise MCNPGeometryError(f'cell {cell.number}: empty intersection')
            return _and(factors)

        def parse_factor():
            nonlocal position
            token = tokens[position]
            position += 1
            if token == '(':
                volume = parse_union()
                position += 1  # ')'
                return volume
            if token == '#(':
                volume = parse_union()
                position += 1
                return ~volume
            if token.startswith('#'):
                other = self.cells[int(token[1:])]
                return ~self.cell_volume(other, placement)
            sense = '-' if token.startswith('-') else '+'
            number_text = token.lstrip('+-')
            facet = None
            if '.' in number_text:
                number_text, facet_text = number_text.split('.')
                facet = int(facet_text)
            number = int(number_text)
            if number not in self.surfaces:
                raise MCNPGeometryError(f'cell {cell.number}: surface {number} ' +
                                        self.surface_errors.get(number, 'not defined'))
            return self.factory.half_volume(self.surfaces[number], placement, sense, facet)

        volume = parse_union()
        if position != len(tokens):
            raise MCNPGeometryError(f'cell {cell.number}: unbalanced parentheses')
        return volume

    def leaf_cells(self, include_outside: bool = False) -> tuple[list[LeafCell], dict[tuple, str]]:
        """Expand fills and return the cells placed in the global system, with any errors."""
        leaves: list[LeafCell] = []
        errors: dict[tuple, str] = {}

        def visit(cell: CellDefinition, placement: Affine, ancestors: list[Volume], path: tuple[int, ...]):
            path = path + (cell.number,)
            try:
                region = self.cell_volume(cell, placement)
            except (MCNPGeometryError, KeyError) as error:
                errors[path] = str(error)
                return
            if cell.fill is not None:
                inner = placement.then(cell.fill_transform) if cell.fill_transform else placement
                for child in self.universes.get(cell.fill, []):
                    visit(child, inner, ancestors + [region], path)
            else:
                leaves.append(LeafCell(path, cell.material, cell.density, _and(ancestors + [region])))

        for cell in self.universes.get(0, []):
            if cell.importance == 0 and not include_outside:
                continue
            visit(cell, Affine.identity(), [], ())

        for number, message in self.cell_errors.items():
            errors[(number,)] = message
        return leaves, errors
