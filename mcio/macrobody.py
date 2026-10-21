"""MCNP macrobody definitions.

Contains the :class:`Macrobody` base class and the BOX, RPP and RCC
macrobodies, which can be written as MCNP macrobody cards or expanded into
volumes of primitive surfaces.
"""
from __future__ import annotations

import math

from typing import Any, TYPE_CHECKING

from mcio import logger
from .formatting import formatting
# from .formatting import MCNP_DP, MCNP_INDENT
from .mcnp import MCNPCard
from .volume import HalfVolume
from .transform import Transform
from .surface import (
    XPlane, YPlane, ZPlane,
    CylinderParallelToZAxis
)

if TYPE_CHECKING:
    from .volume import Volume


class Macrobody:
    """Base class for MCNP macrobodies.

    Parameters
    ----------
    comment : str, optional
        Descriptive text associated with the macrobody, by default None
    transform : Transform, optional
        Transform applied to the macrobody, by default None
    """

    def __init__(self, comment: str | None=None, transform: Transform | None=None) -> None:
        """Initialise the Macrobody base class."""
        self._comment: str | None = comment
        self._transform: Transform | None = transform
        self._number: int | None = None

    @property
    def transform(self) -> Transform | None:
        """Transform applied to the macrobody."""
        return self._transform

    @property
    def number(self) -> int | None:
        """Surface number assigned to the macrobody."""
        return self._number

    @number.setter
    def number(self, number: int | None) -> None:
        """Set the surface number of the macrobody.

        Parameters
        ----------
        number : int or None
            Surface number.
        """
        self._number = number

    def __pos__(self) -> HalfVolume:
        """Return the positive (outside) half volume of the macrobody.

        Returns
        -------
        HalfVolume
            Half volume with direction ``'+'``.
        """
        return HalfVolume('+', self)

    def __neg__(self) -> HalfVolume:
        """Return the negative (inside) half volume of the macrobody.

        Returns
        -------
        HalfVolume
            Half volume with direction ``'-'``.
        """
        return HalfVolume('-', self)

def mcnp_macrobody(macrobody_type: str, parameters: list[float],
                   macrobody_number: int | None,
                   transform: Transform | None) -> MCNPCard:
    """Build an MCNP card for a macrobody.

    Parameters
    ----------
    macrobody_type : str
        MCNP macrobody mnemonic, e.g. ``'RPP'``.
    parameters : list[float]
        Numeric macrobody parameters, written to ``formatting.MCNP_DP`` decimal
        places.
    macrobody_number : int or None
        Surface number of the macrobody.
    transform : Transform or None
        Transform whose number is prepended to the card, if given.

    Returns
    -------
    MCNPCard
        The macrobody surface card.
    """
    rows = [[f'{macrobody_type:<{formatting.MCNP_INDENT}}']]

    for parameter in parameters:
        rows[0].append(f'{parameter:.{formatting.MCNP_DP}f}')

    if transform:
        rows[0].insert(0, str(transform.number))

    return MCNPCard(macrobody_number, rows)


class Box(Macrobody):
    """Arbitrarily oriented orthogonal box (MCNP BOX macrobody).

    Parameters
    ----------
    vx : float, optional
        x coordinate of the corner vertex, by default 0.0
    vy : float, optional
        y coordinate of the corner vertex, by default 0.0
    vz : float, optional
        z coordinate of the corner vertex, by default 0.0
    a1x : float, optional
        x component of the first edge vector, by default 0.0
    a1y : float, optional
        y component of the first edge vector, by default 0.0
    a1z : float, optional
        z component of the first edge vector, by default 0.0
    a2x : float, optional
        x component of the second edge vector, by default 0.0
    a2y : float, optional
        y component of the second edge vector, by default 0.0
    a2z : float, optional
        z component of the second edge vector, by default 0.0
    a3x : float, optional
        x component of the third edge vector, by default 0.0
    a3y : float, optional
        y component of the third edge vector, by default 0.0
    a3z : float, optional
        z component of the third edge vector, by default 0.0
    **kwargs : Any
        Passed to :class:`Macrobody` (``comment``, ``transform``).
    """

    def __init__(self, vx: float=0.0, vy: float=0.0, vz: float=0.0, a1x: float=0.0, a1y: float=0.0, a1z: float=0.0,
                       a2x: float=0.0, a2y: float=0.0, a2z: float=0.0, a3x: float=0.0, a3y: float=0.0, a3z: float=0.0,
                       **kwargs: Any) -> None:
        """Initialise the Box macrobody."""

        super().__init__(**kwargs)

        self._vx: float = vx
        self._vy: float = vy
        self._vz: float = vz
        self._a1x: float = a1x
        self._a1y: float = a1y
        self._a1z: float = a1z
        self._a2x: float = a2x
        self._a2y: float = a2y
        self._a2z: float = a2z
        self._a3x: float = a3x
        self._a3y: float = a3y
        self._a3z: float = a3z

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP BOX surface card for the macrobody."""
        return mcnp_macrobody('BOX',
                              [self._vx, self._vy, self._vz,
                               self._a1x, self._a1y, self._a1z,
                               self._a2x, self._a2y, self._a2z,
                               self._a3x, self._x3y, self._a3z],
                               self._number, self._transform)

    def ocp_shape(self) -> None:
        """Placeholder for building an Open CASCADE shape (not implemented)."""
        pass


class RectangularParallelepiped(Macrobody):
    """Axis-aligned rectangular parallelepiped (MCNP RPP macrobody).

    Parameters
    ----------
    x_min : float, optional
        Minimum x extent, by default 0.0
    x_max : float, optional
        Maximum x extent, by default 0.0
    y_min : float, optional
        Minimum y extent, by default 0.0
    y_max : float, optional
        Maximum y extent, by default 0.0
    z_min : float, optional
        Minimum z extent, by default 0.0
    z_max : float, optional
        Maximum z extent, by default 0.0
    **kwargs : Any
        Passed to :class:`Macrobody` (``comment``, ``transform``).
    """

    def __init__(self, x_min: float=0.0, x_max: float=0.0, y_min: float=0.0,
                 y_max: float=0.0, z_min: float=0.0, z_max: float=0.0, **kwargs: Any) -> None:
        """Initialise the RectangularParallelepiped macrobody."""

        super().__init__(**kwargs)

        self._x_min: float = x_min
        self._x_max: float = x_max
        self._y_min: float = y_min
        self._y_max: float = y_max
        self._z_min: float = z_min
        self._z_max: float = z_max

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP RPP surface card for the macrobody."""
        return mcnp_macrobody('RPP',
                              [self._x_min, self._x_max, self._y_min,
                               self._y_max, self._z_min, self._z_max],
                               self._number, self._transform)

    def as_half_volumes(self) -> list[HalfVolume]:
        """Return the six plane half volumes whose intersection is the box.

        Returns
        -------
        list[HalfVolume]
            Half volumes for the x, y and z min/max planes.
        """
        # set up list of half volumes
        half_volumes = [
            HalfVolume('+', XPlane(x0=self._x_min, transform=self._transform)),
            HalfVolume('-', XPlane(x0=self._x_max, transform=self._transform)),
            HalfVolume('+', YPlane(y0=self._y_min, transform=self._transform)),
            HalfVolume('-', YPlane(y0=self._y_max, transform=self._transform)),
            HalfVolume('+', ZPlane(z0=self._z_min, transform=self._transform)),
            HalfVolume('-', ZPlane(z0=self._z_max, transform=self._transform))
        ]

        return half_volumes

    def as_volume_of_surfaces(self, invert: bool=False) -> Volume:
        """Return the interior of the box as a volume of six planes.

        Parameters
        ----------
        invert : bool, optional
            Unused, by default False

        Returns
        -------
        Volume
            Intersection of the six plane half volumes.
        """
        vol = +XPlane(x0=self._x_min, transform=self._transform) & \
              -XPlane(x0=self._x_max, transform=self._transform) & \
              +YPlane(y0=self._y_min, transform=self._transform) & \
              -YPlane(y0=self._y_max, transform=self._transform) & \
              +ZPlane(z0=self._z_min, transform=self._transform) & \
              -ZPlane(z0=self._z_max, transform=self._transform)

        return vol

        # HalfVolume('+', XPlane(x0=self._x_min, transform=self._transform)),
        # HalfVolume('-', XPlane(x0=self._x_max, transform=self._transform)),
        # HalfVolume('+', YPlane(y0=self._y_min, transform=self._transform)),
        # HalfVolume('-', YPlane(y0=self._y_max, transform=self._transform)),
        # HalfVolume('+', ZPlane(z0=self._z_min, transform=self._transform)),
        # HalfVolume('-', ZPlane(z0=self._z_max, transform=self._transform))

class RightCircularCylinder(Macrobody):
    """Right circular cylinder (MCNP RCC macrobody).

    Parameters
    ----------
    x0 : float, optional
        x coordinate of the base centre, by default 0.0
    y0 : float, optional
        y coordinate of the base centre, by default 0.0
    z0 : float, optional
        z coordinate of the base centre, by default 0.0
    h_x : float, optional
        x component of the height vector, by default 0.0
    h_y : float, optional
        y component of the height vector, by default 0.0
    h_z : float, optional
        z component of the height vector, by default 0.0
    radius : float, optional
        Cylinder radius, by default 0.0
    **kwargs : Any
        Passed to :class:`Macrobody` (``comment``, ``transform``).
    """

    def __init__(self, x0: float=0.0, y0: float=0.0, z0: float=0.0, h_x: float=0.0, h_y: float=0.0, h_z: float=0.0, radius: float=0.0, **kwargs: Any) -> None:
        """Initialise the RightCircularCylinder macrobody."""

        super().__init__(**kwargs)

        self._x0: float = x0
        self._y0: float = y0
        self._z0: float = z0
        self._h_x: float = h_x
        self._h_y: float = h_y
        self._h_z: float = h_z
        self._radius: float = radius

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP RCC surface card for the macrobody."""
        params = [self._x0, self._y0, self._z0, self._h_x, self._h_y, self._h_z, self._radius]
        return mcnp_macrobody('RCC', params, self._number, self._transform)

    def as_volume_of_surfaces(self, invert: bool=False) -> Volume:
        """Return the interior of the cylinder as a volume of primitive surfaces.

        A local frame is built with its z axis along the height vector and its
        origin at the base centre; the volume is the intersection of two
        z-planes and a z-axis cylinder carrying that frame as a transform.

        Parameters
        ----------
        invert : bool, optional
            Unused, by default False

        Returns
        -------
        Volume
            Volume of the cylinder interior.

        Raises
        ------
        ValueError
            If the height vector has zero length.
        """
        # calculate the height
        height = math.sqrt(self._h_x**2 + self._h_y**2 + self._h_z**2)

        if height == 0:
            msg = 'Height vector (h_x, h_y, h_z) cannot be zero.'
            logger.fatal(msg)
            raise ValueError(msg)

        # Local Z-axis unit vector in global coordinates
        uz = (self._h_x / height, self._h_y / height, self._h_z / height)

        # Build an orthogonal coordinate frame (local X, Y, Z axes)
        if abs(uz[0]) < 0.9:
            ref = (1.0, 0.0, 0.0)
        else:
            ref = (0.0, 1.0, 0.0)

        # Local X unit vector = ref x uz
        ux_raw = (
            ref[1] * uz[2] - ref[2] * uz[1],
            ref[2] * uz[0] - ref[0] * uz[2],
            ref[0] * uz[1] - ref[1] * uz[0]
        )
        ux_len = math.sqrt(sum(k**2 for k in ux_raw))
        ux = tuple(k / ux_len for k in ux_raw)

        # Local Y unit vector = uz x ux
        uy = (
            uz[1] * ux[2] - uz[2] * ux[1],
            uz[2] * ux[0] - uz[0] * ux[2],
            uz[0] * ux[1] - uz[1] * ux[0]
        )

        # Rows must be the local axes (X', Y', Z') expressed in global coordinates (X, Y, Z)
        matrix = [
            [ux[0], ux[1], ux[2]],  # x' relative to global X, Y, Z
            [uy[0], uy[1], uy[2]],  # y' relative to global X, Y, Z
            [uz[0], uz[1], uz[2]]   # z' relative to global X, Y, Z
        ]

        lateral = (self._x0, self._y0, self._z0)

        transform = Transform(
            lateral=lateral,
            rotation_matrix=matrix,
            angle_type='cosine'
        )

        # On-axis primitive surfaces in local space (Z-aligned, origin at base)
        bottom_plane = ZPlane(z0=0.0, transform=transform)
        top_plane = ZPlane(z0=height, transform=transform)
        cylinder = CylinderParallelToZAxis(radius=self._radius, transform=transform)

        # produce the volume for the internal space within the macrobody
        vol = +bottom_plane & -top_plane & -cylinder

        # return volume
        return vol

        # # if required, invert the individual half spaces.
        # if not invert:
        #     return [
        #         HalfVolume('-', bottom_plane),  # Inside cylinder is above bottom cap (z > 0)
        #         HalfVolume('+', top_plane),     # Inside cylinder is below top cap (z < height)
        #         HalfVolume('+', cylinder)       # CORRECTED: Inside cylinder is r < R (negative sense)
        #     ]
        # else:
        #     return [
        #         HalfVolume('+', bottom_plane),  # Inside cylinder is above bottom cap (z > 0)
        #         HalfVolume('-', top_plane),     # Inside cylinder is below top cap (z < height)
        #         HalfVolume('-', cylinder)       # CORRECTED: Inside cylinder is r < R (negative sense)
        #     ]
