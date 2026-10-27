"""Transforms which can be applied to surfaces and cells.

Provides the :class:`Transform` class, which can write an MCNP ``TR`` card and
build the equivalent OpenCASCADE ``gp_Trsf``.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import math

try:
    import casadi
    import OCP

    from OCP.gp import (
        gp_Ax3,
        gp_Dir,
        gp_Pnt,
        gp_Trsf
    )
except:
    pass

from .formatting import formatting
from .mcnp import MCNPCard


class Transform():
    """Lateral displacement and rotation matrix applied to surfaces and cells.

    The data maps onto an MCNP transformation card::

        TRCL = o_1, o_2, o_3,
               xx', yx', zx',
               xy', yy', zy',
               xz', yz', zz',
               m

    Parameters
    ----------
    lateral : Sequence[float], optional
        Displacement vector (o_1, o_2, o_3), by default (0, 0, 0)
    rotation_matrix : Sequence[Sequence[float]], optional
        3x3 rotation matrix given as angles (degrees) or cosines depending on
        ``angle_type``, by default the identity in degrees
        ((0, 90, 90), (90, 0, 90), (90, 90, 0))
    angle_type : str, optional
        Either 'degrees' or 'cosine', by default 'degrees'
    displacement_vector_at_origin : bool, optional
        Selects the MCNP ``m`` flag; True writes ``m = -1`` and False writes
        ``m = 1``, by default False
    """
    def __init__(
            self, 
            lateral: Sequence[float]=(0,0,0), 
            rotation_matrix: Sequence[Sequence[float]]=((0,90,90),(90,0,90),(90,90,0)),
            angle_type: str='degrees',
            displacement_vector_at_origin: bool=False) -> None:
        """Initialise the Transform."""

        # store user defined parameters
        self.number: int | None = None
        self.lateral: Sequence[float] = lateral
        self.rotation_matrix: Sequence[Sequence[float]] = rotation_matrix
        self.angle_type: str = angle_type
        self.displacement_vector_at_origin: bool = displacement_vector_at_origin

    @property
    def mcnp_card(self) -> MCNPCard:
        """Build the MCNP transformation card.

        ``TR`` is written when ``angle_type`` is 'cosine' and ``*TR`` when it is
        'degrees'.

        Returns
        -------
        MCNPCard
            The ``TR``/``*TR`` card for this transform.

        Raises
        ------
        Exception
            If ``angle_type`` is neither 'degrees' nor 'cosine'.
        """
        lateral_str = [f'{i:{formatting.MCNP_FORMAT_TRANSFORM_LATTERAL}}' for i in self.lateral]
        matrix_str = [[f'{i:{formatting.MCNP_FORMAT_TRANSFORM_ROTATION}}' for i in row] for row in self.rotation_matrix]

        matrix_data = {'latteral': ' '.join(lateral_str),
                       'xx\' yx\' zx\'': ' '.join(matrix_str[0]),
                       'xy\' yy\' yz\'': ' '.join(matrix_str[1]),
                       'xz\' yz\' zz\'': ' '.join(matrix_str[2])}

        # set the string for the coordinate system
        m_string = '-1' if self.displacement_vector_at_origin else '1'
        
        if self.angle_type == 'cosine':
            return MCNPCard(f'TR{self.number}', [lateral_str, *matrix_str, [m_string]], header_data=matrix_data)
        elif self.angle_type == 'degrees':
            return MCNPCard(f'*TR{self.number}', [lateral_str, *matrix_str, [m_string]], header_data=matrix_data)
        else:
            msg = f'angle_type and only be either degrees or cosine, not {self.angle_type}'
            raise Exception(msg)

    @property
    def ocp_transform(self) -> "gp_Trsf":
        """Convert the MCNP direction angles (TR card) to an active OpenCASCADE gp_Trsf.

        Returns
        -------
        gp_Trsf
            Transformation from the auxiliary (transformed) coordinate system
            to the main system, following the MCNP TR card definition for m = 1
            (``displacement_vector_at_origin`` False) and m = -1 (True).

        Raises
        ------
        ValueError
            If ``angle_type`` is neither 'degrees' nor 'cosine'.
        """
        
        # convert MCNP angles / cosines into direction cosines
        rotation = np.array(self.rotation_matrix, dtype=float)
        if self.angle_type == 'degrees':
            rotation = np.cos(np.radians(rotation))
        elif self.angle_type != 'cosine':
            raise ValueError(f"angle_type must be 'degrees' or 'cosine'")

        # rows are the auxiliary axes x', y', z' in the main system; orthonormalise as
        # decks often give rounded cosines
        u, _, vt = np.linalg.svd(rotation)
        rotation = u @ vt

        lateral = np.array(self.lateral, dtype=float)
        if self.displacement_vector_at_origin:
            # m = -1: the displacement is the origin of the main system in the auxiliary
            # system, x' = R x + o, so x = R^T (x' - o)
            translation = -rotation.T @ lateral
        else:
            # m = 1: the displacement is the origin of the auxiliary system in the main
            # system, x = R^T x' + o
            translation = lateral

        # transformation from the auxiliary (local) system to the main system
        m = rotation.T
        trsf = gp_Trsf()
        trsf.SetValues(m[0, 0], m[0, 1], m[0, 2], translation[0],
                       m[1, 0], m[1, 1], m[1, 2], translation[1],
                       m[2, 0], m[2, 1], m[2, 2], translation[2])

        return trsf
