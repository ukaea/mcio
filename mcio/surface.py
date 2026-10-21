"""Surface classes used to build CSG half-spaces.

Defines the :class:`Surface` base class and the concrete MCNP-style surface types
(planes, spheres, cylinders, cones, quadrics and tori), together with helpers to
write them as MCNP, Serpent and OpenMC surfaces, sort them, and map MCNP surface
mnemonics to classes.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Type, Union

from .formatting import formatting
from .mcnp import MCNPCard
from .serpent2 import SerpentCard
from .transform import Transform
from .volume import HalfVolume

try:
    import openmc
except ModuleNotFoundError:
    pass


class Surface:
    """Base class for all surface types.

    Parameters
    ----------
    reflective : bool, optional
        Whether the surface is reflective, by default False
    comment : str, optional
        Optional descriptive text associated with the surface, by default None
    transform : Transform, optional
        Transform object to be applied to the surface, by default None
    openmc_graveyard : bool, optional
        Flag indicating if the surface belongs to an OpenMC graveyard, by default False
    number : int, optional
        Surface number assigned, by default None
    """

    def __init__(self, 
                 reflective: bool = False, 
                 comment: str | None = None, 
                 transform: Transform | None = None, 
                 openmc_graveyard: bool = False,
                 number: int | None = None) -> None:
        """Initialize the Surface base class.

        Parameters
        ----------
        reflective : bool, optional
            Whether the surface is reflective, by default False
        comment : str, optional
            Optional descriptive text associated with the surface, by default None
        transform : Transform, optional
            Transform object to be applied to the surface, by default None
        openmc_graveyard : bool, optional
            Flag indicating if the surface belongs to an OpenMC graveyard, by default False
        number : int, optional
            Surface number assigned, by default None
        """
        self.reflective: bool = reflective
        self.number: int | None = number
        self.comment: str | None = comment
        self.area: float | None = None
        self.transform: Transform | None = transform
        self.type: str | None = None
        self.openmc_graveyard: bool = openmc_graveyard
        
        self._openmc_surface: Any = None

    def _mcnp_format_card_type(self, card_type: str) -> str:
        """Format the MCNP surface card type string.

        Parameters
        ----------
        card_type : str
            A string denoting the MCNP card type

        Returns
        -------
        str
            MCNP card type string appended with transform number if present
        """
        if self.transform is not None:
            return card_type + f'{self.transform.number}'
        else:
            return card_type

    @classmethod
    def from_mcnp_card(cls, mcnp_card: str) -> "Surface":
        """Construct a Surface object from an MCNP card string.

        Parameters
        ----------
        mcnp_card : str
            Raw MCNP surface card string

        Returns
        -------
        Surface
            Constructed surface instance
        """
        regex_surf = re.compile(
            r'(?:\s*)(?P<surf_number>([0-9]+))(?:\s+)(?P<transform>[0-9]*)(?:\s*)(?P<surf_type>[a-zA-Z/]+)(?P<params>.*)'
        )

        groups_surf = re.search(regex_surf, mcnp_card).groupdict()

        surf_number = int(groups_surf['surf_number'])
        surf_type = groups_surf['surf_type']
        surf_params = [float(v) for v in groups_surf['params'].split()]

        if surf_type.lower() == 'p' and len(surf_params) == 9:
            surf_type = 'P3P'

        surface = MCNP_SURFACE_TYPES[surf_type.upper()](*surf_params)
        surface.number = surf_number

        print('=' * 100)
        print(mcnp_card)
        print(f' -> number {surf_number}')
        print(f' -> type    {surf_type}')
        print(f' -> params  {surf_params}')

        return surface

    def __pos__(self) -> HalfVolume:
        """Return the positive half-space of the surface.

        Returns
        -------
        HalfVolume
            Positive half-volume associated with the surface
        """
        return HalfVolume('+', self)
        
    def __neg__(self) -> HalfVolume:
        """Return the negative half-space of the surface.

        Returns
        -------
        HalfVolume
            Negative half-volume associated with the surface
        """
        return HalfVolume('-', self)

    def __str__(self) -> str:
        """Return string representation of the MCNP card.

        Returns
        -------
        str
            MCNP card string representation
        """
        return self.mcnp_card.as_string


def mcnp_surface(surface_type: str, parameters: List[str], surface: Surface) -> MCNPCard:
    """Construct an MCNPCard representation for a surface.

    Parameters
    ----------
    surface_type : str
        The MCNP surface type designator string
    parameters : list of str
        Parameters formatted as strings for the MCNP surface
    surface : Surface
        The surface object being converted

    Returns
    -------
    MCNPCard
        Constructed MCNP surface card
    """
    rows = [[f'{surface_type:<{formatting.MCNP_INDENT}}']]
    rows[0].extend(parameters)

    if surface.transform:
        rows[0].insert(0, str(surface.transform.number))

    if surface.reflective:
        card_designator = f'*{surface.number}'
    else:
        card_designator = f'{surface.number}'

    return MCNPCard(card_designator, rows)


def serpent_surface(surface_type: str, parameters: List[str], surface: Surface) -> SerpentCard:
    """Construct a SerpentCard representation for a surface.

    Parameters
    ----------
    surface_type : str
        The Serpent surface type designator string
    parameters : list of str
        Parameters formatted as strings for the Serpent surface
    surface : Surface
        The surface object being converted

    Returns
    -------
    SerpentCard
        Constructed Serpent surface card
    """
    rows = [parameters]
    card_designator = f'surf {surface.number} {surface_type:<{formatting.SERPENT_INDENT}}'
    return SerpentCard(card_designator, rows)


def openmc_boundary_type(surface: Surface) -> str:
    """Determine the OpenMC boundary type for a surface.

    Parameters
    ----------
    surface : Surface
        Surface object to check

    Returns
    -------
    str
        OpenMC boundary type string ('vacuum', 'reflective', or 'transmission')
    """
    if surface.openmc_graveyard:
        return 'vacuum'
    elif surface.reflective:
        return 'reflective'
    else:
        return 'transmission'


class Plane(Surface):
    """General plane surface defined by the equation ax + by + cz = d.

    Parameters
    ----------
    a : float, optional
        Coefficient of x, by default 0.0
    b : float, optional
        Coefficient of y, by default 0.0
    c : float, optional
        Coefficient of z, by default 0.0
    d : float, optional
        Displacement constant d, by default 0.0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, a: float = 0.0, b: float = 0.0, c: float = 0.0, d: float = 0.0, **kwargs: Any) -> None:
        """Initialise the Plane surface."""
        super().__init__(**kwargs)
        self.a: float = a
        self.b: float = b
        self.c: float = c
        self.d: float = d

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of the plane."""
        surface_type = 'P'
        surface_params = [
            f'{self.a:{formatting.MCNP_FMT_SURFACE_PLANE_NORMAL}}',
            f'{self.b:{formatting.MCNP_FMT_SURFACE_PLANE_NORMAL}}',
            f'{self.c:{formatting.MCNP_FMT_SURFACE_PLANE_NORMAL}}',
            f'{self.d:{formatting.MCNP_FMT_SURFACE_PLANE_DISPLACEMENT}}'
        ]
        return mcnp_surface(surface_type, surface_params, self)

    @property
    def openmc_surface(self) -> Any:
        """OpenMC surface object representation."""
        if not self._openmc_surface:
            self._openmc_surface = openmc.surface.Plane(
                a=self.a,
                b=self.b,
                c=self.c,
                d=self.d,
                boundary_type=openmc_boundary_type(self)
            )
        return self._openmc_surface

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'plane_' + \
               f'a_{self.a:{formatting.MCNP_FMT_SURFACE_PLANE_NORMAL}}_' + \
               f'b_{self.b:{formatting.MCNP_FMT_SURFACE_PLANE_NORMAL}}_' + \
               f'c_{self.c:{formatting.MCNP_FMT_SURFACE_PLANE_NORMAL}}_' + \
               f'd_{self.d:{formatting.MCNP_FMT_SURFACE_PLANE_DISPLACEMENT}}'

    def __lt__(self, other: "Plane") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : Plane
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        vars_self = [self.a, self.b, self.c, self.d]
        vars_other = [other.a, other.b, other.c, other.d]
        return _less_than_multi_variate(vars_self, vars_other)


class PlaneFromThreePoints(Surface):
    """Plane surface defined by three points in Cartesian coordinates.

    Parameters
    ----------
    x0, y0, z0 : float, optional
        Coordinates of the first point, by default 0.0
    x1, y1, z1 : float, optional
        Coordinates of the second point, by default 0.0
    x2, y2, z2 : float, optional
        Coordinates of the third point, by default 0.0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, x0: float = 0.0, y0: float = 0.0, z0: float = 0.0,
                       x1: float = 0.0, y1: float = 0.0, z1: float = 0.0,
                       x2: float = 0.0, y2: float = 0.0, z2: float = 0.0,
                       **kwargs: Any) -> None:
        """Initialise the PlaneFromThreePoints surface."""
        super().__init__(**kwargs)
        self.x0: float = x0
        self.y0: float = y0
        self.z0: float = z0
        self.x1: float = x1
        self.y1: float = y1
        self.z1: float = z1
        self.x2: float = x2
        self.y2: float = y2
        self.z2: float = z2

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of the plane."""
        surface_type = 'P'
        surface_params = [
            f'{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}', 
            f'{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}', 
            f'{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}', 
            f'{self.x1:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}', 
            f'{self.y1:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}', 
            f'{self.z1:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}', 
            f'{self.x2:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}', 
            f'{self.y2:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}', 
            f'{self.z2:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}'
        ]
        return mcnp_surface(surface_type, surface_params, self)

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'planefromthreepoints_' + \
               f'x0_{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'y0_{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'z0_{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'x1_{self.x1:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'y1_{self.y1:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'z1_{self.z1:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'x2_{self.x2:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'y2_{self.y2:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'z2_{self.z2:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_'

    def __lt__(self, other: "PlaneFromThreePoints") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : PlaneFromThreePoints
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        vars_self = [self.x0, self.y0, self.z0, self.x1, self.y1, self.z1, self.x2, self.y2, self.z2]
        vars_other = [other.x0, other.y0, other.z0, other.x1, other.y1, other.z1, other.x2, other.y2, other.z2]
        return _less_than_multi_variate(vars_self, vars_other)


class XPlane(Surface):
    """Plane perpendicular to the X-axis at x = x0.

    Parameters
    ----------
    x0 : float, optional
        Location on the X-axis, by default 0.0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, x0: float = 0.0, **kwargs: Any) -> None:
        """Initialise the XPlane surface."""
        super().__init__(**kwargs)
        self.x0: float = float(x0)

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of the X-plane."""
        surface_type = 'PX'
        surface_params = [f'{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}']
        return mcnp_surface(surface_type, surface_params, self)
    
    @property
    def openmc_surface(self) -> Any:
        """OpenMC surface object representation."""
        if not self._openmc_surface:
            self._openmc_surface = openmc.surface.XPlane(
                x0=self.x0,
                boundary_type=openmc_boundary_type(self)
            )
        return self._openmc_surface

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'xplane_' + f'{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}'

    def __lt__(self, other: "XPlane") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : XPlane
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        return self.x0 <= other.x0


class YPlane(Surface):
    """Plane perpendicular to the Y-axis at y = y0.

    Parameters
    ----------
    y0 : float, optional
        Location on the Y-axis, by default 0.0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, y0: float = 0.0, **kwargs: Any) -> None:
        """Initialise the YPlane surface."""
        super().__init__(**kwargs)
        self.y0: float = float(y0)

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of the Y-plane."""
        surface_type = 'PY'
        surface_params = [f'{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}']
        return mcnp_surface(surface_type, surface_params, self)

    @property
    def openmc_surface(self) -> Any:
        """OpenMC surface object representation."""
        if not self._openmc_surface:
            self._openmc_surface = openmc.surface.YPlane(
                y0=self.y0,
                boundary_type=openmc_boundary_type(self)
            )
        return self._openmc_surface

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'yplane_' + f'{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}'

    def __lt__(self, other: "YPlane") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : YPlane
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        return self.y0 <= other.y0


class ZPlane(Surface):
    """Plane perpendicular to the Z-axis at z = z0.

    Parameters
    ----------
    z0 : float, optional
        Location on the Z-axis, by default 0.0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, z0: float = 0.0, **kwargs: Any) -> None:
        """Initialise the ZPlane surface."""
        super().__init__(**kwargs)
        self.z0: float = float(z0)

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of the Z-plane."""
        surface_type = 'PZ'
        surface_params = [f'{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}']
        return mcnp_surface(surface_type, surface_params, self)

    @property
    def openmc_surface(self) -> Any:
        """OpenMC surface object representation."""
        if not self._openmc_surface:
            self._openmc_surface = openmc.surface.ZPlane(
                z0=self.z0,
                boundary_type=openmc_boundary_type(self)
            )
        return self._openmc_surface

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'zplane_' + f'{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}'

    def __lt__(self, other: "ZPlane") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : ZPlane
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        return self.z0 <= other.z0


class Sphere(Surface):
    """General sphere centered at (x0, y0, z0) with radius r.

    Parameters
    ----------
    x0 : float, optional
        X-coordinate of center, by default 0.0
    y0 : float, optional
        Y-coordinate of center, by default 0.0
    z0 : float, optional
        Z-coordinate of center, by default 0.0
    radius : float, optional
        Radius of sphere, by default 0.0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, x0: float = 0.0, y0: float = 0.0, z0: float = 0.0, radius: float = 0.0, **kwargs: Any) -> None:
        """Initialise the Sphere surface."""
        super().__init__(**kwargs)
        self.x0: float = float(x0)
        self.y0: float = float(y0)
        self.z0: float = float(z0)
        self.radius: float = float(radius)

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of the sphere."""
        surface_type = 'S'
        surface_params = [
            f'{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}', 
            f'{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.radius:{formatting.MCNP_FMT_SURFACE_RADIUS}}'
        ]
        return mcnp_surface(surface_type, surface_params, self)

    @property
    def serpent_card(self) -> SerpentCard:
        """Serpent card representation of the sphere."""
        surface_type = 'sph'
        surface_params = [
            f'{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}', 
            f'{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.radius:{formatting.MCNP_FMT_SURFACE_RADIUS}}'
        ]
        return serpent_surface(surface_type, surface_params, self)

    @property
    def openmc_surface(self) -> Any:
        """OpenMC surface object representation."""
        if not self._openmc_surface:
            self._openmc_surface = openmc.surface.Sphere(
                r=self.radius,
                x0=self.x0,
                y0=self.y0,
                z0=self.z0,
                boundary_type=openmc_boundary_type(self)
            )
        return self._openmc_surface

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'sphere_' + \
               f'{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' +\
               f'{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' +\
               f'{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' +\
               f'{self.radius:{formatting.MCNP_FMT_SURFACE_RADIUS}}'

    def __lt__(self, other: "Sphere") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : Sphere
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        vars_self = [self.x0, self.y0, self.z0, self.radius]
        vars_other = [other.x0, other.y0, other.z0, other.radius]
        return _less_than_multi_variate(vars_self, vars_other)


class SphereAtOrigin(Surface):
    """Sphere centered at the origin (0, 0, 0) with radius r.

    Parameters
    ----------
    radius : float, optional
        Radius of sphere, by default 0.0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, radius: float = 0.0, **kwargs: Any) -> None:
        """Initialise the SphereAtOrigin surface."""
        super().__init__(**kwargs)
        self.radius: float = float(radius)

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of the sphere at origin."""
        surface_type = 'SO'
        surface_params = [f'{self.radius:{formatting.MCNP_FMT_SURFACE_RADIUS}}']
        return mcnp_surface(surface_type, surface_params, self)

    @property
    def openmc_surface(self) -> Any:
        """OpenMC surface object representation."""
        if not self._openmc_surface:
            return openmc.surface.Sphere(
                r=self.radius,
                boundary_type=openmc_boundary_type(self)
            )
        return self._openmc_surface

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'sphereatorigin_' + f'{self.radius:{formatting.MCNP_FMT_SURFACE_RADIUS}}'

    def __lt__(self, other: "SphereAtOrigin") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : SphereAtOrigin
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        return self.radius <= other.radius


class SphereOnXAxis(Surface):
    """Sphere centered on the X-axis at (x0, 0, 0) with radius r.

    Parameters
    ----------
    x0 : float, optional
        X-coordinate of center, by default 0.0
    radius : float, optional
        Radius of sphere, by default 0.0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, x0: float = 0.0, radius: float = 0.0, **kwargs: Any) -> None:
        """Initialise the SphereOnXAxis surface."""
        super().__init__(**kwargs)
        self.x0: float = x0
        self.radius: float = radius

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of the sphere on X-axis."""
        surface_type = 'SX'
        surface_params = [
            f'{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.radius:{formatting.MCNP_FMT_SURFACE_RADIUS}}'
        ]
        return mcnp_surface(surface_type, surface_params, self)

    @property
    def openmc_surface(self) -> Any:
        """OpenMC surface object representation."""
        if not self._openmc_surface:
            self._openmc_surface = openmc.surface.Sphere(
                r=self.radius,
                x0=self.x0,
                boundary_type=openmc_boundary_type(self)
            )
        return self._openmc_surface

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'sphereonxaxis_' + \
               f'{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}} _' +\
               f'{self.radius}:{formatting.MCNP_FMT_SURFACE_RADIUS}'

    def __lt__(self, other: "SphereOnXAxis") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : SphereOnXAxis
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        vars_self = [self.x0, self.radius]
        vars_other = [other.x0, other.radius]
        return _less_than_multi_variate(vars_self, vars_other)


class SphereOnYAxis(Surface):
    """Sphere centered on the Y-axis at (0, y0, 0) with radius r.

    Parameters
    ----------
    y0 : float, optional
        Y-coordinate of center, by default 0.0
    radius : float, optional
        Radius of sphere, by default 0.0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, y0: float = 0.0, radius: float = 0.0, **kwargs: Any) -> None:
        """Initialise the SphereOnYAxis surface."""
        super().__init__(**kwargs)
        self.y0: float = y0
        self.radius: float = radius

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of the sphere on Y-axis."""
        surface_type = 'SY'
        surface_params = [
            f'{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}', 
            f'{self.radius:{formatting.MCNP_FMT_SURFACE_RADIUS}}'
        ]
        return mcnp_surface(surface_type, surface_params, self)

    @property
    def openmc_surface(self) -> Any:
        """OpenMC surface object representation."""
        if not self._openmc_surface:
            self._openmc_surface = openmc.surface.Sphere(
                r=self.radius,
                y0=self.y0,
                boundary_type=openmc_boundary_type(self)
            )
        return self._openmc_surface

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'sphereonyaxis_' + \
               f'{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' +\
               f'{self.radius:{formatting.MCNP_FMT_SURFACE_RADIUS}}'

    def __lt__(self, other: "SphereOnYAxis") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : SphereOnYAxis
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        vars_self = [self.y0, self.radius]
        vars_other = [other.y0, other.radius]
        return _less_than_multi_variate(vars_self, vars_other)


class SphereOnZAxis(Surface):
    """Sphere centered on the Z-axis at (0, 0, z0) with radius r.

    Parameters
    ----------
    z0 : float, optional
        Z-coordinate of center, by default 0.0
    radius : float, optional
        Radius of sphere, by default 0.0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, z0: float = 0.0, radius: float = 0.0, **kwargs: Any) -> None:
        """Initialise the SphereOnZAxis surface."""
        super().__init__(**kwargs)
        self.z0: float = z0
        self.radius: float = radius

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of the sphere on Z-axis."""
        surface_type = 'SZ'
        surface_params = [
            f'{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}', 
            f'{self.radius:{formatting.MCNP_FMT_SURFACE_RADIUS}}'
        ]
        return mcnp_surface(surface_type, surface_params, self)

    @property
    def openmc_surface(self) -> Any:
        """OpenMC surface object representation."""
        if not self._openmc_surface:
            self._openmc_surface = openmc.surface.Sphere(
                r=self.radius,
                z0=self.z0,
                boundary_type=openmc_boundary_type(self)
            )
        return self._openmc_surface

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'sphereonzaxis_' + \
               f'{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' +\
               f'{self.radius:{formatting.MCNP_FMT_SURFACE_RADIUS}}'

    def __lt__(self, other: "SphereOnZAxis") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : SphereOnZAxis
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        vars_self = [self.z0, self.radius]
        vars_other = [other.z0, other.radius]
        return _less_than_multi_variate(vars_self, vars_other)


class CylinderParallelToXAxis(Surface):
    """Cylinder parallel to the X-axis centered at (y0, z0) with radius r.

    Parameters
    ----------
    y0 : float, optional
        Y-coordinate of cylinder axis, by default 0.0
    z0 : float, optional
        Z-coordinate of cylinder axis, by default 0.0
    radius : float, optional
        Radius of cylinder, by default 0.0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, y0: float = 0.0, z0: float = 0.0, radius: float = 0.0, **kwargs: Any) -> None:
        """Initialise the CylinderParallelToXAxis surface."""
        super().__init__(**kwargs)
        self.y0: float = y0
        self.z0: float = z0
        self.radius: float = radius

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of cylinder parallel to X-axis."""
        surface_type = 'C/X'
        surface_params = [
            f'{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.radius:{formatting.MCNP_FMT_SURFACE_RADIUS}}'
        ]
        return mcnp_surface(surface_type, surface_params, self)
    
    @property
    def openmc_surface(self) -> Any:
        """OpenMC surface object representation."""
        if not self._openmc_surface:
            self._openmc_surface = openmc.surface.XCylinder(
                r=self.radius,
                y0=self.y0,
                z0=self.z0,
                boundary_type=openmc_boundary_type(self)
            )
        return self._openmc_surface

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'cylinderparalleltoxaxis_' + \
               f'{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' +\
               f'{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' +\
               f'{self.radius:{formatting.MCNP_FMT_SURFACE_RADIUS}}'

    def __lt__(self, other: "CylinderParallelToXAxis") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : CylinderParallelToXAxis
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        vars_self = [self.y0, self.z0, self.radius]
        vars_other = [other.y0, other.z0, other.radius]
        return _less_than_multi_variate(vars_self, vars_other)


class CylinderParallelToYAxis(Surface):
    """Cylinder parallel to the Y-axis centered at (x0, z0) with radius r.

    Parameters
    ----------
    x0 : float, optional
        X-coordinate of cylinder axis, by default 0.0
    z0 : float, optional
        Z-coordinate of cylinder axis, by default 0.0
    radius : float, optional
        Radius of cylinder, by default 0.0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, x0: float = 0.0, z0: float = 0.0, radius: float = 0.0, **kwargs: Any) -> None:
        """Initialise the CylinderParallelToYAxis surface."""
        super().__init__(**kwargs)
        self.x0: float = x0
        self.z0: float = z0
        self.radius: float = radius

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of cylinder parallel to Y-axis."""
        surface_type = 'C/Y'
        surface_params = [
            f'{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.radius:{formatting.MCNP_FMT_SURFACE_RADIUS}}'
        ]
        return mcnp_surface(surface_type, surface_params, self)

    @property
    def openmc_surface(self) -> Any:
        """OpenMC surface object representation."""
        if not self._openmc_surface:
            self._openmc_surface = openmc.surface.YCylinder(
                r=self.radius,
                x0=self.x0,
                z0=self.z0,
                boundary_type=openmc_boundary_type(self)
            )
        return self._openmc_surface

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'cylinderparalleltoyaxis_' + \
               f'{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' +\
               f'{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' +\
               f'{self.radius:{formatting.MCNP_FMT_SURFACE_RADIUS}}'

    def __lt__(self, other: "CylinderParallelToYAxis") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : CylinderParallelToYAxis
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        vars_self = [self.x0, self.z0, self.radius]
        vars_other = [other.x0, other.z0, other.radius]
        return _less_than_multi_variate(vars_self, vars_other)


class CylinderParallelToZAxis(Surface):
    """Cylinder parallel to the Z-axis centered at (x0, y0) with radius r.

    Parameters
    ----------
    x0 : float, optional
        X-coordinate of cylinder axis, by default 0.0
    y0 : float, optional
        Y-coordinate of cylinder axis, by default 0.0
    radius : float, optional
        Radius of cylinder, by default 0.0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, x0: float = 0.0, y0: float = 0.0, radius: float = 0.0, **kwargs: Any) -> None:
        """Initialise the CylinderParallelToZAxis surface."""
        super().__init__(**kwargs)
        self.x0: float = x0
        self.y0: float = y0
        self.radius: float = radius

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of cylinder parallel to Z-axis."""
        surface_type = 'C/Z'
        surface_params = [
            f'{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.radius:{formatting.MCNP_FMT_SURFACE_RADIUS}}'
        ]
        return mcnp_surface(surface_type, surface_params, self)
    
    @property
    def openmc_surface(self) -> Any:
        """OpenMC surface object representation."""
        if not self._openmc_surface:
            self._openmc_surface = openmc.surface.ZCylinder(
                r=self.radius,
                x0=self.x0,
                y0=self.y0,
                boundary_type=openmc_boundary_type(self)
            )
        return self._openmc_surface

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'cylinderparalleltozaxis_' + \
               f'{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' +\
               f'{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' +\
               f'{self.radius:{formatting.MCNP_FMT_SURFACE_RADIUS}}'

    def __lt__(self, other: "CylinderParallelToZAxis") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : CylinderParallelToZAxis
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        vars_self = [self.x0, self.y0, self.radius]
        vars_other = [other.x0, other.y0, other.radius]
        return _less_than_multi_variate(vars_self, vars_other)


class CylinderOnXAxis(Surface):
    """Cylinder on the X-axis with radius r.

    Parameters
    ----------
    radius : float, optional
        Radius of cylinder, by default 0.0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, radius: float = 0.0, **kwargs: Any) -> None:
        """Initialise the CylinderOnXAxis surface."""
        super().__init__(**kwargs)
        self.radius: float = radius

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of cylinder on X-axis."""
        surface_type = 'CX'
        surface_params = [f'{self.radius:{formatting.MCNP_FMT_SURFACE_RADIUS}}']
        return mcnp_surface(surface_type, surface_params, self)
    
    @property
    def openmc_surface(self) -> Any:
        """OpenMC surface object representation."""
        if not self._openmc_surface:
            self._openmc_surface = openmc.surface.XCylinder(
                r=self.radius,
                boundary_type=openmc_boundary_type(self)
            )
        return self._openmc_surface

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'cylinderonxaxis_' + f'{self.radius:{formatting.MCNP_FMT_SURFACE_RADIUS}}'

    def __lt__(self, other: "CylinderOnXAxis") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : CylinderOnXAxis
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        return self.radius <= other.radius


class CylinderOnYAxis(Surface):
    """Cylinder on the Y-axis with radius r.

    Parameters
    ----------
    radius : float, optional
        Radius of cylinder, by default 0.0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, radius: float = 0.0, **kwargs: Any) -> None:
        """Initialise the CylinderOnYAxis surface."""
        super().__init__(**kwargs)
        self.radius: float = radius

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of cylinder on Y-axis."""
        surface_type = 'CY'
        surface_params = [f'{self.radius:{formatting.MCNP_FMT_SURFACE_RADIUS}}']
        return mcnp_surface(surface_type, surface_params, self)
    
    @property
    def openmc_surface(self) -> Any:
        """OpenMC surface object representation."""
        if not self._openmc_surface:
            self._openmc_surface = openmc.surface.YCylinder(
                r=self.radius,
                boundary_type=openmc_boundary_type(self)
            )
        return self._openmc_surface

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'cylinderonyaxis_' + f'{self.radius:{formatting.MCNP_FMT_SURFACE_RADIUS}}'

    def __lt__(self, other: "CylinderOnYAxis") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : CylinderOnYAxis
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        return self.radius <= other.radius


class CylinderOnZAxis(Surface):
    """Cylinder on the Z-axis with radius r.

    Parameters
    ----------
    radius : float, optional
        Radius of cylinder, by default 0.0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, radius: float = 0.0, **kwargs: Any) -> None:
        """Initialise the CylinderOnZAxis surface."""
        super().__init__(**kwargs)
        self.radius: float = radius

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of cylinder on Z-axis."""
        surface_type = 'CZ'
        surface_params = [f'{self.radius:{formatting.MCNP_FMT_SURFACE_RADIUS}}']
        return mcnp_surface(surface_type, surface_params, self)

    @property
    def openmc_surface(self) -> Any:
        """OpenMC surface object representation."""
        if not self._openmc_surface:
            self._openmc_surface = openmc.surface.ZCylinder(
                r=self.radius,
                boundary_type=openmc_boundary_type(self)
            )
        return self._openmc_surface

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'cylinderonzaxis_' + f'{self.radius:{formatting.MCNP_FMT_SURFACE_RADIUS}}'

    def __lt__(self, other: "CylinderOnZAxis") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : CylinderOnZAxis
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        return self.radius <= other.radius


class ConeParallelToXAxis(Surface):
    """Cone parallel to the X-axis with vertex at (x0, y0, z0).

    Parameters
    ----------
    x0 : float, optional
        X-coordinate of vertex, by default 0.0
    y0 : float, optional
        Y-coordinate of vertex, by default 0.0
    z0 : float, optional
        Z-coordinate of vertex, by default 0.0
    t_sq : float, optional
        Tangent squared of cone half-angle, by default 0.0
    sheet : int, optional
        Sheet selector (-1, 0, or 1), by default 0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, x0: float = 0.0, y0: float = 0.0, z0: float = 0.0, t_sq: float = 0.0, sheet: int = 0, **kwargs: Any) -> None:
        """Initialise the ConeParallelToXAxis surface."""
        super().__init__(**kwargs)
        self.x0: float = x0
        self.y0: float = y0
        self.z0: float = z0
        self.t_sq: float = t_sq
        self.sheet: int = sheet

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of cone parallel to X-axis."""
        surface_type = 'K/X'
        surface_params = [
            f'{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.t_sq:{formatting.MCNP_FMT_SURFACE_T_SQ}}', 
            f'{self.sheet}'
        ]
        return mcnp_surface(surface_type, surface_params, self)

    @property
    def openmc_surface(self) -> Any:
        """OpenMC surface object representation."""
        if not self._openmc_surface:
            if self.sheet == 0:
                self._openmc_surface = openmc.surface.XCone(
                    x0=self.x0,
                    y0=self.y0,
                    z0=self.z0,
                    r2=self.t_sq,
                    boundary_type=openmc_boundary_type(self)
                )
            elif self.sheet == 1:
                self._openmc_surface = openmc.model.XConeOneSided(
                    x0=self.x0,
                    y0=self.y0,
                    z0=self.z0,
                    r2=self.t_sq,
                    up=True,
                    boundary_type=openmc_boundary_type(self)
                )
            elif self.sheet == -1:
                self._openmc_surface = openmc.model.XConeOneSided(
                    x0=self.x0,
                    y0=self.y0,
                    z0=self.z0,
                    r2=self.t_sq,
                    up=False,
                    boundary_type=openmc_boundary_type(self)
                )
            else:
                msg = f' sheet of cone should be -1, 0, or 1 and not {self.sheet}'
                raise ValueError(msg)

        return self._openmc_surface

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'coneparalleltoxaxis_' + \
               f'x0_{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'y0_{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'z0_{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'tsq_{self.t_sq:{formatting.MCNP_FMT_SURFACE_T_SQ}}_' + \
               f'sheet_{self.sheet}'

    def __lt__(self, other: "ConeParallelToXAxis") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : ConeParallelToXAxis
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        vars_self = [self.x0, self.y0, self.z0, self.t_sq, self.sheet]
        vars_other = [other.x0, other.y0, other.z0, other.t_sq, other.sheet]
        return _less_than_multi_variate(vars_self, vars_other)


class ConeParallelToYAxis(Surface):
    """Cone parallel to the Y-axis with vertex at (x0, y0, z0).

    Parameters
    ----------
    x0 : float, optional
        X-coordinate of vertex, by default 0.0
    y0 : float, optional
        Y-coordinate of vertex, by default 0.0
    z0 : float, optional
        Z-coordinate of vertex, by default 0.0
    t_sq : float, optional
        Tangent squared of cone half-angle, by default 0.0
    sheet : int, optional
        Sheet selector (-1, 0, or 1), by default 0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, x0: float = 0.0, y0: float = 0.0, z0: float = 0.0, t_sq: float = 0.0, sheet: int = 0, **kwargs: Any) -> None:
        """Initialise the ConeParallelToYAxis surface."""
        super().__init__(**kwargs)
        self.x0: float = x0
        self.y0: float = y0
        self.z0: float = z0
        self.t_sq: float = t_sq
        self.sheet: int = sheet

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of cone parallel to Y-axis."""
        surface_type = 'K/Y'
        surface_params = [
            f'{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.t_sq:{formatting.MCNP_FMT_SURFACE_T_SQ}}', 
            f'{self.sheet}'
        ]
        return mcnp_surface(surface_type, surface_params, self)

    @property
    def openmc_surface(self) -> Any:
        """OpenMC surface object representation."""
        if not self._openmc_surface:
            if self.sheet == 0:
                self._openmc_surface = openmc.surface.YCone(
                    x0=self.x0,
                    y0=self.y0,
                    z0=self.z0,
                    r2=self.t_sq,
                    boundary_type=openmc_boundary_type(self)
                )
            elif self.sheet == 1:
                self._openmc_surface = openmc.model.YConeOneSided(
                    x0=self.x0,
                    y0=self.y0,
                    z0=self.z0,
                    r2=self.t_sq,
                    up=True,
                    boundary_type=openmc_boundary_type(self)
                )
            elif self.sheet == -1:
                self._openmc_surface = openmc.model.YConeOneSided(
                    x0=self.x0,
                    y0=self.y0,
                    z0=self.z0,
                    r2=self.t_sq,
                    up=False,
                    boundary_type=openmc_boundary_type(self)
                )

        return self._openmc_surface

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'coneparalleltoyaxis_' + \
               f'x0_{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'y0_{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'z0_{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'tsq_{self.t_sq:{formatting.MCNP_FMT_SURFACE_T_SQ}}_' + \
               f'sheet_{self.sheet}'

    def __lt__(self, other: "ConeParallelToYAxis") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : ConeParallelToYAxis
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        vars_self = [self.x0, self.y0, self.z0, self.t_sq, self.sheet]
        vars_other = [other.x0, other.y0, other.z0, other.t_sq, other.sheet]
        return _less_than_multi_variate(vars_self, vars_other)


class ConeParallelToZAxis(Surface):
    """Cone parallel to the Z-axis with vertex at (x0, y0, z0).

    Parameters
    ----------
    x0 : float, optional
        X-coordinate of vertex, by default 0.0
    y0 : float, optional
        Y-coordinate of vertex, by default 0.0
    z0 : float, optional
        Z-coordinate of vertex, by default 0.0
    t_sq : float, optional
        Tangent squared of cone half-angle, by default 0.0
    sheet : int, optional
        Sheet selector (-1, 0, or 1), by default 0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, x0: float = 0.0, y0: float = 0.0, z0: float = 0.0, t_sq: float = 0.0, sheet: int = 0, **kwargs: Any) -> None:
        """Initialise the ConeParallelToZAxis surface."""
        super().__init__(**kwargs)
        self.x0: float = x0
        self.y0: float = y0
        self.z0: float = z0
        self.t_sq: float = t_sq
        self.sheet: int = sheet

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of cone parallel to Z-axis."""
        surface_type = 'K/Z'
        surface_params = [
            f'{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.t_sq:{formatting.MCNP_FMT_SURFACE_T_SQ}}', 
            f'{self.sheet}'
        ]
        return mcnp_surface(surface_type, surface_params, self)

    @property
    def openmc_surface(self) -> Any:
        """OpenMC surface object representation."""
        if not self._openmc_surface:
            if self.sheet == 0:
                self._openmc_surface = openmc.surface.ZCone(
                    x0=self.x0,
                    y0=self.y0,
                    z0=self.z0,
                    r2=self.t_sq,
                    boundary_type=openmc_boundary_type(self)
                )
            elif self.sheet == 1:
                self._openmc_surface = openmc.model.ZConeOneSided(
                    x0=self.x0,
                    y0=self.y0,
                    z0=self.z0,
                    r2=self.t_sq,
                    up=True,
                    boundary_type=openmc_boundary_type(self)
                )
            elif self.sheet == -1:
                self._openmc_surface = openmc.model.ZConeOneSided(
                    x0=self.x0,
                    y0=self.y0,
                    z0=self.z0,
                    r2=self.t_sq,
                    up=False,
                    boundary_type=openmc_boundary_type(self)
                )
        
        return self._openmc_surface

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'coneparalleltozaxis_' + \
               f'x0_{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'y0_{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'z0_{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'tsq_{self.t_sq:{formatting.MCNP_FMT_SURFACE_T_SQ}}_' + \
               f'sheet_{self.sheet}'

    def __lt__(self, other: "ConeParallelToZAxis") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : ConeParallelToZAxis
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        vars_self = [self.x0, self.y0, self.z0, self.t_sq, self.sheet]
        vars_other = [other.x0, other.y0, other.z0, other.t_sq, other.sheet]
        return _less_than_multi_variate(vars_self, vars_other)


class ConeOnXAxis(Surface):
    """Cone on the X-axis with vertex at (x0, 0, 0).

    Parameters
    ----------
    x0 : float, optional
        X-coordinate of vertex, by default 0.0
    t_sq : float, optional
        Tangent squared of cone half-angle, by default 0.0
    sheet : int, optional
        Sheet selector (-1, 0, or 1), by default 0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, x0: float = 0.0, t_sq: float = 0.0, sheet: int = 0, **kwargs: Any) -> None:
        """Initialise the ConeOnXAxis surface."""
        super().__init__(**kwargs)
        self.x0: float = x0
        self.t_sq: float = t_sq
        self.sheet: int = sheet

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of cone on X-axis."""
        surface_type = 'KX'
        surface_params = [
            f'{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.t_sq:{formatting.MCNP_FMT_SURFACE_T_SQ}}', 
            f'{self.sheet}'
        ]
        return mcnp_surface(surface_type, surface_params, self)

    @property
    def openmc_surface(self) -> Any:
        """OpenMC surface object representation."""
        if not self._openmc_surface:
            if self.sheet == 0:
                self._openmc_surface = openmc.surface.XCone(
                    x0=self.x0,
                    r2=self.t_sq,
                    boundary_type=openmc_boundary_type(self)
                )
            elif self.sheet == 1:
                self._openmc_surface = openmc.model.XConeOneSided(
                    x0=self.x0,
                    r2=self.t_sq,
                    up=True,
                    boundary_type=openmc_boundary_type(self)
                )
            elif self.sheet == -1:
                self._openmc_surface = openmc.model.XConeOneSided(
                    x0=self.x0,
                    r2=self.t_sq,
                    up=False,
                    boundary_type=openmc_boundary_type(self)
                )

        return self._openmc_surface

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'coneonxaxis_' + \
               f'x0_{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'tsq_{self.t_sq:{formatting.MCNP_FMT_SURFACE_T_SQ}}_' + \
               f'sheet_{self.sheet}'

    def __lt__(self, other: "ConeOnXAxis") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : ConeOnXAxis
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        vars_self = [self.x0, self.t_sq, self.sheet]
        vars_other = [other.x0, other.t_sq, other.sheet]
        return _less_than_multi_variate(vars_self, vars_other)


class ConeOnYAxis(Surface):
    """Cone on the Y-axis with vertex at (0, y0, 0).

    Parameters
    ----------
    y0 : float, optional
        Y-coordinate of vertex, by default 0.0
    t_sq : float, optional
        Tangent squared of cone half-angle, by default 0.0
    sheet : int, optional
        Sheet selector (-1, 0, or 1), by default 0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, y0: float = 0.0, t_sq: float = 0.0, sheet: int = 0, **kwargs: Any) -> None:
        """Initialise the ConeOnYAxis surface."""
        super().__init__(**kwargs)
        self.y0: float = y0
        self.t_sq: float = t_sq
        self.sheet: int = sheet

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of cone on Y-axis."""
        surface_type = 'KY'
        surface_params = [
            f'{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.t_sq:{formatting.MCNP_FMT_SURFACE_T_SQ}}', 
            f'{self.sheet}'
        ]
        return mcnp_surface(surface_type, surface_params, self)

    @property
    def openmc_surface(self) -> Any:
        """OpenMC surface object representation."""
        if not self._openmc_surface:
            if self.sheet == 0:
                self._openmc_surface = openmc.surface.YCone(
                    y0=self.y0,
                    r2=self.t_sq,
                    boundary_type=openmc_boundary_type(self)
                )
            elif self.sheet == 1:
                self._openmc_surface = openmc.model.YConeOneSided(
                    y0=self.y0,
                    r2=self.t_sq,
                    up=True,
                    boundary_type=openmc_boundary_type(self)
                )
            elif self.sheet == -1:
                self._openmc_surface = openmc.model.YConeOneSided(
                    y0=self.y0,
                    r2=self.t_sq,
                    up=False,
                    boundary_type=openmc_boundary_type(self)
                )
        
        return self._openmc_surface

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'coneonyaxis_' + \
               f'y0_{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'tsq_{self.t_sq:{formatting.MCNP_FMT_SURFACE_T_SQ}}_' + \
               f'sheet_{self.sheet}'

    def __lt__(self, other: "ConeOnYAxis") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : ConeOnYAxis
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        vars_self = [self.y0, self.t_sq, self.sheet]
        vars_other = [other.y0, other.t_sq, other.sheet]
        return _less_than_multi_variate(vars_self, vars_other)


class ConeOnZAxis(Surface):
    """Cone on the Z-axis with vertex at (0, 0, z0).

    Parameters
    ----------
    z0 : float, optional
        Z-coordinate of vertex, by default 0.0
    t_sq : float, optional
        Tangent squared of cone half-angle, by default 0.0
    sheet : int, optional
        Sheet selector (-1, 0, or 1), by default 0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, z0: float = 0.0, t_sq: float = 0.0, sheet: int = 0, **kwargs: Any) -> None:
        """Initialise the ConeOnZAxis surface."""
        super().__init__(**kwargs)
        self.z0: float = z0
        self.t_sq: float = t_sq
        self.sheet: int = sheet

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of cone on Z-axis."""
        surface_type = 'KZ'
        surface_params = [
            f'{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.t_sq:{formatting.MCNP_FMT_SURFACE_T_SQ}}', 
            f'{self.sheet}'
        ]
        return mcnp_surface(surface_type, surface_params, self)

    @property
    def openmc_surface(self) -> Any:
        """OpenMC surface object representation."""
        if not self._openmc_surface:
            if self.sheet == 0:
                self._openmc_surface = openmc.surface.ZCone(
                    z0=self.z0,
                    r2=self.t_sq,
                    boundary_type=openmc_boundary_type(self)
                )
            elif self.sheet == 1:
                self._openmc_surface = openmc.model.ZConeOneSided(
                    z0=self.z0,
                    r2=self.t_sq,
                    up=True,
                    boundary_type=openmc_boundary_type(self)
                )
            elif self.sheet == -1:
                self._openmc_surface = openmc.model.ZConeOneSided(
                    z0=self.z0,
                    r2=self.t_sq,
                    up=False,
                    boundary_type=openmc_boundary_type(self)
                )

        return self._openmc_surface

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'coneonzaxis_' + \
               f'z0_{self.z0:{formatting.MCNP_FMT_SURFACE_PLANE_NORMAL}}_' + \
               f'tsq_{self.t_sq:{formatting.MCNP_FMT_SURFACE_T_SQ}}_' + \
               f'sheet_{self.sheet}'

    def __lt__(self, other: "ConeOnZAxis") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : ConeOnZAxis
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        vars_self = [self.z0, self.t_sq, self.sheet]
        vars_other = [other.z0, other.t_sq, other.sheet]
        return _less_than_multi_variate(vars_self, vars_other)


class Ellipsoid(Surface):
    """Ellipsoid surface.

    Parameters
    ----------
    a, b, c, d, e, f, g : float, optional
        Quadric equation coefficients, by default 0.0
    x0, y0, z0 : float, optional
        Center point offset coordinates, by default 0.0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, a: float = 0.0, b: float = 0.0, c: float = 0.0, d: float = 0.0, 
                 e: float = 0.0, f: float = 0.0, g: float = 0.0, x0: float = 0.0, 
                 y0: float = 0.0, z0: float = 0.0, **kwargs: Any) -> None:
        """Initialise the Ellipsoid surface."""
        super().__init__(**kwargs)
        self.a: float = a
        self.b: float = b
        self.c: float = c
        self.d: float = d
        self.e: float = e
        self.f: float = f 
        self.g: float = g
        self.x0: float = x0
        self.y0: float = y0
        self.z0: float = z0

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of the ellipsoid."""
        surface_type = 'SQ'
        surface_params = [
            f'{self.a:{formatting.MCNP_FMT_SURFACE_ELIPSOID}}',
            f'{self.b:{formatting.MCNP_FMT_SURFACE_ELIPSOID}}', 
            f'{self.c:{formatting.MCNP_FMT_SURFACE_ELIPSOID}}', 
            f'{self.d:{formatting.MCNP_FMT_SURFACE_ELIPSOID}}', 
            f'{self.e:{formatting.MCNP_FMT_SURFACE_ELIPSOID}}', 
            f'{self.f:{formatting.MCNP_FMT_SURFACE_ELIPSOID}}', 
            f'{self.g:{formatting.MCNP_FMT_SURFACE_ELIPSOID}}', 
            f'{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}'
        ]
        return mcnp_surface(surface_type, surface_params, self)

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'ellipsoid_' + \
               f'a_{self.a:{formatting.MCNP_FMT_SURFACE_ELIPSOID}}_' + \
               f'b_{self.b:{formatting.MCNP_FMT_SURFACE_ELIPSOID}}_' + \
               f'c_{self.c:{formatting.MCNP_FMT_SURFACE_ELIPSOID}}_' + \
               f'd_{self.d:{formatting.MCNP_FMT_SURFACE_ELIPSOID}}_' + \
               f'e_{self.e:{formatting.MCNP_FMT_SURFACE_ELIPSOID}}_' + \
               f'f_{self.f:{formatting.MCNP_FMT_SURFACE_ELIPSOID}}_' + \
               f'g_{self.g:{formatting.MCNP_FMT_SURFACE_ELIPSOID}}_' + \
               f'x0_{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'y0_{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'z0_{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}'
    
    def __lt__(self, other: "Ellipsoid") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : Ellipsoid
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        vars_self = [self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.x0, self.y0, self.z0]
        vars_other = [other.a, other.b, other.c, other.d, other.e, other.f, other.g, other.x0, other.y0, other.z0]
        return _less_than_multi_variate(vars_self, vars_other)


class GeneralQuadratic(Surface):
    """General quadratic surface.

    Parameters
    ----------
    a, b, c, d, e, f, g, h, j, k : float, optional
        General quadratic surface coefficients, by default 0.0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, a: float = 0.0, b: float = 0.0, c: float = 0.0, d: float = 0.0, 
                 e: float = 0.0, f: float = 0.0, g: float = 0.0, h: float = 0.0, 
                 j: float = 0.0, k: float = 0.0, **kwargs: Any) -> None:
        """Initialise the GeneralQuadratic surface."""
        super().__init__(**kwargs)
        self.a: float = a
        self.b: float = b
        self.c: float = c
        self.d: float = d
        self.e: float = e
        self.f: float = f 
        self.g: float = g
        self.h: float = h
        self.j: float = j
        self.k: float = k

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of the general quadratic surface."""
        surface_type = 'GQ'
        surface_params = [
            f'{self.a:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}', 
            f'{self.b:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}', 
            f'{self.c:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}', 
            f'{self.d:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}', 
            f'{self.e:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}', 
            f'{self.f:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}', 
            f'{self.g:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}', 
            f'{self.h:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}', 
            f'{self.j:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}',
            f'{self.k:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}'
        ]
        return mcnp_surface(surface_type, surface_params, self)
    
    @property
    def openmc_surface(self) -> Any:
        """OpenMC surface object representation."""
        if not self._openmc_surface:
            self._openmc_surface = openmc.surface.Quadric(
                a=self.a,
                b=self.b,
                c=self.c,
                d=self.d,
                e=self.e,
                f=self.f,
                g=self.g,
                h=self.h,
                j=self.j,
                k=self.k,
                boundary_type=openmc_boundary_type(self)
            )

        return self._openmc_surface

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'generalquadratic_' + \
               f'a_{self.a:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}_' + \
               f'b_{self.b:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}_' + \
               f'c_{self.c:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}_' + \
               f'd_{self.d:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}_' + \
               f'e_{self.e:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}_' + \
               f'f_{self.f:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}_' + \
               f'g_{self.g:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}_' + \
               f'h_{self.h:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}_' + \
               f'j_{self.j:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}_' + \
               f'k_{self.k:{formatting.MCNP_FMT_SURFACE_GENERAL_QUADRATIC}}'

    def __lt__(self, other: "GeneralQuadratic") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : GeneralQuadratic
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        vars_self = [self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.h, self.j, self.k]
        vars_other = [other.a, other.b, other.c, other.d, other.e, other.f, other.g, other.h, other.j, other.k]
        return _less_than_multi_variate(vars_self, vars_other)


class ToriParralleltoXAxis(Surface):
    """Torus parallel to the X-axis.

    Parameters
    ----------
    x0, y0, z0 : float, optional
        Center coordinates, by default 0.0
    a, b, c : float, optional
        Radii parameters, by default 0.0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, x0: float = 0.0, y0: float = 0.0, z0: float = 0.0, 
                 a: float = 0.0, b: float = 0.0, c: float = 0.0, **kwargs: Any) -> None:
        """Initialise the ToriParralleltoXAxis surface."""
        super().__init__(**kwargs)
        self.x0: float = x0
        self.y0: float = y0
        self.z0: float = z0
        self.a: float = a
        self.b: float = b
        self.c: float = c

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of torus parallel to X-axis."""
        surface_type = 'TX'
        surface_params = [
            f'{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}',
            f'{self.a:{formatting.MCNP_FMT_SURFACE_TORI}}',
            f'{self.b:{formatting.MCNP_FMT_SURFACE_TORI}}',
            f'{self.c:{formatting.MCNP_FMT_SURFACE_TORI}}'
        ]
        return mcnp_surface(surface_type, surface_params, self)
    
    @property
    def openmc_surface(self) -> Any:
        """OpenMC surface object representation."""
        if not self._openmc_surface:
            self._openmc_surface = openmc.surface.XTorus(
                x0=self.x0,
                y0=self.y0,
                z0=self.z0,
                a=self.a,
                b=self.b,
                c=self.c,
                boundary_type=openmc_boundary_type(self)
            )

        return self._openmc_surface

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'toriparaplleltoxaxis_' + \
               f'x0_{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'y0_{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'z0_{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'a_{self.a:{formatting.MCNP_FMT_SURFACE_TORI}}_' + \
               f'b_{self.b:{formatting.MCNP_FMT_SURFACE_TORI}}_' + \
               f'c_{self.c:{formatting.MCNP_FMT_SURFACE_TORI}}'

    def __lt__(self, other: "ToriParralleltoXAxis") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : ToriParralleltoXAxis
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        vars_self = [self.x0, self.y0, self.z0, self.a, self.b, self.c]
        vars_other = [other.x0, other.y0, other.z0, other.a, other.b, other.c]
        return _less_than_multi_variate(vars_self, vars_other)


class ToriParralleltoYAxis(Surface):
    """Torus parallel to the Y-axis.

    Parameters
    ----------
    x0, y0, z0 : float, optional
        Center coordinates, by default 0.0
    a, b, c : float, optional
        Radii parameters, by default 0.0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, x0: float = 0.0, y0: float = 0.0, z0: float = 0.0, 
                 a: float = 0.0, b: float = 0.0, c: float = 0.0, **kwargs: Any) -> None:
        """Initialise the ToriParralleltoYAxis surface."""
        super().__init__(**kwargs)
        self.x0: float = x0
        self.y0: float = y0
        self.z0: float = z0
        self.a: float = a
        self.b: float = b
        self.c: float = c

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of torus parallel to Y-axis."""
        surface_type = 'TY'
        surface_params = [
            f'{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}', 
            f'{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}', 
            f'{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}', 
            f'{self.a:{formatting.MCNP_FMT_SURFACE_TORI}}', 
            f'{self.b:{formatting.MCNP_FMT_SURFACE_TORI}}', 
            f'{self.c:{formatting.MCNP_FMT_SURFACE_TORI}}'
        ]
        return mcnp_surface(surface_type, surface_params, self)

    @property
    def openmc_surface(self) -> Any:
        """OpenMC surface object representation."""
        if not self._openmc_surface:
            self._openmc_surface = openmc.surface.YTorus(
                x0=self.x0,
                y0=self.y0,
                z0=self.z0,
                a=self.a,
                b=self.b,
                c=self.c,
                boundary_type=openmc_boundary_type(self)
            )
        
        return self._openmc_surface

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'toriparaplleltoyaxis_' + \
               f'x0_{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'y0_{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'z0_{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'a_{self.a:{formatting.MCNP_FMT_SURFACE_TORI}}_' + \
               f'b_{self.b:{formatting.MCNP_FMT_SURFACE_TORI}}_' + \
               f'c_{self.c:{formatting.MCNP_FMT_SURFACE_TORI}}'

    def __lt__(self, other: "ToriParralleltoYAxis") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : ToriParralleltoYAxis
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        vars_self = [self.x0, self.y0, self.z0, self.a, self.b, self.c]
        vars_other = [other.x0, other.y0, other.z0, other.a, other.b, other.c]
        return _less_than_multi_variate(vars_self, vars_other)


class ToriParralleltoZAxis(Surface):
    """Torus parallel to the Z-axis.

    Parameters
    ----------
    x0, y0, z0 : float, optional
        Center coordinates, by default 0.0
    a, b, c : float, optional
        Radii parameters, by default 0.0
    **kwargs : Any
        Keyword arguments passed to :class:`Surface` (reflective, comment,
        transform, openmc_graveyard, number).
    """

    def __init__(self, x0: float = 0.0, y0: float = 0.0, z0: float = 0.0, 
                 a: float = 0.0, b: float = 0.0, c: float = 0.0, **kwargs: Any) -> None:
        """Initialise the ToriParralleltoZAxis surface."""
        super().__init__(**kwargs)
        self.x0: float = x0
        self.y0: float = y0
        self.z0: float = z0
        self.a: float = a
        self.b: float = b
        self.c: float = c

    @property
    def mcnp_card(self) -> MCNPCard:
        """MCNP card representation of torus parallel to Z-axis."""
        surface_type = 'TZ'
        surface_params = [
            f'{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}', 
            f'{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}', 
            f'{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}', 
            f'{self.a:{formatting.MCNP_FMT_SURFACE_TORI}}', 
            f'{self.b:{formatting.MCNP_FMT_SURFACE_TORI}}', 
            f'{self.c:{formatting.MCNP_FMT_SURFACE_TORI}}'
        ]
        return mcnp_surface(surface_type, surface_params, self)

    @property
    def openmc_surface(self) -> Any:
        """OpenMC surface object representation."""
        if not self._openmc_surface:
            self._openmc_surface = openmc.surface.ZTorus(
                x0=self.x0,
                y0=self.y0,
                z0=self.z0,
                a=self.a,
                b=self.b,
                c=self.c,
                boundary_type=openmc_boundary_type(self)
            )
        
        return self._openmc_surface

    def __hash_string__(self) -> str:
        """Unique hash string for surface identification."""
        return f'toriparaplleltozaxis_' + \
               f'x0_{self.x0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'y0_{self.y0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'z0_{self.z0:{formatting.MCNP_FMT_SURFACE_CARTESIAN}}_' + \
               f'a_{self.a:{formatting.MCNP_FMT_SURFACE_TORI}}_' + \
               f'b_{self.b:{formatting.MCNP_FMT_SURFACE_TORI}}_' + \
               f'c_{self.c:{formatting.MCNP_FMT_SURFACE_TORI}}'

    def __lt__(self, other: "ToriParralleltoZAxis") -> bool:
        """Return whether this surface orders before another of the same type.

        Used to sort surfaces of one type by their parameters.

        Parameters
        ----------
        other : ToriParralleltoZAxis
            Surface to compare against.

        Returns
        -------
        bool
            True if this surface sorts before ``other``.
        """
        vars_self = [self.x0, self.y0, self.z0, self.a, self.b, self.c]
        vars_other = [other.x0, other.y0, other.z0, other.a, other.b, other.c]
        return _less_than_multi_variate(vars_self, vars_other)


def _less_than_multi_variate(variables_1: List[float], variables_2: List[float]) -> bool:
    """Helper comparison function for multi-variable ordering.

    Parameters
    ----------
    variables_1 : list of float
        First variable set
    variables_2 : list of float
        Second variable set

    Returns
    -------
    bool
        True if variables_1 is strictly less than variables_2 lexicographically
    """
    for var_1, var_2 in zip(variables_1, variables_2):
        if var_1 < var_2:
            return True
        elif var_1 > var_2:
            return False
    return False


def _sort_surfaces_by_type(surfaces: List[Surface]) -> List[Surface]:
    """Sort list of surfaces by surface type and then parameters.

    Parameters
    ----------
    surfaces : list of Surface
        List of surface objects to sort

    Returns
    -------
    list of Surface
        Sorted list of surface objects
    """
    surfaces_sorted_by_type: Dict[Type[Surface], List[Surface]] = {}
    for surface in surfaces:
        if type(surface) not in surfaces_sorted_by_type:
            surfaces_sorted_by_type[type(surface)] = [surface]
        else:
            surfaces_sorted_by_type[type(surface)].append(surface)

    SORT_METHODS = {
        XPlane: sorted,
        YPlane: sorted,
        ZPlane: sorted,
        Plane: sorted,
        PlaneFromThreePoints: sorted,
        SphereAtOrigin: sorted,
        SphereOnXAxis: sorted,
        SphereOnYAxis: sorted,
        SphereOnZAxis: sorted,
        Sphere: sorted,
        CylinderParallelToXAxis: sorted,
        CylinderParallelToYAxis: sorted,
        CylinderParallelToZAxis: sorted,
        CylinderOnXAxis: sorted,
        CylinderOnYAxis: sorted,
        CylinderOnZAxis: sorted,
        ConeOnXAxis: sorted,
        ConeOnYAxis: sorted,
        ConeOnZAxis: sorted,
        ConeParallelToXAxis: sorted,
        ConeParallelToYAxis: sorted,
        ConeParallelToZAxis: sorted,
        GeneralQuadratic: sorted,
        ToriParralleltoXAxis: sorted,
        ToriParralleltoYAxis: sorted,
        ToriParralleltoZAxis: sorted
    }

    surfaces_sorted_by_type_and_order = {}
    for surface_type, surfaces_of_type in surfaces_sorted_by_type.items():
        surfaces_sorted_by_type_and_order[surface_type] = SORT_METHODS[surface_type](surfaces_of_type)

    surfaces_sorted = []
    for surface_type in [s for s in SORT_METHODS if s in surfaces_sorted_by_type_and_order]:
        surfaces_sorted.extend(surfaces_sorted_by_type_and_order[surface_type])
    
    if len(surfaces) != len(surfaces_sorted):
        msg = f'[ERROR] - surfaces returned from _sort_surfaces_by_type not equal to number provided'
        raise Exception(msg)

    return surfaces_sorted


def _sorted_place_holder(surfaces: List[Surface]) -> List[Surface]:
    """Placeholder function where no sort method is yet implemented.

    Parameters
    ----------
    surfaces : list of Surface
        Input surfaces

    Returns
    -------
    list of Surface
        Unchanged input surfaces
    """
    return surfaces


def _map_of_duplicate_surfaces() -> Dict[Type[Surface], Dict[Any, Any]]:
    """Initialize map for storing duplicate surfaces.

    Returns
    -------
    dict
        Dictionary mapping surface types to nested empty dictionaries
    """
    duplicate_surfaces = {
        XPlane: {},
        YPlane: {},
        ZPlane: {},
        Plane: {},
        PlaneFromThreePoints: {},
        SphereAtOrigin: {},
        SphereOnXAxis: {},
        SphereOnYAxis: {},
        SphereOnZAxis: {},
        Sphere: {},
        CylinderParallelToXAxis: {},
        CylinderParallelToYAxis: {},
        CylinderParallelToZAxis: {},
        CylinderOnXAxis: {},
        CylinderOnYAxis: {},
        CylinderOnZAxis: {},
        ConeOnXAxis: {},
        ConeOnYAxis: {},
        ConeOnZAxis: {},
        ConeParallelToXAxis: {},
        ConeParallelToYAxis: {},
        ConeParallelToZAxis: {},
        GeneralQuadratic: {},
        ToriParralleltoXAxis: {},
        ToriParralleltoYAxis: {},
        ToriParralleltoZAxis: {}
    }
    
    return duplicate_surfaces


MCNP_SURFACE_TYPES: Dict[str, Type[Surface]] = {
    'PX': XPlane,
    'PY': YPlane,
    'PZ': ZPlane,
    'P': Plane,
    'P3P': PlaneFromThreePoints,
    'C/X': CylinderParallelToXAxis,
    'C/Y': CylinderParallelToYAxis,
    'C/Z': CylinderParallelToZAxis,
    'CX': CylinderOnXAxis,
    'CY': CylinderOnYAxis,
    'CZ': CylinderOnZAxis,
    'KX': ConeOnXAxis,
    'KY': ConeOnYAxis,
    'KZ': ConeOnZAxis,
    'K/X': ConeParallelToXAxis,
    'K/Y': ConeParallelToYAxis,
    'K/Z': ConeParallelToZAxis,
    'SO': SphereAtOrigin,
    'S': Sphere,
    'SX': SphereOnXAxis,
    'SY': SphereOnYAxis,
    'SZ': SphereOnZAxis,
    'SQ': Ellipsoid,
    'GQ': GeneralQuadratic,
    'TX': ToriParralleltoXAxis,
    'TY': ToriParralleltoYAxis,
    'TZ': ToriParralleltoZAxis
}