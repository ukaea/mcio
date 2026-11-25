"""Compatibility layer between the different OCP (cadquery-ocp) releases.

The OCP bindings changed names between releases, and the PyPI and conda-forge
builds of the same OCP version do not always agree. The differences MCIO needs
to absorb are:

* ``topods_cast.Face`` and the other shape casts lose their ``_s`` suffix (conda-forge
  from OCP 7.9.2, PyPI wheels from OCP 8). OCP 7.8 only has the suffixed names.
* ``TDF_LabelSequence``, ``TopTools_IndexedMapOfShape`` and
  ``XCAFPrs_IndexedDataMapOfShapeStyle`` moved into ``OCP.collections`` (OCP 8).
* ``Bnd_Box.Get()`` no longer converts to a tuple (OCP 8).
* Other static methods (``ShapeTool_s`` etc.) may lose the suffix in a later
  release. :func:`add_missing_static_aliases` adds the suffixed name back, so the
  rest of MCIO can keep calling the ``_s`` spelling.

Everything OCP version dependent is resolved here once, so the rest of MCIO
imports from this module instead of choosing a spelling itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcio import logger

__all__ = [
    "TDF_LabelSequence",
    "TopTools_IndexedMapOfShape",
    "XCAFPrs_IndexedDataMapOfShapeStyle",
    "add_missing_static_aliases",
    "bnd_box_limits",
    "resolve_static",
    "topods_cast",
]

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from OCP.Bnd import Bnd_Box

# ---------------------------------------------------------------------------
# helpers, which do not need OCP so they can be tested on their own
# ---------------------------------------------------------------------------


def resolve_static(owner: Any, name: str) -> Callable[..., Any]:
    """Return a static method or function from ``owner`` under either spelling.

    The plain name is tried first (the OpenCascade spelling), then the name with
    the ``_s`` suffix that older bindings use.

    Parameters
    ----------
    owner : Any
        Class or module holding the static method.
    name : str
        Name without the ``_s`` suffix, e.g. ``'Compound'``.

    Returns
    -------
    Callable
        The static method.

    Raises
    ------
    AttributeError
        If neither spelling exists.
    """
    for candidate in (name, f"{name}_s"):
        method = getattr(owner, candidate, None)
        if method is not None:
            return method
    msg = f"{owner!r} has neither {name!r} nor {name + '_s'!r}"
    raise AttributeError(msg)


def add_missing_static_aliases(owner: Any, names: Iterable[str]) -> list[str]:
    """Give each static method its ``_s`` spelling if only the plain one exists.

    Parameters
    ----------
    owner : Any
        Class holding the static methods.
    names : Iterable[str]
        Names with the ``_s`` suffix, e.g. ``'ShapeTool_s'``.

    Returns
    -------
    list[str]
        The names that were added.
    """
    added = []
    for suffixed in names:
        if hasattr(owner, suffixed):
            continue
        method = getattr(owner, suffixed.removesuffix("_s"), None)
        if method is None:
            continue
        try:
            setattr(owner, suffixed, method)
        except (AttributeError, TypeError):
            # not every binding type accepts new attributes; callers then fail with
            # a clear AttributeError at the call site
            logger.warning(f"could not alias {suffixed} on {owner!r}")
            continue
        added.append(suffixed)
    return added


# ---------------------------------------------------------------------------
# OCP dependent part
# ---------------------------------------------------------------------------

try:
    import importlib

    from OCP.TopoDS import TopoDS

    # --- collection classes, which moved into OCP.collections in OCP 8 ---
    try:
        from OCP.TDF import TDF_LabelSequence
    except ImportError:
        from OCP.collections import Sequence_TDF_Label as TDF_LabelSequence

    try:
        from OCP.TopTools import TopTools_IndexedMapOfShape
    except ImportError:
        from OCP.collections import (
            IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as TopTools_IndexedMapOfShape,
        )

    try:
        from OCP.XCAFPrs import XCAFPrs_IndexedDataMapOfShapeStyle
    except ImportError:
        from OCP.collections import (
            IndexedDataMap_TopoDS_Shape_XCAFPrs_Style_TopTools_ShapeMapHasher as XCAFPrs_IndexedDataMapOfShapeStyle,
        )

    class _TopoDSCasts:
        """The ``TopoDS`` shape casts under one spelling, whatever the OCP version.

        Attributes are the OpenCascade names: ``Compound``, ``CompSolid``,
        ``Solid``, ``Shell``, ``Face``, ``Wire``, ``Edge`` and ``Vertex``. Each takes
        a ``TopoDS_Shape`` and returns the more specific type, raising
        ``Standard_TypeMismatch`` if the shape is not of that type.
        """

        def __init__(self) -> None:
            self.Compound = resolve_static(TopoDS, "Compound")
            self.CompSolid = resolve_static(TopoDS, "CompSolid")
            self.Solid = resolve_static(TopoDS, "Solid")
            self.Shell = resolve_static(TopoDS, "Shell")
            self.Face = resolve_static(TopoDS, "Face")
            self.Wire = resolve_static(TopoDS, "Wire")
            self.Edge = resolve_static(TopoDS, "Edge")
            self.Vertex = resolve_static(TopoDS, "Vertex")

    topods_cast = _TopoDSCasts()

    # --- static methods MCIO calls with the _s spelling ---
    # (module, class, static method names). If a release drops the suffix the name
    # is added back, so the call sites do not have to change.
    _STATICS_USED: tuple[tuple[str, str, tuple[str, ...]], ...] = (
        ("OCP.TDataStd", "TDataStd_Name", ("GetID_s", "Set_s")),
        ("OCP.XCAFDoc", "XCAFDoc_ColorTool", ("GetColor_s",)),
        (
            "OCP.XCAFDoc",
            "XCAFDoc_ShapeTool",
            (
                "GetShape_s",
                "GetSubShapes_s",
                "GetLocation_s",
                "GetComponents_s",
                "IsAssembly_s",
                "GetReferredShape_s",
            ),
        ),
        (
            "OCP.XCAFDoc",
            "XCAFDoc_DocumentTool",
            ("ShapeTool_s", "ColorTool_s", "SetLengthUnit_s"),
        ),
        ("OCP.XCAFApp", "XCAFApp_Application", ("GetApplication_s",)),
        ("OCP.STEPCAFControl", "STEPCAFControl_Controller", ("Init_s",)),
        (
            "OCP.Interface",
            "Interface_Static",
            (
                "SetIVal_s",
                "SetCVal_s",
                "SetRVal_s",
                "IVal_s",
                "CVal_s",
                "RVal_s",
            ),
        ),
        ("OCP.TopExp", "TopExp", ("MapShapes_s",)),
        (
            "OCP.BRep",
            "BRep_Tool",
            ("Curve_s", "Surface_s", "Tolerance_s", "Pnt_s"),
        ),
        (
            "OCP.BRepGProp",
            "BRepGProp",
            ("SurfaceProperties_s", "VolumeProperties_s"),
        ),
        (
            "OCP.BRepBndLib",
            "BRepBndLib",
            ("Add_s", "AddOptimal_s", "AddOBB_s"),
        ),
        ("OCP.ElSLib", "ElSLib", ("Value_s", "Parameters_s")),
        ("OCP.BRepTools", "BRepTools", ("Write_s", "Read_s", "UVBounds_s")),
    )

    for _module_name, _class_name, _names in _STATICS_USED:
        try:
            _owner = getattr(importlib.import_module(_module_name), _class_name)
        except (ImportError, AttributeError):
            continue
        _added = add_missing_static_aliases(_owner, _names)
        if _added:
            logger.info(f"[OCP COMPAT] added {_added} to {_class_name}")

    def bnd_box_limits(box: Bnd_Box) -> tuple[float, float, float, float, float, float]:
        """Extents of a bounding box as ``(x_min, y_min, z_min, x_max, y_max, z_max)``.

        Replaces ``Bnd_Box.Get()``, which OCP 8 no longer converts to a tuple. The
        gap of the box is included, as it is by ``Get()``.

        Parameters
        ----------
        box : Bnd_Box
            A non-void OCP bounding box.

        Returns
        -------
        tuple[float, float, float, float, float, float]
            Minimum then maximum coordinates.
        """
        low, high = box.CornerMin(), box.CornerMax()
        return low.X(), low.Y(), low.Z(), high.X(), high.Y(), high.Z()

except Exception as e:
    logger.warning("failed to load OCP compatibility layer")
    logger.warning(f"{e}")
