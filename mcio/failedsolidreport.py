"""Report on a solid which failed to decompose.

When a solid fails, :func:`write_failed_solid_report` writes a markdown report with what
is known about the failure, and about the solid, to help find out why it failed: what
the solid is in the CAD file, how it failed on each decomposition attempt, a summary of
its geometry with anything suspicious (such as near planar cones and degenerate tori),
the debugging files written, and a possible cause where the error is a known one.

Writing the report never raises: each section is gathered separately, and one which
cannot be is reported as unavailable, so a report can never hide the original failure.
"""

from __future__ import annotations

import collections
import math
import os
import platform
import re
import time
from importlib import metadata
from typing import TYPE_CHECKING, Any

from mcio import logger

try:
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.Geom import Geom_ConicalSurface, Geom_ToroidalSurface
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_FACE

    from .ocpfilereadwrite import (
        _get_ocp_surface_from_ocp_face,
        extract_ocp_shape_type_from_shape,
    )
    from .ocpproperties import _surface_area_of_ocp_solids, _volume_of_ocp_solids
except ImportError:
    pass

if TYPE_CHECKING:
    from .decomposesolid import DecompositionSolid

#: surface types which MCIO can convert
SUPPORTED_SURFACES = (
    "Geom_Plane",
    "Geom_CylindricalSurface",
    "Geom_ConicalSurface",
    "Geom_SphericalSurface",
    "Geom_ToroidalSurface",
)

#: a cone with a semi-angle within this many degrees of 90 is nearly a flat disc
NEAR_PLANAR_CONE_DEGREES = 0.1

#: a cone with a semi-angle below this many degrees is nearly a cylinder
NEAR_CYLINDRICAL_CONE_DEGREES = 0.1

#: faces with an area below this are counted as tiny (in model units squared)
TINY_FACE_AREA = 1e-6

#: how many of each suspicious feature to list
MAX_LISTED = 10

#: (text found in the error, possible cause), checked in order
POSSIBLE_CAUSES: tuple[tuple[str, str], ...] = (
    (
        "no common solids produced",
        "The common of the solid and the half space of an auxiliary surface returned nothing."
        " Check the solid for near planar cones or other extreme surfaces (see below).",
    ),
    (
        "volume not conserved",
        "The pieces made by cutting the solid with an auxiliary surface do not add up to"
        " the volume of the solid, so a boolean operation lost or gained volume."
        " Check the debugging STEP file named in the error.",
    ),
    (
        "required number of solids",
        "A boolean cut or common gave no solids when one was needed. This often means the"
        " half space tool built from a face is invalid, for example for a nearly flat cone"
        " (see below).",
    ),
    (
        "offet_point",
        "The half space of a face could not be built because the face has an internal"
        " orientation. Check the faces of the solid in the CAD model.",
    ),
    (
        "did not cut new volume",
        "A half space made from a face did not cut any volume from its bounding box."
        " Check for faces which are extreme in size or shape.",
    ),
    (
        "time limit",
        "The solid took longer than the time limit (split_solid_timeout for each attempt)."
        " It may have too many faces, or a boolean operation may be stuck on difficult"
        " geometry.",
    ),
    (
        "stopped unexpectedly",
        "The worker process stopped without raising an error, usually a crash inside"
        " OpenCascade on difficult geometry.",
    ),
    (
        "unsupported surface type",
        "The solid has faces which are not planes, cylinders, cones, spheres or tori,"
        " for example splines or surfaces of revolution. Simplify these in the CAD model.",
    ),
    (
        "degenerate torus",
        "The solid has tori with a minor radius larger than the major radius, which"
        " cannot be converted. Replace them in the CAD model (for example with a sphere"
        " or a torus with a smaller minor radius).",
    ),
    (
        "surface type not available",
        "The solid has faces which are not planes, cylinders, cones, spheres or tori,"
        " for example splines or surfaces of revolution. Simplify these in the CAD model.",
    ),
)


def failure_type(reason: str) -> str:
    """Classify a failure from its message.

    Parameters
    ----------
    reason : str
        The failure message.

    Returns
    -------
    str
        ``"timeout"``, ``"worker stopped"`` or ``"error"``.
    """
    if "time limit" in reason:
        return "timeout"
    if "stopped unexpectedly" in reason:
        return "worker stopped"
    return "error"


def possible_causes(reason: str) -> list[str]:
    """Return the possible causes which match the text of a failure message.

    Parameters
    ----------
    reason : str
        The failure message, which may hold the errors of several attempts.

    Returns
    -------
    list[str]
        Each matching possible cause once, in the order of ``POSSIBLE_CAUSES``.
    """
    return [cause for text, cause in POSSIBLE_CAUSES if text in reason]


def debug_files_in(reason: str) -> list[str]:
    """Find the debugging STEP files named in a failure message.

    Parameters
    ----------
    reason : str
        The failure message, which names files as ``see <path>.stp``.

    Returns
    -------
    list[str]
        The paths found, without repeats, in order.
    """
    paths = re.findall(r"see\s+(\S+?\.(?:stp|step))", reason, flags=re.IGNORECASE)
    return list(dict.fromkeys(paths))


def geometry_summary(solid: Any) -> dict[str, Any]:
    """Summarise the geometry of a solid, and anything suspicious in it.

    Each item is gathered on its own, and one which fails is left out.

    Parameters
    ----------
    solid : TopoDS_Solid
        The original solid.

    Returns
    -------
    dict[str, Any]
        Volume, surface area, volume to area ratio, valid flag, face counts by surface
        type, and the suspicious features (near planar and near cylindrical cones,
        degenerate tori, unsupported surfaces and tiny faces).
    """
    summary: dict[str, Any] = {}

    def gather(key: str, function: Any) -> None:
        try:
            summary[key] = function()
        except Exception as exception:
            summary[key] = None
            summary.setdefault("errors", []).append(f"{key}: {exception}")

    gather("volume", lambda: _volume_of_ocp_solids([solid]))
    gather("surface_area", lambda: _surface_area_of_ocp_solids([solid]))
    if summary.get("volume") and summary.get("surface_area"):
        summary["volume_to_area"] = summary["volume"] / summary["surface_area"]
    gather("valid", lambda: BRepCheck_Analyzer(solid).IsValid())

    def bounding_box() -> dict[str, list[float]]:
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib

        box = Bnd_Box()
        BRepBndLib.Add_s(solid, box)
        low, high = box.CornerMin(), box.CornerMax()
        return {
            "min": [low.X(), low.Y(), low.Z()],
            "max": [high.X(), high.Y(), high.Z()],
        }

    gather("bounding_box", bounding_box)

    def faces() -> list[Any]:
        return extract_ocp_shape_type_from_shape(
            solid, TopAbs_FACE, cast_from_shape=True
        )

    face_list: list[Any] = []
    try:
        face_list = faces()
    except Exception as exception:
        summary.setdefault("errors", []).append(f"faces: {exception}")
    summary["n_faces"] = len(face_list)

    counts: collections.Counter[str] = collections.Counter()
    near_planar: list[dict[str, float]] = []
    near_cylindrical: list[dict[str, float]] = []
    degenerate_tori: list[dict[str, float]] = []
    n_tiny = 0
    for face in face_list:
        try:
            surface = _get_ocp_surface_from_ocp_face(face)
        except Exception:
            counts["unreadable surface"] += 1
            continue
        counts[type(surface).__name__] += 1

        if isinstance(surface, Geom_ConicalSurface):
            cone = surface.Cone()
            semi_angle = math.degrees(cone.SemiAngle())
            entry = {
                "semi_angle_degrees": semi_angle,
                "reference_radius": cone.RefRadius(),
            }
            if abs(semi_angle) > 90.0 - NEAR_PLANAR_CONE_DEGREES:
                near_planar.append(entry)
            elif abs(semi_angle) < NEAR_CYLINDRICAL_CONE_DEGREES:
                near_cylindrical.append(entry)
        elif isinstance(surface, Geom_ToroidalSurface):
            if surface.MinorRadius() > surface.MajorRadius():
                degenerate_tori.append(
                    {
                        "major_radius": surface.MajorRadius(),
                        "minor_radius": surface.MinorRadius(),
                    }
                )

        try:
            properties = GProp_GProps()
            BRepGProp.SurfaceProperties_s(face, properties)
            if properties.Mass() < TINY_FACE_AREA:
                n_tiny += 1
        except Exception:
            pass

    summary["face_counts"] = dict(counts)
    summary["unsupported_surfaces"] = {
        name: count for name, count in counts.items() if name not in SUPPORTED_SURFACES
    }
    summary["near_planar_cones"] = near_planar
    summary["near_cylindrical_cones"] = near_cylindrical
    summary["degenerate_tori"] = degenerate_tori
    summary["n_tiny_faces"] = n_tiny
    return summary


def failed_solid_report_data(
    index: int,
    decomposition_solid: DecompositionSolid,
    reason: str,
    traceback_text: str = "",
    files: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Gather the information for the report on a failed solid.

    Parameters
    ----------
    index : int
        Index of the solid, as used in the log and the debugging file names.
    decomposition_solid : DecompositionSolid
        The solid which failed (deserialised).
    reason : str
        The failure message.
    traceback_text : str, optional
        Formatted traceback, by default ""
    files : dict[str, str], optional
        Names and paths of files written for the failure, by default None

    Returns
    -------
    dict[str, Any]
        The report data, see :func:`failed_solid_report_text`.
    """
    data: dict[str, Any] = {
        "index": index,
        "reason": reason,
        "failure_type": failure_type(reason),
        "possible_causes": possible_causes(reason),
        "traceback": traceback_text,
        "files": dict(files or {}),
        "debug_files": debug_files_in(reason),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # identity of the solid in the CAD file
    data["name"] = getattr(decomposition_solid, "name", None)
    data["structure"] = getattr(decomposition_solid, "structure_in_input_file", None)
    rgb = getattr(decomposition_solid, "rgb", None)
    data["colour"] = str(rgb) if rgb else None
    material = getattr(decomposition_solid, "material", None)
    data["material"] = getattr(material, "name", str(material)) if material else None

    # decomposition attempts, and the time allowed
    data["attempts"] = list(getattr(decomposition_solid, "attempts", []) or [])
    params = getattr(decomposition_solid, "decom_params", None)
    if params is not None:
        data["n_attempts"] = params.n_attempts
        data["time_limit_per_attempt"] = params.split_solid_timeout

    # geometry of the solid
    try:
        data["geometry"] = geometry_summary(decomposition_solid._solid_original)
    except Exception as exception:
        data["geometry"] = {"errors": [f"geometry: {exception}"]}

    # versions
    versions = {"python": platform.python_version()}
    for package in ("mcio", "cadquery-ocp", "cadquery"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    data["versions"] = versions
    return data


def _format(value: Any, spec: str = ".6g") -> str:
    """Format a number for the report, with '-' for a missing value."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    return format(value, spec) if isinstance(value, (int, float)) else str(value)


def failed_solid_report_text(data: dict[str, Any]) -> str:
    """Build the markdown text of the report on a failed solid.

    Parameters
    ----------
    data : dict[str, Any]
        Data from :func:`failed_solid_report_data`.

    Returns
    -------
    str
        The report.
    """
    lines = [f"# Failed solid {data['index']}", ""]
    lines.append(f"Generated {data['generated']}")
    lines.append("")

    lines += ["## Solid", ""]
    lines.append(f"- name: {data.get('name') or '-'}")
    structure = data.get("structure")
    lines.append(f"- CAD structure: {' / '.join(structure) if structure else '-'}")
    lines.append(f"- colour: {data.get('colour') or '-'}")
    lines.append(f"- material: {data.get('material') or '-'}")

    lines += ["", "## Failure", ""]
    lines.append(f"- type: {data['failure_type']}")
    lines.append("")
    lines.append("```")
    lines.append(data["reason"].strip())
    lines.append("```")
    if data["possible_causes"]:
        lines += ["", "Possible cause:", ""]
        lines += [f"- {cause}" for cause in data["possible_causes"]]

    lines += ["", "## Decomposition attempts", ""]
    if data["attempts"]:
        lines.append("| attempt | time (s) | result |")
        lines.append("|---|---|---|")
        for attempt in data["attempts"]:
            # a | in the text would break the table
            result = (attempt["error"] or "converted").replace("\n", " ")
            result = result.replace("|", "/")
            lines.append(
                f"| {attempt['attempt']} | {_format(attempt['time'], '.1f')} | {result[:160]} |"
            )
        lines += ["", "Parameter values used:", ""]
        names = list(data["attempts"][0]["values"])
        lines.append(
            "| parameter | "
            + " | ".join(f"attempt {a['attempt']}" for a in data["attempts"])
            + " |"
        )
        lines.append("|---|" + "---|" * len(data["attempts"]))
        for name in names:
            lines.append(
                f"| {name} | "
                + " | ".join(_format(a["values"].get(name)) for a in data["attempts"])
                + " |"
            )
    else:
        lines.append(
            "No attempt information, the solid was stopped before it could report"
            " (a time limit or a stopped worker)."
        )
    if data.get("n_attempts") is not None:
        lines.append("")
        lines.append(
            f"{data['n_attempts']} attempts allowed, time limit "
            f"{_format(data.get('time_limit_per_attempt'))} s for each attempt"
            " (0 is no limit)."
        )

    geometry = data["geometry"]
    lines += ["", "## Geometry of the original solid", ""]
    box = geometry.get("bounding_box")
    if box:
        size = [high - low for low, high in zip(box["min"], box["max"])]
        lines.append(
            f"- bounding box min ({', '.join(_format(v) for v in box['min'])}),"
            f" max ({', '.join(_format(v) for v in box['max'])}),"
            f" size ({', '.join(_format(v) for v in size)})"
        )
    lines.append(f"- volume: {_format(geometry.get('volume'))}")
    lines.append(f"- surface area: {_format(geometry.get('surface_area'))}")
    lines.append(
        f"- volume / surface area: {_format(geometry.get('volume_to_area'))}"
        " (a solid below the degenerate solid limit, bool_degenerate_min_area_to_vol_ratio_values,"
        " would be filtered as a sliver)"
    )
    lines.append(f"- valid (BRepCheck): {_format(geometry.get('valid'))}")
    lines.append(f"- faces: {geometry.get('n_faces', '-')}")
    for name, count in sorted((geometry.get("face_counts") or {}).items()):
        lines.append(f"  - {name}: {count}")

    lines += ["", "## Suspicious features", ""]
    found = False
    for key, title, columns in (
        (
            "near_planar_cones",
            f"cones within {NEAR_PLANAR_CONE_DEGREES} degrees of flat (semi-angle near 90),"
            " which make an extreme half space tool for splitting",
            ("semi_angle_degrees", "reference_radius"),
        ),
        (
            "near_cylindrical_cones",
            f"cones within {NEAR_CYLINDRICAL_CONE_DEGREES} degrees of a cylinder",
            ("semi_angle_degrees", "reference_radius"),
        ),
        (
            "degenerate_tori",
            "tori with a minor radius larger than the major radius",
            ("major_radius", "minor_radius"),
        ),
    ):
        items = geometry.get(key) or []
        if items:
            found = True
            lines.append(f"- {len(items)} {title}:")
            for item in items[:MAX_LISTED]:
                lines.append(
                    "  - " + ", ".join(f"{c} = {_format(item[c])}" for c in columns)
                )
            if len(items) > MAX_LISTED:
                lines.append(f"  - ... and {len(items) - MAX_LISTED} more")
    unsupported = geometry.get("unsupported_surfaces") or {}
    if unsupported:
        found = True
        lines.append(
            "- unsupported surface types: "
            + ", ".join(f"{n} x {c}" for n, c in unsupported.items())
        )
    if geometry.get("n_tiny_faces"):
        found = True
        lines.append(
            f"- {geometry['n_tiny_faces']} faces with an area below {TINY_FACE_AREA:g}"
        )
    if not found:
        lines.append("None found.")
    for error in geometry.get("errors", []):
        lines.append(f"- could not gather {error}")

    lines += ["", "## Files", ""]
    for name, path in data["files"].items():
        lines.append(f"- {name}: `{path}`")
    for path in data["debug_files"]:
        lines.append(f"- named in the error: `{path}`")
    if not data["files"] and not data["debug_files"]:
        lines.append("None.")

    lines += ["", "## Versions", ""]
    for name, version in data["versions"].items():
        lines.append(f"- {name}: {version}")

    if data["traceback"].strip():
        lines += ["", "## Traceback", "", "```", data["traceback"].strip(), "```"]
    lines.append("")
    return "\n".join(lines)


def write_failed_solid_report(
    index: int,
    decomposition_solid: DecompositionSolid,
    reason: str,
    traceback_text: str = "",
    files: dict[str, str] | None = None,
    directory: str | os.PathLike | None = None,
) -> str | None:
    """Write the report on a failed solid.

    This never raises. If the report cannot be written the reason is logged and None
    is returned, so the original failure is never hidden.

    Parameters
    ----------
    index : int
        Index of the solid, as used in the log and the debugging file names.
    decomposition_solid : DecompositionSolid
        The solid which failed (deserialised).
    reason : str
        The failure message.
    traceback_text : str, optional
        Formatted traceback, by default ""
    files : dict[str, str], optional
        Names and paths of files written for the failure, by default None
    directory : str or os.PathLike, optional
        Folder for the report, by default
        ``decom_params.debug_failed_solid_report_directory``.

    Returns
    -------
    str or None
        Path of the report, or None if it could not be written.
    """
    try:
        if directory is None:
            directory = (
                decomposition_solid.decom_params.debug_failed_solid_report_directory
            )
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"solid_{index}_report.md")
        data = failed_solid_report_data(
            index, decomposition_solid, reason, traceback_text, files
        )
        with open(path, "w", encoding="utf-8") as report_file:
            report_file.write(failed_solid_report_text(data))
        return path
    except Exception as exception:
        logger.warning(
            f"could not write the report for failed solid {index}: {exception}"
        )
        return None
