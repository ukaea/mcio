"""Multiprocessing helpers for the CAD decomposition algorithms.

Contains the worker function passed to a process pool (or called directly in
single-core mode) to run a single :class:`DecompositionSolid`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import os
import logging
import traceback

from .ocp import STEPFileWriter
from .decompose import DecompositionSolid
from .ocp import extract_ocp_shape_type_from_shape


def _run_decomposer_solid(
    params: Sequence[Any],
) -> tuple[int, DecompositionSolid, str | None]:
    """Run a single decomposition solid, suitable for a multiprocess pool.

    The solid is run via ``DecompositionSolid.run()``. If a ``RecursionError``
    is raised, the traceback is written to
    ``decom_params.debug_trackback_directory/traceback_solid_<i>.txt``, the
    original solid is written to a STEP file in
    ``decom_params.debug_failed_to_convert_directory`` and the failure is
    logged. Any other exception propagates to the caller.

    Parameters
    ----------
    params : Sequence[Any]
        Sequence (list or tuple) of ``(i, n_total, padding_index,
        decomposition_solid, ...)``: the solid index, total number of solids,
        field width used to pad the index in log messages, and the
        :class:`DecompositionSolid` to run. Callers also pass a fifth ``debug``
        element, which is not used here.

    Returns
    -------
    tuple[int, DecompositionSolid, str | None]
        The solid index, the (run) decomposition solid, and ``None`` on
        success or a string describing the ``RecursionError`` on failure.
    """

    # stop parallel processes from locking the logging output file
    logger = logging.getLogger("mcio")

    # unpack params
    i: int = params[0]
    n_total: int = params[1]
    padding_index: int = params[2]
    decomposition_solid: DecompositionSolid = params[3]

    # store exceptions if generated
    try:
        # run decomposition solid
        decomposition_solid.run()

        # output sucess message to logger
        msg = f'[COMPLETE] : {i: {padding_index}} of {n_total} - time taken' +\
              f'= {decomposition_solid.profile_time_decomposition:.2e} seconds'
        logger.info(msg)

        return i, decomposition_solid, None

    except Exception as exception_from_solid:
        # write exception and traceback to text file
        if not os.path.isdir(
            decomposition_solid.decom_params.debug_trackback_directory
        ):
            os.mkdir(decomposition_solid.decom_params.debug_trackback_directory)

        with open(
            os.path.join(
                decomposition_solid.decom_params.debug_trackback_directory,
                f"traceback_solid_{i}.txt",
            ),
            "w",
        ) as f:
            f.write(f"\n\n=== Exception while exporting solid {i} ===\n")
            f.write(str(exception_from_solid) + "\n")
            f.write(traceback.format_exc())

        # write solid to step file
        if decomposition_solid._is_serialised:
            decomposition_solid.deserialise()
            decomposition_solid.reserialise_required = True

        step_file_writer = STEPFileWriter(decomposition_solid.decom_params)
        step_file_writer.add_solids([decomposition_solid.original_solid], f"cell_{i}")
        output_file_path = step_file_writer.write(
            os.path.join(
                decomposition_solid.decom_params.debug_failed_to_convert_directory,
                f"original_solid_{i}.step",
            )
        )

        # output failure message to logger
        msg = f"solid_{i} failed to convert - see {output_file_path}"
        msg_exception = f"solid_{i} exception :\n{exception_from_solid}"
        logger.fatal(msg)
        logger.fatal(msg_exception)

        # following failure, the decomposition solids must be manually serialised from multiprocess
        if decomposition_solid.reserialise_required:
            decomposition_solid.serialise()
            decomposition_solid.reserialise_required = False

        # return decomposition_solid with its exception
        return i, decomposition_solid, msg_exception
