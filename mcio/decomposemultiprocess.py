"""Multiprocessing helpers for the CAD decomposition algorithms.

Contains the worker function passed to a process pool (or called directly in
single-core mode) to run a single :class:`DecompositionSolid`, and
:func:`run_with_timeout`, a process pool which stops any task that runs for
too long.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import time
import traceback
from collections.abc import Callable, Iterable, Sequence
from multiprocessing.connection import Connection, wait
from typing import Any

# import from where it is defined: decompose imports this module part way through loading
from .decomposesolid import DecompositionSolid
from .ocp import STEPFileWriter


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
        msg = (
            f"[COMPLETE] : {i: {padding_index}} of {n_total} - time taken"
            + f"= {decomposition_solid.profile_time_decomposition:.2e} seconds"
        )
        logger.info(msg)

        return i, decomposition_solid, None

    except Exception as exception_from_solid:
        msg_exception = save_failed_solid(
            i, decomposition_solid, str(exception_from_solid), traceback.format_exc()
        )

        # return decomposition_solid with its exception
        return i, decomposition_solid, msg_exception


def save_failed_solid(
    i: int,
    decomposition_solid: DecompositionSolid,
    reason: str,
    traceback_text: str = "",
) -> str:
    """Save a solid which failed to decompose, and log the failure.

    The reason and traceback are written to
    ``decom_params.debug_trackback_directory/traceback_solid_<i>.txt`` and the
    original solid to ``decom_params.debug_failed_to_convert_directory``. The
    solid is left serialised if it was serialised on entry.

    Parameters
    ----------
    i : int
        Index of the solid.
    decomposition_solid : DecompositionSolid
        The solid which failed.
    reason : str
        Description of the failure (the exception message, or why it was stopped).
    traceback_text : str, optional
        Formatted traceback, by default ""

    Returns
    -------
    str
        The failure message returned in place of the exception.
    """
    logger = logging.getLogger("mcio")
    decom_params = decomposition_solid.decom_params

    # write reason and traceback to text file
    os.makedirs(decom_params.debug_trackback_directory, exist_ok=True)
    with open(
        os.path.join(
            decom_params.debug_trackback_directory, f"traceback_solid_{i}.txt"
        ),
        "w",
    ) as f:
        f.write(f"\n\n=== Exception while exporting solid {i} ===\n")
        f.write(reason + "\n")
        f.write(traceback_text)

    # write solid to step file (a failed run() leaves the solid deserialised,
    # with reserialise_required set if it arrived serialised)
    was_serialised = decomposition_solid._is_serialised
    if was_serialised:
        decomposition_solid.deserialise()

    step_file_writer = STEPFileWriter(decom_params)
    step_file_writer.add_solids([decomposition_solid.original_solid], f"cell_{i}")
    output_file_path = step_file_writer.write(
        os.path.join(
            decom_params.debug_failed_to_convert_directory,
            f"original_solid_{i}.step",
        )
    )

    # output failure message to logger
    msg = f"solid_{i} failed to convert - see {output_file_path}"
    msg_exception = f"solid_{i} exception :\n{reason}"
    logger.fatal(msg)
    logger.fatal(msg_exception)

    # serialise again so the solid can be returned from a worker process
    if was_serialised or decomposition_solid.reserialise_required:
        decomposition_solid.serialise()
        decomposition_solid.reserialise_required = False

    return msg_exception


def _pool_worker(
    connection: Connection,
    function: Callable[[Any], Any],
    initializer: Callable[..., None] | None,
    initargs: tuple[Any, ...],
) -> None:
    """Worker loop for :func:`run_with_timeout`.

    Receives ``(index, task)`` pairs until ``None`` is received. For each task,
    sends ``None`` when it starts (so start up time is not counted against the
    time limit), then ``(True, result)`` or, if ``function`` raises, ``(False,
    traceback)``.
    """
    if initializer is not None:
        initializer(*initargs)

    while True:
        message = connection.recv()
        if message is None:
            return
        _, task = message
        connection.send(None)
        try:
            connection.send((True, function(task)))
        except Exception:  # noqa: BLE001 - any error is reported back as a failed task
            connection.send((False, traceback.format_exc()))


class _PoolProcess:
    """A worker process of :func:`run_with_timeout` and the task it is running."""

    def __init__(self, context: Any, args: tuple[Any, ...]) -> None:
        self.connection, child_connection = context.Pipe()
        self.process = context.Process(
            target=_pool_worker, args=(child_connection, *args), daemon=True
        )
        self.process.start()
        child_connection.close()
        self.index: int | None = None
        self.time_start: float | None = None

    def submit(self, index: int, task: Any) -> None:
        """Send a task to the process; its clock starts when the process replies."""
        self.index, self.time_start = index, None
        self.connection.send((index, task))

    def stop(self, wait_time: float = 0.0) -> None:
        """Stop the process, after waiting up to ``wait_time`` seconds for it to exit."""
        if wait_time > 0:
            self.process.join(wait_time)
        if self.process.is_alive():
            self.process.terminate()
        self.process.join()
        self.connection.close()


def run_with_timeout(
    function: Callable[[Any], Any],
    tasks: Iterable[Any],
    cores: int,
    timeout: float,
    on_failure: Callable[[int, Any, str], Any],
    initializer: Callable[..., None] | None = None,
    initargs: tuple[Any, ...] = (),
    on_result: Callable[[], None] | None = None,
) -> list[Any]:
    """Run ``function`` on each task in a process pool, with a per-task time limit.

    Unlike ``concurrent.futures.ProcessPoolExecutor``, a task that runs for
    longer than ``timeout`` seconds (for example stuck in an OpenCASCADE
    algorithm that never returns) has its worker process stopped and replaced,
    so the remaining tasks still run. A worker process which dies (e.g. a
    segmentation fault) is also replaced.

    Parameters
    ----------
    function : Callable[[Any], Any]
        Picklable function called with each task in a worker process.
    tasks : Iterable[Any]
        Picklable tasks.
    cores : int
        Number of worker processes.
    timeout : float
        Seconds a single task may run for, 0 or less for no limit.
    on_failure : Callable[[int, Any, str], Any]
        Called in this process with ``(index, task, reason)`` when a task times
        out, its worker dies, or ``function`` raises; its return value is used as
        the result of that task.
    initializer : Callable[..., None] | None, optional
        Called with ``initargs`` at the start of each worker process.
    initargs : tuple[Any, ...], optional
        Arguments for ``initializer``.
    on_result : Callable[[], None] | None, optional
        Called in this process each time a task finishes, e.g. to advance a
        progress bar.

    Returns
    -------
    list[Any]
        The results, in the same order as ``tasks``.
    """
    tasks = list(tasks)
    results: list[Any] = [None] * len(tasks)
    context = multiprocessing.get_context()
    args = (function, initializer, initargs)
    pending = iter(enumerate(tasks))
    n_remaining = len(tasks)
    workers: list[_PoolProcess] = []

    def finish(worker: _PoolProcess, result: Any) -> None:
        """Store the result of the worker's task and mark the worker idle."""
        nonlocal n_remaining
        results[worker.index] = result
        worker.index = None
        n_remaining -= 1
        if on_result is not None:
            on_result()

    def submit_next(worker: _PoolProcess) -> None:
        """Send the next pending task, if any, to an idle worker."""
        next_task = next(pending, None)
        if next_task is not None:
            worker.submit(*next_task)

    try:
        for _ in range(min(max(cores, 1), len(tasks))):
            worker = _PoolProcess(context, args)
            workers.append(worker)
            submit_next(worker)

        while n_remaining:
            busy = [worker for worker in workers if worker.index is not None]

            # wait for a result, a worker to die, or the next deadline
            wait_time = None
            started = [w.time_start for w in busy if w.time_start is not None]
            if timeout > 0 and started:
                wait_time = max(0.0, min(started) + timeout - time.monotonic())
            ready = set(
                wait(
                    [w.connection for w in busy] + [w.process.sentinel for w in busy],
                    timeout=wait_time,
                )
            )

            for i_worker, worker in enumerate(workers):
                if worker.index is None:
                    continue

                reason = None
                replace_worker = True
                if worker.connection in ready:
                    try:
                        message = worker.connection.recv()
                    except (EOFError, OSError):
                        reason = "worker process stopped unexpectedly"
                    else:
                        if message is None:
                            worker.time_start = time.monotonic()
                            continue
                        succeeded, result = message
                        if succeeded:
                            finish(worker, result)
                            submit_next(worker)
                            continue
                        reason = f"exception in worker process:\n{result}"
                        replace_worker = False
                elif worker.process.sentinel in ready:
                    reason = (
                        "worker process stopped unexpectedly"
                        + f" (exit code {worker.process.exitcode})"
                    )
                elif (
                    timeout > 0
                    and worker.time_start is not None
                    and time.monotonic() - worker.time_start >= timeout
                ):
                    reason = (
                        f"stopped after exceeding the time limit of {timeout:g} seconds"
                    )

                if reason is None:
                    continue

                # record the failure, and replace the worker if stuck or dead
                index = worker.index
                if replace_worker:
                    worker.stop()
                    worker = workers[i_worker] = _PoolProcess(context, args)
                    worker.index = index
                finish(worker, on_failure(index, tasks[index], reason))
                submit_next(worker)

        # ask workers to exit, so their queued log records are flushed
        for worker in workers:
            worker.connection.send(None)
        for worker in workers:
            worker.stop(wait_time=10.0)
    finally:
        for worker in workers:
            if not worker.connection.closed:
                worker.stop()

    return results
