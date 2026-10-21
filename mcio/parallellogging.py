"""Logging support for multiprocess decomposition.

Log records from worker processes are routed through a multiprocessing queue to
a :class:`logging.handlers.QueueListener` in the main process, which writes them
using the package's file handler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.queues import Queue as MPQueue

import logging
import logging.handlers
import multiprocessing
import os
import sys
import ctypes
from multiprocessing import Queue
import mcio
from mcio import LOGGER_FILE_HANDLER

def start_multiprocess_logging() -> tuple[MPQueue, logging.handlers.QueueListener]:
    """Route ``mcio`` logging through a queue for multiprocess runs.

    The package file handler is detached from the ``mcio`` logger and attached to
    a started :class:`logging.handlers.QueueListener`; a
    :class:`logging.handlers.QueueHandler` is added to the logger instead.

    Returns
    -------
    tuple[MPQueue, logging.handlers.QueueListener]
        Queue to pass to worker processes and the running listener.
    """
    logger = logging.getLogger("mcio")

    # Remove the FileHandler from the logger, but DO NOT close it.
    # The QueueListener will write through it.
    if LOGGER_FILE_HANDLER in logger.handlers:
        logger.removeHandler(LOGGER_FILE_HANDLER)

    # Create queue
    log_queue = Queue()

    # QueueListener writes using the SAME FileHandler
    listener = logging.handlers.QueueListener(log_queue, LOGGER_FILE_HANDLER)
    listener.start()

    # Workers and main process now log via QueueHandler
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger.addHandler(queue_handler)

    return log_queue, listener

def restore_main_process_logging() -> None:
    """Restore the original file handler on the ``mcio`` logger.

    Called after multiprocessing finishes: removes any queue handlers and
    reattaches the package file handler so the main process can continue
    logging normally.
    """
    from mcio import LOGGER_FILE_HANDLER
    logger = logging.getLogger("mcio")

    # Remove QueueHandler(s)
    for h in list(logger.handlers):
        if isinstance(h, logging.handlers.QueueHandler):
            logger.removeHandler(h)

    # Reattach the original file handler
    if LOGGER_FILE_HANDLER not in logger.handlers:
        logger.addHandler(LOGGER_FILE_HANDLER)

def configure_worker_logger(log_queue: MPQueue) -> None:
    """Configure the ``mcio`` logger in a worker process to log via a queue.

    On Windows, native crash dialog boxes and error reporting popups are also
    suppressed.

    Parameters
    ----------
    log_queue : MPQueue
        Queue returned by :func:`start_multiprocess_logging` in the main process.
    """
# Suppress native C++ crash dialog boxes on Windows
    if sys.platform == "win32":
        # SEM_FAILCRITICALERRORS (0x0001) | SEM_NOGPFAULTERRORBOX (0x0002)
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
        try:
            # Disable Windows Error Reporting (WER) popup UI
            ctypes.windll.wer.WerSetFlags(0x0020)
        except AttributeError:
            pass

    # enforce memory safety flags inside worker process runtimes
    # os.environ["MMGT_OPT"] = "0"
    # os.environ["CSF_NUM_THREADS"] = "1"
    # os.environ["OMP_NUM_THREADS"] = "1"

    logger = logging.getLogger("mcio")
    logger.handlers.clear()

    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger.addHandler(queue_handler)
    logger.setLevel(logging.INFO)

