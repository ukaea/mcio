"""Profiling helpers.

Contains the :func:`function_timer` decorator used to time decomposition steps,
and placeholder dataclasses intended to hold profiling data.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import time
import dataclasses
from dataclasses import dataclass


def function_timer(func: Callable[..., Any]) -> Callable[..., tuple[Any, float]]:
    """Decorate a function so that it also returns its wall-clock run time.

    Parameters
    ----------
    func : Callable[..., Any]
        Function to be timed.

    Returns
    -------
    Callable[..., tuple[Any, float]]
        Wrapped function returning a tuple of the original result and the time
        taken in seconds.
    """
    def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, float]:
        """Call the wrapped function and time it.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to the wrapped function.
        **kwargs : Any
            Keyword arguments passed to the wrapped function.

        Returns
        -------
        tuple[Any, float]
            Result of the wrapped function and the time taken in seconds.
        """
        # record start time
        time_start = time.time()

        # call function wrapper has been applied to 
        result = func(*args, **kwargs)

        # calculate total time taken
        time_total = time.time() - time_start

        # return the result of the fucntion wrapper has been applied to, along with total time taken
        return result, time_total
    
    # return the wrapper
    return wrapper


@dataclass
class ProfilerDecompositionSolid:
    """Placeholder for profiling data of a single decomposed solid."""

    pass


@dataclass
class ProfilerDecomposer:
    """Placeholder for profiling data of a full decomposition run."""

    pass
