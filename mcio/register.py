"""This module contains classes to track the number of intsances being processed and
provide optional progress bars to show the user the number of items completed.
"""
from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Any

from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm.notebook import tqdm as tqdm_notebook

from .table import Table
from .utils import (
    in_notebook
)


class ProgressRegister:
    """
    A tracker which counts the number of instances of the `items_to_track`
    privded by the user. This class has an intial setup of the `items_to_track`,
    along with add or remote for whic a list of items in `items_to_track` were
    contained. The progress bar can be updated, so that when a given tracked item has
    no remaining counts, the number of items considered complete will icnrease by one.

    Parameters
    ----------
    items_to_track : list[Hashable]
        A list of hashable items whose progress should be tracked.
    show_progress : bool, optional
        If True, a progress bar is displayed and updated as items complete.
        Default is True.
    title : str, optional
        The label shown on the progress bar. Default is 'ProgressRegister'.

    Attributes
    ----------
    show_progress : bool
        Whether the progress bar is enabled.
    _progress_bar : None
        Return value of ``_setup_register`` (always None); the live progress
        bar is held in ``_progress``.
    _register : dict
        A mapping of each tracked item to its remaining count.
        When a count reaches zero, the item is considered complete.

    """

    def __init__(
        self,
        items_to_track: list[Hashable],
        show_progress: bool = True,
        title: str = "ProgressRegister",
    ) -> None:
        """Initialise the register, starting the progress bar if requested."""
        # store user defined parameters
        self.items_to_track: list[Hashable] = items_to_track
        self.show_progress: bool = show_progress
        self.title: str = title

        # count input items
        self.n_items_to_process: int = len(items_to_track)

        # check if running in notebook
        self._is_notebook: bool = in_notebook()
        self._progress: Progress | Any | None = None
        self._task_id: Any | None = None

        # setup optional progress bar
        self._progress_bar: None = self._setup_register() if show_progress else None

        # set up dictionary to track instances of each key
        self._register: dict[Hashable, int] = {item_to_track: 1 for item_to_track in items_to_track}
        self._n_complete: int = 0

    def _setup_register(self) -> None:
        """Create and start the progress bar.

        Uses a tqdm notebook bar when running in a notebook, otherwise a Rich
        progress bar. The bar is stored in ``self._progress``.
        """

        # set progress bar stage description
        desc_stage = f'[{self.title}]'

        # set up progress bar
        if self._is_notebook:
            desc = f'{desc_stage: <30}'
            self._progress = tqdm_notebook(
                total=self.n_items_to_process,
                desc=self.title,
                unit="item",
            )
        else:

            desc_stage = f'[{self.title}]'
            desc = f'[cyan]{desc_stage: <30}[/cyan]'

            self._progress = Progress(
                TextColumn(desc),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            self._progress.start()
            self._task_id = self._progress.add_task(
                "tracking", 
                total=self.n_items_to_process, 
                completed=0
            )

    def add(self, items_added: Iterable[Hashable]) -> None:
        """Increment the counter for each item in `items_added`.

        Parameters
        ----------
        items_added : Iterable[Hashable]
            Items whose counters should be incremented.
        """
        for item_added in items_added:
            if self._register[item_added] == 0:
                self._n_complete -= 1
            self._register[item_added] += 1

    def remove(self, items_removed: Iterable[Hashable]) -> None:
        """Decrement the counter for each item in `items_removed`.

        Parameters
        ----------
        items_removed : Iterable[Hashable]
            Items whose counters should be decremented.
        """
        for item_removed in items_removed:
            self._register[item_removed] -= 1
            if self._register[item_removed] == 0:
                self._n_complete += 1

    def update(self) -> None:
        """Update the progress bar to reflect the number of completed items.

        An item is considered complete when its counter reaches zero.
        This method reads the completed-item count and updates the progress bar
        accordingly, stopping it once all items are complete.
        """
        # If show_progress has been set to True, but progress bar is not initialized
        if self.show_progress and self._progress is None:
            self._setup_register()

        # Number of completed items (maintained incrementally by add/remove, so
        # update() is O(1) rather than O(number of tracked items))
        n_complete = self._n_complete

        if self.show_progress and self._progress is not None:
            if self._is_notebook:
                # tqdm update requires a delta
                self._progress.n = n_complete
                self._progress.refresh()
            else:
                # rich update
                self._progress.update(self._task_id, completed=n_complete)

        # Check if all tasks are complete
        if n_complete == self.n_items_to_process and self._progress is not None:
            if self._is_notebook:
                self._progress.close()
            else:
                self._progress.stop()

            self._progress = None
            self._task_id = None

    def __str__(self) -> str:
        """Return the register items and their current counts as a table.

        Returns
        -------
        str
            Table of items with their current counts as a string.
        """

        # generate rows for table
        rows: list[list[Any]] = []
        for item, count in self._register.items():
            rows.append([f'{item}', count])

        # set up table with rows and headers
        table: Table = Table(rows=rows, headers=['item', 'count'])

        # return table as string
        return table.as_string
