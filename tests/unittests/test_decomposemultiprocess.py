"""
unittest: run_with_timeout - the decomposition process pool, which stops tasks
that run for too long and replaces workers which die.

Built in functions are used as tasks so they can be sent to spawned processes.
"""

import math
import os
import subprocess
import sys
import tempfile
import time
import unittest

from mcio.decomposemultiprocess import run_with_timeout


def run(function, tasks, cores=2, timeout=0.0):
    """Run tasks, returning (results, failures, number of on_result calls)."""
    failures = {}
    n_results = []

    def on_failure(index, task, reason):
        failures[index] = reason
        return "failed"

    results = run_with_timeout(
        function,
        tasks,
        cores=cores,
        timeout=timeout,
        on_failure=on_failure,
        on_result=lambda: n_results.append(1),
    )
    return results, failures, len(n_results)


class MCIOUnitTestRunWithTimeout(unittest.TestCase):
    def test_results_are_in_task_order(self):
        tasks = [9.0, 1.0, 16.0, 4.0, 25.0, 36.0]
        results, failures, n_results = run(math.sqrt, tasks, cores=3)
        self.assertEqual(results, [3.0, 1.0, 4.0, 2.0, 5.0, 6.0])
        self.assertEqual(failures, {})
        self.assertEqual(n_results, len(tasks))

    def test_task_over_time_limit_is_stopped_and_others_finish(self):
        time_start = time.monotonic()
        results, failures, n_results = run(
            time.sleep, [0.0, 60.0, 0.1, 0.0], cores=2, timeout=2.0
        )
        self.assertLess(time.monotonic() - time_start, 30.0)
        self.assertEqual(results, [None, "failed", None, None])
        self.assertEqual(list(failures), [1])
        self.assertIn("time limit of 2 seconds", failures[1])
        self.assertEqual(n_results, 4)

    def test_every_worker_stuck_still_finishes(self):
        # the case seen on large models: every worker stuck on its own solid
        results, failures, _ = run(time.sleep, [60.0, 60.0, 0.0], cores=2, timeout=1.0)
        self.assertEqual(results, ["failed", "failed", None])
        self.assertEqual(sorted(failures), [0, 1])

    def test_worker_which_dies_is_replaced(self):
        # exec runs each task as code: the second task kills its worker process
        tasks = ["pass", "import os; os._exit(3)", "pass", "pass"]
        results, failures, n_results = run(exec, tasks, cores=1)
        self.assertEqual(results, [None, "failed", None, None])
        self.assertIn("stopped unexpectedly", failures[1])
        self.assertEqual(n_results, 4)

    def test_exception_in_function_is_a_failure(self):
        results, failures, _ = run(math.sqrt, [-1.0])
        self.assertEqual(results, ["failed"])
        self.assertIn("ValueError", failures[0])

    def test_no_time_limit(self):
        results, failures, _ = run(time.sleep, [0.2, 0.2], timeout=0)
        self.assertEqual(results, [None, None])
        self.assertEqual(failures, {})

    def test_no_tasks(self):
        self.assertEqual(run(math.sqrt, [])[0], [])


class MCIOUnitTestWorkerLogFile(unittest.TestCase):
    def import_mcio(self, *argv):
        """Import mcio in a new interpreter, returning whether it opened the log file.

        sys.argv is replaced first, as a spawned worker has the main process
        sys.argv copied in before it imports anything.
        """
        code = "import sys; sys.argv = ['mcio', 'decompose']; import mcio"
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, "-c", code, *argv], cwd=tmp, check=True)
            return os.path.exists(os.path.join(tmp, "analysis", "logging.log"))

    def test_main_process_opens_log_file(self):
        self.assertTrue(self.import_mcio())

    def test_spawned_worker_does_not_open_log_file(self):
        # regression: spawned workers import mcio before parent_process() is
        # set, and each one truncated the main process log file
        self.assertFalse(self.import_mcio("--multiprocessing-fork"))


if __name__ == "__main__":
    unittest.main()
