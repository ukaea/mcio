"""Shared pytest configuration for the mcio test suite.
"""

import os
import sys

import pytest

# Windows only: cadquery imports nlopt (built against the MSVC C++ runtime)
# before casadi (which bundles the MinGW libstdc++/libgcc runtime). With that
# load order the interpreter segfaults at exit, after the tests have passed,
# turning a green run into exit code 139. Loading casadi first avoids it.
# conftest.py is imported before any test module, so this runs first.
if sys.platform == "win32":
    try:
        import casadi  # noqa: F401
    except ImportError:
        pass


def pytest_configure(config):
    """Stop the run if openmc is required (CI) but cannot be imported.

    Tests needing the openmc python api are skipped when it is missing, so a
    contributor without openmc still gets a clean run. CI sets
    ``MCIO_REQUIRE_OPENMC=1`` so a broken openmc install fails loudly there
    instead of silently skipping those tests.
    """
    if os.environ.get('MCIO_REQUIRE_OPENMC', '').lower() in ('1', 'true', 'yes'):
        try:
            import openmc  # noqa: F401
        except Exception as error:
            raise pytest.UsageError(
                f'MCIO_REQUIRE_OPENMC is set but openmc cannot be imported: {error}'
            ) from error
