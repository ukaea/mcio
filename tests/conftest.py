"""Shared pytest configuration for the mcio test suite.
"""

import sys

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
