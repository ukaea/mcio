"""Custom assertions for unit testing
"""

import math
import unittest



class CustomAssertions():
    """Base custom assertions
    """
    def assertRelativeToleranceAlmostEqual(self, a: float, b: float, tol: float) -> None:
        delta = abs(a * tol)
        msg = f'{a:.8e} and {b:.8e} are not within relative tolerance {tol} (delta = {delta})'
        assert ((a - delta) <= b <= (a + delta)), msg

    def assertListsRelativeToleranceAlmostEqual(self, list_a: list[float], list_b: list[float], tol: float) -> None:
        if len(list_a) != len(list_b):
            msg = f'list_a length not equal to list_b ({len(list_a)}, {len(list_b)})'
            raise Exception(msg)
        
        for a, b in zip(list_a, list_b):
            self.assertRelativeToleranceAlmostEqual(a, b, tol)

    def assertvalueInIterable(self, value: float, iterable: iter):
        msg = f'value not found in look up : {value}\n -> ' 
        msg += '\n -> '.join(iterable)
        assert (value in iterable), msg


class CustomParameterChecks():
    def assertParametersInClassInstance(self, cls, parameters, tol):
        for parameter, value in parameters.items():
            try:
                assert hasattr(cls, parameter)
            except:
                print(f'cls = {cls} - parameter = {parameter}')
            self.assertRelativeToleranceAlmostEqual(
                getattr(cls, parameter),
                value,
                tol=tol
            )


class CustomCheckValue():
    """Base custom value check
    """

    def check_values_within_relative_tol(self, a: float, b: float, tol: float):
        delta = abs(a * tol)
        return ((a - delta) <= b <= (a + delta))        


class TemporaryWorkingDirectory():
    """Change into a temporary working directory, restoring the original on exit.

    Decomposer.add_solid writes a STEP file into ``added_solids/`` in the working
    directory for every solid added, so tests that run the decomposer use this
    (as setUpModule / tearDownModule) to keep those files out of the repository.
    """

    def __init__(self):
        self._temp_dir = None
        self._cwd = None

    def enter(self):
        import os
        import tempfile
        self._cwd = os.getcwd()
        self._temp_dir = tempfile.TemporaryDirectory()
        os.chdir(self._temp_dir.name)

    def exit(self):
        import os
        os.chdir(self._cwd)
        self._temp_dir.cleanup()
