"""OpenMC specific tally parameters."""

from __future__ import annotations


class OpenMCTallyParameters:
    """Container for OpenMC specific tally settings.

    Parameters
    ----------
    dose_interpolation : str, optional
        Interpolation scheme for dose function data, by default 'linear-linear'
    """

    def __init__(self, dose_interpolation: str = 'linear-linear') -> None:
        """Initialise the OpenMC tally settings."""
        self.dose_interpolation: str = dose_interpolation

    @property
    def dose_interpolation(self) -> str:
        """str: Interpolation scheme for dose function data."""
        return self._dose_interpolation
    
    @dose_interpolation.setter
    def dose_interpolation(self, dose_interpolation: str) -> None:
        """Set the dose function interpolation scheme.

        Parameters
        ----------
        dose_interpolation : str
            Interpolation scheme, e.g. 'linear-linear' or 'log-log'.
        """
        self._dose_interpolation = dose_interpolation
