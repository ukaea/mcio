"""Classes to store colour meta data."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ColourRGB:
    """Store an RGB colour.

    Attributes
    ----------
    red : int or None
        Red channel value between 0 and 255, by default None
    green : int or None
        Green channel value between 0 and 255, by default None
    blue : int or None
        Blue channel value between 0 and 255, by default None
    """

    red: int | None = None
    green: int | None = None
    blue: int | None = None

    @property
    def rgb(self) -> tuple[int, int, int]:
        """Return the r,g,b integer values between 0 and 255.

        Returns
        -------
        tuple[int, int, int]
            The (red, green, blue) channel values.
        """
        return (self.red, self.green, self.blue)

    def __str__(self) -> str:
        """Return the colour as ``[R,G,B]`` with ints between 0-255.

        Returns
        -------
        str
            String of r,g,b, colour in the format rrr_ggg_bbb.
        """
        return f"[{self.red},{self.green},{self.blue}]"

    def as_string(self) -> str:
        """Return a string of r,g,b in the format rrr_ggg_bbb.

        Returns
        -------
        str
            String of r,g,b, value in the format rrr_ggg_bbb.
        """
        return f"{self.red}_{self.green}_{self.blue}"
