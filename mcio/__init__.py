"""mcio: CAD (STEP) to CSG conversion for Monte Carlo radiation transport codes.

Importing the package configures the ``mcio`` logger (writing to
``analysis/logging.log`` in the main process), imports the core submodules and,
where available, the optional OCP (ocp-cadquery), openmc and cadquery backed
modules. Element and isotope data constants are re-exported from
:mod:`mcio.data`.
"""

from __future__ import annotations

import logging
import os
from multiprocessing import parent_process

import warnings


# filter warning of OCP missing
warnings.filterwarnings("ignore", message=r".*No module named ['\"]OCP['\"].*")

# set up logging
logger: logging.Logger = logging.getLogger("mcio")

# Initialise as None
LOGGER_FILE_HANDLER: logging.FileHandler | None = None

# Robust check: If parent_process() is None, this is the Main Process.
# In "spawn" mode (Windows), child processes will have a parent_process object.
if parent_process() is None:
    # set up logging with a top level handler
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # create output filepath, making directory if required
    file_path = os.path.join("analysis", "logging.log")
    directory = os.path.dirname(file_path)

    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    LOGGER_FILE_HANDLER = logging.FileHandler(file_path, mode="w", encoding="utf-8")
    LOGGER_FILE_HANDLER.setFormatter(formatter)

# set logger params
logger.setLevel(logging.INFO)

# add handler if available
if LOGGER_FILE_HANDLER is not None:
    logger.addHandler(LOGGER_FILE_HANDLER)
else:
    logger.addHandler(logging.NullHandler())

# expose logger and handler
__all__: list[str] = ["logger", "LOGGER_FILE_HANDLER"]

# set logger params
logger.setLevel(logging.INFO)


# store logging levels to allow user to set verbosity
class LoggerLevels:
    """Namespace of logging level constants allowing users to set verbosity.

    Attributes
    ----------
    DEBUG : int
        ``logging.DEBUG`` level.
    INFO : int
        ``logging.INFO`` level.
    WARNING : int
        ``logging.WARNING`` level.
    ERROR : int
        ``logging.ERROR`` level.
    CRITIAL : int
        ``logging.CRITICAL`` level (attribute name is misspelt in the code).
    """

    def __init__(self) -> None:
        """Initialise the logging level attributes."""
        self.DEBUG: int = logging.DEBUG
        self.INFO: int = logging.INFO
        self.WARNING: int = logging.WARNING
        self.ERROR: int = logging.ERROR
        self.CRITIAL: int = logging.CRITICAL


# create lookup for logger levels, strings to int
LOGGER_LEVELS: LoggerLevels = LoggerLevels()


# import command line parser, trial run to check for help print only
# as ocp-cadquery slow to load and not required.
# if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
#     from .__main__ import main
#     args = main()


# import modules
from . import advantg
from . import cad
from . import cell
from . import checkvals
from . import data
from . import decomposeparams
from . import formatting
from . import geant4
from . import macrobody
from . import material
from . import mcnp
from . import model
from . import reader
from . import serpent2
from . import source
from . import surface
from . import tally
from . import transform
from . import volume
from . import universe

# check for optional packages - ocp-cadquery
try:
    import casadi
    import OCP
    from . import decompose
    from . import diagnostic
    # from . import inversion
    from . import ocp
    from . import ocpbooleans
    from . import ocpboundingbox
    from . import ocpfilereadwrite
    from . import ocpproperties
    from . import ocprepair
    from . import ocptomcio

except ModuleNotFoundError as e:
    logger.warning(f'[PACKAGE UNAVAILABLE - unable to load OCP]')
    logger.warning(e)

# check for optional packages - openmc
try:
    import openmc
except ModuleNotFoundError:
    logger.warning(f"[PACKAGE UNAVAILABLE] - unable to load openmc")

# check for optional packages - cadquery
try:
    import cadquery as cq
except ModuleNotFoundError:
    logger.warning(f"[PACKAGE UNAVAILABLE] - unable to load cadquery")

# load constants from data package
from mcio.data import (
    ELEMENT,
    ATOMIC_NUMBER,
    RELATIVE_ATOMIC_MASS,
    ISOTOPIC_COMPOSITION_BY_ATOM_FRACTION,
    ISOTOPIC_COMPOSITION_BY_WEIGHT_FRACTION,
    ZAID_FROM_ISOTOPE,
    ELEMENT_FROM_ISOTOPE,
    ISOTOPE_FROM_ZAID,
)
