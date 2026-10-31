"""GridForge: synthetic transmission-grid generator.

Python conversion of the Snodgrass dissertation MATLAB codebase.
FLO-Grid builds topology; SCALE-Grid builds the electrical model;
GridForge orchestrates. See docs/ARCHITECTURE.md.
"""

from .config import RunConfig
from .model import GridModel

__all__ = ["GridModel", "RunConfig"]
__version__ = "1.0.0"
