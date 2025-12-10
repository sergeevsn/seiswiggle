# Import wiggle_plot from the seiswiggle module
# When using py-modules, seiswiggle.py is installed as a top-level module
try:
    # Try relative import first (if installed as package)
    from .seiswiggle import wiggle_plot
except ImportError:
    # Fallback to absolute import (if installed as module)
    from seiswiggle import wiggle_plot

__version__ = "0.1.0"
__all__ = ["wiggle_plot"]

