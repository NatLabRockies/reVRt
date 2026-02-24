"""Routing analysis library"""

import importlib.metadata

from ._rust import RouteFinder, find_paths, simplify_using_slopes

# Needed for inclusion in Sphinx autodoc documentation generation
RouteFinder.__module__ = "revrt"
find_paths.__module__ = "revrt"
simplify_using_slopes.__module__ = "revrt"

__version__ = version = importlib.metadata.version("NLR-reVRt")
