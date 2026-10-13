"""A Python Tool to Compute Local Symmetry Maps"""
import os

__author__ = "MATERIALab"
__contact__ = "olivieroandreuss@boisestate.edu"
__license__ = "MIT"
__version__ = "0.1.0"
__date__ = "2024-05-24"

try:
    from importlib.metadata import version # python >= 3.8
except Exception :
    try:
        from importlib_metadata import version
    except Exception :
        pass

try:
    __version__ = version("mapsy")
except Exception:
    pass



def __getattr__(name):
    """Lazy public exports keep CLI startup independent of heavy dependencies."""
    if name in ("Maps", "MapsFromFile"):
        from .maps import Maps, MapsFromFile
        return {"Maps": Maps, "MapsFromFile": MapsFromFile}[name]
    if name in ("BioMaps", "BioSystem"):
        from .biomapsy import BioMaps, BioSystem
        return {"BioMaps": BioMaps, "BioSystem": BioSystem}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
