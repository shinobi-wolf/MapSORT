from importlib import import_module

__all__ = ["Detector", "Mapping", "MapSORT", "Tracker"]

# Lazy, so mapsort.data and mapsort.eval load without the vendored boxmot the pipeline needs.
_EXPORTS = {"Detector": ".detector", "Mapping": ".mapping", "MapSORT": ".main", "Tracker": ".tracker"}


def __getattr__(name: str):
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
