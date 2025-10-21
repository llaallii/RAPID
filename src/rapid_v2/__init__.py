"""Bridge package for the RAPID layout migration.

Phase 0 keeps the legacy `rapid_irl_nav` CLI running while new builders
and data pipes come online under this namespace.
"""

from importlib import metadata


def __getattr__(name: str):
    """Expose package metadata without importing heavy submodules."""

    if name == "__version__":
        try:
            return metadata.version("rapid-rl")
        except metadata.PackageNotFoundError:
            return "0.0.0-migration"
    raise AttributeError(name)


__all__ = ["__getattr__"]
