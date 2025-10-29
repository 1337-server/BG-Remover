"""Entry points for the CLI runtime.

The :mod:`runtimes.cli.bgr_cli` module exposes the real implementation of the
command line interface. Importing it eagerly from the package ``__init__``
caused ``python -m runtimes.cli.bgr_cli`` to emit a double-import warning
because the module was already present in :data:`sys.modules` before the
``runpy`` execution hook ran. To keep the public API stable while avoiding the
warning we defer the import until :func:`main` is actually called.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["main"]


def main(argv: Sequence[str] | None = None, /) -> int:
    """Invoke the CLI ``main`` function lazily.

    Parameters
    ----------
    argv:
        Optional argument vector to pass through to
        :func:`runtimes.cli.bgr_cli.main`. ``None`` uses :data:`sys.argv`.

    Returns
    -------
    int
        The exit status returned by :func:`runtimes.cli.bgr_cli.main`.
    """

    from . import bgr_cli

    return bgr_cli.main(argv)
