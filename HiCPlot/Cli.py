#!/usr/bin/env python3
"""
HiCPlot/Cli.py
---------------------------------------------------------------------
Unified command‑line wrapper for the HiCPlot plotting toolkit.

Usage
-----
    $ HiCPlot <tool> [tool‑specific options]

The wrapper exposes each plotting helper (SquHeatmap, TriHeatmap, …) as a
sub‑command and forwards every remaining CLI token untouched.  It also fixes a
long‑standing issue where the chosen sub‑command re‑appeared later in
``sys.argv`` and confused the underlying tool, leading to
``error: unrecognized arguments: <tool>``.

The fix is implemented by temporarily patching ``sys.argv`` so that every
sub‑tool sees *only* its own flags.
"""
from __future__ import annotations

import argparse
import sys
from typing import Callable, Dict, List, Optional

# ----------------------------------------------------------------------
# Import plotting entry points (adjust the import paths to match your
# installation if necessary).
# ----------------------------------------------------------------------
from HiCPlot.SquHeatmap import main as _run_squ
from HiCPlot.SquHeatmapTrans import main as _run_squTrans
from HiCPlot.TriHeatmap import main as _run_tri
from HiCPlot.DiffSquHeatmap import main as _run_diff
from HiCPlot.DiffSquHeatmapTrans import main as _run_diffTrans
from HiCPlot.upper_lower_triangle_heatmap import main as _run_ul
from HiCPlot.NGStrack import main as _run_track

# ----------------------------------------------------------------------
# Sub‑command registry
# ----------------------------------------------------------------------
_SUBCOMMANDS: Dict[str, Callable[[Optional[List[str]] | None], None]] = {
    "SquHeatmap": _run_squ,
    "SquHeatmapTrans": _run_squTrans,
    "TriHeatmap": _run_tri,
    "DiffSquHeatmap": _run_diff,
    "DiffSquHeatmapTrans": _run_diffTrans,
    "upper_lower_triangle_heatmap": _run_ul,
    "NGStrack": _run_track,
}

_SUBCOMMAND_DESCR: Dict[str, str] = {
    "SquHeatmap": "Square intra‑chromosomal heatmap",
    "SquHeatmapTrans": "Square inter‑chromosomal heatmap",
    "TriHeatmap": "Triangular intra‑chromosomal heatmap",
    "DiffSquHeatmap": "Differential square heatmap",
    "DiffSquHeatmapTrans": "Differential square inter‑heatmap",
    "upper_lower_triangle_heatmap": "Split‑triangle heatmap (upper vs lower)",
    "NGStrack": "Plot multiple NGS tracks",
}

# ----------------------------------------------------------------------
# CLI construction helpers
# ----------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Return the top‑level parser with one sub‑parser per plotting tool."""
    parser = argparse.ArgumentParser(
        prog="HiCPlot",
        description="Hi‑C plotting utility (wrapper for individual tools)",
    )

    subparsers = parser.add_subparsers(
        title="Available tools",
        dest="cmd",
        metavar="<tool>",
        required=True,  # Python ≥3.7: a sub‑command is mandatory
    )

    for name in _SUBCOMMANDS:
        help_line = _SUBCOMMAND_DESCR.get(name, "(no description)")
        sp = subparsers.add_parser(
            name,
            help=help_line,
            description=help_line,
            add_help=False,  # let the tool define -h/--help if it wishes
        )
        sp.set_defaults(_entry=_SUBCOMMANDS[name])

    return parser

# ----------------------------------------------------------------------
# Utility context‑manager to swap ``sys.argv`` temporarily
# ----------------------------------------------------------------------

class _patched_sys_argv:
    """Context‑manager that replaces ``sys.argv`` for the lifetime of a block."""

    def __init__(self, replacement: List[str]):
        self._replacement = replacement
        self._original: List[str] | None = None

    def __enter__(self):
        self._original = sys.argv[:]
        sys.argv[:] = self._replacement
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._original is not None:
            sys.argv[:] = self._original

# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------

def main(argv: Optional[List[str]] | None = None) -> None:
    """Parse the wrapper‑level options and dispatch to the chosen sub‑tool."""

    argv = sys.argv[1:] if argv is None else argv
    parser = _build_parser()

    # First‑stage parse: identify the sub‑command and collect the *rest*
    ns, rest = parser.parse_known_args(argv)

    if not hasattr(ns, "_entry"):
        parser.print_help(sys.stderr)
        sys.exit(1)

    entry: Callable[[Optional[List[str]] | None], None] = getattr(ns, "_entry")

    # Special‑case: ``HiCPlot <tool> -h`` should show the tool's own help.
    if rest and rest[0] in ("-h", "--help"):
        entry(["-h"])
        return

    # ------------------------------------------------------------------
    # Forward the remaining CLI tokens to the sub‑tool.  Many HiCPlot
    # helpers read directly from ``sys.argv`` instead of an explicit
    # ``argv`` parameter, so we patch ``sys.argv`` temporarily.
    # ------------------------------------------------------------------
    clean_argv = [sys.argv[0], *rest]  # keep executable name, drop sub‑command

    with _patched_sys_argv(clean_argv):
        # Pass ``None`` so the callee falls back to *its* own ``sys.argv``
        entry(None)


if __name__ == "__main__":
    main()
