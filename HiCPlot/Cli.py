# HiCPlot/cli.py
import sys
import argparse

from HiCPlot.SquHeatmap import main as _run_squ
from HiCPlot.SquHeatmapTrans import main as _run_squTrans
from HiCPlot.TriHeatmap import main as _run_tri
from HiCPlot.DiffSquHeatmap import main as _run_diff
from HiCPlot.DiffSquHeatmapTrans import main as _run_diffTrans
from HiCPlot.upper_lower_triangle_heatmap import main as _run_ul
from HiCPlot.NGStrack import main as _run_track

_SUBCOMMANDS = {
    "SquHeatmap": _run_squ,
    "SquHeatmapTrans": _run_squTrans,
    "TriHeatmap": _run_tri,
    "DiffSquHeatmap": _run_diff,
    "DiffSquHeatmapTrans": _run_diffTrans,
    "upper_lower_triangle_heatmap": _run_ul,
    "NGStrack": _run_track,
}

def _wrapper_parser() -> argparse.ArgumentParser:
    """Parser that knows ONLY the top-level sub-command list.
    It has *no* -h/--help so those flags fall through to sub-tools.
    """
    return argparse.ArgumentParser(
        prog="HiCPlot",
        description="Hi-C plotting utility (wrapper for individual tools)",
        add_help=False,                # <-- key: wrapper owns no -h flag
    )

def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv

    if not argv or argv[0] in ("-h", "--help"):
        p = _wrapper_parser()
        p.add_argument(
            "cmd",
            choices=_SUBCOMMANDS,
            help="Which plotting tool to run",
        )
        p.print_help(sys.stderr)
        sys.exit(0)

    cmd, *rest = argv
    func = _SUBCOMMANDS.get(cmd)
    if func is None:
        sys.stderr.write(f"HiCPlot: unknown sub-command '{cmd}'\n")
        sys.stderr.write("Run 'HiCPlot -h' to see available tools.\n")
        sys.exit(1)

    if rest and rest[0] in ("-h", "--help"):
        func(["-h"])
        return

    func(rest or None)

if __name__ == "__main__":
    main()
