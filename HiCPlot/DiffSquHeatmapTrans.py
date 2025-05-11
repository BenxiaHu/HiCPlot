#!/usr/bin/env python3
import argparse
import os
import sys
import cooler
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
version_py = os.path.join(script_dir, "_version.py")
with open(version_py) as _vf:
    exec(_vf.read())

def _parse_csv_list(val, cast=None):
    if val in (None, ""):
        return []
    parts = [p.strip() for p in str(val).split(',') if p.strip()]
    return [cast(p) if cast else p for p in parts]

def _pad(seq, n, fill=None):
    return list(seq) + [fill] * (n - len(seq))

def _prepare_region(chrom, start, end, chromsizes):
    if chrom not in chromsizes:
        raise ValueError(f"Chromosome '{chrom}' not present")
    size = chromsizes[chrom]
    s = 0 if start is None else int(start)
    e = size if end   is None else int(end)
    if s < 0 or e > size or s >= e:
        raise ValueError(f"Invalid range for {chrom}: {s}-{e} (size {size})")
    return chrom, s, e

def _format_ticks_mb(ax):
    fmt = plt.FuncFormatter(lambda v, _p: f"{v/1e6:.2f}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.tick_params(axis="both", labelsize=8)

def _get_balance_val(clr, spec):
    if spec == "raw":
        return False
    if spec.lower() in ("balance", "ice"):
        return True
    if spec in clr.bins().columns:
        return spec
    sys.stderr.write(f"[warn] balance column '{spec}' missing – using raw\n")
    return False

###############################################################################
# Diff maths (identical to original code 1 – log2 unchanged)                  #
###############################################################################

def _compute_diff(mat1, mat2, operation, division_method):
    if operation == "subtract":
        m1 = np.nan_to_num(mat1, nan=0.0)
        m2 = np.nan_to_num(mat2, nan=0.0)
        return m1 - m2

    # divide family
    m1 = np.nan_to_num(np.maximum(mat1, 0.0), nan=0.0)
    m2 = np.nan_to_num(np.maximum(mat2, 0.0), nan=0.0)

    if division_method == "raw":
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.divide(m1, m2)
        out[~np.isfinite(out)] = np.nan
        return out

    if division_method == "add1":
        out = (m1 + 1) / (m2 + 1)
        out[~np.isfinite(out)] = np.nan
        return out

    if division_method == "log2":
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.divide(m1, m2)
        ratio[(ratio <= 0) | (~np.isfinite(ratio))] = np.nan
        return np.log2(ratio)

    if division_method == "log2_add1":
        ratio = (m1 + 1) / (m2 + 1)
        ratio[~np.isfinite(ratio)] = np.nan
        return np.log2(ratio)

    raise ValueError(f"Unsupported division_method: {division_method}")

###############################################################################
# Plot helpers                                                                #
###############################################################################

def _build_canvas(clr, row_order, col_order, resolution, balance):
    sizes = dict(zip(clr.chromnames, clr.chromsizes))
    row_bins = [int(np.ceil(sizes[c] / resolution)) for c in row_order]
    col_bins = [int(np.ceil(sizes[c] / resolution)) for c in col_order]
    row_edges = np.cumsum([0] + row_bins)
    col_edges = np.cumsum([0] + col_bins)

    canvas = np.full((row_edges[-1], col_edges[-1]), np.nan)
    fetch = clr.matrix(balance=balance, as_pixels=False)

    for iy, ry in enumerate(row_order):
        for ix, cx in enumerate(col_order):
            block = fetch.fetch(ry, cx).astype(float)
            pad_r = row_bins[iy] - block.shape[0]
            pad_c = col_bins[ix] - block.shape[1]
            if pad_r or pad_c:
                block = np.pad(block, ((0, pad_r), (0, pad_c)), constant_values=np.nan)
            r0, r1 = row_edges[iy], row_edges[iy+1]
            c0, c1 = col_edges[ix], col_edges[ix+1]
            canvas[r0:r1, c0:c1] = block
    return canvas, row_edges, col_edges

def _plot_merge(diff, row_edges, col_edges, rows, cols,
                cmap, norm_obj, vmin, vmax, op, title, out_file, size):
    fig, ax = plt.subplots(figsize=(min(size, 40), size))
    im = ax.imshow(np.ma.masked_invalid(diff), origin="upper", aspect="equal",
                   cmap=cmap, norm=norm_obj, vmin=vmin, vmax=vmax)
    # tick labels = chromosome names centred in each block
    ax.set_xticks([(col_edges[i] + col_edges[i+1]) / 2 for i in range(len(cols))])
    ax.set_xticklabels(cols, rotation=90)
    ax.set_yticks([(row_edges[i] + row_edges[i+1]) / 2 for i in range(len(rows))])
    ax.set_yticklabels(rows)
    ax.tick_params(axis="both", length=0, labelsize=8)
    ax.set_xlabel(" / ".join(cols))
    ax.set_ylabel(" / ".join(rows))
    ax.set_title(title, fontsize=10, pad=12)

    div = make_axes_locatable(ax)
    cax = div.append_axes("bottom", size="5%", pad=0.3)
    cb = plt.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7)
    cb.set_label(f"{op} value", fontsize=8, labelpad=3)

    plt.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

def _plot_panel(fig, gs, idx, diff, meta, cmap, norm_obj, vmin, vmax, op):
    ax = fig.add_subplot(gs[idx, 0])
    extent = (meta["x_start"], meta["x_end"], meta["y_end"], meta["y_start"])
    im = ax.imshow(np.ma.masked_invalid(diff), origin="upper", aspect="equal",
                   cmap=cmap, norm=norm_obj, vmin=vmin, vmax=vmax, extent=extent)
    _format_ticks_mb(ax)
    ax.set_xlabel(meta["x_chrom"])
    ax.set_ylabel(meta["y_chrom"])
    ax.set_title(meta["title"], fontsize=9, pad=10)

    div = make_axes_locatable(ax)
    cax = div.append_axes("bottom", size="5%", pad=0.3)
    cb = plt.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7)
    cb.set_label(f"{op} value", fontsize=8, labelpad=3)


def plot_heatmaps(cooler_file1, cooler_file2, resolution,
                  chrid1, chrid2, start1, end1, start2, end2,
                  operation, division_method, fmt,
                  cmap_name, vmin, vmax, diff_title,
                  track_size, track_spacing, output_file,
                  merge_axes):

    # coordinate parsing ------------------------------------------------------
    x_chrs = _parse_csv_list(chrid1)
    y_chrs = _parse_csv_list(chrid2) or x_chrs
    if merge_axes == "panel" and len(x_chrs) != len(y_chrs):
        sys.exit("chrid1 and chrid2 must have the same length in panel layout")

    n_pairs = len(x_chrs)
    xs = _pad(_parse_csv_list(start1, int), n_pairs)
    xe = _pad(_parse_csv_list(end1,   int), n_pairs)
    ys = _pad(_parse_csv_list(start2, int), n_pairs)
    ye = _pad(_parse_csv_list(end2,   int), n_pairs)

    # load coolers ------------------------------------------------------------
    try:
        clr1 = cooler.Cooler(f"{cooler_file1}::resolutions/{resolution}")
        clr2 = cooler.Cooler(f"{cooler_file2}::resolutions/{resolution}")
    except Exception as exc:
        sys.exit(f"Error loading coolers: {exc}")

    bal1 = _get_balance_val(clr1, fmt)
    bal2 = _get_balance_val(clr2, fmt)

    # ------------------------------------------------------------------ MERGE
    if merge_axes == "merge":
        mat1, row_edges, col_edges = _build_canvas(clr1, y_chrs, x_chrs, resolution, bal1)
        mat2, _, _ = _build_canvas(clr2, y_chrs, x_chrs, resolution, bal2)

        # auto-pad smaller matrix so shapes match
        tgt_shape = (max(mat1.shape[0], mat2.shape[0]),
                     max(mat1.shape[1], mat2.shape[1]))

        def _pad_to(arr, tgt):
            if arr.shape == tgt:
                return arr
            pad_r = tgt[0] - arr.shape[0]
            pad_c = tgt[1] - arr.shape[1]
            return np.pad(arr, ((0, pad_r), (0, pad_c)), constant_values=np.nan)

        mat1 = _pad_to(mat1, tgt_shape)
        mat2 = _pad_to(mat2, tgt_shape)

        diff = _compute_diff(mat1, mat2, operation, division_method)

        finite = diff[np.isfinite(diff)]
        auto_min = np.nanmin(finite) if finite.size else -1
        auto_max = np.nanmax(finite) if finite.size else 1

        if operation == "subtract" and (vmin is None or vmax is None):
            abs_max = max(abs(auto_min), abs(auto_max))
            vmin = -abs_max if vmin is None else vmin
            vmax =  abs_max if vmax is None else vmax
        else:
            vmin = auto_min if vmin is None else vmin
            vmax = auto_max if vmax is None else vmax

        norm_obj = None
        if operation == "divide" and division_method in ("raw", "add1"):
            vmin = max(vmin, 1e-9)
            norm_obj = LogNorm(vmin=vmin, vmax=vmax, clip=True)
            vmin = vmax = None

        title = diff_title or f"{os.path.basename(cooler_file1)} vs {os.path.basename(cooler_file2)} ({operation})"
        _plot_merge(diff, row_edges, col_edges, y_chrs, x_chrs,
                    cmap_name, norm_obj, vmin, vmax,
                    operation, title, output_file, track_size)
        return

    # ------------------------------------------------------------------ PANEL
    diff_mats, metas = [], []
    chromsizes = dict(zip(clr1.chromnames, clr1.chromsizes))

    for i in range(n_pairs):
        try:
            x_chr, sx, ex = _prepare_region(x_chrs[i], xs[i], xe[i], chromsizes)
            y_chr, sy, ey = _prepare_region(y_chrs[i], ys[i], ye[i], chromsizes)
        except ValueError as e:
            sys.stderr.write(f"[skip] {e}\n")
            continue

        m1 = clr1.matrix(balance=bal1).fetch((y_chr, sy, ey), (x_chr, sx, ex)).astype(float)
        m2 = clr2.matrix(balance=bal2).fetch((y_chr, sy, ey), (x_chr, sx, ex)).astype(float)
        if m1.shape != m2.shape:
            sys.stderr.write("[skip] shape mismatch\n")
            continue

        diff_mats.append(_compute_diff(m1, m2, operation, division_method))
        metas.append({
            "title": f"{x_chr}:{sx:,}-{ex:,} vs {y_chr}:{sy:,}-{ey:,}",
            "x_chrom": x_chr, "x_start": sx, "x_end": ex,
            "y_chrom": y_chr, "y_start": sy, "y_end": ey
        })

    if not diff_mats:
        sys.exit("No valid matrices – nothing to plot")

    vals = np.concatenate([m[np.isfinite(m)] for m in diff_mats])
    auto_min, auto_max = np.nanmin(vals), np.nanmax(vals)

    if operation == "subtract" and (vmin is None or vmax is None):
        abs_max = max(abs(auto_min), abs(auto_max))
        vmin = -abs_max if vmin is None else vmin
        vmax =  abs_max if vmax is None else vmax
    else:
        vmin = auto_min if vmin is None else vmin
        vmax = auto_max if vmax is None else vmax

    norm_obj = None
    if operation == "divide" and division_method in ("raw", "add1"):
        vmin = max(vmin, 1e-9)
        norm_obj = LogNorm(vmin=vmin, vmax=vmax, clip=True)
        vmin = vmax = None

    n_rows = len(diff_mats)
    fig_h = n_rows * (track_size * 1.12 + track_spacing) - track_spacing
    fig_w = track_size * 1.3
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = plt.GridSpec(n_rows, 1, hspace=track_spacing / track_size)

    if diff_title:
        fig.suptitle(diff_title, fontsize=12, y=0.99)

    for i, diff in enumerate(diff_mats):
        _plot_panel(fig, gs, i, diff, metas[i],
                    cmap_name, norm_obj, vmin, vmax, operation)

    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description="Diff heat-maps between two .mcool files (cis & trans)")

    parser.add_argument('--cooler_file1', type=str, required=True, help='Path to the case .mcool file.')
    parser.add_argument('--cooler_file2', type=str, required=True, help='Path to the control .mcool file.')
    parser.add_argument('--format', type=str, default='balance', choices=['balance', 'ICE'], help='Format of .mcool file.')
    parser.add_argument('--resolution', type=int, required=True, help='Resolution for the cooler data.')

    # coordinates
    parser.add_argument('--chrid1', required=True, help='Chromosome ID(s) for X-axis (comma-sep)')
    parser.add_argument('--start1', type=int, help='Start position for the region of interest 1.')
    parser.add_argument('--end1', type=int, help='End position for the region of interest 1.')
    parser.add_argument('--chrid2', help='Chromosome ID(s) for Y-axis (defaults to chrid2)')
    parser.add_argument('--start2', type=int, help='Start position for the region of interest 2.')
    parser.add_argument('--end2', type=int, help='End position for the region of interest 2.')

    # diff parameters
    parser.add_argument("--operation", choices=["subtract", "divide"], default="subtract",help="Operation to compute the difference matrix: 'subtract' (case - control) or 'divide' (case / control).")
    parser.add_argument("--division_method", choices=["raw", "add1", "log2", "log2_add1"], default="raw",help="Method for division when '--operation divide' is selected: 'raw' (case/control), 'log2' (log2(case/control)), 'add1' ((case+1)/(control+1)), or 'log2_add1' (log2((case+1)/(control+1))).")


    # layout / plotting
    parser.add_argument("--merge_axes", choices=["panel", "merge"], default="merge",help='Concatenate all chromosomes along both axes into one heat-map')
    parser.add_argument('--cmap_name', type=str, default='bwr', help="Colormap for difference matrix. Default is 'bwr' (Blue-White-Red).")
    parser.add_argument('--diff_title', type=str, default=None, help="Title for difference matrix.")
    parser.add_argument('--vmin', type=float, default=None, help='Minimum value for normalization of the combined heatmap.')
    parser.add_argument('--vmax', type=float, default=None, help='Maximum value for normalization of the combined heatmap.')
    parser.add_argument('--track_size', type=float, default=5, help='Height of each track (in inches).')
    parser.add_argument('--track_spacing', type=float, default=0.5, help='Spacing between tracks (in inches).')
    parser.add_argument('--output_file', type=str, default='comparison_heatmap.pdf', help='Filename for the saved comparison heatmap PDF.')
    parser.add_argument("-V", "--version", action="version",version="SquHeatmap {}".format(__version__)\
                      ,help="Print version and exit")
    args = parser.parse_args(argv)

    plot_heatmaps(
        cooler_file1=args.cooler_file1,
        cooler_file2=args.cooler_file2,
        resolution=args.resolution,
        chrid1=args.chrid1,
        chrid2=args.chrid2,
        start1=args.start1,
        end1=args.end1, 
        start2=args.start2,
        end2=args.end2,
        operation=args.operation, 
        division_method=args.division_method, 
        format=args.format,
        cmap_name=args.cmap_name, 
        vmin=args.vmin,
        vmax=args.vmax, 
        diff_title=args.diff_title,
        track_size=args.track_size,
        track_spacing=args.track_spacing,
        output_file=args.output_file,
        merge_axes=args.merge_axes)

if __name__ == "__main__":
    main()
