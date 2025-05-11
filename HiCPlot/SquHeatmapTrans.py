#!/usr/bin/env python3
import argparse
import os
import sys
import cooler
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

script_dir = os.path.dirname(os.path.abspath(__file__))
version_py = os.path.join(script_dir, "_version.py")
with open(version_py) as _vf:
    exec(_vf.read())

def _parse_csv_list(value, cast=None):
    """Return a list from a comma separated string. Empty is[]"""
    if value in (None, ""):
        return []
    items = [v.strip() for v in str(value).split(',') if v.strip()]
    if cast:
        return [cast(v) if v else None for v in items]
    return items


def _pad(seq, n, fill=None):
    """Pad *seq* to length *n* with *fill*."""
    return list(seq) + [fill] * (n - len(seq))


def _prepare_region(chrom, start, end, chromsizes):
    """Convert CLI coordinates is tuple (chrom, start, end) and sanity is check."""
    if chrom not in chromsizes:
        raise ValueError(f"Chromosome '{chrom}' not found.")
    size = chromsizes[chrom]
    start = 0 if start is None else int(start)
    end = size if end is None else int(end)
    if start < 0 or end > size or start >= end:
        raise ValueError(f"Bad range for {chrom}: {start}-{end} (size {size})")
    return chrom, start, end


def _format_ticks(ax):
    """Format axes in Mb with two decimals."""
    million = 1e6
    fmt = plt.FuncFormatter(lambda x, _: f"{x / million:.2f}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.tick_params(axis="both", labelsize=8)


def _normalise(mat, method):
    if mat is None:
        return None
    mat = mat.astype(float)
    if method == "raw":
        return mat
    if method == "logNorm":
        return np.maximum(mat, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        if method == "log2":
            return np.log2(np.where(mat > 0, mat, np.nan))
        if method == "log2_add1":
            return np.log2(mat + 1)
        if method == "log":
            return np.log(np.where(mat > 0, mat, np.nan))
        if method == "log_add1":
            return np.log(mat + 1)
    raise ValueError(f"Unsupported normalisation: {method}")

def plot_heatmaps(cooler_file1,resolution,chrid1,chrid2,start1,end1,
    start2,end2,output_file,format,cmap_name,vmin,vmax,layout,
    cooler_file2,sampleid1,sampleid2,track_size,track_spacing,
    normalization_method,merge_axes):
    """Render heatmaps as requested by command line."""

    try:
        clr1 = cooler.Cooler(f"{cooler_file1}::resolutions/{resolution}")
    except Exception as e:
        sys.exit(f"Error loading {cooler_file1}: {e}")

    chromsizes1 = dict(zip(clr1.chromnames, clr1.chromsizes))

    clr2 = chromsizes2 = None
    if cooler_file2:
        try:
            clr2 = cooler.Cooler(f"{cooler_file2}::resolutions/{resolution}")
            chromsizes2 = dict(zip(clr2.chromnames, clr2.chromsizes))
        except Exception as e:
            sys.exit(f"Error loading {cooler_file2}: {e}")

    use_balance = format == "balance"

    if merge_axes:
        chrom_order = _parse_csv_list(chrid1)
        if not chrom_order:
            sys.exit("--merge_axes needs at least one chromosome in --chrid1")
        if any([start1, end1, start2, end2]):
            print("[warn] start/end ignored with --merge_axes; using whole chromosomes")

        def build_canvas(clr, chromsizes):
            if clr is None:
                return None, None
            bins = [int(np.ceil(chromsizes[c] / resolution)) for c in chrom_order]
            edges = np.cumsum([0] + bins)
            canvas = np.full((edges[-1], edges[-1]), np.nan)
            fetch = clr.matrix(balance=use_balance, as_pixels=False)
            for yi, cy in enumerate(chrom_order):
                for xi, cx in enumerate(chrom_order):
                    block = fetch.fetch(cy, cx).astype(float)
                    # pad if last bin shorter than resolution
                    pad = (bins[yi] - block.shape[0], bins[xi] - block.shape[1])
                    if pad[0] or pad[1]:
                        block = np.pad(block, ((0, pad[0]), (0, pad[1])), constant_values=np.nan)
                    y0, y1 = edges[yi], edges[yi + 1]
                    x0, x1 = edges[xi], edges[xi + 1]
                    canvas[y0:y1, x0:x1] = block
            return canvas, edges

        mat1, edges = build_canvas(clr1, chromsizes1)
        mat2, _ = build_canvas(clr2, chromsizes2) if clr2 else (None, None)

        mats = [m for m in (mat1, mat2) if m is not None]
        mats_norm = [_normalise(m, normalization_method) for m in mats]

        finite = np.concatenate([m[np.isfinite(m) & (m > 0)] for m in mats_norm]) if mats_norm else np.array([])
        if finite.size:
            vmin = np.nanmin(finite) if vmin is None else vmin
            vmax = np.nanmax(finite) if vmax is None else vmax
        else:
            vmin, vmax = 1e-9, 1.0

        norm_obj = None
        if normalization_method == "logNorm":
            vmin = max(vmin, 1e-9)
            norm_obj = LogNorm(vmin=vmin, vmax=vmax, clip=True)
            vmin = vmax = None

        figs = 1 if mat2 is None else 2
        fig_w = figs * track_size + (figs - 1) * track_spacing
        fig, axes = plt.subplots(1, figs, figsize=(min(fig_w, 40), track_size), squeeze=False)

        def show(ax, mat, title):
            im = ax.imshow(np.ma.masked_invalid(mat), origin="upper", aspect="equal", cmap=cmap_name,
                           norm=norm_obj, vmin=vmin, vmax=vmax)
            mids = [(edges[i] + edges[i + 1]) / 2 for i in range(len(chrom_order))]
            ax.set_xticks(mids)
            ax.set_xticklabels(chrom_order, rotation=90)
            ax.set_yticks(mids)
            ax.set_yticklabels(chrom_order)
            ax.tick_params(axis="both", length=0, labelsize=8)
            ax.set_title(title, fontsize=9, pad=10)
            div = make_axes_locatable(ax)
            cax = div.append_axes("bottom", size="5%", pad=0.3)
            cb = plt.colorbar(im, cax=cax, orientation="horizontal")
            cb.ax.tick_params(labelsize=7)
            cb.set_label(normalization_method, fontsize=8, labelpad=3)

        show(axes[0, 0], mats_norm[0], sampleid1 or os.path.basename(cooler_file1))
        if figs == 2:
            show(axes[0, 1], mats_norm[1], sampleid2 or os.path.basename(cooler_file2))

        plt.tight_layout()
        fig.savefig(output_file, bbox_inches='tight')
        plt.close(fig)
        return

    chrid1 = _parse_csv_list(chrid1)
    chrid2 = _parse_csv_list(chrid2) or chrid1

    start1 = _parse_csv_list(start1, cast=lambda x: int(x) if x else None)
    end1 = _parse_csv_list(end1, cast=lambda x: int(x) if x else None)
    start2 = _parse_csv_list(start2, cast=lambda x: int(x) if x else None)
    end2 = _parse_csv_list(end2, cast=lambda x: int(x) if x else None)

    if len(chrid2) != len(chrid1):
        sys.exit("chrid1 and chrid2 must have the same length.")

    n_pairs = len(chrid1)
    chrid1 = _pad(chrid1, n_pairs)
    chrid2 = _pad(chrid2, n_pairs)
    start1 = _pad(start1, n_pairs)
    end1 = _pad(end1, n_pairs)
    start2 = _pad(start2, n_pairs)
    end2 = _pad(end2, n_pairs)

    single_sample = cooler_file2 is None
    mats1_raw, mats2_raw, regions, titles = [], [], [], []

    for i in range(n_pairs):
        c1, c2 = chrid1[i], chrid2[i]
        try:
            r1_c, r1_s, r1_e = _prepare_region(c1, start1[i], end1[i], chromsizes1)
            r2_c, r2_s, r2_e = _prepare_region(c2, start2[i], end2[i], chromsizes1)
        except ValueError as e:
            print(f"[skip] {c1} vs {c2}: {e}")
            continue

        regions.append((r1_c, r1_s, r1_e, r2_c, r2_s, r2_e))
        title = f"{r1_c}:{r1_s:,}-{r1_e:,}\n{r2_c}:{r2_s:,}-{r2_e:,}"
        if r1_s == 0 and r1_e == chromsizes1[r1_c] and r2_s == 0 and r2_e == chromsizes1[r2_c]:
            title = f"{r1_c} vs {r2_c} (whole)"
        titles.append(title)

        fetch1 = clr1.matrix(balance=use_balance, as_pixels=False)
        mats1_raw.append(fetch1.fetch((r1_c, r1_s, r1_e), (r2_c, r2_s, r2_e)).astype(float))

        if not single_sample and clr2:
            fetch2 = clr2.matrix(balance=use_balance, as_pixels=False)
            mats2_raw.append(fetch2.fetch((r1_c, r1_s, r1_e), (r2_c, r2_s, r2_e)).astype(float))

    if not regions:
        sys.exit("No valid regions to plot.")

    mats1 = [_normalise(m, normalization_method) for m in mats1_raw]
    mats2 = [_normalise(m, normalization_method) for m in mats2_raw] if mats2_raw else []

    finite = np.concatenate([m[np.isfinite(m) & (m > 0)] for m in mats1 + mats2])
    if finite.size:
        vmin = np.nanmin(finite) if vmin is None else vmin
        vmax = np.nanmax(finite) if vmax is None else vmax
    else:
        vmin, vmax = 1e-12, 1.0

    norm_obj = None
    if normalization_method == "logNorm":
        vmin = max(vmin, 1e-12)
        norm_obj = LogNorm(vmin=vmin, vmax=vmax, clip=True)
        vmin = vmax = None

    nrows = len(regions)
    ncols = 1 if single_sample else 2
    fig_w = ncols * track_size + (ncols - 1) * track_spacing
    fig_h = nrows * track_size + (nrows - 1) * track_spacing
    fig = plt.figure(figsize=(min(fig_w, 40), min(fig_h, 80)))
    gs = plt.GridSpec(nrows, ncols, hspace=0.4, wspace=0.35)

    for ridx, title in enumerate(titles):
        for sidx in range(ncols):
            ax = fig.add_subplot(gs[ridx, sidx])
            mat = mats1[ridx] if sidx == 0 else mats2[ridx]
            if mat is None:
                ax.axis('off')
                continue
            r1_c, r1_s, r1_e, r2_c, r2_s, r2_e = regions[ridx]
            extent = (r2_s, r2_e, r1_e, r1_s)
            mat = mat.T  # flip so x = chrid1, y = chrid2
            im = ax.imshow(np.ma.masked_invalid(mat), origin='upper', aspect='equal', cmap=cmap_name,
                           norm=norm_obj, vmin=vmin, vmax=vmax, extent=extent)
            _format_ticks(ax)
            ax.set_xlabel(r1_c, fontsize=8, labelpad=4)
            ax.set_ylabel(r2_c, fontsize=8, labelpad=4)
            label = sampleid1 if sidx == 0 else sampleid2
            #ax.set_title(f"{label} - {title}" if label else title, fontsize=9, pad=10)
            ax.set_title(f"{label}\n{title}" if label else title, fontsize=9, pad=10)
            div = make_axes_locatable(ax)
            cax = div.append_axes("bottom", size="5%", pad=0.3)
            cb = plt.colorbar(im, cax=cax, orientation='horizontal')
            cb.ax.tick_params(labelsize=7)
            cb.set_label(normalization_method, fontsize=8, labelpad=3)

    plt.tight_layout()
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description="Plot Hi-C heat-maps (separate or concatenated)")
    parser.add_argument('--cooler_file1', type=str, required=True, help='Path to the case .mcool file.')
    parser.add_argument('--cooler_file2', type=str, help='Path to secondary .mcool (optional)')
    parser.add_argument('--format', type=str, default='balance', choices=['balance', 'ICE'], help='Format of .mcool file.')
    parser.add_argument('--resolution', type=int, required=True, help='Resolution for the cooler data.')

    parser.add_argument('--chrid1', required=True, help='Chromosome ID(s) for X-axis (comma-sep)')
    parser.add_argument('--start1', type=int, help='Start position for the region of interest 1.')
    parser.add_argument('--end1', type=int, help='End position for the region of interest 1.')
    parser.add_argument('--chrid2', help='Chromosome ID(s) for Y-axis (defaults to chrid2)')
    parser.add_argument('--start2', type=int, help='Start position for the region of interest 2.')
    parser.add_argument('--end2', type=int, help='End position for the region of interest 2.')

    parser.add_argument('--sampleid1', default='', help='Label for first sample (optional)')
    parser.add_argument('--sampleid2', default='', help='Label for second sample (optional)')

    parser.add_argument('--cmap_name', default='YlOrRd', help='Matplotlib colormap')
    parser.add_argument('--vmin', type=float); parser.add_argument('--vmax', type=float)
    parser.add_argument('--normalization_method', default='raw', choices=[
        'raw','logNorm','log2','log2_add1','log','log_add1'])
    parser.add_argument('--layout', choices=['vertical','horizontal'], default='vertical', help='Separate-panel layout')
    parser.add_argument('--track_size', type=float, default=5, help='Height of each track (in inches).')
    parser.add_argument('--track_spacing', type=float, default=0.5, help='Spacing between tracks (in inches).')
    parser.add_argument('--output_file', type=str, default='heatmap.pdf', help='Filename for the saved comparison heatmap.')

    parser.add_argument('--merge_axes', action='store_true',help='Concatenate all chromosomes along both axes into one heat-map')
    parser.add_argument("-V", "--version", action="version",version="SquHeatmap {}".format(__version__)\
                      ,help="Print version and exit")
    args = parser.parse_args(argv)

    plot_heatmaps(
        cooler_file1=args.cooler_file1,
        resolution=args.resolution,
        chrid1=args.chrid1,
        chrid2=args.chrid2,
        start1=args.start1,
        end1=args.end1,
        start2=args.start2,
        end2=args.end2,
        output_file=args.output_file,
        format=args.format,
        cmap_name=args.cmap_name,
        vmin=args.vmin,
        vmax=args.vmax,
        layout=args.layout,
        cooler_file2=args.cooler_file2,
        sampleid1=args.sampleid1,
        sampleid2=args.sampleid2,
        track_size=args.track_size,
        track_spacing=args.track_spacing,
        normalization_method=args.normalization_method,
        merge_axes=args.merge_axes,
    )

if __name__ == '__main__':
    main()
