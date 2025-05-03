#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import EngFormatter
import cooler
import argparse
import os
import pandas as pd
from matplotlib.colors import LogNorm
from collections import defaultdict

script_dir = os.path.dirname(os.path.abspath(__file__))
version_py = os.path.join(script_dir, "_version.py")
with open(version_py) as _vf:
    exec(_vf.read())

def pcolormesh_rect(ax, matrix, x_start, x_end, y_start, y_end,
                    cmap='bwr', vmin=None, vmax=None, **kwargs):
    """Plot *matrix* as a heat-map with genomic extents on both axes."""
    # Use imshow for potentially better performance and handling of aspect ratio
    return ax.imshow(matrix,
                     origin='upper',
                     aspect='auto', # Ensures squares aren't forced if extents differ
                     extent=[x_start, x_end, y_end, y_start], # Note: y-axis is inverted for imshow's 'upper' origin
                     cmap=cmap, vmin=vmin, vmax=vmax,
                     interpolation='none', # Avoids blurring pixels
                     **kwargs)


def plot_trans_diff(cooler_file1, cooler_file2,
                    resolution,
                    chrid1, start1, end1,
                    chrid2, start2, end2,
                    operation='subtract',
                    division_method='raw',
                    format='balance',
                    diff_cmap='bwr',
                    vmin=None, vmax=None,
                    diff_title=None,
                    track_size=5,track_spacing=0.5,
                    output_file='comparison_heatmap.pdf'):
    """Compute and plot difference/ratio between two trans blocks."""
    if chrid1 == chrid2:
        raise ValueError("chrid1 and chrid2 must differ – trans only.")

    region1 = f"{chrid1}:{start1}-{end1}"
    region2 = f"{chrid2}:{start2}-{end2}"

    # Load cooler data for case
    clr1 = cooler.Cooler(f'{cooler_file1}::resolutions/{resolution}')
    if format == "balance":
        data1 = clr1.matrix(balance=True).fetch(region1,region2).astype(float)
    elif format == "ICE":
        data1 = clr1.matrix(balance=False).fetch(region1,region2).astype(float)
    else:
        print("input format is wrong")
    # Load cooler data for control
    clr2 = cooler.Cooler(f'{cooler_file2}::resolutions/{resolution}')
    if format == "balance":
        data2 = clr2.matrix(balance=True).fetch(region1,region2).astype(float)
    elif format == "ICE":
        data2 = clr2.matrix(balance=False).fetch(region1,region2).astype(float)
    else:
        print("input format is wrong")

    # Compute difference / ratio
    if operation == 'subtract':
        data1 = np.nan_to_num(data1, nan=0.0)
        data2 = np.nan_to_num(data2, nan=0.0)
        data_diff = data1 - data2
    elif operation == 'divide':
        data1 = np.maximum(data1, 0)
        data2 = np.maximum(data2, 0)

        if division_method == 'raw':
            with np.errstate(divide='ignore', invalid='ignore'):
                data_diff = np.divide(data1, data2)
                data_diff[~np.isfinite(data_diff)] = 0
        elif division_method == 'add1':
            data_diff = (data1 + 1) / (data2 + 1)
        elif division_method == 'log2':
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.divide(data1, data2)
                ratio[(ratio <= 0) | (~np.isfinite(ratio))] = np.nan
                data_diff = np.log2(ratio)
        elif division_method == 'log2_add1':
            ratio = (data1 + 1) / (data2 + 1)
            ratio[~np.isfinite(ratio)] = np.nan
            data_diff = np.log2(ratio)
        else:
            raise ValueError("--division_method must be raw / add1 / log2 / log2_add1")
    else:
        raise ValueError("--operation must be subtract or divide")

    # Colour limits
    if vmin is None and vmax is None:
        vmin = np.nanmin(data_diff)
        vmax = np.nanmax(data_diff)
    elif vmin is None:
        vmin = np.nanmin(data_diff)
    elif vmax is None:
        vmax = np.nanmax(data_diff)

    # Plot
    f  = plt.figure(figsize=(track_size, track_size))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05],hspace=track_spacing)
    ax_diff = f.add_subplot(gs[0, 0])

    im_diff = pcolormesh_rect(ax_diff, data_diff, start1, end1, start2, end2, cmap=diff_cmap, vmin=vmin, vmax=vmax)
    ax_diff.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x / 1e6:.2f}'))
    ax_diff.set_title(diff_title if diff_title else "Difference Heatmap", fontsize=8)
    ax_diff.set_ylim(start2, end2)
    ax_diff.set_xlim(start1, end1)

    fmt = EngFormatter(unit='b')
    ax_diff.set_xlabel(f"{chrid1}:{fmt(start1)}-{fmt(end1)}")
    ax_diff.set_ylabel(f"{chrid2}:{fmt(start2)}-{fmt(end2)}")

    cax_diff = f.add_subplot(gs[1, 0])
    cbar_diff = plt.colorbar(im_diff, cax=cax_diff, orientation='horizontal')
    cbar_diff.ax.tick_params(labelsize=8)
    plt.subplots_adjust(hspace=track_spacing)
    f.savefig(output_file, bbox_inches='tight')
    plt.close(f)


def main(argv=None): 
    parser = argparse.ArgumentParser(description='Plot the difference between two Cooler files for a trans region.')
    parser.add_argument('--cooler_file1', type=str, required=True, help='Path to the case .mcool file.')
    parser.add_argument('--cooler_file2', type=str, required=True, help='Path to the control .mcool file.')
    parser.add_argument('--resolution', type=int, required=True, help='Resolution for the cooler data.')
    parser.add_argument('--format', type=str, default='balance', choices=['balance', 'ICE'], help='Format of .mcool file.')

    parser.add_argument('--start1', type=int, default=None, help='Start position for region 1.')
    parser.add_argument('--end1', type=int, default=None, help='End position for region 1.')
    parser.add_argument('--chrid1', type=str, default=None, help='Chromosome ID for region 1.')
    parser.add_argument('--start2', type=int, required=False, default=None, help='Start position for region 2.')
    parser.add_argument('--end2', type=int, required=False, default=None, help='End position for region 2.')
    parser.add_argument('--chrid2', type=str, required=False, default=None, help='Chromosome ID for region 2.')

    # diff options
    parser.add_argument('--operation', type=str, default='subtract', choices=['subtract', 'divide'],help="Operation for the difference matrix.")
    parser.add_argument('--division_method', type=str, default='raw', choices=['raw', 'log2', 'add1', 'log2_add1'],
                        help="Method for division when '--operation divide' is selected: 'raw' (case/control), 'log2' (log2(case/control)), 'add1' ((case+1)/(control+1)), or 'log2_add1' (log2((case+1)/(control+1))).")
    # visual
    parser.add_argument('--diff_cmap', type=str, default='bwr', help='Colormap to be used for the heatmap.')
    parser.add_argument('--diff_title', type=str, default=None, help='Title for difference matrix.')
    parser.add_argument('--track_size', type=float, default=5, help='Height of each track (in inches).')
    parser.add_argument('--track_spacing', type=float, default=0.5, help='Spacing between tracks (in inches).')
    parser.add_argument('--vmin', type=float, default=None, help='Minimum value for normalization of the heatmap.')
    parser.add_argument('--vmax', type=float, default=None, help='Maximum value for normalization of the combined heatmap.')
    parser.add_argument('--output_file', default='comparison_heatmap.pdf',help='Filename for the saved comparison heatmap PDF.')
    parser.add_argument('-V', '--version', action='version',version=f"trans_diff_heatmap {__version__}")
    args = parser.parse_args()

    plot_trans_diff(cooler_file1=args.cooler_file1,
                    cooler_file2=args.cooler_file2,
                    resolution=args.resolution,
                    start1=args.start1,
                    end1=args.end1,
                    chrid1=args.chrid1,
                    start2=args.start2,
                    end2=args.end2,
                    chrid2=args.chrid2,
                    operation=args.operation,
                    division_method=args.division_method,
                    format=args.format,
                    diff_cmap=args.diff_cmap,
                    vmin=args.vmin,
                    vmax=args.vmax,
                    diff_title=args.diff_title,
                    track_size=args.track_size,
                    track_spacing=args.track_spacing,
                    output_file=args.output_file)


if __name__ == '__main__':
    main()
