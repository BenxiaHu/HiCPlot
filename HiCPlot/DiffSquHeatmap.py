#!/usr/bin/env python
import argparse
import os
import pandas as pd
from matplotlib.colors import Normalize
import pyBigWig
import pyranges as pr
import numpy as np
import matplotlib.pyplot as plt
import cooler
from matplotlib.ticker import EngFormatter
import matplotlib.gridspec as gridspec
from matplotlib.patches import Arc
from collections import defaultdict

dir = os.path.dirname(os.path.abspath(__file__))
version_py = os.path.join(dir, "_version.py")
exec(open(version_py).read())

def plot_genes(ax, gtf_file, region, genes_to_annotate=None, color='blue', track_height=1):
    """
    Plot gene annotations on the given axis.
    Annotate only the specified genes with their names.

    Parameters:
    - ax: Matplotlib axis to plot on.
    - gtf_file: Path to the GTF file.
    - region: Tuple (chromosome, start, end).
    - genes_to_annotate: List of gene names to annotate. If None, no annotations.
    - color: Color for gene lines and exons.
    - track_height: Height of each gene track.
    """
    spacing_factor = 1.5
    chrom, start, end = region
    # Load the GTF file using pyranges
    gtf = pr.read_gtf(gtf_file)
    # Filter relevant region
    region_genes = gtf[(gtf.Chromosome == chrom) & (gtf.Start < end) & (gtf.End > start)]

    if region_genes.empty:
        print("No genes found in the specified region.")
        ax.axis('off')  # Hide the axis if no genes are present
        return

    # Select the longest isoform for each gene
    longest_isoforms = region_genes.df.loc[region_genes.df.groupby('gene_id')['End'].idxmax()]

    y_offset = 0
    y_step = track_height * spacing_factor  # Adjusted vertical step for tighter spacing
    plotted_genes = []

    # Iterate over each gene and plot
    for _, gene in longest_isoforms.iterrows():
        # Determine y_offset to avoid overlap with previously plotted genes
        for plotted_gene in plotted_genes:
            if not (gene['End'] < plotted_gene['Start'] or gene['Start'] > plotted_gene['End']):
                y_offset = max(y_offset, plotted_gene['y_offset'] + y_step)

        # Plot gene line with increased linewidth for better visibility
        ax.plot([gene['Start'], gene['End']], [y_offset, y_offset], color=color, lw=1)

        # Plot exons as larger rectangles for increased height
        exons = region_genes.df[
            (region_genes.df['gene_id'] == gene['gene_id']) & (region_genes.df['Feature'] == 'exon')
        ]
        for _, exon in exons.iterrows():
            ax.add_patch(
                plt.Rectangle(
                    (exon['Start'], y_offset - 0.3 * track_height),  # Lowered to center the exon vertically
                    exon['End'] - exon['Start'],
                    0.6 * track_height,  # Increased height of exon rectangles
                    color=color
                )
            )

        # Conditionally add gene name if it's in the specified list
        if genes_to_annotate and gene['gene_name'] in genes_to_annotate:
            ax.text(
                (gene['Start'] + gene['End']) / 2,
                y_offset - 0.4 * track_height,  # Positioned below the gene line
                gene['gene_name'],
                fontsize=8,  # Increased font size for readability
                ha='center',
                va='top'  # Align text above the specified y position
            )

        # Track the plotted gene's range and offset
        plotted_genes.append({'Start': gene['Start'], 'End': gene['End'], 'y_offset': y_offset})

    # Set y-axis limits based on the final y_offset
    ax.set_ylim(-track_height * 2, y_offset + track_height * 2)  # Expanded lower limit
    ax.set_ylabel('Genes')
    ax.set_yticks([])  # Hide y-ticks for a cleaner look
    ax.set_xlim(start, end)
    ax.set_xlabel("Position (Mb)")

    # Format x-axis to display positions in megabases (Mb)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x / 1e6:.2f}'))

def read_bigwig(file_path, region):
    """
    Read BigWig or bedGraph file and return positions and values.
    """
    chrom, start, end = region
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension in ['.bw', '.bigwig']:
        # Open the BigWig file
        bw = pyBigWig.open(file_path)
        # Fetch values from the region
        values = bw.values(chrom, start, end, numpy=True)
        bw.close()  # Close the BigWig file
        positions = np.linspace(start, end, len(values))
    elif file_extension in ['.bedgraph', '.bg']:
        # Read the bedGraph file using pandas
        # Assuming bedGraph files have columns: chrom, start, end, value
        bedgraph_df = pd.read_csv(file_path, sep='\t', header=None, comment='#',
                                  names=['chrom', 'start', 'end', 'value'])
        # Filter the data for the specified region
        region_data = bedgraph_df[
            (bedgraph_df['chrom'] == chrom) &
            (bedgraph_df['end'] > start) &
            (bedgraph_df['start'] < end)
        ]
        if region_data.empty:
            return None, None
        # Prepare the positions and values
        positions = np.sort(np.unique(np.concatenate([region_data['start'].values,
                                                      region_data['end'].values])))
        values = np.zeros_like(positions, dtype=float)
        for idx in range(len(region_data)):
            s = region_data.iloc[idx]['start']
            e = region_data.iloc[idx]['end']
            v = region_data.iloc[idx]['value']
            mask = (positions >= s) & (positions <= e)
            values[mask] = v
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Supported formats are BigWig (.bw) and bedGraph (.bedgraph, .bg).")
    return positions, values

def get_track_min_max(bigwig_files_sample1, bigwig_labels_sample1,
                      bigwig_files_sample2, bigwig_labels_sample2,
                      region):
    """
    Compute the minimum and maximum values for BigWig tracks per type to ensure consistent y-axis scaling.

    Parameters:
    - bigwig_files_sample1: List of BigWig files for sample 1.
    - bigwig_labels_sample1: List of labels corresponding to BigWig files for sample 1.
    - bigwig_files_sample2: List of BigWig files for sample 2.
    - bigwig_labels_sample2: List of labels corresponding to BigWig files for sample 2.
    - region: Tuple containing (chromosome, start, end).

    Returns:
    - type_min_max: Dictionary with BigWig types as keys and (min, max) tuples as values.
    """
    type_min_max = defaultdict(lambda: {'min': np.inf, 'max': -np.inf})

    # Function to extract type from label (assumes type is the first part before a space)
    def extract_type(label):
        return label.split("_")[1] if label and "_" in label else 'Unknown'


    # Combine sample1 and sample2 BigWig files and labels
    combined_files = bigwig_files_sample1 + bigwig_files_sample2
    combined_labels = bigwig_labels_sample1 + bigwig_labels_sample2

    for file, label in zip(combined_files, combined_labels):
        bw_type = extract_type(label)
        positions, values = read_bigwig(file, region)
        if values is not None and len(values) > 0:
            current_min = np.nanmin(values)
            current_max = np.nanmax(values)
            type_min_max[bw_type]['min'] = min(type_min_max[bw_type]['min'], current_min)
            type_min_max[bw_type]['max'] = max(type_min_max[bw_type]['max'], current_max)

    # Replace infinities with None if no data was found for a type
    for bw_type in type_min_max:
        if type_min_max[bw_type]['min'] == np.inf and type_min_max[bw_type]['max'] == -np.inf:
            type_min_max[bw_type] = (None, None)
        else:
            type_min_max[bw_type] = (type_min_max[bw_type]['min'], type_min_max[bw_type]['max'])

    return type_min_max


def plot_seq(ax, file_path, region, color='blue', y_min=None, y_max=None):
    """
    Plot RNA-seq/ChIP-seq expression from BigWig or bedGraph file on given axis.
    """
    positions, values = read_bigwig(file_path, region)
    if positions is None or values is None:
        print(f"No data found in the specified region ({region[0]}:{region[1]}-{region[2]}) in {file_path}")
        ax.axis('off')
        return

    # Plot the RNA-seq/ChIP-seq expression as a filled line plot
    ax.plot(positions, values, color=color, alpha=0.7)
    ax.set_xlim(region[1], region[2])
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    elif y_max is not None:
        ax.set_ylim(0, y_max)
    elif y_min is not None:
        ax.set_ylim(y_min, 1)  # Default upper limit if only y_min is provided
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x / 1e6:.2f}'))

def plot_bed(ax, bed_file, region, color='green', linewidth=1, label=None):
    """
    Plot BED file annotations on the given axis.
    """
    chrom, start, end = region
    # Read the BED file
    bed_df = pd.read_csv(bed_file, sep='\t', header=None, comment='#',
                         names=['chrom', 'start', 'end'] + [f'col{i}' for i in range(4, 10)])
    # Filter for the region and chromosome
    region_bed = bed_df[
        (bed_df['chrom'] == chrom) &
        (bed_df['end'] > start) &
        (bed_df['start'] < end)
    ]
    if region_bed.empty:
        print(f"No BED entries found in the specified region ({chrom}:{start}-{end}) in {bed_file}.")
        ax.axis('off')
        return

    for _, entry in region_bed.iterrows():
        bed_start = max(entry['start'], start)
        bed_end = min(entry['end'], end)
        ax.add_patch(
            plt.Rectangle(
                (bed_start, 0.1),  # y-coordinate fixed
                bed_end - bed_start,
                0.8,  # Height of the BED feature
                color=color,
                linewidth=linewidth
            )
        )

    ax.set_xlim(start, end)
    ax.set_ylim(0, 1)
    ax.axis('off')  # Hide axis for BED tracks
    if label:
        ax.set_title(label, fontsize=8)

def plot_loops(ax, loop_file, region, color='purple', alpha=0.5, linewidth=1, label=None):
    """
    Plot chromatin loops as arcs on the given axis.

    Parameters:
    - ax: Matplotlib axis to plot on.
    - loop_file: Path to the loop file.
    - region: Tuple (chromosome, start, end).
    - color: Color for the loop arcs.
    - alpha: Transparency level for the arcs.
    - linewidth: Width of the arc lines.
    - label: Label for the loop track (sample name).
    """
    chrom, start, end = region
    # Read the loop file
    loop_df = pd.read_csv(loop_file, sep='\t', header=0, usecols=[0,1,2,3,4,5],
                          names=['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2'])

    # Filter loops where both anchors are within the region and on the same chromosome
    loop_df = loop_df[
        (loop_df['chrom1'] == chrom) &
        (loop_df['chrom2'] == chrom) &
        (loop_df['start1'] >= start) & (loop_df['end1'] <= end) &
        (loop_df['start2'] >= start) & (loop_df['end2'] <= end)
    ]

    if loop_df.empty:
        print(f"No loops detected in the specified region ({chrom}:{start}-{end}) in {loop_file}.")
        return
    else:
        print(f"Loops detected in the specified region ({chrom}:{start}-{end}) in {loop_file}.")

    max_height = 0  # Keep track of the maximum arc height for y-axis scaling

    # Add rectangle background to make arcs stand out
    ax.add_patch(
        plt.Rectangle(
            (start, 0),  # Position of the rectangle
            end - start,
            1.0,  # Height of the rectangle
            #color='black',
            alpha=1,  # Adjusted alpha for visibility
            zorder=3,  # Draw behind other elements
            edgecolor='black',  # Border color
            linewidth=1.0,  # Width of the border line
            facecolor="none",
        )
    )

    # Plot each loop as an arc
    for _, loop in loop_df.iterrows():
        a1 = (loop['start1'] + loop['end1']) / 2  # Midpoint of anchor1
        a2 = (loop['start2'] + loop['end2']) / 2  # Midpoint of anchor2
        if a1 == a2:
            continue  # Skip loops where both anchors are the same

        # Calculate the width and height of the arc
        width = abs(a2 - a1)
        height = width / 2  # Adjust this factor to change the curvature

        # Update max_height
        if height > max_height:
            max_height = height

        # Determine the center of the arc
        mid = (a1 + a2) / 2

        # Create the Arc
        arc = Arc((mid, 0), width=width, height=height*2, angle=0, theta1=0, theta2=180,
                  edgecolor=color, facecolor='none', alpha=alpha, linewidth=linewidth)
        ax.add_patch(arc)

    # Adjust x-limits and y-limits
    ax.set_xlim(start, end)
    ax.set_ylim(0, max_height * 1.1)  # Adjust y-limits based on maximum arc height
    ax.axis('off')  # Hide axis for loop tracks
    if label:
        ax.set_title(label, fontsize=8)  # Add sample name above the loop track

def pcolormesh_square(ax, matrix, start, end, cmap='bwr', vmin=None, vmax=None, *args, **kwargs):
    """
    Plot the difference matrix as a heatmap on the given axis.
    """
    if matrix is None:
        return None
    im = ax.imshow(matrix, aspect='auto', origin='upper',
                   extent=[start, end, end, start], cmap=cmap, vmin=vmin, vmax=vmax, *args, **kwargs)
    return im

def plot_heatmaps(
    cooler_file1,cooler_file2,format="balance",
    bigwig_files_sample1=[], bigwig_labels_sample1=[],colors_sample1="red",
    bed_files_sample1=[], bed_labels_sample1=[],
    loop_file_sample1=None, loop_file_sample2=None,
    gtf_file=None, resolution=None,
    start=None, end=None, chrid=None,
    cmap='autumn_r', vmin=None, vmax=None,
    track_min=None,track_max=None,
    output_file='comparison_heatmap.pdf',
    bigwig_files_sample2=[], bigwig_labels_sample2=[], colors_sample2="blue",
    bed_files_sample2=[], bed_labels_sample2=[],
    track_size=5, track_spacing=0.5,
    operation='subtract', division_method='raw',
    diff_cmap='bwr', diff_title=None,
    genes_to_annotate=None
):
    """
    Plot the difference heatmap along with BigWig, BED tracks, gene annotations, and chromatin loops.

    Parameters:
    - All parameters are as defined in the function signature.
    """
    plt.rcParams['font.size'] = 8
    # Adjust track spacing if needed
    single_sample = len(bigwig_files_sample2) == 0

    region = (chrid, start, end)

    # Load cooler data for case
    clr1 = cooler.Cooler(f'{cooler_file1}::resolutions/{resolution}')
    if format == "balance":
        data1 = clr1.matrix(balance=True).fetch(region).astype(float)
    elif format == "ICE":
        data1 = clr1.matrix(balance=False).fetch(region).astype(float)
    else:
        print("input format is wrong")
    # Load cooler data for control
    clr2 = cooler.Cooler(f'{cooler_file2}::resolutions/{resolution}')
    if format == "balance":
        data2 = clr2.matrix(balance=True).fetch(region).astype(float)
    elif format == "ICE":
            data2 = clr2.matrix(balance=False).fetch(region).astype(float)
    else:
        print("input format is wrong")

    # Compute difference matrix
    data_diff = None  # Initialize

    if operation == 'subtract':
        np.isnan(data1)] = 0
        np.isnan(data2)] = 0
        data_diff = data1 - data2
    elif operation == 'divide':
        if division_method == 'raw':
            # Raw division
            with np.errstate(divide='ignore', invalid='ignore'):
                data1 = np.maximum(data1, 0)
                data2 = np.maximum(data2, 0)
                data_diff = np.divide(data1, data2)
                data_diff[~np.isfinite(data_diff)] = 0  # Replace inf and NaN with 0
        elif division_method == 'add1':
            # (case +1) / (control +1)
            with np.errstate(divide='ignore', invalid='ignore'):
                data1 = np.maximum(data1, 0)
                data2 = np.maximum(data2, 0)
                data_diff = np.divide(data1 + 1, data2 + 1)
        elif division_method == 'log2':
            # Log2(case / control)
            with np.errstate(divide='ignore', invalid='ignore'):
                data1 = np.maximum(data1, 0)
                data2 = np.maximum(data2, 0)
                ratio = np.divide(data1, data2)
                bad = (ratio <= 0) | (~np.isfinite(ratio))
                ratio[bad] = np.nan  # Avoid log2 of non-positive numbers
                data_diff = np.log2(ratio)
        elif division_method == 'log2_add1':
            # log2((case +1) / (control +1))
            with np.errstate(divide='ignore', invalid='ignore'):
                data1 = np.maximum(data1, 0)
                data2 = np.maximum(data2, 0)
                data_diff = np.log2(data1 + 1, data2 + 1)
        else:
            raise ValueError("Invalid division_method. Choose among 'raw', 'log2', 'add1', 'log2_add1'.")
    else:
        raise ValueError("Invalid operation. Choose 'subtract' or 'divide'.")

    # Determine color limits for difference heatmap
    if data_diff is not None:
        # Manually set symmetric vmin and vmax based on the maximum absolute value
        max_abs = np.nanmax(np.abs(data_diff))
        vmin_diff = -max_abs
        vmax_diff = max_abs
    else:
        vmin_diff = None
        vmax_diff = None

    # Define GridSpec for vertical layout
    # Layout:
    # Row0: Difference Heatmap
    # Row1: Colorbar for difference heatmap
    # Row2: Chromatin Loops for sample1
    # Row3: Chromatin Loops for sample2
    # Rows4 to (4 + max_bigwig_bed_tracks): BigWig and BED tracks
    # Last Row: Gene Annotations
    ncols = 1
    max_bigwig_sample = len(bigwig_files_sample1) + len(bigwig_files_sample2)
    max_bed_sample = len(bed_files_sample1) + len(bed_files_sample2)
    max_tracks = max_bigwig_sample + max_bed_sample

    num_colorbars = 1
    num_loops = 0
    if loop_file_sample1:
        num_loops += 1
    if loop_file_sample2:
        num_loops += 1
    num_genes = 1 if gtf_file else 0
    # Each BigWig and BED track has one plot
    num_rows = 1 + num_colorbars + num_loops + max_tracks + num_genes

    # Define height ratios
    small_colorbar_height = 0.1  # Adjust as needed
    track_height_ratio = 0.5  # Adjust as needed for BigWig/BED tracks
    loop_track_height = 0.3
    if num_genes !=0:
        height_ratios = [1] + [small_colorbar_height] + [loop_track_height]*num_loops+[track_height_ratio] * (max_tracks) + [track_height_ratio * num_genes]
    else:
        height_ratios = [1] + [small_colorbar_height] + [loop_track_height]*num_loops+[track_height_ratio] * (max_tracks)
    gs = gridspec.GridSpec(num_rows, 1, height_ratios=height_ratios, hspace=0.3)
    # Define default figsize
    width = track_size
    height = sum(height_ratios) * (track_size / height_ratios[0]) + (num_rows -1)*track_spacing
    figsize = (width, height)

    # Create figure with calculated size
    f = plt.figure(figsize=figsize)

    # Plot Difference Heatmap
    ax_diff = f.add_subplot(gs[0, 0])
    im_diff = pcolormesh_square(ax_diff, data_diff, region[1], region[2], cmap=diff_cmap, vmin=vmin_diff, vmax=vmax_diff)
    # Format x-axis ticks
    ax_diff.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x / 1e6:.2f}'))
    ax_diff.set_title(diff_title if diff_title else "Difference Heatmap", fontsize=8)
    ax_diff.set_ylim(region[2], region[1])  # Flip y-axis to match genomic coordinates
    ax_diff.set_xlim(start, end)

    # Create a colorbar for the difference heatmap
    cax_diff = f.add_subplot(gs[1, 0])
    cbar_diff = plt.colorbar(im_diff, cax=cax_diff, orientation='horizontal')
    cbar_diff.ax.tick_params(labelsize=8)

    # Plot Chromatin Loops
    current_row = 2
    if loop_file_sample1:
        ax_loop1 = f.add_subplot(gs[current_row, 0])
        plot_loops(ax_loop1, loop_file_sample1, region, color=colors_sample1, alpha=0.7, linewidth=1, label='Sample1 Loops')
        current_row += 1
    if loop_file_sample2:
        ax_loop2 = f.add_subplot(gs[current_row, 0])
        plot_loops(ax_loop2, loop_file_sample2, region, color=colors_sample2, alpha=0.7, linewidth=1, label='Sample2 Loops')
        current_row += 1

    # Compute global min and max per BigWig type
    if track_min is not None and track_max is not None:
        type_min_max = defaultdict(lambda: (track_min, track_max))
    else:
        type_min_max = get_track_min_max(bigwig_files_sample1, bigwig_labels_sample1,
                                        bigwig_files_sample2, bigwig_labels_sample2,
                                        region=region)

    # Plot BigWig tracks for Sample1 and Sample2
    # Sample1 BigWig
    track_start_row = current_row
    if bigwig_files_sample1:
        for i in range(len(bigwig_files_sample1)):
            ax_bw = f.add_subplot(gs[track_start_row + i, 0])
            bw_type = bigwig_labels_sample1[i].split("_")[1]
            y_min, y_max = type_min_max[bw_type]
            plot_seq(ax_bw, bigwig_files_sample1[i], region, color=colors_sample1, 
                y_min=y_min, y_max=y_max)
            ax_bw.set_title(f"{bigwig_labels_sample1[i]}", fontsize=8)
            ax_bw.set_xlim(start, end)
            if y_min is not None and y_max is not None:
                ax_bw.set_ylim(y_min, y_max * 1.1)

    # Plot BigWig tracks for Sample2
    if bigwig_files_sample2:
        for j in range(len(bigwig_files_sample2)):
            ax_bw = f.add_subplot(gs[track_start_row + len(bigwig_files_sample1) + j, 0])
            #bw_index = len(bigwig_files_sample2) + j
            bw_type = bigwig_labels_sample2[j].split("_")[1]
            y_min, y_max = type_min_max[bw_type]
            plot_seq(ax_bw, bigwig_files_sample2[j], region, color=colors_sample2, 
                y_min=y_min, y_max=y_max)
            ax_bw.set_title(f"{bigwig_labels_sample2[j]}", fontsize=8)
            ax_bw.set_xlim(start, end)
            if y_min is not None and y_max is not None:
                ax_bw.set_ylim(y_min, y_max * 1.1)

    bed_start_row = track_start_row + len(bigwig_files_sample1) + len(bigwig_files_sample2)

    # Plot BED tracks for Sample1 and Sample2
    # Sample1 BED
    if bed_files_sample1:
        for k in range(len(bed_files_sample1)):
            ax_bed = f.add_subplot(gs[bed_start_row + k, 0])
            label = bed_labels_sample1[k]
            plot_bed(ax_bed, bed_files_sample1[k], region, 
                color=colors_sample1, linewidth=1, label=label)
            ax_bed.set_title(f"{bed_labels_sample1[k]}", fontsize=8)

    # Sample2 BED
    if bed_files_sample2:
        for l in range(len(bed_files_sample2)):
            ax_bed = f.add_subplot(gs[bed_start_row + len(bed_files_sample1) + l, 0])
            label = bed_labels_sample2[l]
            plot_bed(ax_bed, bed_files_sample2[l], region, 
                color=colors_sample2, linewidth=1, label=label)
            ax_bed.set_title(f"{bed_labels_sample2[l]}", fontsize=8)

    # Plot Genes if GTF file is provided
    if gtf_file:
        ax_genes = f.add_subplot(gs[num_rows-1, 0])
        plot_genes(ax_genes, gtf_file, region, genes_to_annotate=genes_to_annotate, color='blue', track_height=1)
        ax_genes.set_xlim(start, end)

    # Adjust layout and save the figure
    plt.subplots_adjust(hspace=track_spacing)
    f.savefig(output_file, bbox_inches='tight')
    plt.close(f)

def main():
    parser = argparse.ArgumentParser(description='Plot difference heatmap from cooler files with BigWig, BED tracks, gene annotations, and chromatin loops.')

    # Required arguments
    parser.add_argument('--cooler_file1', type=str, required=True, help='Path to the case .mcool file.')
    parser.add_argument('--cooler_file2', type=str, required=True, help='Path to the control .mcool file.')
    parser.add_argument('--format', type=str, default='balance', choices=['balance', 'ICE'], help='Format of .mcool file.')
    parser.add_argument('--resolution', type=int, required=True, help='Resolution for the cooler data.')
    parser.add_argument('--start', type=int, required=True, help='Start position for the region of interest.')
    parser.add_argument('--end', type=int, required=True, help='End position for the region of interest.')
    parser.add_argument('--chrid', type=str, required=True, help='Chromosome ID.')

    # Optional arguments
    parser.add_argument('--cmap', type=str, default='autumn_r', help='Colormap to be used for the combined heatmap.')
    parser.add_argument('--vmin', type=float, default=None, help='Minimum value for normalization of the combined heatmap.')
    parser.add_argument('--vmax', type=float, default=None, help='Maximum value for normalization of the combined heatmap.')
    parser.add_argument('--output_file', type=str, default='comparison_heatmap.pdf', help='Filename for the saved comparison heatmap PDF.')

    # BigWig arguments
    parser.add_argument('--bigwig_files_sample1', type=str, nargs='*', help='Paths to BigWig files for case sample.', default=[])
    parser.add_argument('--bigwig_labels_sample1', type=str, nargs='*', help='Labels for BigWig tracks of case sample.', default=[])
    parser.add_argument('--colors_sample1', type=str, default='red', help='Colors for case BigWig tracks.')
    parser.add_argument('--bigwig_files_sample2', type=str, nargs='*', help='Paths to BigWig files for control sample.', default=[])
    parser.add_argument('--bigwig_labels_sample2', type=str, nargs='*', help='Labels for BigWig tracks of control sample.', default=[])
    parser.add_argument('--colors_sample2', type=str, default='blue', help='Colors for control BigWig tracks.')

    # BED arguments
    parser.add_argument('--bed_files_sample1', type=str, nargs='*', help='Paths to BED files for case sample.', default=[])
    parser.add_argument('--bed_labels_sample1', type=str, nargs='*', help='Labels for BED tracks of case sample.', default=[])
    parser.add_argument('--bed_files_sample2', type=str, nargs='*', help='Paths to BED files for control sample.', default=[])
    parser.add_argument('--bed_labels_sample2', type=str, nargs='*', help='Labels for BED tracks of control sample.', default=[])

    # Loop arguments
    parser.add_argument('--loop_file_sample1', type=str, required=False, help='Path to the chromatin loop file for sample1.', default=None)
    parser.add_argument('--loop_file_sample2', type=str, required=False, help='Path to the chromatin loop file for sample2.', default=None)

    # New Arguments for Division Methods and Color Mapping
    parser.add_argument('--operation', type=str, default='subtract', choices=['subtract', 'divide'],
                        help="Operation to compute the difference matrix: 'subtract' (case - control) or 'divide' (case / control).")
    parser.add_argument('--division_method', type=str, default='raw', choices=['raw', 'log2', 'add1', 'log2_add1'],
                        help="Method for division when '--operation divide' is selected: 'raw' (case/control), 'log2' (log2(case/control)), 'add1' ((case+1)/(control+1)), or 'log2_add1' (log2((case+1)/(control+1))).")
    parser.add_argument('--diff_cmap', type=str, default='bwr', help="Colormap for difference matrix. Default is 'bwr' (Blue-White-Red).")
    parser.add_argument('--diff_title', type=str, default=None, help="Title for difference matrix.")

    # Track dimensions and spacing
    parser.add_argument('--track_size', type=float, default=5, help='Height of each track (in inches).')
    parser.add_argument('--track_spacing', type=float, default=0.5, help='Spacing between tracks (in inches).')
    parser.add_argument('--track_min', type=float, default=None, help='Global minimum value for all BigWig tracks.')
    parser.add_argument('--track_max', type=float, default=None, help='Global maximum value for all BigWig tracks.')
    # Gene annotation arguments
    parser.add_argument('--gtf_file', type=str, required=False, help='Path to the GTF file for gene annotations.', default=None)
    parser.add_argument('--genes_to_annotate', type=str, nargs='*', help='Gene names to annotate.', default=None)
    parser.add_argument("-V", "--version", action="version",version="DiffSquHeatmap {}".format(__version__)\
                      ,help="Print version and exit")
    args = parser.parse_args()

    # Call the plot_heatmaps function with the parsed arguments
    plot_heatmaps(
    cooler_file1=args.cooler_file1,
    cooler_file2=args.cooler_file2,
    format=args.format,
    bigwig_files_sample1=args.bigwig_files_sample1,
    bigwig_labels_sample1=args.bigwig_labels_sample1,
    colors_sample1=args.colors_sample1,
    bed_files_sample1=args.bed_files_sample1,
    bed_labels_sample1=args.bed_labels_sample1,
    loop_file_sample1=args.loop_file_sample1,
    loop_file_sample2=args.loop_file_sample2,
    gtf_file=args.gtf_file,
    resolution=args.resolution,
    start=args.start,
    end=args.end,
    chrid=args.chrid,
    cmap=args.cmap,
    vmin=args.vmin,
    vmax=args.vmax,
    track_min=args.track_min,
    track_max=args.track_max,
    output_file=args.output_file,
    bigwig_files_sample2=args.bigwig_files_sample2,
    bigwig_labels_sample2=args.bigwig_labels_sample2,
    colors_sample2=args.colors_sample2,
    bed_files_sample2=args.bed_files_sample2,
    bed_labels_sample2=args.bed_labels_sample2,
    track_size=args.track_size,
    track_spacing=args.track_spacing,
    operation=args.operation,
    division_method=args.division_method,
    diff_cmap=args.diff_cmap,
    diff_title=args.diff_title,
    genes_to_annotate=args.genes_to_annotate
    )


if __name__ == '__main__':
    main()
