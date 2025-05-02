#!/usr/bin/env python
import argparse
import os
import pandas as pd
from matplotlib.colors import LogNorm
import pyBigWig
import pyranges as pr
import numpy as np
import matplotlib.pyplot as plt
import cooler
from matplotlib.ticker import EngFormatter
import itertools
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
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

    Parameters:
    - file_path: Path to the BigWig or bedGraph file.
    - region: Tuple containing (chromosome, start, end).

    Returns:
    - positions: Numpy array of positions.
    - values: Numpy array of values.
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
    Plot RNA-seq/ChIP-seq expression from BigWig or bedGraph file on the given axis.

    Parameters:
    - ax: Matplotlib axis to plot on.
    - file_path: Path to the BigWig or bedGraph file.
    - region: Tuple containing (chromosome, start, end).
    - color: Color for the plot.
    - y_min: Minimum y-axis value.
    - y_max: Maximum y-axis value.
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
            print(f"No data found in the specified region ({chrom}:{start}-{end}) in {file_path}")
            ax.axis('off')  # Hide the axis if no data
            return
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
    
    # Plot the RNA-seq/ChIP-seq expression as a filled line plot
    ax.plot(positions, values, color=color, alpha=0.7)
    ax.set_xlim(start, end)
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    elif y_max is not None:
        ax.set_ylim(0, y_max)
    elif y_min is not None:
        ax.set_ylim(y_min, 1)  # Default upper limit if only y_min is provided
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x / 1e6:.2f}'))

def plot_bed(ax, bed_file, region, color='green', label=None):
    """
    Plot BED regions as rectangles on the given axis.

    Parameters:
    - ax: Matplotlib axis to plot on.
    - bed_file: Path to the BED file.
    - region: Tuple containing (chromosome, start, end).
    - color: Color for the BED regions.
    - label: Label for the BED track.
    """
    chrom, start, end = region
    # Read BED file using pandas
    bed_df = pd.read_csv(bed_file, sep='\t', header=None, comment='#', 
                         names=['chrom', 'start', 'end'] + [f'col{i}' for i in range(3, 15)])
    # Filter for the specified region
    region_bed = bed_df[
        (bed_df['chrom'] == chrom) &
        (bed_df['end'] > start) &
        (bed_df['start'] < end)
    ]
    if region_bed.empty:
        print(f"No BED regions found in the specified region ({chrom}:{start}-{end}) in {bed_file}.")
        ax.axis('off')
        return
    # Plot each BED region as a rectangle
    for _, row in region_bed.iterrows():
        rect_start = max(row['start'], start)
        rect_end = min(row['end'], end)
        ax.add_patch(
            plt.Rectangle(
                (rect_start, 0),
                rect_end - rect_start,
                1,
                color=color,
                alpha=0.5
            )
        )
    ax.set_xlim(start, end)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
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
        ax.axis('off')
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

def pcolormesh_triangle(ax, matrix, start=0, resolution=1, NORM=True,vmin=None, vmax=None, cmap='autumn_r', *args, **kwargs):
    """
    Plot the Hi-C matrix as a triangular heatmap on the given axis.

    Parameters:
    - ax: Matplotlib axis to plot on.
    - matrix: Hi-C contact matrix.
    - start: Start position for the x-axis.
    - resolution: Resolution of the Hi-C matrix.
    - vmin: Minimum value for color scaling.
    - vmax: Maximum value for color scaling.
    - cmap: Colormap for the heatmap.
    """
    n = matrix.shape[0]
    # Triangle orientation
    start_pos_vector = [start + resolution * i for i in range(len(matrix) + 1)]
    t = np.array([[1, 0.5], [-1, 0.5]])
    matrix_a = np.dot(
            np.array([(i[1], i[0]) for i in itertools.product(start_pos_vector[::-1], start_pos_vector)]),
            t)
    x, y = matrix_a[:, 1].reshape(n + 1, n + 1), matrix_a[:, 0].reshape(n + 1, n + 1)
    if NORM:
        log_vmin = vmin if vmin is not None and vmin > 0 else None
        norm = LogNorm(vmin=log_vmin, vmax=vmax, clip=False)
        im = ax.pcolormesh(x, y, np.flipud(matrix), norm=norm, cmap=cmap, *args, **kwargs)
    else:
        im = ax.pcolormesh(x, y, np.flipud(matrix), vmin=vmin, vmax=vmax, cmap=cmap, *args, **kwargs)
    ax.yaxis.set_visible(False)
    im.set_rasterized(True)
    return im

def plot_heatmaps(cooler_file1, sampleid1,format="balance",
                 bigwig_files_sample1=[], bigwig_labels_sample1=[], colors_sample1="red",
                 loop_file_sample1=None, loop_file_sample2=None,
                 bed_files_sample1=[], bed_labels_sample1=[],
                 gtf_file=None,cooler_file2=None, sampleid2=None,
                 bigwig_files_sample2=[], bigwig_labels_sample2=[], colors_sample2="blue",
                 bed_files_sample2=[], bed_labels_sample2=[],
                 resolution=None,start=None, end=None, chrid=None,
                 cmap='autumn_r', vmin=None, vmax=None,
                 track_min=None,track_max=None,
                 output_file='comparison_heatmap.pdf', layout='horizontal',
                 track_width=10, track_height=1, track_spacing=0.5,
                 normalization_method='raw', genes_to_annotate=None):  # Added normalization_method
    """
    Plot Hi-C heatmaps along with BigWig, BED, and gene annotations.

    Parameters:
    - All parameters correspond to command-line arguments.
    """
    plt.rcParams['font.size'] = 8
    track_spacing = track_spacing * 1.2
    # Define a small height for the colorbar
    small_colorbar_height = 0.1  # Adjust this value to make the colorbar height small
    # Set parameters
    region = (chrid, start, end)
    # Load cooler data
    clr1 = cooler.Cooler(f'{cooler_file1}::resolutions/{resolution}')
    if format == "balance":
        data1 = clr1.matrix(balance=True).fetch(region).astype(float)
    elif format == "ICE":
        data1 = clr1.matrix(balance=False).fetch(region).astype(float)
    else:
        print("input format is wrong")
    # Load sample2 data if provided
    single_sample = cooler_file2 is None
    if not single_sample:
        clr2 = cooler.Cooler(f'{cooler_file2}::resolutions/{resolution}')
        if format == "balance":
            data2 = clr2.matrix(balance=True).fetch(region).astype(float)
        elif format == "ICE":
            data2 = clr2.matrix(balance=False).fetch(region).astype(float)
        else:
            print("input format is wrong")

    # Apply normalization to Hi-C matrices
    if normalization_method == 'raw':
        normalized_data1 = data1
        normalized_data2 = data2 if not single_sample else None
    elif normalization_method == 'logNorm':
        normalized_data1 = np.maximum(data1, 0)
        if not single_sample:
            normalized_data2 = np.maximum(data2, 0)
    elif normalization_method == 'log2':
        normalized_data1 = np.log2(data1)
        if not single_sample:
            normalized_data2 = np.log2(data2)
    elif normalization_method == 'log2_add1':
        normalized_data1 = np.log2(data1 + 1)
        if not single_sample:
            normalized_data2 = np.log2(data2 + 1)
    elif normalization_method == 'log':
        normalized_data1 = np.log(data1)
        if not single_sample:
            normalized_data2 = np.log(data2)
    elif normalization_method == 'log_add1':
        normalized_data1 = np.log(data1 + 1)
        if not single_sample:
            normalized_data2 = np.log(data2 + 1)
    else:
        raise ValueError(f"Unsupported normalization method: {normalization_method}")
    
    # Determine vmin and vmax if not provided
    if vmin is None and vmax is None:
        if single_sample:
            vmin = np.nanmin(normalized_data1)
            vmax = np.nanmax(normalized_data1)
        else:
            vmin = min(np.nanmin(normalized_data1), np.nanmin(normalized_data2))
            vmax = max(np.nanmax(normalized_data1), np.nanmax(normalized_data2))
    elif vmin is None:
        if normalization_method.startswith('log'):
            if single_sample:
                vmin = np.nanmin(normalized_data1)
            else:
                vmin = min(np.nanmin(normalized_data1), np.nanmin(normalized_data2))
        else:
            vmin = 0  # For raw data
    elif vmax is None:
        if normalization_method.startswith('log'):
            if single_sample:
                vmax = np.nanmax(normalized_data1)
            else:
                vmax = max(np.nanmax(normalized_data1), np.nanmax(normalized_data2))
        else:
            if single_sample:
                vmax = np.nanmax(normalized_data1)
            else:
                vmax = max(np.nanmax(normalized_data1), np.nanmax(normalized_data2))
    
    # Formatter for big positions
    bp_formatter = EngFormatter()
    def format_ticks(ax, x=True, y=True, rotate=True):
        def format_million(x, pos):
            return f'{x / 1e6:.2f}'
        if y:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(format_million))
        if x:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(format_million))
            ax.xaxis.tick_bottom()
        if rotate:
            ax.tick_params(axis='x', rotation=45)
    
    if layout == 'horizontal':
        # Determine the number of tracks
        num_genes = 1 if gtf_file else 0
        ncols = 1 if single_sample else 2
        # Calculate the maximum number of BigWig and BED tracks per sample
        max_bigwig_sample = max(len(bigwig_files_sample1), len(bigwig_files_sample2)) if not single_sample else len(bigwig_files_sample1)
        max_bed_sample = max(len(bed_files_sample1), len(bed_files_sample2)) if not single_sample else len(bed_files_sample1)
        max_bigwig_bed_tracks = max(max_bigwig_sample, max_bed_sample) if not single_sample else max(len(bigwig_files_sample1), len(bed_files_sample1))
        
        # Total rows:
        # Row0: Heatmaps
        # Row1: Colorbars
        # Rows2 to (2 + max_bigwig_bed_tracks -1): BigWig and BED tracks
        # Last row: Genes (optional)
        # Include loops in the calculation of number of rows
        num_loops = 0
        if loop_file_sample1:
            num_loops = 1
        if loop_file_sample2:
            num_loops = 1
        num_rows = 2 + num_loops + max_bigwig_bed_tracks + num_genes
        
        # Define height ratios
        loop_track_height = 0.3
        if num_genes !=0:
            height_ratios = [1, small_colorbar_height] + [loop_track_height]*num_loops + [0.5] * max_bigwig_bed_tracks + [0.5] * num_genes
        else:
            height_ratios = [1, small_colorbar_height] + [loop_track_height]*num_loops + [0.5] * max_bigwig_bed_tracks
        # Initialize GridSpec
        gs = gridspec.GridSpec(num_rows, ncols, height_ratios=height_ratios, hspace=0.5, wspace=0.3)
        # Define default figsize
        figsize_width = track_width * ncols + (ncols -1)*track_spacing
        figsize_height = sum(height_ratios) * track_height + (num_rows -1)*track_spacing
        # Initialize figure
        f = plt.figure(figsize=(figsize_width, figsize_height))
        
        # Plot Heatmaps
        ax_heatmap1 = f.add_subplot(gs[0, 0])
        if normalization_method == "logNorm":
            im1 = pcolormesh_triangle(ax_heatmap1, normalized_data1, start=region[1], resolution=resolution, NORM=True,vmin=vmin, vmax=vmax,cmap=cmap)
        else:
            im1 = pcolormesh_triangle(ax_heatmap1, normalized_data1, start=region[1], resolution=resolution, NORM=False,vmin=vmin, vmax=vmax,cmap=cmap)
        ax_heatmap1.set_aspect('auto')
        ax_heatmap1.set_ylim(0, normalized_data1.shape[0] * resolution)
        ax_heatmap1.set_xlim(start, end)
        format_ticks(ax_heatmap1, rotate=False)
        ax_heatmap1.set_title(sampleid1, fontsize=8)
        
        if not single_sample:
            ax_heatmap2 = f.add_subplot(gs[0, 1])
            if normalization_method == "logNorm":
                im2 = pcolormesh_triangle(ax_heatmap2, normalized_data2, start=region[1], resolution=resolution, NORM=True,vmin=vmin, vmax=vmax,cmap=cmap)
            else:
                im2 = pcolormesh_triangle(ax_heatmap2, normalized_data2, start=region[1], resolution=resolution, NORM=False,vmin=vmin, vmax=vmax,cmap=cmap)
            ax_heatmap2.set_aspect('auto')
            ax_heatmap2.set_ylim(0, normalized_data2.shape[0] * resolution)
            ax_heatmap2.set_xlim(start, end)
            format_ticks(ax_heatmap2, rotate=False)
            ax_heatmap2.set_title(sampleid2, fontsize=8)
        
        # Create colorbars
        if not single_sample:
            # Colorbar for Sample1
            cax1 = f.add_subplot(gs[1, 0])
            cbar1 = plt.colorbar(im1, cax=cax1, orientation='horizontal')
            cbar1.ax.tick_params(labelsize=8)
            cax1.xaxis.set_label_position('bottom')
            cax1.xaxis.set_ticks_position('bottom')
            cbar1.set_label(normalization_method, labelpad=3)
            cbar1.ax.xaxis.set_label_position('top')
            # Colorbar for Sample2
            cax2 = f.add_subplot(gs[1, 1])
            cbar2 = plt.colorbar(im2, cax=cax2, orientation='horizontal')
            cbar2.ax.tick_params(labelsize=8)
            cax2.xaxis.set_label_position('bottom')
            cax2.xaxis.set_ticks_position('bottom')
            cbar2.set_label(normalization_method, labelpad=3)
            cbar2.ax.xaxis.set_label_position('top')
        else:
            # Single colorbar if only one sample
            cax1 = f.add_subplot(gs[1, 0])
            cbar1 = plt.colorbar(im1, cax=cax1, orientation='horizontal')
            cbar1.ax.tick_params(labelsize=8)
            cax1.xaxis.set_label_position('bottom')
            cax1.xaxis.set_ticks_position('bottom')
            cbar1.set_label(normalization_method, labelpad=3)
            cbar1.ax.xaxis.set_label_position('top')

        # Plot loops if provided
        current_row = 2
        if loop_file_sample1 and not loop_file_sample2:
            ax_loop1 = f.add_subplot(gs[current_row, 0])
            plot_loops(ax_loop1, loop_file_sample1, region, color=colors_sample1, alpha=0.7, linewidth=1, label=f"{sampleid1} Loops")
            current_row += 1
        elif loop_file_sample1 and loop_file_sample2:
            ax_loop1 = f.add_subplot(gs[current_row, 0])
            plot_loops(ax_loop1, loop_file_sample1, region, color=colors_sample1, alpha=0.7, linewidth=1, label=f"{sampleid1} Loops")
            ax_loop2 = f.add_subplot(gs[current_row, 1])
            plot_loops(ax_loop2, loop_file_sample2, region, color=colors_sample2, alpha=0.7, linewidth=1, label=f"{sampleid2} Loops")
            current_row += 1

        # Compute global min and max per BigWig type
        if track_min is not None and track_max is not None:
            type_min_max = defaultdict(lambda: (track_min, track_max))
        else:
            type_min_max = get_track_min_max(bigwig_files_sample1, bigwig_labels_sample1,
                                        bigwig_files_sample2, bigwig_labels_sample2,
                                        region=region)
        # Plot BigWig and BED tracks
        track_start_row = current_row
        # Plot BigWig tracks for Sample1
        if bigwig_files_sample1:
            for i in range(len(bigwig_files_sample1)):
                bw_type = bigwig_labels_sample1[i].split("_")[1]
                y_min, y_max = type_min_max[bw_type]
                ax_bw = f.add_subplot(gs[track_start_row + i, 0])
                plot_seq(ax_bw, bigwig_files_sample1[i], region, color=colors_sample1, 
                         y_min=y_min, y_max=y_max)
                ax_bw.set_title(f"{bigwig_labels_sample1[i]} ({sampleid1})", fontsize=8)
                ax_bw.set_xlim(start, end)
                if y_min is not None and y_max is not None:
                    ax_bw.set_ylim(y_min, y_max * 1.1)
        # Plot BigWig tracks for Sample2 if provided
        if bigwig_files_sample2:
            for j in range(len(bigwig_files_sample2)):
                bw_type = bigwig_labels_sample2[j].split("_")[1]
                y_min, y_max = type_min_max[bw_type]
                ax_bw = f.add_subplot(gs[track_start_row + j, 1])
                # Handle colors_sample2 being a list or single color
                plot_seq(ax_bw, bigwig_files_sample2[j], region, color=colors_sample2, 
                         y_min=y_min, y_max=y_max)
                ax_bw.set_title(f"{bigwig_labels_sample2[j]} ({sampleid2})", fontsize=8)
                ax_bw.set_xlim(start, end)
                if y_min is not None and y_max is not None:
                    ax_bw.set_ylim(y_min, y_max * 1.1)
        # Plot BED tracks for Sample1
        bed_start_row = track_start_row + len(bigwig_files_sample1)
        if bed_files_sample1:
            for k in range(len(bed_files_sample1)):
                ax_bed = f.add_subplot(gs[bed_start_row + k, 0])
                plot_bed(ax_bed, bed_files_sample1[k], region, color=colors_sample1, label=bed_labels_sample1[k])
                ax_bed.set_title(f"{bed_labels_sample1[k]} ({sampleid1})", fontsize=8)
        # Plot BED tracks for Sample2 if provided
        if bed_files_sample2:
            for l in range(len(bed_files_sample2)):
                ax_bed = f.add_subplot(gs[bed_start_row + l, 1])
                plot_bed(ax_bed, bed_files_sample2[l], region, color=colors_sample2, label=bed_labels_sample2[l])
                ax_bed.set_title(f"{bed_labels_sample2[l]} ({sampleid2})", fontsize=8)
        # Plot Genes if GTF file is provided
        if gtf_file:
            gene_row = 2 + num_loops + max_bigwig_bed_tracks
            ax_genes = f.add_subplot(gs[gene_row, 0])
            plot_genes(ax_genes, gtf_file, region, track_height=track_height, genes_to_annotate=genes_to_annotate)
            ax_genes.set_xlim(start, end)
            if not single_sample:
                ax_genes2 = f.add_subplot(gs[gene_row, 1])
                plot_genes(ax_genes2, gtf_file, region, track_height=track_height, genes_to_annotate=genes_to_annotate)
                ax_genes2.set_xlim(start, end)
    elif layout == 'vertical':
        # Vertical layout handling
        # Each track is stacked vertically
        # Row0: Heatmap Sample1
        # Row1: Heatmap Sample2 (if exists)
        # Row2: Colorbars (single)
        # Rows3 to N: BigWig and BED tracks
        # Last row: Genes (optional)
        # Determine the number of tracks
        num_genes = 1 if gtf_file else 0
        ncols = 1
        max_cool_sample = 1 if single_sample else 2
        # Calculate the maximum number of tracks across samples
        max_bigwig_sample = len(bigwig_files_sample1) + len(bigwig_files_sample2)
        max_bed_sample = len(bed_files_sample1) + len(bed_files_sample2)
        max_tracks = max_bigwig_sample + max_bed_sample
        # Calculate colorbar rows
        # In vertical layout, we only want one colorbar
        num_colorbars = 1
        num_loops = 0
        if loop_file_sample1:
            num_loops += 1
        if loop_file_sample2:
            num_loops += 1
        # Total rows:
        # Heatmap1
        # Heatmap2 (if dual sample)
        # Colorbar
        # Tracks
        # Genes (optional)
        num_rows = max_cool_sample + num_colorbars + num_loops + max_tracks + num_genes
        # Define height ratios
        loop_track_height = 0.3
        if num_genes !=0:
            height_ratios = [track_height] * max_cool_sample + [small_colorbar_height] + \
                        [loop_track_height]*num_loops + [track_height] * max_tracks + \
                        [track_height] * num_genes
        else:
            height_ratios = [track_height] * max_cool_sample + [small_colorbar_height] + \
                        [loop_track_height]*num_loops + [track_height] * max_tracks

        # Initialize GridSpec
        gs = gridspec.GridSpec(num_rows, ncols, height_ratios=height_ratios, hspace=0.5)
        # Define default figsize
        width = track_width * ncols
        height = (track_height * num_rows) + small_colorbar_height
        figsize = (width, height)
        # Initialize figure
        f = plt.figure(figsize=figsize)
        # Plot Heatmaps
        ax_heatmap1 = f.add_subplot(gs[0, 0])
        if normalization_method == "logNorm":
            im1 = pcolormesh_triangle(ax_heatmap1, normalized_data1, start=region[1], resolution=resolution, NORM=True,vmin=vmin, vmax=vmax,cmap=cmap)
        else:
            im1 = pcolormesh_triangle(ax_heatmap1, normalized_data1, start=region[1], resolution=resolution, NORM=False,vmin=vmin, vmax=vmax,cmap=cmap)
        ax_heatmap1.set_aspect('auto')
        ax_heatmap1.set_ylim(0, normalized_data1.shape[0] * resolution)
        ax_heatmap1.set_xlim(start, end)
        format_ticks(ax_heatmap1, rotate=False)
        ax_heatmap1.set_title(sampleid1, fontsize=8)
        
        if not single_sample:
            ax_heatmap2 = f.add_subplot(gs[1, 0])
            if normalization_method == "logNorm":
                im2 = pcolormesh_triangle(ax_heatmap2, normalized_data2, start=region[1], resolution=resolution, NORM=True,vmin=vmin, vmax=vmax,cmap=cmap)
            else:
                im2 = pcolormesh_triangle(ax_heatmap2, normalized_data2, start=region[1], resolution=resolution, NORM=False,vmin=vmin, vmax=vmax,cmap=cmap)
            ax_heatmap2.set_aspect('auto')
            ax_heatmap2.set_ylim(0, normalized_data2.shape[0] * resolution)
            ax_heatmap2.set_xlim(start, end)
            format_ticks(ax_heatmap2, rotate=False)
            ax_heatmap2.set_title(sampleid2, fontsize=8)

        # Create a single colorbar for vertical layout
        cax = f.add_subplot(gs[max_cool_sample, 0])
        cbar = plt.colorbar(im1, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(normalization_method, labelpad=3)  # Set the label above
        cbar.ax.xaxis.set_label_position('top')

        # Plot loops
        current_row = max_cool_sample + num_colorbars
        if loop_file_sample1:
            ax_loop1 = f.add_subplot(gs[current_row, 0])
            plot_loops(ax_loop1, loop_file_sample1, region, color=colors_sample1, alpha=0.7, linewidth=1, label=f"{sampleid1} Loops")
            current_row += 1
        if loop_file_sample2:
            ax_loop2 = f.add_subplot(gs[current_row, 0])
            plot_loops(ax_loop2, loop_file_sample2, region, color=colors_sample2, alpha=0.7, linewidth=1, label=f"{sampleid2} Loops")
            current_row += 1

        # Compute global min and max per BigWig type
        if track_min is not None and track_max is not None:
            type_min_max = defaultdict(lambda: (track_min, track_max))
        else:
            type_min_max = get_track_min_max(bigwig_files_sample1, bigwig_labels_sample1,
                                        bigwig_files_sample2, bigwig_labels_sample2,
                                        region=region)
        # Plot BigWig and BED tracks
        # Sample1 BigWig
        track_start_row = current_row
        if bigwig_files_sample1:
            for i in range(len(bigwig_files_sample1)):
                bw_type = bigwig_labels_sample1[i].split("_")[1]
                y_min, y_max = type_min_max[bw_type]
                ax_bw = f.add_subplot(gs[track_start_row + i, 0])
                plot_seq(ax_bw, bigwig_files_sample1[i], region, color=colors_sample1, 
                    y_min=y_min, y_max=y_max)
                ax_bw.set_title(f"{bigwig_labels_sample1[i]} ({sampleid1})", fontsize=8)
                ax_bw.set_xlim(start, end)
                if y_min is not None and y_max is not None:
                    ax_bw.set_ylim(y_min, y_max * 1.1)
        track_start_row = track_start_row + len(bigwig_files_sample1)
        # Sample2 BigWig
        if bigwig_files_sample2:
            for j in range(len(bigwig_files_sample2)):
                bw_type = bigwig_labels_sample2[j].split("_")[1]
                y_min, y_max = type_min_max[bw_type]
                ax_bw = f.add_subplot(gs[track_start_row + j, 0])
                plot_seq(ax_bw, bigwig_files_sample2[j], region, color=colors_sample2, 
                         y_min=y_min, y_max=y_max)
                ax_bw.set_title(f"{bigwig_labels_sample2[j]} ({sampleid2})", fontsize=8)
                ax_bw.set_xlim(start, end)
                if y_min is not None and y_max is not None:
                    ax_bw.set_ylim(y_min, y_max * 1.1)
        # Sample1 BED
        bed_start_row = track_start_row + len(bigwig_files_sample1) + len(bigwig_files_sample2)
        if bed_files_sample1:
            for k in range(len(bed_files_sample1)):
                ax_bed = f.add_subplot(gs[bed_start_row + k, 0])
                plot_bed(ax_bed, bed_files_sample1[k], region, color=colors_sample1, label=bed_labels_sample1[k])
                ax_bed.set_title(f"{bed_labels_sample1[k]} ({sampleid1})", fontsize=8)
        # Sample2 BED
        bed_start_row = track_start_row + len(bigwig_files_sample1) + len(bigwig_files_sample2) + len(bed_files_sample2)
        if bed_files_sample2:
            for l in range(len(bed_files_sample2)):
                ax_bed = f.add_subplot(gs[bed_start_row + l, 0])
                plot_bed(ax_bed, bed_files_sample2[l], region, color=colors_sample2, label=bed_labels_sample2[l])
                ax_bed.set_title(f"{bed_labels_sample2[l]} ({sampleid2})", fontsize=8)
        # Plot Genes if GTF file is provided
        if gtf_file:
            gene_row = max_cool_sample + 1 + max_tracks + num_loops
            ax_genes = f.add_subplot(gs[gene_row, 0])
            plot_genes(ax_genes, gtf_file, region, track_height=track_height, genes_to_annotate=genes_to_annotate)
            ax_genes.set_xlim(start, end)
    else:
        raise ValueError("Invalid layout option. Use 'horizontal' or 'vertical'.")
    
    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    # Save the figure
    f.savefig(output_file, bbox_inches='tight')
    plt.close(f)

def main():
    """
    Main function to parse command-line arguments and initiate plotting.
    """
    parser = argparse.ArgumentParser(description='Plot triangle heatmaps from cooler files with optional BigWig, BED, and GTF annotations.')
    # Required Hi-C matrix
    parser.add_argument('--cooler_file1', type=str, required=True, help='Path to the first .mcool file.')
    # Optional second Hi-C matrix
    parser.add_argument('--cooler_file2', type=str, required=False, help='Path to the second .mcool file.', default=None)
    parser.add_argument('--format', type=str, default='balance', choices=['balance', 'ICE'], help='Format of .mcool file.')

    # Resolution and genomic region
    parser.add_argument('--resolution', type=int, default=10000, help='Resolution for the cooler data.')
    parser.add_argument('--start', type=int, default=10500000, help='Start position for the region of interest.')
    parser.add_argument('--end', type=int, default=13200000, help='End position for the region of interest.')
    parser.add_argument('--chrid', type=str, default='chr2', help='Chromosome ID.')
    # Visualization parameters
    parser.add_argument('--cmap', type=str, default='autumn_r', help='Colormap to be used for plotting.')
    parser.add_argument('--vmin', type=float, default=None, help='Minimum value for Hi-C matrix.')
    parser.add_argument('--vmax', type=float, default=None, help='Maximum value for Hi-C matrix.')
    parser.add_argument('--output_file', type=str, default='comparison_heatmap.pdf', help='Filename for the saved comparison heatmap PDF.')
    parser.add_argument('--layout', type=str, default='horizontal', choices=['horizontal', 'vertical'], help="Layout of the heatmaps: 'horizontal' or 'vertical'.")
    # Sample IDs
    parser.add_argument('--sampleid1', type=str, default='Sample1', help='Sample ID for the first dataset.')
    parser.add_argument('--sampleid2', type=str, default='Sample2', help='Sample ID for the second dataset.')
    # Optional GTF file for gene annotations
    parser.add_argument('--gtf_file', type=str, required=False, help='Path to the GTF file for gene annotations.', default=None)
    
    # BigWig arguments
    parser.add_argument('--bigwig_files_sample1', type=str, nargs='*', help='Paths to BigWig files for sample 1.', default=[])
    parser.add_argument('--bigwig_labels_sample1', type=str, nargs='*', help='Labels for BigWig tracks of sample 1.', default=[])
    parser.add_argument('--colors_sample1', type=str, default="red", help='Colors for sample 1 tracks.')
    parser.add_argument('--bigwig_files_sample2', type=str, nargs='*', help='Paths to BigWig files for sample 2.', default=[])
    parser.add_argument('--bigwig_labels_sample2', type=str, nargs='*', help='Labels for BigWig tracks of sample 2.', default=[])
    parser.add_argument('--colors_sample2', type=str, default="blue", help='Colors for sample 2 tracks.')
    
    # BED arguments
    parser.add_argument('--bed_files_sample1', type=str, nargs='*', help='Paths to BED files for sample 1.', default=[])
    parser.add_argument('--bed_labels_sample1', type=str, nargs='*', help='Labels for BED tracks of sample 1.', default=[])
    parser.add_argument('--bed_files_sample2', type=str, nargs='*', help='Paths to BED files for sample 2.', default=[])
    parser.add_argument('--bed_labels_sample2', type=str, nargs='*', help='Labels for BED tracks of sample 2.', default=[])
    
    # Track dimensions and spacing
    parser.add_argument('--track_width', type=float, default=10, help='Width of each track (in inches).')
    parser.add_argument('--track_height', type=float, default=1, help='Height of each track (in inches).')
    parser.add_argument('--track_spacing', type=float, default=0.5, help='Spacing between tracks (in inches).')
    
    # New Argument for Normalization Method
    parser.add_argument('--normalization_method', type=str, default='raw', choices=['raw', 'logNorm','log2', 'log2_add1','log','log_add1'],
                        help="Method for normalization: 'raw', 'logNorm','log2', 'log2_add1', 'log', or 'log_add1'.")
    # Loop file arguments
    parser.add_argument('--loop_file_sample1', type=str, help='Path to the chromatin loop file for sample 1.', default=None)
    parser.add_argument('--loop_file_sample2', type=str, help='Path to the chromatin loop file for sample 2.', default=None)
    # Gene annotation arguments
    parser.add_argument('--track_min', type=float, default=None, help='Global minimum value for all BigWig tracks.')
    parser.add_argument('--track_max', type=float, default=None, help='Global maximum value for all BigWig tracks.')
    parser.add_argument('--genes_to_annotate', type=str, nargs='*', help='Gene names to annotate.', default=None)
    parser.add_argument("-V", "--version", action="version",version="TriHeatmap {}".format(__version__)\
                      ,help="Print version and exit")
    args = parser.parse_args()
    # Call the plotting function with the normalization_method argument
    plot_heatmaps(
        cooler_file1=args.cooler_file1,
        sampleid1=args.sampleid1,
        bigwig_files_sample1=args.bigwig_files_sample1,
        bigwig_labels_sample1=args.bigwig_labels_sample1,
        colors_sample1=args.colors_sample1,
        loop_file_sample1=args.loop_file_sample1,
        loop_file_sample2=args.loop_file_sample2,
        bed_files_sample1=args.bed_files_sample1,
        bed_labels_sample1=args.bed_labels_sample1,
        gtf_file=args.gtf_file,
        cooler_file2=args.cooler_file2,
        sampleid2=args.sampleid2,
        bigwig_files_sample2=args.bigwig_files_sample2,
        bigwig_labels_sample2=args.bigwig_labels_sample2,
        colors_sample2=args.colors_sample2,
        bed_files_sample2=args.bed_files_sample2,
        bed_labels_sample2=args.bed_labels_sample2,
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
        layout=args.layout,
        track_width=args.track_width,
        track_height=args.track_height,
        track_spacing=args.track_spacing,
        normalization_method=args.normalization_method,
        genes_to_annotate=args.genes_to_annotate,
        format=args.format
    )

if __name__ == '__main__':
    main()
