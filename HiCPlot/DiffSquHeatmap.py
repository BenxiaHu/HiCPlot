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

dir = os.path.dirname(os.path.abspath(__file__))
version_py = os.path.join(dir, "_version.py")
exec(open(version_py).read())

def plot_genes(ax, gtf_file, region, color='blue', track_height=1):
    """
    Plot gene annotations on the given axis.
    """
    spacing_factor = 1.5
    chrom, start, end = region
    # Load the GTF file using pyranges
    gtf = pr.read_gtf(gtf_file)
    # Filter relevant region and keep only the longest isoform for each gene
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
        
        # Add gene name at the center of the gene, adjusted vertically
        ax.text(
            (gene['Start'] + gene['End']) / 2,
            y_offset + 0.4 * track_height,  # Adjusted position for better alignment
            gene['gene_name'],
            fontsize=8,  # Increased font size for readability
            ha='center',
            va='bottom'  # Align text below the exon
        )
        
        # Track the plotted gene's range and offset
        plotted_genes.append({'Start': gene['Start'], 'End': gene['End'], 'y_offset': y_offset})
    
    # Set y-axis limits based on the final y_offset
    ax.set_ylim(-track_height, y_offset + track_height * 2)
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

def get_track_min_max(bigwig_files_sample1, bigwig_files_sample2, layoutid, region):
    """
    Determine the minimum and maximum values across all BigWig files to set consistent y-axis limits.
    """
    max_num_tracks = len(bigwig_files_sample1) + len(bigwig_files_sample2)
    min_max_list = []

    for i in range(max_num_tracks):
        min_val = np.inf
        max_val = -np.inf

        if i < len(bigwig_files_sample1):
            positions, values = read_bigwig(bigwig_files_sample1[i], region)
            if values is not None and len(values) > 0:
                current_min = np.nanmin(values)
                current_max = np.nanmax(values)
                min_val = min(min_val, current_min)
                max_val = max(max_val, current_max)

        if i < len(bigwig_files_sample2):
            positions, values = read_bigwig(bigwig_files_sample2[i], region)
            if values is not None and len(values) > 0:
                current_min = np.nanmin(values)
                current_max = np.nanmax(values)
                min_val = min(min_val, current_min)
                max_val = max(max_val, current_max)

        # Handle cases where no data was found for the track
        if min_val == np.inf and max_val == -np.inf:
            min_max_list.append((None, None))
        else:
            min_max_list.append((min_val, max_val))

    return min_max_list

def plot_seq(ax, file_path, region, color='blue', y_min=None, y_max=None):
    """
    Plot RNA-seq/ChIP-seq expression from BigWig or bedGraph file on given axis.
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
    ax.fill_between(positions, values, color=color, alpha=0.7)
    ax.set_xlim(start, end)
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    elif y_max is not None:
        ax.set_ylim(0, y_max)
    elif y_min is not None:
        ax.set_ylim(y_min, 1)  # Default upper limit if only y_min is provided
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x / 1e6:.2f}'))

def plot_bed(ax, bed_file, region, color='green', linewidth=1):
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
        print(f"No BED entries found in the specified region ({chrom}:{start}-{end}) in {bed_file}")
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

def pcolormesh_square(ax, matrix, start, end, cmap='bwr', vmin=None, vmax=None, *args, **kwargs):
    """
    Plot the difference matrix as a heatmap on the given axis.
    """
    if matrix is None:
        return None
    im = ax.imshow(matrix, aspect='auto', origin='upper',
                   extent=[start, end, end, start], cmap=cmap, vmin=vmin, vmax=vmax, *args, **kwargs)
    return im

def plot_heatmaps(cooler_file1,
                 bigwig_files_sample1, bigwig_labels_sample1, colors_sample1,
                 bed_files_sample1, bed_labels_sample1, colors_bed_sample1,
                 gtf_file, resolution,
                 start, end, chrid,
                 cmap='autumn_r', vmin=None, vmax=None,
                 output_file='comparison_heatmap.pdf',
                 cooler_file2=None,
                 bigwig_files_sample2=[], bigwig_labels_sample2=[], colors_sample2=[],
                 bed_files_sample2=[], bed_labels_sample2=[], colors_bed_sample2=[],
                 track_size=5, track_spacing=0.5,
                 operation='subtract', division_method='raw',
                 diff_cmap='bwr',diff_title=None):
    """
    Plot the difference heatmap along with BigWig, BED tracks, and gene annotations.
    """
    plt.rcParams['font.size'] = 8
        # Set parameters
    region = (chrid, start, end)
    # Load cooler data for case
    clr1 = cooler.Cooler(f'{cooler_file1}::resolutions/{resolution}')
    data1 = clr1.matrix(balance=True).fetch(region).astype(float)
    
    # Load cooler data for control
    clr2 = cooler.Cooler(f'{cooler_file2}::resolutions/{resolution}')
    data2 = clr2.matrix(balance=True).fetch(region).astype(float)
    
    # Compute difference matrix
    data_diff = None  # Initialize
    diff_title = diff_title
    
    if operation == 'subtract':
        data_diff = data1 - data2
        diff_title = diff_title
    elif operation == 'divide':
        if division_method == 'raw':
            # Raw division
            with np.errstate(divide='ignore', invalid='ignore'):
                data_diff = np.divide(data1, data2)
                data_diff[~np.isfinite(data_diff)] = 0  # Replace inf and NaN with 0
            diff_title = diff_title
        elif division_method == 'log2':
            # Log2(case / control)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.divide(data1, data2)
                ratio[ratio <= 0] = np.nan  # Avoid log2 of non-positive numbers
                data_diff = np.log2(ratio)
            diff_title = diff_title
        elif division_method == 'add1':
            # (case +1) / (control +1)
            with np.errstate(divide='ignore', invalid='ignore'):
                data_diff = np.divide(data1 + 1, data2 + 1)
                data_diff[~np.isfinite(data_diff)] = 0
            diff_title = diff_title
        elif division_method == 'log2_add1':
            # log2((case +1) / (control +1))
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.divide(data1 + 1, data2 + 1)
                ratio[ratio <= 0] = np.nan
                data_diff = np.log2(ratio)
            diff_title = diff_title
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
    # Rows2 to (2 + max_bigwig_bed_tracks * 2): BigWig tracks for Sample1 and Sample2
    # RowsN: BED tracks for Sample1 and Sample2
    # Last Row: Gene Annotations
    ncols = 1
    max_bigwig_sample = len(bigwig_files_sample1) + len(bigwig_files_sample2)
    max_bed_sample = len(bed_files_sample1) + len(bed_files_sample2)
    max_tracks = max_bigwig_sample + max_bed_sample
    
    num_colorbars = 1
    num_genes = 1 if gtf_file else 0
    # Each BigWig track has two plots (Sample1 and Sample2)
    num_rows = 1 + num_colorbars + max_tracks + num_genes
    
    # Define height ratios
    small_colorbar_height = 0.1
    #height_ratios = track_height[1] + [small_colorbar_height] + [0.5] * max_tracks + [0.5] * num_genes

    # Define height ratios
    height_ratios = [track_size]*1 + [small_colorbar_height] + [track_size/5]* (max_tracks) + [track_size/5]

    gs = gridspec.GridSpec(num_rows, 1, height_ratios=height_ratios, hspace=0.5)
    # Define default figsize
    width = track_size
    height = sum(height_ratios) + 2
    figsize = (width, height)

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

    # Create figure with calculated size
    f = plt.figure(figsize=figsize)
    
    # Plot Difference Heatmap
    ax_diff = f.add_subplot(gs[0, 0])
    im_diff = pcolormesh_square(ax_diff, data_diff, region[1], region[2], cmap=diff_cmap, vmin=vmin_diff, vmax=vmax_diff)
    format_ticks(ax_diff, rotate=False)
    ax_diff.set_title(diff_title, fontsize=8)
    #ax_diff.set_aspect('auto')
    ax_diff.set_ylim(end, start)
    ax_diff.set_xlim(start, end)
    #ax_diff.tick_params(axis='y', labelsize=8)
    #ax_diff.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x / 1e6:.2f}'))
    
    # Create a colorbar for the difference heatmap
    cax_diff = f.add_subplot(gs[1, 0])
    cbar_diff = plt.colorbar(im_diff, cax=cax_diff, orientation='horizontal')
    #cbar_diff.set_label(diff_title, fontsize=8)
    cbar_diff.ax.tick_params(labelsize=8)
    #cbar_diff.set_label(normalization_method, labelpad=3)
    #cbar_diff.ax.xaxis.set_label_position('top')

    # Compute y_max_list for BigWig tracks to ensure consistent y-axis across samples per track row
    y_min_max_list_bigwig = get_track_min_max(bigwig_files_sample1, bigwig_files_sample2, 'vertical', region)
    
    # Plot BigWig tracks for Sample1 and Sample2
    # Sample1 BigWig
    track_start_row = 2  # Starting from row index 2
    if len(bigwig_files_sample1):
        for i in range(len(bigwig_files_sample1)):
            ax_bw = f.add_subplot(gs[track_start_row + i, 0])
            plot_seq(ax_bw, bigwig_files_sample1[i], region, color=colors_sample1[i], 
                    y_min=y_min_max_list_bigwig[i][0], y_max=y_min_max_list_bigwig[i][1])
            ax_bw.set_title(f"{bigwig_labels_sample1[i]}", fontsize=8)
            ax_bw.set_xlim(start, end)
            ax_bw.set_ylim(y_min_max_list_bigwig[i][0], y_min_max_list_bigwig[i][1] * 1.1)
    track_start_row = 2 + len(bigwig_files_sample1)
    # Plot BigWig for Sample2
    if len(bigwig_files_sample2):
        for j in range(len(bigwig_files_sample2)):
            ax_bw = f.add_subplot(gs[track_start_row + j, 0])
            plot_seq(ax_bw, bigwig_files_sample2[j], region, color=colors_sample2[j], 
            y_min=y_min_max_list_bigwig[j][0], y_max=y_min_max_list_bigwig[j][1])
            ax_bw.set_title(f"{bigwig_labels_sample2[j]}", fontsize=8)
            ax_bw.set_xlim(start, end)
            ax_bw.set_ylim(y_min_max_list_bigwig[j][0], y_min_max_list_bigwig[j][1] * 1.1)
    track_start_row = 2 + len(bigwig_files_sample1) + len(bigwig_files_sample2)
    # Plot BED tracks for Sample1 and Sample2
    # Sample1 BED
    if len(bed_files_sample1):
        for k in range(len(bed_files_sample1)):
            ax_bed = f.add_subplot(gs[track_start_row + k, 0])
            plot_bed(ax_bed, bed_files_sample1[k], region, color=bed_colors_sample1[k], label=bed_labels_sample1[k])
            ax_bed.set_title(f"{bed_labels_sample1[k]}", fontsize=8)
    track_start_row = 2 + len(bigwig_files_sample1) + len(bigwig_files_sample2) + len(bed_files_sample1)
    # Sample2 BED
    if len(bed_files_sample2):
        for l in range(len(bed_files_sample2)):
            ax_bed = f.add_subplot(gs[track_start_row + l, 0])
            plot_bed(ax_bed, bed_files_sample2[l], region, color=bed_colors_sample2[l], label=bed_labels_sample2[l])
            ax_bed.set_title(f"{bed_labels_sample2[l]}", fontsize=8)
    # Plot Genes if GTF file is provided
    if gtf_file:
        gene_row = 2 + max_tracks
        ax_genes = f.add_subplot(gs[gene_row, 0])
        plot_genes(ax_genes, gtf_file, region)
        ax_genes.set_xlim(start, end)
    
    # Adjust layout and save the figure
    plt.tight_layout()
    f.savefig(output_file, bbox_inches='tight')
    plt.close(f)

def main():
    parser = argparse.ArgumentParser(description='Plot difference heatmap from cooler files with BigWig and BED tracks.')
    
    # Required arguments
    parser.add_argument('--cooler_file1', type=str, required=True, help='Path to the case .cool or .mcool file.')
    parser.add_argument('--cooler_file2', type=str, required=False, help='Path to the control .cool or .mcool file.', default=None)
    parser.add_argument('--resolution', type=int, required=True, help='Resolution for the cooler data.')
    parser.add_argument('--start', type=int, required=True, help='Start position for the region of interest.')
    parser.add_argument('--end', type=int, required=True, help='End position for the region of interest.')
    parser.add_argument('--chrid', type=str, required=True, help='Chromosome ID.')
    parser.add_argument('--gtf_file', type=str, required=True, help='Path to the GTF file for gene annotations.')
    
    # Optional arguments
    parser.add_argument('--cmap', type=str, default='autumn_r', help='Colormap to be used for plotting other tracks.')
    parser.add_argument('--vmin', type=float, default=None, help='Minimum value for normalization of other tracks.')
    parser.add_argument('--vmax', type=float, default=None, help='Maximum value for normalization of other tracks.')
    parser.add_argument('--output_file', type=str, default='comparison_heatmap.pdf', help='Filename for the saved comparison heatmap PDF.')
    
    # BigWig arguments
    parser.add_argument('--bigwig_files_sample1', type=str, nargs='*', help='Paths to BigWig files for case sample.', default=[])
    parser.add_argument('--bigwig_labels_sample1', type=str, nargs='*', help='Labels for BigWig tracks of case sample.', default=[])
    parser.add_argument('--colors_sample1', type=str, nargs='+', help='Colors for case BigWig tracks.', default=None)
    parser.add_argument('--bigwig_files_sample2', type=str, nargs='*', help='Paths to BigWig files for control sample.', default=[])
    parser.add_argument('--bigwig_labels_sample2', type=str, nargs='*', help='Labels for BigWig tracks of control sample.', default=[])
    parser.add_argument('--colors_sample2', type=str, nargs='+', help='Colors for control BigWig tracks.', default=None)
    
    # BED arguments
    parser.add_argument('--bed_files_sample1', type=str, nargs='*', help='Paths to BED files for case sample.', default=[])
    parser.add_argument('--bed_labels_sample1', type=str, nargs='*', help='Labels for BED tracks of case sample.', default=[])
    parser.add_argument('--colors_bed_sample1', type=str, nargs='+', help='Colors for case BED tracks.', default=None)
    parser.add_argument('--bed_files_sample2', type=str, nargs='*', help='Paths to BED files for control sample.', default=[])
    parser.add_argument('--bed_labels_sample2', type=str, nargs='*', help='Labels for BED tracks of control sample.', default=[])
    parser.add_argument('--colors_bed_sample2', type=str, nargs='+', help='Colors for control BED tracks.', default=None)
    
    # New Arguments for Division Methods and Color Mapping
    parser.add_argument('--operation', type=str, default='subtract', choices=['subtract', 'divide'],
                        help="Operation to compute the difference matrix: 'subtract' (case - control) or 'divide' (case / control).")
    parser.add_argument('--division_method', type=str, default='raw', choices=['raw', 'log2', 'add1', 'log2_add1'],
                        help="Method for division when '--operation divide' is selected: 'raw' (case/control), 'log2' (log2(case/control)), 'add1' ((case+1)/(control+1)), or 'log2_add1' (log2((case+1)/(control+1))).")
    parser.add_argument('--diff_cmap', type=str, default='bwr', help="Colormap for difference matrix. Default is 'bwr' (Blue-White-Red).")
    parser.add_argument('--diff_title', type=str, default=None, help="titlw for difference matrix")
    # Track dimensions and spacing
    parser.add_argument('--track_size', type=float, default=5, help='Width of each track (in inches).')
    parser.add_argument('--track_spacing', type=float, default=0.5, help='Spacing between tracks (in inches).')
    parser.add_argument("-V", "--version", action="version", version=f"HiCHeatmap {__version__}",
                      help="Print version and exit")
    args = parser.parse_args()

if __name__ == '__main__':
    main()