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

dir = os.path.dirname(os.path.abspath(__file__))
version_py = os.path.join(dir, "_version.py")
exec(open(version_py).read())

def plot_genes(ax, gtf_file, region, color='blue', track_height=1):
    """Plot genes from GTF file on given axis with exons and introns."""
    chrom, start, end = region
    # Load the GTF file using pyranges
    gtf = pr.read_gtf(gtf_file)
    # Filter relevant region and keep only the longest isoform for each gene
    region_genes = gtf[(gtf.df['Chromosome'] == chrom) & (gtf.df['Start'] < end) & (gtf.df['End'] > start)]
    if region_genes.empty:
        print("No genes found in the specified region.")
        return
    # Select the longest isoform for each gene
    longest_isoforms = region_genes.df.loc[region_genes.df.groupby('gene_id')['End'].idxmax()]
    y_offset = 0
    y_step = track_height * 1.2  # Vertical step to add space between stacked genes
    plotted_genes = []
    # Iterate over each gene and plot
    for i, (_, gene) in enumerate(longest_isoforms.iterrows()):
        # Determine y_offset to avoid overlap with previously plotted genes
        for plotted_gene in plotted_genes:
            if not (gene['End'] < plotted_gene['Start'] or gene['Start'] > plotted_gene['End']):
                y_offset = max(y_offset, plotted_gene['y_offset'] + y_step)
        # Plot gene line
        ax.plot([gene['Start'], gene['End']], [y_offset, y_offset], color=color, lw=1)
        # Plot exons as boxes
        exons = region_genes.df[
            (region_genes.df['gene_id'] == gene['gene_id']) & (region_genes.df['Feature'] == 'exon')
        ]
        for _, exon in exons.iterrows():
            ax.add_patch(
                plt.Rectangle(
                    (exon['Start'], y_offset - 0.1 * track_height),
                    exon['End'] - exon['Start'],
                    0.2 * track_height,
                    color=color
                )
            )
        # Add gene name at the center of the gene, adjusted vertically to avoid overlap
        ax.text(
            (gene['Start'] + gene['End']) / 2,
            y_offset + 0.3 * track_height,
            gene['gene_name'],
            fontsize=8,
            ha='center'
        )
        # Track the plotted gene's range and offset
        plotted_genes.append({'Start': gene['Start'], 'End': gene['End'], 'y_offset': y_offset})
        # Increment y_offset for the next gene if needed
        if (i + 1) % 5 == 0:
            y_offset += y_step
    ax.set_ylim(-1, y_offset + 1)
    ax.set_ylabel('Genes')
    ax.set_yticks([])
    ax.set_xlim(start, end)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x / 1e6:.2f}'))

def read_bigwig(file_path, region):
    """Read BigWig or bedGraph file and return positions and values."""
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

def get_track_max(bigwig_files_sample1, bigwig_files_sample2, layoutid,region):
    """Compute the maximum y-value for each BigWig track index across both samples."""
    if layoutid=="horizontal":
        max_num_tracks = max(len(bigwig_files_sample1), len(bigwig_files_sample2))
    elif layoutid=="vertical":
        max_num_tracks = len(bigwig_files_sample1)+len(bigwig_files_sample2)
    else:
        print("bigwig file wrong")
    y_max_list = []
    for i in range(max_num_tracks):
        max_val = 0
        # Sample1
        if i < len(bigwig_files_sample1):
            positions, values = read_bigwig(bigwig_files_sample1[i], region)
            if values is not None:
                track_max = np.nanmax(values)
                max_val = max(max_val, track_max)
        # Sample2
        if i < len(bigwig_files_sample2):
            positions, values = read_bigwig(bigwig_files_sample2[i], region)
            if values is not None:
                track_max = np.nanmax(values)
                max_val = max(max_val, track_max)
        y_max_list.append(max_val)
    return y_max_list

def plot_seq(ax, file_path, region, color='blue', y_max=None):
    """Plot RNA-seq/ChIP-seq expression from BigWig or bedGraph file on given axis."""
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
    if y_max:
        ax.set_ylim(0, y_max)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x / 1e6:.2f}'))

def plot_heatmaps(cooler_file1, sampleid1,
                 bigwig_files_sample1, bigwig_labels_sample1, colors_sample1,
                 gtf_file, resolution=10000,
                 start=10500000, end=13200000, chrid="chr2",
                 cmap='autumn_r', vmin=None, vmax=None,
                 output_file='comparison_heatmap.pdf', layout='horizontal',
                 cooler_file2=None, sampleid2=None,
                 bigwig_files_sample2=[], bigwig_labels_sample2=[], colors_sample2=[],
                 track_width=10, track_height=1, track_spacing=0.5):
    plt.rcParams['font.size'] = 10
    # Define a small height for the colorbar
    small_colorbar_height = 0.1  # Adjust this value to make the colorbar height small
    # Set parameters
    region = (chrid, start, end)
    # Load cooler data
    clr1 = cooler.Cooler(f'{cooler_file1}::resolutions/{resolution}')
    data1 = clr1.matrix(balance=True).fetch(region)
    # Load sample2 data if provided
    single_sample = cooler_file2 is None
    if not single_sample:
        clr2 = cooler.Cooler(f'{cooler_file2}::resolutions/{resolution}')
        data2 = clr2.matrix(balance=True).fetch(region)
    # Define normalization
    norm1 = LogNorm(vmin=vmin, vmax=vmax)
    norm2 = LogNorm(vmin=vmin, vmax=vmax) if not single_sample else None
    # Function to plot triangle heatmaps
    def pcolormesh_triangle(ax, matrix, start=0, resolution=1, norm=None, cmap='autumn_r', *args, **kwargs):
        n = matrix.shape[0]
        # Triangle orientation
        start_pos_vector = [start + resolution * i for i in range(len(matrix) + 1)]
        t = np.array([[1, 0.5], [-1, 0.5]])
        matrix_a = np.dot(
            np.array([(i[1], i[0]) for i in itertools.product(start_pos_vector[::-1], start_pos_vector)]),
            t
        )
        x, y = matrix_a[:, 1].reshape(n + 1, n + 1), matrix_a[:, 0].reshape(n + 1, n + 1)
        im = ax.pcolormesh(x, y, np.flipud(matrix), norm=norm, cmap=cmap, *args, **kwargs)
        ax.yaxis.set_visible(False)
        im.set_rasterized(True)
        return im

    # Formatter for big positions
    bp_formatter = EngFormatter('b')
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
        ncols = 1 if single_sample else 2
        # Calculate the number of BigWig tracks per sample
        num_bigwig_sample1 = len(bigwig_files_sample1)
        num_bigwig_sample2 = len(bigwig_files_sample2) if not single_sample else 0
        num_bigwig_sample1 = num_bigwig_sample1 / ncols
        num_bigwig_sample2 = num_bigwig_sample2 / ncols
        max_num_bigwig_files = int(num_bigwig_sample1 + num_bigwig_sample2)
        # Total rows:
        # Row0: Heatmaps
        # Row1: Colorbars
        # Rows2 to (2 + max_num_bigwig_files -1): BigWig tracks
        # Last row: Genes
        num_rows = 2 + max_num_bigwig_files + 1  # Heatmaps + colorbars + BigWig + Genes
        # Define height ratios: [Heatmaps, Colorbars, BigWig tracks..., Genes]
        height_ratios = [track_height]*1 + [small_colorbar_height]*1 + [track_height]*max_num_bigwig_files + [track_height]
        # Initialize GridSpec
        gs = gridspec.GridSpec(num_rows, ncols, height_ratios=height_ratios, hspace=track_spacing/(track_height))
        # Define default figsize if not provided
        width = track_width * ncols
        height = (track_height * num_rows) + (track_spacing * (num_rows - 1)) + small_colorbar_height
        figsize = (width, height)
        # Initialize figure
        f = plt.figure(figsize=figsize)
        # Plot Heatmaps
        ax_heatmap1 = f.add_subplot(gs[0, 0])
        im1 = pcolormesh_triangle(ax_heatmap1, data1, start=region[1], resolution=resolution, norm=norm1, cmap=cmap)
        ax_heatmap1.set_aspect('auto')
        ax_heatmap1.set_ylim(0, data1.shape[0] * resolution)
        ax_heatmap1.set_xlim(start, end)
        #ax_heatmap1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x / 1e6:.2f}'))
        format_ticks(ax_heatmap1, rotate=False)
        ax_heatmap1.set_title(sampleid1, fontsize=10)
        
        if not single_sample:
            ax_heatmap2 = f.add_subplot(gs[0, 1])
            im2 = pcolormesh_triangle(ax_heatmap2, data2, start=region[1], resolution=resolution, norm=norm2, cmap=cmap)
            ax_heatmap2.set_aspect('auto')
            ax_heatmap2.set_ylim(0, data2.shape[0] * resolution)
            ax_heatmap2.set_xlim(start, end)
            #ax_heatmap2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x / 1e6:.2f}'))
            format_ticks(ax_heatmap2, rotate=False)
            ax_heatmap2.set_title(sampleid2, fontsize=10)
        
        # Create separate colorbars for each heatmap
        if not single_sample:
            # Colorbar for Sample1
            cax1 = f.add_subplot(gs[1, 0])
            sm1 = plt.cm.ScalarMappable(cmap=cmap, norm=norm1)
            sm1.set_array([])
            cbar1 = f.colorbar(sm1, cax=cax1, orientation='horizontal')
            #cbar1.set_label('Interaction Frequency')  # Customize as needed
            # Colorbar for Sample2
            cax2 = f.add_subplot(gs[1, 1])
            sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm2)
            sm2.set_array([])
            cbar2 = f.colorbar(sm2, cax=cax2, orientation='horizontal')
            #cbar2.set_label('Interaction Frequency')  # Customize as needed
        else:
            # Single colorbar if only one sample
            cax = f.add_subplot(gs[1, 0])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm1)
            sm.set_array([])
            cbar = f.colorbar(sm, cax=cax, orientation='horizontal')
            #cbar.set_label('Interaction Frequency')  # Customize as needed
        
        # Compute y_max_list for BigWig tracks to ensure consistent y-axis across samples per track row
        y_max_list = get_track_max(bigwig_files_sample1, bigwig_files_sample2, layout,region)
        
        # Plot BigWig tracks for Sample1
        for i, (bw_file, bw_label, color) in enumerate(zip(bigwig_files_sample1, bigwig_labels_sample1, colors_sample1)):
            ax_bw = f.add_subplot(gs[2 + i, 0])
            plot_seq(ax_bw, bw_file, region, color=color, y_max=y_max_list[i])
            ax_bw.set_title(f"{bw_label} ({sampleid1})", fontsize=10)
            ax_bw.set_xlim(start, end)
        
        # Plot BigWig tracks for Sample2 if provided
        if not single_sample:
            for i, (bw_file, bw_label, color) in enumerate(zip(bigwig_files_sample2, bigwig_labels_sample2, colors_sample2)):
                ax_bw = f.add_subplot(gs[2 + i, 1])
                plot_seq(ax_bw, bw_file, region, color=color, y_max=y_max_list[i])
                ax_bw.set_title(f"{bw_label} ({sampleid2})", fontsize=10)
                ax_bw.set_xlim(start, end)
        
        # Plot Genes
        gene_row = 2 + max_num_bigwig_files
        ax_genes = f.add_subplot(gs[gene_row, 0])
        plot_genes(ax_genes, gtf_file, region, track_height=track_height)
        ax_genes.set_xlim(start, end)
        if not single_sample:
            ax_genes2 = f.add_subplot(gs[gene_row, 1])
            plot_genes(ax_genes2, gtf_file, region, track_height=track_height)
            ax_genes2.set_xlim(start, end)
        
    elif layout == 'vertical':
        # Similar approach but stacking vertically
        # Row0: Heatmap Sample1
        # Row1: Heatmap Sample2
        # Row2: Colorbar spanning all columns
        # Rows3 to N: BigWig tracks
        # Last row: Genes
        num_heatmaps = 1 if single_sample else 2
        num_bigwig_sample1 = len(bigwig_files_sample1)
        num_bigwig_sample2 = len(bigwig_files_sample2) if not single_sample else 0
        max_num_bigwig_files = num_bigwig_sample1 + num_bigwig_sample2
        num_rows = num_heatmaps + 1 + max_num_bigwig_files + 1  # Heatmaps + colorbar + BigWig + Genes
        # Define height ratios: [Heatmaps..., Colorbar, BigWig tracks..., Genes]
        height_ratios = [track_height]*num_heatmaps + [small_colorbar_height] + [track_height]*max_num_bigwig_files + [track_height]
        # Initialize GridSpec
        gs = gridspec.GridSpec(num_rows, 1, height_ratios=height_ratios, hspace=track_spacing/(track_height))
        # Define default figsize if not provided
        width = track_width
        height = (track_height * num_rows) + (track_spacing * (num_rows - 1)) + small_colorbar_height
        figsize = (width, height)
        # Initialize figure
        f = plt.figure(figsize=figsize)
        # Plot Heatmaps
        ax_heatmap1 = f.add_subplot(gs[0, 0])
        im1 = pcolormesh_triangle(ax_heatmap1, data1, start=region[1], resolution=resolution, norm=norm1, cmap=cmap)
        ax_heatmap1.set_aspect('auto')
        ax_heatmap1.set_ylim(0, data1.shape[0] * resolution)
        ax_heatmap1.set_xlim(start, end)
        format_ticks(ax_heatmap1, rotate=False)
        ax_heatmap1.set_title(sampleid1, fontsize=10)
        
        if not single_sample:
            ax_heatmap2 = f.add_subplot(gs[1, 0])
            im2 = pcolormesh_triangle(ax_heatmap2, data2, start=region[1], resolution=resolution, norm=norm2, cmap=cmap)
            ax_heatmap2.set_aspect('auto')
            ax_heatmap2.set_ylim(0, data2.shape[0] * resolution)
            ax_heatmap2.set_xlim(start, end)
            format_ticks(ax_heatmap2, rotate=False)
            ax_heatmap2.set_title(sampleid2, fontsize=10)
        
        # Create separate colorbars for each heatmap
        if not single_sample:
            # Colorbar for Sample1
            cax1 = f.add_subplot(gs[num_heatmaps, 0])
            sm1 = plt.cm.ScalarMappable(cmap=cmap, norm=norm1)
            sm1.set_array([])
            cbar1 = f.colorbar(sm1, cax=cax1, orientation='horizontal')
            #cbar1.set_label('Interaction Frequency')  # Customize as needed
            # Colorbar for Sample2
            #cax2 = f.add_subplot(gs[num_heatmaps + 1, 0])
            #sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm2)
            #sm2.set_array([])
            #cbar2 = f.colorbar(sm2, cax=cax2, orientation='horizontal')
            #cbar2.set_label('Interaction Frequency')  # Customize as needed
        else:
            # Single colorbar if only one sample
            cax = f.add_subplot(gs[num_heatmaps, 0])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm1)
            sm.set_array([])
            cbar = f.colorbar(sm, cax=cax, orientation='horizontal')
            #cbar.set_label('Interaction Frequency')  # Customize as needed
        
        # Compute y_max_list for BigWig tracks to ensure consistent y-axis across samples per track row
        y_max_list = get_track_max(bigwig_files_sample1, bigwig_files_sample2, layout,region)
        
        # Plot BigWig tracks for Sample1
        for i, (bw_file, bw_label, color) in enumerate(zip(bigwig_files_sample1, bigwig_labels_sample1, colors_sample1)):
            ax_bw = f.add_subplot(gs[num_heatmaps + 1 + i, 0])
            plot_seq(ax_bw, bw_file, region, color=color, y_max=y_max_list[i])
            ax_bw.set_title(f"{bw_label} ({sampleid1})", fontsize=10)
            ax_bw.set_xlim(start, end)
        
        # Plot BigWig tracks for Sample2 if provided
        if not single_sample:
            sample2_start_row = num_heatmaps + 1 + len(bigwig_files_sample1)
            for i, (bw_file, bw_label, color) in enumerate(zip(bigwig_files_sample2, bigwig_labels_sample2, colors_sample2)):
                ax_bw = f.add_subplot(gs[sample2_start_row + i, 0])
                plot_seq(ax_bw, bw_file, region, color=color, y_max=y_max_list[i])
                ax_bw.set_title(f"{bw_label} ({sampleid2})", fontsize=10)
                ax_bw.set_xlim(start, end)
        
        # Plot Genes
        gene_row = num_heatmaps + 1 + max_num_bigwig_files
        ax_genes = f.add_subplot(gs[gene_row, 0])
        plot_genes(ax_genes, gtf_file, region, track_height=track_height)
        ax_genes.set_xlim(start, end)
        if not single_sample:
            ax_genes2 = f.add_subplot(gs[gene_row, 0])
            plot_genes(ax_genes2, gtf_file, region, track_height=track_height)
            ax_genes2.set_xlim(start, end)
    else:
        raise ValueError("Invalid layout option. Use 'horizontal' or 'vertical'.")
    
    plt.figtext(0.5, 0.02, "Position (Mb)", ha="center", fontsize=10)
    # Adjust layout using subplots_adjust to prevent overlap
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)
    # Save the figure
    f.savefig(output_file, bbox_inches='tight')
    plt.close(f)

def main():
    parser = argparse.ArgumentParser(description='Plot heatmaps from cooler files.')
    parser.add_argument('--cooler_file1', type=str, required=True, help='Path to the first .mcool file.')
    parser.add_argument('--cooler_file2', type=str, required=False, help='Path to the second .mcool file.', default=None)
    parser.add_argument('--resolution', type=int, default=10000, help='Resolution for the cooler data.')
    parser.add_argument('--start', type=int, default=10500000, help='Start position for the region of interest.')
    parser.add_argument('--end', type=int, default=13200000, help='End position for the region of interest.')
    parser.add_argument('--chrid', type=str, default='chr2', help='Chromosome ID.')
    parser.add_argument('--cmap', type=str, default='autumn_r', help='Colormap to be used for plotting.')
    parser.add_argument('--vmin', type=float, default=None, help='Minimum value for LogNorm scaling.')
    parser.add_argument('--vmax', type=float, default=None, help='Maximum value for LogNorm scaling.')
    parser.add_argument('--output_file', type=str, default='comparison_heatmap.pdf', help='Filename for the saved comparison heatmap PDF.')
    parser.add_argument('--layout', type=str, default='horizontal', choices=['horizontal', 'vertical'], help="Layout of the heatmaps: 'horizontal' or 'vertical'.")
    parser.add_argument('--sampleid1', type=str, default='example1', help='Sample ID for the first dataset.')
    parser.add_argument('--sampleid2', type=str, default=None, help='Sample ID for the second dataset.')
    parser.add_argument('--gtf_file', type=str, required=True, help='Path to the GTF file for gene annotations.')
    parser.add_argument('--bigwig_files_sample1', type=str, nargs='*', help='Paths to BigWig files for sample 1.', default=[])
    parser.add_argument('--bigwig_labels_sample1', type=str, nargs='*', help='Labels for BigWig tracks of sample 1.', default=[])
    parser.add_argument('--colors_sample1', type=str, nargs='+', help='Colors for sample 1 tracks.', default=None)
    parser.add_argument('--bigwig_files_sample2', type=str, nargs='*', help='Paths to BigWig files for sample 2.', default=[])
    parser.add_argument('--bigwig_labels_sample2', type=str, nargs='*', help='Labels for BigWig tracks of sample 2.', default=[])
    parser.add_argument('--colors_sample2', type=str, nargs='+', help='Colors for sample 2 tracks.', default=None)
    parser.add_argument('--track_width', type=float, default=10, help='Width of each track (in inches).')
    parser.add_argument('--track_height', type=float, default=1, help='Height of each track (in inches).')
    parser.add_argument('--track_spacing', type=float, default=0.5, help='Spacing between tracks (in inches).')
    parser.add_argument("-V", "--version", action="version",version="DLR_ICF_comparison {}".format(__version__)\
                      ,help="Print version and exit")
    args = parser.parse_args()

if __name__ == '__main__':
    main()
