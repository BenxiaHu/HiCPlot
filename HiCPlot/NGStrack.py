#!/usr/bin/env python
import argparse
import os
import pandas as pd
from matplotlib.colors import LogNorm
import pyBigWig
import pyranges as pr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

dir = os.path.dirname(os.path.abspath(__file__))
version_py = os.path.join(dir, "_version.py")
exec(open(version_py).read())

def plot_genes(ax, gtf_file, region, color='blue', track_height=1):
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

def get_track_min_max(bigwig_files_sample1, bigwig_files_sample2, layoutid, region):
    if layoutid == "horizontal":
        max_num_tracks = max(len(bigwig_files_sample1), len(bigwig_files_sample2))
    elif layoutid == "vertical":
        max_num_tracks = len(bigwig_files_sample1) + len(bigwig_files_sample2)
    else:
        raise ValueError("Invalid layoutid. Use 'horizontal' or 'vertical'.")

    min_max_list = []

    for i in range(max_num_tracks):
        min_val = np.inf
        max_val = -np.inf

        if layoutid == "horizontal":
            # In horizontal layout, track i corresponds to sample1[i] and sample2[i]
            # Sample 1
            if i < len(bigwig_files_sample1):
                positions, values = read_bigwig(bigwig_files_sample1[i], region)
                if values is not None and len(values) > 0:
                    current_min = np.nanmin(values)
                    current_max = np.nanmax(values)
                    min_val = min(min_val, current_min)
                    max_val = max(max_val, current_max)

            # Sample 2
            if i < len(bigwig_files_sample2):
                positions, values = read_bigwig(bigwig_files_sample2[i], region)
                if values is not None and len(values) > 0:
                    current_min = np.nanmin(values)
                    current_max = np.nanmax(values)
                    min_val = min(min_val, current_min)
                    max_val = max(max_val, current_max)

        elif layoutid == "vertical":
            # In vertical layout, first all sample1 tracks, then sample2 tracks
            if i < len(bigwig_files_sample1):
                # Sample 1 tracks
                positions, values = read_bigwig(bigwig_files_sample1[i], region)
                if values is not None and len(values) > 0:
                    current_min = np.nanmin(values)
                    current_max = np.nanmax(values)
                    min_val = min(min_val, current_min)
                    max_val = max(max_val, current_max)
            else:
                # Sample 2 tracks
                sample2_idx = i - len(bigwig_files_sample1)
                if sample2_idx < len(bigwig_files_sample2):
                    positions, values = read_bigwig(bigwig_files_sample2[sample2_idx], region)
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
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    elif y_max is not None:
        ax.set_ylim(0, y_max)
    elif y_min is not None:
        ax.set_ylim(y_min, 1)  # Default upper limit if only y_min is provided
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x / 1e6:.2f}'))

def plot_bed(ax, bed_file, region, color='green', label=None):
    """Plot BED file annotations on the given axis."""
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
                linewidth=1
            )
        )
    
    ax.set_xlim(start, end)
    ax.set_ylim(0, 1)
    ax.axis('off')  # Hide axis for BED tracks
    if label:
        ax.set_title(label, fontsize=8)

def plot_tracks(
    bigwig_files_sample1, bigwig_labels_sample1, colors_sample1=[],
    bigwig_files_sample2=[], bigwig_labels_sample2=[], colors_sample2=[],
    bed_files_sample1=[], bed_labels_sample1=[], bed_colors_sample1=[],
    bed_files_sample2=[], bed_labels_sample2=[], bed_colors_sample2=[],
    gtf_file=None,
    chrom,start,end,
    vmin=None, vmax=None,
    output_file='comparison_tracks.pdf',
    layout='vertical',
    track_width=10, track_height=1, track_spacing=0.5
):
    plt.rcParams['font.size'] = 8
    track_spacing = track_spacing * 1.2
    single_sample = bigwig_files_sample2 is None

    if layout == 'horizontal':
        num_genes = 1 if gtf_file else 0
        ncols = 1 if single_sample else 2
        # Calculate the maximum number of BigWig and BED tracks per sample
        max_bigwig_sample = max(len(bigwig_files_sample1), len(bigwig_files_sample2))
        max_bed_sample = max(len(bed_files_sample1), len(bed_files_sample2))
        max_bigwig_bed_tracks = max_bigwig_sample + max_bed_sample

        num_rows = max_bigwig_bed_tracks + num_genes
        height_ratios = [track_height] * max_bigwig_bed_tracks + [track_height] * num_genes
        gs = gridspec.GridSpec(num_rows, ncols, height_ratios=height_ratios, hspace=0.5, wspace=0.3)
        width = track_width * ncols
        height = (track_height * num_rows)
        figsize = (width, height)
        f = plt.figure(figsize=figsize)
        
        # Compute y_max_list for BigWig tracks
        y_min_max_list_bigwig = get_track_min_max(bigwig_files_sample1, bigwig_files_sample2, layout, region) if (bigwig_files_sample1 or bigwig_files_sample2) else []

        # Sample1 BigWig
        track_start_row = 0
        for i in range(len(bigwig_files_sample1)):
            ax_bw1 = f.add_subplot(gs[track_start_row, 0])
            plot_seq(ax_bw1, bigwig_files_sample1[i], region, color=colors_sample1[i],
                     y_min=y_min_max_list_bigwig[i][0], y_max=y_min_max_list_bigwig[i][1])
            ax_bw1.set_title(f"{bigwig_labels_sample1[i]}", fontsize=8)
            ax_bw1.set_xlim(start, end)
            ax_bw1.set_ylim(y_min_max_list_bigwig[i][0], y_min_max_list_bigwig[i][1] * 1.1)

        # Sample2 BigWig
        track_start_row = 0
        if len(bigwig_files_sample2):
            for j in range(len(bigwig_files_sample2)):
                ax_bw2 = f.add_subplot(gs[j, 1])
                plot_seq(ax_bw2, bigwig_files_sample2[j], region, color=colors_sample2[j],
                    y_min=y_min_max_list_bigwig[j][0], y_max=y_min_max_list_bigwig[j][1])
                ax_bw2.set_title(f"{bigwig_labels_sample2[j]}", fontsize=8)
                ax_bw2.set_xlim(start, end)
                ax_bw2.set_ylim(y_min_max_list_bigwig[j][0], y_min_max_list_bigwig[j][1] * 1.1)
        # Plot BED tracks
        # Sample1 BED
        track_start_row = 0 + len(bigwig_files_sample1)
        if len(bed_files_sample1):
            for k in range(len(bed_files_sample1)):
                ax_bed = f.add_subplot(gs[track_start_row + k, 0])
                plot_bed(ax_bed, bed_files_sample1[k], region, color=bed_colors_sample1[k], label=bed_labels_sample1[k])
                ax_bed.set_title(f"{bed_labels_sample1[k]}", fontsize=8)
            
        # Sample2 BED
        track_start_row = 0 + len(bigwig_files_sample1)
        if len(bed_files_sample2):
            for l in range(len(bed_files_sample2)):
                ax_bed = f.add_subplot(gs[track_start_row + l, 1])
                plot_bed(ax_bed, bed_files_sample2[l], region, color=bed_colors_sample2[l], label=bed_labels_sample2[l])
                ax_bed.set_title(f"{bed_labels_sample2[l]}", fontsize=8)

        # Plot Genes if GTF file is provided
        if gtf_file:
            gene_row = max_bigwig_bed_tracks
            ax_genes = f.add_subplot(gs[gene_row, 0])
            plot_genes(ax_genes, gtf_file, region, track_height=track_height)
            ax_genes.set_xlim(start, end)
            #ax_genes.set_aspect('auto')  # Default aspect
            if not single_sample:
                ax_genes = f.add_subplot(gs[gene_row, 1])
                plot_genes(ax_genes, gtf_file, region, track_height=track_height)
                ax_genes.set_xlim(start, end)
                #ax_genes.set_aspect('auto')  # Default aspect

    elif layout == 'vertical':
        num_genes = 1 if gtf_file else 0
        ncols = 1
        # Calculate the maximum number of tracks across samples
        max_bigwig_sample = len(bigwig_files_sample1) + len(bigwig_files_sample2)
        max_bed_sample = len(bed_files_sample1) + len(bed_files_sample2)
        max_tracks = max_bigwig_sample + max_bed_sample
        num_rows = max_tracks + num_genes
        height_ratios = [track_height] * max_tracks + [track_height] * num_genes
        gs = gridspec.GridSpec(num_rows, ncols, height_ratios=height_ratios, hspace=track_spacing/(track_height))
        width = track_width * ncols
        height = (track_height * num_rows)
        figsize = (width, height)
        f = plt.figure(figsize=figsize)
        
        y_min_max_list_bigwig = get_track_min_max(bigwig_files_sample1, bigwig_files_sample2, layout, region) if (bigwig_files_sample1 or bigwig_files_sample2) else []
        # Plot BigWig and BED tracks
        # Sample1 BigWig
        track_start_row = 0
        if len(bigwig_files_sample1):
            for i in range(len(bigwig_files_sample1)):
                ax_bw = f.add_subplot(gs[track_start_row + i, 0])
                plot_seq(ax_bw, bigwig_files_sample1[i], region, color=colors_sample1[i], 
                    y_min=y_min_max_list_bigwig[i][0], y_max=y_min_max_list_bigwig[i][1])
                ax_bw.set_title(f"{bigwig_labels_sample1[i]}", fontsize=8)
                ax_bw.set_xlim(start, end)
                ax_bw.set_ylim(y_min_max_list_bigwig[i][0], y_min_max_list_bigwig[i][1] * 1.1)
        track_start_row = 0 + len(bigwig_files_sample1)
        # Plot BigWig files for Sample2
        if len(bigwig_files_sample2):
            for j in range(len(bigwig_files_sample2)):
                ax_bw = f.add_subplot(gs[track_start_row + j, 0])
                plot_seq(ax_bw, bigwig_files_sample2[j], region, color=colors_sample2[j], 
                y_min=y_min_max_list_bigwig[j][0], y_max=y_min_max_list_bigwig[j][1])
                ax_bw.set_title(f"{bigwig_labels_sample2[j]}", fontsize=8)
                ax_bw.set_xlim(start, end)
                ax_bw.set_ylim(y_min_max_list_bigwig[j][0], y_min_max_list_bigwig[j][1] * 1.1)
        
        track_start_row = 0 + len(bigwig_files_sample1) + len(bigwig_files_sample2)
        # Plot BED files for Sample1
        if len(bed_files_sample1):
            for k in range(len(bed_files_sample1)):
                ax_bed = f.add_subplot(gs[track_start_row + k, 0])
                plot_bed(ax_bed, bed_files_sample1[k], region, color=bed_colors_sample1[k], label=bed_labels_sample1[k])
                ax_bed.set_title(f"{bed_labels_sample1[k]}", fontsize=8)
        track_start_row = 0 + len(bigwig_files_sample1) + len(bigwig_files_sample2) + len(bed_files_sample1)
        # Plot BED files for Sample2
        if len(bed_files_sample2):
            for l in range(len(bed_files_sample2)):
                ax_bed = f.add_subplot(gs[track_start_row + l, 0])
                plot_bed(ax_bed, bed_files_sample2[l], region, color=bed_colors_sample2[l], label=bed_labels_sample2[l])
                ax_bed.set_title(f"{bed_labels_sample2[l]}", fontsize=8)
        
        # Plot Genes if GTF file is provided
        if gtf_file:
            gene_row = max_tracks
            ax_genes = f.add_subplot(gs[gene_row, 0])
            plot_genes(ax_genes, gtf_file, region, track_height=track_height)
            ax_genes.set_xlim(start, end)
    else:
        raise ValueError("Invalid layout option. Use 'horizontal' or 'vertical'.")
    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    # Save the figure
    f.savefig(output_file, bbox_inches='tight')
    plt.close(f)

def main():
    parser = argparse.ArgumentParser(description='Plot BigWig, BED, and GTF tracks with customizable layout.')

    # Required BigWig files for Sample1
    parser.add_argument('--bigwig_files_sample1', type=str, nargs='+', required=True, help='Paths to BigWig files for sample 1.')
    parser.add_argument('--bigwig_labels_sample1', type=str, nargs='+', required=True, help='Labels for BigWig tracks of sample 1.')

    # Optional BigWig files for Sample2
    parser.add_argument('--bigwig_files_sample2', type=str, nargs='*', help='Paths to BigWig files for sample 2.', default=[])
    parser.add_argument('--bigwig_labels_sample2', type=str, nargs='*', help='Labels for BigWig tracks of sample 2.', default=[])

    # Optional BED files for Sample1
    parser.add_argument('--bed_files_sample1', type=str, nargs='*', help='Paths to BED files for sample 1.', default=[])
    parser.add_argument('--bed_labels_sample1', type=str, nargs='*', help='Labels for BED tracks of sample 1.', default=[])

    # Optional BED files for Sample2
    parser.add_argument('--bed_files_sample2', type=str, nargs='*', help='Paths to BED files for sample 2.', default=[])
    parser.add_argument('--bed_labels_sample2', type=str, nargs='*', help='Labels for BED tracks of sample 2.', default=[])

    # Optional GTF file for gene annotations
    parser.add_argument('--gtf_file', type=str, required=False, help='Path to the GTF file for gene annotations.', default=None)

    # Genomic region
    parser.add_argument('--start', type=int, required=True, help='Start position for the region of interest.')
    parser.add_argument('--end', type=int,required=True, help='End position for the region of interest.')
    parser.add_argument('--chrid', type=str,required=True, help='Chromosome ID.')

    # Visualization parameters
    parser.add_argument('--vmin', type=float, default=None, help='Minimum value for LogNorm scaling.')
    parser.add_argument('--vmax', type=float, default=None, help='Maximum value for LogNorm scaling.')
    parser.add_argument('--output_file', type=str, default='comparison_tracks.pdf', help='Filename for the saved comparison tracks PDF.')

    # Track dimensions and spacing
    parser.add_argument('--track_width', type=float, default=10, help='Width of each track (in inches).')
    parser.add_argument('--track_height', type=float, default=1, help='Height of each track (in inches).')
    parser.add_argument('--track_spacing', type=float, default=0.5, help='Spacing between tracks (in inches).')

    # Colors for BigWig and BED tracks
    parser.add_argument('--colors_sample1', type=str, nargs='*', help='Colors for sample 1 BigWig tracks.', default=[])
    parser.add_argument('--colors_sample2', type=str, nargs='*', help='Colors for sample 2 BigWig tracks.', default=[])
    parser.add_argument('--colors_bed_sample1', type=str, nargs='*', help='Colors for sample 1 BED tracks.', default=[])
    parser.add_argument('--colors_bed_sample2', type=str, nargs='*', help='Colors for sample 2 BED tracks.', default=[])

    # Layout argument
    parser.add_argument('--layout', type=str, default='vertical', choices=['horizontal', 'vertical'],
                        help="Layout of the tracks: 'horizontal' or 'vertical'.")
    parser.add_argument("-V", "--version", action="version",version="NGStrack {}".format(__version__)\
                      ,help="Print version and exit")
    args = parser.parse_args()

if __name__ == '__main__':
    main()
