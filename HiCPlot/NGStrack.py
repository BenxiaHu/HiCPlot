import argparse
import os
import pandas as pd
from matplotlib.colors import LogNorm
import pyBigWig
import pyranges as pr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

dir = os.path.dirname(os.path.abspath(__file__))
version_py = os.path.join(dir, "_version.py")
exec(open(version_py).read())

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
    - layoutid: Layout type ('horizontal' or 'vertical').
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

def plot_genes(ax, gtf_file, region, genes=None, color='blue', track_height=1):
    """
    Plot gene lines for all genes in the region.
    Annotate only the specified genes with their names.
    
    Parameters:
    - ax: Matplotlib axis to plot on.
    - gtf_file: Path to the GTF file.
    - region: Tuple (chromosome, start, end).
    - genes: List of gene names to annotate. If None, no annotations.
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
        if genes and gene['gene_name'] in genes:
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
    # Expanded lower limit to accommodate gene names below the lines
    ax.set_ylim(-track_height * 2, y_offset + track_height * 2)
    ax.set_ylabel('Genes')
    ax.set_yticks([])  # Hide y-ticks for a cleaner look
    ax.set_xlim(start, end)
    ax.set_xlabel("Position (Mb)")
    
    # Format x-axis to display positions in megabases (Mb)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x / 1e6:.2f}'))

def read_bigwig(file_path, region):
    """Read BigWig or bedGraph file and return positions and values."""
    chrid, start, end = region
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension in ['.bw', '.bigwig']:
        # Open the BigWig file
        bw = pyBigWig.open(file_path)
        # Fetch values from the region
        values = bw.values(chrid, start, end, numpy=True)
        bw.close()  # Close the BigWig file
        positions = np.linspace(start, end, len(values))
    elif file_extension in ['.bedgraph', '.bg']:
        # Read the bedGraph file using pandas
        # Assuming bedGraph files have columns: chrid, start, end, value
        bedgraph_df = pd.read_csv(file_path, sep='\t', header=None, comment='#', 
                                  names=['chrom', 'start', 'end', 'value'])
        # Filter the data for the specified region
        region_data = bedgraph_df[
            (bedgraph_df['chrom'] == chrid) &
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
    ax.plot(positions, values, color=color, alpha=0.7)
    ax.set_xlim(start, end)
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    elif y_max is not None:
        ax.set_ylim(0, y_max)
    elif y_min is not None:
        ax.set_ylim(y_min, 1)  # Default upper limit if only y_min is provided
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x / 1e6:.2f}'))

def plot_tracks(
    bigwig_files_sample1, bigwig_labels_sample1, 
    bigwig_files_sample2=[], bigwig_labels_sample2=[], 
    colors_sample1="red",colors_sample2="blue",
    bed_files_sample1=[], bed_labels_sample1=[],
    bed_files_sample2=[], bed_labels_sample2=[],
    gtf_file=None,
    genes_to_annotate=None,
    chrid=None, start=None, end=None,
    track_min=None, track_max=None,
    output_file='comparison_tracks.pdf',
    layout='vertical',
    track_width=10, track_height=1, track_spacing=0.5
):
    """
    Plot BigWig, BED, and GTF tracks with customizable layout.

    Parameters:
    - bigwig_files_sample1: List of BigWig file paths for sample 1.
    - bigwig_labels_sample1: List of labels for BigWig tracks of sample 1.
    - colors_sample1: List of colors for BigWig tracks of sample 1.
    - bigwig_files_sample2: List of BigWig file paths for sample 2.
    - bigwig_labels_sample2: List of labels for BigWig tracks of sample 2.
    - colors_sample2: List of colors for BigWig tracks of sample 2.
    - bed_files_sample1: List of BED file paths for sample 1.
    - bed_labels_sample1: List of labels for BED tracks of sample 1.
    - bed_files_sample2: List of BED file paths for sample 2.
    - bed_labels_sample2: List of labels for BED tracks of sample 2.
    - gtf_file: Path to the GTF file for gene annotations.
    - genes_to_annotate: List of gene names to annotate.
    - chrid: Chromosome ID.
    - start: Start position of the region.
    - end: End position of the region.
    - track_min: Minimum value for scaling.
    - track_max: Maximum value for scaling.
    - output_file: Filename for the saved PDF.
    - layout: 'horizontal' or 'vertical'.
    - track_width: Width of each track in inches.
    - track_height: Height of each track in inches.
    - track_spacing: Spacing between tracks in inches.
    """
    plt.rcParams['font.size'] = 8
    track_spacing = track_spacing * 1.2
    single_sample = len(bigwig_files_sample2) == 0
    region = (chrid, start, end)
    if layout == 'horizontal':
        num_genes = 1 if gtf_file else 0
        ncols = 1 if single_sample else 2
        # Calculate the maximum number of BigWig and BED tracks per sample
        max_bigwig_sample = max(len(bigwig_files_sample1), len(bigwig_files_sample2)) if not single_sample else len(bigwig_files_sample1)
        max_bed_sample = max(len(bed_files_sample1), len(bed_files_sample2)) if not single_sample else len(bed_files_sample1)
        max_bigwig_bed_tracks = max_bigwig_sample + max_bed_sample

        num_rows = max_bigwig_bed_tracks + num_genes
        if num_genes !=0:
            height_ratios = [track_height] * max_bigwig_bed_tracks + [track_height] * num_genes
        else:
            height_ratios = [track_height] * max_bigwig_bed_tracks
        gs = gridspec.GridSpec(num_rows, ncols, height_ratios=height_ratios, hspace=0.5, wspace=0.3)
        width = track_width * ncols
        height = (track_height * num_rows)
        figsize = (width, height)
        f = plt.figure(figsize=figsize)

        # Compute global min and max per BigWig type
        if track_min is not None and track_max is not None:
            type_min_max = defaultdict(lambda: (track_min, track_max))
        else:
            type_min_max = get_track_min_max(bigwig_files_sample1, bigwig_labels_sample1,
                                        bigwig_files_sample2, bigwig_labels_sample2,
                                        region=region)

        # Sample1 BigWig
        track_start_row = 0
        if bigwig_files_sample2:
            for i in range(len(bigwig_files_sample1)):
                ax_bw1 = f.add_subplot(gs[track_start_row + i, 0])
                bw_type = bigwig_labels_sample1[i].split("_")[1]
                y_min, y_max = type_min_max[bw_type]
                plot_seq(ax_bw1, bigwig_files_sample1[i], (chrid, start, end), color=colors_sample1,
                     y_min=y_min, y_max=y_max)
                ax_bw1.set_title(f"{bigwig_labels_sample1[i]}", fontsize=8)
                ax_bw1.set_xlim(start, end)
                if y_min is not None and y_max is not None:
                    ax_bw1.set_ylim(y_min, y_max * 1.1)

        # Sample2 BigWig
        track_start_row = 0
        if bigwig_files_sample2:
            for j in range(len(bigwig_files_sample2)):
                ax_bw2 = f.add_subplot(gs[track_start_row + j, 1])
                bw_type = bigwig_labels_sample2[j].split("_")[1]
                y_min, y_max = type_min_max[bw_type]
                plot_seq(ax_bw2, bigwig_files_sample2[j], (chrid, start, end), color=colors_sample2,
                     y_min=y_min, y_max=y_max)
                ax_bw2.set_title(f"{bigwig_labels_sample2[j]}", fontsize=8)
                ax_bw2.set_xlim(start, end)
                if y_min is not None and y_max is not None:
                    ax_bw2.set_ylim(y_min, y_max * 1.1)
        
        # Plot BED tracks
        # Sample1 BED
        track_start_row = len(bigwig_files_sample1)
        if bed_files_sample1:
            for k in range(len(bed_files_sample1)):
                ax_bed = f.add_subplot(gs[track_start_row + k, 0])
                label = bed_labels_sample1[k]
                plot_bed(ax_bed, bed_files_sample1[k], (chrid, start, end), color=colors_sample1, label=label)
                ax_bed.set_title(f"{bed_labels_sample1[k]}", fontsize=8)
            
        # Sample2 BED
        if bed_files_sample2:
            track_start_row = len(bigwig_files_sample1)
            for l in range(len(bed_files_sample2)):
                ax_bed = f.add_subplot(gs[track_start_row + l, 1])
                label = bed_labels_sample2[l] if l < len(bed_labels_sample2) else None
                plot_bed(ax_bed, bed_files_sample2[l], (chrid, start, end), color=colors_sample2, label=label)
                ax_bed.set_title(f"{bed_labels_sample2[l]}", fontsize=8)

        # Plot Genes if GTF file is provided
        if gtf_file:
            gene_row = max_bigwig_bed_tracks
            ax_genes = f.add_subplot(gs[gene_row, 0])
            plot_genes(ax_genes, gtf_file, (chrid, start, end), genes=genes_to_annotate, track_height=track_height)
            ax_genes.set_xlim(start, end)
            if not single_sample:
                ax_genes = f.add_subplot(gs[gene_row, 1])
                plot_genes(ax_genes, gtf_file, (chrid, start, end), genes=genes_to_annotate, track_height=track_height)
                ax_genes.set_xlim(start, end)

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
        
        # Compute global min and max per BigWig type
        if track_min is not None and track_max is not None:
            type_min_max = defaultdict(lambda: (track_min, track_max))
        else:
            type_min_max = get_track_min_max(bigwig_files_sample1, bigwig_labels_sample1,
                                        bigwig_files_sample2, bigwig_labels_sample2,
                                        region=region)
        # Plot BigWig and BED tracks
        # Sample1 BigWig
        track_start_row = 0
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

        track_start_row = len(bigwig_files_sample1)
        # Plot BigWig files for Sample2
        if bigwig_files_sample2:
            for j in range(len(bigwig_files_sample2)):
                ax_bw = f.add_subplot(gs[track_start_row + j, 0])
                bw_type = bigwig_labels_sample2[j].split("_")[1]
                y_min, y_max = type_min_max[bw_type]
                plot_seq(ax_bw, bigwig_files_sample2[j], region, color=colors_sample2, 
                    y_min=y_min, y_max=y_max)
                ax_bw.set_title(f"{bigwig_labels_sample2[j]}", fontsize=8)
                ax_bw.set_xlim(start, end)
                if y_min is not None and y_max is not None:
                    ax_bw.set_ylim(y_min, y_max * 1.1)
        
        track_start_row = len(bigwig_files_sample1) + len(bigwig_files_sample2)
        # Plot BED files for Sample1
        if bed_files_sample1:
            for k in range(len(bed_files_sample1)):
                ax_bed = f.add_subplot(gs[track_start_row + k, 0])
                label = bed_labels_sample1[k] if k < len(bed_labels_sample1) else None
                plot_bed(ax_bed, bed_files_sample1[k], (chrid, start, end), color=colors_sample1, label=label)
                ax_bed.set_title(f"{bed_labels_sample1[k]}", fontsize=8)
        track_start_row = len(bigwig_files_sample1) + len(bigwig_files_sample2) + len(bed_files_sample1)
        # Plot BED files for Sample2
        if bed_files_sample2:
            for l in range(len(bed_files_sample2)):
                ax_bed = f.add_subplot(gs[track_start_row + l, 0])
                label = bed_labels_sample2[l] if l < len(bed_labels_sample2) else None
                plot_bed(ax_bed, bed_files_sample2[l], (chrid, start, end), color=colors_sample2, label=label)
                ax_bed.set_title(f"{bed_labels_sample2[l]}", fontsize=8)
        
        # Plot Genes if GTF file is provided
        if gtf_file:
            gene_row = max_tracks
            ax_genes = f.add_subplot(gs[gene_row, 0])
            plot_genes(ax_genes, gtf_file, (chrid, start, end), genes=genes_to_annotate, track_height=track_height)
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

    # Optional Gene names
    parser.add_argument('--genes_to_annotate', type=str, nargs='*', help='Gene names to display.', default=None)

    # Genomic region
    parser.add_argument('--start', type=int, required=True, help='Start position for the region of interest.')
    parser.add_argument('--end', type=int, required=True, help='End position for the region of interest.')
    parser.add_argument('--chrid', type=str, required=True, help='Chromosome ID.')

    # Visualization parameters
    parser.add_argument('--track_min', type=float, default=None, help='Global minimum value for all BigWig tracks.')
    parser.add_argument('--track_max', type=float, default=None, help='Global maximum value for all BigWig tracks.')
    parser.add_argument('--output_file', type=str, default='comparison_tracks.pdf', help='Filename for the saved comparison tracks PDF.')

    # Track dimensions and spacing
    parser.add_argument('--track_width', type=float, default=10, help='Width of each track (in inches).')
    parser.add_argument('--track_height', type=float, default=1, help='Height of each BigWig/BED track (in inches).')
    parser.add_argument('--track_spacing', type=float, default=0.5, help='Spacing between tracks (in inches).')

    # Colors for BigWig and BED tracks
    parser.add_argument('--colors_sample1', type=str, default="red", help='Colors for sample 1 BigWig tracks.')
    parser.add_argument('--colors_sample2', type=str, default="blue", help='Colors for sample 2 BigWig tracks.')

    # Layout argument
    parser.add_argument('--layout', type=str, default='vertical', choices=['horizontal', 'vertical'],
                        help="Layout of the tracks: 'horizontal' or 'vertical'.")
    parser.add_argument("-V", "--version", action="version",version="NGStrack {}".format(__version__)\
                      ,help="Print version and exit")
    args = parser.parse_args()

    # Call plot_tracks with the parsed arguments
    plot_tracks(
        bigwig_files_sample1=args.bigwig_files_sample1,
        bigwig_labels_sample1=args.bigwig_labels_sample1,
        colors_sample1=args.colors_sample1,
        bigwig_files_sample2=args.bigwig_files_sample2,
        bigwig_labels_sample2=args.bigwig_labels_sample2,
        colors_sample2=args.colors_sample2,
        bed_files_sample1=args.bed_files_sample1,
        bed_labels_sample1=args.bed_labels_sample1,
        bed_files_sample2=args.bed_files_sample2,
        bed_labels_sample2=args.bed_labels_sample2,
        gtf_file=args.gtf_file,
        genes_to_annotate=args.genes_to_annotate,
        chrid=args.chrid,
        start=args.start,
        end=args.end,
        track_min=args.track_min,
        track_max=args.track_max,
        output_file=args.output_file,
        layout=args.layout,
        track_width=args.track_width,
        track_height=args.track_height,
        track_spacing=args.track_spacing
    )

if __name__ == '__main__':
    main()
