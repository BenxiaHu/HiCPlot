import argparse
import os
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pyBigWig
import pyranges as pr
import numpy as np
import matplotlib.pyplot as plt
import cooler
from matplotlib.ticker import EngFormatter
from matplotlib.colors import LogNorm
import itertools
import sys
import scipy.sparse
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Arc
from collections import defaultdict
from matplotlib import rcParams
rcParams['font.family'] = 'DejaVu Sans'

script_dir = os.path.dirname(os.path.abspath(__file__))
version_py = os.path.join(script_dir, "_version.py")
if os.path.exists(version_py):
    with open(version_py) as _vf:
        exec(_vf.read())
else:
    __version__ = "0.0.1" # Fallback if version file is missing

def plot_genes(ax, gtf_file, region, genes_to_annotate=None, color='blue', track_height=1):
    """
    Plot gene annotations on the given axis.
    """
    spacing_factor = 1.5
    chrom, start, end = region
    # Load the GTF file using pyranges
    gtf = pr.read_gtf(gtf_file)
    # Filter relevant region
    region_genes = gtf[(gtf.Chromosome == chrom) & (gtf.Start < end) & (gtf.End > start)]

    if region_genes.empty:
        print("No genes found in the specified region.")
        ax.axis('off')
        return

    # Select the longest isoform for each gene
    longest_isoforms = region_genes.df.loc[region_genes.df.groupby('gene_id')['End'].idxmax()]

    y_offset = 0
    y_step = track_height * spacing_factor
    plotted_genes = []

    for _, gene in longest_isoforms.iterrows():
        for plotted_gene in plotted_genes:
            if not (gene['End'] < plotted_gene['Start'] or gene['Start'] > plotted_gene['End']):
                y_offset = max(y_offset, plotted_gene['y_offset'] + y_step)

        ax.plot([gene['Start'], gene['End']], [y_offset, y_offset], color=color, lw=1)

        exons = region_genes.df[
            (region_genes.df['gene_id'] == gene['gene_id']) & (region_genes.df['Feature'] == 'exon')
        ]
        for _, exon in exons.iterrows():
            ax.add_patch(
                plt.Rectangle(
                    (exon['Start'], y_offset - 0.3 * track_height),
                    exon['End'] - exon['Start'],
                    0.6 * track_height,
                    color=color
                )
            )

        if genes_to_annotate and gene['gene_name'] in genes_to_annotate:
            ax.text(
                (gene['Start'] + gene['End']) / 2,
                y_offset - 0.4 * track_height,
                gene['gene_name'],
                fontsize=8,
                ha='center',
                va='top'
            )

        plotted_genes.append({'Start': gene['Start'], 'End': gene['End'], 'y_offset': y_offset})

    ax.set_ylim(-track_height * 2, y_offset + track_height * 2)
    ax.set_ylabel('Genes')
    ax.set_yticks([])
    ax.set_xlim(start, end)
    ax.set_xlabel("Position (Mb)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x / 1e6:.2f}'))

def read_bigwig(file_path, region):
    """Read BigWig or bedGraph file and return positions and values."""
    chrom, start, end = region
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension in ['.bw', '.bigwig']:
        bw = pyBigWig.open(file_path)
        values = bw.values(chrom, start, end, numpy=True)
        bw.close()
        positions = np.linspace(start, end, len(values))
    elif file_extension in ['.bedgraph', '.bg']:
        bedgraph_df = pd.read_csv(file_path, sep='\t', header=None, comment='#', 
                                  names=['chrom', 'start', 'end', 'value'])
        region_data = bedgraph_df[
            (bedgraph_df['chrom'] == chrom) &
            (bedgraph_df['end'] > start) &
            (bedgraph_df['start'] < end)
        ]
        if region_data.empty:
            return None, None
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
    """Compute the minimum and maximum values for BigWig tracks per type."""
    type_min_max = defaultdict(lambda: {'min': np.inf, 'max': -np.inf})

    def extract_type(label):
        return label.split("_")[1] if label and "_" in label else 'Unknown'

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

    for bw_type in type_min_max:
        if type_min_max[bw_type]['min'] == np.inf and type_min_max[bw_type]['max'] == -np.inf:
            type_min_max[bw_type] = (None, None)
        else:
            type_min_max[bw_type] = (type_min_max[bw_type]['min'], type_min_max[bw_type]['max'])

    return type_min_max

def plot_seq(ax, file_path, region, color='blue', y_min=None, y_max=None):
    """Plot RNA-seq/ChIP-seq expression from BigWig or bedGraph file on given axis."""
    chrom, start, end = region
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension in ['.bw', '.bigwig']:
        bw = pyBigWig.open(file_path)
        values = bw.values(chrom, start, end, numpy=True)
        bw.close()
        positions = np.linspace(start, end, len(values))
    elif file_extension in ['.bedgraph', '.bg']:
        bedgraph_df = pd.read_csv(file_path, sep='\t', header=None, comment='#', 
                                  names=['chrom', 'start', 'end', 'value'])
        region_data = bedgraph_df[
            (bedgraph_df['chrom'] == chrom) &
            (bedgraph_df['end'] > start) &
            (bedgraph_df['start'] < end)
        ]
        if region_data.empty:
            print(f"No data found in the specified region ({chrom}:{start}-{end}) in {file_path}")
            ax.axis('off')
            return
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
    
    ax.plot(positions, values, color=color, alpha=0.7)
    ax.set_xlim(start, end)
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    elif y_max is not None:
        ax.set_ylim(0, y_max)
    elif y_min is not None:
        ax.set_ylim(y_min, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x / 1e6:.2f}'))

def plot_bed(ax, bed_file, region, color='green', linewidth=1):
    """Plot BED file annotations on the given axis."""
    chrom, start, end = region
    bed_df = pd.read_csv(bed_file, sep='\t', header=None, comment='#', 
                         names=['chrom', 'start', 'end'] + [f'col{i}' for i in range(4, 10)])
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
                (bed_start, 0.1),
                bed_end - bed_start,
                0.8,
                color=color,
                linewidth=linewidth
            )
        )
    
    ax.set_xlim(start, end)
    ax.set_ylim(0, 1)
    ax.axis('off')

def pcolormesh_square(ax, matrix, start, end, NORM=True,cmap='autumn_r', vmin=None, vmax=None, *args, **kwargs):
    """Plot the difference matrix as a heatmap on the given axis."""
    if matrix is None:
        return None
    
    if NORM:
        log_vmin = vmin if vmin is not None and vmin > 0 else None
        norm = LogNorm(vmin=log_vmin, vmax=vmax, clip=False)
        im = ax.imshow(matrix, aspect='auto', origin='upper',norm=norm,
                extent=[start, end, end, start], cmap=cmap, *args, **kwargs)
    else:
        im = ax.imshow(matrix, aspect='auto', origin='upper',
                   extent=[start, end, end, start], cmap=cmap, vmin=vmin, vmax=vmax, *args, **kwargs)
    return im

def plot_loops(ax, loop_file, region, color='purple', alpha=0.5, linewidth=1, label=None):
    """Plot chromatin loops as arcs on the given axis."""
    chrom, start, end = region
    loop_df = pd.read_csv(loop_file, sep='\t', header=0, usecols=[0,1,2,3,4,5],
                          names=['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2'])

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

    max_height = 0 

    # Add rectangle background
    ax.add_patch(
        plt.Rectangle(
            (start, 0),
            end - start,
            1.0,
            alpha=1,
            zorder=3,
            edgecolor='black',
            linewidth=1.0,
            facecolor="none",
        )
    )

    for _, loop in loop_df.iterrows():
        a1 = (loop['start1'] + loop['end1']) / 2
        a2 = (loop['start2'] + loop['end2']) / 2
        if a1 == a2:
            continue

        width = abs(a2 - a1)
        height = width / 2
        if height > max_height:
            max_height = height
        mid = (a1 + a2) / 2

        arc = Arc((mid, 0), width=width, height=height*2, angle=0, theta1=0, theta2=180,
                  edgecolor=color, facecolor='none', alpha=alpha, linewidth=linewidth)
        ax.add_patch(arc)

    ax.set_xlim(start, end)
    ax.set_ylim(0, max_height * 1.1)
    ax.axis('off')
    if label:
        ax.set_title(label, fontsize=8)

def plot_heatmaps(cooler_file1=None, sampleid1=None,format="balance",
                 bigwig_files_sample1=[], bigwig_labels_sample1=[], colors_sample1="red",
                 bed_files_sample1=[], bed_labels_sample1=[],
                 loop_file_sample1=None, loop_file_sample2=None,
                 gtf_file=None, resolution=None,
                 start=None, end=None, chrid=None,
                 cmap='autumn_r', vmin=None, vmax=None,
                 track_min=None,track_max=None,
                 output_file='comparison_heatmap.pdf', layout='horizontal',
                 cooler_file2=None, sampleid2=None,
                 bigwig_files_sample2=[], bigwig_labels_sample2=[], colors_sample2="blue",
                 bed_files_sample2=[], bed_labels_sample2=[], 
                 track_size=5, track_spacing=0.5, normalization_method='raw',
                 genes_to_annotate=None):
    plt.rcParams['font.size'] = 8
    
    region = (chrid, start, end)
    has_hic = cooler_file1 is not None
    single_sample = True
    
    normalized_data1 = None
    normalized_data2 = None

    if has_hic:
        # Load cooler data
        clr1 = cooler.Cooler(f'{cooler_file1}::resolutions/{resolution}')
        if format == "balance":
            data1 = clr1.matrix(balance=True).fetch(region).astype(float)
        elif format == "ICE":
            data1 = clr1.matrix(balance=False).fetch(region).astype(float)
        else:
            print("input format is wrong")
            return

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
                return

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

    # Set up the figure based on layout
    if layout == 'horizontal':
        # Even without cooler, we might have two columns if sample 2 tracks/loops exist
        # But if no cooler is provided, usually we assume sample1/sample2 context from other files
        has_sample2_tracks = (len(bigwig_files_sample2) > 0 or len(bed_files_sample2) > 0 or loop_file_sample2 is not None)
        ncols = 2 if (cooler_file2 is not None or has_sample2_tracks) else 1
        
        num_genes = 1 if gtf_file else 0
        max_num_bigwig_files = max(len(bigwig_files_sample1), len(bigwig_files_sample2))
        max_num_bed_files = max(len(bed_files_sample1), len(bed_files_sample2))
        max_bigwig_bed_tracks = (max_num_bigwig_files + max_num_bed_files)
        
        # Calculate Rows
        # Heatmap (1) + Colorbar (1) OR 0 if no hic
        num_heatmap_rows = 2 if has_hic else 0
        
        num_loops = 0
        if loop_file_sample1: num_loops = 1
        if loop_file_sample2: num_loops = 1 # Assuming they share the row if horizontal
        
        num_rows = num_heatmap_rows + num_loops + max_bigwig_bed_tracks + num_genes
        
        # Define height ratios
        small_colorbar_height = 0.1
        loop_track_height = 0.3
        track_height = 0.5
        
        height_ratios = []
        if has_hic:
            height_ratios = [1, small_colorbar_height]
        
        if num_loops > 0:
            height_ratios += [loop_track_height] * num_loops
            
        height_ratios += [track_height] * max_bigwig_bed_tracks
        
        if num_genes != 0:
            height_ratios += [track_height * num_genes]

        # Safety check if height_ratios is empty (no tracks at all)
        if not height_ratios:
             height_ratios = [1]
             num_rows = 1

        # Calculate figure size
        # If no heatmap, we base scale on track_size or arbitrary
        base_unit = 1 if not has_hic else height_ratios[0]
        per_unit = track_size / base_unit
        
        figsize_height = sum(hr * per_unit for hr in height_ratios) + (num_rows -1)*track_spacing
        figsize_width = ncols * track_size + (ncols -1)*track_spacing

        gs = gridspec.GridSpec(num_rows, ncols, height_ratios=height_ratios, hspace=0.3, wspace=0.3)
        f = plt.figure(figsize=(figsize_width, figsize_height))
        
        current_row = 0
        
        # Plot Heatmaps
        if has_hic:
            # Map 1
            ax1 = f.add_subplot(gs[current_row, 0])
            norm_bool = (normalization_method == "logNorm")
            im1 = pcolormesh_square(ax1, normalized_data1, region[1], region[2], cmap=cmap, NORM=norm_bool, vmin=vmin, vmax=vmax)
            format_ticks(ax1, rotate=False)
            ax1.set_title(sampleid1, fontsize=10)
            ax1.set_aspect('equal')
            ax1.set_ylim(end, start)
            ax1.set_xlim(start, end)
            
            # Cbar 1
            cax1 = f.add_subplot(gs[current_row + 1, 0])
            cbar1 = plt.colorbar(im1, cax=cax1, orientation='horizontal')
            cbar1.ax.tick_params(labelsize=8)
            cax1.xaxis.set_label_position('bottom')
            cax1.xaxis.set_ticks_position('bottom')
            cbar1.set_label(normalization_method, labelpad=3)
            cbar1.ax.xaxis.set_label_position('top')
            
            # Map 2
            if ncols > 1 and cooler_file2:
                ax2 = f.add_subplot(gs[current_row, 1])
                im2 = pcolormesh_square(ax2, normalized_data2, region[1], region[2], cmap=cmap, NORM=norm_bool, vmin=vmin, vmax=vmax)
                format_ticks(ax2, rotate=False)
                ax2.set_title(sampleid2, fontsize=10)
                ax2.set_aspect('equal')
                ax2.set_ylim(end, start)
                ax2.set_xlim(start, end)
                
                cax2 = f.add_subplot(gs[current_row + 1, 1])
                cbar2 = plt.colorbar(im2, cax=cax2, orientation='horizontal')
                cbar2.ax.tick_params(labelsize=8)
                cax2.xaxis.set_label_position('bottom')
                cax2.xaxis.set_ticks_position('bottom')
                cbar2.set_label(normalization_method, labelpad=3)
                cbar2.ax.xaxis.set_label_position('top')
            
            current_row += 2

        # Plot Loops
        if loop_file_sample1 or loop_file_sample2:
            loop_start_row = current_row
            if loop_file_sample1:
                ax_loop1 = f.add_subplot(gs[loop_start_row, 0])
                plot_loops(ax_loop1, loop_file_sample1, region, color=colors_sample1, alpha=0.7, linewidth=1, label=f"{sampleid1} Loops")
            
            if loop_file_sample2 and ncols > 1:
                ax_loop2 = f.add_subplot(gs[loop_start_row, 1])
                plot_loops(ax_loop2, loop_file_sample2, region, color=colors_sample2, alpha=0.7, linewidth=1, label=f"{sampleid2} Loops")
            
            current_row += 1

        # Tracks
        track_start_row = current_row
        if track_min is not None and track_max is not None:
            type_min_max = defaultdict(lambda: (track_min, track_max))
        else:
            type_min_max = get_track_min_max(bigwig_files_sample1, bigwig_labels_sample1,
                                        bigwig_files_sample2, bigwig_labels_sample2,
                                        region=region)
        
        # BigWig Sample 1
        for i in range(len(bigwig_files_sample1)):
            ax_bw = f.add_subplot(gs[track_start_row + i, 0])
            bw_type = bigwig_labels_sample1[i].split("_")[1] if "_" in bigwig_labels_sample1[i] else "Unknown"
            y_min, y_max = type_min_max[bw_type]
            plot_seq(ax_bw, bigwig_files_sample1[i], region, color=colors_sample1, y_min=y_min, y_max=y_max)
            ax_bw.set_title(f"{bigwig_labels_sample1[i]}", fontsize=8)
        
        # BigWig Sample 2
        if ncols > 1:
            for j in range(len(bigwig_files_sample2)):
                ax_bw = f.add_subplot(gs[track_start_row + j, 1])
                bw_type = bigwig_labels_sample2[j].split("_")[1] if "_" in bigwig_labels_sample2[j] else "Unknown"
                y_min, y_max = type_min_max[bw_type]
                plot_seq(ax_bw, bigwig_files_sample2[j], region, color=colors_sample2, y_min=y_min, y_max=y_max)
                ax_bw.set_title(f"{bigwig_labels_sample2[j]}", fontsize=8)
        
        bed_start_row = track_start_row + max_num_bigwig_files
        
        # BED Sample 1
        for k in range(len(bed_files_sample1)):
            ax_bed = f.add_subplot(gs[bed_start_row + k, 0])
            plot_bed(ax_bed, bed_files_sample1[k], region, color=colors_sample1, label=bed_labels_sample1[k])
            ax_bed.set_title(f"{bed_labels_sample1[k]}", fontsize=8)

        # BED Sample 2
        if ncols > 1:
            for l in range(len(bed_files_sample2)):
                ax_bed = f.add_subplot(gs[bed_start_row + l, 1])
                plot_bed(ax_bed, bed_files_sample2[l], region, color=colors_sample2, label=bed_labels_sample2[l])
                ax_bed.set_title(f"{bed_labels_sample2[l]}", fontsize=8)
                
        # Genes
        if gtf_file:
            gene_row = bed_start_row + max_num_bed_files
            ax_genes = f.add_subplot(gs[gene_row, 0])
            plot_genes(ax_genes, gtf_file, region, genes_to_annotate=genes_to_annotate, color='blue')
            format_ticks(ax_genes, rotate=False)
            
            if ncols > 1:
                ax_genes2 = f.add_subplot(gs[gene_row, 1])
                plot_genes(ax_genes2, gtf_file, region, genes_to_annotate=genes_to_annotate, color='blue')
                format_ticks(ax_genes2, rotate=False)

    elif layout == 'vertical':
        num_genes = 1 if gtf_file else 0
        
        # Heatmap rows
        if has_hic:
            max_cool_sample = 1 if single_sample else 2
            num_colorbars = 1
        else:
            max_cool_sample = 0
            num_colorbars = 0
            
        max_bigwig_sample = len(bigwig_files_sample1) + len(bigwig_files_sample2)
        max_bed_sample = len(bed_files_sample1) + len(bed_files_sample2)
        max_tracks = max_bigwig_sample + max_bed_sample
        
        num_loops = 0
        if loop_file_sample1: num_loops += 1
        if loop_file_sample2: num_loops += 1
            
        num_rows = max_cool_sample + num_colorbars + num_loops + max_tracks + num_genes
        
        small_colorbar_height = 0.1
        track_height_ratio = 0.5
        loop_track_height = 0.3
        
        height_ratios = []
        if has_hic:
            height_ratios += [1] * max_cool_sample
            height_ratios += [small_colorbar_height]
        
        if num_loops > 0:
            height_ratios += [loop_track_height] * num_loops
            
        height_ratios += [track_height_ratio] * max_tracks
        
        if num_genes > 0:
            height_ratios += [track_height_ratio * num_genes]

        if not height_ratios:
             height_ratios = [1]
             num_rows = 1

        # Figure size
        base_unit = 1 if has_hic else 1 # Scale doesn't heavily depend on map in vertical
        figsize_width = track_size
        figsize_height = sum(height_ratios) * (track_size / (height_ratios[0] if height_ratios else 1)) + (num_rows -1)*track_spacing
        
        gs = gridspec.GridSpec(num_rows, 1, height_ratios=height_ratios, hspace=0.3)
        f = plt.figure(figsize=(figsize_width, figsize_height))
        
        current_row = 0
        
        if has_hic:
            ax_heatmap1 = f.add_subplot(gs[current_row, 0])
            norm_bool = (normalization_method == "logNorm")
            im1 = pcolormesh_square(ax_heatmap1, normalized_data1, region[1], region[2], cmap=cmap, NORM=norm_bool, vmin=vmin, vmax=vmax)
            ax_heatmap1.set_ylim(end, start)
            ax_heatmap1.set_xlim(start, end)
            format_ticks(ax_heatmap1, rotate=False)
            ax_heatmap1.set_title(sampleid1, fontsize=8)
            current_row += 1
            
            if not single_sample:
                ax_heatmap2 = f.add_subplot(gs[current_row, 0])
                im2 = pcolormesh_square(ax_heatmap2, normalized_data2, region[1], region[2], cmap=cmap, NORM=norm_bool, vmin=vmin, vmax=vmax)
                ax_heatmap2.set_ylim(end, start)
                ax_heatmap2.set_xlim(start, end)
                format_ticks(ax_heatmap2, rotate=False)
                ax_heatmap2.set_title(sampleid2, fontsize=10)
                current_row += 1
            
            cax = f.add_subplot(gs[current_row, 0])
            cbar = plt.colorbar(im1, cax=cax, orientation='horizontal')
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label(normalization_method, labelpad=3)
            cbar.ax.xaxis.set_label_position('top')
            current_row += 1
            
        # Loops
        if loop_file_sample1:
            ax_loop1 = f.add_subplot(gs[current_row, 0])
            plot_loops(ax_loop1, loop_file_sample1, region, color=colors_sample1, alpha=0.7, linewidth=1, label=f"{sampleid1} Loops")
            current_row += 1
        if loop_file_sample2:
            ax_loop2 = f.add_subplot(gs[current_row, 0])
            plot_loops(ax_loop2, loop_file_sample2, region, color=colors_sample2, alpha=0.7, linewidth=1, label=f"{sampleid2} Loops")
            current_row += 1
            
        if track_min is not None and track_max is not None:
            type_min_max = defaultdict(lambda: (track_min, track_max))
        else:
            type_min_max = get_track_min_max(bigwig_files_sample1, bigwig_labels_sample1,
                                        bigwig_files_sample2, bigwig_labels_sample2,
                                        region=region)
        
        # BigWig 1
        for i in range(len(bigwig_files_sample1)):
            ax_bw = f.add_subplot(gs[current_row, 0])
            bw_type = bigwig_labels_sample1[i].split("_")[1] if "_" in bigwig_labels_sample1[i] else "Unknown"
            y_min, y_max = type_min_max[bw_type]
            plot_seq(ax_bw, bigwig_files_sample1[i], region, color=colors_sample1, y_min=y_min, y_max=y_max)
            ax_bw.set_title(f"{bigwig_labels_sample1[i]}", fontsize=8)
            current_row += 1

        # BigWig 2
        for j in range(len(bigwig_files_sample2)):
            ax_bw = f.add_subplot(gs[current_row, 0])
            bw_type = bigwig_labels_sample2[j].split("_")[1] if "_" in bigwig_labels_sample2[j] else "Unknown"
            y_min, y_max = type_min_max[bw_type]
            plot_seq(ax_bw, bigwig_files_sample2[j], region, color=colors_sample2, y_min=y_min, y_max=y_max)
            ax_bw.set_title(f"{bigwig_labels_sample2[j]}", fontsize=8)
            current_row += 1

        # BED 1
        for k in range(len(bed_files_sample1)):
            ax_bed = f.add_subplot(gs[current_row, 0])
            plot_bed(ax_bed, bed_files_sample1[k], region, color=colors_sample1, label=bed_labels_sample1[k])
            ax_bed.set_title(f"{bed_labels_sample1[k]}", fontsize=8)
            current_row += 1
            
        # BED 2
        for l in range(len(bed_files_sample2)):
            ax_bed = f.add_subplot(gs[current_row, 0])
            plot_bed(ax_bed, bed_files_sample2[l], region, color=colors_sample2, label=bed_labels_sample2[l])
            ax_bed.set_title(f"{bed_labels_sample2[l]}", fontsize=8)
            current_row += 1
            
        if gtf_file:
            ax_genes = f.add_subplot(gs[current_row, 0])
            plot_genes(ax_genes, gtf_file, region, genes_to_annotate=genes_to_annotate, color='blue')
            ax_genes.set_xlim(start, end)

    else:
        raise ValueError("Invalid layout option. Use 'horizontal' or 'vertical'.")
    
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    f.savefig(output_file, bbox_inches='tight')
    plt.close(f)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description='Plot heatmaps from cooler files.')
    # changed required=True to False
    parser.add_argument('--cooler_file1', type=str, required=False, help='Path to the first .cool or .mcool file.', default=None)
    parser.add_argument('--cooler_file2', type=str, required=False, help='Path to the second .cool or .mcool file.', default=None)
    parser.add_argument('--format', type=str, default='balance', choices=['balance', 'ICE'], help='Format of .mcool file.')

    parser.add_argument('--resolution', type=int, help='Resolution for the cooler data.')
    parser.add_argument('--start', type=int, help='Start position for the region of interest.')
    parser.add_argument('--end', type=int, help='End position for the region of interest.')
    parser.add_argument('--chrid', type=str, help='Chromosome ID.')
    parser.add_argument('--cmap', type=str, default='autumn_r', help='Colormap to be used for plotting.')
    parser.add_argument('--vmin', type=float, default=None, help='Minimum value for Hi-C matrix.')
    parser.add_argument('--vmax', type=float, default=None, help='Maximum value for Hi-C matrix.')
    parser.add_argument('--output_file', type=str, default='comparison_heatmap.pdf', help='Filename for the saved comparison heatmap PDF.')
    parser.add_argument('--layout', type=str, default='horizontal', choices=['horizontal', 'vertical'],
                        help="Layout of the heatmaps: 'horizontal' or 'vertical'.")
    parser.add_argument('--sampleid1', type=str, default='Sample1', help='Sample ID for the first dataset.')
    parser.add_argument('--sampleid2', type=str, default='Sample2', help='Sample ID for the second dataset.')
    
    # BigWig arguments
    parser.add_argument('--bigwig_files_sample1', type=str, nargs='*', help='Paths to BigWig files for sample 1.', default=[])
    parser.add_argument('--bigwig_labels_sample1', type=str, nargs='*', help='Labels for BigWig tracks of sample 1.', default=[])
    parser.add_argument('--colors_sample1', type=str, default="red", help='Colors for sample 1 BigWig tracks.')
    parser.add_argument('--bigwig_files_sample2', type=str, nargs='*', help='Paths to BigWig files for sample 2.', default=[])
    parser.add_argument('--bigwig_labels_sample2', type=str, nargs='*', help='Labels for BigWig tracks of sample 2.', default=[])
    parser.add_argument('--colors_sample2', type=str, default="blue", help='Colors for sample 2 BigWig tracks.')
    
    # BED arguments
    parser.add_argument('--bed_files_sample1', type=str, nargs='*', help='Paths to BED files for sample 1.', default=[])
    parser.add_argument('--bed_labels_sample1', type=str, nargs='*', help='Labels for BED tracks of sample 1.', default=[])
    parser.add_argument('--bed_files_sample2', type=str, nargs='*', help='Paths to BED files for sample 2.', default=[])
    parser.add_argument('--bed_labels_sample2', type=str, nargs='*', help='Labels for BED tracks of sample 2.', default=[])
    
    parser.add_argument('--normalization_method', type=str, default='raw', choices=['raw', 'logNorm','log2', 'log2_add1','log','log_add1'],
                        help="Method for normalization: 'raw', 'logNorm','log2', 'log2_add1', 'log', or 'log_add1'.")
    
    parser.add_argument('--track_size', type=float, default=5, help='Width of each track (in inches).')
    parser.add_argument('--track_spacing', type=float, default=0.5, help='Spacing between tracks (in inches).')

    # Loop file arguments
    parser.add_argument('--loop_file_sample1', type=str, help='Path to the chromatin loop file for sample 1.', default=None)
    parser.add_argument('--loop_file_sample2', type=str, help='Path to the chromatin loop file for sample 2.', default=None)
    # Gene annotation arguments
    parser.add_argument('--gtf_file', type=str, required=False, help='Path to the GTF file for gene annotations.', default=None)
    parser.add_argument('--genes_to_annotate', type=str, nargs='*', help='Gene names to annotate.', default=None)
    parser.add_argument("-V", "--version", action="version",version="SquHeatmap {}".format(__version__)\
                      ,help="Print version and exit")
    parser.add_argument('--track_min', type=float, default=None, help='Global minimum value for all BigWig tracks.')
    parser.add_argument('--track_max', type=float, default=None, help='Global maximum value for all BigWig tracks.')
    
    args = parser.parse_args(argv)

    # Added validation check
    if args.cooler_file1 is None and (args.start is None or args.end is None or args.chrid is None):
        parser.error("If --cooler_file1 is not provided, --start, --end, and --chrid must be specified to define the region.")

    plot_heatmaps(
        cooler_file1=args.cooler_file1,
        sampleid1=args.sampleid1,
        bigwig_files_sample1=args.bigwig_files_sample1,
        bigwig_labels_sample1=args.bigwig_labels_sample1,
        colors_sample1=args.colors_sample1,
        colors_sample2=args.colors_sample2,
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
        layout=args.layout,
        cooler_file2=args.cooler_file2,
        sampleid2=args.sampleid2,
        bigwig_files_sample2=args.bigwig_files_sample2,
        bigwig_labels_sample2=args.bigwig_labels_sample2,
        bed_files_sample2=args.bed_files_sample2,
        bed_labels_sample2=args.bed_labels_sample2,
        track_size=args.track_size,
        track_spacing=args.track_spacing,
        normalization_method=args.normalization_method,
        genes_to_annotate=args.genes_to_annotate,
        format=args.format
    )
if __name__ == '__main__':
    main()