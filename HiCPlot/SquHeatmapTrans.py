import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cooler
import matplotlib.gridspec as gridspec
from matplotlib.ticker import EngFormatter
from matplotlib.colors import LogNorm
from collections import defaultdict

script_dir = os.path.dirname(os.path.abspath(__file__))
version_py = os.path.join(script_dir, "_version.py")
with open(version_py) as _vf:
    exec(_vf.read())

def pcolormesh_square(ax, matrix, start, end, NORM=True, cmap='autumn_r', vmin=None, vmax=None, *args, **kwargs):
    """
    Plot the matrix as a heatmap on the given axis.
    """
    if matrix is None or matrix.size == 0:
        return None

    # Handle NaN values by masking
    masked_matrix = np.ma.masked_invalid(matrix)

    if NORM:
        log_vmin = vmin if vmin is not None and vmin > 0 else None
        norm = LogNorm(vmin=log_vmin, vmax=vmax, clip=False)
        im = ax.imshow(masked_matrix, aspect='auto', origin='upper', norm=norm,
                extent=[start, end, end, start], cmap=cmap, *args, **kwargs)
    else:
        im = ax.imshow(masked_matrix, aspect='auto', origin='upper',
                   extent=[start, end, end, start], cmap=cmap, vmin=vmin, vmax=vmax, *args, **kwargs)
    return im


def plot_heatmaps(cooler_file1, sampleid1=None, format="balance",
                 resolution=None,
                 start1=None, end1=None, chrid1=None,
                 start2=None, end2=None, chrid2=None,
                 cmap='autumn_r', vmin=None, vmax=None,
                 track_min=None, track_max=None,
                 output_file='comparison_heatmap.pdf', layout='horizontal',
                 cooler_file2=None, sampleid2=None,
                 track_size=5, track_spacing=0.5, normalization_method='raw'):

    plt.rcParams['font.size'] = 8

    # Define regions
    region1 = (chrid1, start1, end1)
    single_region_mode = (chrid2 is None and start2 is None and end2 is None) or \
                         (chrid1 == chrid2 and start1 == start2 and end1 == end2)

    if single_region_mode:
        regions = [region1]
    else:
        region2 = (chrid2, start2, end2)
        regions = [region1, region2]

    num_regions = len(regions)
    single_sample = cooler_file2 is None
    num_samples = 1 if single_sample else 2

    # Load cooler data for all regions and samples
    clr1 = cooler.Cooler(f'{cooler_file1}::resolutions/{resolution}')
    data1 = {}
    data2 = {} # Only used if not single_sample

    for i, region in enumerate(regions):
        r_key = f'region{i+1}'
        chrom, start, end = region
        try:
            if format == "balance":
                data1[r_key] = clr1.matrix(balance=True).fetch(chrom, start, end).astype(float)
            elif format == "ICE":
                data1[r_key] = clr1.matrix(balance=False).fetch(chrom, start, end).astype(float)
            else:
                 print("Input format is wrong")
                 return # Exit if format is wrong

            if not single_sample:
                clr2 = cooler.Cooler(f'{cooler_file2}::resolutions/{resolution}')
                if format == "balance":
                    data2[r_key] = clr2.matrix(balance=True).fetch(chrom, start, end).astype(float)
                elif format == "ICE":
                    data2[r_key] = clr2.matrix(balance=False).fetch(chrom, start, end).astype(float)
                else:
                    print("Input format is wrong")
                    return # Exit if format is wrong
        except Exception as e:
            print(f"Error fetching cooler data for {chrom}:{start}-{end}: {e}")
            data1[r_key] = None
            if not single_sample:
                data2[r_key] = None

    # Apply normalization
    normalized_data1 = {}
    normalized_data2 = {} if not single_sample else None

    for r_key in data1:
        if data1[r_key] is not None:
            if normalization_method == 'raw':
                normalized_data1[r_key] = data1[r_key]
            elif normalization_method == 'logNorm':
                 normalized_data1[r_key] = np.maximum(data1[r_key], 0) # Handle negative values before log
            elif normalization_method == 'log2':
                 # Avoid log2(0) which is -inf, replace with NaN or a small value
                 normalized_data1[r_key] = np.log2(data1[r_key], where=data1[r_key]>0, out=np.nan*data1[r_key])
            elif normalization_method == 'log2_add1':
                normalized_data1[r_key] = np.log2(data1[r_key] + 1)
            elif normalization_method == 'log':
                # Avoid log(0) which is -inf, replace with NaN or a small value
                normalized_data1[r_key] = np.log(data1[r_key], where=data1[r_key]>0, out=np.nan*data1[r_key])
            elif normalization_method == 'log_add1':
                normalized_data1[r_key] = np.log(data1[r_key] + 1)
            else:
                raise ValueError(f"Unsupported normalization method: {normalization_method}")

        if not single_sample and data2 and data2[r_key] is not None:
             if normalization_method == 'raw':
                 normalized_data2[r_key] = data2[r_key]
             elif normalization_method == 'logNorm':
                  normalized_data2[r_key] = np.maximum(data2[r_key], 0) # Handle negative values before log
             elif normalization_method == 'log2':
                  # Avoid log2(0) which is -inf, replace with NaN or a small value
                 normalized_data2[r_key] = np.log2(data2[r_key], where=data2[r_key]>0, out=np.nan*data2[r_key])
             elif normalization_method == 'log2_add1':
                 normalized_data2[r_key] = np.log2(data2[r_key] + 1)
             elif normalization_method == 'log':
                 # Avoid log(0) which is -inf, replace with NaN or a small value
                 normalized_data2[r_key] = np.log(data2[r_key], where=data2[r_key]>0, out=np.nan*data2[r_key])
             elif normalization_method == 'log_add1':
                 normalized_data2[r_key] = np.log(data2[r_key] + 1)
             else:
                 raise ValueError(f"Unsupported normalization method: {normalization_method}")

    # Determine global vmin and vmax across all heatmaps if not provided
    if vmin is None or vmax is None:
        all_heatmap_data = []
        for r_key in normalized_data1:
            if normalized_data1[r_key] is not None:
                all_heatmap_data.append(normalized_data1[r_key].flatten())
            if not single_sample and normalized_data2 and normalized_data2[r_key] is not None:
                all_heatmap_data.append(normalized_data2[r_key].flatten())

        if all_heatmap_data:
            combined_data = np.concatenate(all_heatmap_data)
            # Filter out NaN, Inf, and -Inf before calculating min/max
            finite_data = combined_data[np.isfinite(combined_data)]

            if finite_data.size > 0:
                if vmin is None:
                     vmin = np.nanmin(finite_data) if normalization_method.startswith('log') else 0
                if vmax is None:
                     vmax = np.nanmax(finite_data)
            else:
                 vmin, vmax = 0, 1 # Default if no valid data

        else:
             vmin, vmax = 0, 1 # Default if no heatmap data at all

    # Helper function to format tick labels
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

    # Set up the figure
    if layout == 'vertical':
        ncols = num_samples
        
        # Calculate figure size
        figsize_width = num_samples * track_size + (num_samples - 1) * track_spacing
        figsize_height = num_regions * track_size * 1.5  # Simple estimation for figure height
        
        # Create figure and gridspec for heatmaps
        fig = plt.figure(figsize=(figsize_width, figsize_height))
        gs = gridspec.GridSpec(num_regions, ncols, hspace=0.3, wspace=0.3)
        
        for i, region in enumerate(regions):
            r_key = f'region{i+1}'
            chrom, start, end = region
            region_title = f'{chrom}:{start}-{end}'
            
            # Plot heatmap for the first sample
            ax_heatmap1 = fig.add_subplot(gs[i, 0])
            if normalized_data1.get(r_key) is not None:
                if normalization_method == "logNorm":
                    im1 = pcolormesh_square(ax_heatmap1, normalized_data1.get(r_key), start, end, 
                                           cmap=cmap, NORM=True, vmin=vmin, vmax=vmax)
                else:
                    im1 = pcolormesh_square(ax_heatmap1, normalized_data1.get(r_key), start, end, 
                                           cmap=cmap, NORM=False, vmin=vmin, vmax=vmax)
                format_ticks(ax_heatmap1, rotate=False)
                ax_heatmap1.set_title(f"{sampleid1} - {region_title}", fontsize=10)
                ax_heatmap1.set_aspect('equal')
                ax_heatmap1.set_ylim(end, start)
                ax_heatmap1.set_xlim(start, end)
                
                # Add colorbar
                divider = make_axes_locatable(ax_heatmap1)
                cax1 = divider.append_axes("bottom", size="5%", pad=0.1)
                cbar1 = plt.colorbar(im1, cax=cax1, orientation='horizontal')
                cbar1.ax.tick_params(labelsize=8)
                cbar1.set_label(normalization_method, labelpad=3)
            else:
                ax_heatmap1.set_title(f"{sampleid1} - {region_title} (No Data)", fontsize=10)
                ax_heatmap1.axis('off')
                
            # Plot heatmap for the second sample if applicable
            if not single_sample:
                ax_heatmap2 = fig.add_subplot(gs[i, 1]) 
                if normalized_data2 and normalized_data2.get(r_key) is not None:
                    if normalization_method == "logNorm":
                        im2 = pcolormesh_square(ax_heatmap2, normalized_data2.get(r_key), start, end, 
                                               cmap=cmap, NORM=True, vmin=vmin, vmax=vmax)
                    else:
                        im2 = pcolormesh_square(ax_heatmap2, normalized_data2.get(r_key), start, end, 
                                               cmap=cmap, NORM=False, vmin=vmin, vmax=vmax)
                    format_ticks(ax_heatmap2, rotate=False)
                    ax_heatmap2.set_title(f"{sampleid2} - {region_title}", fontsize=10)
                    ax_heatmap2.set_aspect('equal')
                    ax_heatmap2.set_ylim(end, start)
                    ax_heatmap2.set_xlim(start, end)
                    
                    # Add colorbar
                    divider = make_axes_locatable(ax_heatmap2)
                    cax2 = divider.append_axes("bottom", size="5%", pad=0.1)
                    cbar2 = plt.colorbar(im2, cax=cax2, orientation='horizontal')
                    cbar2.ax.tick_params(labelsize=8)
                    cbar2.set_label(normalization_method, labelpad=3)
                else:
                    ax_heatmap2.set_title(f"{sampleid2} - {region_title} (No Data)", fontsize=10)
                    ax_heatmap2.axis('off')
    else:
        # Simple horizontal layout for single region
        fig, axes = plt.subplots(1, num_samples, figsize=(track_size * num_samples, track_size))
        
        r_key = 'region1'
        chrom, start, end = regions[0]
        region_title = f'{chrom}:{start}-{end}'
        
        # Ensure axes is always a list-like object
        if num_samples == 1:
            axes = [axes]
            
        # Plot first sample
        if normalized_data1.get(r_key) is not None:
            if normalization_method == "logNorm":
                im1 = pcolormesh_square(axes[0], normalized_data1.get(r_key), start, end, 
                                       cmap=cmap, NORM=True, vmin=vmin, vmax=vmax)
            else:
                im1 = pcolormesh_square(axes[0], normalized_data1.get(r_key), start, end, 
                                       cmap=cmap, NORM=False, vmin=vmin, vmax=vmax)
            format_ticks(axes[0], rotate=False)
            axes[0].set_title(f"{sampleid1} - {region_title}", fontsize=10)
            axes[0].set_aspect('equal')
            axes[0].set_ylim(end, start)
            axes[0].set_xlim(start, end)
            
            # Add colorbar
            divider = make_axes_locatable(axes[0])
            cax1 = divider.append_axes("bottom", size="5%", pad=0.1)
            cbar1 = plt.colorbar(im1, cax=cax1, orientation='horizontal')
            cbar1.ax.tick_params(labelsize=8)
            cbar1.set_label(normalization_method, labelpad=3)
        else:
            axes[0].set_title(f"{sampleid1} - {region_title} (No Data)", fontsize=10)
            axes[0].axis('off')
            
        # Plot second sample if applicable
        if not single_sample and normalized_data2 and normalized_data2.get(r_key) is not None:
            if normalization_method == "logNorm":
                im2 = pcolormesh_square(axes[1], normalized_data2.get(r_key), start, end, 
                                       cmap=cmap, NORM=True, vmin=vmin, vmax=vmax)
            else:
                im2 = pcolormesh_square(axes[1], normalized_data2.get(r_key), start, end, 
                                       cmap=cmap, NORM=False, vmin=vmin, vmax=vmax)
            format_ticks(axes[1], rotate=False)
            axes[1].set_title(f"{sampleid2} - {region_title}", fontsize=10)
            axes[1].set_aspect('equal')
            axes[1].set_ylim(end, start)
            axes[1].set_xlim(start, end)
            
            # Add colorbar
            divider = make_axes_locatable(axes[1])
            cax2 = divider.append_axes("bottom", size="5%", pad=0.1)
            cbar2 = plt.colorbar(im2, cax=cax2, orientation='horizontal')
            cbar2.ax.tick_params(labelsize=8)
            cbar2.set_label(normalization_method, labelpad=3)
        elif not single_sample:
            axes[1].set_title(f"{sampleid2} - {region_title} (No Data)", fontsize=10)
            axes[1].axis('off')

    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)


def main(argv=None): 
    parser = argparse.ArgumentParser(description='Plot heatmaps from cooler files in the same or different chromosomes.')

    parser.add_argument('--cooler_file1', type=str, required=True, help='Path to the first .cool or .mcool file.')
    parser.add_argument('--cooler_file2', type=str, required=False, help='Path to the second .cool or .mcool file.', default=None)
    parser.add_argument('--format', type=str, default='balance', choices=['balance', 'ICE'], help='Format of .mcool file.')
    parser.add_argument('--resolution', type=int, default=None, help='Resolution for the cooler data.')
    parser.add_argument('--start1', type=int, default=None, help='Start position for region 1.')
    parser.add_argument('--end1', type=int, default=None, help='End position for region 1.')
    parser.add_argument('--chrid1', type=str, default=None, help='Chromosome ID for region 1.')
    parser.add_argument('--start2', type=int, required=False, default=None, help='Start position for region 2.')
    parser.add_argument('--end2', type=int, required=False, default=None, help='End position for region 2.')
    parser.add_argument('--chrid2', type=str, required=False, default=None, help='Chromosome ID for region 2.')
    parser.add_argument('--cmap', type=str, default='autumn_r', help='Colormap to be used for plotting.')
    parser.add_argument('--vmin', type=float, default=None, help='Minimum value for Hi-C matrix.')
    parser.add_argument('--vmax', type=float, default=None, help='Maximum value for Hi-C matrix.')
    parser.add_argument('--output_file', type=str, default='comparison_heatmap.pdf', help='Filename for the saved comparison heatmap PDF.')
    parser.add_argument('--layout', type=str, default='vertical', choices=['vertical', 'horizontal'], 
                        help="Layout of the heatmaps: 'vertical' or 'horizontal'.")
    parser.add_argument('--sampleid1', type=str, default='Sample1', help='Sample ID for the first dataset.')
    parser.add_argument('--sampleid2', type=str, default='Sample2', help='Sample ID for the second dataset.')
    parser.add_argument('--normalization_method', type=str, default='raw', 
                        choices=['raw', 'logNorm', 'log2', 'log2_add1', 'log', 'log_add1'],
                        help="Method for normalization: 'raw', 'logNorm', 'log2', 'log2_add1', 'log', or 'log_add1'.")
    parser.add_argument('--track_size', type=float, default=5, help='Size of each track (in inches).')
    parser.add_argument('--track_spacing', type=float, default=0.5, help='Spacing between tracks (in inches).')
    parser.add_argument('--track_min', type=float, default=None, help='Global minimum value for all tracks.')
    parser.add_argument('--track_max', type=float, default=None, help='Global maximum value for all tracks.')
    parser.add_argument('-V', '--version', action='version', version='SquHeatmap 1.0.0', help='Print version and exit')
    
    args = parser.parse_args()

    # Call the plotting function
    plot_heatmaps(
        cooler_file1=args.cooler_file1,
        sampleid1=args.sampleid1,
        resolution=args.resolution,
        start1=args.start1,
        end1=args.end1,
        chrid1=args.chrid1,
        start2=args.start2,
        end2=args.end2,
        chrid2=args.chrid2,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        output_file=args.output_file,
        layout=args.layout,
        cooler_file2=args.cooler_file2,
        sampleid2=args.sampleid2,
        track_size=args.track_size,
        track_spacing=args.track_spacing,
        normalization_method=args.normalization_method,
        format=args.format
    )

if __name__ == '__main__':
    main()