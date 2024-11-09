# HiCPlot  

### HiCPlot can be used to plot square and triangle heatmaps from Hi-C matrices and tracks from bigwig files.  

#### plot square heatmaps for individual/two Hi-C contact matrices
the format of input file is cool format.  
the output file is heatmaps and genome tracks.  
#### usage:
``` 
    SquHeatmap \
    --cooler_file1 "sample1.mcool" \
    --cooler_file2 "sample2.mcool" \
    --sampleid1 "sample1" --sampleid2 "sample2" \
    --bigwig_files_sample1 "sample1.bw" \
    --bigwig_labels_sample1 "sample1 RNAseq" \
    --colors_sample1 "red" \
    --bigwig_files_sample2 "sample2.bw" \
    --bigwig_labels_sample2 "sample2 RNAseq" \
    --colors_sample2 "green" \
    --gtf_file "gencode.v38.annotation.gtf" \
    --resolution 10000 --chrid "chr2" --start 1120000 --end 1320000 \
    --cmap "autumn_r" --layout 'horizontal' \
    --output_file "twosamples_heatmap.pdf" \
    --track_size 4 \
    --track_spacing 0.5
```
**Square and Horizontal Heatmap**  
![Square and Horizontal Heatmap](./images/Square_horizontal_heatmap.png)

**Square and Vertical Heatmap**  
![Square and Vertical Heatmap](./images/Square_vertical_heatmap.png)


#### plot triangle heatmaps for individual/two Hi-C contact matrices
#### usage: 
``` 
    TriHeatmap \
    --cooler_file1 "sample1.mcool" \
    --cooler_file2 "sample2.mcool" \
    --sampleid1 "sample1" --sampleid2 "sample2" \
    --bigwig_files_sample1 "sample1.bw" \
    --bigwig_labels_sample1 "sample1 RNAseq" \
    --colors_sample1 "red" \
    --bigwig_files_sample2 "sample2.bw" \
    --bigwig_labels_sample2 "sample2 RNAseq" \
    --colors_sample2 "green" \
    --gtf_file "gencode.v38.annotation.gtf" \
    --resolution 10000 --chrid "chr2" --start 1120000 --end 1320000 \
    --cmap "autumn_r" --layout 'horizontal' \
    --output_file "twosamples_heatmap.pdf" \
    --track_width 4 \
    --track_height 1.5 \
    --track_spacing 0.5
``` 
**Triangle and Horizontal Heatmap**  
![Triangle and Horizontal Heatmap](./images/Triangle_horizontal_heatmap.png)

**Triangle and Vertical Heatmap**  
![Triangle and Vertical Heatmap](./images/Triangle_vertical_heatmap.png)

### Installation 
#### requirement for installation
python>=3.12  
numpy  
pandas  
argparse  
cooler  
matplotlib  
pyBigWig  
pyranges  

#### pip install HiCPlot==1.0.1
https://pypi.org/project/HiCPlot/1.0.1/  


