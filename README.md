# HiCPlot  

### HiCPlot can be used to plot square and triangle heatmaps from Hi-C matrices and tracks from bigwig files.  

#### plot square heatmaps for individual/two Hi-C contact matrices
the format of input file is cool format.  
the output file is heatmaps and genome tracks.  
#### usage:
``` 
    SquHeatmap \
    --cooler_file1 "Sample1.mcool" \
    --cooler_file2 "Sample2.mcool" \
    --bigwig_files_sample1 "Sample1_RNAseq.bw" "Sample1_ChIPseq.bw" \
    --bigwig_labels_sample1 Sample1_RNA Sample1_ChIP \
    --colors_sample1 "red" \
    --bigwig_files_sample2 "Sample2_RNAseq.bw" "Sample2_ChIP.bw" \
    --bigwig_labels_sample2 Sample2_RNA Sample2_ChIP \
    --colors_sample2 "green" \
    --gtf_file "gencode.v38.annotation.gtf" \
    --resolution 10000 --chrid "chr16" --start 67500000 --end 67700000 \
    --cmap "autumn_r" \
    --output_file "Square_horizontal_heatmap.pdf" \
    --track_size 4 \
    --track_spacing 0.5 \
    --normalization_method log2_add1 \
    --diff_cmap bwr --diff_title "log2((sample1+1)/(sample2+1))" \
    --loop_file_sample1 Sample1_loops.csv \
    --loop_file_sample2 Sample2_loops.csv \
    --genes_to_annotate "CTCF" "GFOD2" \
    --format "balance"
```
**Square and Horizontal Heatmap**  
![Square and Horizontal Heatmap](./images/Square_horizontal_heatmap.png)


#### plot triangle heatmaps for individual/two Hi-C contact matrices
#### usage: 
``` 
    TriHeatmap \
    --cooler_file1 "Sample1.mcool" \
    --cooler_file2 "Sample2.mcool" \
    --bigwig_files_sample1 "Sample1_RNAseq.bw" "Sample1_ChIPseq.bw" \
    --bigwig_labels_sample1 Sample1_RNA Sample1_ChIP \
    --colors_sample1 "red" \
    --bigwig_files_sample2 "Sample2_RNAseq.bw" "Sample2_ChIP.bw" \
    --bigwig_labels_sample2 Sample2_RNA Sample2_ChIP \
    --colors_sample2 "green" \
    --resolution 10000 --chrid "chr16" --start 67500000 --end 67700000 \
    --cmap "autumn_r" --layout 'horizontal' \
    --output_file "Triangle_horizontal_heatmap.pdf" \
    --track_width 4 \
    --track_height 1.5 \
    --track_spacing 0.5 \
    --gtf_file "gencode.v38.annotation.gtf" \
    --normalization_method "log" \
    --loop_file_sample1 Sample1_loops.csv \
    --loop_file_sample2 Sample2_loops.csv \
    --genes_to_annotate "CTCF" "GFOD2" \
    --format "balance"
``` 
**Triangle and Horizontal Heatmap**  
![Triangle and Horizontal Heatmap](./images/Triangle_horizontal_heatmap.png)

#### plot lower and upper combined square heatmaps from two Hi-C contact matrices
the format of input file is cool format.  
the output file is heatmaps and genome tracks.
#### usage:
```
    upper_lower_triangle_heatmap \
    --cooler_file1 "Sample1.mcool" \
    --cooler_file2 "Sample2.mcool" \
    --sampleid1 Sample1 \
    --sampleid2 Sample2 \
    --bigwig_files_sample1 "Sample1.bw" \
    --bigwig_labels_sample1 Sample1_RNA \
    --colors_sample1 "red" \
    --bigwig_files_sample2 "Sample2.bw" \
    --bigwig_labels_sample2 Sample2_RNA \
    --colors_sample2 "green" \
    --gtf_file "gencode.v38.annotation.gtf" \
    --resolution 10000 --chrid "chr16" --start 67500000 --end 67700000 \
    --cmap "autumn_r" \
    --output_file "lower_upper_combined_square_Heatmap.pdf" \
    --track_size 4 \
    --track_spacing 0.5 \
    --loop_file_sample1 /data/bxhu/project/DLR_AD/result/HiC/mustache/AD_merged_5kb_loops.csv \
    --loop_file_sample2 /data/bxhu/project/DLR_AD/result/HiC/mustache/Old_merged_5kb_loops.csv \
    --genes_to_annotate "CTCF" "GFOD2" \
    --normalization_method 'raw' --title "Sample1 vs Sample2" \
    --format "balance"
```

**lower and upper combined square Heatmap**  
![lower and upper combined square Heatmap](./images/lower_upper_combined_square_Heatmap.png)

#### plot square heatmaps for difference betwee two Hi-C contact matrices
the format of input file is cool format.  
the output file is heatmaps and genome tracks.  
#### usage:
``` 
    DiffSquHeatmap \
    --cooler_file1 "Sample1.mcool" \
    --cooler_file2 "Sample2.mcool" \
    --bigwig_files_sample1 "Sample1_RNAseq.bw" "Sample1_ChIPseq.bw" \
    --bigwig_labels_sample1 Sample1_RNA Sample1_ChIP \
    --colors_sample1 "red" \
    --bigwig_files_sample2 "Sample2_RNAseq.bw" "Sample2_ChIPseq.bw" \
    --bigwig_labels_sample2 Sample2_RNA Sample2_ChIP \
    --colors_sample2 "green" \
    --gtf_file "gencode.v38.annotation.gtf" \
    --resolution 10000 --chrid "chr16" --start 67500000 --end 67700000 \
    --cmap "autumn_r" \
    --output_file "Division_Square_vertical_heatmap.pdf" \
    --track_size 4 \
    --track_spacing 0.5 \
    --operation divide \
    --division_method log2_add1 \
    --diff_cmap bwr --diff_title "log2((sample1+1)/(sample2+1))" \
    --loop_file_sample1 Sample1_loops.csv \
    --loop_file_sample2 Sample2_loops.csv \
    --genes_to_annotate "CTCF" "GFOD2" \
    --format "balance"
```

**Square division Heatmap**  
![Square division Heatmap](./images/Division_Square_vertical_heatmap.png)

#### plot genomic tracks based on bigwig files
#### usage: 
``` 
    NGStrack \
    --chrid "chr16" --start 67500000 --end 67700000 \
    --layout 'horizontal' \
    --track_width 4 \
    --track_height 1.5 \
    --track_spacing 0.5 \
    --bigwig_files_sample1 "Sample1_RNAseq.bw" "Sample1_ChIPseq.bw" \
    --bigwig_labels_sample1 Sample1_RNA Sample1_ChIP \
    --colors_sample1 "red" \
    --bigwig_files_sample2 "Sample2_RNAseq.bw" "Sample2_ChIPseq.bw" \
    --bigwig_labels_sample2 Sample2_RNA Sample2_ChIP \
    --colors_sample2 "green" \
    --gtf_file "gencode.v38.annotation.gtf" \
    --output_file "track_horizontal.pdf" \
    --genes_to_annotate "CTCF" "GFOD2"
```
**Horizontal Track**  
![Horizontal track](./images/track_horizontal.png)


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

#### pip install HiCPlot==1.29
https://pypi.org/project/HiCPlot/1.29/


