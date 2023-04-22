# Crop2Seg

`Crop2Seg` is repository for my Master's thesis project.

Main goal is to perform semantic segmentation of crop types in time-series of Sentinel-2 tiles in automatic manner

Our area of interest is Czech Republic.

## What is planned
0. ![example workflow](https://badgen.net/badge/progress/99%25/green) Review methods & models for semantic segmentation of crop types in time-series of satellite images 
1. ![example workflow](https://badgen.net/badge/progress/65%25/purple) Propose enhancements if possible 

2. ![example workflow](https://badgen.net/badge/progress/75%25/cyan) Prepare public dataset of time-series of Sentinel-2 tiles over Czech Republic, with ground truth
    based on [LPIS data](https://eagri.cz/public/web/mze/farmar/LPIS/export-lpis-rocni-shp.html).
3. ![example workflow](https://badgen.net/badge/progress/0%25/red) Fine-tune [U-TAE model](https://github.com/VSainteuf/utae-paps) on created dataset
4. ![example workflow](https://badgen.net/badge/progress/70%25/purple) Explore possibilities of dealing with pixel mixing at boundary of crop fields 
5. ![example workflow](https://badgen.net/badge/progress/20%25/orange) Propose & implement method for self-supervised pretraining of [U-TAE model](https://github.com/VSainteuf/utae-paps) on Sentinel-2 time-series
6. ![example workflow](https://badgen.net/badge/progress/50%25/blue) Prepare unsupervised dataset of Sentinel-2 time-series for self-supervised pretraining
7. ![example workflow](https://badgen.net/badge/progress/0%25/red) Pretrain & fine-tune [U-TAE model](https://github.com/VSainteuf/utae-paps)
8. ![example workflow](https://badgen.net/badge/progress/15%25/orange) Propose & implement automatic pipeline/application for crop type segmentation based on fine-tuned model 
   * Users input will be AOI within Czech republic and date range
   * Output will be exported GeoTiff file (which can be viewed e.g. in Arcgis Pro) and static/dynamic map view
     (see static demo [T33UWR 17.2.2019](https://raw.githack.com/Many98/Crop2Seg/main/data/T33UWR_20190217_sample_static.html)) 

## Requirements
#### TODO

## Examples
#### TODO

## Usage
#### TODO

## Model
#### TODO
Based on [utae-paps](https://github.com/VSainteuf/utae-paps)

## Dataset

## TODO

[Static preview demo of T33UWR tile from 17.2.2019](https://raw.githack.com/Many98/Crop2Seg/main/data/T33UWR_20190217_sample_static.html)

