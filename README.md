# Crop2Seg 

- Main goal is to perform semantic segmentation of crop types in time-series of Sentinel-2 tiles in automatic manner

- Our area of interest is Czech Republic.

## What is planned
0. ![example workflow](https://badgen.net/badge/progress/100%25/green) Review methods & models for semantic segmentation of crop types in time-series of satellite images.
1. ![example workflow](https://badgen.net/badge/progress/100%25/green) Propose enhancements of reviewed methods & models if possible.

2. ![example workflow](https://badgen.net/badge/progress/100%25/green) Prepare public dataset of time-series of Sentinel-2 tiles over Czech Republic, with ground truth
    based on [LPIS data](https://eagri.cz/public/web/mze/farmar/LPIS/export-lpis-rocni-shp.html).
3. ![example workflow](https://badgen.net/badge/progress/95%25/green) Explore possibilities of dealing with pixel mixing at boundary of crop fields .
4. ![example workflow](https://badgen.net/badge/progress/85%25/cyan) Propose & implement automatic pipeline/application for crop type segmentation based on fine-tuned model.
   * User's input could be AOI within Czech republic and date range
   * Output will be GeoTiff file representing crop type map (which can be viewed e.g. in Arcgis Pro) and dynamic map view
     (E.g. see this static demo [T33UWR 17.2.2019](https://raw.githack.com/Many98/Crop2Seg/main/data/T33UWR_20190217_sample_static.html)) 

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

![](https://raw.githubusercontent.com/Many98/Crop2Seg/main/data/ts_sample.gif)

#### TODO

[Static preview demo of T33UWR tile from 17.2.2019](https://raw.githack.com/Many98/Crop2Seg/main/data/T33UWR_20190217_sample_static.html)

