# Crop2Seg


Main goal  of Crop2Seg project is to explore semantic segmentation of crop types in time-series of Sentinel-2 tiles as part of author's Master's thesis.
We also want to provide standalone dataset for semantic segmentation of crop types from Sentinel-2 time-series for Czech Republic in 2019.

Finally we want to provide web application which can process Sentinel-2 data and perform semantic segmentation of crop types in automatic manner.

## Requirements
Please see requirement.txt file (TODO update it)

## Usage


### Script train.py
Main script for model training/testing/finetuning etc.

### Script crop2seg.py
 Main script for web application (see Demo below)
 
 To run web app on localhost use: 
```bash
streamlit run crop2seg.py
```

## Models

Proposed models are based on [U-TAE](https://github.com/VSainteuf/utae-paps).
Modifications are mainly made within order of processing. Also few adjustments in positional encoding are proposed.


| Time-Unet     | W-TAE         |
| ------------- | ------------- |
| ![](https://raw.githubusercontent.com/Many98/Crop2Seg/main/data/models/timeunet_final_model.png)  | ![](https://raw.githubusercontent.com/Many98/Crop2Seg/main/data/models/wtae_final_model.png)  |



## Dataset S2TSCzCrop
TODO make it publicly available

![](https://raw.githubusercontent.com/Many98/Crop2Seg/main/data/ts_sample.gif)

Dataset consits of 6945 Sentinel-2 time-series patches over Czech Republic for year 2019 divided into train/validation/test sub-datasets.
Similarly to [PASTIS](https://github.com/VSainteuf/pastis-benchmark) we use patches of shape T x C x H x W where C=10, H=W=128. Note that lenght of time-series is irregular i.e. time-series can have from 27 to 61 acquisitions.

#### Nomenclature
Dataset contains 13 classes of crops and 2 auxiliary classes (Background & Not classified). Also class representing boundary can be generated on-the-fly from ground truth (see  [on-the-fly boundary extraction](https://raw.githubusercontent.com/Many98/Crop2Seg/main/data/misc/boundary_extract.png)).

![](https://raw.githubusercontent.com/Many98/Crop2Seg/main/data/misc/img.png) ![](https://raw.githubusercontent.com/Many98/Crop2Seg/main/data/misc/legend.png)

#### Temporal characteristics

NDVI Temporal profiles of all used crop types can be found [here](https://github.com/Many98/Crop2Seg/tree/main/data/temporal_profiles)

![](https://raw.githubusercontent.com/Many98/Crop2Seg/main/data/temporal_profiles/profile.png)

#### Class Distribution
![](https://raw.githubusercontent.com/Many98/Crop2Seg/main/data/misc/class_distrib.png)

## Automatic processing pipeline

#### Demo (still WIP)

![](https://raw.githubusercontent.com/Many98/Crop2Seg/main/data/demo/demo.gif)

Web interface is created using streamlit and based on this [streamlit-template](https://github.com/giswqs/streamlit-template)

#### Schema of processing pipeline

![](https://raw.githubusercontent.com/Many98/Crop2Seg/main/data/demo/pipeline_schema.png)




