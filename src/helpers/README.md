## Helpers

This directory contains helper modules which are from another project of mine.

They mainly help to process Sentinel data, work with geospatial and raster data and create dataset.

Among others one can use them to:
- Download Sentinel-2 (L1C, or L2A) data from https://dhr1.cesnet.cz/ (`sentinel.py`, `sentinel_cli.py`)
- Download time-series of Sentinel-2 (L1C, or L2A) data (`sentinel.py`, `sentinel_cli.py`)
- Apply Sen2Cor processor on L1C data (`sentinel_cli.py`)
- Use raster-like interface for Sentinel-2 (.SAFE) data (`sentinel2raster.py`)
- Create time-series dataset from Sentinel-2 data (`dataset_creator.py`)

### Notes:

- Modules and functionalities heavily depend on configuration file located in
`config` directory
- To apply Sen2Cor processor one need to install (https://step.esa.int/main/snap-supported-plugins/sen2cor/sen2cor-v2-11/) it prior to use it
  (tested only on Linux)

### Examples

## TODO add examples