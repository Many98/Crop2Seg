import sys
import logging
import os
import re
import argparse
import numpy as np

import subprocess

# ### small boiler plate to add src to sys path
import sys
from pathlib import Path
file = Path(__file__).resolve()
root = str(file).split('src')[0]
sys.path.append(root)
# --------------------------------------

from src.global_vars import SENTINEL_PATH_DATASET
from src.helpers.sentinel import sentinel, time_series_s2

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Basic operations with Sentinel-1 and Sentinel-2. See configuration file to set specific settings"
                    "Capabilities are: "
                    "1) downloading Sentinel-1 and Sentinel-2 data (also Sentinel-2 time-series see config)"
                    "2) performing sen2cor processing from L1C products (Top of atmosphere) to L2A (Bottom of atmosphere)"
                    "Note that all operations will be performed in the place by default. This behavior can be changed"
                    "by setting parameter --inplace"
                    ,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # PART FOR DOWNLOADING DATA (ALSO SEE CONFIGURATION FILE)
    parser.add_argument(
        "--download",
        "-d",
        action='store_true',
        help="Whether to perform download.",

    )

    parser.add_argument(
        "--path",
        "-pa",
        default=SENTINEL_PATH_DATASET,
        help="Path to the folder where Sentinel-1 and Sentinel-2 data will be downloaded",
    )
    # MORE SPECIFIC SETTINGS WHICH WILL TAKE PLACE FOR DOWNLOADING
    parser.add_argument(
        "--platform",
        "-pl",
        default='Sentinel-2',
        help="Specifies the name of mission"
             "Can be: * `Sentinel-1`"
             "        * Sentinel-2`",
    )
    parser.add_argument(
        "--count",
        "-c",
        default=4,
        help="Number of products to be downloaded",
    )
    parser.add_argument(
        "--polygon",
        "-p",
        default='[[14.31, 49.44], [13.89, 50.28], [15.55, 50.28]]',
        help='Polygon defining Area of interest. E.g. `--polygon "[[14.31, 49.44], [13.89, 50.28], [15.55, 50.28]]"`',
    )
    parser.add_argument(
        "--filename",
        "-f",
        default=None,
        help="Filename of product to be downloaded",
    )
    parser.add_argument(
        "--product",
        "-pr",
        default="S2MSI2A",
        help=(
            "Specifies the type of product of particular Sentinel mission."
            "For Sentinel_1 : * 'SLC'"
            "                 * 'GRD'"
            "For Sentinel_2 : * 'S2MSI2A' (in other parts code this is referred as 'L2A')"
            "                 * 'S2MSI1C' (in other parts code this is referred as 'L1C')"
            "If None then both (all) product types will be downloaded "
            "(but only if 'filename' parameter is not used (set to None))"
        ),
    )
    parser.add_argument(
        "--begin",
        "-b",
        help='Specifies sensing start date (Specifies interval e.g. `--begin "[NOW-30DAYS TO NOW]"`)'
             'To download Sentinel-2 time series use --s2_time_series argument'
        ,
        default="[NOW-30DAYS TO NOW]"
    )

    # Sentinel-1 specific
    parser.add_argument(
        "--polarisation",
        "-pol",
        help="Specifies the polarisation mode of Sentinel 1 radar."
             "Can be: * 'HH'"
             "        * 'VV'"
             "        * 'HV'"
             "        * 'VH'"
             "        * 'HH HV'"
             "        * 'VV VH'",
        default='VV VH'
    )
    parser.add_argument(
        "--sensor",
        "-s",
        help="Specifies the sensor operational mode of Sentinel 1 radar."
             "Can be: * 'SM'"
             "        * 'IW' (usually used)"
             "        * 'EW'"
             "        * 'WV'",
        default='IW'
    )

    # Sentinel-2 specific
    parser.add_argument(
        "--tilename",
        "-t",
        help="Specifies name of particular tile. e.g. `--tilename T33UWQ``"
             "This can be done instead of performing search based on 'polygon' parameter.",
    )
    parser.add_argument(
        "--cloud",
        "-cl",
        default=['[0 TO 5]'],
        nargs='*',
        help='Specifies interval of allowed overall cloud coverage percentage of Sentinel-2 tile.'
             'E.g. `--cloud "[0 TO 5.5]"` is translated as [0 TO 5.5] for API used'
    )
    parser.add_argument(
        "--s2timeseries",
        "-ts",
        action='store_true',
        help="Whether to download whole time series of Sentinel-2 data."
             "Works only with Sentinel-2."
             "See configuration to set tiles, dates and cloud cover percentages",
    )



    args = parser.parse_args()


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    if args.download:
        args.cloud = args.cloud[0]
        a = re.findall("\d+\.\d+", args.polygon)
        b = np.array([[a[2 * i], a[2 * i + 1]] for i in range(len(a) // 2)])
        args.polygon = b
        logging.info('DOWNLOADING DATA... \n')
        sentinel(args.polygon, args.tilename, args.count, args.platform,
                 args.path,
                 args.product, args.filename, args.begin,
                 args.cloud, args.polarisation,
                 args.sensor)
        

    if args.s2timeseries:
        logging.info('DOWNLOADING S2 TIMESERIES... \n')
        time_series_s2(args.count, args.product)

    logging.info('All operations completed succesfully!')
