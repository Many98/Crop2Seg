import sys
import logging
import os
import re
import argparse
import numpy as np
from sentinel import sentinel, sentinel_sen2cor, time_series_s2

import subprocess

# ### small boiler plate to add src to sys path
import sys
from pathlib import Path
file = Path(__file__).resolve()
root = str(file).split('src')[0]
sys.path.append(root)
# --------------------------------------

from src.global_vars import SENTINEL_PATH_DATASET, SEN2COR

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Basic operations with Sentinel-1 and Sentinel-2. See configuration file to set specific settings"
                    "Capabilities are: "
                    "1) downloading Sentinel-1 and Sentinel-2 data (also Sentinel-2 time-series see config)"
                    "2) performing sen2cor processing from L1C products (Top of atmosphere) to L2A (Bottom of atmosphere)"
                    "Note that all operations will be performed in the place by default. This behavior can be changed"
                    "by setting parameter --inplace"
                    "Sen2cor processor will process all ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # PART FOR DOWNLOADING DATA (ALSO SEE CONFIGURATION FILE)
    parser.add_argument(
        "--download",
        "-d",
        action='store_true',
        help="Whether to perform download.",

    )
    # MORE SPECIFIC SETTINGS WHICH WILL TAKE PLACE FOR DOWNLOADING
    parser.add_argument(
        "--platformname",
        default='Sentinel-2',
        help="Specifies the name of mission"
             "Can be: * `Sentinel-1`"
             "        * Sentinel-2`",
    )
    parser.add_argument(
        "--count",
        default=4,
        help="Number of products to be downloaded",
    )
    parser.add_argument(
        "--polygon",
        default=None,
        help='Polygon defining Area of interest. E.g. `--polygon "[[14.31, 49.44], [13.89, 50.28], [15.55, 50.28]]"`',
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="Filename of product to be downloaded",
    )
    parser.add_argument(
        "--producttype",
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
        "--beginposition",
        help='Specifies sensing start date (Specifies interval e.g. `--beginposition "[NOW-30DAYS TO NOW]"`)'
             'To download Sentinel-2 time series use --s2_time_series argument'
        ,
        default="[NOW-30DAYS TO NOW]"
    )
    # Sentinel-1 specific
    parser.add_argument(
        "--polarisationmode",
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
        "--sensoroperationalmode",
        help="Specifies the sensor operational mode of Sentinel 1 radar."
             "Can be: * 'SM'"
             "        * 'IW' (usually used)"
             "        * 'EW'"
             "        * 'WV'",
        default='IW'
    )
    # Sentinel-2 specific
    parser.add_argument(
        "--tile_name",
        help="Specifies name of particular tile. e.g. `--tile_name T33UWQ``"
             "This can be done instead of performing search based on 'polygon' parameter.",
    )
    parser.add_argument(
        "--cloudcoverpercentage",
        default=['[0 TO 5]'],
        nargs='*',
        help='Specifies interval of allowed overall cloud coverage percentage of Sentinel-2 tile.'
             'E.g. `--cloudcoverpercentage "[0 TO 5.5]"` is translated as [0 TO 5.5] for API used'
    )
    parser.add_argument(
        "--s2_time_series",
        action='store_true',
        help="Whether to download whole time series of Sentinel-2 data."
             "Works only with Sentinel-2."
             "See configuration to set tiles, dates and cloud cover percentages",
    )
    # PART FOR SEN2COR PROCESSING
    parser.add_argument(
        "--sen2cor",
        action='store_true',
        help="Whether to perform sen2cor processing of Sentinel-2 tiles from L1C to L2A",
    )
    parser.add_argument(
        "--n_jobs",
        default=5,
        help=(
            'Number of cpus for concurrent processing when used sen2cor processor'
        ),
    )
    #  Superresolution specific args
    #parser.add_argument(
    #    "--superresolve",
    #    action='store_true',
    #    help=(
    #        'Whether to superresolution using DSen2 model on Sentinel-2 tiles to 10m'
    #    ),
    #)
    parser.add_argument(
        "--copy_original_bands",
        action='store_true',
        help="The default is not to copy the original selected 10m bands into the output file in addition "
             "to the super-resolved bands. If this flag is used, the output file may be used as a 10m "
             "version of the original Sentinel-2 file.",
    )
    parser.add_argument(
        "--input_dir", default=None, help="Directory of input Sentinel-2 tiles to be superresolved"
    )
    # --other args
    parser.add_argument(
        "--inplace",
        action='store_true',
        help=(
            'Whether to perform all operations (sen2cor and superresolving) in place '
            'meaning that old data will be removed'
        ),
    )

    args = parser.parse_args()

    if args.download:
        args.cloudcoverpercentage = args.cloudcoverpercentage[0]
        a = re.findall("\d+\.\d+", args.polygon)
        b = np.array([[a[2 * i], a[2 * i + 1]] for i in range(len(a) // 2)])
        args.polygon = b
        logging.info('DOWNLOADING DATA... \n')
        sentinel(args.polygon, args.tile_name, args.count, args.platformname,
                 args.producttype, args.filename, args.beginposition,
                 args.cloudcoverpercentage, args.polarisationmode,
                 args.sensoroperationalmode,
                 path_to_save=SENTINEL_PATH_DATASET)
    if args.s2_time_series:
        logging.info('DOWNLOADING S2 TIMESERIES... \n')
        time_series_s2(args.count, args.producttype)
    if args.sen2cor:
        logging.info('APPLYING SEN2COR... \n')
        sentinel_sen2cor(path_dataset=SENTINEL_PATH_DATASET, n_jobs=args.n_jobs, inplace=args.inplace,
                         sen2cor_path=SEN2COR)
    '''
    if args.superresolve:
        logging.info('APPLYING SUPERRESOLUTION... \n')
        if args.copy_original_bands and not args.inplace:
            subprocess.call(
                ['python3', 'helpers/DSen2/testing/s2_tiles_supres.py',  # TODO fix this problem with relative path
                 SENTINEL_PATH_DATASET, '--copy_original_bands',
                 ])
        elif args.inplace and not args.copy_original_bands:
            subprocess.call(
                ['python3', 'helpers/DSen2/testing/s2_tiles_supres.py',  # TODO fix this problem with relative path
                 SENTINEL_PATH_DATASET,
                 '--inplace'])
        elif args.inplace and args.copy_original_bands:
            subprocess.call(
                ['python3', 'helpers/DSen2/testing/s2_tiles_supres.py',  # TODO fix this problem with relative path
                 SENTINEL_PATH_DATASET,
                 '--inplace', '--copy_original_bands'])
        else:
            subprocess.call(
                ['python3', 'helpers/DSen2/testing/s2_tiles_supres.py',  # TODO fix this problem with relative path
                 SENTINEL_PATH_DATASET])
    '''
    logging.info('All operations completed succesfully!')
