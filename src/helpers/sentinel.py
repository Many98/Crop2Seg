import os
import sys
import json
import time

from geojson import Polygon, Point
from datetime import datetime, timedelta
import pandas as pd
import math
import xml.etree.ElementTree as ET
import zipfile
import shutil
import requests
from urllib.parse import urljoin
from shapely import geometry
from osgeo import gdal
from tqdm.auto import tqdm
import logging

import rasterio
from rasterio import mask

import numpy as np
import cv2
import ray
from filelock import FileLock
import subprocess
import re


# ### small boiler plate to add src to sys path
import sys
from pathlib import Path
file = Path(__file__).resolve()
root = str(file).split('src')[0]
sys.path.append(root)
# --------------------------------------

from src.helpers.utils import distribute_args

from src.global_vars import ODATA_URI, ODATA_RESOURCE, OPENSEARCH_URI, ACCOUNT, PASSWORD, \
    SENTINEL_PATH_DATASET, SEN2COR, TILES, DATES, CLOUDS, MAX_CLOUD, MAX_SNOW, MIN_SIZE_L2A, MIN_SIZE_L1C

logging.getLogger().setLevel(logging.INFO)


######################
# HANDLE DOWNLOADING #
######################
def create_keywords(polygon, **kwargs):
    """
    Creates main part of text query.

    Parameters
    ----------
    polygon : numpy.array or None
        Specifies vertices of polygon which defines Area of Interest (AOI).
        If None then search based on polygon is not performed.
    **kwargs
        See https://scihub.copernicus.eu/userguide/FullTextSearch

    Returns
    ----------
    keywords : str
        Main part of text query.
    """
    keywords = []
    for key, value in kwargs.items():
        keywords.append(f'{key}:{value}')

    # IF POLYGON SPECIFIES SOME AREA OF INTEREST
    # (this should bring a bit flexibility to search based on name of tile [e.g. T33UWQ] what practically defines AOI
    # instead of searches requiring AOI defined by 'polygon' parameter)
    if polygon is not None:
        vyrez = ""
        for idx_bod in range(len(polygon)):
            vyrez += f'{polygon[idx_bod, 0]} {polygon[idx_bod, 1]}, '
        vyrez += f'{polygon[0, 0]} {polygon[0, 1]}'

        keywords.append(f'footprint:"Intersects(POLYGON(({vyrez})))"')
    keywords = " AND ".join(keywords)
    return keywords


def sentinel_query(polygon, opensearch_uri=OPENSEARCH_URI, count=5, account=ACCOUNT, password=PASSWORD, **kwargs):
    """
    Queries the OpenSearch API.

    Parameters
    ----------
    polygon : numpy.array or None
        Specifies vertices of polygon which defines Area of Interest (AOI).
        If None then search based on polygon is not performed.
    opensearch_uri : str
    count : int
        Specifies number of tiles to query. Only the best ones according to size and cloud percentage will be returned
    account : str
    password : str
    **kwargs
        See https://scihub.copernicus.eu/userguide/FullTextSearch

    Returns
    ----------
    id_list : list
        Ids (UUIDs) satisfying the query.
    json_feed : dict
        Contains the full information about query.
    total_results : int
        Number of ids satisfying the query.

    """

    def rank(tile_type, cloud, size):
        """

        Parameters
        ----------
        size: float
            Size of tile file in MB. It somehow describes area of relevant data (not zeros)
        cloud: float
            Represents cloud percentage of tile
        tile_type: str
            Tile type; possible are `L1C` and `L2A`.
            This is required because of different tile sizes of L1C and L2A products
        Returns
        -------

        """
        if tile_type == 'L1C':
            return (-(cloud / (MAX_CLOUD // 10)) + 10) * ((size / 100) - MIN_SIZE_L1C/100) if (size >= MIN_SIZE_L1C and cloud <= MAX_CLOUD) else 0.0
        elif tile_type == 'L2A':
            return (-(cloud / (MAX_CLOUD // 10)) + 10) * ((size / 100) - MIN_SIZE_L2A/100) if (size >= MIN_SIZE_L2A and cloud <= MAX_CLOUD) else 0.0
        else:
            return None

    if count > 100:
        sys.exit('Error: more than 100 results. Maximum 100 results can be queried, aborting.')

    keywords = create_keywords(polygon, **kwargs)

    query_parts = ['search?q=(' + keywords + ')', 'format=JSON', f'rows={100}']
    query_full = "&".join(query_parts)

    url = urljoin(opensearch_uri, query_full)
    logging.info(f'Querying ... \n {url}')

    response = requests.get(url, auth=(account, password))

    if response.status_code == 401:
        raise Exception('Unauthorized access to Opensearch API!')
    else:
        json_feed = response.json()['feed']

        id_list = []
        tile_type_list = []
        cloud_percentage_list = []
        snow_percentage_list = []
        size_list = []
        title_list = []
        if not 'entry' in json_feed:
            raise RuntimeError('No results, matching set conditions was found. Check if cloud condition is not too.'
                               'restrictive')
        if type(json_feed['entry']) is list:
            for i, entry in enumerate(json_feed['entry']):
                _ = [j for j in json_feed['entry'][i]['str'] if j['name'] == 'processinglevel']
                tile_type = _[0]['content'][-2:]
                tile_type_list.append('L1C' if tile_type == '1C' else 'L2A' if tile_type == '2A' else 'other')
                _ = [j for j in json_feed['entry'][i]['str'] if j['name'] == 'size']
                size = _[0]['content'].split(' ')
                size_list.append(float(size[0]) if size[1] == 'MB' else float(size[0]) * 1000)
                _ = [j for j in json_feed['entry'][i]['double'] if j['name'] == 'cloudcoverpercentage']
                cloud_percentage_list.append(float(_[0]['content']))
                _ = [j for j in json_feed['entry'][i]['double'] if j['name'] == 'snowicepercentage']
                snow_percentage_list.append(float(_[0]['content']))
                id_list.append(entry['id'])
                title_list.append(json_feed['entry'][i]['title'])

            df = pd.DataFrame(list(zip(title_list, id_list, tile_type_list, cloud_percentage_list, snow_percentage_list,
                                       size_list, [None] * len(id_list))),
                              columns=['title', 'ids', 'types', 'clouds', 'snow', 'size', 'rank'])

            logging.info(f"APPLYING SNOW COVER PERCENTAGE FILTER (<={MAX_SNOW}%)")
            df = df[df['snow'] <= MAX_SNOW]
            logging.info(f"LEFT {df.shape[0]}/{int(json_feed['opensearch:totalResults'])} TILE CANDIDATES")

            if not df.empty:
                # RANK THE PRODUCTS ACCORDING TO CLOUD PERCENTAGE AND SIZE
                df['rank'] = df.apply(lambda x: rank(x['types'], x['clouds'], x['size']), axis=1)
                df.sort_values('rank', ascending=False, inplace=True)

            logging.info(f"APPLYING SIZE AND CLOUD COVER PERCENTAGE FILTERS (<={MAX_CLOUD}%)")
            df = df[df['rank'] > 0.0]
            logging.info(f"LEFT {df.shape[0]}/{int(json_feed['opensearch:totalResults'])} TILE CANDIDATES")

            id_list = df.head(count)['ids'].to_list()
        elif type(json_feed['entry']) is dict:
            _type = [j for j in json_feed['entry']['str'] if j['name'] == 'processinglevel']
            tile_type = _type[0]['content'][-2:]
            _snow = [j for j in json_feed['entry']['double'] if j['name'] == 'snowicepercentage']
            _size = [j for j in json_feed['entry']['str'] if j['name'] == 'size']
            _cloud = [j for j in json_feed['entry']['double'] if j['name'] == 'cloudcoverpercentage']
            size = _size[0]['content'].split(' ')
            size_filter = MIN_SIZE_L1C if tile_type == 'L1C' else MIN_SIZE_L2A
            size = float(size[0]) if size[1] == 'MB' else float(size[0]) * 1000

            if float(_snow[0]['content']) <= MAX_SNOW and size >= size_filter and float(_cloud[0]['content']) <= MAX_CLOUD:
                id_list.append(json_feed['entry']['id'])  # in this case only one id is in json_feed
            else:
                logging.info(f"SKIPPING DUE TO FILTER RESTRICTIONS")

        total_results = int(json_feed['opensearch:totalResults'])

        logging.info(f'OVERALL NUMBER OF RESULTS TO BE DOWNLOADED: {len(id_list)}/{total_results}')

        return id_list, json_feed, total_results


def sentinel_download(id_list, json_feed,
                      path_dataset=SENTINEL_PATH_DATASET,
                      odata_uri=ODATA_URI, odata_resource=ODATA_RESOURCE, opensearch_uri=OPENSEARCH_URI,
                      account=ACCOUNT, password=PASSWORD):
    """"
    Downloads the UUIDs' tiles and informative .json via the OData API.

    Parameters
    ----------
    id_list : list
        List of uuids of tiles to be downloaded.
    json_feed : dict
        Json response from query.
    path_dataset: str
        Specifies (path) where to download: SENTINEL_PATH_DATASET
    odata_uri : str
    odata_resource : str
    account : str
    password : str
    """
    for idx, sentinel_uuid in enumerate(tqdm(id_list, position=0)):
        url = urljoin(odata_uri, odata_resource)
        url_full = f"{url}('{sentinel_uuid}')/$value"  # should be https://dhr1.cesnet.cz/odata/v1/Products(uuid)/$value
        logging.info(f'Downloading from ... \n {url_full}')

        if type(json_feed['entry']) is list:
            path = os.path.join(path_dataset, json_feed['entry'][idx]['title'])

            # SAVE THE JSON RESPONSE
            with open(path + '.json', 'w') as outfile:
                json.dump(json_feed['entry'][idx], outfile)

            # CHECK WHETHER THE .ZIP IS ALREADY DOWNLOADED (BUT ZIP CAN BE STILL DOWNLOADED ONLY PARTIALLY)
            if (json_feed['entry'][idx]['title'] + '.zip') in os.listdir(path_dataset) or \
                    (json_feed['entry'][idx]['title'] + '.SAFE') in os.listdir(path_dataset):
                logging.info('This tile has already been downloaded.')
                continue

        # IT IS NOT LIST IF THERE IS ONLY ONE TILE TO BE DOWNLOADED
        else:
            path = os.path.join(path_dataset, json_feed['entry']['title'])

            # SAVE THE JSON RESPONSE
            with open(path + '.json', 'w') as outfile:
                json.dump(json_feed['entry'], outfile)

            # CHECK WHETHER THE .ZIP IS ALREADY DOWNLOADED (BUT ZIP CAN BE STILL DOWNLOADED ONLY PARTIALLY)
            if (json_feed['entry']['title'] + '.zip') in os.listdir(path_dataset) or \
                    (json_feed['entry']['title'] + '.SAFE') in os.listdir(path_dataset):
                logging.info('This tile has already been downloaded.')
                continue

        # DOWNLOAD THE TILE
        try:
            with requests.get(url_full, auth=(account, password), stream=True) as r:
                chunk_size = 1024
                r.raise_for_status()
                with open(path + '.zip', 'wb') as f:
                    pbar = tqdm(desc=f"Downloading {id_list[idx]}", unit="B", total=int(r.headers['Content-Length']),
                                position=0)
                    for chunk in tqdm(r.iter_content(chunk_size=chunk_size)):
                        pbar.update(len(chunk))
                        f.write(chunk)
        except requests.exceptions.HTTPError as e:
            logging.info(f'Http error occurred with response {e.response}')


def sentinel_unzip(path_dataset=SENTINEL_PATH_DATASET):
    """
    Unzips the tiles.

    Parameters
    -----------
    path_dataset: str
        Specifies the path where to unzip downloaded (zipped) tiles.
    """
    zip_folders = [f for f in os.listdir(path_dataset) if f.endswith('.zip')]
    for path_tile in tqdm(zip_folders, desc='Unziping tiles'):
        # CHECK WHETHER THE ZIP IS UNZIPPED
        if (path_tile[:-4] + '.SAFE') in os.listdir(path_dataset):
            continue

        # CHECK IF ZIP IS TREATED AS ZIP (ZIPFILE MODULE DID NOT WORK ON ALL ZIPPED TILES)
        if zipfile.is_zipfile(path_tile):
            with zipfile.ZipFile(os.path.join(path_dataset, path_tile), "r") as zip_ref:
                zip_ref.extractall(path_dataset)

        # OTHERWISE USE SHUTIL MODULE
        else:
            try:
                shutil.unpack_archive(os.path.join(path_dataset, path_tile), extract_dir=path_dataset)
            except Exception:
                os.remove(os.path.join(path_dataset, path_tile))
                raise Exception(f"File {path_tile} is damaged. Try to download it again.")
        # finally remove zip file
        os.remove(os.path.join(path_dataset, path_tile))


def sentinel(polygon=None, tile_name=None, count=4, platformname='Sentinel-2', path_to_save=SENTINEL_PATH_DATASET,
             producttype='S2MSI2A', filename=None, beginposition='[NOW-30DAYS TO NOW]',
             cloudcoverpercentage='[0 TO 5]', polarisationmode='VV VH',
             sensoroperationalmode='IW', **kwargs):
    """
    Downloads and unzip given Sentinel product.
    Unsuccessful searches can be caused because of interference of arguments ('intersections' of conditions
    derived from arguments is not True).

    More info: https://scihub.copernicus.eu/userguide/FullTextSearch

    Parameters
    ----------
    path_to_save: str
        Specifies path where should be downloaded tiles stored.
        Default is storing to SENTINEL_PATH_DATASET (For Sentinel 2 tiles).
        It must be explicitly set if Sentinel 1 tiles are downloaded to not have them in same directory.
    tile_name: str or None
        Specifies name of particular tile. e.g. 'T33UWQ'
        This can be done instead of performing search based on 'polygon' parameter.
        If None then this parameter will not be used. (If not None then parameter 'polygon' must be set to None
        to use search based on tile_name.)
    polygon: numpy.array or None
        Specifies vertices of polygon (AOI).
        If None then search based on Area of Interest is not performed.
        If polygon edges cross each other then search can raise error.
        If None then this parameter will not be used.
        (For both Sentinel 1 and 2 products.)
    count: int
        Specifies number of products (tiles) to download.
        (For both Sentinel 1 and 2 products.)
    platformname: str
        Specifies the name of mission
        Can be: * 'Sentinel-1'
                * 'Sentinel-2'
        Required parameter.
        (For both Sentinel 1 and 2 products.)
    producttype: str or None
        Specifies the type of product of particular Sentinel mission.
        For Sentinel_1 : * 'SLC'
                         * 'GRD'
        For Sentinel_2 : * 'S2MSI2A' (in other parts code this is referred as 'L2A')
                         * 'S2MSI1C' (in other parts code this is referred as 'L1C')
        If None then both (all) product types will be downloaded (but only if 'filename' parameter is not used (set to None))
        (For both Sentinel 1 and 2 products.)
    filename: str or None
        Name of file containing product.
        E.g. 'S2A_MSIL1C_20201203T104421_N0209_R008_T30PVU_20201203T142136.SAFE',
              'S1A_IW_GRDH_1SDV_20200913T165157_20200913T165222_034343_03FE2F_56A8.SAFE',
              'S1A_IW_GRDH_1SDV_*',
              'S2A_MSIL1C*',
              'S2A_MSIL1C_*T30PVU_20201203T142136.SAFE'
        Use this to perform search based on particular name of file of a product.
        (This has less search priority than search based on AOI, in other words
        'polygon' parameter must be set to None to perform search based on filename)
        (For both Sentinel 1 and 2 products.)
        If None then this parameter will not be used.
    beginposition: str or None
        Specifies sensing start date (Specifies interval e.g. [NOW-30DAYS TO NOW])
        The general form to be used is: [<timestamp> TO <timestamp>]
        Where <timestamp> can be:   * yyyy-MM-ddThh:mm:ss.SSSZ (ISO8601 format)
                                    * NOW
                                    * NOW-<n>MINUTE(S)
                                    * NOW-<n>HOUR(S)
                                    * NOW-<n>DAY(S)
                                    * NOW-<n>MONTH(S)
        There is another option to use parameter 'endposition' as kwarg but usually its interval
        is same as 'beginposition' so there is no need to specify 'endposition' argument.
        If None then this parameter will not be used.
        (For both Sentinel 1 and 2 products.)
    cloudcoverpercentage: str or None
        Specifies interval of allowed overall cloud coverage percentage of tile.
        E.g. [0 TO 5.5].
        If None then this parameter will not be used.
        (Only for Sentinel 2 products.)
    polarisationmode: str or None
        Specifies the polarisation mode of Sentinel 1 radar.
        Can be: * 'HH'
                * 'VV'
                * 'HV'
                * 'VH'
                * 'HH HV'
                * 'VV VH'
        If None then this parameter will not be used.
        (Only for Sentinel 1 products.)
    sensoroperationalmode: str or None
        Specifies the sensor operational mode of Sentinel 1 radar.
         Can be: * 'SM'
                 * 'IW' (usually used)
                 * 'EW'
                 * 'WV'
        If None then this parameter will not be used.
        (Only for Sentinel 1 products.)
    Examples
    --------------
    * To perform search and download:
    * 10 SENTINEL 2 L2A TILES BASED ON AREA OF INTEREST (AOI) IN TIME INTERVAL FROM NOW TO 5 MONTHS BACK WITH ALLOWED
      0-5 % CLOUD COVERAGE -> sentinel(polygon=np.array([[14.31, 49.44], [13.89, 50.28], [15.55, 50.28]]),
      platformname='Sentinel-2', producttype='S2MSI2A', count=10, cloudcoverpercentage='[0, 5]',
      beginposition='[NOW-5MONTH TO NOW]')

    * 1 SENTINEL 1  TILES BASED ONLY ON FILENAME -> sentinel(polygon=None, tile_name=None,
      filename='S1A_IW_GRDH_1SDV_20200913T165157_20200913T165222_034343_03FE2F_56A8.SAFE',
      platformname='Sentinel-1', producttype=None, beginposition=None, path_to_save=SENTINEL_PATH_DATASET_S1,
      polarisationmode=None, sensoroperationalmode=None)

    * 10 SENTINEL 1  TILES BASED ONLY ON FILENAME STARTING WITH 'S1A_IW_GRDH' -> sentinel(polygon=None, tile_name=None,
      filename='S1A_IW_GRDH*', polarisationmode=None, sensoroperationalmode=None, beginposition=None,
      platformname='Sentinel-1', producttype=None, count=10, path_to_save=SENTINEL_PATH_DATASET_S1)

    * 10 SENTINEL 2 TILES WITH PARTICULAR NAME T33UUR (THIS EXPLICITLY DEFINES AOI) -> sentinel(polygon=None,
      tile_name='T33UUR', filename=None, platformname='Sentinel-2', producttype=None, count=10, cloudcoverpercentage=None,
      beginposition=None)

    * 10 SENTINEL 2 L2A TILES WITH PARTICULAR NAME T33UUR (THIS EXPLICITLY DEFINES AOI)
      -> sentinel(polygon=None, tile_name='T33UUR', filename=None,
      platformname='Sentinel-2', producttype='S2MSI2A', count=10, cloudcoverpercentage=None)

    * 10 SENTINEL 2 L2A TILES WITH PARTICULAR NAME T33UUR (THIS EXPLICITLY DEFINES AOI) FROM NOW TO 120 DAYS BACK
      -> sentinel(polygon=None, tile_name='T33UUR', filename=None, beginposition='[NOW-120DAYS TO NOW]',
      platformname='Sentinel-2', producttype='S2MSI2A', count=10, cloudcoverpercentage=None)

    * 10 SENTINEL 2 L2A TILES WITH NAME T33UUR, ALLOWED CLOUD COVERAGE 0-10 %, FROM 1.6.2020 TO 31.7.2020 ->
      sentinel(polygon=None, tile_name='T33UUR', filename=None,
      beginposition='[2020-06-01T00:00:00.000Z TO 2020-07-31T00:00:00.000Z]',
      platformname='Sentinel-2', producttype='S2MSI2A', count=10, cloudcoverpercentage='[0 TO 10]')

    """
    # ARGUMENTS TO BE SET
    args = None

    ######################################
    # PERFORM SEARCH BASED ON TILE_NAME  #
    # INSTEAD OF SEARCH BASED ON POLYGON #
    # POLYGON IS NOW EXPLICITLY SET TO   #
    # NONE                               #
    ######################################
    if tile_name is not None and polygon is None:

        # TILE NAME MUST BE CORRECT
        if len(tile_name) == 6:
            if producttype in ['S2MSI2A', 'S2MSI1C', None]:
                args = {'filename': '*' + tile_name + '*',
                        'producttype': producttype,
                        'beginposition': beginposition,
                        'cloudcoverpercentage': cloudcoverpercentage}  # PARAMETER SPECIFIC TO SENTINEL 2
            else:
                raise Exception(f"Invalid 'producttype' parameter [{producttype}]. \n"
                                f"Possible values are ['S2MSI2A', 'S2MSI1C', None]")
        else:
            raise Exception(f"Tile [{tile_name}] name seems to be wrong.")

    ######################################
    # PERFORM SEARCH BASED ON            #
    # ANOTHER PARAMETERS THAN TILE_NAME  #
    ######################################
    else:
        # FIRST CHECK PLATFORM-NAME TO DETERMINE IF DOWNLOAD SENTINEL 2 OR 1 TILES
        if platformname == "Sentinel-2":
            # DOWNLOAD SPECIFIED PRODUCTTYPE IN PARTICULAR PLATFORM
            # NONE MEANS TO DOWNLOAD ALL PRODUCT TYPES IN PARTICULAR PLATFORM
            if producttype in ['S2MSI2A', 'S2MSI1C', None]:
                args = {'platformname': platformname, 'beginposition': beginposition, 'producttype': producttype,
                        'cloudcoverpercentage': cloudcoverpercentage}  # PARAMETER SPECIFIC TO SENTINEL 2
            else:
                raise Exception(f"Invalid 'producttype' parameter [{producttype}] \n"
                                f"Possible values are ['S2MSI2A', 'S2MSI1C', None]")

        elif platformname == "Sentinel-1":
            if sensoroperationalmode not in ['SM', 'IW', 'EW', 'WV', None]:
                raise Exception(f"Invalid 'sensoroperationalmode' parameter [{sensoroperationalmode}] \n"
                                f"Possible values are ['SM', 'IW', 'EW', 'WV', None]")
            if polarisationmode not in ['HH', 'VV', 'HV', 'VH', 'HH HV', 'VV VH', None]:
                raise Exception(f"Invalid 'polarisationmode' parameter [{polarisationmode}] \n"
                                f"Possible values are ['HH', 'VV', 'HV', 'VH', 'HH HV', 'VV VH', None]")

            # DOWNLOAD SPECIFIED PRODUCTTYPE IN PARTICULAR PLATFORM
            # NONE MEANS TO DOWNLOAD ALL PRODUCT TYPES IN PARTICULAR PLATFORM
            if producttype in ['SLC', 'GRD', 'OCN', None]:
                args = {'platformname': platformname, 'beginposition': beginposition, 'producttype': producttype,
                        'sensoroperationalmode': sensoroperationalmode,  # PARAMETER SPECIFIC TO SENTINEL 1
                        'polarisationmode': polarisationmode}  # PARAMETER SPECIFIC TO SENTINEL 1
            else:
                raise Exception(f"Invalid 'producttype' parameter [{producttype}] \n"
                                f"Possible values are ['SLC', 'GRD', 'OCN', None]")

            ###########################################
            # PERFORM SEARCH BASED ON                 #
            # PARTICULAR NAME OF FILE OF PRODUCT      #
            # REQUIRES CORRECT NAMING CONVENTIONS     #
            ###########################################
            if filename is not None:
                logging.warning(
                    "It is assumed that 'filename' parameter fits the naming convention of particular tile. \n"
                    "To obtain results based only on filename explicitly set parameters 'polygon' and \n "
                    "'tile_name' to None otherwise these parameters are also included in search.\n "
                    "E.g. filename='S2*T33UUR*' AND platformname='Sentinel-1'  will not work."
                    "For more info about naming conventions of files \n "
                    "see https://scihub.copernicus.eu/userguide/FullTextSearch ")

                args['filename'] = filename
        else:
            raise Exception(f"Invalid 'platformname' parameter {platformname}. Possible values are \n "
                            f"{['Sentinel-1', 'Sentinel-2']}")

    # FILTER OUT ARGS WHICH ARE SET TO NONE
    args = {k: v for k, v in args.items() if v is not None}

    # FINALLY CALL FUNCTION TO PERFORM SEARCH
    id_list, json_feed, num_results = sentinel_query(polygon, count=count, **args, **kwargs)

    # DOWNLOAD TILES
    sentinel_download(id_list, json_feed, path_dataset=path_to_save)

    # in last pass it always left some unziped tiles
    time.sleep(10)

    # UNZIP TILES
    sentinel_unzip(path_dataset=path_to_save)


def sentinel_sen2cor(path_dataset=SENTINEL_PATH_DATASET, n_jobs=15, inplace=True,
                     sen2cor_path=SEN2COR):
    """
    Performs sen2cor processing on all L1C tiles.
    This assumes that sen2cor was already installed. (See.: https://step.esa.int/main/snap-supported-plugins/sen2cor/)
    Parameters
    ----------
    path_dataset: str
        Path where are stored L1C tiles to be processed
    n_jobs: int
        Number of concurrent jobs to be run
    sen2cor_path: str
        Path to sen2cor executable
    Returns
    -------

    """
    os.chdir(path_dataset)
    already_processed = ['_'.join([f.split('_')[2], f.split('_')[5]]) for f in os.listdir(path_dataset) if
                         re.search(r'(L2A).*\.SAFE$', f)]
    tiles_all = [f for f in os.listdir(path_dataset) if re.search(r'(L1C).*\.SAFE$', f)]
    tiles_all = [tile for tile in tiles_all if
                 '_'.join([tile.split('_')[2], tile.split('_')[5]]) not in already_processed]
    args = distribute_args(tiles_all, n_jobs)
    tiles_all = [[tiles_all[j] for j in range(i[0], i[1])] for i in args]

    n_jobs = len(tiles_all)

    @ray.remote
    def run_sen2cor(tiles_part):
        for tile in tqdm(tiles_part, desc=f'Processing with sen2cor...'):
            if tile is not None:
                _ = subprocess.run([sen2cor_path, tile])
                with FileLock("processed.txt.lock"):
                    with open("processed.txt", "a") as f:
                        f.write(tile)
                if inplace:
                    try:
                        shutil.rmtree(tile)
                    except OSError as e:
                        logging.info(f"Error: {tile} : {e.strerror}")
                        with FileLock("log.txt.lock"):
                            with open("log.txt", "a") as f:
                                f.write(f"Error deleting: {tile} : {e.strerror}")
                        continue

    ray.init(num_cpus=n_jobs)

    results_id = [run_sen2cor.remote(tiles_part) for tiles_part in tiles_all]
    _ = [ray.get(results_id)]

    ray.shutdown()


def sentinel_scl_dictionary(inverse=False):
    dic = {'no_data': 0, 'saturated_or_defective': 1, 'dark_area_pixels': 2, 'cloud_shadows': 3, 'vegetation': 4,
           'not_vegetated': 5, 'water': 6, 'unclassified': 7, 'cloud_medium_probability': 8,
           'cloud_high_probability': 9, 'thin_cirrus': 10, 'snow': 11}
    return dic if not inverse else {v: k for k, v in dic.items()}


###############################################
# HANDLE DATA (TILES) PROCESSING/LOADING ETC. #
###############################################
def sentinel_rescale(data_channel, ratio=1 / 2, interpolation=cv2.INTER_AREA):
    """
    Rescales Sentinel 2 image data.

    Parameters
    ----------
    data_channel : numpy.array
        Specifies particular channel (set of bands with same resolution, e.g. 20m) to be rescaled.
    ratio : float
        Species the rescaling ratio.
    interpolation :
        Specifies the algorithm to be used for rescaling.
        E.g.  cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_CUBIC

    Returns
    -------
    data_rescaled: numpy.array
    """

    dim = (int(data_channel.shape[1] * ratio), int(data_channel.shape[2] * ratio))
    data_rescaled = np.empty(
        (data_channel.shape[0], int(data_channel.shape[1] * ratio), int(data_channel.shape[2] * ratio)),
        dtype=data_channel.dtype)

    for channel in range(data_channel.shape[0]):  # CHANNEL IS HERE MEANT AS BAND
        data_rescaled[channel] = cv2.resize(data_channel[channel, :, :], dim, interpolation=interpolation)

    return data_rescaled


def sentinel_load_tile(path_tile, path_dataset=SENTINEL_PATH_DATASET, use_only_20m_resolution=True):
    """
    Loads the specified Sentinel 2 tile. Channels R10m and R60m are resized to be the same shape as R20m.
    For L2A it loads 9 bands in 20m resolution.
    For L1C it loads 6 bands in 20m resolution.

    Parameters
    ----------
    use_only_20m_resolution: bool
        Specifies if load only bands in 20m resolution.
    path_dataset : str
        Specifies absolute path where are stored Sentinel 2 tiles to be loaded (SENTINEL_PATH_DATASET).
    path_tile : str
        Relative path in .SAFE format path of a tile (of the Sentinel 2).

    Returns
    ---------
        data: ndarray
    """

    # S2_L2A R10m:B02,B03,B04,B08
    #        R20m:B02,B03,B04,B05,B06,B07,B8A,B11,B12
    #        R60m:B01,B02,B03,B04,B05,B06,B07,B8A,B09,B11,B12

    data = np.array([])

    channels = ['R20m'] if use_only_20m_resolution else ['R10m', 'R20m', 'R60m']

    for idx, channel in enumerate(tqdm(channels, desc=f"Loading channels")):

        # LOAD SPECIFIED BANDS IN PARTICULAR RESOLUTION
        data_channel = sentinel_load_channel(path_tile, path_dataset, channel=channel)

        if channel == 'R10m':
            data_channel = sentinel_rescale(data_channel, ratio=1 / 2)
        if channel == 'R60m':
            data_channel = sentinel_rescale(data_channel, ratio=3)

        data = np.concatenate([data, data_channel], axis=0) if data.size else data_channel

    return data


def sentinel_load_channel(path_tile, path_dataset=SENTINEL_PATH_DATASET, channel="R20m", band=None,
                          return_reader=False):
    """
    Loads the specified channel (set of bands within particular resolution). Channel could be R10m, R20m or R60m.
    If parameter band is specified then function loads specified band in specified channel,
    if no such band exists then empty array is returned.

    Parameters
    ----------
    band : str or None
        Specifies particular band to be loaded.
        Can be 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'
        If None then all bands within particular channel (resolution will be loaded)
    path_dataset : str
        Specifies absolute path where are stored tiles to be loaded (SENTINEL_PATH_DATASET).
    path_tile : str
        Relative path in .SAFE format path of a tile.
    channel: str
          Specifies channel resolution.
    return_reader : bool
        Specifies if channel should be returned as raster reader in mode 'r'
        Default is False

    Returns
    ----------
        : ndarray if return_reader set to False otherwise returns list of raster readers in 'r' mode
    """
    # S2_L2A R10m:B02,B03,B04,B08
    #        R20m:B02,B03,B04,B05,B06,B07,B8A,B11,B12
    #        R60m:B01,B02,B03,B04,B05,B06,B07,B8A,B09,B11,B12

    resolutions_of_bands = {'B01': 'R60m', 'B02': 'R10m', 'B03': 'R10m', 'B04': 'R10m', 'B05': 'R20m',
                            'B06': 'R20m',
                            'B07': 'R20m', 'B08': 'R10m', 'B8A': 'R20m', 'B09': 'R60m', 'B10': 'R60m',
                            'B11': 'R20m',
                            'B12': 'R20m'}

    if band not in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', None]:
        raise Exception(f"Invalid 'band' parameter [{band}]. \n"
                        f"Possible values are\n"
                        f"['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', "
                        f"'B09', 'B11', 'B12', None] ")
    if channel not in ['R10m', 'R20m', 'R60m']:
        raise Exception(f"Invalid 'channel' parameter [{channel}]. \n"
                        f"Possible values are ['R10m', 'R20m', 'R60m']")

    a = os.path.join(path_dataset, path_tile, 'GRANULE')
    a = os.path.join(a, os.listdir(a)[0], 'IMG_DATA')
    arrs = []
    jp2s = []

    # CONSIDER ONLY SENTINEL 2 L2A TILES TO BE LOADED
    if path_tile.split('_')[1].endswith('2A'):
        a = os.path.join(a, channel)
        # SET ONLY SPECIFIED BAND OF SPECIFIED RESOLUTION (CHANNEL) TO BE LOADED
        if band in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']:
            jp2s = [os.path.join(a, f) for f in os.listdir(a) if f.split('_')[2] == band]
        # SET ALL BANDS OF SPECIFIED RESOLUTION TO BE LOADED
        elif band is None:
            # THIS GUARANTEE ORDER IN R20M CHANNEL TO BE [B02, B03, B04, B05, B06, B07, B11, B12, B8A]
            bands = [f for f in sorted(os.listdir(a), key=lambda x: x.split('_')[2]) if f.split('_')[2].startswith('B')]
            jp2s = [os.path.join(a, f) for f in bands]
        else:
            raise Exception(f"Invalid 'band' parameter [{band}]. \n"
                            f"Possible values are\n"
                            f"['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', "
                            f"'B09', 'B11', 'B12', None] ")

    # CONSIDER ONLY SENTINEL 2 L1C TILES TO BE LOADED
    elif path_tile.split('_')[1].endswith('1C'):
        # SET ONLY SPECIFIED BAND OF SPECIFIED RESOLUTION TO BE LOADED
        if band in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']:
            jp2s = [os.path.join(a, f) for f in os.listdir(a) if f.split('_')[2].startswith(band) and
                    resolutions_of_bands[(f.split('_')[2]).split('.')[0]] == channel]
        # SET ALL BANDS OF SPECIFIED RESOLUTION TO BE LOADED
        elif band is None:
            bands = [f for f in sorted(os.listdir(a), key=lambda x: x.split('_')[2]) if f.split('_')[2].startswith('B') \
                     and resolutions_of_bands[(f.split('_')[2]).split('.')[0]] == channel]
            jp2s = [os.path.join(a, f) for f in bands]
        else:
            raise Exception(f"Invalid 'band' parameter [{band}]. \n"
                            f"Possible values are\n"
                            f"['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', "
                            f"'B09', 'B10', 'B11', 'B12', None] ")

    # LOAD ALL SELECTED BANDS
    # for jp2 in tqdm(jp2s, desc="Loading bands"):
    for jp2 in jp2s:
        if return_reader:
            arrs.append(rasterio.open(jp2))
        else:
            with rasterio.open(jp2) as f:
                arrs.append(f.read(1))
    if return_reader:
        return arrs
    else:
        return np.array(arrs, dtype=arrs[0].dtype) if arrs else np.array(arrs)


def sentinel_load_clouds(path_tile, path_dataset=SENTINEL_PATH_DATASET, resolution='20m', mask_type='CLOUDS',
                         return_reader=False):
    """
    Loads mask in jp2 format from Sentinel 2  (L2A only) data.
    Possible masks to be loaded: CLOUDS, SNOW and CLASSIFICATION
    (Classification mask contains also cloud mask.)
    (See https://earth.esa.int/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm )

    Parameters
    ----------
    mask_type : str
        Specifies the type of mask to be loaded.
        Can be 'CLOUDS', 'SNOW' , 'CLASSIFICATION'
    path_dataset : str
        Specifies absolute path where are stored tiles to be loaded (SENTINEL_PATH_DATASET).
    path_tile : str
        Relative path in .SAFE format path of a tile.
    resolution : str
        Valid values are '20m', '60m'. Only for those are masks present.
    return_reader : bool
        Specifies if channel should be returned as raster reader in mode 'r'
        If false then numpy.ndarray is returned.
    Returns
    -------
    arrs : ndarray or list
        ndarray if return_reader parameter is set to False  otherwise returns [raster opened in mode 'r'] (list)
    """
    if mask_type not in ['CLOUDS', 'SNOW', 'CLASSIFICATION']:
        raise Exception(f"Invalid 'mask_type' parameter [{mask_type}]. \n"
                        f"Possible values are ['CLOUDS', 'SNOW' , 'CLASSIFICATION'].")
    if resolution not in ['20m', '60m']:
        raise Exception(f"Invalid 'mask_type' parameter [{resolution}]. \n"
                        f"Possible values are ['20m', '60m'].")

    mask_file = None
    arrs = []
    if path_tile.split('_')[1].endswith('L2A'):
        a = os.path.join(path_dataset, path_tile, 'GRANULE')
        if mask_type == 'CLASSIFICATION':
            a = os.path.join(a, os.listdir(a)[0], 'IMG_DATA', 'R' + resolution)
            msk = [f for f in os.listdir(a) if f.split('_')[2] == 'SCL']
            mask_file = os.path.join(a, msk[0])
        else:
            a = os.path.join(a, os.listdir(a)[0], 'QI_DATA')

        if mask_type == 'CLOUDS':
            mask_file = os.path.join(a, 'MSK_CLDPRB_' + resolution + '.jp2')
        elif mask_type == 'SNOW':
            mask_file = os.path.join(a, 'MSK_SNWPRB_' + resolution + '.jp2')

    if mask_file is not None:

        if return_reader:
            arrs.append(rasterio.open(mask_file, 'r'))
            return arrs
        else:
            with rasterio.open(mask_file) as f:
                arrs.append(f.read(1))
                return np.array(arrs, dtype=arrs[0].dtype) if len(arrs) > 0 else arrs


def sentinel_load_json(path_dataset, path_tile):
    """
    Loads the json of the specified tile.

    Parameters
    ----------
    path_dataset : str
        Specifies absolute path where are stored tiles to be loaded (SENTINEL_PATH_DATASET).
    path_tile : str
        Relative path in .SAFE format path of a tile.
    Returns
    -----------
    json_feed : dict
        Json response from sentinel query.
    """
    with open(os.path.join(path_dataset, path_tile)) as f:
        json_feed = json.load(f)

    return json_feed


def sentinel_get_tilebounds(path_tile, tiles_path=SENTINEL_PATH_DATASET):
    """
    Gets the bounding box of a specified tile.
    Works for Sentinel 2 (tiles) and also for Sentinel 1 (products).

    Parameters
    ----------
    path_tile : str
        Relative path in .SAFE format path of a tile.
    tiles_path: str
        absolute path to directory where are stored S2 tiles
    Returns
    -------
    bounds : numpy.array
        [left, bottom, right, top]
    """
    bounds = []

    # CALCULATE BOUNDS FOR SENTINEL 2 TILES
    if path_tile.startswith('S2'):
        path = os.path.join(tiles_path, path_tile, 'GRANULE')
        path = os.path.join(path, os.listdir(path)[0], 'IMG_DATA')

        path = os.path.join(path, 'R10m')
        jp2 = [os.path.join(path, f) for f in os.listdir(path) if f.split('_')[2].startswith('B')][0]

        bounds = np.array(rasterio.open(jp2).bounds).astype('int')

    # CALCULATE BOUNDS FOR SENTINEL 1 TILES
    elif path_tile.startswith('S1'):
        # FIRSTLY APPLY RE-PROJECTION TO EPSG:32633
        sentinel1_reproject_data(path_tile)

        path = os.path.join(tiles_path, path_tile, 'measurement')
        tif = [os.path.join(path, f) for f in os.listdir(path) if len(f.split('-')) == 10][0]

        bounds = np.array(rasterio.open(tif).bounds).astype('int')
    return bounds


def sentinel_load_mask(path_tile, where_to_export, path_dataset=SENTINEL_PATH_DATASET, mask_type='CLOUDS', out_resolution=20,
                       return_reader=False):
    """
    Loads specified mask for L1C or L2A and returns np.array representation of this mask in specified resolution
    (rows x columns)
    Every tile is square and should cover 109800 m^2 in UTM zone 33N, so e.g. for 20m resolution it is needed
    109800 / 20 rows and columns.
    CLOUDS or NODATA pixs are set to 1.0 other pixs to 0.0.

    Parameters
    ----------
    where_to_export: str
        Specifies path where to store generated masks.
    return_reader : bool
        Specifies if channel should be returned as raster reader in mode 'r'
        default is False
    path_tile : str
        Relative path in .SAFE format path of a tile.
    path_dataset : str
        Specifies absolute path where are stored tiles to be loaded (SENTINEL_PATH_DATASET).
    mask_type : str
        Can be 'CLOUDS', 'DEFECT', 'NODATA'
    out_resolution: int
        Specifies resolution of mask to be loaded.
        Can be 10 , 20 , 60 meant as 10m 20m and 60m
    Returns
    -------
    arr : ndarray or list
          (masked image) CLOUDS or NODATA pixs are set to 1.0 other pixs to 0.0.
          if return_reader parameter is set to False then returned is ndarray
          otherwise returns [raster opened in mode 'r'] (list)
    """

    a = os.path.join(path_dataset, path_tile, 'GRANULE')
    a = os.path.join(a, os.listdir(a)[0], 'QI_DATA')

    rows = int(109800 / out_resolution)
    name_tile = os.path.splitext(path_tile)[0]
    name = mask_type + '_mask_' + 'R' + str(out_resolution) + 'm'
    out_image = []

    # SET PATH WHERE SHOULD BE SAVED GENERATED MASK
    if not os.path.isdir(where_to_export):
        try:
            os.mkdir(where_to_export)
        except OSError:
            logging.info(f"Creation of the directory {where_to_export} failed")

    #  IF MASK IS NOT ALREADY GENERATED
    if not os.path.isfile(os.path.join(where_to_export, f'{name_tile}{"_" + name}.tif')):

        # POLYGONS REPRESENT CLOUDS, NODATA REGIONS ETC.
        polygons = []
        # COORDINATES OF MASK_TYPE PIXELS IN UTM 33N ARE IN POSLIST ELEMENT
        source = '{http://www.opengis.net/gml/3.2}posList'
        source2 = '{http://www.opengis.net/eop/2.0}maskType'

        if mask_type in ['CLOUDS', 'DEFECT', 'NODATA'] and out_resolution in [10, 20, 60]:
            masks = [os.path.join(a, mask) for mask in os.listdir(a) if mask.startswith('MSK_' + mask_type)]

            for mask_ in masks:
                tree = ET.parse(mask_)
                root = tree.getroot()

                pos_lists = [pos_list.text.split(' ') for pos_list in root.iter(source)]

                for i, pos in enumerate(pos_lists):
                    # IF NEEDED MASK IS CLOUDS THEN IN COORDINATES IS STORED ADDITIONAL INFO ABOUT TYPE OF CLOUD
                    if mask_type == 'CLOUDS':
                        coordinates = [geometry.Point(float(pos[2 * j]), float(pos[2 * j + 1])) for j
                                       in range(int(len(pos) / 2))]
                    # IN NODATA AND DEFECT MASK SI SOME ADDITIONAL INFO SO IT IS NEEDED TO 'JUMP' OVER 3RD ELEMENT
                    # TO OBTAIN ONLY PIXEL COORDINATES
                    else:
                        coordinates = [geometry.Point(float(pos[3 * j]), float(pos[3 * j + 1])) for j in
                                       range(int(len(pos) / 3))]

                    poly = geometry.Polygon([[p.x, p.y] for p in coordinates])
                    polygons.append(poly)

        else:
            raise Exception(f"Invalid 'mask_type' [{mask_type}] or 'out_resolution' [{out_resolution}] parameter. \n"
                            f"Possible values are \n"
                            f"['CLOUDS', 'DEFECT', 'NODATA'] \n"
                            f"[10, 20, 60]")

        # THERE IS NO 'MASK_TYPE' PIXELS SO RETURN NDARRAY FULL OF ZEROS
        if len(polygons) == 0:
            out_image = np.zeros((1, rows, rows),
                                 dtype=np.uint32)
        else:
            # REPRESENTS TILE COVERING 109800 M^2 IN 'OUT_RESOLUTION' RESOLUTION
            arr = np.ones((1, rows, rows),
                          dtype=np.uint32)
            # CREATES GEOTIFF FULL OF ONES BECAUSE RASTERIO.MASK.MASK REQUIRES RASTER READER IN 'R' MODE
            export_to_geotif(arr, path_tile, out_resolution='R' + str(out_resolution) + 'm',
                             where_to_export=where_to_export,
                             name=mask_type + '_mask_' + 'R' + str(out_resolution) + 'm')

            band = rasterio.open(os.path.join(where_to_export, f'{name_tile}{"_" + name}.tif'))
            out_image, out_transform = rasterio.mask.mask(band, polygons, filled=True)

            # CLOSE THE SOCKET
            band.close()

        # FINALLY EXPORT MASK TO GEOTIFF FORMAT
        export_to_geotif(out_image, path_tile, out_resolution='R' + str(out_resolution) + 'm',
                         where_to_export=where_to_export, name=mask_type + '_mask_' + 'R' + str(out_resolution) + 'm')
    arrs = []
    if return_reader:
        arrs.append(rasterio.open(os.path.join(where_to_export, f'{name_tile}{"_" + name}.tif')))
    else:
        out_image = rasterio.open(os.path.join(where_to_export, f'{name_tile}{"_" + name}.tif')).read()

    return out_image if not return_reader else arrs


def sentinel1_reproject_data(path_tile, tiles_path=SENTINEL_PATH_DATASET):
    """
    Applies reprojection to EPSG:32633 to Sentinel 1 tiffs.

    TODO consider use of rasterio.warp.reproject() (https://rasterio.readthedocs.io/en/latest/topics/reproject.html) \
         mainly for compatibility and some errors in readings of input SRS
    Parameters
    ----------
    path_tile : str
        Relative path in .SAFE format path of a tile (Sentinel 1).
    tiles_path: str
        absolute path to directory where are stored S1 tiles

    """
    a = os.path.join(tiles_path, path_tile, 'measurement')

    tifs_utm = [os.path.join(a, f) for f in os.listdir(a) if len(f.split('-')) == 10]
    tifs_not_reprojected = [(os.path.join(a, f), f) for f in os.listdir(a) if len(f.split('-')) == 9]

    # RE-PROJECTION IS APPLIED ONLY IF IT HAVE NOT BEEN APPLIED BEFORE
    if len(tifs_utm) != len(tifs_not_reprojected):
        for tif in tqdm(tifs_not_reprojected, desc='Applying re-projection to EPSG:32633'):
            im = rasterio.open(tif[0])
            w, h = im.width, im.height
            options = gdal.WarpOptions(format='GTiff', tps=True, width=w, height=h, resampleAlg='bilinear',
                                       srcSRS='EPSG:4326', dstSRS='EPSG:32633', dstNodata=0)
            out_name = tif[1].split('.')[0] + '-utm.tiff'
            gdal.Warp(os.path.join(a, out_name), tif[0], options=options)


def sentinel_load_measurement(path_tile, path_dataset=SENTINEL_PATH_DATASET, swath="1", polarisation=None,
                              return_reader=False):
    """
    Loads measurements from SENTINEL 1 data (similar to load_tile but instead of BAND there is POLARISATION).
    Currently works only with GRD and SLC in IW mode.

    Parameters
    ----------
    return_reader : bool
        Specifies if channel should be returned as raster reader in mode 'r'
        default is False
    polarisation : str or None
        Specifies the polarisation of particular Sentinel 1 product.
        Can be 'VV', 'VH', 'HH', 'HV' .
        If None then both polarisations will be loaded (it seems that HH, HV do not work)
        If 'tile' do not contain such polarisation ('VV', 'VH', 'HH', 'HV') simply empty array will be returned.
    swath : str
        Specifies the swath (need only if SLC product tile have been chosen).
        Can be '1', '2', '3'.
    path_dataset : str
        Specifies absolute path where are stored tiles to be loaded (SENTINEL_PATH_DATASET_S1).
    path_tile : str
        Relative path in .SAFE format path of a tile (Sentinel 1).
    Returns
    -------
        :ndarray if return_reader parameter is set to False  otherwise returns list of [raster opened in mode 'r'] (list)
    """

    a = os.path.join(path_dataset, path_tile, 'measurement')
    arrs = []
    tifs = []
    polaristation_in_tile = os.listdir(a)[0].split('-')[3]

    # APPLY REPROJECTION TO SENTINEL 1 TILES IF NOT ALREADY RE-PROJECTED
    sentinel1_reproject_data(path_tile)

    # CONSIDER ONLY SENTINEL 1 GRD TILES
    if path_tile.split('_')[2].startswith('GRD'):
        # SET ONLY PARTICULAR POLARISATION
        if polarisation in ['VV', 'VH', 'HH', 'HV'] and polaristation_in_tile[0] == polarisation.lower()[0]:
            tifs = [os.path.join(a, f) for f in os.listdir(a) if
                    f.split('-')[3] == polarisation.lower() and len(f.split('-')) == 10]
        # SET ALL POLARISATION PRESENT IN PARTICULAR PRODUCT
        elif polarisation is None:
            tifs = [os.path.join(a, f) for f in os.listdir(a) if len(f.split('-')) == 10]
        else:
            raise Exception(f"Invalid 'polarisation' parameter [{polarisation}]. \n"
                            f"Possible values are ['VV', 'VH', 'HH', 'HV', None]")

    # CONSIDER ONLY SENTINEL 1 SLC TILES
    elif path_tile.split('_')[2].endswith('SLC'):
        # SET ONLY PARTICULAR POLARISATION WITH THE PARTICULAR SWATH
        if polarisation in ['VV', 'VH', 'HH', 'HV']:
            tifs = [os.path.join(a, f) for f in os.listdir(a) if f.split('-')[3] == polarisation.lower() and
                    f.split('-')[1].endswith(swath)]
        # SET ALL POLARISATION PRESENT IN PARTICULAR PRODUCT WITH THE PARTICULAR SWATH
        elif polarisation is None:
            tifs = [os.path.join(a, f) for f in os.listdir(a) if f.split('-')[1].endswith(swath)]
        else:
            raise Exception(f"Invalid 'polarisation' parameter [{polarisation}]. \n"
                            f"Possible values are ['VV', 'VH', 'HH', 'HV', None]")

    # LOAD ALL SELECTED POLARISATIONS
    for tif in tqdm(tifs, desc="Loading polarisations"):
        if return_reader:
            arrs.append(rasterio.open(tif))
        else:
            with rasterio.open(tif) as f:
                arrs.append(f.read(1))
    if return_reader:
        return arrs
    else:
        return np.array(arrs, dtype=arrs[0].dtype) if arrs else np.array(arrs)


def sentinel_crop_shape(tile, shape, output_as_list=True, all_touched=True):
    """
    Trims tile to fit any polygon defined by shape (from shapefile)
    It wraps function mask from rasterio.mask module
    (https://rasterio.readthedocs.io/en/latest/api/rasterio.mask.html)
    TODO consider vectorization or another way to make this function faster, theoretically tile can be exported \
         to tiff with all needed bands and then use same code as below only with for loop (for time in tile:...)

    Parameters
    ----------
    all_touched : bool
        Include a pixel in the mask if it touches any of the shapes. Defaults True
    tile : np.ndarray or list
        [TIME x BAND_reader in mode 'r']
    shape : shapely.Polygon or simply shape from shapefile
        Defines the area to be trimmed.
    output_as_list : bool
        Specifies whether the output should be returned as a list (mainly when partial arrays have different shapes)
    Returns
    shape_trimmed : list or np.ma.MaskedArray
        If output_as_list is True otherwise np.ma.MaskedArray
        [TIME x BAND x X x Y]
    """
    shape_trimmed = []
    for time in tile:
        band_trimmed = []
        for band in time:
            out_image, out_transform = rasterio.mask.mask(band, [shape], crop=True,
                                                          all_touched=all_touched, filled=False)
            #  if nothing is masked out then it is needed to hardmask at least one pix to
            #  be possible of use of np.ma.array otherwise only np.array is created
            #  if such situation occurs then left top pix will be masked out
            if np.ma.count_masked(out_image) == 0:
                out_image.mask[0][0][0] = True
            band_trimmed.append(out_image)
        shape_trimmed.append(np.ma.concatenate(band_trimmed, axis=0))

    return shape_trimmed if output_as_list else np.ma.array(shape_trimmed)


def export_to_geotif(img, path_tile, where_to_export, path_dataset=SENTINEL_PATH_DATASET, out_resolution='R20m',
                     name='unknown', normalize=False, norm=cv2.NORM_MINMAX,
                     alpha=0, beta=10000, return_reader=False):
    """
    Exports calculated index to geotif image format.
    TODO consider to merge this function with function `merge_bands`
    Parameters
    ----------
    norm:
        Specifies the norm used to normalize.
        E.g. cv2.NORM_L1, cv2.NORM_L2, cv2.NORM_MINMAX
    beta: float
        Upper range boundary in case of the range normalization; it is not used for the norm normalization.
    alpha: float
        Norm value to normalize to or the lower range boundary in case of the range normalization.
    normalize : bool
        Specifies whether the exported index should be normalized.
        It will be easier for later processing.
    img : ndarray
        [BAND x X x Y], shape (1, X_dim, Y_dim)
    path_tile : str
        Relative path to tile ending with .SAFE.
    path_dataset: str
        Specifies absolute path where are stored tiles to be loaded (SENTINEL_PATH_DATASET).
    out_resolution : str
        Can be 'R20m', 'R10m', 'R60m'
    where_to_export : str
        Path where to save exported image.
    name : str
        Name to be added at the end of file name.
    return_reader: bool
        Specifies whether to return reader of generated raster
    Returns
    -------

    """
    if not os.path.isdir(where_to_export):
        try:
            os.mkdir(where_to_export)
        except OSError:
            logging.info(f"Creation of the directory {where_to_export} failed")

    if normalize:
        # PERFORM NORMALIZATION
        img[0] = cv2.normalize(img[0], None, alpha=alpha, beta=beta, norm_type=norm, dtype=cv2.CV_32F)
        img = img.astype('int16')

        # SET THE NAME OF TIFF FILE
        name_tile = os.path.splitext(path_tile)[0]  # GET RID OF 'SAFE' EXTENSION
        path_exported = os.path.join(where_to_export,
                                     name_tile + '_' + name + '_normalized_' + out_resolution + '.jp2')
    else:
        # SET THE NAME OF TIFF FILE
        name_tile = os.path.splitext(path_tile)[0]  # GET RID OF 'SAFE' EXTENSION
        path_exported = os.path.join(where_to_export, name_tile + '_' + name + '_' + out_resolution + '.jp2')

    # MERGE BANDS ONLY IF ALREADY NOT MERGED
    if not os.path.isfile(path_exported):
        # LOAD BAND TO GET WIDTH, HEIGHT, TRANSFROM ETC.
        bands = sentinel_load_channel(path_tile, channel=out_resolution, band='B03', return_reader=True,
                                      path_dataset=path_dataset)
        exported = rasterio.open(path_exported, 'w', driver='JP2OpenJPEG',
                                 width=bands[0].width, height=bands[0].height,
                                 count=len(bands),
                                 crs=bands[0].crs,
                                 transform=bands[0].transform,
                                 dtype='int16')

        # WRITE IMAGE TO TIFF FILE
        for i, band in enumerate(bands):
            exported.write(img[0], i + 1)

        # CLOSE OPENED SOCKETS
        exported.close()
        for band in bands:
            band.close()
    if return_reader:
        exported = rasterio.open(path_exported, 'r')
        return exported


def merge_bands(path_tile, where_to_export, resolution='R20m', path_dataset=SENTINEL_PATH_DATASET):
    """
    Merges bands of tile to one tiff raster.

    Parameters
    ----------
    resolution: str
        Specifies resolution of merged bands.
        Can be 'R20m', 'R10m', 'R60m'
    where_to_export: str
        Path where to save exported image.
    path_dataset: str
        Specifies path where are stored tiles: SENTINEL_PATH_DATASET
    path_tile: str
        Relative path to tile ending with .SAFE to be merged.
    Returns
    -------

    """
    if resolution not in ['R20m', 'R10m', 'R60m']:
        raise Exception(f"Invalid 'resolution' parameter [{resolution}]. \n"
                        f"Possible values are ['R20m', 'R10m', 'R60m']")
    # SET THE NAME OF TIFF FILE
    name_tile = os.path.splitext(path_tile)[0]  # GET RID OF 'SAFE' EXTENSION
    path_merged = os.path.join(where_to_export, name_tile + '_merged_' + resolution + '.tiff')

    # MERGE BANDS ONLY IF ALREADY NOT MERGED
    if not os.path.isfile(path_merged):
        bands = sentinel_load_channel(path_tile, channel=resolution, return_reader=True, path_dataset=path_dataset)
        merged = rasterio.open(path_merged, 'w', driver='Gtiff',
                               width=bands[0].width, height=bands[0].height,
                               count=len(bands),
                               crs=bands[0].crs,
                               transform=bands[0].transform,
                               dtype='uint16')
        for i, band in enumerate(bands):
            merged.write(band.read(1), i + 1)
        # CLOSE OPENED SOCKETS
        merged.close()
        for band in bands:
            band.close()


def sentinel_load_merged(path_tile, path_to_save, path_dataset=SENTINEL_PATH_DATASET,
                         resolution='R20m', return_reader=False):
    """
    Loads merged raster containing merged bands from particular tile.

    Parameters
    ----------
    path_to_save: str
        Specifies path where to save merged bands as tiff.
    return_reader: bool
        Specifies if channel should be returned as raster reader in mode 'r'
        Default is False
    path_tile: str
        Relative path to tile ending with .SAFE .
    path_dataset: str
        Specifies path where are stored merged tiffs. SENTINEL_PATH_DATASET
    resolution: str
        Specifies resolution of merged bands.
        Can be 'R20m', 'R10m', 'R60m'
    Returns
    -------
        : ndarray if return_reader is False otherwise rasterio reader in r mode
    """
    if resolution not in ['R20m', 'R10m', 'R60m']:
        raise Exception(f"Invalid 'resolution' parameter [{resolution}]. \n"
                        f"Possible values are ['R20m', 'R10m', 'R60m']")
    name_tile = os.path.splitext(path_tile)[0]  # GET RID OF 'SAFE' EXTENSION
    path_merged = os.path.join(path_to_save, name_tile + '_merged_' + resolution + '.tiff')

    # IF DOES NOT EXIST MERGED TIFF THEN CREATE IT
    if not os.path.isfile(path_merged):
        merge_bands(path_tile, path_dataset=path_dataset, resolution=resolution, where_to_export=path_to_save)

    if return_reader:
        return rasterio.open(path_merged)
    else:
        with rasterio.open(path_merged) as f:
            return f.read()


def sentinel2_overpasses(aoi=(19.59, 49.90, 20.33, 50.21), days_after=7,
                         export_csv=True,
                         api_key='081bd4280cce5a418890f6c27f9d85d5a21bde9f8fc30094335f2597467591e5',
                         satellites=None,
                         satellite_cycle=None):
    """
    Tracks Sentinel's overpasses over given AOI.
    (Reference API: https://api.spectator.earth/?language=Python#high-resolution-image)
    (See animation of Sentinel 2 coverage at
    https://www.esa.int/ESA_Multimedia/Videos/2016/08/Sentinel-2_global_coverage)
    (S2A 9 * 86 400 + 85904.543842969 s for one cycle (9.994265554 days) 9 days 23 hod 51 min 44.543844 s
    S2B 9.994297322 days)
    Parameters
    ----------
    api_key: str
        API key of user in https://spectator.earth/. Given to every registered user.
    days_after: int
        Specifies number of days (from current date) for which the prediction should be performed.
    aoi : tuple
        Should be one of following forms: (x_left y_bottom x_right y_top) -> bounding box of area of interest
                                          (longitude, latitude) -> point of interest
    export_csv: bool
        Whether export results to csv file.
    satellites: tuple
        Currently unsupported...
        Tuple of strings specifying names of satellites to be included.
        Other Sentinel missions are also included as well as Landsat missions. See  https://spectator.earth/
        for more information.
    satellite_cycle: int
        Currently unsupported... TODO duration of satellite cycle will be need if this function will be used with
                                    another mission e.g. Sentinel 1 etc.
    Returns
    -------
    final_frame: pandas.Dataframe
        Contains info about date of overpass, satellite, acquisition status (if available) and coordinates of
        corresponding point within trajectory of satellite nearest to queried area of interest
    """
    now = datetime.now()
    time_delta = timedelta(days=days_after)
    to_date = now + time_delta
    sentinel_cycle = timedelta(days=10)  # sentinel 2 duration of one cycle (consists of 143 orbits)
    periods = math.ceil(days_after / 10)

    if len(aoi) == 4:
        geo = Polygon([[(aoi[0], aoi[1]), (aoi[2], aoi[1]), (aoi[2], aoi[3]), (aoi[0], aoi[3])]])
    elif len(aoi) == 2:
        geo = Point([aoi[0], aoi[1]])
    else:
        raise Exception("AOI does not follow expected form")

    satellites = ','.join(('Sentinel-2A', 'Sentinel-2B'))
    # it would be enough to query just first 10 days and other just replicate, but never minds
    url = 'https://api.spectator.earth/overpass/?api_key={api_key}&geometry={geometry}&satellites={satellites}' \
          '&days_after={days_after}&days_before={days_before}'.format(
        api_key=api_key, geometry=geo, satellites=satellites, days_after=days_after, days_before=0)

    response = requests.get(url)
    data = response.json()
    overpasses = data['overpasses']

    list_of_overpasses = []
    for o in overpasses:
        list_of_overpasses.append([o['date'], o['satellite'], o['acquisition'],
                                   o['geometry']['coordinates'][0], o['geometry']['coordinates'][1]])
    df_of_overpasses = pd.DataFrame(list_of_overpasses, columns=["date", "satellite", "acquisition",
                                                                 "longitude", "latitude"])

    # convert strings to datetime object
    df_of_overpasses['date'] = df_of_overpasses['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ"))

    replicated_overpasses = []
    for i in range(periods):
        c = df_of_overpasses.copy(deep=True)
        replicated_overpasses.append(c)
        df_of_overpasses['date'] = df_of_overpasses['date'].apply(lambda x: x + sentinel_cycle)
        df_of_overpasses['acquisition'] = df_of_overpasses['acquisition'].apply(lambda x: None)

    final_data = pd.concat(replicated_overpasses, ignore_index=True)
    final_data = final_data.sort_values('date', ignore_index=True)
    final_data = final_data.drop_duplicates(subset=['date'], ignore_index=True)
    final_data = final_data.loc[final_data['date'] <= to_date]

    if export_csv:
        final_data.to_csv('overpasses.csv', encoding='utf-8', index=False)
    return final_data


def time_series_s2(count, producttype='S2MSI1C', path_to_save=SENTINEL_PATH_DATASET):
    """
    Auxiliary function for downloading time series of Sentinel-2 tiles
    See config file to set specific CLOUD percentages, TILES and DATES
    Parameters
    ----------
    count:int
    producttype: str
        One of `S2MSI1C` or S2MSI2A

    Returns
    -------

    """
    ###########################################
    # DOWNLOAD L1C TIME SERIES DATA           #
    ###########################################
    # DOWNLOAD SENTINEL 2 TILES
    for cloud, date in zip(CLOUDS, DATES):
        for tile in TILES:
            try:
                # DOWNLOAD L1C because in 2018 there was lack of L2A tiles and also we can use most recent version
                sentinel(polygon=None, tile_name=tile, platformname='Sentinel-2', producttype=producttype, count=count,
                         beginposition=date,
                         cloudcoverpercentage=f'[0 TO {cloud}]', path_to_save=path_to_save)
            except RuntimeError as e:
                logging.info(e.__str__())
                logging.info(f'Skipping date:tile {date}:{tile}')
                continue
