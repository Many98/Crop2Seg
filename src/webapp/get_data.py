# ############
# SCRIPT CONTAINING ALL RELEVANT FUNCTIONS FOR LOADING/DOWNLOADING SENTINEL-2 TIME SERIES BASED ON USER INPUT
# ############

import os
import zipfile
import shutil
from functools import reduce
import git
import requests
import numpy as np
import datetime
import geopandas as gpd
from shapely.geometry import box as shapely_box
from shapely.geometry import Polygon
from shapely.prepared import prep
from src.helpers.dataset_creator import DatasetCreator
from src.global_vars import ACCOUNT, PASSWORD
import streamlit as st


def get_s2_shape(prefix: str):
    """
    downloads git repo with Sentinel-2 grid shapefile
    https://github.com/justinelliotmeyers/Sentinel-2-Shapefile-Index
    """
    os.makedirs(os.path.join(prefix, 'data/s2_grid/'), exist_ok=True)
    git.Git(os.path.join(prefix, 'data/s2_grid/')).clone(
        "https://github.com/justinelliotmeyers/Sentinel-2-Shapefile-Index.git")


def grid_bounds(geom, parts=6):
    minx, miny, maxx, maxy = round(geom.bounds[0]), round(geom.bounds[1]), round(geom.bounds[2]), round(geom.bounds[3])
    gx, gy = np.linspace(minx, maxx, parts), np.linspace(miny, maxy, parts)
    grid = []
    for i in range(len(gx) - 1):
        for j in range(len(gy) - 1):
            poly_ij = Polygon([[gx[i], gy[j]], [gx[i], gy[j + 1]], [gx[i + 1], gy[j + 1]], [gx[i + 1], gy[j]]])
            grid.append(poly_ij)
    return grid


def partition(geom, parts=6):
    prepared_geom = prep(geom)
    grid = list(filter(prepared_geom.intersects, grid_bounds(geom, parts)))
    return grid


def generate_grid(prefix: str):
    if not os.path.isfile(
            os.path.join(prefix, 'data/s2_grid/Sentinel-2-Shapefile-Index/sentinel_2_index_shapefile.shp')):
        get_s2_shape(prefix)
    if not os.path.isfile(os.path.join(prefix, 'data/s2_grid/grid.shp')):
        grid_s2 = gpd.read_file(
            os.path.join(prefix, 'data/s2_grid/Sentinel-2-Shapefile-Index/sentinel_2_index_shapefile.shp'),
            bbox=shapely_box(11.8, 48.48, 18.9, 51.12))
        grid_s2 = grid_s2[grid_s2.Name.isin(['33UVS', '33UWS', '33UUR', '33UVR', '33UWR', '33UXR',
                                             '33UYR', '33UUQ', '33UVQ', '33UWQ', '33UXQ', '33UYQ'])]
        grid_s2 = grid_s2.to_crs(32633)

        pols = []
        tiles = []
        for _, row in grid_s2.iterrows():
            pols.append(partition(row['geometry']))
            tiles.append([row['Name']] * 25)
        grid = gpd.GeoDataFrame(data={'tile': list(reduce(lambda x, y: x + y, tiles, []))},
                                geometry=list(reduce(lambda x, y: x + y, pols, [])), crs='EPSG:32633')
        grid.to_file(os.path.join(prefix, 'data/s2_grid/grid.shp'), driver='ESRI Shapefile')


def get_LPIS(prefix, year):
    """
    downloads lpis (Czech republic) data for year
    https://eagri.cz/public/portal/mze/farmar/LPIS/export-lpis-rocni-shp
    """
    assert year > 2014 and year < datetime.date.today().year, 'Year must be > 2014 and < current year'

    os.makedirs(os.path.join(prefix, 'data/lpis/'), exist_ok=True)
    if not os.path.isfile(f"{os.path.join(prefix, 'data/lpis/')}{year}1231-CR-DPB-SHP.zip") and \
            not os.path.isfile(f"{os.path.join(prefix, 'data/lpis/')}{year}1231-CR-DPB-SHP.shp"):
        try:
            with requests.get(f'https://eagri.cz/public/app/eagriapp/LpisData/{year}1231-CR-DPB-SHP.zip',
                              stream=True) as r:
                total_size_in_bytes = int(r.headers.get('content-length', 0))
                block_size = 1024
                # progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                my_bar = st.progress(0, text=f'Downloading vector data:  {0}%')
                done = 0
                with open(f"{os.path.join(prefix, 'data/lpis/')}{year}1231-CR-DPB-SHP.zip", mode="wb") as f:
                    # for chunk in stqdm(r.iter_content(chunk_size=block_size), total=total_size_in_bytes, unit='kB',
                    #                   unit_scale=True):
                    for chunk in r.iter_content(chunk_size=block_size):
                        done += block_size
                        # progress_bar.update(len(chunk))
                        my_bar.progress(min((done / total_size_in_bytes), 1.0),
                                        text=f'Downloading vector data: {round(min((done / total_size_in_bytes), 1.0) * 100, 1)}%')
                        f.write(chunk)
                # progress_bar.close()
        except Exception as e:
            print(e)
            raise Exception(f'Download error: LPIS <{year}>')

    if os.path.isfile(f"{os.path.join(prefix, 'data/lpis/')}{year}1231-CR-DPB-SHP.zip") and \
            not os.path.isfile(f"{os.path.join(prefix, 'data/lpis/')}{year}1231-CR-DPB-SHP.shp"):

        st.write('Unzipping LPIS data ...')
        # CHECK IF ZIP IS TREATED AS ZIP (ZIPFILE MODULE DID NOT WORK ON ALL ZIPPED TILES)
        if zipfile.is_zipfile(f"{os.path.join(prefix, 'data/lpis/')}{year}1231-CR-DPB-SHP.zip"):
            with zipfile.ZipFile(f"{os.path.join(prefix, 'data/lpis/')}{year}1231-CR-DPB-SHP.zip", "r") as zip_ref:
                zip_ref.extractall(f"{os.path.join(prefix, 'data/lpis/')}")

        # OTHERWISE USE SHUTIL MODULE
        else:
            try:
                shutil.unpack_archive(f"{os.path.join(prefix, 'data/lpis/')}{year}1231-CR-DPB-SHP.zip",
                                      extract_dir=f"{os.path.join(prefix, 'data/lpis/')}")
            except Exception:
                os.remove(f"{os.path.join(prefix, 'data/lpis/')}{year}1231-CR-DPB-SHP.zip")
                raise Exception(f"File {year}1231-CR-DPB-SHP.zip is damaged. Try to download it again.")
        try:
            # workaround to get proper prj file (which is not contained within downloaded LPIS data)
            shutil.copy(f"data/inference/YYYY1231-CR-DPB-SHP.prj",
                        f"{os.path.join(prefix, 'data/lpis/')}{year}1231-CR-DPB-SHP.prj")
        except:
            pass

        # finally remove zip file
        if not st.session_state['cache_enabled']:
            os.remove(f"{os.path.join(prefix, 'data/lpis/')}{year}1231-CR-DPB-SHP.zip")


def get_info(index):
    """
    returns tilename, patch_bounds
    """
    grid = gpd.read_file('src/webapp/data/s2_grid/grid.shp')
    grid = grid.reset_index()

    patch = grid[grid['index'] == index]
    patch = patch.to_crs(32633)

    return patch['tile'].values[0], patch.total_bounds


def get_ts(tilename, patch_id, bounds, year, start_dt, end_dt, account, password, cache=False):
    temp_dir = f'src/webapp/data/s2_patches/{tilename}_{patch_id}_{year}'
    if not os.path.isdir(temp_dir):
        try:
            shutil.rmtree('src/webapp/data/s2_patches/')
        except:
            pass

    download = True

    s2_data_dir = f'src/webapp/data/s2_tiles/tmp_{tilename}_{year}'

    # if os.path.isdir(s2_data_dir):
    #    if len(os.listdir(s2_data_dir)) != 0:
    #        download = False

    os.makedirs(s2_data_dir, exist_ok=True)
    # TODO user should have properly set config mainly credentials for S2 API

    dc = DatasetCreator(output_dataset_path=temp_dir, features_path='',
                        tiles_path=s2_data_dir,
                        for_inference=True,
                        download=download,
                        delete_source_tiles=False if cache else True)
    tiles = ['T' + tilename]
    clouds = np.array([65, 65, 65, 65, 65, 45, 65, 65, 55, 55, 65, 65, 45, 45])
    dates = np.array([
        f'[{year - 1}-09-01T00:00:00.000Z TO {year - 1}-09-30T00:00:00.000Z]',
        f'[{year - 1}-10-01T00:00:00.000Z TO {year - 1}-10-31T00:00:00.000Z]',
        f'[{year - 1}-11-01T00:00:00.000Z TO {year - 1}-11-30T00:00:00.000Z]',
        f'[{year - 1}-12-01T00:00:00.000Z TO {year - 1}-12-31T00:00:00.000Z]',
        f'[{year}-01-01T00:00:00.000Z TO {year}-01-31T00:00:00.000Z]',
        f'[{year}-02-01T00:00:00.000Z TO {year}-02-28T00:00:00.000Z]',
        f'[{year}-03-01T00:00:00.000Z TO {year}-03-31T00:00:00.000Z]',
        f'[{year}-04-01T00:00:00.000Z TO {year}-04-30T00:00:00.000Z]',
        f'[{year}-05-01T00:00:00.000Z TO {year}-05-31T00:00:00.000Z]',
        f'[{year}-06-01T00:00:00.000Z TO {year}-06-30T00:00:00.000Z]',
        f'[{year}-07-01T00:00:00.000Z TO {year}-07-31T00:00:00.000Z]',
        f'[{year}-08-01T00:00:00.000Z TO {year}-08-31T00:00:00.000Z]',
        f'[{year}-09-01T00:00:00.000Z TO {year}-09-30T00:00:00.000Z]',
        f'[{year}-10-01T00:00:00.000Z TO {year}-10-31T00:00:00.000Z]'
    ])
    dates = dates[start_dt:end_dt]
    clouds = clouds[start_dt:end_dt]
    try:
        dates = dc(tile_names=tiles, clouds=clouds, dates=dates, bounds=bounds, account=account,
                   password=password)
    except Exception as e:
        if str(e) == 'Unauthorized access to Opensearch API!':
            return None, 401
        else:
            clouds = np.array([85, 85, 65, 65, 65, 65, 65, 75, 75, 85, 85, 85, 85, 85])
            st.warning(f'Problem occured when downloading S2 data. \n'
                       f'Rerunning with less restrictive cloud filter')
            st.error(f'Exception: {e}')
            try:
                dates = dc(tile_names=tiles, clouds=clouds, dates=dates, bounds=bounds, account=account,
                           password=password)
            except Exception as e:
                st.error(f'Error occured. \n'
                         f'Please create issue at https://github.com/Many98/Crop2Seg/issues')
                st.error(f'Exception: {e}')
                return
    # TODO at the end delete all create files/dirs if not required caching

    return dates


if __name__ == '__main__':
    '''
    get_ts('T33UWR', [500000, 5500000, 501280, 5501280], year=2020, cache=False,
           account='', password='')
    '''
    generate_grid('src/webapp')
