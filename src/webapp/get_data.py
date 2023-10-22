# ############
# SCRIPT CONTAINING ALL RELEVANT FUNCTIONS FOR LOADING/DOWNLOADING SENTINEL-2 TIME SERIES BASED ON USER INPUT
# ############

import os
from src.helpers.dataset_creator import DatasetCreator
from src.global_vars import ACCOUNT, PASSWORD

def get_info(x, y):
    """
    returns tilename, patch_id, patch_tranform, patch_bounds
    """

    return 'T33UWR', 5, None, [500000, 5500000, 501280, 5501280]


def get_ts(tilename, bounds, year=2022, cache=False, account=ACCOUNT, password=PASSWORD):

    temp_dir = f'data/tmp_{tilename}_{year}'
    download = True

    s2_data_dir = os.path.join(temp_dir, 'DATA_S2')

    if os.path.isdir(s2_data_dir):
        if len(os.listdir(s2_data_dir)) != 0:
            download = False

    os.makedirs(temp_dir, exist_ok=True)

    # TODO user should have properly set config mainly credentials for S2 API

    dc = DatasetCreator(output_dataset_path=temp_dir, features_path='',
                        tiles_path=s2_data_dir,
                        for_inference=True,
                        download=download,
                        delete_source_tiles=False if cache else True)
    tiles = [tilename]
    clouds = [65, 65, 65, 65, 65, 35, 65, 20, 55, 15, 25, 55, 45, 35]
    dates = [
            f'[{year-1}-09-01T00:00:00.000Z TO {year-1}-09-30T00:00:00.000Z]',
            f'[{year-1}-10-01T00:00:00.000Z TO {year-1}-10-31T00:00:00.000Z]',
            f'[{year-1}-11-01T00:00:00.000Z TO {year-1}-11-30T00:00:00.000Z]',
            f'[{year-1}-12-01T00:00:00.000Z TO {year-1}-12-31T00:00:00.000Z]',
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
            ]


    try:
        time_series, dates = dc(tile_names=tiles, clouds=clouds, dates=dates, bounds=bounds, account=account,
                                password=password)
    except Exception as e:
        if str(e) == 'Unauthorized access to Opensearch API!':
            return None, 401
        else:
            raise Exception('Unknown problem occured when obtaining data')

    # TODO at the end delete all create files/dirs

    return time_series, dates

if __name__ == '__main__':

    get_ts('T33UWR', [500000, 5500000, 501280, 5501280], year=2020, cache=False,
           account='', password='')
