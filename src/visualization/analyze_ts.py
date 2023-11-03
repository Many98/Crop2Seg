import torch

from src.helpers.dataset_creator import DatasetCreator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import pandas as pd

from src.visualization.visualize import plot_rgb, plot_lulc, plot_ndvi
from src.learning.utils import get_dilated
from src.datasets.s2_ts_cz_crop import crop_cmap


# https://eos.com/industries/agriculture/ndre/
# https://eos.com/industries/agriculture/msavi/
# https://custom-scripts.sentinel-hub.com/sentinel-2/evi/
# https://en.wikipedia.org/wiki/Soil-adjusted_vegetation_index

def get_date(date):
    """
    transform string date to datetime.datetime date
    """
    return datetime(int(str(date)[:4]), int(str(date)[4:6]), int(str(date)[6:]))


def ndvi_ts(ts: np.ndarray, red_edge: bool = False):
    """
    generates time series of ndvi
    Parameters
    ----------
    ts: np.ndarray
        4D array [T x C x H x W]
        representing time-series of Sentinel-2 images
    red_edge: bool
        Whether calculate red edge NDVI index
    """
    if red_edge:
        nir = ts[:, 3, ...].astype(np.float16)
        red_edge = ts[:, 4, ...].astype(np.float16)
        return (nir - red_edge) / (nir + red_edge)
    else:
        nir = ts[:, 3, ...].astype(np.float16)
        red = ts[:, 0, ...].astype(np.float16)
        return (nir - red) / (nir + red)


def ts_profile(ndvi_ts: np.ndarray, segmentation_mask: np.ndarray, class_code: int, stat: str = 'mean',
               q: float = 0.75):
    """
    calculates temporal ndvi profile
    Parameters
    ----------
    ndvi_ts: np.ndarray
        3D array [T x H x W]
        representing time-series of ndvi index
    segmentation_mask: np.ndarray
        2D array [H x W]
        with class codes
    class_code: int
        class code which should be analyzed
    stat: str
        name of statistic to be used. Can be one of `mean`, 'std', 'median', 'quantile'
    q: float
        optional quantile value if used `stat = 'quantile'`
    Returns
    -------
    returns matplotlib.pyplot.Figure object
    """
    nd = np.where(segmentation_mask == class_code, ndvi_ts, np.nan).astype(np.float32)

    if stat == 'mean':
        prof = np.nanmean(nd.reshape(nd.shape[0], -1), axis=-1)
    elif stat == 'std':
        prof = np.nanstd(nd.reshape(nd.shape[0], -1), axis=-1)
    elif stat == 'median':
        prof = np.nanmedian(nd.reshape(nd.shape[0], -1), axis=-1)
    elif stat == 'quantile':
        prof = np.nanquantile(nd.reshape(nd.shape[0], -1), q=q, axis=-1)
    else:
        raise Exception('Unsupported stat')

    return prof, np.where(prof != 0.)[0]


def plot_profile(ndvi_ts: np.ndarray, dates: list, c: list, segmentation: np.ndarray, smooth: int = 3):
    """
    plot ndvi temporal profile
    Parameters
    ----------
    ndvi_ts: np.ndarray
        3D array [T x H x W]
        representing time-series of ndvi index
    dates: list
        list of dates used for indexing e.g. ['20190820', '20190825'...]
    c: list
        list of class codes which should be analyzed
    segmentation: np.ndarray
        2D array [H x W]
        with class codes
    smooth: int
        smoothing constant used in moving average
    Returns
    -------
    returns matplotlib.pyplot.Figure object
    """
    labels_super_short_2 = ['Background', 'Grassland', 'Fruit/vegetable', 'Summer cereals',
                            'Winter cereals', 'Rapeseed', 'Maize', 'Forage crops', 'Sugar beet', 'Flax/Hemp',
                            'Permanent fruit', 'Hopyards', 'Vineyards', 'Other crops', 'Not classified']
    sns.set_style("whitegrid")
    fig, ax = plt.subplots()

    for cc in c:
        ts_prof, mask = ts_profile(ndvi_ts, segmentation, cc, stat='mean')
        date_index = pd.DatetimeIndex(data=np.array(dates)[mask]).values

        # date_index = date_index.to_series().loc[lambda x: (x > '2019-02-25') & (x < '2019-10-05')].index.values
        ts_prof_q, mask_q = ts_profile(ndvi_ts, segmentation, cc, stat='std')

        df = pd.DataFrame(data={'date': date_index, 'mean': ts_prof[mask], 'std': ts_prof_q[mask_q]})
        df[f'mean_{smooth}'] = df['mean'].rolling(smooth).mean()
        df[f'std_{smooth}'] = df['std'].rolling(smooth).mean()
        df = df[(df.date > '2019-02-25') & (df.date < '2019-10-05')]

        ax.plot(df['date'].values, df[f'mean_{smooth}'].values, label=labels_super_short_2[cc],
                # color=crop_cmap()[c],
                marker='+')
        ax.fill_between(df['date'].values, df[f'mean_{smooth}'].values - 2 * df[f'std_{smooth}'].values,
                        df[f'mean_{smooth}'].values + 2 * df[f'std_{smooth}'].values,
                        alpha=0.2  # , color=crop_cmap()[c]
                        )
    ax.set_ylabel('Smoothed NDVI')
    # plt.xlim(120, 400)
    plt.xticks(rotation=25)
    plt.ylim(-0.1, 1)
    plt.legend()

    return fig


if __name__ == '__main__':

    d = DatasetCreator(output_dataset_path='')

    time_series, bbox, affine, crs, file_names, dates = d._load_s2(
        'T33UVR')  # , bounds=[400000, 5500000, 440280, 5540280])
    time_series = d._preprocess(time_series)
    segmentation_mask = d._create_segmentation(time_series.shape[-2:], affine, bbox)

    dilated = get_dilated(torch.from_numpy(segmentation_mask)[None, ...], 15, 'cpu', 4).numpy()[0]

    segmentation_mask = np.where(dilated.sum(0) > 1, 14, segmentation_mask)  # reclass boundaries to not classified

    ndvi = ndvi_ts(time_series, red_edge=False)
    sns.set_style("whitegrid")

    ref_date = get_date('20180901')

    plot_profile(ndvi, dates, [3, 4, 5, 8], segmentation_mask)
    plt.show()

    plot_ndvi(ndvi[27, 3000:3128, 3000:3128])
    plt.show()

    labels_super_short_2 = ['Background', 'Grassland', 'Fruit/vegetable', 'Summer cereals',
                            'Winter cereals', 'Rapeseed', 'Maize', 'Forage crops', 'Sugar beet', 'Flax/Hemp',
                            'Permanent fruit', 'Hopyards', 'Vineyards', 'Other crops', 'Not classified']

    plot_rgb(time_series[27, [0, 1, 2], 3000:3128, 3000:3128])
    plt.show()

    plot_lulc(segmentation_mask[3000:3128, 3000:3128], labels_super_short_2, crop_cmap())
    plt.show()
