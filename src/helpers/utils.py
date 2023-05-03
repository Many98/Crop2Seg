from pyproj import Proj, Transformer, transform
import numpy as np


def progress_bar(j, count, size=50, prefix=""):
    x = int(size * j / count)
    if j == count:
        print(f'{prefix} [{u"█" * x}{"." * (size - x)}]', flush=True)
    elif j < count:
        print(f'{prefix} [{u"█" * x}{"." * (size - x)}]', end="\r", flush=True)


def distribute_args(iterable, num_cpus):
    s = int(len(iterable) / num_cpus)
    args = [[i * s, (i + 1) * s] for i in range(num_cpus)]
    if (len(iterable)) % num_cpus != 0:
        args[-1][1] = len(iterable)

    return args


def transform_from_crs(left, bottom, right, top, src_crs="EPSG:4326", dst_crs="EPSG:32633"):
    # transformer = Transformer.from_crs(src_crs, dst_crs)
    #  insert as array of x-coordinates, array of y cordinates (longitude (x) then latitude (y))
    # transformed = transformer.transform([left, right], [bottom, top])
    # above method not working so we will use deprecated method

    p_src = Proj(init=src_crs, preserve_units=False)
    p_dst = Proj(init=dst_crs, preserve_units=False)

    x1, y1 = p_src((left, right), (bottom, top))
    x2, y2 = transform(p_src, p_dst, x1, y1)

    return {'left': x2[0], 'bottom': y2[0],
            'right': x2[1], 'top': y2[1]}


def UTMtoWGS(vyrez):
    """Converts UTM zone 33N to WGS84.
    EPSG:32633 UTM zone 33N 
    EPSG:4326 WGS84 - World Geodetic System 1984; used in GPS

    Parameters
    vyrez : np.ndarray
        [[bottom, left], [top, right]]
    Returns
        [[left, bottom], [right, top]]
    """
    transformer = Transformer.from_crs("EPSG:32633", "EPSG:4326")
    a = transformer.transform(vyrez[:, 0], vyrez[:, 1])
    return np.transpose(np.array(a))  # TODO this joke needs to be done in another way (at least np.flip via axis=0)


def WGStoUTM(vyrez):
    """Converts WGS84 to UTM zone 33N.
    EPSG:32633 UTM zone 33N 
    EPSG:4326 WGS84 - World Geodetic System 1984; used in GPS

    Parameters
    vyrez : np.ndarray
        [[bottom, left], [top, right]]
    Returns
        [[left, bottom], [right, top]]
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633")
    a = transformer.transform(vyrez[:, 0], vyrez[:, 1])

    return np.transpose(np.array(a))  # Longitude, Latitude  TODO and this joke as well (at least np.flip via axis=1)

def get_row_col(patch_id: int, size: int = 82):
    """
    Auxiliary function to get row and column indices of
    patch based on its id.
    Parameters
    patch_id: int
        id of patch within tile
    size: int
        Size of one side of tile (in number of pixels)
    Returns
        (row_index, col_index)
    """
    return patch_id // size, patch_id % size

def calc_subtile_id(patch_id: int, size_subtile: int=6, size_tile: int = 82, size_border: int=2):
    """
    Auxiliary function to get id of subtile based on patch_id within tile.
    It is used to better distribute patches to train/val/test sets 
    Parameters
    patch_id: int
        id of patch within tile
    size_subtile: int
        Size of tile (in number of subtiles) which should partition whole tile
    size_tile: int
        Size of one side of tile (in number of pixels)
    size_border: int
        Size of border (in number of patches) used to separate created subtiles.
    Returns
        id of subtile, if patch_id is on patch creating border it will return -1
    """
    num_borders = size_subtile - 1

    assert (size_tile - (num_borders * size_border) % size_subtile) == 0, 'Provided sizes are not compatible'

    row, col = get_row_col(patch_id) 

    pass