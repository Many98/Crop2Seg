import shapefile
import os
import numpy as np
import logging

from helpers.utils import WGStoUTM
from global_vars import AGRI_PATH_DATASET


def agri_load(path_dataset=AGRI_PATH_DATASET):
    """
    Loads the shapefile.

    Parameters
    ----------
    path_dataset : str
        Path to the folder containing shapefile (.dbf, .shp and .shx)
    
    Returns
    ----------
    r : shapefile.Reader
    """
    path_agrishapefile = []
    for agrishapefile in sorted(os.listdir(path_dataset)):
        # READ SHAPEFILES ALPHABETICALLY
        if agrishapefile.endswith(('.dbf', '.shp', '.shx')):
            path_agrishapefile.append(agrishapefile)

    r = shapefile.Reader(dbf=open(os.path.join(path_dataset, path_agrishapefile[0]), "rb"),
                         shp=open(os.path.join(path_dataset, path_agrishapefile[1]), "rb"),
                         shx=open(os.path.join(path_dataset, path_agrishapefile[2]), "rb"))

    return r


def agri_trim(reader, vyrez, path_dataset=AGRI_PATH_DATASET,
              shapefile_name=os.path.join('trimmed', 'trimmed')):
    """
    Saves the trimmed shapefile based on vyrez which is meant as Area of interest.

    Parameters
    ----------
    reader: shapefile.Reader
        Reader containing the original shapefile
    vyrez : np.ndarray
        [[bottom, left], [top, right]] TODO bit confusing it should be [left, bottom, right, top]
    path_dataset : str
        Path to the folder containing shapefile (.dbf, .shp and .shx)
    shapefile_name : str
        Name of the trimmed shapefile.
    """

    # CONVERT TO STANDARD COORDINATE SYSTEM ("EPSG:4326" -> "EPSG:32633") UTM 33N
    vyrez = WGStoUTM(vyrez)

    w = shapefile.Writer(os.path.join(path_dataset, shapefile_name))
    w.fields = reader.fields[1:]  # skip first deletion field

    for info in reader.iterShapeRecords():
        rectangle = info.shape.bbox
        if vyrez[0, 0] < rectangle[0] and vyrez[0, 1] < rectangle[1] and vyrez[1, 0] > rectangle[2] and vyrez[1, 1] > \
                rectangle[3]:
            w.record(*info.record)
            w.shape(info.shape)
    w.close()

    pass


def agri(vyrez=np.array([[49.9, 14.6], [50.1, 14.7]])):
    """
    Creates new shapefile within given Area of interest.

    Parameters
    ----------
    vyrez : np.ndarray [[bottom, left], [top, right]]
        values are in EPSG:4326

    Returns
    -------
    r_trimmed: shapefile.Reader
    """
    r = agri_load(AGRI_PATH_DATASET)
    logging.info(f"Pocet poli: {len(r.shapeRecords())}")

    agri_trim(r, vyrez, shapefile_name=os.path.join('trimmed', 'trimmed'))

    r_trimmed = agri_load(os.path.join(AGRI_PATH_DATASET, 'trimmed'))
    logging.info(f"Pocet poli po vyrezu: {len(r_trimmed.shapeRecords())}")

    return r_trimmed

