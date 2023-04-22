import configparser
import json
import os

current_dir = os.path.abspath(os.curdir)
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # this will ensure to always start from src directory

conf = configparser.ConfigParser()
conf.read('../config/config.ini')

os.chdir(current_dir)
################################
# PARAMETERS USED TO           #
# DOWNLOADING TILES            #
# FROM ODATA-API/OPENSERCH API #
################################
ODATA_URI = conf['sentinel']['odata_uri']
ODATA_RESOURCE = conf['sentinel'][
    'odata_resource']  # FOR MORE RESOURCES SEE https://scihub.copernicus.eu/userguide/ODataAPI
OPENSEARCH_URI = conf['sentinel']['opensearch_uri']
ACCOUNT = conf['sentinel']['account']
PASSWORD = conf['sentinel']['password']

#########################################
# PATHS WHERE ARE STORED SENTINEL TILES #
#########################################
SENTINEL_PATH_DATASET = conf['sentinel']['path_dataset']  # WHERE ARE STORED SENTINEL 2 TILES

#####################################
# PATHS WHERE ARE STORED SHAPEFILES #
#####################################
AGRI_PATH_DATASET = conf['agri']['path_dataset']  # SHAPEFILE OF AGRICULTURE FIELDS (ONLY)

####################
# SET RANDOM STATE #
####################
RANDOM_STATE = 42

############
# EPSILON  #
############
EPS = 1e-10


##################################################
# TILES USED WHEN DOWNLOADING FROM BIGGER AREA   #
##################################################
TILES = [i.replace(' ', '').replace('\n', '') for i in conf['tiles']['tiles'].split(',')]

###############################
# DATES USED FOR TIME SERIES  #
###############################
DATES = [i.replace(' ', '').replace('\n', '').replace('TO', ' TO ') for i in conf['dates']['dates'].split(',')]

##############
CLOUDS = [int(i.replace(' ', '').replace('\n', '')) for i in conf['clouds']['clouds'].split(',')]


###############
SEN2COR = conf['sen2cor']['sen2cor']
