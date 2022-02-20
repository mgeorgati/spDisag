import os

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
parent_path = os.path.dirname(ROOT_DIR) 
city="cph"
year=2018
ancillary_path = parent_path + '/AncillaryData/{0}'.format(city)
pop_path = os.path.join(parent_path + "/{}_ProjectData/PopData/".format(city))
#path to folder with gdal executable files
gdal_rasterize_path = r'~/anaconda3/envs/spdisag_env/Library/bin'
python_scripts_folder_path = r'~/anaconda3/envs/spdisag_env/spdisag_env/Scripts'

temp_shp= ancillary_path + "/temp_shp/"
temp_tif= ancillary_path + "/temp_tif/"



