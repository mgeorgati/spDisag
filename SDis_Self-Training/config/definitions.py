import os

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
parent_path = os.path.dirname(ROOT_DIR) 

year = 2018
ancillary_path = parent_path + '/AncillaryData/'
pop_path = "C:/FUME/Dasymetric_Mapping/GroundTruth/"
#path to folder with gdal executable files
gdal_rasterize_path = r'/home/ubuntu/anaconda3/envs/spdisag_env/bin'
python_scripts_folder_path = r'~/anaconda3/envs/spdisag_env/spdisag_env/Scripts'

temp_shp= ancillary_path + "/temp_shp/"
temp_tif= ancillary_path + "/temp_tif/"
