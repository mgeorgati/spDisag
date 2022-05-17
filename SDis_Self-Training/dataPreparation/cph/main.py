# Main Script for data preparation -------------------------------------------------------------------------------------
# imports
import itertools
import os, sys, subprocess
import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
import osgeoutils as osgu
import gdalutils
from config.definitions import ROOT_DIR, python_scripts_folder_path, gdal_rasterize_path
from dataPrepDK.dataPrep import restructureData
from evaluateFunctions import zonalStat
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.mainFunctions.format_conversions import shptoraster
city ='cph'

#-------- SELECT PROCESS --------

#-------- PROCESS: GHS RPREPARATION --------
init_ghs = "no"
raster_file = 'C:/FUME/PopNetV2/data_prep/{0}_ProjectData/temp_tif/{0}_CLC_2012_2018.tif'.format(city)
init_population = "yes"
init_templates = "no"
init_esm = "no"
process_ghs = "no"

year=2018
 
def process_data():
    if init_ghs == "yes":
        #-------- PROCESS: GHS RPREPARATION --------
        projAlgorithms =['near' ] #,'bilinear','cubic','cubicspline','lanczos','average','mode','rms', 'max', 'min','med', 'q1','q3', 'sum'
        resampleAlgorithms =['cubicspline'] #'nearest','bilinear','cubic','lanczos','average','mode'
        for projAlg in projAlgorithms:
            inputfile = ROOT_DIR + '/Rasters/GHS/GHS_POP_E2015_GLOBE_R2019A_54009_250_V1_0_18_2.tif'
            temp_outputfile = ROOT_DIR + '/Rasters/{0}_GHS/GHS_POP_250_{1}.tif'.format(city, projAlg)
            gdalutils.reprojectionGDAL(inputfile, temp_outputfile, projAlg)
            for resampleAlg in resampleAlgorithms:
            
                temp_outputfileVRT = ROOT_DIR + '/Rasters/{2}_GHS/GHS_POP_100_{0}_{1}.vrt'.format(projAlg,resampleAlg,city)
                outputfile = ROOT_DIR + '/Rasters/{2}_GHS/GHS_POP_100_{0}_{1}.tif'.format(projAlg,resampleAlg, city)
            
                gdalutils.resampleGDAL(temp_outputfile, temp_outputfileVRT, 100, 100, resampleAlg)
                gdalutils.vrt2tifGDAL(raster_file, temp_outputfileVRT, outputfile)
    if init_population == "yes":
        restructureData(ROOT_DIR, city, year)
        
    if init_templates == "yes":
        input = ROOT_DIR + '/Shapefiles/{1}/{0}_{1}.shp'.format(year,city)
        src = gpd.read_file(input)
        if 'index' not in src.columns:
            src=src.reset_index()
            src.to_file(input, driver='ESRI Shapefile', crs= "EPSG:3035")
        
        outputfile = ROOT_DIR + '/Rasters/{0}_template/{0}_template_100.tif'.format(city)
        gdalutils.shp2tifGDAL(raster_file,city, input, outputfile, 100, 100, attribute=None, burnValues=1)
    
    if init_esm == "yes":
        ancillary_data_folder_path =  "C:/Users/NM12LQ/OneDrive - Aalborg Universitet/DasymetricMapping/cph_ProjectData/AncillaryData/ESM/ESM2015"
        source_raster_path = ancillary_data_folder_path + "/Data/ESM_BUILT_VHR2015CLASS_R2019_3035_N36E44_010m_v010.tif"
        destination_raster_path = ancillary_data_folder_path + "/esm_residential_10.tif"
    
        print("------------------------------ Splitting ESM raster to categories: Residential------------------------------")
        cmds = 'python {2}/gdal_calc.py -A "{0}" --A_band=1 --outfile="{1}" --calc="(A==255)"'.format(source_raster_path, destination_raster_path, python_scripts_folder_path)
        print(cmds)
        subprocess.call(cmds, shell=True)
        
        dstFile = ancillary_data_folder_path + "/{0}_residCellCount.geojson".format(city)
        polyPath = "C:/Users/NM12LQ/OneDrive - Aalborg Universitet/PopNetV2_backup/data_prep/cph_ProjectData/temp_shp/cph_grid.geojson"
        zonalStat("{0}/esm_residential_10.tif".format(ancillary_data_folder_path), dstFile, polyPath, 'sum')
        
        dst_file = ancillary_data_folder_path + "/{0}_residential.tif".format(city)
        shptoraster(raster_file, dstFile, gdal_rasterize_path, dst_file, 'sum_', xres=100, yres=100)
        
    if process_ghs == "yes":
        ancillary_data_folder_path =  "C:/Users/NM12LQ/OneDrive - Aalborg Universitet/DasymetricMapping/cph_ProjectData/AncillaryData/ESM/ESM2015"
        # Make Inverted prcentage of water coverage
        src = rasterio.open("C:/FUME/PopNetV2/data_prep/cph_ProjectData/temp_tif/corine/waterComb_cph_CLC_2012_2018.tif")
        ancdataset, rastergeo = osgu.readRaster("C:/FUME/PopNetV2/data_prep/cph_ProjectData/temp_tif/corine/waterComb_cph_CLC_2012_2018.tif")
        slice22 = src.read(1)
        slice22[slice22 < 0] = np.nan
        neo = (1 - slice22)*100
        neo[neo < 0] = np.nan
        #osgu.writeRaster(neo[:,:], rastergeo, "C:/FUME/PopNetV2/data_prep/cph_ProjectData/temp_tif/corine/waterComb_cph_CLC_2012_2018Inverted.tif")
        
        water = rasterio.open("C:/FUME/PopNetV2/data_prep/cph_ProjectData/temp_tif/corine/waterComb_cph_CLC_2012_2018Inverted.tif")
        water_array = water.read(1)
        # Read ESM and add factor of 1
        esm_file = ancillary_data_folder_path + "/{0}_residential.tif".format(city)
        esm = rasterio.open(esm_file)
        esm_array = esm.read(1)
        # Read GHS
        projAlg ='near'  #,'bilinear','cubic','cubicspline','lanczos','average','mode','rms', 'max', 'min','med', 'q1','q3', 'sum'
        resampleAlg ='cubicspline'
        outputfile = ROOT_DIR + '/Rasters/{2}_GHS/GHS_POP_100_{0}_{1}.tif'.format(projAlg,resampleAlg, city)
        ghs = rasterio.open(outputfile)
        ghs_array = ghs.read(1)
        # Multiply GHS with ESM and Water coverage
        array = ghs_array * (water_array/100) * ((esm_array/100)+1)
        array[array < 0] = np.nan
        osgu.writeRaster(array[:,:], rastergeo, ROOT_DIR + '/Rasters/{2}_GHS/GHS_POP_100_near_cubicsplineWaterIESM_new.tif'.format(projAlg,resampleAlg, city))

process_data()
