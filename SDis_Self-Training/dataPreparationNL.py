# Main Script for data preparation -------------------------------------------------------------------------------------
# imports
import itertools
import os
import subprocess

import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import pandas as pd

import gdalutils
from config.definitions import ROOT_DIR, ancillary_path,  engine, pop_path, cur,conn, gdal_rasterize_path
from dataPrep.dataPrepCBS import getStatisticsCBS, plotCBS, popDataPrepCBS
from dataPrep.dataPrepOIS import popDataPrepOIS, getStatisticsOIS, plotOISg, compOISn_OISg, plot_dif, restructureData
from dataPrep.CBS_OIS import joinCBS_OIS, getStatisticsCBS_OIS, computePopByArea
#-------- SELECT PROCESS --------
print(pop_path)
#-------- PROCESS: GHS RPREPARATION --------
init_ghs = "no"
raster_file = ancillary_path + '/temp_tif/grootams_CLC_2012_2018.tif'
init_population = "no"
init_templates = "no"
clip_data = "yes"
city ='ams'
year=2018
popraster = 'GHS_POP_100_near_cubicsplineWaterIESM_new.tif'
 
def process_data():
    if init_ghs == "yes":
        #-------- PROCESS: GHS RPREPARATION --------
        projAlgorithms =['near' ,'bilinear','cubic','cubicspline','lanczos','average','mode','rms', 'max', 'min','med', 'q1','q3', 'sum']
        resampleAlgorithms =['nearest' ,'bilinear','cubic','cubicspline','lanczos','average','mode']
        for projAlg in projAlgorithms:
            inputfile = ROOT_DIR + '/Rasters/GHS/GHS_POP_E2015_GLOBE_R2019A_54009_250_V1_0_18_2.tif'
            temp_outputfile = ROOT_DIR + '/Rasters/GHS/GHS_POP_250_{}.tif'.format(projAlg)
            gdalutils.reprojectionGDAL(inputfile, temp_outputfile, projAlg)
            for resampleAlg in resampleAlgorithms:
            
                temp_outputfileVRT = ROOT_DIR + '/Rasters/GHS/GHS_POP_100_{0}_{1}.vrt'.format(projAlg,resampleAlg)
                outputfile = ROOT_DIR + '/Rasters/GHS/GHS_POP_100_{0}_{1}.tif'.format(projAlg,resampleAlg)
            
                gdalutils.resampleGDAL(temp_outputfile, temp_outputfileVRT, 100, 100, resampleAlg)
                gdalutils.vrt2tifGDAL(raster_file, temp_outputfileVRT, outputfile)
    if init_population == "yes":
        print(1)
        #popDataPrepCBS(year, cur,conn, engine, ROOT_DIR, city, pop_path, ancillary_path)
        print(2)
        #getStatisticsCBS(year, ROOT_DIR, city)
        print(3)
        #plotCBS(year, ROOT_DIR, city,ancillary_path, engine)
        print(4)
        #popDataPrepOIS(year,ROOT_DIR, city, pop_path)
        print(5)
        #getStatisticsOIS(year, ROOT_DIR)
        print(6)
        #joinCBS_OIS(year, city, pop_path)
        #computePopByArea(cur, conn, engine)
        #getStatisticsCBS_OIS(year, ROOT_DIR, city, ancillary_path )
        #plotOISg(year, city, ancillary_path, pop_path, ROOT_DIR)
        #compOISn_OISg(year, city, pop_path, cur, conn, engine, ancillary_path)
        #plot_dif(ROOT_DIR,city, year, ancillary_path)
        restructureData(pop_path, raster_file, gdal_rasterize_path , city, year, popraster)
    if init_templates == "yes":
        input = ROOT_DIR + '/Shapefiles/{0}_{1}.shp'.format(year,city)
        
        src = gpd.read_file(input)
        if 'index' not in src.columns:
            src=src.reset_index()
            src.to_file(input, driver='ESRI Shapefile', crs= "EPSG:3035")
        
        if not os.path.isfile(ROOT_DIR + '/Shapefiles/{}_ams.shp'.format(year)):
            df = src.loc[src['BU_CODE'].str.contains('BU0363')]
            df.to_file(ROOT_DIR + '/Shapefiles/{}_ams.shp'.format(year), driver='ESRI Shapefile', crs= "EPSG:3035")
        
        amsPath = ROOT_DIR + '/Shapefiles/{}_ams.shp'.format(year)
        outputfile = ROOT_DIR + '/Rasters/template/{}_template_100.tif'.format(city)
        gdalutils.shp2tifGDAL(raster_file, input, outputfile, 100, 100, attribute=None, burnValues=1)
        
        outputfileNeigh = ROOT_DIR + '/Rasters/template/{}_templateNeigh_100.tif'.format(city)
        gdalutils.shp2tifGDAL(raster_file, input, outputfileNeigh, 100, 100, attribute='index', burnValues=None)
        
        outputfile = ROOT_DIR + '/Rasters/template/ams_template_100.tif'
        gdalutils.shp2tifGDAL(raster_file, amsPath, outputfile, 100, 100, attribute=None, burnValues=1)        
        
        outputfileNeigh = ROOT_DIR + '/Rasters/template/ams_templateNeigh_100.tif'
        gdalutils.shp2tifGDAL(raster_file, amsPath, outputfileNeigh, 100, 100, attribute='index', burnValues=None)
    if clip_data =='yes':
        print("------------------------------ Clipping Corine rasters by extent of case study area ------------------------------")
        temp_shp_path = 'C:/Users/NM12LQ/OneDrive - Aalborg Universitet/DasymetricMapping/ams_ProjectData/AncillaryData/temp_shp'
        temp_tif_path = 'C:/Users/NM12LQ/OneDrive - Aalborg Universitet/DasymetricMapping/ams_ProjectData/AncillaryData/constr_year'
        
        bboxPath = temp_shp_path + "/{}_bbox.geojson".format(city)
        for file in os.listdir(temp_tif_path):
            if file.endswith('fillnodata100.tif'):
                name = file.split('_')[1]
                print(name)
                cmds = '{3}/gdalwarp.exe -of GTiff -t_srs EPSG:3035 -cutline "{0}" -crop_to_cutline -nosrcalpha "{1}/{2}" "C:/FUME/project/AncillaryData/ams/temp_tif/{2}"'.format(bboxPath, temp_tif_path, file, gdal_rasterize_path,name)
                print(cmds)
                subprocess.call(cmds, shell=True)
        
    

process_data()
