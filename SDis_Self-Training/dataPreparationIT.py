# Main Script for data preparation -------------------------------------------------------------------------------------
# imports
import itertools
import os

import geopandas as gpd
import numpy as np
import pandas as pd

import gdalutils
from config.definitions import ROOT_DIR, ancillary_path, engine, pop_path, cur,conn

#-------- SELECT PROCESS --------
print(pop_path)
#-------- PROCESS: GHS RPREPARATION --------
getBasicSHP = "yes"
init_ghs = "no"
raster_file = 'C:/FUME/PopNetV2/data_prep/rom_ProjectData/temp_tif/province/corine/water_rom_CLC_2012_2018.tif'
init_population = "no"
init_templates = "no"

city ='rom'

def process_data():
    if getBasicSHP == "yes":
        year_list = [ 2015, 2016, 2017, 2018, 2019, 2020]
        for year in year_list:
            df = gpd.read_file("C:/FUME/PopNetV2/data_prep/rom_ProjectData/PopData/CensusTractsMunicipality/{}_komCensusTracts.geojson".format(year))
            df = df.rename(columns={'Census tracts_new':'CT_CODE', 'Total population ':'totalpop', 'Foreigners':'foreigners', 'Italians':'ita', 'EU':'EU', 'Europe (non EU)':'EurNotEU', 
                                    'Italians - 0-5':'ita0-5', 'Italians - 6-19':'ita6-19', 'Italians - 20-29':'ita20-29', 'Italians - 30-44':'ita30-44', 'Italians - 45-64':'ita45-64', 'Italians - 65-79':'ita65-79', 'Italians – 80+':'ita80+', 
                                    'Foreigners - 0-5':'mig0-5', 'Foreigners - 6-19':'mig6-19', 'Foreigners - 20-29':'mig20-29', 'Foreigners - 30-44':'mig30-44', 'Foreigners - 45-64':'mig45-64', 'Foreigners - 65-79':'mig65-79', 'Foreigners - 80+':'mig80+'})
            df['CT_CODE'] = df['CT_CODE'].astype('str')
            df['notEU'] = df['foreigners'] - df['EU']
            df['children'] = df['ita0-5'] + df['ita6-19'] + df['mig0-5'] + df['mig6-19']
            df['students'] = df['ita20-29'] + df['mig20-29'] 
            df['mobadults']= df['ita30-44'] + df['mig30-44'] 
            df['nmobadults']= df['ita45-64'] + df['mig45-64'] 
            df['elderly']= df['ita65-79'] + df['ita80+'] + df['mig65-79'] + df['mig80+']
            print(df.columns.to_list())
            # CAREFUL HERE BC THE ITALIAN INCLUDE BIG VALUES NOT ASSIGNED TO TRACTS!!!
            df.dropna(subset=['geometry'], inplace=True)
            df = df.fillna(0)
            df.to_file(ROOT_DIR + '/Shapefiles/{1}/Merged/{0}_{1}.shp'.format(year,city), driver='ESRI Shapefile', crs= "EPSG:3035")
            gdf = df[['CT_CODE','geometry']]
            
            gdf.to_file(ROOT_DIR + '/Shapefiles/{1}/{0}_{1}.shp'.format(year,city), driver='ESRI Shapefile', crs= "EPSG:3035")
            df = df.drop(columns='geometry')
            df.to_csv(ROOT_DIR + '/Statistics/{1}/{0}_{1}.csv'.format(year,city))
            
    if init_ghs == "yes":
        years=[1990, 2000, 2015]
        #-------- PROCESS: GHS RPREPARATION --------
        for year in years:
            projAlgorithms =['near']
            resampleAlgorithms =['cubicspline']
            for projAlg in projAlgorithms:
                inputfile = ROOT_DIR + '/Rasters/GHS/RomeKrakow/{0}_GHS_POP.tif'.format(year)
                temp_outputfile = ROOT_DIR + '/Rasters/GHS/RomeKrakow/{0}_GHS_POP_250_{1}.tif'.format(year,projAlg)
                gdalutils.reprojectionGDAL(inputfile, temp_outputfile, projAlg)
                for resampleAlg in resampleAlgorithms:
                
                    temp_outputfileVRT = ROOT_DIR + '/Rasters/GHS/RomeKrakow/{0}_GHS_POP_100_{1}_{2}.vrt'.format(year,projAlg,resampleAlg)
                    outputfile = ROOT_DIR + '/Rasters/{3}_GHS/{0}_GHS_POP_100_{1}_{2}.tif'.format(year,projAlg,resampleAlg,city)
                
                    gdalutils.resampleGDAL(temp_outputfile, temp_outputfileVRT, 100, 100, resampleAlg)
                    gdalutils.vrt2tifGDAL(raster_file, temp_outputfileVRT, outputfile)
    if init_population == "yes":
        print(1)

    if init_templates == "yes":
        year_list = [2015, 2016, 2017, 2018, 2019, 2020]
        for year in year_list:
            input = ROOT_DIR + '/Shapefiles/{1}/{0}_{1}.shp'.format(year,city)
            
            src = gpd.read_file(input)
            if 'index' not in src.columns:
                src=src.reset_index()
                src.to_file(input, driver='ESRI Shapefile', crs= "EPSG:3035")
            outputfile = ROOT_DIR + '/Rasters/{1}_template/{0}_{1}_template_100.tif'.format(year,city)
            gdalutils.shp2tifGDAL(raster_file, input, outputfile, 100, 100, attribute=None, burnValues=1)
            
            outputfileNeigh = ROOT_DIR + '/Rasters/{1}_template/{0}_{1}_templateNeigh_100.tif'.format(year, city)
            gdalutils.shp2tifGDAL(raster_file, input, outputfileNeigh, 100, 100, attribute='index', burnValues=None)
        
    

process_data()
