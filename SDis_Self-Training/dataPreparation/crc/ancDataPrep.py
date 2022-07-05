# Main Script for data preparation -------------------------------------------------------------------------------------
# imports
from utils import gdalutils, osgu
from mainFunctions import zonalStat, shptoraster
from config import ancillary_path, python_scripts_folder_path, gdal_rasterize_path
import rasterio, os, subprocess, glob
import numpy as np

city ='crc'
        
            
def ancData_crc(init_esm, init_ghs, process_ghs, year): 
    if year < 2000 :
        raster_file = ancillary_path + "/{0}/corine/waterComb_{0}_CLC_1990_2000.tif".format(city)
    elif year >= 2000 and year < 2006 :
        raster_file = ancillary_path + "/{0}/corine/waterComb_{0}_CLC_2000_2006.tif".format(city)
    elif year >= 2006 and year < 2012 :
        raster_file = ancillary_path + "/{0}/corine/waterComb_{0}_CLC_2006_2012.tif".format(city)
    elif year >= 2012 :
        raster_file = ancillary_path + "/{0}/corine/waterComb_{0}_CLC_2012_2018.tif".format(city) 
    
    if init_esm == "yes":
        #Merge and reproject ESM files
        esmFiles= glob.glob( ancillary_path + "/euroData/ESM/2015/{0}/*.tif".format(city))
        projAlg = 'near'
        esmFile = ancillary_path + "/{0}/ESM/ESM_merged.tif".format(city)
        gdalutils.reprojectionGDAL(esmFiles, esmFile, projAlg)  

        #Clip to extent
        outputfile = ancillary_path + "/{0}/ESM/crc_ESM.tif".format(city)
        gdalutils.vrt2tifGDAL(raster_file, esmFile, outputfile)
        destination_raster_path = ancillary_path + "/{0}/ESM/esm_residential_10.tif".format(city)

        # Select residential cells
        print("------------------------------ Splitting ESM raster to categories: Residential------------------------------")
        cmds = 'python {2}/gdal_calc.py -A "{0}" --A_band=1 --outfile="{1}" --calc="(A==255)"'.format(outputfile, destination_raster_path, python_scripts_folder_path)
        print(cmds)
        subprocess.call(cmds, shell=True)
        
        #Aggregate to 100m grid cells
        dstFile = ancillary_path + "/{0}/ESM/{0}_residCellCount.geojson".format(city)
        polyPath = "C:/Users/NM12LQ/OneDrive - Aalborg Universitet/PopNetV2_backup/data_prep/{0}_ProjectData/temp_shp/{0}_grid.geojson".format(city)
        zonalStat(destination_raster_path, dstFile, polyPath, 'sum')
        
        #Export to final raster file
        dst_file = ancillary_path + "/{0}/ESM/{0}_residential.tif".format(city)
        shptoraster(raster_file, dstFile, gdal_rasterize_path, dst_file, 'sum_', xres=100, yres=100)
         
    if init_ghs == "yes":
            projAlgorithms =['near']
            resampleAlgorithms =['cubicspline']
            for projAlg in projAlgorithms:
                inputfile = ancillary_path + '/euroData/GHS/{1}/GHS_POP_E{0}_GLOBE_R2019A_54009_250_V1_0_19_3.tif'.format(year,city)
                temp_outputfile = ancillary_path + '/euroData/GHS/{2}/{0}_GHS_POP_250_V1_0_19_3_{1}.tif'.format(year,projAlg,city)
                gdalutils.reprojectionGDAL(inputfile, temp_outputfile, projAlg)
                for resampleAlg in resampleAlgorithms:
                
                    temp_outputfileVRT =  ancillary_path + '/euroData/GHS/{3}/{0}_GHS_POP_100_V1_0_19_3_{1}_{2}.vrt'.format(year,projAlg,resampleAlg,city)
                    outputfile = ancillary_path + '/{3}/GHS/{0}_GHS_POP_100_{1}_{2}.tif'.format(year,projAlg,resampleAlg,city)
                
                    gdalutils.resampleGDAL(temp_outputfile, temp_outputfileVRT, 100, 100, resampleAlg)
                    gdalutils.vrt2tifGDAL(raster_file, temp_outputfileVRT, outputfile)
    
    if process_ghs == "yes":
        # Make Inverted prcentage of water coverage
        src = rasterio.open(raster_file)
        ancdataset, rastergeo = osgu.readRaster(raster_file)
        slice22 = src.read(1)
        slice22[slice22 < 0] = np.nan
        neo = (1 - slice22)*100
        neo[neo < 0] = np.nan
        osgu.writeRaster(neo[:,:], rastergeo, ancillary_path + "/{0}/corine/{1}_waterComb_{0}_Inverted.tif".format(city, year))
        
        water = rasterio.open(ancillary_path + "/{0}/corine/{1}_waterComb_{0}_Inverted.tif".format(city, year))
        water_array = water.read(1)
        # Read ESM and add factor of 1
        esm_file = ancillary_path + "/{0}/ESM/{0}_residential.tif".format(city)
        esm = rasterio.open(esm_file)
        esm_array = esm.read(1)
        # Read GHS
        projAlg ='near'  #,'bilinear','cubic','cubicspline','lanczos','average','mode','rms', 'max', 'min','med', 'q1','q3', 'sum'
        resampleAlg ='cubicspline'
        outputfile = ancillary_path + '/{3}/GHS/{0}_GHS_POP_100_{1}_{2}.tif'.format(year,projAlg,resampleAlg,city)
        ghs = rasterio.open(outputfile)
        ghs_array = ghs.read(1)
        # Multiply GHS with ESM and Water coverage
        array = ghs_array * (water_array/100) * ((esm_array/100)+1)
        array[array < 0] = np.nan
        osgu.writeRaster(array[:,:], rastergeo, ancillary_path + "/{0}/GHS/{1}_GHS_POP_100_processed.tif".format(city, year))


        

    """
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
    """    