import glob
import os
import subprocess
import sys
from pathlib import Path

import attr
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
from osgeo import gdal

import osgeoutils as osgu
from evaluateFunctions import (percentage_error ,div_error, mae_error, nmae_error, nrmse_error,
                               prop_error, rmse_error)
from mainFunctions.basic import zonalStat, createFolder
from gdalutils import maskRaster
from plotting.plotRaster import plot_map
from plotting.plotVectors import plot_mapVectorPolygons

"""
    EVALUATION OF THE PREDICTIONS WITH THE GROUND TRUTH DATA OF DSTg DATASET
"""
def eval_Results_GC(ROOT_DIR, pop_path, ancillary_path, year, city, attr_value):
    evalPath = ROOT_DIR + "/Evaluation/{}/".format(city)

    # Required files
    actualPath = ROOT_DIR + "/Evaluation/{1}_groundTruth/{0}_{1}_{2}.tif".format(year,city,attr_value)
    raster_file = ancillary_path + '/{0}_template/{0}_template_100.tif'.format(city)
    polyPath = ROOT_DIR + "/Shapefiles/{1}/{0}_{1}.shp".format(year,city)
    districtPath = "K:/FUME/demo_popnet-Commit01/data_prep/cph_ProjectData/AncillaryData/CaseStudy/sogn.shp"
    waterPath = ancillary_path + '/corine/waterComb_cph_CLC_2012_2018.tif'.format(city)
   
    aggr_outfileSUMGT = ROOT_DIR + "/Shapefiles/Comb/{0}_{1}.geojson".format(year,city)
    ##### -------- PLOT Ground truth at Grid Cells -------- #####
    print("----- Plotting Population Distribution -----") 
    print("----- Ground Truth for Copenhagen at Grid Cells -----") 
    templatePathGA = raster_file
    #ds,rastergeo  = osgu.readRaster(actualPath)
    # Clip GT to extent of Municipality 
    data = gdal.Open(templatePathGA)
    geoTransform = data.GetGeoTransform()
    minx = geoTransform[0] 
    maxy = geoTransform[3] 
    maxx = minx + geoTransform[1] * data.RasterXSize 
    miny = maxy + geoTransform[5] * data.RasterYSize
    data = None
    bbox = (minx,maxy,maxx,miny)
    outputGTL =  ROOT_DIR + "/Evaluation/{1}_groundTruth/{0}_{1}_{2}.tif".format(year,city,attr_value)
    if not os.path.exists(outputGTL):
        gdal.Translate(outputGTL, actualPath, projWin=bbox)

    outputGT =  ROOT_DIR + "/Evaluation/{1}_groundTruth/{0}_{1}_{2}.tif".format(year,city,attr_value)
    #if not os.path.exists(outputGT):
        #maskRaster(polyPath, outputGTL, outputGT)
    
    exportPath = evalPath + "/{0}_GT_{1}.png".format(city,attr_value)
    if not os.path.exists(exportPath):
        title ="Population Distribution (persons)\n(Ground Truth: {})(2018)".format(attr_value)
        LegendTitle = "Population (persons)"
        src = rasterio.open(outputGT)
        plot_map(city,'popdistribution', src, exportPath, title, LegendTitle, districtPath = polyPath, neighPath = districtPath , waterPath = waterPath, invertArea = None, addLabels=True)
    
    ##### -------- Get files to be processed -------- #####
    print("----- Get prediction files to be evaluated -----") 
    # Get all spatial data for neighborhoods in list
    evalFiles = []
    # All files ending with .shp with depth of 2 folder
    evalFiles1 = glob.glob(ROOT_DIR + "/Results/{0}/apcatbr/*{0}_*_{1}.tif".format(city,attr_value))
    evalFiles2 = glob.glob(ROOT_DIR + "/Results/{0}/aprf/*{0}_*_{1}.tif".format(city,attr_value))
    evalFiles3 = glob.glob(ROOT_DIR + "/Results/{0}/Dasy/*_{0}_{1}_dasyWIN.tif".format(city,attr_value))
    evalFiles4 = glob.glob(ROOT_DIR + "/Results/{0}/Pycno/*_{0}_{1}_pycno.tif".format(city,attr_value))
    evalFiles5 = glob.glob(ROOT_DIR + "/Results/{0}/GHS/*.tif".format(city))
    evalFiles.extend(evalFiles1)
    evalFiles.extend(evalFiles2)
    evalFiles.extend(evalFiles3)
    #evalFiles.extend(evalFiles4)
    #evalFiles.extend(evalFiles5)
    print(evalFiles)
    print("----- {} files to be evaluated -----".format(len(evalFiles))) 

    
    ##### -------- PLOT PREDICTIONS for Greater Copenhagen at Grid Cells -------- #####
    print("----- Plotting Population Distribution -----") 
    print("----- ", len(evalFiles), "files for Greater Copenhagen at Grid Cells -----") 
    evalPathGA = ROOT_DIR + "/Evaluation/{0}_groundTruth/".format(city)
    for k in evalFiles:
        path = Path(k)
        name = path.stem 
        exportPath = evalPathGA + "/{}.png".format(name)
        if not os.path.exists(exportPath):
            title ="Population Distribution (persons)\n({})(2018)".format(name)
            LegendTitle = "Population (persons)"
            src = rasterio.open(path)
            plot_map(city,'popdistribution', src, exportPath, title, LegendTitle, districtPath = polyPath , neighPath = districtPath, waterPath = waterPath, invertArea = None, addLabels=True)
    
    filenamemetrics2e = evalPath + '/{0}_Evaluation_{1}.csv'.format(city, attr_value)
    if os.path.exists(filenamemetrics2e):
        os.remove(filenamemetrics2e)

    ##### -------- Process Evaluation: Steps -------- #####
    print("----- Plotting Population Distribution -----") 
    print("----- Ground Truth for Copenhagen at Grid Cells -----")      
    for file in evalFiles:
        path = Path(file)
        fileName = path.stem
        method = fileName.split("_",1)[1]
        print(fileName)
        # Clip the predictions to AMS extent 
        input = file
        if 'aprf' in fileName:
            if 'dissever00' in fileName:
                outputPath = evalPath + "/aprf/dissever00"
                createFolder(outputPath)
            else:
                outputPath = evalPath + "/aprf/dissever01"
                createFolder(outputPath)
        elif 'apcatbr' in fileName:
            outputPath = evalPath + "/apcatbr"
            createFolder(outputPath)
        elif 'GHS' in fileName:
            outputPath = evalPath + "/GHS"
            createFolder(outputPath)
        elif fileName.endswith('_dasyWIN'):
            outputPath = evalPath + "/Dasy"
            createFolder(outputPath)
        elif fileName.endswith('_pycno'):
            outputPath = evalPath + "/Pycno"
            createFolder(outputPath)
        else: 
            outputPath = evalPath
            
        outputfile = outputPath + "/{0}_{1}.tif".format(city,fileName)
        #if not os.path.exists(outputfile):
           # maskRaster(polyPath, input, outputfile)
        # Plot the population distribution of the predictions 
        exportPath = outputPath + "/{0}_{1}.png".format(city,fileName)
        if not os.path.exists(exportPath):
            print("----- Step #1: Plotting Population Distribution -----")
            title ="Population Distribution (persons)\n({})(2018)".format(fileName)
            LegendTitle = "Population (persons)"
            src = rasterio.open(path)
            plot_map(city,'popdistribution', src, exportPath, title, LegendTitle, districtPath =  polyPath, neighPath = districtPath , waterPath = waterPath, invertArea = None, addLabels=True)
        
        print("----- Step #2: Calculating Metrics -----")
        src_real = rasterio.open(outputGT)
        actual = src_real.read(1)
        
        src_pred = rasterio.open(input)
        predicted = src_pred.read(1)
        predicted[(np.where((predicted <= -100000)))] = np.nan
        predicted = np.nan_to_num(predicted, nan=0) 
        actSum = round(np.nansum(actual),2)
        predSum = round(np.nansum(predicted),2)
        
        actMax = round(np.nanmax(actual),2)
        predMax = round(np.nanmax(predicted),2)
        
        actMean = round(np.nanmean(actual),2)
        predMean = round(np.nanmean(predicted),2)
        # Read raster to get extent for writing the rasters later
        ds, rastergeo = osgu.readRaster(input)

        r1, MAEdataset = mae_error(actual, predicted)
        r2 = round(rmse_error(actual, predicted), 4)
        r3 = round(nmae_error(actual, predicted), 4)
        r4 = round(nrmse_error(actual, predicted), 4)

        r5 = prop_error(actual, predicted)[0]
        MAEdataset = prop_error(actual, predicted)[1]
        r6 = div_error(actual, predicted)[0]
        #DIVdataset = div_error(actual, predicted)[1]
        r6 = percentage_error(actual, predicted)[0]
        DIVdataset = percentage_error(actual, predicted)[1]

        stdActual = round(np.std(actual, dtype=np.float64),2)
        stdPred = round(np.std(predicted, dtype=np.float64),2)
        
        print("----- Step #2: Writing CSV with Metrics -----")    
        if os.path.exists(filenamemetrics2e):
            with open(filenamemetrics2e, 'a') as myfile:
                myfile.write(fileName + ';' + str(round(r1,3)) + ';'+ str(r2) + ';' + str(r3) + ';' + str(r4) + ';' + str(round(r5,2)) + ';' 
                            + str(round(r6,2)) +';' + str(actSum) + ';' + str(predSum) +  ';' + str(actMax) + ';' + str(predMax) + ';' + str(round(actMean,2)) + ';' + str(round(predMean,2)) + ';' + str(stdActual) + ';' + str(stdPred) + '\n')       
        else:
            with open(filenamemetrics2e, 'w+') as myfile:
                myfile.write('Comparison among the predictions and the ground truth data for the Municipality of Copenhagen\n')
                myfile.write('Method;MAE;RMSE;MAEMEAN;RMSEMEAN;PrE;PE;ActualSum;PredictedSum;ActualMax;PredictedMax;ActualMean;PredictedMean;ActualSTD;PredictedSTD\n')
                myfile.write(fileName + ';' + str(round(r1,3)) + ';'+ str(r2) + ';' + str(r3) + ';' + str(r4) + ';' + str(round(r5,2)) + ';' 
                            + str(round(r6,2)) + ';' + str(actSum) + ';' + str(predSum) +  ';' + str(actMax) + ';' + str(predMax) + ';' + str(round(actMean,2)) + ';' + str(round(predMean,2)) + ';' + str(stdActual) + ';' + str(stdPred) + '\n')
               
        # Write the difference and the quotient TIF files (gridcells) 
        print("----- Step #3: Writing TIF files with Difference and quotient -----") 
        #outfileMAECL = outputPath + "/mae_{0}_{1}CL.tif".format(city,fileName)
        #outfileDivCL = outputPath + "/div_{0}_{1}CL.tif".format(city,fileName)
        outfileMAE = outputPath + "/mae_{0}_{1}.tif".format(city,fileName)
        outfileDiv = outputPath + "/div_{0}_{1}.tif".format(city,fileName)
        osgu.writeRaster(MAEdataset[:,:], rastergeo, outfileMAE)
        osgu.writeRaster(DIVdataset[:,:], rastergeo, outfileDiv)
        
        
        #maskRaster(polyPath,outfileMAECL, outfileMAE)
        #maskRaster(polyPath,outfileDivCL, outfileDiv)
        
        #os.remove(outfileMAECL)
        #os.remove(outfileDivCL)
        
        # Write the difference and the quotient TIF files (gridcells) 
        print("----- Step #3A: Plotting Difference and quotient -----") 
        exportPath = outputPath + "/mae_{}_Grid.png".format(fileName)
        if not os.path.exists(exportPath):
            title ="Mean Absolute Error by Neighborhood (persons)\n({})(2018)".format(fileName)
            LegendTitle = "Absolute Error (persons)"
            src = rasterio.open(outfileMAE)
            plot_map(city,'mae', src, exportPath, title, LegendTitle, districtPath =  polyPath , neighPath = districtPath, waterPath = waterPath, invertArea = None, addLabels=True)
        
        exportPath = outputPath + "/div007_{}_Grid.png".format(fileName)
        if not os.path.exists(exportPath):
            title ="Percentage Error (%)\n({})(2018)".format(fileName)
            LegendTitle = "Error (%)"
            src = rasterio.open(outfileDiv)
            plot_map(city,'div007', src, exportPath, title, LegendTitle, districtPath = polyPath, neighPath = districtPath , waterPath = waterPath, invertArea = None, addLabels=True)
        
        """
        # Calculate the mean difference and the quotient by neighborhood 
        # Write Zonal Statistics file and csv
        print("----- Step #4: Calculating the mean difference and the quotient by neighborhood -----")
        aggr_outfileMAE = outputPath + "/mae_{0}_{1}.geojson".format(city,fileName)
        aggr_outfileDiv = outputPath + "/div_{0}_{1}.geojson".format(city,fileName)
        statistics = "mean"
        print("----- Step #4A: Calculating ZSTAT (mean) -----")
        if not os.path.exists(aggr_outfileMAE):
            print('NO ZONAL STAT CALCULATED')
            #zonalStat(outfileMAE, aggr_outfileMAE, polyPath, statistics)
        if not os.path.exists(aggr_outfileDiv):
            print('NO ZONAL STAT CALCULATED')
            #zonalStat(outfileDiv, aggr_outfileDiv, polyPath, statistics)
        
        print("----- Step #4B: Plotting the mean difference and the quotient by neighborhood -----")
        src = gpd.read_file(aggr_outfileMAE)
        #src = src.loc[src['BU_CODE'].str.contains('BU0363')]
        exportPath = outputPath + "/mae_{}_Polyg.png".format(fileName)
        if not os.path.exists(exportPath):
            title ="Mean Absolute Error by Neighborhood (persons)\n({})(2018)".format(fileName)
            LegendTitle = "Absolute Error (persons)"
            #plot_mapVectorPolygons(city,'mae', src, exportPath, title, LegendTitle, 'mean_', districtPath = districtPath, neighPath = polyPath, waterPath = waterPath, invertArea = None, addLabels=True)
        
        src = gpd.read_file(aggr_outfileDiv)
        #src = src.loc[src['BU_CODE'].str.contains('BU0363')]
        exportPath = outputPath + "/div_{}_Polyg.png".format(fileName)
        if not os.path.exists(exportPath):
            title ="Mean Percentage Accuracy by Neighborhood (%)\n({})(2018)".format(fileName)
            LegendTitle = "Mean Accuracy (%)"
            #plot_mapVectorPolygons(city,'div', src, exportPath, title, LegendTitle, 'mean_', districtPath=districtPath, neighPath=polyPath, waterPath=waterPath, invertArea= None, addLabels=True)
                  
        # Calculate the sum of population by neighborhood 
        # Write Zonal Statistics file and csv
        print("----- Step #5: Calculating the sum of the population by neighborhood ZSTAT -----")
        aggr_outfileSUM = outputPath+ "/{0}_{1}.geojson".format(city,fileName)
        statistics = "sum"
        print("----- Step #5A: Calculating ZSTAT (sum)-----")
        if not os.path.exists(aggr_outfileSUM):
            print('NO ZONAL STAT CALCULATED')
            #zonalStat(outputfile, aggr_outfileSUM, polyPath, statistics)
        
        print("----- Step #5B: Plotting the total population by neighborhood for predictions -----")
        src = gpd.read_file(aggr_outfileSUM)
        exportPath = outputPath + "{}_Polyg.png".format(fileName)
        if not os.path.exists(exportPath):
            title ="Population Distribution (persons)\n({})(2018)".format(fileName)
            LegendTitle = "Population (persons)"
            #plot_mapVectorPolygons(city,'popdistributionPolyg', src, exportPath, title, LegendTitle, 'sum_', districtPath = districtPath, neighPath = polyPath, waterPath = waterPath, invertArea = None, addLabels=True)

        print("----- Step #5C: Plotting the difference and the quotient of the sum by neighborhood for predictions to the original -----")
        frame = gpd.read_file(aggr_outfileSUMGT) 
        frame = frame.set_index('Buurtcode').join(src.set_index('Buurtcode'), lsuffix='_l')
        frame["dif_{}".format(attr_value)] = frame['sum_'] - frame["{}".format(attr_value)]
        frame["prop_{}".format(attr_value)] = (frame["{}".format(attr_value)] / frame['sum_'])*100
        exportPath = outputPath + "/mae0_{}_Polyg.png".format(fileName)
        if not os.path.exists(exportPath):
            title ="Error by Neighborhood (persons)\n({})(2018)".format(fileName)
            LegendTitle = "Error (persons)"
            #plot_mapVectorPolygons(city,'mae', frame, exportPath, title, LegendTitle, "dif_{}".format(attr_value), districtPath = districtPath, neighPath = polyPath, waterPath = waterPath, invertArea = None, addLabels=True)
            
        exportPath = outputPath + "/div0_{}_Polyg.png".format(fileName)
        if not os.path.exists(exportPath):
            title ="Percentage Accuracy by Neighborhood (%)\n({})(2018)".format(fileName)
            LegendTitle = "Accuracy (%)"
            #plot_mapVectorPolygons(city,'div', frame, exportPath, title, LegendTitle, "prop_{}".format(attr_value), districtPath=districtPath, neighPath=polyPath, waterPath=waterPath, invertArea= None, addLabels=True)
        """
    
    print("----- Step #6: Calculating and Plotting the total population by neighborhood for ground truth-----")   
    src = gpd.read_file(aggr_outfileSUMGT)   
    exportPath = evalPath + "{0}_{1}_{2}_Polyg.png".format(year,city,attr_value)
    if not os.path.exists(exportPath):
        title ="Population Distribution (persons)\n(source:DST, Municipality)(2018)"
        LegendTitle = "Population (persons)"
        plot_mapVectorPolygons(city,'popdistributionPolyg', src, exportPath, title, LegendTitle, '{}'.format(attr_value), districtPath = polyPath , neighPath = districtPath, waterPath = waterPath, invertArea = None, addLabels=True)
    
