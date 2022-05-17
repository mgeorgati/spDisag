from fileinput import filename
import os
import sys
from pathlib import Path
import csv
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
from osgeo import gdal
import glob 
import osgeoutils as osgu
from evaluateFunctions import (mae_error, rmse_error, mape_error, percentage_error)
from gdalutils import maskRaster
from plotting.plotRaster import plot_map
from plotting.plotVectors import plot_mapVectorPolygons

print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mainFunctions.basic import createFolder

"""
    EVALUATION OF THE PREDICTIONS WITH THE GROUND TRUTH DATA OF OISg DATASET
"""
def eval_Results_ams(ROOT_DIR, pop_path, ancillary_path, year, city, attr_value):
    evalPath = ROOT_DIR + "/Evaluation/{}_20220323/".format(city)
    
    # Required files
    actualPath = pop_path + "/{0}/GridCells/rasters/{1}_{0}_{2}.tif".format(city, year, attr_value)
    polyPath = ROOT_DIR + "/Shapefiles/{1}/{0}_{1}.shp".format(year,city)
    districtPath = ancillary_path + '/adm/{0}_districts.geojson'.format(city)    
    waterPath = ancillary_path + '/corine/waterComb_{0}_CLC_2012_2018.tif'.format(city)
   
    aggr_outfileSUMGT = ROOT_DIR + "/Shapefiles/Comb/{0}_ams_ois.geojson".format(year,city)
    ##### -------- PLOT Ground truth at Grid Cells -------- #####
    print("----- Plotting Population Distribution -----") 
    
    templatePathGA = ancillary_path + '/template/{0}_template_100.tif'.format(city)
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
    #outputGTL =  ROOT_DIR + "/Evaluation/{1}_groundTruth/{0}_{1}_{2}.tif".format(year,city,attr_value)
    #if not os.path.exists(outputGTL):
        #gdal.Translate(outputGTL, actualPath, projWin=bbox)

    outputGT =  pop_path + "GridCells/rasters/{0}_{1}_{2}.tif".format(year,city,attr_value)
    #if not os.path.exists(outputGT):
        #maskRaster(polyPath, outputGTL, outputGT)
    
    print("----- Ground Truth for Amsterdam at Grid Cells -----") 
    exportPath = evalPath + "/GT/ams_GT_{1}.png".format(city,attr_value)
    if not os.path.exists(exportPath):
        title ="Population Distribution (persons)\n(Ground Truth: {})(2018)".format(attr_value)
        LegendTitle = "Population (persons)"
        src = rasterio.open(outputGT)
        #plot_map(city, 'popdistribution', src, exportPath, title, LegendTitle, districtPath = districtPath, neighPath = polyPath, waterPath = waterPath, invertArea = None, addLabels=True)
    
    ##### -------- Get files to be processed -------- #####
    print("----- Select prediction files to be evaluated -----") 
    evalFiles = [ROOT_DIR + "/Results/{0}/apcnn/dissever01_CLF_2018_ams_Dasy_16unet_10epochspi_AIL12_it3_{1}.tif".format(city,attr_value),
                 ROOT_DIR + "/Results/{0}/apcnn/dissever01_CLF1_2018_ams_Dasy_16unet_10epochspi_AIL10_it10_{1}.tif".format(city,attr_value),
                 ROOT_DIR + "/Results/{0}/apcnn/dissever01_CLF2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{1}.tif".format(city,attr_value),
                 #ROOT_DIR + "/Results/{0}/apcnn/dissever01_RMSE2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{1}.tif".format(city,attr_value),
                 #ROOT_DIR + "/Results/{0}/apcnn/dissever01_CLF2018_ams_Dasy_16unet_10epochspi_AIL12_it2_{1}.tif".format(city,attr_value),
                 #ROOT_DIR + "/Results/{0}/apcnn/dissever01_CLF2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{1}.tif".format(city,attr_value),
                 ROOT_DIR + "/Results/{0}/apcatbr/dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{1}.tif".format(city,attr_value)
                 ]

    print(evalFiles)
    print("----- {} files to be evaluated -----".format(len(evalFiles))) 

    metrics2eMAE = evalPath + '/{}_MAE1.csv'.format(city)
    metrics2eRMSE = evalPath + '/{}_RMSE1.csv'.format(city)
    metrics2ePE = evalPath + '/{}_MAPE1.csv'.format(city)
    

    fileNames = []
    MAE_metrics = []
    RMSE_metrics = []
    MAPE_metrics = []
    ##### -------- Process Evaluation: Steps -------- #####
    print("----- Plotting Population Distribution -----") 
    print("----- Ground Truth for Amsterdam at Grid Cells -----")      
    for i in range(len(evalFiles)):
        file = evalFiles[i]
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
        elif 'unet' in fileName:
            outputPath = evalPath + "/apcnn"
            createFolder(outputPath)
        elif 'apcatbr' in fileName:
            outputPath = evalPath + "/apcatbr"
            createFolder(outputPath)
        elif 'GHS' in fileName:
            outputPath = evalPath + "/GHS"
            createFolder(outputPath)
        elif fileName.endswith('_dasyWIESMN'):
            outputPath = evalPath + "/Dasy"
            createFolder(outputPath)
        elif fileName.endswith('_pycno'):
            outputPath = evalPath + "/Pycno"
            createFolder(outputPath)
        else: 
            outputPath = evalPath
            
        outputfile = outputPath + "/ams_{}.tif".format(fileName)
        if not os.path.exists(outputfile):
            maskRaster(polyPath, input, outputfile)
        # Plot the population distribution of the predictions 
        exportPath = outputPath + "/ams_{}.png".format(fileName)
        if not os.path.exists(exportPath):
            print("----- Step #1: Plotting Population Distribution -----")
            title ="Population Distribution (persons)\n({})(2018)".format(fileName)
            LegendTitle = "Population (persons)"
            src = rasterio.open(path)
            #plot_map(city,'popdistributionPred', src, exportPath, title, LegendTitle, districtPath = districtPath, neighPath = polyPath, waterPath = waterPath, invertArea = None, addLabels=True)
        
        os.remove(outputfile)
        
        print("----- Step #2: Calculating Metrics -----")
        src_real = rasterio.open(outputGT)
        actual = src_real.read(1)
        
        src_pred = rasterio.open(input)
        predicted = src_pred.read(1)
        predicted[(np.where((predicted <= -100000)))] = np.nan
        predicted = np.nan_to_num(predicted, nan=0) 
        actSum = np.nansum(actual)
        predSum =np.nansum(predicted)
        
        actMax = np.nanmax(actual)
        predMax =np.nanmax(predicted)
        
        actMean = np.nanmean(actual)
        predMean =np.nanmean(predicted)
        # Read raster to get extent for writing the rasters later
        ds, rastergeo = osgu.readRaster(input)

        r1, MAEdataset, std = mae_error(actual, predicted) 
        r2 = rmse_error(actual, predicted)
        r3 = mape_error(actual,predicted)
        
        #r6 = percentage_error(actual, predicted)[0]
        DIVdataset = np.absolute(percentage_error(actual, predicted)[1])
        DIVdataset[(np.where(DIVdataset == 100))] = 0
                
        stdActual = round(np.std(actual, dtype=np.float64),2)
        stdPred = round(np.std(predicted, dtype=np.float64),2)
        
        fileNames.append(fileName)
        print("{0}/±{1}".format(r1,std))
        MAE_metrics.append("{0}/±{1}".format(r1,std))
        RMSE_metrics.append(r2)
        MAPE_metrics.append(r3)
        print("----- Step #2: Writing CSV with Metrics -----") 
        
        # Write the difference and the quotient TIF files (gridcells) 
        print("----- Step #3: Writing TIF files with Difference and quotient -----") 
        outfileMAECL = outputPath + "/mae_ams_{}CL.tif".format(fileName)
        outfileDivCL = outputPath + "/div_ams_{}CL.tif".format(fileName)
        
        osgu.writeRaster(MAEdataset[:,:], rastergeo, outfileMAECL)
        osgu.writeRaster(DIVdataset[:,:], rastergeo, outfileDivCL)
        
        outfileMAE = outputPath + "/mae_ams_{}.tif".format(fileName)
        outfileDiv = outputPath + "/div_ams_{}.tif".format(fileName)
        maskRaster(polyPath,outfileMAECL, outfileMAE)
        maskRaster(polyPath,outfileDivCL, outfileDiv)
        
        # Write the difference and the quotient TIF files (gridcells) 
        print("----- Step #3A: Plotting Difference and quotient -----") 
        exportPath = outputPath + "/mae_{}_Grid.png".format(fileName)
        if not os.path.exists(exportPath):
            title ="Absolute Error (persons)\n({})(2018)".format(fileName)
            LegendTitle = "Absolute Error (persons)"
            src = rasterio.open(outfileMAE)
            #plot_map(city,'mae', src, exportPath, title, LegendTitle, districtPath = districtPath, neighPath = polyPath, waterPath = waterPath, invertArea = None, addLabels=True)
        
        exportPath = outputPath + "/div_{}_Grid.png".format(fileName)
        if not os.path.exists(exportPath):
            title ="Absolute Percentage Error (%)\n({})(2018)".format(fileName)
            LegendTitle = "Error (%)"
            src = rasterio.open(outfileDiv)
            plot_map(city,'pe', src, exportPath, title, LegendTitle, districtPath = districtPath, neighPath = polyPath, waterPath = waterPath, invertArea = None, addLabels=True)
        
        
        """
        # Calculate the mean difference and the quotient by neighborhood 
        # Write Zonal Statistics file and csv
        print("----- Step #4: Calculating the mean difference and the quotient by neighborhood -----")
        aggr_outfileMAE = outputPath + "/mae_ams_{}.geojson".format(fileName)
        aggr_outfileDiv = outputPath + "/div_ams_{}.geojson".format(fileName)
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
        aggr_outfileSUM = outputPath+ "/ams_{}.geojson".format(fileName)
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
    MAE_metrics.insert(0, attr_value)
    RMSE_metrics.insert(0, attr_value)
    MAPE_metrics.insert(0, attr_value)
    fileNames.insert(0, "Model")
    if os.path.exists(metrics2eMAE):
        with open(metrics2eMAE, 'a', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(MAE_metrics) 
    else:
        with open(metrics2eMAE, 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            
            wr.writerow(['MAE for the Municipality of Amsterdam'])
            wr.writerow(fileNames)
            wr.writerow(MAE_metrics)
    
    if os.path.exists(metrics2eRMSE):
        with open(metrics2eRMSE, 'a', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(RMSE_metrics) 
    else:
        with open(metrics2eRMSE, 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(['RMSE for the Municipality of Amsterdam'])
            wr.writerow(fileNames)
            wr.writerow(RMSE_metrics)
    
    if os.path.exists(metrics2ePE):
        with open(metrics2ePE, 'a', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(MAPE_metrics) 
    else:
        with open(metrics2ePE, 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            
            wr.writerow(['MAPE for the Municipality of Amsterdam'])
            wr.writerow(fileNames)
            wr.writerow(MAPE_metrics)
    print("----- Step #6: Calculating and Plotting the total population by neighborhood for ground truth-----")   
    src = gpd.read_file(aggr_outfileSUMGT)   
    exportPath = evalPath + "{0}_{1}_{2}_Polyg.png".format(year,city,attr_value)
    if not os.path.exists(exportPath):
        title ="Population Distribution (persons)\n(source:OIS, buurt)(2018)"
        LegendTitle = "Population (persons)"
        #plot_mapVectorPolygons(city,'popdistributionPolyg', src, exportPath, title, LegendTitle, '{}'.format(attr_value), districtPath = districtPath, neighPath = polyPath, waterPath = waterPath, invertArea = None, addLabels=True)
    

    files_in_dir = glob.glob(outputPath + '/*.tif')
    for _file in files_in_dir:
        print(_file) # just to be sure, you know how it is...
        os.remove(_file)