import csv
import glob
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
from mainFunctions import createFolder, csvTOlatex
from plotting import plot_map, plot_mapVectorPolygons, plotMatrix, scatterplot
from plotting.createScatterPlot import histplot
import osgeoutils as osgu
from sklearn.preprocessing import StandardScaler
from evaluating import (Rsquared, mae_error, mape_error, percentage_error,
                        residual, rmse_error)

def writeMetrics(pathFile, fileNames, metrics, title):
    if os.path.exists(pathFile):
        with open(pathFile, 'a', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(metrics) 
    else:
        with open(pathFile, 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            
            wr.writerow([title])
            wr.writerow(fileNames)
            wr.writerow(metrics)

def eval_Results_Metrics(evalFiles, outputGT,  city, attr_value, evalPath):
    
    print("----- Calculating metrices for {} -----".format(city)) 
    print("----- {} files to be evaluated -----".format(len(evalFiles))) 

    metrics2eMAE = evalPath + '/{}_MAE.csv'.format(city)
    metrics2eRMSE = evalPath + '/{}_RMSE.csv'.format(city)
    metrics2ePE = evalPath + '/{}_MAPE.csv'.format(city)
    metrics2eR2 = evalPath + '/{}_R2.csv'.format(city)

    fileNames = []
    MAE_metrics = []
    RMSE_metrics = []
    MAPE_metrics = []
    r2_metrics = []
    
    ##### -------- Process Evaluation: Steps -------- #####
    print("----- Ground Truth for {} at Grid Cells -----".format(city))      
    for i in range(len(evalFiles)):
        file = evalFiles[i]
        path = Path(file)
        fileName = path.stem
        # Clip the predictions to AMS extent 
        input = file
                    
        print("----- Step #2: Calculating Metrics {}-----".format(attr_value))
        src_real = rasterio.open(outputGT)
        actual = src_real.read(1)
        
        src_pred = rasterio.open(input)
        predicted = src_pred.read(1)
        predicted[(np.where((predicted <= -100000)))] = np.nan
        predicted = np.nan_to_num(predicted, nan=0) 

        r1, MAEdataset, std = mae_error(actual, predicted) 
        r2 = rmse_error(actual, predicted)
        r3 = mape_error(actual,predicted)
        
        r4 = Rsquared(actual,predicted)
        #r6 = percentage_error(actual, predicted)[0]
        DIVdataset = np.absolute(percentage_error(actual, predicted)[1])
        DIVdataset[(np.where(DIVdataset == 100))] = 0
        
        fileNames.append(fileName)
        MAE_metrics.append("{0}/±{1}".format(r1,std))
        RMSE_metrics.append(r2)
        MAPE_metrics.append(r3)  
        r2_metrics.append(r4)  

    MAE_metrics.insert(0, attr_value)
    RMSE_metrics.insert(0, attr_value)
    MAPE_metrics.insert(0, attr_value)
    r2_metrics.insert(0, attr_value)
    fileNames.insert(0, "Model")

    writeMetrics(metrics2eMAE, fileNames, MAE_metrics,'MAE for {}'.format(city))
    writeMetrics(metrics2eRMSE, fileNames, RMSE_metrics,'RMSE for {}'.format(city))
    writeMetrics(metrics2ePE, fileNames, MAPE_metrics,'MAPE for {}'.format(city))
    writeMetrics(metrics2eR2, fileNames, r2_metrics,'r2 for {}'.format(city))

    csvTOlatex(metrics2eMAE, evalPath + '/{}_MAE.tex'.format(city))  
    csvTOlatex(metrics2eRMSE, evalPath + '/{}_RMSE.tex'.format(city)) 
    csvTOlatex(metrics2eR2, evalPath + '/{}_R2.tex'.format(city))  
    csvTOlatex(metrics2ePE, evalPath + '/{}_MAPE.tex'.format(city))

def eval_Results(evalPath, outputGT, exportPathGT, polyPath, districtPath, waterPath, invertArea,evalFiles, year, city, attr_value, aggr_outfileSUMGT , scatterPath):
    ##### -------- PLOT Ground truth at Grid Cells -------- #####
    print("----- Plotting Population Distribution -----") 
    print("----- Ground Truth for {} at Grid Cells -----".format(city)) 
    exportPath = exportPathGT + "/{0}_GT_{1}.png".format(city,attr_value)
    if not os.path.exists(exportPath):
        title ="Population Distribution (persons)\n(Ground Truth: {})(2018)".format(attr_value)
        LegendTitle = "Population (persons)"
        src = rasterio.open(outputGT)
        plot_map(city, 'popdistribution', src, exportPath, title, LegendTitle, districtPath = districtPath, neighPath = polyPath, waterPath = waterPath, invertArea = None, addLabels=True)

    ##### -------- Process Evaluation: Steps -------- #####
    print("----- Plotting Population Distribution -----") 
    print("----- Predictions for Amsterdam at Grid Cells -----")      
    for i in range(len(evalFiles)):
        file = evalFiles[i]
        path = Path(file)
        fileName = path.stem
        method = path.parent.stem
    
        # Clip the predictions to AMS extent 
        input = file
        outputPath = evalPath + "/{}".format(method)
        createFolder(outputPath)
            
        # Plot the population distribution of the predictions 
        exportPath = outputPath + "/{}.png".format(fileName)
        if not os.path.exists(exportPath):
            print("----- Step #1: Plotting Population Distribution -----")
            title ="Population Distribution (persons)\n({})(2018)".format(fileName)
            LegendTitle = "Population (persons)"
            src = rasterio.open(path)
            plot_map(city,'popdistributionPred', src, exportPath, title, LegendTitle, districtPath = districtPath, neighPath = polyPath, waterPath = waterPath, invertArea = None, addLabels=True)
                
        print("----- Step #2: Calculating Metrics {}-----".format(attr_value))
        src_real = rasterio.open(outputGT)
        actual = src_real.read(1)
        
        src_pred = rasterio.open(input)
        predicted = src_pred.read(1)
        predicted[(np.where((predicted <= -100000)))] = np.nan
        predicted = np.nan_to_num(predicted, nan=0) 
        
        # Read raster to get extent for writing the rasters later
        ds, rastergeo = osgu.readRaster(input)

        MAEdataset = mae_error(actual, predicted)[1]
        
        RESdataset = residual(actual, predicted)[0]
        DIFdataset = residual(actual, predicted)[1]
        #r6 = percentage_error(actual, predicted)[0]
        DIVdataset = np.absolute(percentage_error(actual, predicted)[1])
        DIVdataset[(np.where(DIVdataset == 100))] = 0

        # Write the difference and the quotient TIF files (gridcells) 
        print("----- Step #3: Writing TIF files with Difference and quotient -----") 
        outfileMAE = outputPath + "/mae_ams_{}.tif".format(fileName)
        outfileDiv = outputPath + "/div_ams_{}.tif".format(fileName)
        outfileRes = outputPath + "/res_ams_{}.tif".format(fileName)
        outfileDif = outputPath + "/dif_ams_{}.tif".format(fileName)
        
        osgu.writeRaster(MAEdataset[:,:], rastergeo, outfileMAE)
        osgu.writeRaster(DIVdataset[:,:], rastergeo, outfileDiv)
        osgu.writeRaster(RESdataset[:,:], rastergeo, outfileRes)
        osgu.writeRaster(DIFdataset[:,:], rastergeo, outfileDif)
        
        # Write the difference and the quotient TIF files (gridcells) 
        print("----- Step #3A: Plotting Difference and quotient {}-----".format(attr_value)) 
        exportPath = outputPath + "/mae_{}_Grid.png".format(fileName)
        if not os.path.exists(exportPath):
            title ="Absolute Error (persons)\n({})(2018)".format(fileName)
            LegendTitle = "Absolute Error (persons)"
            src = rasterio.open(outfileMAE)
            plot_map(city,'mae', src, exportPath, title, LegendTitle, districtPath = districtPath, neighPath = polyPath, waterPath = waterPath, invertArea = invertArea, addLabels=True)
        
        exportPath = outputPath + "/div_{}_Grid.png".format(fileName)
        if not os.path.exists(exportPath):
            title ="Absolute Percentage Error (%)\n({})(2018)".format(fileName)
            LegendTitle = "Error (%)"
            src = rasterio.open(outfileDiv)
            plot_map(city,'pe', src, exportPath, title, LegendTitle, districtPath = districtPath, neighPath = polyPath, waterPath = waterPath, invertArea = None, addLabels=True)

        # Scatterplot for GT in x-axis and predictions in y-axis
        exportPath = scatterPath + "/scat_{}.png".format(fileName)   
        #scatterplot(path, exportPath, outputGT, 'Ground Truth (persons)', 'Predictions (persons)', 'Scatter plot of Predictions-Ground Truth', 0, 100, 0, 100)
        
        # Scatterplot of proportion predictions per gt in y-axis and ground truth in x-axis
        exportPath = scatterPath + "/scat_div_{}.png".format(fileName) 
        #scatterplot(Path(outputGT), exportPath, Path(outfileRes), 'Quotient (PR/GT *100)', 'Ground Truth (persons)', 'Distribution of rate Predictions per Ground Truth by Ground Truth', 0,200, 0, 100)
        
        # Scatterplot of proportion predictions per gt in y-axis and ground truth in x-axis
        exportPath = scatterPath + "/scat_dif_{}.png".format(fileName) 
        #scatterplot(Path(outfileDif), exportPath, Path(path), 'Difference (GT-Pred)', 'Ground Truth (persons)', 'Distribution of error by Ground Truth', -100, 100, 0, 100)
            
    print("----- Step #6: Calculating and Plotting the total population by neighborhood for ground truth {}-----".format(attr_value))   
    src = gpd.read_file(aggr_outfileSUMGT)   
    exportPath = exportPathGT +  "/{0}_GT_{1}_Polyg.png".format(year,city,attr_value)
    if not os.path.exists(exportPath):
        title ="Population Distribution (persons)\n(2018)"
        LegendTitle = "Population (persons)"
        plot_mapVectorPolygons(city,'popdistributionPolyg', src, exportPath, title, LegendTitle, '{}'.format(attr_value), districtPath = districtPath, neighPath = polyPath, waterPath = waterPath, invertArea = None, addLabels=True)

    files_in_dir = glob.glob(outputPath + '*/*.tif')
    print('Files to be removed:', len(files_in_dir))
    for _file in files_in_dir:
        os.remove(_file)

def eval_Scatterplots(evalPath, outputGT, evalFiles, city, attr_value, scatterPath):
    ##### -------- Process Evaluation: Steps -------- #####
    print("----- Plotting Scatterplots for {0}, {1} -----".format(attr_value,city)) 
    for i in range(len(evalFiles)):
        file = evalFiles[i]
        path = Path(file)
        fileName = path.stem
        method = path.parent.stem
    
        # Clip the predictions to AMS extent 
        input = file
        outputPath = evalPath + "/{}".format(method)
        createFolder(outputPath)
            
        print("----- Step #2: Calculating Metrics {}-----".format(attr_value))
        src_real = rasterio.open(outputGT)
        actual = src_real.read(1)
        
        src_pred = rasterio.open(input)
        predicted = src_pred.read(1)
        predicted[(np.where((predicted <= -100000)))] = np.nan
        predicted = np.nan_to_num(predicted, nan=0) 
        
        # Read raster to get extent for writing the rasters later
        ds, rastergeo = osgu.readRaster(input)

        MAEdataset = mae_error(actual, predicted)[1]
        
        RESdataset = residual(actual, predicted)[0]
        DIFdataset = residual(actual, predicted)[1]
        print(DIFdataset.shape)
        stand_residual = StandardScaler().fit_transform(residual(actual, predicted)[1])
        #r6 = percentage_error(actual, predicted)[0]
        
        DIVdataset = np.absolute(percentage_error(actual, predicted)[1])
        DIVdataset[(np.where(DIVdataset == 100))] = 0

        # Write the difference and the quotient TIF files (gridcells) 
        print("----- Step #3: Writing TIF files with Difference and quotient -----") 
        outfileMAE = outputPath + "/mae_ams_{}.tif".format(fileName)
        outfileDiv = outputPath + "/div_ams_{}.tif".format(fileName)
        outfileRes = outputPath + "/res_ams_{}.tif".format(fileName)
        outfileDif = outputPath + "/dif_ams_{}.tif".format(fileName)
        outfileStandRes = outputPath + "/standRes_ams_{}.tif".format(fileName)
        
        osgu.writeRaster(stand_residual[:,:], rastergeo, outfileStandRes)
        osgu.writeRaster(DIVdataset[:,:], rastergeo, outfileDiv)
        osgu.writeRaster(RESdataset[:,:], rastergeo, outfileRes)
        osgu.writeRaster(DIFdataset[:,:], rastergeo, outfileDif)
        
        # Scatterplot for GT in x-axis and predictions in y-axis
        exportPath = scatterPath + "/scatP_{}.png".format(fileName)   
        scatterplot(Path(path), Path(outputGT), exportPath, 'Predictions (persons)', 'Ground Truth (persons)',  'Predicted vs Actual', 0, 100, 0, 100)
        
        # Scatterplot for GT in x-axis and predictions in y-axis
        exportPath = scatterPath + "/hist_{}.png".format(fileName)   
        histplot(Path(path), Path(outputGT), exportPath, 'Predictions (persons)', 'Ground Truth (persons)',  'Predicted vs Actual', x1=None, x2=100, y1=None, y2=None)
        
        # Scatterplot for GT in x-axis and predictions in y-axis
        exportPath = scatterPath + "/scat_{}.png".format(fileName)   
        scatterplot(Path(outputGT), Path(path), exportPath, 'Ground Truth (persons)', 'Predictions (persons)',  'Predicted vs Actual', 0, 100, 0, 100)
        
        # Scatterplot for predictions in x-axis and standardized residuals in y-axis
        exportPath = scatterPath + "/scatResidualsP_{}.png".format(fileName)   
        scatterplot(Path(path), Path(outfileStandRes), exportPath, 'Predictions (persons)', 'Standardized Residuals', 'Residuals', x1=0, x2=100, y1=-10, y2=10)
        
        # Scatterplot for predictions in x-axis and standardized residuals in y-axis
        exportPath = scatterPath + "/scatResiduals_{}.png".format(fileName)   
        scatterplot(Path(outputGT), Path(outfileStandRes), exportPath, 'Ground Truth (persons)', 'Standardized Residuals', 'Residuals', x1=0, x2=100, y1=-10, y2=10)
        
        # Scatterplot of proportion predictions per gt in y-axis and ground truth in x-axis
        exportPath = scatterPath + "/scat_div_{}.png".format(fileName) 
        #scatterplot(Path(outputGT), exportPath, Path(outfileRes), 'Quotient (PR/GT *100)', 'Ground Truth (persons)', 'Distribution of rate Predictions per Ground Truth by Ground Truth', 0,200, 0, 100)
        
        # Scatterplot of proportion predictions per gt in y-axis and ground truth in x-axis
        exportPath = scatterPath + "/scat_dif_{}.png".format(fileName) 
        #scatterplot(Path(outfileDif), exportPath, Path(path), 'Difference (GT-Pred)', 'Ground Truth (persons)', 'Distribution of error by Ground Truth', -100, 100, 0, 100)
            
    files_in_dir = glob.glob(outputPath + '*/*.tif')
    print('Files to be removed:', len(files_in_dir))
    for _file in files_in_dir:
        os.remove(_file)  

def createMatrices(evalPath, city, attr_value, evalPathKNN, evalPathKNNgf, evalPathMatrices, exportPathGT, fileName, method, scatterPath=None):
    # Population distribution
    e= [os.path.dirname(os.path.dirname(evalPath)) + "/{0}_GT/{0}_{0}/{0}_GT_{1}.png".format(city,attr_value),
        evalPath + "/apcnn/dissever01_RMSE_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}.png".format(attr_value),
        evalPath + "/apcnn/dissever01_RMSEflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}.png".format(attr_value),
        evalPath + "/apcnn/dissever01_CLF_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}.png".format(attr_value),
        evalPath + "/apcnn/dissever01_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}.png".format(attr_value),
        evalPath + "/apcnn/dissever01_00RBLF_2018_ams_Dasy_16unet_10epochspi_12AIL12_it10_{}.png".format(attr_value),
        evalPath + "/apcnn/dissever01_tr2008CLF_2018_ams_Dasy_16unet_10epochspi_2AIL20_it10_{}.png".format(attr_value),
        evalPath + "/apcnn/dissever01_CLF_2018_ams_Dasy_16unet_10epochspi_2AIL21_it10_{1}.png".format(city,attr_value),
        evalPath + "/apcatbr/dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{}.png".format(attr_value)]
    path = evalPathMatrices + "/popdis_LF_HD_{0}.png".format(attr_value)
    title = "Population Distribution: {0}".format(attr_value)
    plotMatrix(e, path, title) 
    
    """# Population distribution
    e= [exportPathGT + "/{0}_GT_{1}.png".format(city,attr_value),
        evalPath + "/{1}/{0}.png".format(fileName,method),
        evalPath + "/{1}/mae_{0}_Grid.png".format(fileName,method),
        evalPath + "/{1}/div_{0}_Grid.png".format(fileName,method),
        #evalPathKNN + "/convDif_50_{0}.png".format(fileName),
        #evalPathKNN + "/convDif_100_{0}.png".format(fileName),
        #evalPathKNN + "/convDif_200_{0}.png".format(fileName),
        #evalPathKNN + "/convDif_500_{0}.png".format(fileName),
        evalPathKNNgf + "/convDif_50_{0}.png".format(fileName),
        evalPathKNNgf + "/convDif_100_{0}.png".format(fileName),
        evalPathKNNgf + "/convDif_200_{0}.png".format(fileName),
        evalPathKNNgf + "/convDif_500_{0}.png".format(fileName)]
    title = "Collection: {0}".format(attr_value)
    plotMatrix(e, evalPathMatrices, title) """
    """
    # Population distribution
    e= [scatterPath + "/scat_dissever01_RMSEflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}.png".format(attr_value),
        scatterPath + "/scat_dissever01_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{0}.png".format(attr_value),
        scatterPath + "/scat_dissever01_2_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{0}.png".format(attr_value),
        scatterPath + "/scat_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{0}.png".format(attr_value),
        scatterPath + "/scatResiduals_dissever01_RMSEflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}.png".format(attr_value),
        scatterPath + "/scatResiduals_dissever01_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{0}.png".format(attr_value),
        scatterPath + "/scatResiduals_dissever01_2_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{0}.png".format(attr_value),
        scatterPath + "/scatResiduals_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{0}.png".format(attr_value),
        scatterPath + "/scatResidualsP_dissever01_RMSEflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}.png".format(attr_value),
        scatterPath + "/scatResidualsP_dissever01_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{0}.png".format(attr_value),
        scatterPath + "/scatResidualsP_dissever01_2_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{0}.png".format(attr_value),
        scatterPath + "/scatResidualsP_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{0}.png".format(attr_value)]
    title = "Collection: {0}".format(attr_value)
    plotMatrix(e, evalPathMatrices, title) 
    """
    """
    # Population distribution
    e= [os.path.dirname(os.path.dirname(evalPath)) + "/{0}_GT/{0}_GT_{1}.png".format(city,attr_value),
        evalPath + "/apcnn/dissever01_RMSE_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}.png".format(attr_value),
        evalPath + "/apcnn/dissever01_CLF_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}.png".format(attr_value),
        evalPath + "/apcnn/dissever01_RMSEflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}.png".format(attr_value),
        evalPath + "/apcnn/dissever01_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}.png".format(attr_value),
        evalPath + "/apcnn/dissever01_1_CLFflipped2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}.png".format(attr_value),
        evalPath + "/apcnn/dissever01_2_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}.png".format(attr_value),
        evalPath + "/apcatbr/dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{}.png".format(attr_value)]
    path = evalPath + "/matrices/popdistribution_{0}.png".format(attr_value)
    title = "Population Distribution: {0}".format(attr_value)
    plotMatrix(e, path, title) 
    
    # Mean Absolute error
    e= [evalPath + "/apcnn/mae_dissever01_RMSE_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}_Grid.png".format(attr_value),
        evalPath + "/apcnn/mae_dissever01_CLF_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}_Grid.png".format(attr_value),
        evalPath + "/apcnn/mae_dissever01_RMSEflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}_Grid.png".format(attr_value),
        evalPath + "/apcnn/mae_dissever01_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}_Grid.png".format(attr_value),
        evalPath + "/apcnn/mae_dissever01_1_CLFflipped2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}_Grid.png".format(attr_value),
        evalPath + "/apcnn/mae_dissever01_2_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}_Grid.png".format(attr_value),
        evalPath + "/apcatbr/mae_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{0}_Grid.png".format(attr_value)]
    path = evalPath + "/matrices/mae_{0}.png".format(attr_value) 
    title = "Mean Absolute Error: {0}".format(attr_value)
    plotMatrix(e, path, title) 
    
    # MAPE
    e= [evalPath + "/apcnn/div_dissever01_RMSE_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}_Grid.png".format(attr_value),
        evalPath + "/apcnn/div_dissever01_CLF_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}_Grid.png".format(attr_value),
        evalPath + "/apcnn/div_dissever01_RMSEflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}_Grid.png".format(attr_value),
        evalPath + "/apcnn/div_dissever01_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}_Grid.png".format(attr_value),
        evalPath + "/apcnn/div_dissever01_1_CLFflipped2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}_Grid.png".format(attr_value),
        evalPath + "/apcnn/div_dissever01_2_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}_Grid.png".format(attr_value),
        evalPath + "/apcatbr/div_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{0}_Grid.png".format(attr_value)]
    path = evalPath + "/matrices/mape_{0}.png".format(attr_value)
    title = "Mean Absolute Percentage Error: {0}".format(attr_value)
    plotMatrix(e, path, title) 
    
    # Scatterplots
    e= [evalPath + "/apcnn/scat_div_dissever01_RMSE_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}.png".format(attr_value),
        evalPath + "/apcnn/scat_div_dissever01_CLF_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}.png".format(attr_value),
        evalPath + "/apcnn/scat_div_dissever01_RMSEflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}.png".format(attr_value),
        evalPath + "/apcnn/scat_div_dissever01_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}.png".format(attr_value),
        evalPath + "/apcnn/scat_div_dissever01_1_CLFflipped2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}.png".format(attr_value),
        evalPath + "/apcnn/scat_div_dissever01_2_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}.png".format(attr_value),
        evalPath + "/apcatbr/scat_div_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{0}.png".format(attr_value)]
    path = evalPath + "/matrices/scat_div_{0}.png".format(attr_value)
    title = "Difference per Ground Truth: {0}".format(attr_value)
    plotMatrix(e, path, title) 
    
    e= [evalPath + "/apcnn/scat_dissever01_RMSE_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}.png".format(attr_value),
        evalPath + "/apcnn/scat_dissever01_CLF_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}.png".format(attr_value),
        evalPath + "/apcnn/scat_dissever01_RMSEflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}.png".format(attr_value),
        evalPath + "/apcnn/scat_dissever01_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}.png".format(attr_value),
        evalPath + "/apcnn/scat_dissever01_1_CLFflipped2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}.png".format(attr_value),
        evalPath + "/apcnn/scat_dissever01_2_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}.png".format(attr_value),
        evalPath + "/apcatbr/scat_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{0}.png".format(attr_value)]
    path = evalPath + "/matrices/scat_{0}.png".format(attr_value)
    title = "Quotient per Ground Truth: {0}".format(attr_value)
    plotMatrix(e, path, title) """
    
    
    
    
