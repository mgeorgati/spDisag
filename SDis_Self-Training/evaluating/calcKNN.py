import os
from msilib.schema import CreateFolder
from pathlib import Path

import numpy as np
import rasterio
import rasterio.plot
from mainFunctions.basic import createFolder, csvTOlatex
from plotting.plotRaster import plot_map
from scipy.ndimage import convolve, generic_filter
from utils import gdalutils

import evaluating.evalResults as evalRs
from evaluating.evaluateFunctions import mae_error


def calcConv(inputPath, nnn, outPath, city, ancillary_path):
    templ = rasterio.open(ancillary_path + "/{0}/template/{0}_templateClipped.tif".format(city))
    temp = templ.read(1)
    
    waterPath = rasterio.open(ancillary_path + '/{0}/corine/waterComb_{0}_CLC_2012_2018.tif'.format(city))
    water = waterPath.read(1)
    water = np.where(water>0.3, water, 0)
    mask = temp * water
    mask = np.where(mask==1, np.nan, mask)
    
    src = rasterio.open(inputPath)
    arr = src.read(1)
    arr[np.isnan(mask)] = np.nan
    
    kernel0 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    kernel1 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    kernel2 = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]])
    kernel3 = np.array([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]])
    kernel4 = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    kernel5 = np.array([[0, 0, 0, 1, 0, 0, 0], 
                        [0, 0, 1, 1, 1, 0, 0], 
                        [0, 1, 1, 1, 1, 1, 0], 
                        [1, 1, 1, 1, 1, 1, 1], 
                        [0, 1, 1, 1, 1, 1, 0], 
                        [0, 0, 1, 1, 1, 0, 0], 
                        [0, 0, 0, 1, 0, 0, 0]])

    kernel6 = np.array([[0, 0, 1, 1, 1, 0, 0], 
                        [0, 1, 1, 1, 1, 1, 0], 
                        [1, 1, 1, 1, 1, 1, 1], 
                        [1, 1, 1, 1, 1, 1, 1], 
                        [1, 1, 1, 1, 1, 1, 1], 
                        [0, 1, 1, 1, 1, 1, 0], 
                        [0, 0, 1, 1, 1, 0, 0]])

    kernel7 = np.array([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]])

    c0 = generic_filter(arr, np.nansum, footprint=kernel0, mode='constant', cval=np.NaN)
    c0 = np.where(c0>=nnn, 0, 1)
    c1 = generic_filter(arr, np.nansum, footprint=kernel1, mode='constant', cval=np.NaN)
    c1 = np.where(c1>=nnn, 0, 1)
    c2 = generic_filter(arr, np.nansum, footprint=kernel2, mode='constant', cval=np.NaN)
    c2 = np.where(c2>=nnn, 0, 1)
    c3 = generic_filter(arr, np.nansum, footprint=kernel3, mode='constant', cval=np.NaN)
    c3 = np.where(c3>=nnn, 0, 1)
    c4 = generic_filter(arr, np.nansum, footprint=kernel4, mode='constant', cval=np.NaN)
    c4 = np.where(c4>=nnn, 0, 1)
    c5 = generic_filter(arr, np.nansum, footprint=kernel5, mode='constant', cval=np.NaN)
    c5 = np.where(c5>=nnn, 0, 1 )
    c6 = generic_filter(arr, np.nansum, footprint=kernel6, mode='constant', cval=np.NaN)
    c6 = np.where(c6>=nnn, 0, 1)
    c7 = generic_filter(arr, np.nansum, footprint=kernel7, mode='constant', cval=np.NaN)
    print(np.max(c7))
    c7 = np.where(c7>=nnn, 0, 1)
    c = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7
    cf = np.where(temp != 0, c, np.nan)
    cf[np.isnan(mask)] = np.nan  
    gdalutils.writeRaster(inputPath, outPath, cf)
    return cf

"""
def calcConv(inputPath, nnn, outPath):
    templ = rasterio.open(ancillary_path + "/{0}/template/{0}_templateClipped.tif".format(city))
    temp = templ.read(1)
    
    waterPath = rasterio.open(ancillary_path + '/{0}/corine/waterComb_{0}_CLC_2012_2018.tif'.format(city))
    water = waterPath.read(1)
    water = np.where(water>0.3, water, 0)
    mask = temp * water
    mask = np.where(mask==1, np.nan, mask)
    
    src = rasterio.open(inputPath)
    arr = src.read(1)
    arr[np.isnan(mask)] = np.nan
    
    kernel0 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    kernel1 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    kernel2 = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]])
    kernel3 = np.array([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]])
    kernel4 = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    kernel5 = np.array([[0, 0, 0, 1, 0, 0, 0], 
                        [0, 0, 1, 1, 1, 0, 0], 
                        [0, 1, 1, 1, 1, 1, 0], 
                        [1, 1, 1, 1, 1, 1, 1], 
                        [0, 1, 1, 1, 1, 1, 0], 
                        [0, 0, 1, 1, 1, 0, 0], 
                        [0, 0, 0, 1, 0, 0, 0]])

    kernel6 = np.array([[0, 0, 1, 1, 1, 0, 0], 
                        [0, 1, 1, 1, 1, 1, 0], 
                        [1, 1, 1, 1, 1, 1, 1], 
                        [1, 1, 1, 1, 1, 1, 1], 
                        [1, 1, 1, 1, 1, 1, 1], 
                        [0, 1, 1, 1, 1, 1, 0], 
                        [0, 0, 1, 1, 1, 0, 0]])

    kernel7 = np.array([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]])

    c0 = convolve(arr, kernel0, mode='constant')
    c0 = np.where(c0>=nnn, 0, 1)
    c1 = convolve(arr, kernel1, mode='constant')
    c1 = np.where(c1>=nnn, 0, 1)
    c2 = convolve(arr, kernel2, mode='constant')
    c2 = np.where(c2>=nnn, 0, 1)
    c3 = convolve(arr, kernel3, mode='constant')
    c3 = np.where(c3>=nnn, 0, 1)
    c4 = convolve(arr, kernel4, mode='constant')
    c4 = np.where(c4>=nnn, 0, 1)
    c5 = convolve(arr, kernel5, mode='constant')
    c5 = np.where(c5>=nnn, 0, 1 )
    c6 = convolve(arr, kernel6, mode='constant')
    c6 = np.where(c6>=nnn, 0, 1)
    c7 = convolve(arr, kernel7, mode='constant')
    print(np.max(c7))
    c7 = np.where(c7>=nnn, 0, 1)
    c = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7
    cf = np.where(temp != 0, c, np.nan)  
    gdalutils.writeRaster(inputPath, outPath, cf)
    return cf"""
    
def computeKNN_Metrices(city, attr_value, thres, evalPath, outputGT, outGT, evalFiles, ancillary_path):
    
    gt = calcConv(outputGT, thres, outGT, city, ancillary_path)
    
    metrics2eMAE = evalPath + '/{}_MAEconv.csv'.format(city)
    metrics2eProp = evalPath + '/{}_Propconv.csv'.format(city)
    metrics2eGT = evalPath + '/{}_GTconv.csv'.format(city)
    fileNames = []
    MAE_metrics = []
    Prop_metrics = []
    
    uniqueGT, countsGT = np.unique(gt, return_counts=True)
    b = [dict(zip(uniqueGT, np.round(countsGT/countsGT.sum(),decimals=2)))]
    convPath =  evalPath + "/tifs/conv"
    createFolder(convPath)
    
    for i in evalFiles:
        path = Path(i)
        fileName = path.stem 
        outPath = convPath + "/conv_{0}_{1}.tif".format(thres, fileName)
        pred = calcConv(i, thres, outPath, city, ancillary_path)
        unique, counts = np.unique(pred, return_counts=True)
        a = dict(zip(unique, np.round(counts/counts.sum(),decimals=2)))           
        
        gt = np.nan_to_num(gt, posinf=0, neginf=0, nan=0) 
        pred = np.nan_to_num(pred, posinf=0, neginf=0, nan=0) 
        r1, MAEdataset, std = mae_error(gt, pred) 

        fileNames.append(fileName)
        MAE_metrics.append("{0}/±{1}".format(r1,std))
        Prop_metrics.append(a)
    
    gtMetrics=[]
    gtMetrics.append(b)
    MAE_metrics.insert(0, "{0}_{1}".format(attr_value,thres))
    fileNames.insert(0, "Model")
    evalRs.writeMetrics(metrics2eMAE, fileNames, MAE_metrics,'MAE of NNN for {}'.format(city))
    
    Prop_metrics.insert(0, "{0}_{1}".format(attr_value,thres))
    gtMetrics.insert(0, 'gt')
    evalRs.writeMetrics(metrics2eProp, fileNames, Prop_metrics,'Prop of NNN for {}'.format(city))
    
    #gtMetrics.insert(0, 'gt')
    #evalRs.writeMetrics(metrics2eGT, fileNames, gtMetrics,'Prop of GT NNN for {}'.format(city))

    csvTOlatex(metrics2eMAE, evalPath + '/{}_MAEconv.tex'.format(city))  
    csvTOlatex(metrics2eProp, evalPath + '/{}_Propconv.tex'.format(city))
    
def plotKNN(city, thres, outputGT, outGT, evalFiles, evalPath, districtPath, polyPath, waterPath, ancillary_path):
    
    gt = calcConv(outputGT, thres, outGT, city, ancillary_path)
    convPath =  evalPath + "/tifs/conv"
    createFolder(convPath)
    
    convDifPath =  evalPath + "/tifs/dif"
    createFolder(convDifPath)
    for i in evalFiles:
        path = Path(i)
        fileName = path.stem 
        outPath = convPath + "//conv_{0}_{1}.tif".format(thres, fileName)
        pred = calcConv(i, thres, outPath, city, ancillary_path)
       
        dif = gt - pred
        
        outPathDif = convDifPath + "/convDif_{0}_{1}.tif".format(thres, fileName)
        gdalutils.writeRaster(path, outPathDif, dif)
        
        # Plot the population distribution of the predictions 
        exportPath = evalPath + "/convDif_{0}_{1}.png".format(thres, fileName)
        if not os.path.exists(exportPath):
            print("----- Step #1: Plotting Population Distribution -----")
            title ="'Spatial Error'\nDifference between the convolutioned GT & Pred\n({0}, KNN:{1})(2018)".format(fileName, thres)
            LegendTitle = "Kernels"
            src = rasterio.open(outPathDif)
            plot_map(city,'dif_KNN', src, exportPath, title, LegendTitle, districtPath = districtPath, neighPath = polyPath, waterPath = waterPath, invertArea = None, addLabels=True)     
    