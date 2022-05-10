import osgeoutils as osgu, nputils as npu
import geopandas as gpd, numpy as np
import os, collections
from pathlib import Path
from evaluateFunctions import rmse_error, mae_error, nrmse_error ,nmae_error, prop_error

def verifyMassPreserv(fshapea, fcsv, key, evalList, csv_output, attr_value):
    """[Returns difference between predictions and ground truth data]
    ##### ----- MODIFY CSV FOR EACH CASE ----- #####

    Args:
        fshapea ([str]): [directory to shapefile]
        fcsv ([str]): [directory to csv]
        key ([str]): [common key between fshapea and fcsv]
        evalList ([list]): [list of directories to file to be evaluated]
        ROOT_DIR ([str]): [parent directory]
    """
    fshape = osgu.copyShape(fshapea, 'verifymass')
    osgu.addAttr2Shapefile(fshape, fcsv, [key], encoding='utf-8')

    for i in evalList:
        fileName = Path(i).stem
        method = fileName.split("_",3)[-1]
        print(fileName, method)
        raster, rastergeo = osgu.readRaster(i)
        nrowsds = raster.shape[1]
        ncolsds = raster.shape[0]

        idsdataset = osgu.ogr2raster(fshape, attr='ID', template=[rastergeo, nrowsds, ncolsds])[0]
        polygonvaluesdataset = osgu.ogr2raster(fshape, attr=attr_value, template=[rastergeo, nrowsds, ncolsds])[0]

        # Summarizes the values within each polygon
        stats_predicted = npu.statsByID(raster, idsdataset, 'sum')
        stats_true = npu.polygonValuesByID(polygonvaluesdataset, idsdataset)

        predicted = np.fromiter(collections.OrderedDict(sorted(stats_predicted.items())).values(), dtype=float)
        true = np.fromiter(collections.OrderedDict(sorted(stats_true.items())).values(), dtype=float)
        diff = predicted - true
        if np.abs(np.mean(diff)) > (0.00001 * np.mean(true)):
            mae = np.abs(np.mean(diff))
            maxVal = 0.00001 * np.mean(true)
            print('Problem with file', fileName)
            print('- Difference:', np.abs(np.mean(diff)))
            print('- Max value:', 0.00001 * np.mean(true))

            print("-----CALCULATING METRICS-----")
            actSum = np.nansum(true)
            predSum =np.nansum(predicted)
            
            r1 = mae_error(true, predicted)[0]
            r2 = round(rmse_error(true, predicted), 1)
            r3 = round(nmae_error(true, predicted), 4)
            r4 = round(nrmse_error(true, predicted), 4)
            r5 = prop_error(true, predicted)[0]
            #r6 = div_error(true, predicted)[0]
            
            stdtrue = round(np.std(true, dtype=np.float64),3)
            stdPred = round(np.std(predicted, dtype=np.float64),3)
            print("-----WRITING CSV-----")
            # Calculate the mean difference and the quotient of total grid cells and write it in csv
            filenamemetrics2e = csv_output
            if os.path.exists(filenamemetrics2e):
                with open(filenamemetrics2e, 'a') as myfile:
                    myfile.write(method + ';' + str(r1) + ';'+ str(r2) + ';' + str(r3) + ';' + str(r4) + ';' + str(r5) + ';' 
                                 + str(actSum) + ';' + str(predSum) + ';' + str(stdtrue) + ';' + str(stdPred) + '\n')       
            else:
                with open(filenamemetrics2e, 'w+') as myfile:
                    myfile.write('Comparison among the predictions and the ground truth data for the Municipality of Amsterdam\n')
                    myfile.write('Method;MAE;RMSE;MAEMEAN;RMSEMEAN;PrE;PE;ActualSum;PredictedSum;ActualSTD;PredictedSTD\n')
                    myfile.write(method + ';' + str(r1) + ';'+ str(r2) + ';' + str(r3) + ';' + str(r4) + ';' + str(r5) + ';' 
                                + str(actSum) + ';' + str(predSum) + ';' + str(stdtrue) + ';' + str(stdPred) + '\n')
        else:
            print("----- ALL GOOD FOR {} -----\n YOU MAY PROCEED WITH THE ANALYSIS!".format(attr_value))    
    osgu.removeShape(fshape)


