import numpy as np, rasterio
from scipy.stats.stats import pearsonr 
from evaluating import evalRs
from pathlib import Path

def corCSV(attr_value, outputGT, evalFiles, city, evalPath):
       
    metrics2eCOR = evalPath + '/{}_COR.csv'.format(city)

    fileNames = []
    COR_metrics = []
    
    gr = rasterio.open(outputGT)
    arrayGR = gr.read(1)
    for i in range(len(evalFiles)):
        file = evalFiles[i]
        path = Path(file)
        fileName = path.stem
        
        pr = rasterio.open(path)
        arrayPR = pr.read(1)
    
        arrayGR = np.where(arrayGR!= np.nan , arrayGR, 0)
        arrayPR = np.nan_to_num(arrayPR, nan=0, posinf=0)
        cor = np.round(np.nanmean(pearsonr(arrayPR.ravel(),arrayGR.ravel())),2)
        
        fileNames.append(fileName)
        COR_metrics.append(cor)
          
    COR_metrics.insert(0, attr_value)
    
    fileNames.insert(0, "Model")

    evalRs.writeMetrics(metrics2eCOR, fileNames, COR_metrics, 'Correlation for {}'.format(city))
    

    