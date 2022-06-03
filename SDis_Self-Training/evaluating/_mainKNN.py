from fileinput import filename
import os, numpy as np
from pathlib import Path
from attr import attr

import matplotlib.pyplot as plt
import rasterio
from config import ROOT_DIR, pop_path, year, ancillary_path
from utils import gdalutils
from evaluating import calcConv, mae_error, evalResultsNL
from plotting import plot_map, plotMatrix
from mainFunctions import csvTOlatex

city='ams'
evalPath ="C:/Users/NM12LQ/OneDrive - Aalborg Universitet/Dasymetric_Mapping/SDis_Self-Training/Evaluation/{}_KNN/".format(city)

cnnPath = "C:/Users/NM12LQ/OneDrive - Aalborg Universitet/Dasymetric_Mapping/SDis_Self-Training/Results/{}/apcnn/".format(city)
gbPath = "C:/Users/NM12LQ/OneDrive - Aalborg Universitet/Dasymetric_Mapping/SDis_Self-Training/Results/AgilePaper/{}/apcatbr/".format(city)
districtPath = 'C:/Users/NM12LQ/OneDrive - Aalborg Universitet/Dasymetric_Mapping/AncillaryData/{0}/adm/{0}_districts.geojson'.format(city)


# Required files for plotting
polyPath = ROOT_DIR + "/Shapefiles/{1}/{0}_{1}.shp".format(year,city)
districtPath = ancillary_path + '/{0}/adm/{0}_districts_west.geojson'.format(city)    
waterPath = ancillary_path + '{0}/corine/waterComb_{0}_CLC_2012_2018.tif'.format(city)


    
    
def computeKNN(ROOT_DIR, year, city, attr_value, thres):
    outputGT =  pop_path + "/{1}/GridCells/rasters/{0}_{1}_{2}.tif".format(year,city,attr_value)
    
    print("----- Select prediction files to be evaluated -----") 
    evalFiles = [ROOT_DIR + "/Results/{0}/apcnn/dissever01_RMSE_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{1}.tif".format(city,attr_value),
                 #ROOT_DIR + "/Results/{0}/apcnn/dissever01_CLF_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{1}.tif".format(city,attr_value),
                 #ROOT_DIR + "/Results/{0}/apcnn/dissever01_RMSEflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{1}.tif".format(city,attr_value),
                 #ROOT_DIR + "/Results/{0}/apcnn/dissever01_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{1}.tif".format(city,attr_value),
                 #ROOT_DIR + "/Results/{0}/apcnn/dissever01_1_CLFflipped2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{1}.tif".format(city,attr_value),
                 #ROOT_DIR + "/Results/{0}/apcnn/dissever01_2_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{1}.tif".format(city,attr_value),
                 #ROOT_DIR + "/Results/{0}/apcatbr/dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{1}.tif".format(city,attr_value)
                 ]
    outGT = pop_path + '/{1}/GridCells/convolutions/conv_{0}_{1}_{2}.tif'.format(year,city,attr_value)
    gt = calcConv(outputGT, thres, outGT)
    
    
    for i in evalFiles:
        path = Path(i)
        fileName = path.stem 
        outPath = evalPath + "/tifs/conv/conv_{0}_{1}.tif".format(thres, fileName)
        pred = calcConv(i, thres, outPath)
       
        dif = gt - pred
        
        outPathDif = evalPath + "/tifs/dif/convDif_{0}_{1}.tif".format(thres, fileName)
        gdalutils.writeRaster(path, outPathDif, dif)
        
        # Plot the population distribution of the predictions 
        exportPath = evalPath + "/convDif_{0}_{1}.png".format(thres, fileName)
        if not os.path.exists(exportPath):
            print("----- Step #1: Plotting Population Distribution -----")
            title ="'Spatial Error'\nDifference between the convolutioned GT & Pred\n({0}, KNN:{1})(2018)".format(fileName, thres)
            LegendTitle = "Kernels"
            src = rasterio.open(outPathDif)
            plot_map(city,'dif_KNN', src, exportPath, title, LegendTitle, districtPath = districtPath, neighPath = polyPath, waterPath = waterPath, invertArea = None, addLabels=True)     
    
attr_value = ['nonwestern', 'tur', 'mar']
thres = [10, 25, 50, 100, 200]
for i in attr_value:
    for k in thres:       
        computeKNN(ROOT_DIR, year, city, i, k)


    # Population distribution
    e= [os.path.dirname(os.path.dirname(evalPath)) + "/{0}_GT/{0}_GT_{1}.png".format(city,i),
        evalPath + "/convDif_10_dissever01_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{0}.png".format(i, thres),
        #evalPath + "/convDif_10_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{0}.png".format(i, thres),
        evalPath + "/convDif_25_dissever01_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{0}.png".format(i, thres),
        #evalPath + "/convDif_25_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{0}.png".format(i, thres),
        evalPath + "/convDif_50_dissever01_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{0}.png".format(i, thres),
        #evalPath + "/convDif_50_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{0}.png".format(i, thres),
        evalPath + "/convDif_100_dissever01_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{0}.png".format(i, thres),
        #evalPath + "/convDif_100_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{0}.png".format(i, thres),
        evalPath + "/convDif_200_dissever01_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{0}.png".format(i, thres),
        #evalPath + "/convDif_200_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{0}.png".format(i, thres)
        ]
    """e= [os.path.dirname(os.path.dirname(evalPath)) + "/{0}_GT/{0}_GT_{1}.png".format(city,i),
        evalPath + "/convDif_{1}_dissever01_RMSE_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{0}.png".format(i, thres),
        #evalPath + "/apcnn/dissever01_CLF_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{}.png".format(i),
        evalPath + "/convDif_{1}_dissever01_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{0}.png".format(i, thres),
        evalPath + "/convDif_{1}_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{0}.png".format(i, thres)]"""
    path = evalPath + "/matrices/convDif_CLFflipped_{0}.png".format(i, thres)
    title = "Spatial Error: {0}".format(i)
    #plotMatrix(e, path, title) 