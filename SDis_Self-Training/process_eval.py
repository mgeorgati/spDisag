import glob
import os
from datetime import datetime
from pathlib import Path
from config import ROOT_DIR, ancillary_path, pop_path, year
from evaluating import calcKNN, corCSV, evalRs
from mainFunctions import createFolder
from verifyMassPreserving import verifyMassPreserv


def process_eval(attr_value, city, key, methodopts, verMassPreserv, run_Evaluation, calc_Metrics, calc_Corr, 
                 plot_Scatter, plot_evalMaps, calc_Metrics_knn, plot_evalMaps_knn, plot_Matrices):
    ancillary_path_case = ancillary_path +"{}".format(city)
        
    if verMassPreserv == "yes":
        ##### -------- PROCESS: VERIFY MASS PRESERVATION  -------- #####
        fshapea = ROOT_DIR + "/Shapefiles/{1}/{0}_{1}.shp".format(year,city)
        fcsv = ROOT_DIR + "/Statistics/{1}/{0}_{1}.csv".format(year,city) 
        #ymethodopts = ['aprf'] #'Dasy', 'Pycno', 'aprf'
        for ymethod in methodopts:
            if isinstance(attr_value, list):
                for i in attr_value:
                    evalList = glob.glob(ROOT_DIR + "/Results/{3}/{0}/dissever01_*it10_{1}.tif".format(ymethod,i,ymethod.lower(),city))
                    #evalList = glob.glob(ROOT_DIR + "/Results/{0}/*_{1}_pycno.tif".format(ymethod,i))
                    csv_output = ROOT_DIR + '/Results/{1}/{3}/{1}_Eval_{2}.csv'.format(year,city,i,ymethod)
                    verifyMassPreserv(fshapea, city, fcsv, key, evalList, csv_output, i)
            else:
                evalList = glob.glob(ROOT_DIR + "/Results/{0}/dissever00*{1}*.tif".format(ymethod,attr_value,ymethod.lower()))
                print(evalList)
                #evalList = glob.glob(ROOT_DIR + "/Results/{0}/*_{1}_pycno.tif".format(ymethod,i))
                csv_output = ROOT_DIR + '/Results/{1}/{3}/{1}_Eval_{2}.csv'.format(year,city,attr_value,ymethod)
                verifyMassPreserv(fshapea, city, fcsv, key, evalList, csv_output,attr_value)
                
    if run_Evaluation == "yes":       
        date = datetime.now().strftime("%Y%m%d")
        
        for i in attr_value:
            print("Evaluating {0}, {1}".format(city, i))
            if city == 'ams':
                evalFiles = [ROOT_DIR + "/Results/{0}/apcnn/dissever01_RMSE_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{1}.tif".format(city,i),
                    #ROOT_DIR + "/Results/{0}/apcnn/dissever01_CLF_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{1}.tif".format(city,i),
                    ROOT_DIR + "/Results/{0}/apcnn/dissever01_RMSEflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{1}.tif".format(city,i),
                    #ROOT_DIR + "/Results/{0}/apcnn/dissever01_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{1}.tif".format(city,i),
                    #ROOT_DIR + "/Results/{0}/apcnn/dissever01_1_CLFflipped2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{1}.tif".format(city,i),
                    #ROOT_DIR + "/Results/{0}/apcnn/dissever01_2_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{1}.tif".format(city,i),
                    #ROOT_DIR + "/Results/{0}/apcnn/dissever01_true_CLFflipped_2018_ams_Dasy_16unet_10epochspi_AIL12_it10_{1}.tif".format(city,i),
                    ROOT_DIR + "/Results/{0}/apcatbr/dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{1}.tif".format(city,i)
                    ]
                
                aggr_outfileSUMGT = ROOT_DIR + "/Shapefiles/Comb/{0}_{1}_ois.geojson".format(year,city)

                outputGT =  pop_path + "/{1}/GridCells/rasters/{0}_{1}_{2}.tif".format(year,city,i)
                # Required files for plotting
                polyPath = ROOT_DIR + "/Shapefiles/{1}/{0}_{1}.shp".format(year,city)
                districtPath = ancillary_path + '/{0}/adm/{0}_districts_west.geojson'.format(city)    
                waterPath = ancillary_path + '{0}/corine/waterComb_{0}_CLC_2012_2018.tif'.format(city)
                invertArea = ancillary_path + '/{0}/adm/grootams_noAms.geojson'.format(city)
                
                outGT = pop_path + '/{1}/GridCells/convolutions/conv_{0}_{1}_{2}.tif'.format(year,city,i)
                thres = [50, 100, 200, 500]
            if city == 'cph':
                evalFiles = [ROOT_DIR + "/Results/{0}/apcnn/dissever01_RMSE_2018_{0}_Dasy_16unet_10epochspi_AIL12_it10_{1}.tif".format(city,i),
                    #ROOT_DIR + "/Results/{0}/apcnn/dissever01_CLF_2018_{0}_Dasy_16unet_10epochspi_AIL12_it10_{1}.tif".format(city,i),
                    ROOT_DIR + "/Results/{0}/apcnn/dissever01_RMSEflipped_2018_{0}_Dasy_16unet_10epochspi_AIL12_it10_{1}.tif".format(city,i),
                    #ROOT_DIR + "/Results/{0}/apcnn/dissever01_CLFflipped_2018_{0}_Dasy_16unet_10epochspi_AIL12_it10_{1}.tif".format(city,i),
                    #ROOT_DIR + "/Results/{0}/apcnn/dissever01_1_CLFflipped2018_{0}_Dasy_16unet_10epochspi_AIL12_it10_{1}.tif".format(city,i),
                    #ROOT_DIR + "/Results/{0}/apcnn/dissever01_2_CLFflipped_2018_{0}_Dasy_16unet_10epochspi_AIL12_it10_{1}.tif".format(city,i),
                    #ROOT_DIR + "/Results/{0}/apcnn/dissever01_true_CLFflipped_2018_{0}_Dasy_16unet_10epochspi_AIL12_it10_{1}.tif".format(city,i),
                    ROOT_DIR + "/Results/{0}/apcatbr/dissever01WIESMN_500_2018_{0}_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{1}.tif".format(city,i)
                    ]
                #evalRsDK.eval_Results_cph(ROOT_DIR, pop_path, ancillary_path_case, year, city, i)
            
            if calc_Metrics == "yes":
                print('----- Plot Metrices -----')
                evalPath = ROOT_DIR + "/Evaluation/{0}_{1}/".format(city, 'metrices') #date

                if not os.path.exists(evalPath):
                    createFolder(evalPath)
                evalRs.eval_Results_Metrics(evalFiles, outputGT, city, i, evalPath)
            
            if calc_Corr == "yes":
                print('----- Plot Correlations -----')
                evalPath = ROOT_DIR + "/Evaluation/{0}_{1}/".format(city, 'metrices')
                corCSV(i, outputGT, evalFiles, city, evalPath)
            
            if plot_Scatter == "yes":
                print('----- Plot Scatterplots -----')
                scatterPath = ROOT_DIR + "/Evaluation/{0}_scatterplots1/".format(city) #date
                if not os.path.exists(scatterPath):
                    createFolder(scatterPath)
                evalRs.eval_Scatterplots(scatterPath, outputGT, evalFiles, city, attr_value, scatterPath)
            
            if plot_evalMaps== "yes":
                evalPath = ROOT_DIR + "/Evaluation/{0}_{1}/".format(city, 'west') #date
                if not os.path.exists(evalPath):
                    createFolder(evalPath)
                    
                exportPathGT = ROOT_DIR + "/Evaluation/{0}_GT/{0}_{1}/".format(city, 'west') #date
                if not os.path.exists(exportPathGT):
                    createFolder(exportPathGT)
                
                scatterPath = ROOT_DIR + "/Evaluation/{0}_scatterplots/".format(city) #date
                if not os.path.exists(scatterPath):
                    createFolder(scatterPath)
                #Plot Maps
                evalRs.eval_Results(evalPath, outputGT, exportPathGT, polyPath, districtPath, waterPath, invertArea, evalFiles, year, city, i, aggr_outfileSUMGT, scatterPath)
            
            for k in thres:
                if calc_Metrics_knn == "yes":
                    evalPath = ROOT_DIR + "/Evaluation/{0}_{1}/".format(city, 'knnGF') #date
                    if not os.path.exists(evalPath):
                        createFolder(evalPath)
                    calcKNN.computeKNN_Metrices(city, i, k, evalPath, outputGT, outGT, evalFiles, ancillary_path)
                
                if plot_evalMaps_knn == "yes":
                    evalPath = ROOT_DIR + "/Evaluation/{0}_knnGF/{0}_{1}/".format(city, 'knn_west') #date
                    if not os.path.exists(evalPath):
                        createFolder(evalPath)
                    calcKNN.plotKNN(city, k, outputGT, outGT, evalFiles, evalPath, districtPath, polyPath, waterPath, ancillary_path)
            
            if plot_Matrices == "yes" :
                for x in range(len(evalFiles)):
                    file = evalFiles[x]
                    path = Path(file)
                    fileName = path.stem   
                    method = path.parent.stem 
                    print(path, fileName)
                    exportPathGT = ROOT_DIR + "/Evaluation/{0}_GT/{0}_{1}/".format(city, 'west')
                    evalPath = ROOT_DIR + "/Evaluation/{0}_{1}/".format(city, 'west') #date
                    evalPathKNN = ROOT_DIR + "/Evaluation/{0}_knn/{0}_{1}/".format(city, 'knn_west')
                    evalPathKNNgf = ROOT_DIR + "/Evaluation/{0}_knnGF/{0}_{1}/".format(city, 'knn_west')
                    #evalPathMatrices = ROOT_DIR + "/Evaluation/{0}_{1}/west_{2}.png".format(city, 'matrices', fileName)
                    
                    scatterPath = ROOT_DIR + "/Evaluation/{0}_scatterplots/".format(city)
                    evalPathMatrices = ROOT_DIR + "/Evaluation/{0}_{1}/combScatterplots.png".format(city, 'matrices', fileName)
                    
                    if not os.path.exists(ROOT_DIR + "/Evaluation/{0}_{1}".format(city, 'matrices')):
                        createFolder(ROOT_DIR + "/Evaluation/{0}_{1}".format(city, 'matrices'))
                    # Plot matrces
                    evalRs.createMatrices(evalPath, city, i, evalPathKNN, evalPathKNNgf, evalPathMatrices, exportPathGT, fileName, method, scatterPath)
                    
    else:
        print("Nothing selected")
