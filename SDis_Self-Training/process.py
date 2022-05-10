import glob

from config.definitions import (ROOT_DIR, ancillary_path,  python_scripts_folder_path,
                                pop_path, year)
#from evalResultsDK import eval_Results_cph
#from evalResultsNL import eval_Results_ams
from mainFunctions.basic import createFolder
from runDasymetricMapping import run_dasy
from runDisaggregation import run_disaggregation
from runPycnophylacticInterpolation import run_pycno
from verifyMassPreserving import verifyMassPreserv

def process_data(attr_value, city, popraster, key, run_Pycno, run_Dasy, run_Disaggregation, maxIters, methodopts, ymethodopts, inputDataset, verMassPreserv, run_Evaluation):
    ancillary_path_case = ancillary_path +"{}".format(city)
    
    createFolder(ROOT_DIR + "/Temp/{}/".format(city))
    createFolder(ROOT_DIR + "/TempRaster/{}/".format(city))
    createFolder(ROOT_DIR + "/TempCSV/{}/".format(city))
    
    if run_Pycno == "yes":
        createFolder(ROOT_DIR + "/Results/{}/Pycno/".format(city))
        ##### -------- PROCESS: RUN PYCNOPHYLACTIC -------- #####
        for i in attr_value:
            run_pycno(ROOT_DIR,ancillary_path, year, city, i, popraster, key)
    if run_Dasy == "yes":
        print(attr_value)
        createFolder(ROOT_DIR + "/Results/{}/Dasy/".format(city))
        ##### -------- PROCESS: RUN DASYMETRIC  -------- #####
        templateraster = '{}_template_100.tif'.format(city)
        if isinstance(attr_value, list):
            for i in attr_value:
                outputNameDasy = "/Results/{}/Dasy/".format(city) + str(year) + '_' + city + '_' + i + '_dasy.tif'
                run_dasy(ancillary_path, year, city, i, outputNameDasy, ROOT_DIR, popraster, key) 
        else:
            outputNameDasy = "/Results/{}/Dasy/".format(city) + str(year) + '_' + city + '_' + attr_value + '_dasy.tif'
            run_dasy(ancillary_path, year, city, attr_value, outputNameDasy, ROOT_DIR, popraster, key) 
    
    if run_Disaggregation == "yes":
        print(methodopts, ymethodopts, inputDataset)
        ##### -------- PROCESS: TRAIN REGRESSION MODEL  -------- #####
        #methodopts = ['aprf'] # aplm (linear model), aprf (random forest), apcatbr (Catboost Regressor), apcnn (CNN), 'apmltr', 'aptfbtr' (Tensorflow BoostedTreesRegressor)
        #ymethodopts = ['Dasy'] #'Pycno', Dasy# pycno, td, tdnoise10, td1pycno, average25p75td
        #cnnmodelopts = ['unet'] # lenet, vgg, uenc, unet, 2runet (this and the following are only aplicable if method == CNN) 
        # The ancillary datasets are defined in runDisaggregation
        #inputDataset = [ 'AIL1'] # 'AIL0', 'AIL1', 'AIL2','AIL3', 'AIL4', 'AIL5','AIL6', 'AIL7', #'AIL5',
        #iterMax = 2
        for i in inputDataset:
            run_disaggregation(ancillary_path_case, ROOT_DIR, methodopts, ymethodopts, city, year, attr_value, key, i, maxIters, python_scripts_folder_path)
    
    if verMassPreserv == "yes":
        ##### -------- PROCESS: VERIFY MASS PRESERVATION  -------- #####
        fshapea = ROOT_DIR + "/Shapefiles/{1}/{0}_{1}.shp".format(year,city)
        fcsv = ROOT_DIR + "/Statistics/{1}/{0}_{1}.csv".format(year,city) 
        ymethodopts = ['aprf'] #'Dasy', 'Pycno', 'aprf'
        for ymethod in ymethodopts:
            if isinstance(attr_value, list):
                for i in attr_value:
                    evalList = glob.glob(ROOT_DIR + "/Results/{3}/{0}/dissever01_*it7*{1}.tif".format(ymethod,i,ymethod.lower(),city))
                    #evalList = glob.glob(ROOT_DIR + "/Results/{0}/*_{1}_pycno.tif".format(ymethod,i))
                    csv_output = ROOT_DIR + '/Results/{3}/{0}_{1}_Eval_{2}.csv'.format(year,city,i,ymethod)
                    verifyMassPreserv(fshapea, fcsv, key, evalList, csv_output, i)
            else:
                evalList = glob.glob(ROOT_DIR + "/Results/{0}/dissever00*{1}*.tif".format(ymethod,attr_value,ymethod.lower()))
                print(evalList)
                #evalList = glob.glob(ROOT_DIR + "/Results/{0}/*_{1}_pycno.tif".format(ymethod,i))
                csv_output = ROOT_DIR + '/Results/{3}/{0}_{1}_Eval_{2}.csv'.format(year,city,attr_value,ymethod)
                verifyMassPreserv(fshapea, fcsv, key, evalList, csv_output,attr_value)
                
    if run_Evaluation == "yes":
        pop_path_case = pop_path + "/{}/".format(city)
        if isinstance(attr_value, list):
            for i in attr_value:
                print("Evaluation possible")
                if city == 'ams':
                    print("Evaluation Not possible")
                    #eval_Results_ams(ROOT_DIR, pop_path_case, ancillary_path_case, year, city, i)
                elif city == 'cph':
                    print("Evaluation Not possible")
                    #eval_Results_cph(ROOT_DIR, pop_path_case, ancillary_path_case, year, city, i)
        else:
            print("Evaluation Not possible")
            #eval_Results_ams(ROOT_DIR, pop_path_case, ancillary_path_case, year, city, attr_value)
    
    
#process_data(attr_value, city, popraster, key, run_Pycno, run_Dasy, run_Disaggregation, iterMax, methodopts, ymethodopts, inputDataset, verMassPreserv, run_Evaluation)
