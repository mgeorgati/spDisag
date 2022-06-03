import glob

from config.definitions import (ROOT_DIR, ancillary_path, pop_path,
                                gdal_rasterize_path, year)
from evalResultsNL import eval_Results_ams
from mainFunctions.basic import createFolder
from runDasymetricMapping import run_dasy
from runDisaggregation import run_disaggregation
from runPycnophylacticInterpolation import run_pycno
from verifyMassPreserving import verifyMassPreserv

#-------- GLOBAL ARGUMENTS --------
city='ams'
popraster = 'GHS_POP_100_near_cubicspline.tif'.format(city) 
key = 'Buurtcode' #'BU_CODE'
ancillary_path_case = ancillary_path +"{}".format(city)  

#-------- SELECT DEMOGRAPHIC GROUP OR LIST OF GROUPS --------
# If it is a list it will be calculated in multi-output model 
# Total Population : 'totalpop',
# 5 Age Groups : 'children', 'students','mobadults', 'nmobadults', 'elderly', 
# 7 Migrant Groups : 'sur', 'ant', 'mar','tur', 'nonwestern', 'western', 'autoch'

#'totalpop', 'students','mobadults', 'nmobadults', 'elderly', 'sur', 'ant', 'mar','tur', 'nonwestern', 'western', 'autoch'
attr_value = ['children', 'students','mobadults', 'nmobadults', 'elderly', 'sur', 'ant', 'mar','tur', 'nonwestern', 'western', 'autoch' ]   
  
#-------- SELECT PROCESS --------
#1. CALCULATE SIMPLE HEURISTIC ESTIMATES WITH PYCHNOPHYLACTIC OR DASYMETRIC MAPPING
run_Pycno = "no"
run_Dasy = "no"

#2. TRAIN REGRESSION MODEL (FURTHER CHOICES NEED TO BE DEFINED BELOW)
run_Disaggregation = "no"
    # 2.1. SELECT METHOD/MODEL 
        # aplm (linear model) 
        # aprf (random forest) 
        # apcatbr (Catboost Regressor)
    # 2.2. SELECT DISAGGREGATED METHOD TO BE USED AS INPUT
        # Pycno
        # Dasy
    # 2.3 SELECT ANCILLARY DATASET

# 3. VERIFY MASS PRESERVATION
verMassPreserv = "yes"

# 4. EVALUATE RESULTS
run_EvaluationGC_ams = "no"

def process_data(attr_value):
    createFolder(ROOT_DIR + "/Temp/{}/".format(city))
    createFolder(ROOT_DIR + "/TempRaster/{}/".format(city))
    createFolder(ROOT_DIR + "/TempCSV/{}/".format(city))
    if run_Pycno == "yes":
        createFolder(ROOT_DIR + "/Results/{}/Pycno/".format(city))
        ##### -------- PROCESS: RUN PYCNOPHYLACTIC -------- #####
        for i in attr_value:
            run_pycno(ROOT_DIR,ancillary_path, year, city, i, popraster, key)
    if run_Dasy == "yes":
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
        ##### -------- PROCESS: TRAIN REGRESSION MODEL  -------- #####
        methodopts = ['aprf'] # aplm (linear model), aprf (random forest), apcatbr (Catboost Regressor), apcnn (CNN), 'apmltr', 'aptfbtr' (Tensorflow BoostedTreesRegressor)
        ymethodopts = ['Dasy'] #'Pycno', Dasy# pycno, td, tdnoise10, td1pycno, average25p75td
        cnnmodelopts = ['unet'] # lenet, vgg, uenc, unet, 2runet (this and the following are only aplicable if method == CNN) 
        # The ancillary datasets are defined in runDisaggregation
        inputDataset = ['AIL1'] # 'AIL0', 'AIL1', 'AIL2','AIL3', 'AIL4', 'AIL5','AIL6', 'AIL7', #'AIL5',
        iterMax = 2
        for i in inputDataset:
            run_disaggregation(ancillary_path_case, ROOT_DIR, methodopts, ymethodopts, cnnmodelopts, city, year, attr_value, key, i, iterMax, gdal_rasterize_path)
    
    if verMassPreserv == "yes":
        ##### -------- PROCESS: VERIFY MASS PRESERVATION  -------- #####
        fshapea = ROOT_DIR + "/Shapefiles/{1}/{0}_{1}.shp".format(year,city)
        fcsv = ROOT_DIR + "/Statistics/{1}/{0}_{1}.csv".format(year,city) 
        ymethodopts = ['apcnn'] #'Dasy', 'Pycno', 'aprf'
        for ymethod in ymethodopts:
            if isinstance(attr_value, list):
                for i in attr_value:
                    evalList = glob.glob(ROOT_DIR + "/Results/{3}/{0}/dissever01*it10_{1}.tif".format(ymethod,i,ymethod.lower(),city))
                    print('lista', evalList)
                    #evalList = glob.glob(ROOT_DIR + "/Results/{0}/*_{1}_pycno.tif".format(ymethod,i))
                    csv_output = ROOT_DIR + '/Results/{1}/{3}/{0}_{1}_Eval_{2}.csv'.format(year,city,i,ymethod)
                    verifyMassPreserv(fshapea, city, fcsv, key, evalList, csv_output, i)
            else:
                evalList = glob.glob(ROOT_DIR + "/Results/{0}/dissever00*_it10_{1}.tif".format(ymethod,attr_value,ymethod.lower()))
                print(evalList)
                #evalList = glob.glob(ROOT_DIR + "/Results/{0}/*_{1}_pycno.tif".format(ymethod,i))
                csv_output = ROOT_DIR + '/Results/{3}/{0}_{1}_Eval_{2}.csv'.format(year,city,attr_value,ymethod)
                #verifyMassPreserv(fshapea, city, fcsv, key, evalList, csv_output,attr_value)
                
    if run_EvaluationGC_ams == "yes":
        pop_path_case = pop_path + "/{}/".format(city)
        if isinstance(attr_value, list):
            for i in attr_value:
                print("Evaluation possible")
                eval_Results_ams(ROOT_DIR, pop_path_case, ancillary_path_case, year, city, i)
        else:
            print("Evaluation Not possible")
            #eval_Results_ams(ROOT_DIR, pop_path_case, ancillary_path_case, year, city, attr_value)
    
    
process_data(attr_value)
