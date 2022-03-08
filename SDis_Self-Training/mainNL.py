import glob

from config.definitions import (ROOT_DIR, ancillary_path, pop_path,
                                python_scripts_folder_path, year)
#from evalResultsNL import eval_Results_ams
from mainFunctions.basic import createFolder
from runDasymetricMapping import run_dasy
from runDisaggregation import run_disaggregation
from runPycnophylacticInterpolation import run_pycno
from verifyMassPreserving import verifyMassPreserv

#-------- GLOBAL ARGUMENTS --------
city='ams'
popraster = 'GHS_POP_100_near_cubicsplineWaterIESM_new.tif'.format(city) 
key = 'Buurtcode' #'BU_CODE'
ancillary_path_case = ancillary_path +"{}".format(city)     
  
#-------- PROCESS: GHS RPREPARATION --------
run_Pycno = "no"
run_Dasy = "no"
run_Disaggregation = "yes"
verMassPreserv = "no"
run_EvaluationGC_ams = "no"

#-------- SELECT DEMOGRAPHIC GROUP OR LIST OF GROUPS --------
# If it is a list it will be calculated in multi-output model 
# Total Population : 'totalpop',
# 5 Age Groups : 'children', 'students','mobadults', 'nmobadults', 'elderly', 
# 7 Migrant Groups : 'sur', 'ant', 'mar','tur', 'nonwestern', 'western', 'autoch'

#'students','mobadults', 'nmobadults', 'elderly', 'sur', 'ant', 'mar','tur', 'nonwestern', 'western', 'autoch'
attr_value = ['children' ]
#print(attr_value[0:5], attr_value[-7:])

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
        methodopts = ['apcnn'] # aplm (linear model), aprf (random forest), apxgbtree (XGBoost), apcnn (CNN), 'apcatbr' (Catboost Regressor), 'apmltr', 'aptfbtr' (Tensorflow BoostedTreesRegressor)
        ymethodopts = ['Dasy'] #'Pycno', Dasy# pycno, td, tdnoise10, td1pycno, average25p75td
        cnnmodelopts = ['unet'] # lenet, vgg, uenc, unet, 2runet (this and the following are only aplicable if method == CNN) 
        #'GHS_ESM_corine': '8AIL0', 'GHS_ESM_corine_transp':12AIL1, 'GHS_ESM_corine_transpA': 12AIL2
        inputDataset = [ 'AIL12'] # 'AIL0', 'AIL1', 'AIL2','AIL3', 'AIL4', 'AIL5','AIL6', 'AIL7', #'AIL5',
        iterMax = 2
        for i in inputDataset:
            for k in attr_value:
                run_disaggregation(ancillary_path_case, ROOT_DIR, methodopts, ymethodopts, cnnmodelopts, city, year, k, key, i, iterMax, python_scripts_folder_path)
    
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
                    #verifyMassPreserv(fshapea, fcsv, key, evalList, csv_output, i)
            else:
                
                evalList = glob.glob(ROOT_DIR + "/Results/{0}/dissever00*{1}*.tif".format(ymethod,attr_value,ymethod.lower()))
                print(evalList)
                #evalList = glob.glob(ROOT_DIR + "/Results/{0}/*_{1}_pycno.tif".format(ymethod,i))
                csv_output = ROOT_DIR + '/Results/{3}/{0}_{1}_Eval_{2}.csv'.format(year,city,attr_value,ymethod)
                #verifyMassPreserv(fshapea, fcsv, key, evalList, csv_output,attr_value)
                
    if run_EvaluationGC_ams == "yes":
        pop_path_case = pop_path + "/{}/".format(city)
        if isinstance(attr_value, list):
            for i in attr_value:
                print("Evaluation Not possible")
                #eval_Results_ams(ROOT_DIR, pop_path_case, ancillary_path_case, year, city, i)
        else:
            print("Evaluation Not possible")
            #eval_Results_ams(ROOT_DIR, pop_path_case, ancillary_path_case, year, city, attr_value)
    
    
process_data(attr_value)
