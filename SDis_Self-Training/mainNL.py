from config.definitions import ROOT_DIR, ancillary_path, pop_path, year, python_scripts_folder_path
from evalResultsNL import eval_Results_GC_ams
from mainFunctions.basic import createFolder
from runDasymetricMapping import run_dasy
#from verifyMassPreserving import verifyMassPreserv
from runDisaggregation import run_disaggregation
from runPycnophylacticInterpolation import run_pycno
import glob
city='ams'
#-------- GLOBAL ARGUMENTS --------
popraster = 'GHS_POP_100_near_cubicsplineWaterIESM_new.tif'.format(city) #### <<<< ----- GHS POP ----- >>>> #### --> FIX AFTER EVALUATION OF DISAGGREGATION
key = 'Buurtcode' #'BU_CODE'
        
#-------- PROCESS: GHS RPREPARATION --------
run_Pycno = "no"
run_Dasy = "no"
run_Disaggregation = "no"
verMassPreserv = "no"
run_EvaluationGC_ams = "yes"
#'totalpop','children', 'students','mobadults', 'nmobadults', 'elderly', 'sur', 'ant', 'mar','tur', 'nonwestern','western', 'autoch'
attr_value = ['totalpop']

def process_data(attr_value):
    createFolder(ROOT_DIR + "/Temp/")
    createFolder(ROOT_DIR + "/TempRaster/")
    createFolder(ROOT_DIR + "/TempCSV/")
    if run_Pycno == "yes":
        ##### -------- PROCESS: RUN PYCNOPHYLACTIC -------- #####
        for i in attr_value:
            run_pycno(year, city, i, popraster, key)
    if run_Dasy == "yes":
        ##### -------- PROCESS: RUN DASYMETRIC  -------- #####
        templateraster = '{}_template_100.tif'.format(city)
        if isinstance(attr_value, list):
            for i in attr_value:
                run_dasy(year, city, i, popraster,  key) 
        else:
            run_dasy(year, city, attr_value, popraster,  key) 
    
    if run_Disaggregation == "yes":
        methodopts = ['aprf'] # aplm (linear model), aprf (random forest), apxgbtree (XGBoost), apcnn (CNN), 'apcatbr' (Catboost Regressor), 'apmltr', 'aptfbtr' (Tensorflow BoostedTreesRegressor)
        ymethodopts = ['Dasy'] #'Pycno', Dasy# pycno, td, tdnoise10, td1pycno, average25p75td
        cnnmodelopts = ['unet'] # lenet, vgg, uenc, unet, 2runet (this and the following are only aplicable if method == CNN) 
        #'GHS_ESM_corine': '8AIL0', 'GHS_ESM_corine_transp':12AIL1, 'GHS_ESM_corine_transpA': 12AIL2
        inputDataset = [ 'AIL12'] # 'AIL0', 'AIL1', 'AIL2','AIL3', 'AIL4', 'AIL5','AIL6', 'AIL7', #'AIL5',
        iterMax = 10
        for i in inputDataset:
            for k in attr_value:
                run_disaggregation(methodopts, ymethodopts, cnnmodelopts, city, year, k, key, i, iterMax, python_scripts_folder_path)
    """
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
                print("kati edi")
                evalList = glob.glob(ROOT_DIR + "/Results/{0}/dissever00*{1}*.tif".format(ymethod,attr_value,ymethod.lower()))
                print(evalList)
                #evalList = glob.glob(ROOT_DIR + "/Results/{0}/*_{1}_pycno.tif".format(ymethod,i))
                csv_output = ROOT_DIR + '/Results/{3}/{0}_{1}_Eval_{2}.csv'.format(year,city,attr_value,ymethod)
                verifyMassPreserv(fshapea, fcsv, key, evalList, csv_output,attr_value)
    """            
    if run_EvaluationGC_ams == "yes":
        if isinstance(attr_value, list):
            for i in attr_value:
                
                eval_Results_GC_ams(ROOT_DIR, pop_path, ancillary_path, year, city, i)
        else:
            
            eval_Results_GC_ams(ROOT_DIR, pop_path, ancillary_path, year, city, attr_value)
    
    
process_data(attr_value)
