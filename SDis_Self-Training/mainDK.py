import glob

from config.definitions import (ROOT_DIR, ancillary_path, pop_path,
                                python_scripts_folder_path, year, city)
#from evalResults import eval_Results_GC
from mainFunctions.basic import createFolder
#from runPycnophylacticInterpolation import run_pycno
#from runDasymetricMapping import run_dasy
from runDisaggregation import run_disaggregation
#from verifyMassPreserving import verifyMassPreserv


#-------- GLOBAL ARGUMENTS --------
popraster = ancillary_path + '/GHS/GHS_POP_100_near_cubicsplineWaterIESM_new.tif'
key = 'KOMNAME' #'BU_CODE'
        
#-------- PROCESS: GHS RPREPARATION --------
run_Pycno = "no"
run_Dasy = "no"
run_Disaggregation = "yes"
verMassPreserv = "no"
run_EvaluationGC = "no"

#attr_value = 'totalpop'
attr_value = [ 'children', 'students', 'mobadults', 'nmobadults', 'elderly',
                'AuNZ', 'CentAsia', 'EastAsia','EastEur', 'LAC', 'Melanesia', 'Micronesia', 'NorthAfr', 
                'NorthAm', 'NorthEur', 'OTH', 'Polynesia', 'SEastAsia', 'SouthAsia', 'SouthEur', 'STA', 'SubSahAfr', 
                'WestAsia', 'WestEur','DNK', 
                'EU', 'notEU'] 

def process_data(attr_value):
    createFolder(ROOT_DIR + "/Temp/")
    createFolder(ROOT_DIR + "/TempRaster/")
    createFolder(ROOT_DIR + "/TempCSV/")
    """if run_Pycno == "yes":
        
        createFolder(ROOT_DIR + "/Results/{}/Pycno".format(city))
        ##### -------- PROCESS: RUN PYCNOPHYLACTIC -------- #####
        if isinstance(attr_value, list):ls
        
            for i in attr_value:
                run_pycno(ROOT_DIR, year, city, i, popraster, key)
        else: run_pycno(ROOT_DIR, year, city, attr_value, popraster, key)
    if run_Dasy == "yes":
        ##### -------- PROCESS: RUN DASYMETRIC  -------- #####
        createFolder(ROOT_DIR + "/Results/{}/Dasy".format(city))
        if isinstance(attr_value, list):
            for i in attr_value:
                outputNameDasy = "/Results/{}/Dasy/".format(city) + str(year) + '_' + city + '_' + i + '_dasyWIN.tif'
                run_dasy(year, city, i, outputNameDasy, ROOT_DIR, popraster, key) 
        else:
            outputNameDasy = "/Results/{}/Dasy/".format(city) + str(year) + '_' + city + '_' + attr_value + '_dasyWIN.tif'
            run_dasy(year, city, attr_value,outputNameDasy, ROOT_DIR, popraster, key)
    """
    if run_Disaggregation == "yes":
        methodopts = ['apcatbr'] # aplm (linear model), aprf (random forest), apxgbtree (XGBoost), apcnn (CNN), 'apcatbr' (Catboost Regressor), 'apmltr'
        createFolder(ROOT_DIR + "/Results/{0}/{1}".format(city, methodopts[0] ))
        ymethodopts = ['Dasy'] #'Pycno', Dasy# pycno, td, tdnoise10, td1pycno, average25p75td
        cnnmodelopts = ['unet'] # lenet, vgg, uenc, unet, 2runet (this and the following are only aplicable if method == CNN) 
        #'GHS_ESM_corine': '8AIL0', 'GHS_ESM_corine_transp':12AIL1, 'GHS_ESM_corine_transpA': 12AIL2
        inputDataset = ['AIL8'] # 'AIL0', 'AIL1', 'AIL2'
        iterMax = 2
        for i in inputDataset:
            run_disaggregation(methodopts, ymethodopts, cnnmodelopts, city, year, attr_value, key, i, iterMax, python_scripts_folder_path)
    """
    if verMassPreserv == "yes":
        ##### -------- PROCESS: VERIFY MASS PRESERVATION  -------- #####
        fshapea = ROOT_DIR + "/Shapefiles/{1}/{0}_{1}.shp".format(year,city)
        fcsv = ROOT_DIR + "/Statistics/{1}/{0}_{1}.csv".format(year,city) 
        ymethodopts = ['aprf'] #'Dasy', 'Pycno', 'aprf'
        for ymethod in ymethodopts:
            if isinstance(attr_value, list):
                for i in attr_value:
                    evalList = glob.glob(ROOT_DIR + "/Results/{3}/{0}/*_{1}_*.tif".format(ymethod,i,ymethod.lower(),city))
                    #evalList = glob.glob(ROOT_DIR + "/Results/{0}/*_{1}_pycno.tif".format(ymethod,i))
                    csv_output = ROOT_DIR + '/Results/{1}/{3}/{0}_{1}_Eval_{2}.csv'.format(year,city,i,ymethod)
                    verifyMassPreserv(fshapea, fcsv, key, evalList, csv_output, i)
            else:
                evalList = glob.glob(ROOT_DIR + "/Results/{3}/{0}/*.tif".format(ymethod,attr_value,ymethod.lower(),city))
                print(evalList)
                #evalList = glob.glob(ROOT_DIR + "/Results/{0}/*_{1}_pycno.tif".format(ymethod,i))
                csv_output = ROOT_DIR + '/Results/{1}/{3}/{0}_{1}_Eval_{2}.csv'.format(year,city,attr_value,ymethod)
                verifyMassPreserv(fshapea, fcsv, key, evalList, csv_output,attr_value)
    """
    """           
    if run_EvaluationGC == "yes":
        #EVALUATION OF THE PREDICTIONS WITH THE GROUND TRUTH DATA OF MNC DATASET
        if isinstance(attr_value, list):
            for i in attr_value:
                eval_Results_GC(ROOT_DIR, pop_path, ancillary_path, year, city, i)
        else: eval_Results_GC(ROOT_DIR, pop_path, ancillary_path, year, city, attr_value)
    """
process_data(attr_value)

