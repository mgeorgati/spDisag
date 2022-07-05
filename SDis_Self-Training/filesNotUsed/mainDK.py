import glob

from config.definitions import (ROOT_DIR, ancillary_path, pop_path,
                                python_scripts_folder_path, year)
from evalResultsDK import eval_Results_cph
from mainFunctions.basic import createFolder
from runDasymetricMapping import run_dasy
from runDisaggregation import run_disaggregation
from runPycnophylacticInterpolation import run_pycno
from verifyMassPreserving import verifyMassPreserv

#-------- GLOBAL ARGUMENTS --------
city="cph"
popraster = 'GHS_POP_100_near_cubicsplineWaterIESM_new.tif'
key = 'KOMNAME' #'BU_CODE'
ancillary_path_case = ancillary_path +"{}".format(city)
        
#-------- PROCESS: GHS RPREPARATION --------
run_Pycno = "no"
run_Dasy = "no"
run_Disaggregation = "yes"
verMassPreserv = "no"
run_EvaluationGC = "no"

#-------- SELECT DEMOGRAPHIC GROUP OR LIST OF GROUPS --------
# If it is a list it will be calculated in multi-output model 
# Total Population : 'totalpop',
# 5 Age Groups : 'children', 'students','mobadults', 'nmobadults', 'elderly', 
# Group Migrants1: 7 Migrant Groups : 'AuNZ', 'CentAsia', 'EastAsia','EastEur', 'LAC', 'Melanesia', 'Micronesia', 'NorthAfr', 
                #'NorthAm', 'NorthEur', 'OTH', 'Polynesia', 'SEastAsia', 'SouthAsia', 'SouthEur', 'STA', 'SubSahAfr', 
                #'WestAsia', 'WestEur','DNK', 
# Group Migrants 2:'EU', 'notEU' 
#-------- CAREFUL IN 2 GROUPS OF MIGRANTS TO INCLUDE NATIVES IN BOTH GROUPS -------- 

attr_value = [ 'children', 'students', 'mobadults', 'nmobadults', 'elderly',
                'AuNZ', 'CentAsia', 'EastAsia','EastEur', 'LAC', 'Melanesia', 'Micronesia', 'NorthAfr', 
                'NorthAm', 'NorthEur', 'OTH', 'Polynesia', 'SEastAsia', 'SouthAsia', 'SouthEur', 'STA', 'SubSahAfr', 
                'WestAsia', 'WestEur','DNK', 
                'EU', 'notEU'] 
print(attr_value[0:5], attr_value[5:-2], attr_value[-3:])
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
        methodopts = ['apcatbr'] # aplm (linear model), aprf (random forest), apxgbtree (XGBoost), apcnn (CNN), 'apcatbr' (Catboost Regressor), 'apmltr'
        createFolder(ROOT_DIR + "/Results/{0}/{1}".format(city, methodopts[0] ))
        ymethodopts = ['Dasy'] #'Pycno', Dasy# pycno, td, tdnoise10, td1pycno, average25p75td
        cnnmodelopts = ['unet'] # lenet, vgg, uenc, unet, 2runet (this and the following are only aplicable if method == CNN) 
        #'GHS_ESM_corine': '8AIL0', 'GHS_ESM_corine_transp':12AIL1, 'GHS_ESM_corine_transpA': 12AIL2
        inputDataset = ['AIL8'] # 'AIL0', 'AIL1', 'AIL2'
        iterMax = 2
        for i in inputDataset:
            run_disaggregation(ancillary_path_case, ROOT_DIR, methodopts, ymethodopts, cnnmodelopts, city, year, attr_value, key, i, iterMax, python_scripts_folder_path)
    
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
   
    if run_EvaluationGC == "yes":
        #EVALUATION OF THE PREDICTIONS WITH THE GROUND TRUTH DATA OF MNC DATASET
        if isinstance(attr_value, list):
            for i in attr_value:
                eval_Results_cph(ROOT_DIR, pop_path, ancillary_path, year, city, i)
        else: eval_Results_cph(ROOT_DIR, pop_path, ancillary_path, year, city, attr_value)

process_data(attr_value)

