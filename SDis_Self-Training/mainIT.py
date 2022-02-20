from config.definitions import ROOT_DIR, ancillary_path, pop_path, year, python_scripts_folder_path
from evalResults import eval_Results_GC_ams
from runDasymetricMapping import run_dasy
from verifyMassPreserving import verifyMassPreserv
from runDisaggregation import run_disaggregation
from runPycnophylacticInterpolation import run_pycno
import glob
import time
city='rom'
#-------- GLOBAL ARGUMENTS --------

key = 'CT_CODE'
        
#-------- PROCESS: GHS RPREPARATION --------
run_Pycno = "no"
run_Dasy = "no"
run_Disaggregation = "no"
verMassPreserv = "yes"
run_EvaluationGC_ams = "no"
#['CT_CODE', 'Census tracts (sezioni di censimento)', 'totalpop', 'foreigners', 'ita', 'Africa', 'America', 'Asia & Oceania', 'EU', 'EurNotEU', 
# 'ita0-5', 'ita6-19', 'ita20-29', 'ita30-44', 'ita45-64', 'ita65-79', 'ita80+', 
# 'mig0-5', 'mig6-19', 'mig20-29', 'mig30-44', 'mig45-64', 'mig65-79', 'mig80+', 'COD_REG', 'COD_ISTAT', 'PRO_COM', 'SEZ', 'Shape_Leng', 'Shape_Area', 'geometry', 
# 'notEU', 'children', 'students', 'mobadults', 'nmobadults', 'elderly']
attr_value = [ 'totalpop', 'foreigners', 'ita', 'EU', 'EurNotEU', 'notEU', 'children', 'students', 'mobadults', 'nmobadults', 'elderly'] #'totalpop', 'foreigners',

def process_data(attr_value):
    year_list = [ 2015, 2016, 2017, 2018, 2019, 2020]
    for year in year_list:
        if year <= 2000:
            popraster = '1990_GHS_POP_100_near_cubicspline.tif' 
        elif 2000 < year <= 2015:
            popraster = '2000_GHS_POP_100_near_cubicspline.tif'
        elif year>2015 :
            popraster = '2015_GHS_POP_100_near_cubicspline.tif'
        else:
            print('----- NO GHS FILE -----')
        print(year, popraster)
        if run_Pycno == "yes":
            ##### -------- PROCESS: RUN PYCNOPHYLACTIC -------- #####
            for i in attr_value:
                run_pycno(year, city, i, popraster, key)
        if run_Dasy == "yes":
            ##### -------- PROCESS: RUN DASYMETRIC  -------- #####
            #templateraster = '{}_template_100.tif'.format(city)
            for i in attr_value:
                start_total_algorithm_timer = time.time()
                run_dasy(year, city, i, popraster, key)
                # stop total algorithm time timer ----------------------------------------------------------------------------------
                stop_total_algorithm_timer = time.time()
                # calculate total runtime
                total_time_elapsed = (stop_total_algorithm_timer - start_total_algorithm_timer)/60
                print("Total preparation time for ... is {} minutes".format( total_time_elapsed))
        
        if run_Disaggregation == "yes":
            methodopts = ['aprf'] # aplm (linear model), aprf (random forest), apxgbtree (XGBoost), apcnn (CNN)
            ymethodopts = ['Pycno', 'Dasy'] #'Pycno',# pycno, td, tdnoise10, td1pycno, average25p75td
            cnnmodelopts = ['unet'] # lenet, vgg, uenc, unet, 2runet (this and the following are only aplicable if method == CNN) 
            #'GHS_ESM_corine': '8AIL0', 'GHS_ESM_corine_transp':12AIL1, 'GHS_ESM_corine_transpA': 12AIL2
            inputDataset = ['AIL0'] # 'AIL0', 'AIL1', 'AIL2'
            iterMax = 5
            for i in inputDataset:
                run_disaggregation (methodopts, ymethodopts, cnnmodelopts, city, year, attr_value, key, i, iterMax, python_scripts_folder_path)
        
        if verMassPreserv == "yes":
            ##### -------- PROCESS: VERIFY MASS PRESERVATION  -------- #####
            fshapea = ROOT_DIR + "/Shapefiles/{1}/{0}_{1}.shp".format(year,city)
            fcsv = ROOT_DIR + "/Statistics/{1}/{0}_{1}.csv".format(year,city) 
            ymethodopts = ['Dasy'] #'Dasy', 'Pycno', 'aprf'
            for ymethod in ymethodopts:
                if isinstance(attr_value, list):
                    for i in attr_value:
                        evalList = glob.glob(ROOT_DIR + "/Results/{3}/{0}/*_{1}_{2}.tif".format(ymethod,i,ymethod.lower(),city))
                        #evalList = glob.glob(ROOT_DIR + "/Results/{0}/*_{1}_pycno.tif".format(ymethod,i))
                        csv_output = ROOT_DIR + '/Results/{1}/{3}/{0}_{1}_Eval_{2}.csv'.format(year,city,i,ymethod)
                        verifyMassPreserv(fshapea, fcsv, key, evalList, csv_output, i)
                else:
                    print("kati edi")
                    evalList = glob.glob(ROOT_DIR + "/Results/{0}/dissever00*{1}*.tif".format(ymethod,attr_value,ymethod.lower()))
                    print(evalList)
                    #evalList = glob.glob(ROOT_DIR + "/Results/{0}/*_{1}_pycno.tif".format(ymethod,i))
                    csv_output = ROOT_DIR + '/Results/{3}/{0}_{1}_Eval_{2}.csv'.format(year,city,attr_value,ymethod)
                    verifyMassPreserv(fshapea, fcsv, key, evalList, csv_output,attr_value)
                    
        if run_EvaluationGC_ams == "yes":
            #EVALUATION OF THE PREDICTIONS WITH THE GROUND TRUTH DATA OF MNC DATASET
            for i in attr_value:
                eval_Results_GC_ams(ROOT_DIR, pop_path, ancillary_path, year, city, i)
    
process_data(attr_value)
