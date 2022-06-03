import glob, os
from datetime import datetime
from config import (ROOT_DIR, ancillary_path,  python_scripts_folder_path,
                                pop_path, year)
from evaluating import evalRsNL, corCSV
from mainFunctions import createFolder
#from runDasymetricMapping import run_dasy
#from runDisaggregation import run_disaggregation
#from runPycnophylacticInterpolation import run_pycno
from verifyMassPreserving import verifyMassPreserv

def process_data(attr_value, city, group_split, popraster, key, run_Pycno, run_Dasy, run_Disaggregation, maxIters, methodopts, ymethodopts, inputDataset, verMassPreserv, run_Evaluation):
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
        if isinstance(attr_value, list):
            for i in attr_value:
                outputNameDasy = "/Results/{}/Dasy/".format(city) + str(year) + '_' + city + '_' + i + '_dasy.tif'
                run_dasy(ancillary_path, year, city, i, outputNameDasy, ROOT_DIR, popraster, key) 
        else:
            outputNameDasy = "/Results/{}/Dasy/".format(city) + str(year) + '_' + city + '_' + attr_value + '_dasy.tif'
            run_dasy(ancillary_path, year, city, attr_value, outputNameDasy, ROOT_DIR, popraster, key) 
    
    if run_Disaggregation == "yes":
        print(methodopts, ymethodopts, inputDataset)
        for i in inputDataset:
            run_disaggregation(ancillary_path_case, ROOT_DIR, methodopts, ymethodopts, city, year, attr_value, group_split, key, i, maxIters, python_scripts_folder_path)
    
    
