import glob
from datetime import datetime
from config import (ROOT_DIR, ancillary_path, python_scripts_folder_path,
                                pop_path, year, gdal_rasterize_path)

from mainFunctions.basic import createFolder

def process_data(attr_value, city, group_split, nmodelpred, popraster, key, run_Pycno, run_Dasy, run_Disaggregation, maxIters, methodopts, ymethodopts, inputDataset, verMassPreserv, run_Evaluation):
    ancillary_path_case = ancillary_path +"{}".format(city)
    
    createFolder(ROOT_DIR + "/Temp/{}/".format(city))
    createFolder(ROOT_DIR + "/TempRaster/{}/".format(city))
    createFolder(ROOT_DIR + "/TempCSV/{}/".format(city))
    
    print(attr_value[0:group_split[0]],attr_value[group_split[0]:group_split[1]])
    if run_Pycno == "yes":
        createFolder(ROOT_DIR + "/Results/{}/Pycno/".format(city))
        ##### -------- PROCESS: RUN PYCNOPHYLACTIC -------- #####
        for i in attr_value:
            from processes.runPycnophylacticInterpolation import run_pycno
            run_pycno(ROOT_DIR, ancillary_path, year, city, i, popraster, key)
    if run_Dasy == "yes":
        print(attr_value)
        createFolder(ROOT_DIR + "/Results/{}/Dasy/".format(city))
        ##### -------- PROCESS: RUN DASYMETRIC  -------- #####
        if isinstance(attr_value, list):
            for i in attr_value:
                outputNameDasy = "/Results/{}/Dasy/".format(city) + str(year) + '_' + city + '_' + i + '_dasy.tif'
                from processes.runDasymetricMapping import run_dasy
                run_dasy(ancillary_path, year, city, i, outputNameDasy, ROOT_DIR, popraster, key) 
        else:
            outputNameDasy = "/Results/{}/Dasy/".format(city) + str(year) + '_' + city + '_' + attr_value + '_dasy.tif'
            run_dasy(ancillary_path, year, city, attr_value, outputNameDasy, ROOT_DIR, popraster, key) 
    
    if run_Disaggregation == "yes":
        if len(methodopts) == 1:
            if not methodopts[0].endswith('cnn'):
                for i in inputDataset:
                    from processes.runDisaggregation import run_disaggregation
                    run_disaggregation(ancillary_path_case, ROOT_DIR, pop_path, methodopts, ymethodopts, city, year, attr_value, group_split, key, i, maxIters, python_scripts_folder_path)
            else:
                cnnmodelopts = ['unet']
                print(inputDataset)
                for i in inputDataset:
                    from processes.runDisaggregationTF import run_disaggregationTF
                    run_disaggregationTF(ancillary_path_case, ROOT_DIR, pop_path, methodopts, ymethodopts, cnnmodelopts, city, year, attr_value, group_split, nmodelpred, key, i, maxIters, gdal_rasterize_path)
        else:
            print("---------- YOU NEED TO DEFINE METHODOPTS ----------")
    