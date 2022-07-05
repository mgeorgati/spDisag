from config import (ROOT_DIR, ancillary_path, python_scripts_folder_path,
                                year, gdal_rasterize_path, pop_path)

from mainFunctions import createFolder
from runDasymetricMapping import run_dasy
from runDisaggregation import run_disaggregation

from runPycnophylacticInterpolation import run_pycno

def process_data(attr_value, city, group_split, nmodelpred, popraster, key, run_Pycno, run_Dasy, run_Disaggregation, maxIters, methodopts, ymethodopts, inputDataset):
    ancillary_path_case = ancillary_path +"{}".format(city)
    
    createFolder(ROOT_DIR + "/Temp/{}/".format(city))
    createFolder(ROOT_DIR + "/TempRaster/{}/".format(city))
    createFolder(ROOT_DIR + "/TempCSV/{}/".format(city))
    
    if run_Pycno == "yes":
        createFolder(ROOT_DIR + "/Results/{}/Pycno/".format(city))
        ##### -------- PROCESS: RUN PYCNOPHYLACTIC -------- #####
        for i in attr_value:
            run_pycno(ROOT_DIR, ancillary_path, year, city, i, popraster, key)
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
        if len(methodopts) == 1:
            if not methodopts[0].endswith('cnn'):
                for i in inputDataset:
                    run_disaggregation(ancillary_path_case, ROOT_DIR, pop_path, methodopts, ymethodopts, city, year, attr_value, group_split, key, i, maxIters, gdal_rasterize_path)
            else:
                cnnmodelopts = ['unet']
                print(inputDataset)
                for i in inputDataset:
                    from runDisaggregationTF import run_disaggregationTF
                    run_disaggregationTF(ancillary_path_case, ROOT_DIR, pop_path, methodopts, ymethodopts, cnnmodelopts, city, year, attr_value, group_split, nmodelpred, key, i, maxIters, gdal_rasterize_path)
        else:
            print("---------- YOU NEED TO DEFINE METHODOPTS ----------")
    
                
    
    
