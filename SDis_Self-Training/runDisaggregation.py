from email import utils
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import itertools

from dataSelection import selectAncDt
from mainFunctions import createFolder
from utils import osgu
SEED = 42
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Turn off GPU

def run_disaggregation(ancillary_path, ROOT_DIR, pop_path, methodopts, ymethodopts, city, year, attr_value, group_split, key, inputDataset, iterMax, gdal_rasterize_path):
    """_summary_ is an intermediate function connecting to dissever and disseverM  

    Args:
        ancillary_path (str): Path to ancillary data
        ROOT_DIR (str): Path to directory with main scripts
        pop_path (str): Path to directory with ground truth/historical data
        methodopts (str): _description_
        ymethodopts (str): _description_
        city (str): _description_
        year (int): _description_
        attr_value (list or str): _description_
        group_split (list): _description_
        key (str): _description_
        inputDataset (str): _description_
        iterMax (int): _description_
        gdal_rasterize_path (str): Directory to gdal_merge.py
    """      
    psamplesopts = [[1]]
    fshapea = ROOT_DIR + "/Shapefiles/{1}/{0}_{1}.shp".format(year,city)
    fcsv = ROOT_DIR + "/Statistics/{1}/{0}_{1}.csv".format(year,city)
    ancdatasets, rastergeo = selectAncDt(city, year, inputDataset, ancillary_path, pop_path)
    
    ancdatasetsopts = [ancdatasets]
    
    fshape = osgu.copyShape(fshapea, 'dissever', city)
    
    if not ymethodopts: osgu.addAttr2Shapefile(fshape, fcsv, '{0}_{1}'.format(year,city))

    for (ancdts, psamples, method, ymethod) in itertools.product(ancdatasetsopts, psamplesopts, methodopts, ymethodopts):
        createFolder(ROOT_DIR + "/Results/{0}/{1}/".format(city, method))
                                                                                
        if isinstance(attr_value, list):
            listOfFiles = []
            converted_list = [str(element) for element in attr_value]
            joined_string = '_'.join(converted_list)
            outfile =  ROOT_DIR + '/TempRaster/{0}/merged_'.format(city) + str(len(attr_value)) + '_{0}.tif'.format(ymethod.lower())
            if not os.path.exists(outfile):
                for i in attr_value:
                    filePaths = ROOT_DIR + '/Results/{1}/{4}/{0}_{1}_{2}_{3}.tif'.format(year,city,i,ymethod.lower(),ymethod)
                    listOfFiles.append(filePaths) 
                
                #Create txt file with number of band --> Name of File
                f = open(ROOT_DIR + '/TempRaster/{1}/{0}_{1}_{2}_{3}.txt'.format(year,city,len(attr_value),ymethod.lower()), "w+")
                str_files = " ".join([""" "{}" """.format(listOfFiles[i]) for i in range(len(listOfFiles))])
                for i,each in enumerate(listOfFiles,start=1):
                    f.write("{}.{}".format(i,each))
                f.close()
                print(outfile) 
                #Write the merged tif file 
                cmd_tif_merge = """python "{0}/gdal_merge.py" -o "{1}" -separate {2} """.format(gdal_rasterize_path, outfile, str_files)
                print(cmd_tif_merge)
                subprocess.call(cmd_tif_merge, shell=True)
                yraster = outfile
                print(yraster)
                print('| INPUT POPULATION DATASET SUCCESFULLY PRODUCED')
                
                
            else:
                print('| INPUT POPULATION DATASET ALREADY EXISTS')
                yraster = outfile
                
            if not method.endswith('cnn'):
                print('\n--- Running dissever leveraging a', method, 'model')
            
                #dissever_year_city_ymethod_method_psamples_lAIL(length&type)_lIL_itl _attrValue
                casestudy = str(year) + '_' + city + '_' + ymethod + '_' + method + \
                            '_' + str(ancdts.shape[2]) + inputDataset + '_' + str(len(attr_value)) + 'IL_' + 'it' + str(iterMax)
                from processes import disseverM
                print(yraster)
                dissdatasetList = disseverM.runDissever(city, fshape, ancdts, attr_value, group_split, ROOT_DIR, min_iter=3, max_iter=iterMax,
                                                            poly2agg = key,
                                                            rastergeo=rastergeo, method=method, p = psamples,
                                                            yraster=yraster, casestudy=casestudy,
                                                            verbose=True) #yraster, 29, 30

                for i in range(len(dissdatasetList)):
                    dissdataset = dissdatasetList[i]
                    val = attr_value[i]
                    print('- Writing raster to disk...')
                    osgu.writeRaster(dissdataset[:,:], rastergeo, ROOT_DIR + '/Results/{}/'.format(city) + method + '/dissever01_' + casestudy + '_' + val + '.tif')#
            else:
                print("You need to change environment and include tensorflow! This refers to the NN")
                sys.exit()           
                
        # ______________________
        # if input is not a list
        else: 
            if ymethod == 'Pycno':
                yraster = ROOT_DIR + '/Results/{2}/{0}/{1}_{2}_{3}_pycno.tif'.format(ymethod, year, city, attr_value)
            elif ymethod == 'Dasy':
                yraster = ROOT_DIR + '/Results/{2}/{0}/{1}_{2}_{3}_dasy.tif'.format(ymethod, year, city, attr_value)    
            else:
                yraster = None
            
            if not method.endswith('cnn'):
                print('\n--- Running dissever leveraging a', method, 'model with 1 Variable')
                #dissever_year_city_ymethod_method_psamples_lAIL(length&type)_lIL_itl _attrValue
                casestudy = str(year) + '_' + city + '_' + ymethod + '_' + method + \
                            '_' + str(ancdts.shape[2]) + inputDataset + '_' + str(1) + 'IL_' + 'it' + str(iterMax) 
                from processes import dissever
                dissdataset, rastergeo = dissever.runDissever(city, fshape, ancdts, attr_value,ROOT_DIR, min_iter=3, max_iter=iterMax,
                                                            poly2agg = key,
                                                            rastergeo=rastergeo, method=method, p=psamples,
                                                            yraster=yraster, casestudy=casestudy,
                                                            verbose=True) #yraster, 29, 30
                print('- Writing raster to disk...')
                osgu.writeRaster(dissdataset, rastergeo, ROOT_DIR + '/Results/{}/'.format(city) + method + '/dissever00_' + casestudy + '_' + attr_value + '.tif')#
            else:
                print("You need to change environment and include tensorflow! This refers to the NN")
                sys.exit()
                
                

    osgu.removeShape(fshape, city)

    