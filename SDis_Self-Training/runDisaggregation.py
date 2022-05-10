import os, sys
import warnings
import subprocess
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import itertools 
from mainFunctions.basic import createFolder
from ancilDt import selectAncDt
import dissever
import disseverM 
import osgeoutils as osgu

SEED = 42
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Turn off GPU

def run_disaggregation (ancillary_path, ROOT_DIR, methodopts, ymethodopts, city, year, attr_value, key, inputDataset, iterMax, python_scripts_folder_path):
    """[summary]

    Args:
        methodopts ([type]): [description]
        ymethodopts ([type]): [description]
        city ([type]): [description]
        year ([type]): [description]
        inputDataset ([type]): [description]
        iterMax ([type]): [description]
    """   
    fshapea = ROOT_DIR + "/Shapefiles/{1}/{0}_{1}.shp".format(year,city)
    fcsv = ROOT_DIR + "/Statistics/{1}/{0}_{1}.csv".format(year,city)
    ancdatasets, rastergeo = selectAncDt(city, year, inputDataset, ancillary_path)
    
    ancdatasetsopts = [ancdatasets]

    fshape = osgu.copyShape(fshapea, 'dissever', city)
    
    if not ymethodopts: osgu.addAttr2Shapefile(fshape, fcsv, '{0}_{1}'.format(year,city))

    for (ancdts, method, perc2evaluate, ymethod) in itertools.product(ancdatasetsopts, methodopts, ymethodopts):
        createFolder(ROOT_DIR + "/Results/{0}/{1}/".format(city, method))
                                                                                
        if isinstance(attr_value, list):
            listOfFiles = []
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
                
                #Write the merged tif file 
                cmd_tif_merge = """python {0}/gdal_merge.py -o "{1}" -separate {2} """.format(python_scripts_folder_path, outfile, str_files)
                subprocess.call(cmd_tif_merge, shell=False)
                print('| INPUT POPULATION DATASET SUCCESFULLY PRODUCED')
                yraster = outfile
                
            else:
                print('| INPUT POPULATION DATASET ALREADY EXISTS')
                yraster = outfile
                
            if not method.endswith('cnn'):
                print('\n--- Running dissever leveraging a', method, 'model')
            
                #dissever_year_city_ymethod_method_psamples_lAIL(length&type)_lIL_itl _attrValue
                casestudy = str(year) + '_' + city + '_' + ymethod + '_' + method + \
                            '_' + str(ancdts.shape[2]) + inputDataset + '_' + str(len(attr_value)) + 'IL_' + 'it' + str(iterMax)
                dissdatasetList = disseverM.runDissever(city, fshape, ancdts, attr_value,ROOT_DIR, min_iter=3, max_iter=iterMax,
                                                            perc2evaluate=perc2evaluate, poly2agg= key,
                                                            rastergeo=rastergeo, method=method, 
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
                print('none')
            
            if not method.endswith('cnn'):
                print('\n--- Running dissever leveraging a', method, 'model with 1 Variable')
                print(yraster)
                #dissever_year_city_ymethod_method_psamples_lAIL(length&type)_lIL_itl _attrValue
                casestudy = str(year) + '_' + city + '_' + ymethod + '_' + method + \
                            '_' + str(ancdts.shape[2]) + inputDataset + '_' + str(1) + 'IL_' + 'it' + str(iterMax) 
                dissdataset, rastergeo = dissever.runDissever(city, fshape, ancdts, attr_value,ROOT_DIR, min_iter=3, max_iter=iterMax,
                                                            perc2evaluate=perc2evaluate, poly2agg = key,
                                                            rastergeo=rastergeo, method=method,
                                                            yraster=yraster, casestudy=casestudy,
                                                            verbose=True) #yraster, 29, 30
                print('- Writing raster to disk...')
                osgu.writeRaster(dissdataset, rastergeo, ROOT_DIR + '/Results/{}/'.format(city) + method + '/dissever00_' + casestudy + '_' + attr_value + '.tif')#
            else:
                print("You need to change environment and include tensorflow! This refers to the NN")
                sys.exit()
                
                

    osgu.removeShape(fshape, city)

    