import os
import subprocess
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import itertools

from dataSelection.ancilDt import selectAncDt
from mainFunctions.basic import createFolder
from utils import gdalutils, osgu

SEED = 42
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Turn off GPU

def run_disaggregation (ancillary_path, ROOT_DIR, pop_path, methodopts, ymethodopts, city, year, attr_value, group_split, key, inputDataset, iterMax, gdal_rasterize_path):
    """[summary]

    Args:
        methodopts ([type]): [description]
        ymethodopts ([type]): [description]
        cnnmodelopts ([type]): [description]
        city ([type]): [description]
        year ([type]): [description]
        inputDataset ([type]): [description]
        iterMax ([type]): [description]
    """    
    psamplesopts = [[1]] # [0.0625], [[0.0625], [0.03125]], [[0.03125, 0.0625]]]
    
    perc2evaluateopts = [1]
    
    fshapea = ROOT_DIR + "/Shapefiles/{1}/{0}_{1}.shp".format(year,city)
    fcsv = ROOT_DIR + "/Statistics/{1}/{0}_{1}.csv".format(year,city)
    print('this is the file')
    ancdatasets, rastergeo = selectAncDt(city, year, inputDataset, ancillary_path, pop_path)
        
    ancdatasetsopts = [ancdatasets]

    fshape = osgu.copyShape(fshapea, 'dissever', city)
    # if not yraster: osgu.addAttr2Shapefile(fshape, fcsv, [admboundary1.upper(), admboundary2.upper()])
    if not ymethodopts: osgu.addAttr2Shapefile(fshape, fcsv, '{0}_{1}'.format(year,city))

    for (ancdts, psamples, method, perc2evaluate, ymethod) in itertools.product(ancdatasetsopts,
                                                                                psamplesopts,
                                                                                methodopts,
                                                                                perc2evaluateopts,
                                                                                ymethodopts):
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
                cmd_tif_merge = """python {0}/gdal_merge.py -o "{1}" -separate {2} """.format(gdal_rasterize_path, outfile, str_files)
                print(cmd_tif_merge)
                subprocess.call(cmd_tif_merge, shell=False)
                yraster = outfile
                print(yraster)
              
            else:
                yraster = outfile
                print(yraster)
            if not method.endswith('cnn'):
                print('\n--- Running dissever leveraging a', method, 'model')
                 
                #dissever_year_city_ymethod_method_psamples_lAIL(length&type)_lIL_itl _attrValue
                casestudy = str(year) + '_' + city + '_' + ymethod + 'C_' + method + '_p' + str(psamples) + \
                            '_' + str(ancdts.shape[2]) + inputDataset + '_' + str(len(attr_value)) + 'IL_' + 'it' + str(iterMax)
                
                from processes.disseverM import runDissever as runDisseverM                
                dissdatasetList = runDisseverM(city, fshape, ancdts, attr_value,ROOT_DIR,group_split, yraster=yraster,rastergeo=rastergeo,
                                              poly2agg= key, method=method, p=psamples, min_iter=2, max_iter=iterMax, converge=2,
                                                casestudy=casestudy, verbose=True) #yraster, 29, 30

                for i in range(len(dissdatasetList)):
                    dissdataset = dissdatasetList[i]
                    val = attr_value[i]
                    #print(dissdataset.shape, np.nanmax(dissdataset))
                    print('- Writing raster to disk...')
                    outfile = ROOT_DIR + '/Results/{}/'.format(city) + method + '/dissever01_500_01' + casestudy
                    osgu.writeRaster(dissdataset[:,:], rastergeo, outfile + '_' + val + '.tif')
            else:
                print("Method not defined") 
            
            # Calculate total population as average of the sum of the age groups and migrant groups
            gdalutils.calcTotalPop(outfile, attr_value, city, gdal_rasterize_path)  
        
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
                casestudy = str(year) + '_' + city + '_' + ymethod + '_' + method + '_p' + str(psamples) + \
                            '_' + str(ancdts.shape[2]) + inputDataset + '_' + str(1) + 'IL_' + 'it' + str(iterMax)
                import processes.dissever as dissever 
                dissdataset, rastergeo = dissever.runDissever(city, fshape, ancdts, attr_value,ROOT_DIR, min_iter=3, max_iter=iterMax,
                                                            perc2evaluate=perc2evaluate, poly2agg = key,
                                                            rastergeo=rastergeo, method=method, p=psamples,
                                                            yraster=yraster, casestudy=casestudy,
                                                            verbose=True) #yraster, 29, 30
                print('- Writing raster to disk...')
                osgu.writeRaster(dissdataset, rastergeo, ROOT_DIR + '/Results/{}/'.format(city) + method + '/dissever00_' + casestudy + '_' + attr_value + '.tif')#
            else:
                print("Method not defined")

    
    osgu.removeShape(fshape, city)

    

    