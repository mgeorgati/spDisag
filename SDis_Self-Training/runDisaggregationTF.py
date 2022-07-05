import os
import subprocess
import warnings
from utils import gdalutils

warnings.filterwarnings("ignore", category=FutureWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import itertools
import random

import numpy as np
#import tensorflow as tf

from processes import dissever
from dataSelection import selectAncDt
from mainFunctions import createFolder
#import disseverM 
from processes import disseverTF
from utils import osgu

SEED = 42
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Turn off GPU

def run_disaggregationTF(ancillary_path, ROOT_DIR, methodopts, ymethodopts, cnnmodelopts, city, year, attr_value, group_split, nmodelpred, key, inputDataset, 
        iterMax, gdal_rasterize_path):
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
    epochspiopts = [10]
    batchsizeopts = [64] # 256, 1024ß
    learningrateopts = [0.001] # 0.0, 0.001, census-0.0001 #changed from 0.001
    
    useFlippedImages= 'no'
    loss_function = 'rblf' # 'clf', 'rmse', 'rblf
    
    extendeddatasetopts = [None] # None, '2T6'
    lossweightsopts = [[0.1, 0.9]]
    perc2evaluateopts = [1]
    hubervalueopts = [1] # 0.5, 1
    stdivalueopts = [1] # 0.1, 0.5, 1
    dropoutopts = [0.5] # 0.25, 0.5, 0.75
    
    fshapea = ROOT_DIR + "/Shapefiles/{1}/{0}_{1}.shp".format(year,city)
    fcsv = ROOT_DIR + "/Statistics/{1}/{0}_{1}.csv".format(year,city)
    ancdatasets, rastergeo = selectAncDt(city, year, inputDataset, ancillary_path)        
        
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

        ################################
        
        #outfile = ROOT_DIR + '/Results/{}/apcnn/dissever01_RMSE_2018_cph_Dasy_16unet_10epochspi_AIL3_it10'.format(city) 
        #gdalutils.calcTotalPop(outfile, attr_value, city, gdal_rasterize_path)

        ##################################
                                                                                
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
              
            else:
                yraster = outfile
            
            
            for cnnmodel in cnnmodelopts:
                if cnnmodel == 'lenet':
                    print('Lenet')
                    filtersopts = [[14, 28, 56, 112, 224]]
                    patchsizeopts = [7]
                elif cnnmodel == 'vgg':
                    print('VGG')
                    filtersopts = [[8, 16, 32, 64, 512]]
                    patchsizeopts = [32]
                elif cnnmodel.endswith('unet'):
                    print('U-Net')
                    filtersopts = [[8, 16, 32, 64, 128]]
                    patchsizeopts = [16]
                elif cnnmodel == 'uenc':
                    print('U-Net Encoder')
                    filtersopts = [[8, 16, 32, 64, 128]]
                    patchsizeopts = [16]
                else:
                    filtersopts = [[14, 28, 56, 112, 224]]
                    patchsizeopts = [7]

                for (lossweights, batchsize, epochpi, learningrate,
                    filters, patchsize, extendeddataset,
                    hubervalue, stdivalue, dropout) in itertools.product(lossweightsopts,
                                                                        batchsizeopts,
                                                                        epochspiopts,
                                                                        learningrateopts,
                                                                        filtersopts,
                                                                        patchsizeopts,
                                                                        extendeddatasetopts,
                                                                        hubervalueopts,
                                                                        stdivalueopts,
                                                                        dropoutopts):
                    print('\n--- Running dissever with the following CNN configuration:')
                    print('- Method:', cnnmodel, '| Percentage of sampling:', psamples,
                        '| Epochs per iteration:', epochpi, '| Batch size:', batchsize,
                        '| Learning rate:', learningrate, '| Loss Function:', loss_function.upper(), '| Flipped Images:', useFlippedImages, 
                        '| Filters:', filters,
                        '| Loss weights:', lossweights, '| Patch size:', patchsize,
                        '| Extended dataset:', extendeddataset, '| Huber value:', hubervalue,
                        '| Stdi value:', stdivalue, '| Dropout:', dropout)

                    random.seed(SEED)
                    np.random.seed(SEED)
                    
                    
                    tf.keras.backend.clear_session()
                    tf.random.set_seed(SEED)
                    physical_devices = tf.config.experimental.list_physical_devices('GPU')
                    for gpu_instance in physical_devices:
                        tf.config.experimental.set_memory_growth(gpu_instance, True)

                    yraster = outfile
                    if useFlippedImages =='yes':
                        flipped = 'flipped'
                    else:
                        flipped =''
                    casestudy = loss_function.upper() + flipped + '_' + str(year) + '_' + city + '_' + ymethod + '_' + str(patchsize) + cnnmodel + \
                                '_' + str(epochpi) + 'epochspi' + '_' + str(ancdts.shape[2]) + str(inputDataset) + '_it' + str(iterMax)
                    
                    dissdatasetList = disseverTF.runDissever(city, fshape, ancdts, attr_value, ROOT_DIR, group_split, nmodelpred, 
                                                                min_iter=3, max_iter=iterMax,
                                                                perc2evaluate=perc2evaluate,
                                                                poly2agg=key,
                                                                rastergeo=rastergeo, method=method, p=psamples,
                                                                cnnmod=cnnmodel, patchsize=patchsize, batchsize=batchsize,
                                                                epochspi=epochpi, lrate=learningrate, loss_function=loss_function, useFlippedImages=useFlippedImages, filters=filters,
                                                                lweights=lossweights, extdataset=extendeddataset,
                                                                yraster=yraster, converge=1.5,
                                                                hubervalue=hubervalue, stdivalue=stdivalue,
                                                                dropout=dropout,
                                                                casestudy=casestudy,
                                                                verbose=True)

                    for i in range(len(dissdatasetList)):
                        dissdataset = dissdatasetList[i]
                        val = attr_value[i]
                        #print(dissdataset.shape, np.nanmax(dissdataset))
                        print('- Writing raster to disk...')
                        
                        outfile = ROOT_DIR + '/Results/{}/'.format(city) + method + '/dissever01_' + casestudy
                        osgu.writeRaster(dissdataset[:,:], rastergeo, outfile + '_' + val + '.tif')
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
            
            """if not method.endswith('cnn'):
                print('\n--- Running dissever leveraging a', method, 'model with 1 Variable')
                print(yraster)
                #dissever_year_city_ymethod_method_psamples_lAIL(length&type)_lIL_itl _attrValue
                casestudy = str(year) + '_' + city + '_' + ymethod + '_' + method + '_p' + str(psamples) + \
                            '_' + str(ancdts.shape[2]) + inputDataset + '_' + str(1) + 'IL_' + 'it' + str(iterMax) 
                dissdataset, rastergeo = dissever.runDissever(city, fshape, ancdts, attr_value,ROOT_DIR, min_iter=3, max_iter=iterMax,
                                                            perc2evaluate=perc2evaluate, poly2agg = key,
                                                            rastergeo=rastergeo, method=method, p=psamples,
                                                            yraster=yraster, casestudy=casestudy,
                                                            verbose=True) #yraster, 29, 30
                print('- Writing raster to disk...')
                osgu.writeRaster(dissdataset, rastergeo, ROOT_DIR + '/Results/{}/'.format(city) + method + '/dissever00_' + casestudy + '_' + attr_value + '.tif')#
            else:"""
            print("This is for the CNN") #UNCOMMENT IT FOR CNN
            
            for cnnmodel in cnnmodelopts:
                if cnnmodel == 'lenet':
                    print('Lenet')
                    filtersopts = [[14, 28, 56, 112, 224]]
                    patchsizeopts = [7]
                elif cnnmodel == 'vgg':
                    print('VGG')
                    filtersopts = [[8, 16, 32, 64, 512]]
                    patchsizeopts = [32]
                elif cnnmodel.endswith('unet'):
                    print('U-Net')
                    filtersopts = [[8, 16, 32, 64, 128]]
                    patchsizeopts = [16]
                elif cnnmodel == 'uenc':
                    print('U-Net Encoder')
                    filtersopts = [[8, 16, 32, 64, 128]]
                    patchsizeopts = [16]
                else:
                    filtersopts = [[14, 28, 56, 112, 224]]
                    patchsizeopts = [7]

                for (lossweights, batchsize, epochpi, learningrate,
                    filters, patchsize, extendeddataset,
                    hubervalue, stdivalue, dropout) in itertools.product(lossweightsopts,
                                                                        batchsizeopts,
                                                                        epochspiopts,
                                                                        learningrateopts,
                                                                        filtersopts,
                                                                        patchsizeopts,
                                                                        extendeddatasetopts,
                                                                        hubervalueopts,
                                                                        stdivalueopts,
                                                                        dropoutopts):
                    print('\n--- Running dissever with the following CNN configuration:')
                    print('- Method:', cnnmodel, '| Percentage of sampling:', psamples,
                        '| Epochs per iteration:', epochpi, '| Batch size:', batchsize,
                        '| Learning rate:', learningrate, '| Filters:', filters,
                        '| Loss weights:', lossweights, '| Patch size:', patchsize,
                        '| Extended dataset:', extendeddataset, '| Huber value:', hubervalue,
                        '| Stdi value:', stdivalue, '| Dropout:', dropout)

                    random.seed(SEED)
                    np.random.seed(SEED)
                    
                    tf.keras.backend.clear_session()
                    tf.random.set_seed(SEED)
                    physical_devices = tf.config.experimental.list_physical_devices('GPU')
                    for gpu_instance in physical_devices:
                        tf.config.experimental.set_memory_growth(gpu_instance, True)

                    yraster = ROOT_DIR + '/Results/{2}/{0}/{1}_{2}_{3}_dasy.tif'.format(ymethod, year, city, attr_value)
                    casestudy = str(year) + '_' + city + '_' + ymethod + '_' + str(patchsize) + cnnmodel + '_huber' + str(hubervalue) + \
                                '_stdi' + str(stdivalue) + '_dropout' + str(dropout) + \
                                '_' + str(epochpi) + 'epochspi' + '_' + str(ancdts.shape[2]) + '-ploop-t1'
                    dissdataset, rastergeo = dissever.runDissever(city, fshape, ancdts, attr_value, ROOT_DIR, group_split, nmodelpred,
                                                                min_iter=1, max_iter=2,
                                                                perc2evaluate=perc2evaluate,
                                                                poly2agg=key,
                                                                rastergeo=rastergeo, method=method, p=psamples,
                                                                cnnmod=cnnmodel, patchsize=patchsize, batchsize=batchsize,
                                                                epochspi=epochpi, lrate=learningrate, filters=filters,
                                                                lweights=lossweights, extdataset=extendeddataset,
                                                                yraster=yraster, converge=1.5,
                                                                hubervalue=hubervalue, stdivalue=stdivalue,
                                                                dropout=dropout,
                                                                casestudy=casestudy,
                                                                verbose=True)

                    print('- Writing raster to disk...')
                    outfile = ROOT_DIR + '/Results/{}/'.format(city)  + method + 'dissever00_' + casestudy

                    osgu.writeRaster(dissdataset, rastergeo,  outfile + '.tif')
                
    
    osgu.removeShape(fshape, city)
    # Calculate total population as average of the sum of the age groups and migrant groups
    gdalutils.calcTotalPop(outfile, attr_value, city, gdal_rasterize_path)
    