import os
import time
import numpy as np
import rasterio
from sklearn import metrics

import caret
import gputils as gput
import metricsev as mev
import neigPairs
import nputils as npu
import osgeoutils as osgu
import kerasutils as ku
import pycno
import matplotlib.pyplot as plt

def runDissever(city, fshape, ancdatasets, attr_value, ROOT_DIR, yraster=None, rastergeo=None, perc2evaluate = 0.1, poly2agg = None,
                method='lm', cnnmod='unet', patchsize=7, epochspi=1, batchsize=1024, lrate=0.001, filters=[2,4,8,16,32],
                lweights=[1/2, 1/2], extdataset=None, p=[1], min_iter=3, max_iter=100, converge=2,
                hubervalue=0.5, stdivalue=0.01, dropout=0.5,
                casestudy='pcounts', tempfileid=None, verbose=False):

    print('| DISSEVER MULTIPLE VARIABLES')
    indicator = casestudy.split('_')[0]
    filenamemetrics2e = ROOT_DIR + '/TempCSV/{}/pcounts1_'.format(city) + casestudy + '_2e0.csv'

    if patchsize >= 16 and (cnnmod == 'lenet' or cnnmod == 'uenc' or cnnmod == 'vgg'):
        cstudyad = indicator + '_ghspghsbualcnl_' + str(patchsize) + '_wpadd_extended'
    elif patchsize >= 16:
        cstudyad = indicator + '_ghspghsbualcnl_' + str(patchsize) + '_nopadd_extended'
    else:
        cstudyad = None

    print("here is the 1st ancdatasets: ancillary:", ancdatasets.shape)
    nrowsds = ancdatasets[:,:,0].shape[1]
    ncolsds = ancdatasets[:,:,0].shape[0]
    idsdataset = osgu.ogr2raster(fshape, attr='ID', template=[rastergeo, nrowsds, ncolsds],city=city)[0] # ID's polígonos originais

    print('| Computing polygons areas')
    polygonareas = gput.computeAreas(fshape)
    
    if yraster:
        listOfArrays=[]
        idpolvaluesList =[]
        src= rasterio.open(yraster)
        arrays = src.read()
        print(arrays.shape)
        if arrays.shape[0]>=1:
            for i in range(1, arrays.shape[0]+1):
                array, rastergeo = osgu.readmultiBandRaster(yraster,i)
                array = np.nan_to_num(array, nan=0, posinf=0, neginf=0)
                #print(np.nanmax(array))
                listOfArrays.append(array)
                idpolvaluesInitial = npu.statsByID(array, idsdataset, 'sum')
                idpolvaluesList.append(idpolvaluesInitial)
            
            #This is a nd array. Each array corresponds to a raster of initial estimates.
            disseverdatasetA = np.dstack((listOfArrays))
            
            #disseverdatasetList = [disseverdatasetA]
            #This is a list of dictionaries. Each dictionary has the aggregated values that will be used to preserve the mass later
            idpolvalues = idpolvaluesList
            
        else:
            disseverdatasetA, rastergeo = osgu.readRaster(yraster)   
            idpolvalues = npu.statsByID(disseverdatasetA, idsdataset, 'sum')
    else:
        polygonvaluesdataset, rastergeo = osgu.ogr2raster(fshape, attr=attr_value, template=[rastergeo, nrowsds, ncolsds], city=city)
        idpolvalues = npu.polygonValuesByID(polygonvaluesdataset, idsdataset)
        disseverdataset, rastergeo = pycno.runPycno(idsdataset, polygonvaluesdataset, rastergeo, tempfileid)

    iterdissolv = False
    if method.startswith('ap'):
        if iterdissolv:
            adjpolygons = gput.computeNeighbors(fshape, polyg = poly2agg, verbose=True)
        else:
            adjpairs, newpairs = neigPairs.createNeigSF(fshape, polyg=poly2agg)

    dissmask = np.copy(idsdataset)
    dissmask[~np.isnan(dissmask)] = 1
    ancvarsmask = np.dstack([dissmask] * ancdatasets.shape[2])
    print("ancvarsmask:", ancvarsmask.shape )
    dissmaskList=[]
    for i in range(len(attr_value)):
        dissmaskList.append(dissmask)
    dissmask = np.dstack((dissmaskList))
    # the mask for the predictions that keep the neighborhoods ID in n=2 
    dissmaskA = dissmask.reshape(dissmask.shape[0],dissmask.shape[1], 1, dissmask.shape[2])
    
    olddisseverdataset = disseverdatasetA
     
    if method.endswith('cnn'):
        print("This is for the CNN") 
        
        # Create anc variables patches (includes replacing nans by 0, and 0 by nans)
        print('| Creating ancillary variables patches')
        # ancdatasets[np.isnan(ancdatasets)] = 0
        print(cnnmod)
        padd = True if cnnmod == 'lenet' or cnnmod == 'uenc' or cnnmod == 'vgg' else False
        print(cstudyad)
        ancpatches = ku.createpatches(ancdatasets,city, ROOT_DIR, patchsize, padding=padd, stride=1, cstudy=cstudyad)
        print("ancpatches = patch of the ancillary:", ancpatches.shape)
        ancdatasets = ancdatasets * ancvarsmask

        # Compile model and save initial weights
        cnnobj = ku.compilecnnmodel(cnnmod, [patchsize, patchsize, ancdatasets.shape[2]], lrate, dropout,
                                    filters=filters, lweights=lweights, hubervalue=hubervalue, stdivalue=stdivalue)
        print("not Saving the weigths") 
        cnnobj.save_weights(ROOT_DIR + '/Temp/{}/models_'.format(city) + casestudy + '.h5')
        
    print("Saving the weigths")
    lasterror = -np.inf
    lowesterror = np.inf
    start_timeT = time.time()
    for k in range(1, max_iter+1):
        print('| - Iteration', k)
        start_time = time.time()
        # if (k%10) == 0:
        #     epochspi = epochspi + 10
           
        if method.endswith('cnn'):
            print("This is for the CNN") #UNCOMMENT IT FOR CNN
            
            print('| -- Updating dissever patches')
            # disseverdataset[np.isnan(disseverdataset)] = 0
            disseverdatasetA = disseverdatasetA * dissmask
            disspatches = ku.createpatches(disseverdatasetA, city, ROOT_DIR, patchsize, padding=padd, stride=1)
            print(disspatches.shape, "this the demo input")
            print('| -- Fitting the model')
            fithistory = caret.fitcnn(ancpatches, disspatches, p, ROOT_DIR, city, cnnmod=cnnmod, cnnobj=cnnobj, casestudy=casestudy,
                                    epochs=epochspi, batchsize=batchsize, extdataset=extdataset)

            # New
            # mod = caret.fit(ancdatasets, disseverdataset, p, 'aplm', batchsize, lrate, epochspi)

            print('| -- Predicting new values')
            predictedmaps = caret.predictcnn(cnnobj, cnnmod, fithistory, casestudy,
                                            ancpatches, disseverdatasetA.shape, batchsize=batchsize)
            print("predicted maps:", len(predictedmaps))
            for i in range(len(predictedmaps)): predictedmaps[i] = np.expand_dims(predictedmaps[i], axis=2)
            #predictedmaps= predictedmaps[0]
            #predictedmaps = predictedmaps.reshape(predictedmaps.shape[0],predictedmaps.shape[1], 1, predictedmaps.shape[2])
            #This is a list of arrays of (440,445,1,n)  
            
            # # Including variance
            # predictedmaps = caret.predictcnn(cnnobj, cnnmod, fithistory, casestudy,
            #                                  ancpatches, disseverdataset.shape, batchsize=batchsize)
            # print(predictedmaps[1][:,:,0].shape)
            # osgu.writeRaster(predictedmaps[1][:,:,0], rastergeo,
            #                  'pcounts_' + casestudy + '_' + str(k).zfill(2) + 'it_variance.tif')
            # predictedmaps = predictedmaps[0]

            # New
            # ancdatasets[np.isnan(ancdatasets)] = 0
            # predictedmapslm = caret.predict(mod, ancdatasets)
            # for i in range(len(predictedmapslm)): predictedmapslm[i] = np.expand_dims(predictedmapslm[i], axis=2)
            # for i in range(len(predictedmaps)): predictedmaps[i] = 0.9 * predictedmaps[i] + 0.1 * predictedmapslm[i]
            
        else:
            print('| -- Fitting the model')
            # Replace NaN's by 0
            ancdatasets[np.isnan(ancdatasets)] = 0
            #disseverdataset = disseverdataset * dissmask
            #### <<<< ----- THIS CHANGED ----- >>>> ####
            disseverdatasetA = disseverdatasetA * dissmask
            
            mod = caret.fitM(ancdatasets, disseverdatasetA, p, method, batchsize, lrate, epochspi, ROOT_DIR, casestudy,city) 
            
            print('| -- Predicting new values')
            predictedmaps = caret.predictM(mod, ancdatasets, attr_value)
            print(len(predictedmaps))
            print(predictedmaps[0].shape)
            #This is a list of arrays of (440,445,1,n)  
            for i in range(len(predictedmaps)): predictedmaps[i] = np.expand_dims(predictedmaps[i], axis=2)
        
        print("--", len(predictedmaps)) 
        bestmaepredictedmaps = float("inf")
        newPredList=[]
        
        for i, predmap in enumerate(predictedmaps):
            
            val = attr_value[i]
            print("---", val, "---")
            # Replace NaN zones by Nan
            predmap = predmap * dissmaskA
            predmap[predmap < 0] = 0
            predmap2e = np.copy(predmap)
            ancdatasets = ancdatasets * ancvarsmask
            metricsmap = mev.report_sdev_map(predmap)
            
            print("k",np.nanmax(predmap), predmap.shape) 
            idpolvalues = idpolvaluesList[i]
            previdpolvalues = idpolvalues # Original polygon values
            
            #adjpairs--> list of lists with adjustent 
            idsdataset2e = osgu.ogr2raster(fshape, attr='ID', template=[rastergeo, nrowsds, ncolsds],city=city)[0]
            
            # Edit idpolvalues
            pairslist = [item for t in adjpairs for item in t]
            
            ids2keep = list(set(previdpolvalues.keys()))
            
            idpolvalues = dict((k, previdpolvalues[k]) for k in ids2keep)
            #previdpolvalues = {int(m):v for m,v in previdpolvalues.items()}
            
            idpolvalues2e = dict((k, previdpolvalues[k]) for k in pairslist if k in list(previdpolvalues.keys())) 
            polygonarea2e = dict((k, polygonareas[k]) for k in pairslist if k in list(polygonareas.keys()))
            
            predmap = predmap[:,:,:,0]
            predmap2e = predmap2e[:,:,:,0]
            
            if verbose: print('| -- Computing adjustement factor')
            stats = npu.statsByID(predmap, idsdataset, 'sum')

            if method.startswith('ap'):
                stats2e = npu.statsByID(predmap2e, idsdataset2e, 'sum')
                stats2e = dict((k, stats2e[k]) for k in pairslist if k in list(stats2e.keys())) 
            
            # Horrible hack, avoid division by 0
            for s in stats: stats[s] = stats[s] + 0.00001
            for s in stats2e: stats2e[s] = stats2e[s] + 0.00001
            
            polygonratios = {k: idpolvalues[k] / stats[k] for k in stats.keys() & idpolvalues}
            polygonratios2e = {k: idpolvalues2e[k] / stats2e[k] for k in stats2e.keys() & idpolvalues2e}
            idpolvalues = previdpolvalues
            
            # Mass-preserving adjustment
            for polid in polygonratios:
                predmap[idsdataset == polid] = (predmap[idsdataset == polid] * polygonratios[polid])
            for polid in polygonratios2e:
                predmap2e[idsdataset2e == polid] = (predmap2e[idsdataset2e == polid] * polygonratios2e[polid])
            

            # Compute metrics for the evaluation municipalities
            actual2e = list(idpolvalues2e.values())
            predicted2e = list(stats2e.values())
            areas2e = list(polygonarea2e.values())
            range2e = max(actual2e) - min(actual2e)

            mae2e, wae2e = mev.mean_absolute_error(actual2e, predicted2e, areas2e)
            rmse2e = np.sqrt(metrics.mean_squared_error(actual2e, predicted2e))
            metricsmae2e = mev.report_mae_y(actual2e, predicted2e)
            metricsrmse2e = mev.report_rmse_y(actual2e, predicted2e)
            metricsr22e = mev.report_r2_y(actual2e, predicted2e)

            elapsed_time = time.time() - start_time
            if os.path.exists(filenamemetrics2e):
                with open(filenamemetrics2e, 'a') as myfile:
                    myfile.write(val)
                    myfile.write(';' + str(k) + ';' + str(elapsed_time) + ';' + str(mae2e) + ';' + str(rmse2e))
                    for metric in metricsmap: myfile.write(';' + str(metric))
                    for metric in metricsmae2e: myfile.write(';' + str(metric))
                    for metric in metricsrmse2e: myfile.write(';' + str(metric))
                    for metric in metricsr22e: myfile.write(';' + str(metric) + '\n')
            else:
                with open(filenamemetrics2e, 'w+') as myfile:
                    myfile.write('Val;IT;Time;MAE;RMSE;STDMEAN;MAEMEAN;RMSEMEAN;R2MEAN;R2ITR;ERROR2IT\n')
                    myfile.write(val)
                    myfile.write(';' + str(k) + ';' + str(elapsed_time) + ';' + str(mae2e) + ';' + str(rmse2e))
                    for metric in metricsmap: myfile.write(';' + str(metric))
                    for metric in metricsmae2e: myfile.write(';' + str(metric))
                    for metric in metricsrmse2e: myfile.write(';' + str(metric))
                    for metric in metricsr22e: myfile.write(';' + str(metric) + '\n')
            print("MAE:", metricsmae2e[0])
            
            if metricsmae2e[0] < bestmaepredictedmaps:
                bestmaepredictedmaps = metricsmae2e[0]
            
            #predmap = predmap.reshape(predmap.shape[0], predmap.shape[1], predmap.shape[3]) 
            predmap1 = predmap[:,:,0]
           
            osgu.writeRaster(predmap1, rastergeo, ROOT_DIR + "/TempRaster/{}/".format(city) +'pcounts1_{}_'.format(val) + casestudy + '_' + str(k).zfill(2) + 'it.tif')

            newPredList.append(predmap1)
            print("---- This is where the loop ends ----")
        
        # Check if the algorithm converged
        disseverdatasetA = np.dstack((newPredList))
        error = np.nanmean(abs(disseverdatasetA - olddisseverdataset))
        
        with open(filenamemetrics2e, 'a') as myfile: myfile.write(';' + str(error) + '\n')
        errorrat = (error/lasterror) if lasterror>0 else np.inf
        lasterror = error
        print('Error:', error)
        print('Errorrat:', errorrat, min_iter)
        if k >= min_iter:
            if errorrat < converge:
                if error < lowesterror:
                    lowesterror = error
                    lowesterriter = k
                    print('NOT Retaining model fitted at iteration', lowesterriter)
                    lowesterrdisseverdataset = np.copy(disseverdatasetA)
            else:
                if k == min_iter:
                    lowesterriter = k
                else:
                    disseverdatasetA = lowesterrdisseverdataset #itan disseverdatasetA
                print('Retaining model fitted at iteration', lowesterriter)
                break
        olddisseverdataset = disseverdatasetA
    
    elapsed_timeT = time.time() - start_timeT
    with open(filenamemetrics2e, 'a') as myfile: myfile.write(';' + str(elapsed_timeT) + '\n')
    print('Total time:', elapsed_timeT)
    
    if tempfileid:
        tempfile = ROOT_DIR + "/TempRaster/tempfiledissever1_" + tempfileid + '.tif'
        osgu.writeRaster(disseverdataset, rastergeo, tempfile)
    
    #resultlist = np.dsplit(disseverdataset, len(attr_value))
    resultlist = newPredList
    return resultlist #disseverdatasetA[:,:,len(attr_value)-1], rastergeo
    
