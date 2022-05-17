import os

import numpy as np
from evaluation import mev
from models import caret, ku
from sklearn import metrics
from utils import gput, neigPairs, npu, osgu

from processes import pycno


def runDissever(city, fshape, ancdatasets, attr_value, ROOT_DIR, yraster=None, rastergeo=None, perc2evaluate = 0.1, poly2agg = None,
                method='lm', cnnmod='unet', patchsize=7, epochspi=1, batchsize=1024, lrate=0.001, filters=[2,4,8,16,32],
                lweights=[1/2, 1/2], extdataset=None, p=[1], min_iter=3, max_iter=100, converge=1, #htan 2
                hubervalue=0.5, stdivalue=0.01, dropout=0.5,
                casestudy='pcounts', tempfileid=None, verbose=False):

    print('| DISSEVER')
    
    indicator = casestudy.split('_')[0]
    filenamemetrics2e = ROOT_DIR + '/TempCSV/pcounts_' + casestudy + '_2e.csv'
    
    if patchsize >= 16 and (cnnmod == 'lenet' or cnnmod == 'uenc' or cnnmod == 'vgg'):
        cstudyad = indicator + '_ghspghsbualcnl_' + str(patchsize) + '_wpadd_extended'
    elif patchsize >= 16:
        cstudyad = indicator + '_ghspghsbualcnl_' + str(patchsize) + '_nopadd_extended'
    else:
        cstudyad = None
    
    nrowsds = ancdatasets[:,:,0].shape[1]
    ncolsds = ancdatasets[:,:,0].shape[0]
    idsdataset = osgu.ogr2raster(fshape, attr='ID', template=[rastergeo, nrowsds, ncolsds],city=city)[0] # ID's polígonos originais

    print('| Computing polygons areas')
    polygonareas = gput.computeAreas(fshape)

    if yraster:
        disseverdataset, rastergeo = osgu.readRaster(yraster)
        idpolvalues = npu.statsByID(disseverdataset, idsdataset, 'sum')
    else:
        polygonvaluesdataset, rastergeo = osgu.ogr2raster(fshape, attr='VALUE', template=[rastergeo, nrowsds, ncolsds], city=city)
        idpolvalues = npu.polygonValuesByID(polygonvaluesdataset, idsdataset)
        disseverdataset, rastergeo = pycno.runPycno(idsdataset, polygonvaluesdataset, rastergeo, tempfileid)

    iterdissolv = False
    if method.startswith('ap'):
        if iterdissolv:
            adjpolygons = gput.computeNeighbors(fshape, polyg = poly2agg, verbose=True)
        else:
            #This is used!
            adjpairs, newpairs = neigPairs.createNeigSF(fshape, polyg=poly2agg)
            # a list of pairs(in list) neighborhoodsd

    dissmask = np.copy(idsdataset)
    dissmask[~np.isnan(dissmask)] = 1
    ancvarsmask = np.dstack([dissmask] * ancdatasets.shape[2])

    olddisseverdataset = disseverdataset
    

    if method.endswith('cnn'):
        # Create anc variables patches (includes replacing nans by 0, and 0 by nans)
        print('| Creating ancillary variables patches')
        # ancdatasets[np.isnan(ancdatasets)] = 0
        padd = True if cnnmod == 'lenet' or cnnmod == 'uenc' or cnnmod == 'vgg' else False
        ancpatches = ku.createpatches(ancdatasets, patchsize, padding=padd, stride=1, cstudy=cstudyad)
        ancdatasets = ancdatasets * ancvarsmask

        # Compile model and save initial weights
        cnnobj = ku.compilecnnmodel(cnnmod, [patchsize, patchsize, ancdatasets.shape[2]], lrate, dropout,
                                    filters=filters, lweights=lweights, hubervalue=hubervalue, stdivalue=stdivalue)
        cnnobj.save_weights('Temp/models_' + casestudy + '.h5')
    
    #strat = True
    
    if method.startswith('ap'):
        if iterdissolv:
            if strat:
                numpols = round((perc2evaluate/2) * (len(adjpolygons)))
                pquartilles = gput.polygonsQuartilles(idpolvalues, numpols)
                adjpairs = gput.createAdjPairs(adjpolygons, perc2evaluate/2, strat=pquartilles, verbose=True)
                print("5",len(adjpairs))
            else:
                adjpairs = gput.createAdjPairs(adjpolygons, perc2evaluate/2, verbose=True)
                print("2",len(adjpairs))
            initadjpairs = adjpairs
            if(verbose): print('Fixed adjacent pairs (' + str(len(initadjpairs)) + ') -', initadjpairs)
    
    lasterror = -np.inf
    lowesterror = np.inf
    for k in range(1, max_iter+1):
        print('| - Iteration', k)

        # if (k%10) == 0:
        #     epochspi = epochspi + 10

        previdpolvalues = idpolvalues # Original polygon values
        if method.startswith('ap'):
            
            if iterdissolv:
                if strat:
                    pquartilles = gput.polygonsQuartilles(idpolvalues, numpols)
                    adjpairs = gput.createAdjPairs(adjpolygons, perc2evaluate / 2, strat=pquartilles, initadjpairs=initadjpairs, verbose=True)
                    print("4",len(adjpairs))
                else:
                    adjpairs = gput.createAdjPairs(adjpolygons, perc2evaluate/2, initadjpairs=initadjpairs)
                    print("3",len(adjpairs))
                if (verbose): print('Adjacent pairs (' + str(len(adjpairs)) + ') -', adjpairs)
                newshape, newpairs = gput.dissolvePairs(fshape, adjpairs)

                # osgu.removeShapefile(newshape)
            
            idsdataset2e = osgu.ogr2raster(fshape, attr='ID', template=[rastergeo, nrowsds, ncolsds], city=city)[0]

            # Edit idpolvalues
            pairslist = [item for t in adjpairs for item in t]

            ids2keep = list(set(previdpolvalues.keys()))
            idpolvalues = dict((k, previdpolvalues[k]) for k in ids2keep)

            idpolvalues2e = dict((k, previdpolvalues[k]) for k in pairslist)
            polygonarea2e = dict((k, polygonareas[k]) for k in pairslist)
            
        
        if method.endswith('cnn'):
            print('| -- Updating dissever patches')
            
            # disseverdataset[np.isnan(disseverdataset)] = 0
            disseverdataset = disseverdataset * dissmask
            disspatches = ku.createpatches(disseverdataset, patchsize, padding=padd, stride=1)

            print('| -- Fitting the model')
            fithistory = caret.fitcnn(ancpatches, disspatches, p, cnnmod=cnnmod, cnnobj=cnnobj, casestudy=casestudy,
                                      epochs=epochspi, batchsize=batchsize, extdataset=extdataset)

            print('| -- Predicting new values')
            predictedmaps = caret.predictcnn(cnnobj, cnnmod, fithistory, casestudy,
                                             ancpatches, disseverdataset.shape, batchsize=batchsize)

            
        else:
            print('| -- Fitting the model')
            # Replace NaN's by 0
            ancdatasets[np.isnan(ancdatasets)] = 0
            disseverdataset = disseverdataset * dissmask

            mod = caret.fit(ancdatasets, disseverdataset, p, method, batchsize, lrate, epochspi, ROOT_DIR, casestudy,city)

            print('| -- Predicting new values')
            predictedmaps = caret.predict(mod, ancdatasets)
            print("se",predictedmaps[0].shape)
            for i in range(len(predictedmaps)): predictedmaps[i] = np.expand_dims(predictedmaps[i], axis=2)
            print("ne", predictedmaps[0].shape)
        print("--", len(predictedmaps))
        bestmaepredictedmaps = float("inf")
        for i, predmap in enumerate(predictedmaps):
            print(np.nanmax(predmap), predmap.shape)
            # Replace NaN zones by Nan
            predmap = predmap * dissmask
            predmap[predmap < 0] = 0
            predmap2e = np.copy(predmap)
            ancdatasets = ancdatasets * ancvarsmask
            metricsmap = mev.report_sdev_map(predmap)

            if verbose: print('| -- Computing adjustement factor')
            stats = npu.statsByID(predmap, idsdataset, 'sum')

            if method.startswith('ap'):
                stats2e = npu.statsByID(predmap2e, idsdataset2e, 'sum')
                stats2e = dict((k, stats2e[k]) for k in pairslist)
            #print("7", stats)
            #print("8", stats2e)
            # Horrible hack, avoid division by 0
            for s in stats: stats[s] = stats[s] + 0.00001
            for s in stats2e: stats2e[s] = stats2e[s] + 0.00001

            polygonratios = {k: idpolvalues[k] / stats[k] for k in stats.keys() & idpolvalues}
            polygonratios2e = {k: idpolvalues2e[k] / stats2e[k] for k in stats2e.keys() & idpolvalues2e}
            idpolvalues = previdpolvalues
            #print("9", polygonratios)
            print("10", polygonratios)
            print("11", idsdataset)
            # Mass-preserving adjustment
            print("12", predmap.shape)
            print(predmap[idsdataset == 474.0])
            for polid in polygonratios:
                predmap[idsdataset == polid] = (predmap[idsdataset == polid] * polygonratios[polid])
                if polid == 474.0:
                    print("www",polid, predmap[idsdataset == polid])
            for polid in polygonratios2e:
                predmap2e[idsdataset2e == polid] = (predmap2e[idsdataset2e == polid] * polygonratios2e[polid])
                
            
            if method.startswith('ap'):
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

                if os.path.exists(filenamemetrics2e):
                    with open(filenamemetrics2e, 'a') as myfile:
                        myfile.write(str(k) + ';' + str(mae2e) + ';' + str(rmse2e))
                        for metric in metricsmap: myfile.write(';' + str(metric))
                        for metric in metricsmae2e: myfile.write(';' + str(metric))
                        for metric in metricsrmse2e: myfile.write(';' + str(metric))
                        for metric in metricsr22e: myfile.write(';' + str(metric))
                else:
                    with open(filenamemetrics2e, 'w+') as myfile:
                        myfile.write('IT;MAE;RMSE;STDMEAN;MAEMEAN;RMSEMEAN;R2MEAN;R2ITR;ERROR2IT\n')
                        myfile.write(str(k) + ';' + str(mae2e) + ';' + str(rmse2e))
                        for metric in metricsmap: myfile.write(';' + str(metric))
                        for metric in metricsmae2e: myfile.write(';' + str(metric))
                        for metric in metricsrmse2e: myfile.write(';' + str(metric))
                        for metric in metricsr22e: myfile.write(';' + str(metric))

                if metricsmae2e[0] < bestmaepredictedmaps:
                    bestmaepredictedmaps = metricsmae2e[0]
            disseverdataset = predmap

        osgu.writeRaster(disseverdataset[:, :, 0], rastergeo, ROOT_DIR + "/TempRaster/" + 'pcounts_' + casestudy + '_' + str(k).zfill(2) + 'it.tif')

        # Check if the algorithm converged
        error = np.nanmean(abs(disseverdataset-olddisseverdataset))
        with open(filenamemetrics2e, 'a') as myfile: myfile.write(';' + str(error) + '\n')
        errorrat = (error/lasterror) if lasterror>0 else np.inf
        lasterror = error
        print('Error:', error)
        print('Errorrat:', errorrat, min_iter)
        
        if k >= min_iter:
            print(errorrat, converge)
            if errorrat < converge:
                if error < lowesterror:
                    lowesterror = error
                    lowesterriter = k
                    print('--- Retaining model fitted at iteration', lowesterriter)
                    lowesterrdisseverdataset = np.copy(disseverdataset)
            else:
                if k == min_iter:
                    lowesterriter = k
                else:
                    disseverdataset = lowesterrdisseverdataset
                print('Retaining model fitted at iteration', lowesterriter)
                break
        olddisseverdataset = disseverdataset

    if tempfileid:
        tempfile = ROOT_DIR + '/TempRaster/tempfiledissever_' + tempfileid + '.tif'
        osgu.writeRaster(disseverdataset, rastergeo, tempfile)

    return disseverdataset[:,:,0], rastergeo

