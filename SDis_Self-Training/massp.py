import numpy as np
import osgeoutils as osgu

def runMassPreserving(idsdataset, polygonvaluesdataset, rastergeo, tempfileid=None):
    print('|| MASS-PRESERVING AREAL WEIGHTING')
    unique, counts = np.unique(idsdataset[~np.isnan(idsdataset)], return_counts=True)
    counts = dict(zip(unique, counts))
    countsmp = np.copy(idsdataset)

    for polid in counts:
        countsmp[countsmp == polid] = counts[polid]

    masspdataset = polygonvaluesdataset/countsmp

    if tempfileid:
        tempfile = tempfileid + '_tempfilemp.tif'
        osgu.writeRaster(masspdataset[:, :, 0], rastergeo, tempfile)

    return masspdataset, rastergeo

 

