import os

import numpy as np

from processes import dm
from utils import osgu


def run_dasy(ancillary_path, year, city, attr_value,outputNameDasy, ROOT_DIR, popraster, key):
    print('--- Running dasymetric mapping for the indicator', year, city, attr_value)
    ds, rastergeo = osgu.readRaster( ancillary_path + "{0}/GHS/{1}".format(city, popraster))
    #dstemplate = osgu.readRaster(os.path.join(ROOT_DIR, 'Rasters', '{}_template'.format(city), templateraster))[0]
    nrowsds = ds.shape[1]
    ncolsds = ds.shape[0]
    fshapeaMerged = ROOT_DIR + "/Shapefiles/{1}/Merged/{0}_{1}.shp".format(year,city)
    if not os.path.exists(fshapeaMerged):
        fshapea = ROOT_DIR + "/Shapefiles/{1}/{0}_{1}.shp".format(year,city)
        fshape = osgu.copyShape(fshapea, 'dasymapping', city)
        fcsv = ROOT_DIR + "/Statistics/{1}/{0}_{1}.csv".format(year,city)
        osgu.removeAttrFromShapefile(fshape, ['ID', 'VALUE'])
        osgu.addAttr2Shapefile(fshape, fcsv, [key],  encoding='UTF-8')
    else:
        fshape = osgu.copyShape(fshapeaMerged, 'dasymapping', city)

    fancdataset = ancillary_path + "{0}/GHS/{1}".format(city, popraster)
    tempfileid = None #None
    idsdataset = osgu.ogr2raster(fshape, attr='ID', template=[rastergeo, nrowsds, ncolsds], city=city)[0]
    polygonvaluesdataset, rastergeo = osgu.ogr2raster(fshape, attr=attr_value, template=[rastergeo, nrowsds, ncolsds], city=city)
    
    ancdataset = osgu.readRaster(fancdataset)[0]
    tddataset, rastergeo = dm.rundasymmapping(idsdataset, polygonvaluesdataset, ancdataset, rastergeo, tempfileid=tempfileid)
    
    #idpolvalues = npu.polygonValuesByID(polygonvaluesdataset, idsdataset)
    pycnomask = np.copy(idsdataset)
    pycnomask[~np.isnan(pycnomask)] = 1
    tddataset = tddataset * pycnomask
    inputRaster = ROOT_DIR + outputNameDasy 
    osgu.writeRaster(tddataset[:,:,0], rastergeo, inputRaster ) #

    osgu.removeShape(fshape, city)
    
   