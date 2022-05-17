from utils import osgu
import pycno


def run_pycno(ROOT_DIR,ancillary_path, year, city, attr_value, popraster, key):
    print('--- Running pycnophylactic interpolation for the indicator', year, city, attr_value)
    ds, rastergeo = osgu.readRaster(ancillary_path + "{0}/GHS/{1}".format(city, popraster))
    nrowsds = ds.shape[1]
    ncolsds = ds.shape[0]

    fshapea = ROOT_DIR + "/Shapefiles/{1}/{0}_{1}.shp".format(year,city)
    fshape = osgu.copyShape(fshapea, 'pycno', city)
    fcsv = ROOT_DIR + "/Statistics/{1}/{0}_{1}.csv".format(year,city)

    osgu.addAttr2Shapefile(fshape, fcsv, [key], encoding='utf-8') #indicator[2].upper() #### <<<< ----- CHANGED ----- >>>> ####

    # tempfileid = None
    tempfileid = ROOT_DIR + "/TempRaster/{}/tempfilepycno_".format(city) + str(year) + city + attr_value
    idsdataset = osgu.ogr2raster(fshape, attr='ID', template=[rastergeo, nrowsds, ncolsds], city=city)[0]
    polygonvaluesdataset, rastergeo = osgu.ogr2raster(fshape, attr=attr_value, template=[rastergeo, nrowsds, ncolsds], city=city)
    pycnodataset, rastergeo = pycno.runPycno(ROOT_DIR, city, idsdataset, polygonvaluesdataset, rastergeo, attr_value, tempfileid=tempfileid)

    osgu.writeRaster(pycnodataset[:, :, 0], rastergeo, ROOT_DIR + "/Results/{1}/Pycno/{0}_{1}_{2}_pycnoWI.tif".format(year, city, attr_value))

    osgu.removeShape(fshape,city)
    
