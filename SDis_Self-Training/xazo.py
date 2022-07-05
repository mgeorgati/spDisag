import os
import geopandas as gpd
from pathlib import Path
from config import year, ROOT_DIR
from utils import osgeoutils 

def copyShape(fshapea, meth, city):
    fshapea = Path(fshapea)
    fname = fshapea.stem
    print(fname, fshapea)
    #fname = fshapea.split('\\',-1)[-1].split('.shp')[0]    
    fshape = ROOT_DIR + '/Temp/{}/'.format(city) + fname + '_' + meth + '_' + str(os.getpid()) + '.shp'
    print(fshape)
    df = gpd.read_file(fshapea)
    print(df.head)
    df.to_file(fshape, driver='ESRI Shapefile')
    return fshape
#from mainFunctions.basic import excelTOlatex 
#path = 'C:/Users/NM12LQ/OneDrive - Aalborg Universitet/Dasymetric_Mapping/SDis_Self-Training/Evaluation/ams_metrices'
#excelTOlatex(path + '/plot.xlsx', path + '/maeLF.tex')
#def xazi():
    #fshapea = ROOT_DIR + "/Shapefiles/{1}/{0}_{1}.shp".format(year,'ams')
    #fshape = osgeoutils.copyShape(fshapea, 'dissever', 'ams')
    #return fshape


