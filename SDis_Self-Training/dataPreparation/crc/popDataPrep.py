# Rasterize shp by totalmig 
# Aggregate to district level by sum --> export csv  
from mainFunctions import shptoraster, zonalStat, jsonTOxlxs
from config import gdal_rasterize_path, pop_path, ancillary_path, parent_path
import geopandas as gpd
import pandas as pd
city ='crc'
raster_file = ancillary_path + "/{0}/corine/waterComb_crc_CLC_2006_2012.tif".format(city)

def popData_crc(year):
    #src_file = "C:/FUME/PopNetV2/data_prep/{0}_ProjectData/PopData/combined/griddedDataset/{1}_dataVectorGrid.gpkg".format(city, year)
    src_file = "C:/FUME/PopNetV2/data_prep/{0}_ProjectData/PopData/voivodeship/{1}/temp_shp/{1}_dataVectorGridMUW.gpkg".format(city, year)
    column_name = "totalmig"
    dst_file = pop_path + "/{0}/GridCells/rasters/{1}_totalmig.tif".format(city, year)
    shptoraster(raster_file, src_file, gdal_rasterize_path, dst_file, column_name, xres=100, yres=100)
    
    polyPath = parent_path + "/SDis_Self-Training/Shapefiles/{0}/2013_{0}.shp".format(city, year)
    dst_fileAggr = parent_path + "/SDis_Self-Training/Statistics/{0}/{1}_{0}.geojson".format(city, year)
    zonalStat(dst_file, dst_fileAggr, polyPath, 'sum')

    output = parent_path + "/SDis_Self-Training/Statistics/{0}/{1}_{0}A.csv".format(city, year)
    gdf = gpd.read_file(dst_fileAggr)
    gdf = gdf[['id', 'DZIELNICA', 'sum_']]
    gdf = gdf.rename(columns={'sum_': 'totalmig'})
    with open(parent_path + "/SDis_Self-Training/Statistics/{0}/{1}_{0}.xlsx".format(city, year)) as f:
        print(f)
    df = pd.read_excel(parent_path + "/SDis_Self-Training/Statistics/{0}/{1}_{0}.xlsx".format(city, year))
    ndf = pd.merge(df, gdf, on='DZIELNICA', how='outer')
    
    print(ndf)
    ndf.to_csv(output, encoding = "utf-8")
    
    