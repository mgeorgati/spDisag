# Main Script for data preparation -------------------------------------------------------------------------------------
# imports

import os,sys

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.wkt import loads
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.definitions import ROOT_DIR,year, city, ancillary_path,  engine, pop_path, cur,conn, gdal_rasterize_path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.mainFunctions.format_conversions import shptoraster

def csvtoshp(ROOT_DIR, city, year):
    
    pathI = ROOT_DIR + "/Evaluation/{}_groundTruth/population_20190101_ethnicity_grids.xlsx".format(city)
    ndf = pd.read_excel(pathI, header=0)
    print(ndf.head())
    ndf.dropna(subset=['geom_epsg3035'], inplace=True)
                     
    print(ndf.head())
    ndf = ndf.rename(columns={'Suriname':'sur', 'Antilles':'ant', 'Turkey':'tur', 'Morocco':'mar', 'non_western':'nonwestern','Western':'western', 'Dutch':'autoch' })
    # Create geopandas for large dataframe with all cleaned attributes 
    frame = gpd.GeoDataFrame(ndf, geometry=ndf['geom_epsg3035'].apply(loads), crs='epsg:3035')
    print(frame.head(2))
    frame = frame[['grid_id','sur', 'ant', 'tur', 'mar', 'nonwestern','western', 'autoch', 'geometry' ]]
    
    print("------------------------------ Creating shapefile:{0} on VectorGrid------------------------------".format(year))
    src_file = ROOT_DIR + "/Evaluation/{0}_groundTruth/{1}_dataVectorGridM.geojson".format(city,year)
    frame.to_file(src_file, driver='GeoJSON',crs="EPSG:3035")

    raster_file = ancillary_path + 'corine/waterComb_grootams_CLC_2012_2018.tif'
    for col in frame.columns:
        if col!='gridid' and col!='geometry':
            dst_file = ROOT_DIR + "/Evaluation/{1}_groundTruth/{0}_{1}_{2}.tif".format(year,city,col)
            shptoraster(raster_file, src_file, gdal_rasterize_path, dst_file, col, xres=100, yres=100)
        
    df = gpd.read_file(pop_path + '/OIS/gridCells/2018_dataVectorGrid.geojson')
    comp = df[['grid_id', "l1_totalpop",  "l2_children",  "l3_students",  "l4_mobile_adults",  "l5_not_mobile_adults",  "l6_elderly",  "geometry" ]]
    comp = comp.rename(columns={'l1_totalpop':'totalpop', 'l2_children':'children', 'l3_students':'students', 'l4_mobile_adults':'mobadults', 'l5_not_mobile_adults':'nmobadults','l6_elderly':'elderly' })
    src_file1 = ROOT_DIR + "/Evaluation/{0}_groundTruth/{1}_dataVectorGrid.geojson".format(city,year)
    comp.to_file(src_file1, driver='GeoJSON',crs="EPSG:3035")
    for col1 in comp.columns:
        if  col1!='gridid' and col1!='geometry':
            dst_file = ROOT_DIR + "/Evaluation/{1}_groundTruth/{0}_{1}_{2}.tif".format(year,city,col1)
            shptoraster(raster_file, src_file1, gdal_rasterize_path, dst_file, col1, xres=100, yres=100)
    
def compareData(ROOT_DIR, city, year):   
    
    gdf2 = gpd.read_file(ROOT_DIR + "/Evaluation/{0}_groundTruth/{1}_dataVectorGridM.geojson".format(city,year))
    gdf1 = gpd.read_file(ROOT_DIR + "/Evaluation/{0}_groundTruth/{1}_dataVectorGrid.geojson".format(city,year))
    gdf = gdf1.set_index('grid_id').join(gdf2.set_index('grid_id'), lsuffix='_l')
    df = pd.read_csv(ROOT_DIR + "/Statistics/{0}/{1}_{0}.csv".format(city,year))
    df =df[['totalpop'  ,'children' , 'students' , 'mobadults',  'nmobadults', 'elderly' ,  'sur',  'ant' ,  'tur',  'mar' ,'western'  ,'nonwestern',  'autoch' ]]
    print(gdf.head(2))
    print(gdf1.head(2))
    print(df.head(2))
    filenamemetrics2e = ROOT_DIR + "/Evaluation/{}_groundTruth/grid_neigh.csv".format(city)
    for col in df.columns:
        if (col!='KOMNAME') and (col!='Unnamed: 0'):
            gc = int(gdf['{}'.format(col)].sum())
            ng = int(df['{}'.format(col)].sum())
            dif = int(gc - ng) 
            print("----- Step #2: Writing CSV with Metrics -----")    
            if os.path.exists(filenamemetrics2e):
                with open(filenamemetrics2e, 'a') as myfile:
                    myfile.write(col + ';' + str(gc) + ';'+ str(ng) + ';' + str(dif) + '\n')       
            else:
                with open(filenamemetrics2e, 'w+') as myfile:
                    myfile.write('Comparison among the sum of the groups from municipality and grid cell ground truth data for Greater Copenhagen\n')
                    myfile.write('Value;GridCells;Municipalities;\n')
                    myfile.write(col + ';' + str(gc) + ';'+ str(ng) + ';' + str(dif) +'\n')
                    
#csvtoshp(ROOT_DIR, city, year)
compareData(ROOT_DIR, city, year)