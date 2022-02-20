import os
import sys
from osgeo import gdal
import numpy as np
import geopandas as gpd
import pandas as pd
from config.definitions import (ROOT_DIR, ancillary_path, pop_path,
                                gdal_rasterize_path, year, city)
from mainFunctions.format_conversions import shptoraster

def non_match_elements(list_a, list_b):
    non_match = []
    for i in list_a:
        if i not in list_b:
            non_match.append(i)
    return non_match

def restructureData(ROOT_DIR, city, year):
    df = gpd.read_file("K:/FUME/demo_popnet-Commit01/data_prep/cph_ProjectData/PopData/2018/temp_shp/2018_dataVectorGrid.geojson")
    df['children']= df[ "L2_P00_19"] * df["L1_SUM_POPULATION" ]
    df['students']= df["L3_P20_29" ] * df["L1_SUM_POPULATION"]
    df['mobadults']= df["L4_P30_44"] * df["L1_SUM_POPULATION"]
    df['nmobadults']= df["L5_P45_64"] * df["L1_SUM_POPULATION"]
    df['elderly']= df["L6_P65"] * df["L1_SUM_POPULATION"]
    
    #df.columns= df.columns.str.lower()
    df.columns= [col.replace('L10_', '') for col in df.columns]

    codes = pd.read_excel('K:/FUME/demo_popnet-Commit01/data_prep/cph_ProjectData/EXCEL/unsd_dk.xlsx')
    
    a = codes['abbr'].to_list()
    b = df.columns.to_list()
    #print(a,b)
    print(non_match_elements(a, b))
    #Rename columns
    #ldf = ndfC.rename(columns=codes.set_index('country')['abbr'])
    ldf =df.copy()
    regions = codes['region_abbr'].unique().tolist()
    print(regions)
    for key in regions:
        keyFrame = codes.loc[codes.region =='{}'.format(key)]
        select = keyFrame['abbr'].tolist()
        print(select)
        print(ldf.head(2))
        #ldf['{}'.format(key)] = ldf.loc[:, select].sum(axis=1)
        ldf['{}'.format(key)] = ldf[ldf.columns.intersection(select)].sum(axis=1)
        ldf['{}'.format(key)].astype(int)
        print(ldf['{}'.format(key)].sum())
    
    regionsEU = codes['eu'].unique().tolist()
    print(regionsEU)
    for key in regionsEU:
        keyFrame = codes.loc[codes.eu =='{}'.format(key)]
        select = keyFrame['abbr'].tolist()
        print(select)
        print(ldf.head(2))
        #ldf['{}'.format(key)] = ldf.loc[:, select].sum(axis=1)
        ldf['{}'.format(key)] = ldf[ldf.columns.intersection(select)].sum(axis=1)
        ldf['{}'.format(key)].astype(int)
        print(ldf['{}'.format(key)].sum())
    print(ldf.head(2))

    frame = ldf.set_index('Municipality').join(ndf.set_index('Municipality'))
    frame = ldf.reset_index()
    frame = frame.rename(columns={'no':'notEU', 'L1_SUM_POPULATION':'totalpop'})
    print(frame.head(3))
    frame['EU'] = frame['totalpop'] - (frame['yes'] + frame['efta'] + frame['uk'])
    
    
    selectedColumns= ['gridid','geometry','totalpop', 'children', 'students', 'mobadults', 'nmobadults', 'elderly', 
                        'DNK', 'AuNZ', 'CentAsia', 'EastAsia', 'EastEur', 'LAC', 'Melanesia', 'Micronesia', 'NorthAfr', 
                        'NorthAm', 'NorthEur', 'OTH', 'Polynesia', 'SEastAsia', 'SouthAsia', 'SouthEur', 'STA', 'SubSahAfr', 
                        'WestAsia', 'WestEur',
                        'EU', 'notEU']
    frame = frame[frame.columns.intersection(selectedColumns)]
    print(frame.head(10))
    #frame.to_csv(ROOT_DIR + "/Statistics/{1}/{0}_{1}.csv".format(year,city), encoding="utf-8")
    src_file = ROOT_DIR + "/Evaluation/groundTruth/{}_dataVectorGrid.geojson".format(year)
    frame.to_file(src_file, driver='GeoJSON',crs="EPSG:3035")

    raster_file = ancillary_path + '/corine/waterComb_cph_CLC_2012_2018.tif'.format(city)
    for col in selectedColumns:
        if col!='gridid' and col!='geometry':
            dst_file = ROOT_DIR + "/Evaluation/groundTruth/{0}_{1}_{2}.tif".format(year,city,col)
            shptoraster(raster_file, src_file, gdal_rasterize_path, dst_file, col, xres=100, yres=100)

def compareData(ROOT_DIR, city, year):
    gdf = gpd.read_file(ROOT_DIR + "/Evaluation/groundTruth/{}_dataVectorGrid.geojson".format(year))
    df = pd.read_csv(ROOT_DIR + "/Statistics/{0}/{1}_{0}.csv".format(city,year))
    print(gdf.head(2))
    print(df.head(2))
    filenamemetrics2e = ROOT_DIR + "/Evaluation/groundTruth/grid_neigh.csv"
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

def joinStatShap(ROOT_DIR, city, year):
    df = pd.read_csv(ROOT_DIR + "/Statistics/{0}/{1}_{0}.csv".format(city,year))
    gdf = gpd.read_file(ROOT_DIR + "/Shapefiles/{0}/{1}_{0}.shp".format(city,year))
    print(gdf.head(2))
    print(df.head(2))

    frame = df.set_index('KOMNAME').join(gdf.set_index('KOMNAME'))
    src_file = ROOT_DIR + "/Shapefiles/Comb/{0}_{1}.geojson".format(year,city)
    frame = gpd.GeoDataFrame(frame, geometry='geometry')
    frame.to_file(src_file, driver='GeoJSON',crs="EPSG:3035")

#restructureData(ROOT_DIR, city, year)
#compareData(ROOT_DIR, city, year)
joinStatShap(ROOT_DIR, city, year)
        