import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import glob
from pathlib import Path

from evaluateFunctions import zonalStat, zonalStat1, zonalStat2
from osgeoutils import *
from plotting.plotBoxPlot import BoxPlotCBS, BoxPlotCBS_NO
from plotting.plotVectors import plot_mapVectorPolygons
from dataPrepDK.dataPrep import non_match_elements

sys.path.append(os.path.dirname((os.path.dirname(os.path.abspath(__file__)))))
print(os.path.dirname((os.path.dirname(os.path.abspath(__file__)))))
from scripts.mainFunctions.format_conversions import shptoraster

def plotOISg(year, city, ancillary_path, pop_path, ROOT_DIR):
    polyPath = ROOT_DIR + "/Shapefiles/Withdrawals/{0}_{1}.shp".format(year,city)
    districtPath = ancillary_path + '/adm/ams_districts.geojson'
    waterPath = ancillary_path + 'corine/waterComb_grootams_CLC_2012_2018.tif'
    
    df = gpd.read_file(pop_path + '/OIS/gridCells/2018_dataVectorGrid1.geojson')
    comp = df[[ "l1_totalpop",  "l2_children",  "l3_students",  "l4_mobile_adults",  "l5_not_mobile_adults",  "l6_elderly",  "nonWestern",  "Western",
               "mar",  "ant",  "sur",  "tur",  "nld" ]]
    
    for col in comp:
        exportPath = ROOT_DIR + "/dataPrep/OIS/oisg_{0}_Grid.png".format(col)
        title ="Population Distribution (persons)\n({0})({1})".format(col, year)
        LegendTitle = "Population Distribution (persons)"
        plot_mapVectorPolygons(city,'popdistributionGrid', df, exportPath, title, LegendTitle, '{}'.format(col), districtPath = districtPath, neighPath = polyPath, waterPath = waterPath, invertArea = None, addLabels=True)
        
        
def popDataPrepOIS(year,ROOT_DIR, city, pop_path):
    """[Prepares the OIS data for Statistics.
    The OIS needs to be rasterized first.
    This function takes the neighborhood data of OIS and produces SHP without pop values and aggregates the grid cell data from MNC to this level]

    Args:
        year (int): [The reference year to work upon]
        engine ([Engine object based on a URL]): [Connection to database]
        ROOT_DIR ([str]): [parent directory]
        city ([str]): [City to work on]
        pop_path ([str]): [Directory to Population data with processed MNC dataset (TIF)]
    
    Returns:
        CSV in Statistics with all variables selected for predictions
        SHP in Shapefiles/Comb with all variables selected from Raw MNC data
    """ 
    ams = gpd.read_file(pop_path + '/OIS/buurt/ams_neighborhoods.json')
    ams = ams.to_crs('EPSG:3035')
    ams.to_file(ROOT_DIR + '/Shapefiles/Withdrawals/{0}_ams.shp'.format(year), driver='ESRI Shapefile', crs='EPSG:3035')
    
    popData = pd.read_excel(pop_path + '/OIS/buurt/2021_BBGA.xlsx')
    print(popData.head(2))
    pop = popData.loc[popData['jaar'] == year+1]
    pop.to_csv(ROOT_DIR + "/dataPrep/OIS/{}_ams.csv".format(year))
    
    pop = pd.read_csv(ROOT_DIR + "/dataPrep/OIS/{}_ams.csv".format(year))
    pop = pop.loc[pop['niveau'] == 8]
    dataToKeep = pop[['sdnaam', 'gebiedcode15', 'gebiedcodenaam','gebiednaam','BEVTOTAAL', 'BEV0_4','BEV5_9','BEV10_14','BEV15_19','BEV20_24',
                      'BEV25_29','BEV30_34','BEV35_39','BEV40_44','BEV45_49','BEV50_54','BEV55_59','BEV60_64','BEV65_69','BEV70_74','BEV75_79','BEV80_84','BEV85_89',
                        'BEV90_94','BEV95_99','BEV100PLUS', 'BEVSUR','BEVANTIL','BEVTURK','BEVMAROK','BEVOVNW','BEVWEST','BEVAUTOCH']]
    
    dataToKeep['Children'] = dataToKeep['BEV0_4'] + dataToKeep['BEV5_9'] + dataToKeep['BEV10_14'] + dataToKeep['BEV15_19']
    dataToKeep['Students'] =  dataToKeep['BEV20_24'] +  dataToKeep['BEV25_29']
    dataToKeep['MobileAdults'] =  dataToKeep['BEV30_34'] + dataToKeep['BEV35_39'] + dataToKeep['BEV40_44'] 
    dataToKeep['NotMobileAdults'] = dataToKeep['BEV45_49'] + dataToKeep['BEV50_54'] + dataToKeep['BEV55_59'] + dataToKeep['BEV60_64']
    dataToKeep['Elderly'] = dataToKeep['BEV100PLUS'] + dataToKeep['BEV95_99'] + dataToKeep['BEV90_94'] + dataToKeep['BEV85_89'] + dataToKeep['BEV80_84'] + dataToKeep['BEV75_79'] + dataToKeep['BEV70_74'] + dataToKeep['BEV65_69'] 
    dataToKeep = dataToKeep.rename(columns={'gebiedcode15':'Buurtcode','BEVTOTAAL':'totalpop','BEVSUR':'l10_sur', 'BEVANTIL':'l10_ant','BEVTURK':'l10_tur','BEVMAROK':'l10_mar',
                                            'BEVOVNW':'l10_nonwestern','BEVWEST':'l10_western','BEVAUTOCH':'autoch'})
    
    dataToKeep1 = dataToKeep[['sdnaam', 'Buurtcode', 'gebiedcodenaam','gebiednaam','totalpop', 'Children','Students','MobileAdults',
                             'NotMobileAdults', 'Elderly', 'l10_sur', 'l10_ant','l10_tur','l10_mar','l10_western','l10_nonwestern','autoch']]
    dataToKeep1.to_csv(ROOT_DIR + "/Statistics/Withdrawals/{}_ams.csv".format(year))
    
    frame = ams.set_index('Buurtcode').join(dataToKeep1.set_index('Buurtcode'), lsuffix='_l')
    for col in frame.columns:      
        if col.endswith("_l"):
            frame = frame.drop(columns=[col])
    
    frame = frame.fillna(0)
    frame.to_file(ROOT_DIR + '/Shapefiles/Comb/{0}_ams_ois.shp'.format(year), driver='ESRI Shapefile', crs='EPSG:3035')

def getStatisticsOIS(year, ROOT_DIR):
    fcsv = ROOT_DIR + "/Statistics/Withdrawals/{}_ams.csv".format(year)
    df =pd.read_csv(fcsv)
    filenamemetrics2e = ROOT_DIR + '/dataPrep/OIS/ams_ois_summary.csv'
    index = len(df.index)
    nanPop = df['totalpop'].isna().sum()
    r = df['totalpop'].sum()
    r1 = df['Children'].sum()
    r2 = df['Students'].sum()
    r3 = df['MobileAdults'].sum()
    r4 = df['NotMobileAdults'].sum()
    r5 = df['Elderly'].sum()
    
    r6 = df['l10_sur'].sum()
    r7 = df['l10_ant'].sum()
    r8 = df['l10_mar'].sum()
    r9 = df['l10_tur'].sum()
    r10 = df['l10_western'].sum()
    r11 = df['l10_nonwestern'].sum()
    r12 = df['autoch'].sum()
    
    print("----- Writing CSV with Metrics -----")    
    if os.path.exists(filenamemetrics2e):
        with open(filenamemetrics2e, 'a') as myfile:
            myfile.write(str(year) + ';' + str(r) + ';' + str(r1) + ';'+ str(r2) + ';' + str(r3) + ';' + str(r4) + ';' + str(r5) + ';' 
                        + str(r6) + ';' + str(r7) + ';' + str(r8) +  ';' + str(r9) + ';' + str(r10) + ';' + str(r11) + ';' + str(r12) + ';' + str(index) + ';' + str(nanPop) + '\n')       
    else:
        with open(filenamemetrics2e, 'w+') as myfile:
            myfile.write('Summary of Data for the Municipality of Amsterdam, source:OIS\n')
            myfile.write('Year;TotalPopulation;Children;Students;MobileAdults;NotMobileAdults;Elderly;Sur;Ant;Mar;Tur;Western;nonWestern;Natives;Neighborhoods;EmptyNeighborhoods\n')
            myfile.write(str(year) + ';' + str(r) + ';' + str(r1) + ';'+ str(r2) + ';' + str(r3) + ';' + str(r4) + ';' + str(r5) + ';' 
                        + str(r6) + ';' + str(r7) + ';' + str(r8) +  ';' + str(r9) + ';' + str(r10) + ';' + str(r11) + ';' + str(r12) + ';' + str(index) + ';' + str(nanPop) + '\n')
            

def compOISn_OISg(year, city, pop_path, cur, conn, engine, ancillary_path):
    """[Makes a common file for CBS, OIS nad aggregated OIS dataset in order to inticate differences]

    Args:
        year (int): [The reference year to work upon]
        engine ([Engine object based on a URL]): [Connection to database]
        ROOT_DIR ([str]): [parent directory]
        city ([str]): [City to work on]
        pop_path ([str]): [Directory to Population data with processed MNC dataset (TIF)]
    
    Returns:
        CSV in Statistics with all variables selected for predictions
        SHP in Shapefiles/Comb with all variables selected from Raw MNC data
    """ 
    
    #Step1: get OISn data
    ois = gpd.read_file(ROOT_DIR + '/Shapefiles/Comb/{0}_ams_ois.shp'.format(year,city))
    ois_cols =["Buurtcode", "Buurtnaam", "totalpop", "children","students","mobadults","nmobadults","elderly","sur","ant","mar","tur","wester","nonwes" ]
    frame = ois[ois_cols]
    for col in frame.columns:
        if col != 'Buurtcode' and col != 'geometry' and col != 'Buurtnaam':
            frame = frame.rename(columns={'{}'.format(col):'ois_{}'.format(col)})        
    
    #frame = frame.join(ois_frame.set_index('Buurtnaam'), on='WijkenEnBu', lsuffix='_l')
    
    #Step2: get OISg tiffs and aggregate with 2ways of zonal statitistc with rasterstat
    tifPaths = glob.glob(pop_path + "/OIS/gridCells/{}_*.tif".format(year))
    ams_path= ROOT_DIR + '/Shapefiles/Withdrawals/{0}_ams.shp'.format(year,city)
    ams_neigh = gpd.read_file(ams_path)
    ams_neigh = ams_neigh[['Buurtcode', 'Buurtnaam','geometry']]
    statistics='sum'
    for i in tifPaths:
        path = Path(i)
        fileName = path.stem 
    
        dst_file = ROOT_DIR + '/dataPrep/OIS/aggregated0/{0}.geojson'.format(fileName)
        print(path, fileName, dst_file)
        zonalStat(path, dst_file, ams_path, statistics)
        ois_gc = gpd.read_file(dst_file)
        ois_gc = ois_gc.rename(columns={'sum_':'ois_gc_{}'.format(fileName)})        
        frame = frame.join(ois_gc.set_index('Buurtcode'), on='Buurtcode', lsuffix='_l') 
 
        dst_file = ROOT_DIR + '/dataPrep/OIS/aggregated0/{0}_1.geojson'.format(fileName)
        zonalStat1(path, dst_file, ams_path, statistics)
        ois_gc = gpd.read_file(dst_file)
        ois_gc = ois_gc.rename(columns={'sum_':'ois_gc1_{}'.format(fileName)})        
        frame = frame.join(ois_gc.set_index('Buurtcode'), on='Buurtcode', lsuffix='_l')
        
        dst_file = ROOT_DIR + '/dataPrep/OIS/aggregated0/{0}_1.geojson'.format(fileName)
        zonalStat2(path, dst_file, ams_path, statistics)
        ois_gc = gpd.read_file(dst_file)
        ois_gc = ois_gc.rename(columns={'sum_':'ois_gc2_{}'.format(fileName)})        
        frame = frame.join(ois_gc.set_index('Buurtcode'), on='Buurtcode', lsuffix='_l')
    
    # Step 3: Compute based on area with PostGIS 
    
    vectorPopPath = pop_path + "/OIS/gridCells/{}_dataVectorGrid.geojson".format(year)
    vectorPop = gpd.read_file(vectorPopPath)

    # Create Table for population
    print("---------- Creating table for city, if it doesn't exist ----------")
    print("Checking {0} Pop table".format(year))
    cur.execute("SELECT EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = '{0}_dataVectorGrid');".format(year))
    check = cur.fetchone()
    if check[0] == True:
        print("{0}_dataVectorGrid already exists".format(year))

    else:
        print("Creating {0} pop table ".format(city))
        vectorPop.to_postgis('{0}_dataVectorGrid'.format(year),engine)
    
    # Create Table for neighborhoods
    print("---------- Creating table for city, if it doesn't exist ----------")
    print("Checking {0} Pop table".format(year))
    cur.execute("SELECT EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = '{0}_{1}_ois');".format(year,city))
    check = cur.fetchone()
    if check[0] == True:
        print("{0}_{1}_ois already exists".format(year,city))
    else:
        print("Creating {0} pop table ".format(city))
        ois.to_postgis('{0}_{1}_ois'.format(year,city),engine)
        

    query = """Select * FROM "2018_ams_ois" """
    inter = gpd.read_postgis(query, engine)
    inter = inter.rename(columns={'geom':'geometry'})
    nf = gpd.GeoDataFrame(inter, geometry= 'geometry')
    frame = frame.join(nf.set_index('Buurtcode'), on='Buurtcode', lsuffix='_l')
    nf.to_file(ROOT_DIR + '/dataPrep/OIS/aggregated0/OIS_OISaggrPostgis.geojson',driver='GeoJSON', crs='EPSG:3035')
    
    print("kati allo edo")             
    frame = frame.loc[:,~frame.columns.duplicated()]
    for col in frame.columns:
        if col.endswith('_l'):
            frame = frame.drop(columns=col)
    frame = frame.reset_index()
    print(frame.head(2))
    
    frame = gpd.GeoDataFrame(frame, geometry='geometry')
    selection =[]
    for col in frame.columns:
        if col.startswith('ois_'):
            selection.append(col)
    extra =['Buurtcode','Buurtnaam','geometry']
    selection.extend(extra)
    frame = frame[selection]
    frame['dif_gc_totalpop']  = frame["ois_totalpop"] - frame["ois_gc_2018_totalpop"]
    frame['dif_gc1_totalpop'] = frame["ois_totalpop"] - frame["ois_gc1_2018_totalpop"]
    frame['dif_gc2_totalpop'] = frame["ois_totalpop"] - frame["ois_gc2_2018_totalpop"]
    frame['dif_gc3_totalpop'] = frame["ois_totalpop"] - frame["ois_gc3_l1_totalpop"]
    
    frame['pe_gc_totalpop']  = (frame["ois_gc_2018_totalpop"] / frame["ois_totalpop"] ) *100
    frame['pe_gc1_totalpop'] = (frame["ois_gc1_2018_totalpop"] / frame["ois_totalpop"]) *100
    frame['pe_gc2_totalpop'] = (frame["ois_gc2_2018_totalpop"] / frame["ois_totalpop"]) *100
    frame['pe_gc3_totalpop'] = (frame["ois_gc3_l1_totalpop"]  / frame["ois_totalpop"] ) *100 
    frame.to_file(ROOT_DIR + '/dataPrep/OIS/{0}_amsOISngComparison.geojson'.format(year),driver='GeoJSON', crs='EPSG:3035')
    nframe = frame.drop(columns='geometry')
    nframe.to_csv(ROOT_DIR + '/dataPrep/OIS/{0}_amsOISngComparison.csv'.format(year),index=False)
    
def plot_dif(ROOT_DIR,city, year, ancillary_path):
    frame = gpd.read_file(ROOT_DIR + '/dataPrep/OIS/{0}_amsOISngComparison.geojson'.format(year))
    polyPath = ROOT_DIR + "/Shapefiles/Withdrawals/{0}_{1}.shp".format(year,city)
    districtPath = ancillary_path + '/adm/ams_districts.geojson'
    waterPath = ancillary_path + 'corine/waterComb_grootams_CLC_2012_2018.tif'
    
    for col in frame.columns:
        if col.startswith('dif_'):
            exportPath = ROOT_DIR + "/dataPrep/OIS/aggregated0/{}_Polyg.png".format(col)
            title ="Mean Error (persons)\n(Comparison between the OISn and the aggregated OISg, source:CBS)\n({0},{1})".format(col, year)
            LegendTitle = "Mean Error (persons)"
            plot_mapVectorPolygons(city, 'mae', frame , exportPath, title, LegendTitle, '{}'.format(col) , districtPath = districtPath, neighPath = polyPath, waterPath = waterPath, invertArea = None, addLabels=True)
        if col.startswith('pe_'):
            exportPath = ROOT_DIR + "/dataPrep/OIS/aggregated0/{}_Polyg.png".format(col)
            title ="Percentage Accuracy by Neighborhood (Comparison between the OISn and the aggregated OISg, source:CBS)\n({0},{1})".format(col, year)
            LegendTitle = "Accuracy (%)"
            plot_mapVectorPolygons(city, 'div', frame , exportPath, title, LegendTitle, '{}'.format(col) , districtPath = districtPath, neighPath = polyPath, waterPath = waterPath, invertArea = None, addLabels=True)          


def restructureData(pop_path, raster_file, gdal_rasterize_path , city, year, popraster):
    df = gpd.read_file(pop_path + "/OIS/gridCells/2018_dataVectorGrid.geojson")
    
    codes = pd.read_excel('C:/FUME/PopNetV2/data_prep/ams_ProjectData/PopData/EXCEL/codes.xlsx')
    
    a = codes['abbr'].to_list()
    string_list = [each_string.lower() for each_string in a]

    #b = df.columns.to_list()
    #print(a,string_list)
    #print(non_match_elements(string_list, b))
    #Rename columns
    b = codes['wnw'].unique().tolist()
    cleanedList = [x for x in b if str(x) != 'nan']
    print(b)
    regions = [each_string.lower() for each_string in cleanedList]
    print(regions)

    for key in regions:
        keyFrame = codes.loc[codes.wnw =='{}'.format(key)]
        selectS = keyFrame['abbr'].tolist()
        select  = [each_string.lower() for each_string in selectS]
        print(select)
        #ldf['{}'.format(key)] = ldf.loc[:, select].sum(axis=1)
        df['{}'.format(key)] = df[df.columns.intersection(select)].sum(axis=1)
        df['{}'.format(key)].astype(int)
        print(df['{}'.format(key)].sum())
    
    frame = df.rename(columns={'l1_totalpop':'totalpop'})
    selectedColumns= ['grid_id','totalpop', 'sur','ant', 'tur', 'mar', 'western', 'nonwestern', 'autoch','geometry' ]
    #colsExtra= ["totalpop", "children","students","mobadults","nmobadults","elderly"]
    cols =['sur','ant', 'tur', 'mar', 'western', 'nonwestern', 'autoch']
    frame = frame[selectedColumns]
    #print(frame.head(3))
    src_file = pop_path + "/OIS/gridCells/2018_dataVectorGridSelection.geojson"
    frame.to_file(src_file, driver='GeoJSON', crs='epsg:3035')
    
    for i in cols:
        dst_file = 'C:/FUME/DasymetricMapping/SDis_Self-Training/Evaluation/{1}_groundTruth/backup/{0}_{1}_{2}.tif'.format(year,city,i)
        #shptoraster(raster_file, src_file, gdal_rasterize_path, dst_file, i, xres=100, yres=100)

        ds, rastergeo = osgu.readRaster(os.path.join(ROOT_DIR, 'Rasters', '{}_GHS'.format(city), popraster))
        nrowsds = ds.shape[1]
        ncolsds = ds.shape[0]
        fshapea = ROOT_DIR + "/Shapefiles/{1}/{0}_{1}.shp".format(year,city)
        fshape = osgu.copyShape(fshapea, 'converting')
        idsdataset = osgu.ogr2raster(fshape, attr='ID', template=[rastergeo, nrowsds, ncolsds])[0]

        pycnomask = np.copy(idsdataset)
        pycnomask[~np.isnan(pycnomask)] = 1
        tddataset, rastergeo = osgu.readRaster(dst_file)
        tddataset = tddataset * pycnomask
    
        inputRaster = 'C:/FUME/DasymetricMapping/SDis_Self-Training/Evaluation/{1}_groundTruth/{0}_{1}_{2}.tif'.format(year,city,i)
        osgu.writeRaster(tddataset[:,:,0], rastergeo, inputRaster ) #

                       
#########################################################################
def getStatisticsOIS_GC(year, ROOT_DIR):
    rawMNC= "C:/FUME/PopNetV2/data_prep/ams_ProjectData/PopData/rawData/nidi"
    path = rawMNC +  "/nidi{}.xlsx".format(year)
    df = pd.read_excel(path, header=0, skiprows=2 )
    
    'TOTALPOP', 'CHILDREN', 'STUDENTS', 'MOBILE_ADULTS', 'NOT_MOBILE_ADULTS', 'ELDERLY', 'IMMIGRANTS', 'EU_EFTA_IMMIGRANTS', 'NON_EU_EFTA_IMMIGRANTS',
    
    filenamemetrics2e = ROOT_DIR + '/dataPrep/OIS/ams_oisGC_summary.csv'
    index = len(df.index)
    nanPop = df['totalpop'].isna().sum()
    r = df['totalpop'].sum()
    r1 = df['Children'].sum()
    r2 = df['Students'].sum()
    r3 = df['MobileAdults'].sum()
    r4 = df['NotMobileAdults'].sum()
    r5 = df['Elderly'].sum()
    
    r6 = df['l10_sur'].sum()
    r7 = df['l10_ant'].sum()
    r8 = df['l10_mar'].sum()
    r9 = df['l10_tur'].sum()
    r10 = df['l10_western'].sum()
    r11 = df['l10_nonwestern'].sum()
    r12 = df['autoch'].sum()
    
    print("----- Writing CSV with Metrics -----")    
    if os.path.exists(filenamemetrics2e):
        with open(filenamemetrics2e, 'a') as myfile:
            myfile.write(str(year) + ';' + str(r) + ';' + str(r1) + ';'+ str(r2) + ';' + str(r3) + ';' + str(r4) + ';' + str(r5) + ';' 
                        + str(r6) + ';' + str(r7) + ';' + str(r8) +  ';' + str(r9) + ';' + str(r10) + ';' + str(r11) + ';' + str(r12) + ';' + str(index) + ';' + str(nanPop) + '\n')       
    else:
        with open(filenamemetrics2e, 'w+') as myfile:
            myfile.write('Summary of Data for the Municipality of Amsterdam, source:OIS\n')
            myfile.write('Year;TotalPopulation;Children;Students;MobileAdults;NotMobileAdults;Elderly;Sur;Ant;Mar;Tur;Western;nonWestern;Natives;Neighborhoods;EmptyNeighborhoods\n')
            myfile.write(str(year) + ';' + str(r) + ';' + str(r1) + ';'+ str(r2) + ';' + str(r3) + ';' + str(r4) + ';' + str(r5) + ';' 
                        + str(r6) + ';' + str(r7) + ';' + str(r8) +  ';' + str(r9) + ';' + str(r10) + ';' + str(r11) + ';' + str(r12) + ';' + str(index) + ';' + str(nanPop) + '\n')
 
 
def popDataPrepMNC(year,ROOT_DIR, city, pop_path):
    """[Prepares the MNC data for Statistics.
    The MNC needs to be rasterized first.
    This function takes the neighborhoods of Munic of Amsterdam and produces SHP without pop values and aggregates the grid cell data from MNC to this level]

    Args:
        year (int): [The reference year to work upon]
        engine ([Engine object based on a URL]): [Connection to database]
        ROOT_DIR ([str]): [parent directory]
        city ([str]): [City to work on]
        pop_path ([str]): [Directory to Population data with processed MNC dataset (TIF)]
    
    Returns:
        CSV in Statistics with all variables selected for predictions
        SHP in Shapefiles/Comb with all variables selected from Raw MNC data
    """ 
    src = gpd.read_file(ROOT_DIR + '/Shapefiles/Withdrawals/{0}_{1}.shp'.format(year,city))
    df = src.loc[src['BU_CODE'].str.contains('BU0363')]
    ams_neigh = ROOT_DIR + '/Shapefiles/{0}_ams.shp'.format(year)
    df.to_file(ams_neigh,driver='ESRI Shapefile', crs='EPSG:3035')
    
    cbs = gpd.read_file(ROOT_DIR + '/Shapefiles/Comb/{0}_{1}.shp'.format(year,city))
    cbs = cbs.loc[cbs['BU_CODE'].str.contains('BU0363')]
    cols = ['BU_CODE','l1_totalpo',"l2_childre","l3_student","l4_mobile_","l5_not_mob","l6_elderly"]
    frame = cbs[cols]
    for col in frame.columns:
        if col != 'BU_CODE' and col != 'geometry':
            frame = frame.rename(columns={col:'cbs_{}'.format(col)})
    frame= frame.set_index('BU_CODE')
    statistics = 'sum'
    tifPaths = glob.glob(pop_path + "/gridCells/{}_*.tif".format(year))

    for i in tifPaths:
        path = Path(i)
        
        fileName = path.stem 
        dst_file = ROOT_DIR + '/dataPrep/MNC/{0}.geojson'.format(fileName)
        print(path, fileName, dst_file)
        zonalStat(path, dst_file, ams_neigh, statistics)
        mnc = gpd.read_file(dst_file)
        mnc = mnc.rename(columns={'sum_':'mnc_{}'.format(fileName)})        
        frame = frame.join(mnc.set_index('BU_CODE'), on='BU_CODE', lsuffix='_l')
            
    frame = frame.loc[:,~frame.columns.duplicated()]
    for col in frame.columns:
        if col.endswith('_l'):
            frame = frame.drop(columns=col)
    frame = frame.reset_index()
    
    frame = gpd.GeoDataFrame(frame, geometry='geometry')
    frame.to_file(ROOT_DIR + '/dataPrep/CBS_MNC/{0}_amsComparison.geojson'.format(year),driver='GeoJSON', crs='EPSG:3035')
    nframe = frame.drop(columns='geometry')
    nframe.to_csv(ROOT_DIR + '/dataPrep/CBS_MNC/{0}_amsComparison.csv'.format(year),index=False)
    
    for col in frame.columns:
        if col.startswith('cbs'):
            frame = frame.drop(columns=col)
    frame.to_file(ROOT_DIR + '/Shapefiles/Comb/{0}_ams.shp'.format(year), driver='ESRI Shapefile', crs='EPSG:3035')
    gframe = frame.drop(columns='geometry')
    gframe.to_csv(ROOT_DIR + '/Statistics/Withdrawals/{0}_ams.csv'.format(year),index=False)
    

def updateCBSfromOIS(year,ROOT_DIR):
    """[Updates CBS data for Statistics.
    The MNC needs to be rasterized first.
    This function takes the neighborhoods of Munic of Amsterdam and produces SHP without pop values and aggregates the grid cell data from MNC to this level]

    Args:
        year (int): [The reference year to work upon]
        engine ([Engine object based on a URL]): [Connection to database]
        ROOT_DIR ([str]): [parent directory]
        city ([str]): [City to work on]
        pop_path ([str]): [Directory to Population data with processed MNC dataset (TIF)]
    
    Returns:
        CSV in Statistics with all variables selected for predictions
    """ 
    cbs_mnc = pd.read_csv(ROOT_DIR + '/dataPrep/CBS_MNC/{0}_amsComparison.csv'.format(year))
    print(cbs_mnc.columns)
    cbs_mnc.reset_index()
    for index, row in cbs_mnc.iterrows():
        if row['cbs_l1_totalpo'] == 0 and row['mnc_2018_l1_totalpop']> 0:
            cbs_mnc.at[index,'cbs_l1_totalpo'] = cbs_mnc.at[index,'mnc_2018_l1_totalpop']
        
        if row['cbs_l2_childre'] == 0 and row['mnc_2018_l2_children']> 0:
            cbs_mnc.at[index,'cbs_l2_childre'] = cbs_mnc.at[index,'mnc_2018_l2_children']
        
        if row['cbs_l3_student'] == 0 and row['mnc_2018_l3_students']> 0:
            cbs_mnc.at[index,'cbs_l3_student'] = cbs_mnc.at[index,'mnc_2018_l3_students']
        
        if row['cbs_l4_mobile_'] == 0 and row['mnc_2018_l4_mobile_adults']> 0:
            cbs_mnc.at[index,'cbs_l4_mobile_'] = cbs_mnc.at[index,'mnc_2018_l4_mobile_adults']
        
        if row['cbs_l5_not_mob'] == 0 and row['mnc_2018_l5_not_mobile_adults']> 0:
            cbs_mnc.at[index,'cbs_l5_not_mob'] = cbs_mnc.at[index,'mnc_2018_l5_not_mobile_adults']
        
        if row['cbs_l6_elderly'] == 0 and row['mnc_2018_l6_elderly']> 0:
            cbs_mnc.at[index,'cbs_l6_elderly'] = cbs_mnc.at[index,'mnc_2018_l6_elderly']

    cbs_mnc.to_csv(ROOT_DIR + '/dataPrep/CBS_MNC/{0}_amsComparisonUpdated.csv'.format(year),index=False)
    
    
