import os
import sys
import numpy as np
import geopandas as gpd
import pandas as pd
import cbsodata

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from plotting.plotBoxPlot import BoxPlotCBS, BoxPlotCBS_NO
from plotting.plotVectors import plot_mapVectorPolygons

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.mainFunctions.basic import createFolder
from scripts.mainFunctions.excel_sql import dfTOxls
# In thi case there 
def downloadDataCBS(pop_path):
    tables = pd.DataFrame(cbsodata.get_table_list())
    tables = tables[["Identifier", "Title","ShortTitle", "Period"]]
    # Search with key words in tables
    mask = np.column_stack([tables["ShortTitle"].str.contains("migrat")])
    migrat_tables = tables.loc[mask.any(axis=1)]
    outTables = os.path.dirname(pop_path) + "/{}_ProjectData/keyTables/"

    maskNeigh = np.column_stack([tables["ShortTitle"].str.contains("wijk")])
    neigh_tables = tables.loc[maskNeigh.any(axis=1)]
    #dfTOxls(outTables, 'NeighborhoodTables', neigh_tables)
    maskPop = np.column_stack([neigh_tables["ShortTitle"].str.contains("Kerncijfers wijken en buurten")])
    pop_tables = neigh_tables.loc[maskPop.any(axis=1)]
    pop_tablesSeries = pop_tables[pop_tables.Period.str.contains("-") == True]
    pop_tablesYears = pop_tables[pop_tables.Period.str.contains("-") == False]
    # Download data for separate years (2014 - 2020) + (1995 - 2003)
    id = pop_tablesYears["Identifier"].tolist()
    rawDataPath = pop_path + '/rawDataMigration'
    createFolder(rawDataPath)
    for i in range(len(id)):
        year = pop_tablesYears.iloc[i]["Period"]
        year = int(year)
        if year == 2019 :
            print(id[i],year)
            data = pd.DataFrame(
                cbsodata.get_data("{}".format(id[i])))
            dfTOxls(rawDataPath, "{}_keyFig".format(year), data)   
        else:
            print('Year: {} is not selected'.format(year))  

def popDataPrepCBS(year, cur,conn, engine, ROOT_DIR, city, pop_path, ancillary_path):
    """[Prepares the CBS data for Statistics.
    This function takes the raw CBS tables and joins it to the isolated neighborhood polygons that have the majority of their area in the NUTS 3 area 
    (with intersectedBUURT.area > intersectedBUURT.area/2), renames attributes based on neighData_keyFig/abbr_codes.xlsx
    Produces CSV with desired attributes and SHP in Comb with all attributes]

    Args:
        year (int): [The reference year to work upon]
        engine ([Engine object based on a URL]): [Connection to database]
        ROOT_DIR ([str]): [parent directory]
        city ([str]): [City to work on]
        pop_path ([str]): [Directory to Population data with raw CBS dataset]
    
    Returns:
        CSV in Statistics/temp
        CSV in Statistics
        SHP in Shapefiles/Comb with all variables selected from Raw CBS data
    """
    """    
    gdf = gpd.read_file(ancillary_path + "/adm/neighborhoodYears/buurt_{0}/buurt_{0}_v3.shp".format(year+1))
    # Create Table for Railway
    print("---------- Creating table for buurt, if it doesn't exist ----------")
    print("Checking {0} Case Study table".format(city))
    cur.execute("SELECT EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = '{0}_buurt');".format(year+1))
    check = cur.fetchone()
    if check[0] == True:
        print("{0} streets table already exists".format(city))
        cur.execute("DROP TABLE IF EXISTS {0}_streets;".format(city))
        conn.commit()
    else:
        print("Creating {0} streets ".format(city))
        gdf.to_postgis('{0}_buurt'.format(year+1), engine)
    """
    #Preparing data for disaggregatio, take the processed dataset from CleanSelection, dissolve, and save shp to Shapefiles, and population groups in csv in Statistics
    query = """Select "{0}_buurt"."BU_CODE","{0}_buurt"."BU_NAAM", ST_Transform("{0}_buurt".geometry,3035) as geom from "{0}_buurt" ,  "nuts3_grootAms" \
                WHERE ST_Intersects( ST_Transform("{0}_buurt".geometry,3035),  "nuts3_grootAms".geom) \
                AND ST_AREA(ST_Intersection(ST_Transform("{0}_buurt".geometry,3035),"nuts3_grootAms".geom) ) > ST_AREA(ST_Transform("{0}_buurt".geometry,3035))/2""".format(year+1)
    inter = gpd.read_postgis(query, engine)
    inter = inter.rename(columns={'geom':'geometry'})
    nf = gpd.GeoDataFrame(inter, geometry= 'geometry')
    print(inter.head(2))
    nf.to_file(ROOT_DIR + '/Shapefiles/{0}_{1}.shp'.format(year,city),driver='ESRI Shapefile', crs='EPSG:3035')
    
    rawDataPath = pop_path + '/CBS/rawDataMigration'
    path = rawDataPath + "/{}_keyFig.xlsx".format(year+1)
    df = pd.read_excel(path, header=0)
    ndf = df.iloc[: , 1:] 
    if 'Perioden' in df.columns: 
        ndf = ndf.drop(columns=['Perioden'])
    if 'BU_CODE' not in ndf.columns:
        print('Key column is renamed')
        ndf = ndf.rename(columns={'BU_2004':'BU_CODE'}) 
        ndf = ndf.rename(columns={'Codering_3':'BU_CODE'})
    
    frame = nf.set_index('BU_CODE').join(ndf.set_index('BU_CODE'), lsuffix='_l')

    frame.to_csv(ROOT_DIR + '/Statistics/temp/{0}_{1}.csv'.format(year,city),index=False)
    
    codes = pd.read_excel(pop_path + "/CBS/abbr_codes.xlsx", header=0)
    codes_df = codes.iloc[:,0:2]
    for index, row in codes_df.iterrows():
        character = "_" + row['columnName'].split("_")[-1]
        row['columnName'] = row['columnName'].split("{}".format(character))[0]
        print(row['columnName'])
    
    dictionary = dict(zip(codes_df["columnName"], codes_df["abb"]))
    
    keys = []
    values = []
    for col in frame.columns:
        char = "_" + col.split("_")[-1]
        columnname = col.split("{}".format(char))[0] 
        keys.append(col)
        values.append(columnname) 
    dictionary1 = dict(zip(keys, values))
    ndf = frame.copy()
    ndf = ndf.rename(columns = dictionary1)
    ng_new = ndf.rename(columns = dictionary)
    index = ng_new.index
    number_of_rows = len(index) 
    print(number_of_rows)
    print(ng_new.head(3))
    
    listCol = list(set(codes_df["abb"].to_list()))
    listColA = frame.iloc[:, 0:10].columns.to_list()
    listCol.extend(listColA)
    listCol.append('geometry')
    
    ndfL = ng_new[ng_new.columns.intersection(listCol)]
    ndfL = ndfL.loc[:,~ndfL.columns.duplicated()]
    ndfL = ndfL.fillna(0).reset_index()
    nframe = ndfL
    nframe = nframe[['BU_CODE','l1_totalpop', 'l2_children', 'l3_students', 'l4_mobile_adults', 'l5_not_mobile_adults', 'l6_elderly']]    
    nframe.to_csv(ROOT_DIR + '/Statistics/{0}_{1}.csv'.format(year,city),index=False)
    gdf = gpd.GeoDataFrame(ndfL, geometry='geometry')
    gdf.to_file(ROOT_DIR + '/Shapefiles/Comb/{0}_{1}.shp'.format(year,city),driver='ESRI Shapefile', crs='EPSG:3035')
    
def getStatisticsCBS(year, ROOT_DIR, city):
    """[Summarizes the statistscs of the CBS, comparison total population to age groups]

    Args:
        year (int): [The reference year to work upon]
        ROOT_DIR ([str]): [parent directory]
        city ([str]): [City to work on]
    
    Returns:
        CSV CBS statistics
    """ 
    df = gpd.read_file(ROOT_DIR + '/Shapefiles/Comb/{0}_{1}.shp'.format(year,city))
    codes = df['BU_CODE'].unique()
    munCodes = []
    for i in codes:
        code = i[2:6]
        munCodes.append(code)
    munCodes = set(list( munCodes))
    print(munCodes)
    for x in munCodes:
        ndf = df.loc[df['BU_CODE'].str.contains('BU{}'.format(x))]
        sum1 = ndf['l1_totalpo'].sum()
        sum2 = ndf["l2_childre"].sum()
        sum3 = ndf["l3_student"].sum()
        sum4 = ndf["l4_mobile_"].sum()
        sum5 = ndf["l5_not_mob"].sum()
        sum6 = ndf["l6_elderly"].sum()
        
        print('The total Population of municipality:', x,':', sum)
        print("-----WRITING CSV-----")
        # Calculate the mean difference and the quotient of total grid cells and write it in csv
        filenamemetrics2e = ROOT_DIR + '/dataPrep/CBS/{0}_{1}_statistics.csv'.format(year,city)
        
        if os.path.exists(filenamemetrics2e):
            with open(filenamemetrics2e, 'a') as myfile:
                myfile.write(x + ';' + str(sum1) + ';' + str(sum2)+ ';' + str(sum3) + ';' + str(sum4) + ';' + str(sum5) + ';' + str(sum6) + '\n')       
        else:
            with open(filenamemetrics2e, 'w+') as myfile:
                myfile.write('Statistics on Neighborhood data (source:CBS)\n')
                myfile.write('MunicipalityCode;Population;Children;Students;MobileAdults;NotMobileAdults;Elderly\n')
                myfile.write(x + ';' + str(sum1) + ';' + str(sum2)+ ';' + str(sum3) + ';' + str(sum4) + ';' + str(sum5) + ';' + str(sum6) + '\n')
 
def plotCBS(year, ROOT_DIR, city,ancillary_path, engine):
    """[Prepares the CBS data for Statistics]

    Args:
        year (int): [The reference year to work upon]
        engine ([Engine object based on a URL]): [Connection to database]
        ROOT_DIR ([str]): [parent directory]
        city ([str]): [City to work on]
        ancillary_path ([str]): [Directory to Ancillary Data]
    
    Returns:
        Plots for comparison of total population to age groups
    """
    
    neighPath= ROOT_DIR + "/Shapefiles/Withdrawals/2018_grootams.shp"
    waterPath = ancillary_path + 'corine/waterComb_grootams_CLC_2012_2018.tif'
        
    #Preparing data for disaggregatio, take the processed dataset from CleanSelection, dissolve, and save shp to Shapefiles, and population groups in csv in Statistics
    query = """Select "{0}_municipalities"."gm_code","{0}_municipalities"."gm_naam", ST_Transform("{0}_municipalities".geom,3035) as geom from "{0}_municipalities" ,  "nuts3_grootAms" \
                WHERE ST_Intersects( ST_Transform("{0}_municipalities".geom,3035),  "nuts3_grootAms".geom) \
                AND ST_AREA(ST_Intersection(ST_Transform("{0}_municipalities".geom,3035),"nuts3_grootAms".geom) ) > ST_AREA(ST_Transform("{0}_municipalities".geom,3035))/2""".format(year)
    inter = gpd.read_postgis(query, engine)
    
    munic = ancillary_path + '/adm/municipalities/{0}_{1}_GM.shp'.format(year,city)
    inter.to_file(munic,driver='ESRI Shapefile', crs='EPSG:3035')
    
    src = gpd.read_file(ROOT_DIR + '/Shapefiles/Comb/{0}_{1}.shp'.format(year,city))
    #src = gpd.read_file(path)
    #src = src.loc[src['BU_CODE'].str.contains('BU0363')]
    print(src.head(4))
    cols = ['l1_totalpo',"l2_childre","l3_student","l4_mobile_","l5_not_mob","l6_elderly"]
    for col in cols:
        name='CBS_{0}'.format(col)
        exportPath = ROOT_DIR + "/dataPrep/CBS/{0}_Polyg.png".format(name)
        title ="Population Distribution (persons)\n({0})({1})".format(name, year)
        LegendTitle = "Population Distribution (persons)"
        plot_mapVectorPolygons('popdistributionPolyg', src, exportPath, title, LegendTitle, col, districtPath = munic, neighPath = neighPath, waterPath = waterPath, invertArea = None, addLabels=True)
    
    ndf = src[cols]
    xLegend = 'Population Range'
    directory0 = ROOT_DIR + "/dataPrep/CBS/CBS_BoxPlot.png"
    BoxPlotCBS(directory0, ndf, xLegend)
    directoryNO0 = ROOT_DIR + "/dataPrep/CBS/CBS_BoxPlotNO.png"
    BoxPlotCBS_NO(directoryNO0, ndf, xLegend)
    src['totalAgeGroups'] = src["l2_childre"]+src["l3_student"] + src["l4_mobile_"] + src["l5_not_mob"] + src["l6_elderly"]
    
    a = src[['l1_totalpo', 'totalAgeGroups']]
    directoryNO0 = ROOT_DIR + "/dataPrep/CBS/dif_CBS_BoxPlot.png"
    BoxPlotCBS(directoryNO0, a, xLegend)
    
    src['mae'] = src['l1_totalpo'] - src['totalAgeGroups']
    exportPath = ROOT_DIR + "/dataPrep/CBS/mae_totalPopAgeGroups_Polyg.png"
    title ="Mean Error (persons)\n(Comparison between the total population and the sum of the 5age groups, source:CBS)({1})".format(name, year)
    LegendTitle = "Mean Error (persons)"
    plot_mapVectorPolygons('mae', src , exportPath, title, LegendTitle, 'mae', districtPath = munic, neighPath = neighPath, waterPath = waterPath, invertArea = None, addLabels=True)
    
