import os
import geopandas as gpd
import pandas as pd
from mainFunctions import shptoraster, dfTOxls

def non_match_elements(list_a, list_b):
    non_match = []
    for i in list_a:
        if i not in list_b:
            non_match.append(i)
    return non_match

def restructureDataAG(ROOT_DIR):
    df = pd.read_excel(ROOT_DIR + '/dataPreparation/cph/rawMunicipality_DST/FOLK1C_age_2019Q1.xlsx')
    df = df.fillna(method='ffill')
    ndf = df.set_index('Municipality') 
    
    mdf = ndf.pivot( columns='Age groups' )
    print(mdf.head)
    dfTOxls(ROOT_DIR + '/dataPreparation/cph/rawMunicipality_DST/', 'FOLK1C_age_2019Q1_re', mdf)
    
def restructureDataMB(ROOT_DIR):
    df = pd.read_excel(ROOT_DIR + '/dataPreparation/cph/rawMunicipality_DST/FOLK1C_coo_2019Q1.xlsx')
    df = df.fillna(method='ffill')
    ndf = df.set_index('Municipality') 
    
    mdf = ndf.pivot( columns='Countries' )
    print(mdf.head)
    #dfTOxls(ROOT_DIR + '/dataPreparation/cph/rawMunicipality_DST/', 'FOLK1C_coo_2019Q1_re', mdf)
    ndf = pd.read_excel(ROOT_DIR + '/dataPreparation/cph/rawMunicipality_DST/FOLK1C_coo_2019Q1_re.xlsx')

def combineMB_AG(ROOT_DIR, city, year):
    # SUM BY MIGRATION BACKGROUND 
    ndf = pd.read_excel(ROOT_DIR + '/dataPreparation/cph/rawMunicipality_DST/FOLK1C_coo_2019Q1_re.xlsx')
    codes = pd.read_excel('C:/Users/NM12LQ/OneDrive - Aalborg Universitet/FUME/wp6/UNSD — Methodology.xlsx')
    
    a = codes['DK'].to_list()
    b = ndf.columns.to_list()
    #print(a,b)
    
    
    ldf = ndf.copy()
    regions = codes['DK'].unique().tolist() #region_abbr
    print(regions)
    usedCoo= []
    for key in regions:
        print('key', key)
        keyFrame = codes.loc[codes.DK =='{}'.format(key)]
        select = keyFrame['Country or Area'].tolist()
        
        usedCoo.extend(select) #abbr
        print('selection:', select)
        print(ldf.head(2))
        #ldf['{}'.format(key)] = ldf.loc[:, select].sum(axis=1)
        ldf['{}'.format(key)] = ldf[ldf.columns.intersection(select)].sum(axis=1)
        ldf['{}'.format(key)].astype(int)
    print('----')
    print(non_match_elements( b, usedCoo))
    frame = ldf.set_index('Municipality')
    frame = ldf.reset_index()
    frame = frame.rename(columns={'Total':'totalpop'}) #'no':'notEU',
    
    # SUM BY AGE GROUPS    
    ag = pd.read_excel(ROOT_DIR + '/dataPreparation/cph/rawMunicipality_DST/FOLK1C_age_2019Q1_re.xlsx')
    ag['children'] = ag['0-4 years'] + ag['5-9 years'] + ag['10-14 years'] + ag['15-19 years'] 
    ag['younadults'] = ag['20-24 years'] + ag['25-29 years'] 
    ag['mobadults'] = ag['30-34 years'] + ag['35-39 years'] + ag['40-44 years'] 
    ag['nmobadults'] = ag['45-49 years'] + ag['50-54 years'] + ag['55-59 years'] + ag['60-64 years']
    ag['elderly'] = ag['65-69 years'] + ag['70-74 years'] + ag['75-79 years'] + ag['80-84 years'] +ag['85-89 years'] + ag['90-94 years'] + ag['95-99 years'] + ag['100 years and over']
    ag = ag.rename(columns={'Total':'totalpop_ag'})
    
    nframe = frame.set_index('Municipality').join(ag.set_index('Municipality'))
    selectedColumns= ['id','Municipality', 'totalpop', 'totalpop_ag', 'children', 'younadults', 'mobadults', 'nmobadults', 'elderly', 
                        'DNK', 'W_EU_EFTA', 'E_EU', 'Europe_nonEU', 'MENAP',  'TUR', 'OtherWestern', 'OtherNonWestern'                        
                        ]
    nframe = nframe[nframe.columns.intersection(selectedColumns)]
    print(nframe.head(10))
    nframe.to_csv(ROOT_DIR + "/Statistics/{1}/{0}_{1}.csv".format(year,city), encoding="utf-8")
     
def restructureData_old(ROOT_DIR, city, year, ancillary_path, gdal_rasterize_path):
    df = gpd.read_file("K:/FUME/demo_popnet-Commit01/data_prep/cph_ProjectData/PopData/2018/temp_shp/2018_dataVectorGrid.geojson")
    df['children']= df[ "L2_P00_19"] * df["L1_SUM_POPULATION" ]
    df['students']= df["L3_P20_29" ] * df["L1_SUM_POPULATION"]
    df['mobadults']= df["L4_P30_44"] * df["L1_SUM_POPULATION"]
    df['nmobadults']= df["L5_P45_64"] * df["L1_SUM_POPULATION"]
    df['elderly']= df["L6_P65"] * df["L1_SUM_POPULATION"]
    
    #df.columns= df.columns.str.lower()
    df.columns= [col.replace('L10_', '') for col in df.columns]

    #codes = pd.read_excel('K:/FUME/demo_popnet-Commit01/data_prep/cph_ProjectData/EXCEL/unsd_dk.xlsx')
    codes = pd.read_excel('C:/Users/NM12LQ/OneDrive - Aalborg Universitet/FUME/wp6/UNSD — Methodology.xlsx')
    
    a = codes['abbr'].to_list()
    b = df.columns.to_list()
    #print(a,b)
    print(non_match_elements(a, b))
    #Rename columns
    #ldf = ndfC.rename(columns=codes.set_index('country')['abbr'])
    ldf =df.copy()
    regions = codes['DK'].unique().tolist() #region_abbr
    print(regions)
    for key in regions:
        keyFrame = codes.loc[codes.region =='{}'.format(key)]
        select = keyFrame['ISO-alpha3 Code'].tolist() #abbr
        print(select)
        print(ldf.head(2))
        #ldf['{}'.format(key)] = ldf.loc[:, select].sum(axis=1)
        ldf['{}'.format(key)] = ldf[ldf.columns.intersection(select)].sum(axis=1)
        ldf['{}'.format(key)].astype(int)
        print(ldf['{}'.format(key)].sum())
    """
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
    print(ldf.head(2))"""

    frame = ldf.set_index('Municipality').join(ndf.set_index('Municipality'))
    frame = ldf.reset_index()
    frame = frame.rename(columns={ 'L1_SUM_POPULATION':'totalpop'}) #'no':'notEU',
    print(frame.head(3))
    #frame['EU'] = frame['totalpop'] - (frame['yes'] + frame['efta'] + frame['uk'])
    
    
    selectedColumns= ['gridid','geometry','totalpop', 'children', 'students', 'mobadults', 'nmobadults', 'elderly', 
                        'DNK', 'W_EU_EFTA', 'E_EU', 'Europe_nonEU', 'MENAP',  'TUR', 'OtherWestern', 'OtherNonWestern'
                        #'AuNZ', 'CentAsia', 'EastAsia', 'EastEur', 'LAC', 'Melanesia', 'Micronesia', 'NorthAfr', 
                        #'NorthAm', 'NorthEur', 'OTH', 'Polynesia', 'SEastAsia', 'SouthAsia', 'SouthEur', 'STA', 'SubSahAfr', 
                        #'WestAsia', 'WestEur',
                        #'EU', 'notEU'
                        ]
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


        