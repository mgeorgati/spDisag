import os
import sys
import numpy as np
import geopandas as gpd
import pandas as pd
import cbsodata

cur_path = os.path.dirname(os.path.abspath(__file__))

def non_match_elements(list_a, list_b):
    non_match = []
    for i in list_a:
        if i not in list_b:
            non_match.append(i)
    return non_match


def restructureData(ROOT_DIR, city, year):
    df = pd.read_excel(cur_path + "/rawMunicipality_DST/FOLK1C_age_2019Q1.xlsx")
    df['Municipality'] = df['Municipality'].fillna(method='ffill')
    ndf = df.groupby([df['Municipality'], 'Age groups'])['2019Q1'].first().unstack()
    #print(ndf.columns.to_list())
    ndf['children'] = ndf['0-4 years'] + ndf['5-9 years'] + ndf['10-14 years'] + ndf['15-19 years']
    ndf['students'] =  ndf['20-24 years'] +  ndf['25-29 years']
    ndf['mobadults'] =  ndf['30-34 years'] + ndf['35-39 years'] + ndf['40-44 years'] 
    ndf['nmobadults'] = ndf['45-49 years'] + ndf['50-54 years'] + ndf['55-59 years'] + ndf['60-64 years']
    ndf['elderly'] = ndf['100 years and over'] + ndf['95-99 years'] + ndf['90-94 years'] + ndf['85-89 years'] + ndf['80-84 years'] + ndf['75-79 years'] + ndf['70-74 years'] + ndf['65-69 years'] 
    ndf = ndf.rename(columns={'Total':'totalpop'}).reset_index()

    dfC = pd.read_excel(cur_path + "/rawMunicipality_DST/FOLK1C_coo_2019Q1.xlsx")
    dfC['Municipality'] = dfC['Municipality'].fillna(method='ffill')
    ndfC = dfC.groupby([dfC['Municipality'], 'Countries'])['2019Q1'].first().unstack()
    ndfC = ndfC.reset_index()
    
    codes = pd.read_excel('C:/FUME/PopNetV2/data_prep/ams_ProjectData/PopData/EXCEL/unsd_dk.xlsx')
    
    a = codes['country'].to_list()
    b = ndfC.columns.to_list()
    #print(a,b)
    #print(non_match_elements(a, b))
    #Rename columns
    ldf = ndfC.rename(columns=codes.set_index('country')['abbr'])
    print(ldf.head(5))
   
    regions = codes['region_abbr'].unique().tolist()
    print(codes.head(3))
    print(regions)
    for key in regions:
        print(key)
        keyFrame = codes.loc[codes.region_abbr =='{}'.format(key)]
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
    print(ndf.head(2))
    
    frame = ldf.set_index('Municipality').join(ndf.set_index('Municipality'))
    
    frame = frame.reset_index()
    print(frame.head(3))
    frame['EU'] = frame['totalpop'] - (frame['yes'] + frame['efta'] + frame['uk'])
    frame['Municipality'] = frame['Municipality'].replace('ø','oe', regex=True)
    frame['Municipality'] = frame['Municipality'].replace('æ','ae', regex=True)
    frame['Municipality'] = frame['Municipality'].replace('å','aa', regex=True)
    
    frame = frame.rename(columns={'no':'notEU', 'Municipality':'KOMNAME'})
    selectedColumns= ['KOMNAME','totalpop', 'children', 'students', 'mobadults', 'nmobadults', 'elderly', 
                      'DNK', 'AuNZ', 'CentAsia', 'EastAsia', 'EastEur', 'LAC', 'Melanesia', 'Micronesia', 'NorthAfr', 
                      'NorthAm', 'NorthEur', 'OTH', 'Polynesia', 'SEastAsia', 'SouthAsia', 'SouthEur', 'STA', 'SubSahAfr', 'WestAsia', 'WestEur', 
                      'EU', 'notEU']
    frame = frame[frame.columns.intersection(selectedColumns)]
    print(frame.head(10))
    frame.to_csv(ROOT_DIR + "/Statistics/{1}/{0}_{1}.csv".format(year,city), encoding="utf-8")
    
     