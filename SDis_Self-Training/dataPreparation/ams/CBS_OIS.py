import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import glob
from pathlib import Path

from evaluateFunctions import zonalStat, zonalStat1
from osgeoutils import *
from plotting.plotBoxPlot import BoxPlotCBS, BoxPlotCBS_NO
from plotting.plotVectors import plot_mapVectorPolygons
"""[summary]
    WITH a as(
    Select "2018_ams_ois"."Buurtcode", "2018_dataVectorGrid".l3_students as stud,  "2018_dataVectorGrid".geom as geom, "2018_dataVectorGrid".grid_id as id, 
    ST_AREA( ST_Intersection( ST_Transform("2018_ams_ois".geometry,3035),  "2018_dataVectorGrid".geom) ) as area 
    from "2018_ams_ois" ,  "2018_dataVectorGrid" 
    WHERE  "2018_ams_ois"."Buurtcode"= 'A02c' 
    And  ST_Intersects( ST_Transform("2018_ams_ois".geometry,3035),  "2018_dataVectorGrid".geom) 
    AND ST_AREA( ST_Intersection( ST_Transform("2018_ams_ois".geometry,3035),  "2018_dataVectorGrid".geom) ) ) SELECT SUM(stud*area/10000) as pop FROM a


    with a as (
    Select "2018_ams_ois"."Buurtcode", "2018_dataVectorGrid".l3_students as stud,  ST_AREA( ST_Intersection( ST_Transform("2018_ams_ois".geometry,3035),  "2018_dataVectorGrid".geom) ) as area,
    "2018_dataVectorGrid".geom as geom, "2018_dataVectorGrid".grid_id as id  
    from "2018_ams_ois" ,  "2018_dataVectorGrid" 
    WHERE  "2018_ams_ois"."Buurtcode"= 'A01h' 
    And  ST_Intersects( ST_Transform("2018_ams_ois".geometry,3035),  "2018_dataVectorGrid".geom) 
    )  Select SUM(stud*area/10000) as pop from a 

    with a as (
    Select "2018_ams_ois"."Buurtcode", "2018_dataVectorGrid".l3_students as stud,  "2018_dataVectorGrid".geom as geom 
    from "2018_ams_ois" ,  "2018_dataVectorGrid" 
    WHERE  "2018_ams_ois"."Buurtcode"= 'A01h' 
    And  ST_Intersects( ST_Transform("2018_ams_ois".geometry,3035),  "2018_dataVectorGrid".geom) 
    ) Select sum(a.stud) from a

"""

def joinCBS_OIS(year, city, pop_path):
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
    cbs = gpd.read_file(ROOT_DIR + '/Shapefiles/Comb/{0}_{1}.shp'.format(year,city))
    cbs = cbs.loc[cbs['BU_CODE'].str.contains('BU0363')]
    cols = ['BU_CODE','WijkenEnBu','l1_totalpo',"l2_childre","l3_student","l4_mobile_","l5_not_mob","l6_elderly", "l10_wester", "l10_nonwes",  "l10_mar" , "l10_ant",  "l10_sur",  "l10_tur" ]
    frame = cbs[cols]
    for col in frame.columns:
        if col != 'BU_CODE' and col != 'geometry' and col != 'WijkenEnBu':
            frame = frame.rename(columns={col:'cbs_{}'.format(col)})
    frame = frame.set_index('WijkenEnBu')
    print(frame.head(2))
    
    #Step2: OIS
    ois = gpd.read_file(ROOT_DIR + '/Shapefiles/Comb/{0}_ams_ois.shp'.format(year,city))
    ois_cols =["Buurtcode", "Buurtnaam","totalpop", "Children","Students","MobileAdul","NotMobileA","Elderly","l10_sur","l10_ant","l10_mar","l10_tur","l10_wester","l10_nonwes" ]
    ois_frame = ois[ois_cols]
    for col in ois_frame.columns:
        if col != 'Buurtcode' and col != 'geometry' and col != 'Buurtnaam':
            ois_frame = ois_frame.rename(columns={'{}'.format(col):'ois_{}'.format(col)})        
    
    frame = frame.join(ois_frame.set_index('Buurtnaam'), on='WijkenEnBu', lsuffix='_l')
    
    #Step3: OIS aggregated
    tifPaths = glob.glob(pop_path + "/OIS/gridCells/{}_*.tif".format(year))
    ams_path= ROOT_DIR + '/Shapefiles/Withdrawals/{0}_ams.shp'.format(year,city)
    ams_neigh = gpd.read_file(ams_path)
    ams_neigh = ams_neigh[['Buurtcode', 'Buurtnaam','geometry']]
    statistics='sum'
    for i in tifPaths:
        path = Path(i)
        print(path)
        fileName = path.stem 
        print(fileName)
        dst_file = ROOT_DIR + '/dataPrep/OIS/aggregated0/{0}.geojson'.format(fileName)
        print(path, fileName, dst_file)
        zonalStat(path, dst_file, ams_path, statistics)
        ois_gc = gpd.read_file(dst_file)
        ois_gc = ois_gc.rename(columns={'sum_':'ois_gc_{}'.format(fileName)})        
        frame = frame.join(ois_gc.set_index('Buurtnaam'), on='WijkenEnBu', lsuffix='_l') 
 
        dst_file = ROOT_DIR + '/dataPrep/OIS/aggregated0/{0}_1.geojson'.format(fileName)
        zonalStat1(path, dst_file, ams_path, statistics)
        ois_gc = gpd.read_file(dst_file)
        ois_gc = ois_gc.rename(columns={'sum_':'ois_gc1_{}'.format(fileName)})        
        frame = frame.join(ois_gc.set_index('Buurtnaam'), on='WijkenEnBu', lsuffix='_l')
        """
        dst_file = ROOT_DIR + '/dataPrep/OIS/aggregated0/{0}_2.geojson'.format(fileName)
        zonalStat(path, dst_file, ams_path, statistics, all_touched = True, percent_cover_selection=40, percent_cover_weighting=True, #0.5-->dubled the population
                percent_cover_scale=100000)
        ois_gc = gpd.read_file(dst_file)
        ois_gc = ois_gc.rename(columns={'sum_':'ois_gc2_{}'.format(fileName)})        
        frame = frame.join(ois_gc.set_index('Buurtnaam'), on='WijkenEnBu', lsuffix='_l')
        
        dst_file = ROOT_DIR + '/dataPrep/OIS/aggregated0/{0}_3.geojson'.format(fileName)
        zonalStat(path, dst_file, ams_path, statistics, all_touched = True, percent_cover_selection=90, percent_cover_weighting=True, #0.5-->dubled the population
                percent_cover_scale=100000)
        ois_gc = gpd.read_file(dst_file)
        ois_gc = ois_gc.rename(columns={'sum_':'ois_gc3_{}'.format(fileName)})        
        frame = frame.join(ois_gc.set_index('Buurtnaam'), on='WijkenEnBu', lsuffix='_l')
        """
    print("kati allo edo")             
    frame = frame.loc[:,~frame.columns.duplicated()]
    for col in frame.columns:
        if col.endswith('_l'):
            frame = frame.drop(columns=col)
    frame = frame.reset_index()
    print(frame.head(2))
    
    frame = gpd.GeoDataFrame(frame, geometry='geometry')
    frame.to_file(ROOT_DIR + '/dataPrep/CBS_OIS/{0}_amsComparison.geojson'.format(year),driver='GeoJSON', crs='EPSG:3035')
    nframe = frame.drop(columns='geometry')
    nframe.to_csv(ROOT_DIR + '/dataPrep/CBS_OIS/{0}_amsComparison.csv'.format(year),index=False)

def computePopByArea(cur, conn,engine):
    fields= [ "l1_totalpop",  "l2_children",  "l3_students",  "l4_mobile_adults",  "l5_not_mobile_adults",  "l6_elderly",  "tur",  "mar",  "ant",  "sur" ]
    for i in fields:
        cur.execute("""ALTER TABLE "2018_ams_ois" ADD COLUMN agg_{} double precision;""".format(i))
        conn.commit()
        #cur.execute("""ALTER TABLE "2018_ams_ois" ALTER COLUMN calc_totalpop SET DEFAULT 0.0;""")
        ids = []
        cur.execute("""SELECT "Buurtcode" FROM "2018_ams_ois";""")
        chunk_id = cur.fetchall()
        
        # saving ids to list
        for id in chunk_id:
            ids.append(id[0])
        
        # iterate through chunks
        for chunk in ids:    
            cur.execute( """ WITH b as (
                WITH a as(
                    Select "2018_ams_ois"."Buurtcode" as code, "2018_dataVectorGrid".{1} as stud,  "2018_dataVectorGrid".geom as geom, "2018_dataVectorGrid".grid_id as id, 
                    ST_AREA( ST_Intersection( ST_Transform("2018_ams_ois".geom,3035),  "2018_dataVectorGrid".geom) ) as area 
                    from "2018_ams_ois",  "2018_dataVectorGrid" 
                    WHERE  "2018_ams_ois"."Buurtcode"= '{0}' 
                    AND  ST_Intersects( ST_Transform("2018_ams_ois".geom,3035),  "2018_dataVectorGrid".geom) )
                Select SUM(stud*area/10000) as pop from a)
            UPDATE "2018_ams_ois" SET agg_{1} = pop from b where "2018_ams_ois"."Buurtcode"='{0}';
            """.format(chunk, i))
            conn.commit()
    query = """Select * FROM  "2018_ams_ois" """
    inter = gpd.read_postgis(query, engine)
    inter = inter.rename(columns={'geom':'geometry'})
    nf = gpd.GeoDataFrame(inter, geometry= 'geometry')
    nf.to_file(ROOT_DIR + '/dataPrep/CBS_OIS/OIS_OISaggrPostgis1.geojson',driver='GeoJSON', crs='EPSG:3035')
        
            
    
def getStatisticsCBS_OIS(year, ROOT_DIR, city, ancillary_path ):
    """[Compares the OIS and the CBS Dataset]

    Args:
        year (int): [The reference year to work upon]
        ROOT_DIR ([str]): [parent directory]
        city ([str]): [City to work on]
        ancillary_path ([str]): [Directory to Ancillary Data]
    
    Returns:
        CSV in CBS_OIS with differences and 
        Plots for pop distribution at neighborhood level, difference between the 2datasets and deviation
    """
    df = gpd.read_file(ROOT_DIR + '/dataPrep/CBS_OIS/{0}_amsComparison.geojson'.format(year))
    
    df['dif_l1_totalpop'] = df['ois_totalpop']- df['cbs_l1_totalpo'] 
    df['dif_l2_children'] = df['ois_Children']- df['cbs_l2_childre']  
    df['dif_l3_students'] = df['ois_Students']- df['cbs_l3_student'] 
    df['dif_l4_mobile_adults'] = df['ois_MobileAdul'] - df['cbs_l4_mobile_'] 
    df['dif_l5_not_mobile_adults'] = df['ois_NotMobileA']- df['cbs_l5_not_mob'] 
    df['dif_l6_elderly'] = df['cbs_l6_elderly'] - df['ois_Elderly']
    
    mae1 = df['dif_l1_totalpop'].mean()
    mae2 = df['dif_l2_children'].mean()
    mae3 = df['dif_l3_students'].mean()
    mae4 = df['dif_l4_mobile_adults'].mean()
    mae5 = df['dif_l5_not_mobile_adults'].mean()
    mae6 = df['dif_l6_elderly'].mean()
    
    cbsSum = df['cbs_l1_totalpo'].sum()
    OISSum = df['ois_totalpop'].sum()
    df['pdif_l1_totalpop'] = (df['cbs_l1_totalpo'] / df['ois_totalpop']) *100
    df['pdif_l2_children'] = (df['cbs_l2_childre'] / df['ois_Children']) *100
    df['pdif_l3_students'] = (df['cbs_l3_student'] / df['ois_Students']) *100
    df['pdif_l4_mobile_adults'] = (df['cbs_l4_mobile_'] / df['ois_MobileAdul']) *100
    df['pdif_l5_not_mobile_adults'] = (df['cbs_l5_not_mob']/ df['ois_NotMobileA']) *100
    df['pdif_l6_elderly'] = (df['cbs_l6_elderly'] / df['ois_Elderly']) *100
    print(df.head(2))
    
    colDif = []
    for col in df.columns:
        if col.startswith('dif_'):
            colDif.append(col)
    sa = df[colDif]

    #Population 
    xLegend = 'Population Difference'
    directory0 = ROOT_DIR + "/dataPrep/CBS_OIS/dif_CBS_OIS_BoxPlot.png"
    BoxPlotCBS(directory0, sa, xLegend)
    directoryNO0 = ROOT_DIR + "/dataPrep/CBS_OIS/dif_CBS_OIS_BoxPlotNO.png"
    BoxPlotCBS_NO(directoryNO0, sa, xLegend)

    comp = df[['cbs_l1_totalpo','cbs_l2_childre', 'cbs_l3_student', 'cbs_l4_mobile_', 'cbs_l5_not_mob',  'cbs_l6_elderly', 'ois_totalpop',
              'ois_Children','ois_Students','ois_MobileAdul','ois_NotMobileA','ois_Elderly']]
    
    #Population 
    xLegend = 'Population Range'
    directory0 = ROOT_DIR + "/dataPrep/CBS_OIS/CBS_OIS_BoxPlot.png"
    BoxPlotCBS(directory0, comp, xLegend)
    directoryNO0 = ROOT_DIR + "/dataPrep/CBS_OIS/CBS_OIS_BoxPlotNO.png"
    BoxPlotCBS_NO(directoryNO0, comp, xLegend)
    
    print("-----WRITING CSV-----")
    # Calculate the mean difference and the quotient of total grid cells and write it in csv
    filenamemetrics2e = ROOT_DIR + '/dataPrep/CBS_OIS/CBS_OIS.csv'.format(year,city)
    
    if os.path.exists(filenamemetrics2e):
        with open(filenamemetrics2e, 'a') as myfile:
            myfile.write(str(mae1) + ';' + str(mae2)+ ';' + str(mae3) + ';' + str(mae4) + ';' + str(mae5) + ';' + str(mae6) + ';' + str(cbsSum) + ';' + str(OISSum)+ '\n')       
    else:
        with open(filenamemetrics2e, 'w+') as myfile:
            myfile.write('Statistics on Neighborhood data (source:CBS)\n')
            myfile.write('MEPopulation;MEChildren;MEStudents;MEMobileAdults;MENotMobileAdults;MEElderly;CBStotal;OIStotal\n')
            myfile.write( str(mae1) + ';' + str(mae2)+ ';' + str(mae3) + ';' + str(mae4) + ';' + str(mae5) + ';' + str(mae6) + ';' + str(cbsSum) + ';' + str(OISSum) + '\n')
    
    polyPath = ROOT_DIR + "/Shapefiles/Withdrawals/{0}_{1}.shp".format(year,city)
    districtPath = ancillary_path + '/adm/ams_districts.geojson'
    waterPath = ancillary_path + 'corine/waterComb_grootams_CLC_2012_2018.tif'
    
    for col in df.columns:
        if col.startswith('dif_'):
            exportPath = ROOT_DIR + "/dataPrep/CBS_OIS/mae_CBS_OIS_{}_Polyg.png".format(col)
            title ="Comparison between the CBS and the OIS dataset for {0} (CBSi-OISi) ({1})".format(col, year)
            LegendTitle = "Difference (persons)"
            plot_mapVectorPolygons(city,'mae', df , exportPath, title, LegendTitle, col, districtPath = districtPath, neighPath = polyPath, waterPath = waterPath, invertArea = None, addLabels=True)       
        
        if col.startswith('ois_') or col.startswith('cbs_') :
            if col!='ois_Buurtcode' and col!='ois_Wijkcode' and col!='ois_gebiednaam' and col!='ois_gebiedcode' and col!='ois_Stadsdeelc' \
                and col!='ois_Oppervlakt' and col!='ois_BuurtID' and col!='ois_sdnaam' and col!='ois_gebiedcode':
                exportPath = ROOT_DIR + "/dataPrep/CBS_OIS/{0}_Polyg.png".format(col)
                title ="Population Distribution (persons)\n({0})({1})".format(col, year)
                LegendTitle = "Population Distribution (persons)"
                plot_mapVectorPolygons(city,'popdistributionPolyg', df, exportPath, title, LegendTitle, '{}'.format(col), districtPath = districtPath, neighPath = polyPath, waterPath = waterPath, invertArea = None, addLabels=True)
        
        if col.startswith('pdif_'):
            exportPath = ROOT_DIR + "/dataPrep/CBS_OIS/PrE00_CBS_OIS_{}_Polyg.png".format(col)
            title ="Comparison between the CBS and the OIS dataset for {0} ((CBS*100/OIS) ({1})".format(col, year)
            LegendTitle = "Proportional Error (persons)"
            plot_mapVectorPolygons(city,'div', df , exportPath, title, LegendTitle, col, districtPath = districtPath, neighPath = polyPath, waterPath = waterPath, invertArea = None, addLabels=True) 

