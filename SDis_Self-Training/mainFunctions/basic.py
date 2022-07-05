import os
import numpy as np
#Python get unique values from list using numpy.unique

# function to get unique values
def unique(list1):
    x = np.array(list1)
    print(np.unique(x))

## ## ## ## ## ----- CREATE NEW FOLDER  ----- ## ## ## ## ##
def createFolder(path):
    if not os.path.exists(path):
        print("------------------------------ Creating Folder : {} ------------------------------".format(path))
        os.makedirs(path)
    else: 
        print("------------------------------ Folder already exists------------------------------")

import pandas as pd
def csvTOlatex(input, output):
    df = pd.read_csv(input, encoding='cp1252')
    with open(output, 'w', encoding='cp1252') as tf:
        tf.write(df.to_latex())

def excelTOlatex(input, output):
    df = pd.read_excel(input)
    with open(output, 'w') as tf:
        tf.write(df.to_latex()) 

import openpyxl
def jsonTOxlxs(input, output):
    df = gpd.read_file(input)
    df.to_csv(output)
"""
## ## ## ## ## ----- SAVE DATAFRAME TO EXCEL FILE  ----- ## ## ## ## ##
def dfTOxls(dest_path, fileName, frame):
    # Create a Pandas Excel writer using XlsxWriter as the engine
    writer = pd.ExcelWriter(dest_path + "/{}.xlsx".format(fileName),  index = False, header=True)
    # Convert the dataframe to an XlsxWriter Excel object.
    frame.to_excel(writer, sheet_name='Sheet1')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
"""  
import geopandas as gpd
import json
from rasterstats import zonal_stats
# Calculate zonal statistics from tiffs
def zonalStat(src_file, dst_file, polyPath, statistics):  
    # Read Files
    districts = gpd.read_file(polyPath)
    districts = districts.to_crs("EPSG:3035")
    
    zs = zonal_stats(districts, src_file,
                stats='{}'.format(statistics), all_touched = False, percent_cover_selection=None, percent_cover_weighting=False, #0.5-->dubled the population
                percent_cover_scale=None,geojson_out=True)
    
    for row in zs:
        newDict = row['properties']
        for i in newDict.keys():
            if i == '{}'.format(statistics):
                newDict['{}_'.format(statistics)] = newDict.pop(i)
        
    result = {"type": "FeatureCollection", "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:EPSG::3035" } }, "features": zs}
    #dst_file = dstPath + "{0}".format(dstFile) #
    with open(dst_file , 'w') as outfile:
        json.dump(result, outfile)
