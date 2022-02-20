import geopandas as gpd
import random, os
import numpy as np
from config.definitions import ROOT_DIR

def computeNeighbors(fshape, polyg = 'NUTSIII', verbose=False):
    adjpolygons = {}

    gdf = gpd.read_file(fshape)

    if(verbose): print('Shapefile with', gdf.shape[0], 'polygons')

    for index, pol in gdf.iterrows():
        pname = pol[polyg]
        pid = pol.ID

        neigdf = gdf[~gdf.geometry.disjoint(pol.geometry)]
        for index, neig in neigdf.iterrows():
            if neig[polyg] != pname:
                neigid = neig.ID
                if pid in adjpolygons:
                    adjpolygons[pid].append(neigid)
                else:
                    adjpolygons[pid] = [neigid]

    return adjpolygons

def stratifiedSamples(qvalues, numpolygons):
    import random
    pols2merge = []
    maxids = numpolygons
    while maxids > 0:
        for i in qvalues.values():
            if (maxids > 0) and (len(i) > 0):
                pols2merge.append(random.choice(i))
                maxids = maxids - 1
            else:
                continue

    return pols2merge

#strat=quartilles
def createAdjPairs(adjpolygons, pairperc, initadjpairs=[], strat=None, verbose=False):
    nummun = len(adjpolygons)
    numfinalmun = round(pairperc*nummun)
    adjpairs = initadjpairs.copy()
    cpids = [x for t in initadjpairs for x in t]
    stratsamples = stratifiedSamples(strat, nummun)

    i = 1
    while i <= numfinalmun:
        if strat:
            cp1 = int(stratsamples[i - 1])
            if cp1 in cpids:
                stratsamples.remove(cp1)
                cp1 = int(stratsamples[i - 1])
        else:
            cp1 = random.randint(1, nummun)

        if (cp1 not in cpids) and (cp1 in adjpolygons):
            neigcp1 = adjpolygons[cp1]
            for j in range(len(neigcp1)):
                cp2 = neigcp1[j]
                if cp2 not in cpids:
                    cpids.append(cp1)
                    cpids.append(cp2)
                    adjpairs.append([cp1, cp2])
                    i = i+1
                    break
                else:
                    if j == (len(neigcp1) - 1):
                        i = i+1
                        break

    if(verbose): print(pairperc)
    return adjpairs

def dissolvePairs(fshape, adjpairs):
    newpairs = {}

    i = 1
    gdf = gpd.read_file(fshape)
    for pair in adjpairs:
        gdf.loc[gdf['ID'] == pair[0], 'ID'] = int(str(99999) + str(i))
        gdf.loc[gdf['ID'] == pair[1], 'ID'] = int(str(99999) + str(i))
        newpairs[int(str(99999) + str(i))] = [pair[0], pair[1]]
        i = i + 1

    dissolveddf = gdf.dissolve(by='ID')
    dissolveddf['ID'] = dissolveddf.index

    prj = [l.strip() for l in open(fshape.replace('.shp', '.prj'), 'r')][0]
    fshape_dissolved = ROOT_DIR + '/Temp/fshape_dissolved_' + str(os.getpid()) + '.shp'
    dissolveddf.to_file(driver='ESRI Shapefile', filename=fshape_dissolved, crs_wkt=prj)
    return fshape_dissolved, newpairs

def polygonsQuartilles(idpolvalues, numpolygons):
    quart1, quart2, quart3, quart4 = np.percentile(list(idpolvalues.values()), [25, 50, 75, 100])
    idsq1 = [key for (key, value) in idpolvalues.items() if value >= 0 and value < quart1]
    idsq2 = [key for (key, value) in idpolvalues.items() if value >= quart1 and value < quart2]
    idsq3 = [key for (key, value) in idpolvalues.items() if value >= quart2 and value < quart3]
    idsq4 = [key for (key, value) in idpolvalues.items() if value >= quart3 and value < quart4]
    qvalues = {1: idsq1, 2: idsq2, 3: idsq3, 4: idsq4}

    return qvalues

def computeAreas(fshape):
    polareas = {}
    gdf = gpd.read_file(fshape)
    for index, row in gdf.iterrows():
        poly_id = row['ID']
        poly_area = row['geometry'].area * 10**6 # Areas in m2
        polareas[poly_id] = poly_area
    return polareas

