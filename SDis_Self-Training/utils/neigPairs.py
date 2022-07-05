import geopandas as gpd
import random, os

def createNeigSF(fshape, polyg):
    gdf = gpd.read_file(fshape, encoding='latin-1')

    ids = list(range(1, gdf['geometry'].count() + 1))
    gdf['ID'] = ids

    adjpolygons = {}

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

    cpids = []
    adjpairs = []
    for pol1 in adjpolygons:
        if pol1 not in cpids:
            candidates = set(adjpolygons[pol1]) - set(cpids)
            if len(candidates) > 0:
                pol2 = list(candidates)[random.randint(0, len(candidates)-1)]
                cpids.append(pol1)
                cpids.append(pol2)
                adjpairs.append([pol1, pol2])

    newgdf = gdf.copy()
    newpairs = {}
    i = 1
    for pair in adjpairs:
        newgdf.loc[newgdf['ID'] == pair[0], 'ID'] = int(str(99999) + str(i))
        newgdf.loc[newgdf['ID'] == pair[1], 'ID'] = int(str(99999) + str(i))
        newpairs[int(str(99999) + str(i))] = [pair[0], pair[1]]
        i = i + 1

    dissolveddf = newgdf.dissolve(by='ID')
    dissolveddf['ID'] = dissolveddf.index

    # prj = [l.strip() for l in open(fshape.replace('.shp', '.prj'), 'r')][0]
    # fshape_dissolved = 'Temp/fshape_dissolved_proc-' + str(os.getpid()) + '.shp'
    # dissolveddf.to_file(driver='ESRI Shapefile', filename=fshape_dissolved, crs_wkt=prj)

    return adjpairs, newpairs

