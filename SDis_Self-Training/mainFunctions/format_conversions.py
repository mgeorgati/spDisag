from shapely.geometry import Polygon, MultiPolygon, shape, Point
import geopandas as gp
def convert_3D_2D(geometry):
    '''
    Takes a GeoSeries of 3D Multi/Polygons (has_z) and returns a list of 2D Multi/Polygons
    '''
    new_geo = []
    for p in geometry:
        if p.has_z:
            if p.geom_type == 'Polygon':
                lines = [xy[:2] for xy in list(p.exterior.coords)]
                new_p = Polygon(lines)
                new_geo.append(new_p)
            elif p.geom_type == 'MultiPolygon':
                new_multi_p = []
                for ap in p:
                    lines = [xy[:2] for xy in list(ap.exterior.coords)]
                    new_p = Polygon(lines)
                    new_multi_p.append(new_p)
                new_geo.append(MultiPolygon(new_multi_p))
    return new_geo

def checkGeom(df):
    '''
    Takes a GeoDataFrame of 2D/3D Multi/Polygons and returns a GeoDataFrame of 2D Polygons
    '''
    gdf = df.copy()
    for index,i in gdf.iterrows():
        geom = i['geometry']
        if geom.geom_type != 'Polygon':
            if geom.geom_type == 'POLYGON Z':
                gdf.geometry = convert_3D_2D(gdf.geometry)
            elif geom.geom_type == 'MultiPolygon':
                gdf = gdf.explode()
    gdf = gdf.reset_index(drop=True)
    return gdf

import subprocess
from osgeo import gdal
def shptoraster(raster_file, src_file, gdal_rasterize_path, dst_file, column_name, xres=100, yres=100):
    '''
    Takes the path of GeoDataframe and converts it to raster
        raster_file         : str
            path to base raster, from which the extent of the new raster is calculated 
        src_file            : str
            path to source file (SHP,GEOJSON, GPKG) 
        gdal_rasterize_path : str
            path to execute gdal_rasterize.exe
        dst_file            : str
            path and name of the destination file
        column_name         : str
            Field to use for rasterizing
    '''
    data = gdal.Open(raster_file)
    geoTransform = data.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * data.RasterXSize
    miny = maxy + geoTransform[5] * data.RasterYSize
    data = None    
    cmd = '{0}/gdal_rasterize.exe -a "{9}" -te {1} {2} {3} {4} -tr {5} {6} "{7}" "{8}"'\
                .format(gdal_rasterize_path, minx, miny, maxx, maxy, xres, yres, src_file, dst_file, column_name)
                
    print(cmd)
    subprocess.call(cmd, shell=True)

import geopandas as gpd    
def dbTOraster(city, gdal_rasterize_path, engine, raster_file, temp_shp_path, temp_tif_path, column_name, layerName):
    # Create SQL Query
    sql = """SELECT id, "{0}", geometry FROM {1}_cover_analysis""".format(column_name, city)
    # Read the data with Geopandas
    gdf = gpd.GeoDataFrame.from_postgis(sql, engine, geom_col='geometry' )

    # exporting water cover from postgres
    print("Exporting {0} from postgres".format(column_name))
    src_file = temp_shp_path + "/{0}Grid.geojson".format(layerName)
    gdf.to_file(src_file,  driver="GeoJSON")   
    dst_file = temp_tif_path + "/{0}.tif".format(layerName)
    
    shptoraster(raster_file, src_file, gdal_rasterize_path, dst_file, column_name, xres=100, yres=100)
    

