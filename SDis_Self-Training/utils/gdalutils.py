import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
import os
from osgeo import gdal
from utils import osgu
#from osgeoutils import readRaster, addAttr2Shapefile

def progress_cb(complete, message, cb_data):
    '''Emit progress report in numbers for 10% intervals and dots for 3%'''
    if int(complete*100) % 10 == 0:
        print(f'{complete*100:.0f}', end='', flush=True)
    elif int(complete*100) % 3 == 0:
        print(f'{cb_data}', end='', flush=True)
        
def reprojectionGDAL(inputfile, temp_outputfile,resamplingAlgorithm):   
    wrp_options1 = gdal.WarpOptions(resampleAlg=resamplingAlgorithm, targetAlignedPixels= True )
    gdal.Warp(temp_outputfile, inputfile, dstSRS="EPSG:3035", callback=progress_cb, callback_data='.')

def resampleGDAL(  temp_outputfile, temp_outputfileVRT,  xres, yres, resamplingAlgorithm):
    vrt_options1 = gdal.BuildVRTOptions(xRes=xres, yRes=yres, resampleAlg=resamplingAlgorithm, targetAlignedPixels= True )
    print(temp_outputfileVRT, temp_outputfile)
    gdal.BuildVRT(temp_outputfileVRT, temp_outputfile, options= vrt_options1, callback=progress_cb, callback_data='.')
    
def vrt2tifGDAL(raster_file, temp_outputfileVRT, outputfile):
    data = gdal.Open(raster_file)
    geoTransform = data.GetGeoTransform()
    minx = geoTransform[0] 
    maxy = geoTransform[3] 
    maxx = minx + geoTransform[1] * data.RasterXSize 
    miny = maxy + geoTransform[5] * data.RasterYSize
    data = None
    bbox = (minx,maxy,maxx,miny)
    #Clip 
    gdal.Translate(outputfile, temp_outputfileVRT, projWin=bbox, callback=progress_cb, callback_data='.')

def shp2tifGDAL(template,city, fshape,  outputfile, xres, yres, attribute, burnValues):
    print('| Converting shapefile to raster:', fshape, '-', attribute)

    if attribute != 'ID':
        osgu.addAttr2Shapefile(fshape, attr='ID')

    print('|| Converting...')
    data = gdal.Open(template)
    geoTransform = data.GetGeoTransform()
    minx = geoTransform[0] 
    maxy = geoTransform[3] 
    maxx = minx + geoTransform[1] * data.RasterXSize 
    miny = maxy + geoTransform[5] * data.RasterYSize
    data = None
    bbox = (minx,miny,maxx,maxy)
    #tempfile = ROOT_DIR + '/TempRaster/{0}/'.format(city) + 'tempfileo2r_' + attribute + '_' + str(os.getpid()) + '.tif'
    rst_options = gdal.RasterizeOptions(outputBounds=bbox, attribute = attribute, outputSRS="EPSG:3035", xRes=xres, yRes=yres, burnValues = burnValues , targetAlignedPixels= True )
    gdal.Rasterize( outputfile, fshape, options= rst_options)
    
    dataset, rastergeo = osgu.readRaster( outputfile)
    return dataset, rastergeo
  
import rasterio  
def writeRaster(raster_file, outfile, out_array):
    with rasterio.open(raster_file) as src:
        new_dataset = rasterio.open(
            outfile,
            'w',
            driver='GTiff',
            height=src.shape[0],
            width=src.shape[1],
            count=1,
            dtype=out_array.dtype,
            crs=src.crs,
            transform= src.transform
            )
    new_dataset.write(out_array, 1)
    new_dataset.close()

import fiona
import rasterio
import rasterio.mask    
def maskRaster(shpPath, input, output):
    with fiona.open(shpPath, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile] 
    
    with rasterio.open(input) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes)
        out_meta = src.meta
    
    out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

    with rasterio.open(output, "w", **out_meta) as dest:
        dest.write(out_image)
                
