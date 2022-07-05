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
                
import subprocess
def calcTotalPop(resultsPath, attr_value, city, python_scripts_folder_path):
    if city =='ams':
        cmds = """python {0}/gdal_calc.py -A "{1}_{2}.tif" -B "{1}_{3}.tif"  \
                    -C "{1}_{4}.tif" -D "{1}_{5}.tif" -E "{1}_{6}.tif" \
                    --A_band=1 --B_band=1 --C_band=1 --D_band=1 --E_band=1 \
                    --outfile="{1}_ag_totalpop.tif" \
                    --calc="A+B+C+D+E" """.format(python_scripts_folder_path, resultsPath, attr_value[0], attr_value[1], attr_value[2], attr_value[3], attr_value[4])
        subprocess.call(cmds, shell=True)

        cmds = """python {0}/gdal_calc.py -A "{1}_{2}.tif" -B "{1}_{3}.tif"  \
                    -C "{1}_{4}.tif" -D "{1}_{5}.tif" -E "{1}_{6}.tif" -F "{1}_{7}.tif" -G "{1}_{8}.tif"\
                    --A_band=1 --B_band=1 --C_band=1 --D_band=1 --E_band=1 --F_band=1 --G_band=1\
                    --outfile="{1}_mg_totalpop.tif" \
                    --calc="A+B+C+D+E+F+G" """.format(python_scripts_folder_path, resultsPath, attr_value[5], attr_value[6], attr_value[7], attr_value[8], attr_value[9], attr_value[10], attr_value[11])
        subprocess.call(cmds, shell=True)

        cmds = """python {0}/gdal_calc.py -A "{1}_ag_totalpop.tif" -B "{1}_mg_totalpop.tif"  \
                    --A_band=1 --B_band=1 \
                    --outfile="{1}_totalpop.tif" \
                    --calc="(A+B)/2" """.format(python_scripts_folder_path, resultsPath)
        subprocess.call(cmds, shell=True)
    
    elif city == 'cph':
        cmds = """python {0}/gdal_calc.py -A "{1}_{2}.tif" -B "{1}_{3}.tif"  \
                    -C "{1}_{4}.tif" -D "{1}_{5}.tif" -E "{1}_{6}.tif" \
                    --A_band=1 --B_band=1 --C_band=1 --D_band=1 --E_band=1 \
                    --outfile="{1}_ag_totalpop.tif" \
                    --calc="A+B+C+D+E" """.format(python_scripts_folder_path, resultsPath, attr_value[0], attr_value[1], attr_value[2], attr_value[3], attr_value[4])
        subprocess.call(cmds, shell=True)
        cmds = """python {0}/gdal_calc.py -A "{1}_{2}.tif" -B "{1}_{3}.tif"  \
                    -C "{1}_{4}.tif" -D "{1}_{5}.tif" -E "{1}_{6}.tif" -F "{1}_{7}.tif" -G "{1}_{8}.tif" -H "{1}_{9}.tif"\
                    --A_band=1 --B_band=1 --C_band=1 --D_band=1 --E_band=1 --F_band=1 --G_band=1 --H_band=1\
                    --outfile="{1}_mg_totalpop.tif" \
                    --calc="A+B+C+D+E+F+G+H" """.format(python_scripts_folder_path, resultsPath, attr_value[5], attr_value[6], attr_value[7], attr_value[8], attr_value[9], attr_value[10], attr_value[11], attr_value[12])
        subprocess.call(cmds, shell=True)

        cmds = """python {0}/gdal_calc.py -A "{1}_ag_totalpop.tif" -B "{1}_mg_totalpop.tif"  \
                    --A_band=1 --B_band=1 \
                    --outfile="{1}_totalpop.tif" \
                    --calc="(A+B)/2" """.format(python_scripts_folder_path, resultsPath)
        subprocess.call(cmds, shell=True)
