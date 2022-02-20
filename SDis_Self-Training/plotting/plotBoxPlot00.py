import sys, os, seaborn as sns, rasterio, pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.definitions import ROOT_DIR, ancillary_path, city,year
from plotting.plotBoxPlot import BoxPlot, BoxPlotNoOutliers
from plotting.createScatterPlot import plotMatrix

attr_value = ['students','mobadults', 'nmobadults', 'elderly', 'sur', 'ant', 'mar','tur', 'nonwestern','western', 'autoch']


path="C:/FUME/DasymetricMapping/SDis_Self-Training/Evaluation/ams/matrixPE_{}.tif".format(attr_value)
title = "PE: Mobile Adults"
 

def plotBoxPlot(attr_value):
    evalFilesMAEbp = [ROOT_DIR + "/Evaluation/{0}/Pycno/mae_{0}_{2}_{0}_{1}_pycno.tif".format(city,attr_value,year),
                ROOT_DIR + "/Evaluation/{0}/Dasy/mae_{0}_{2}_{0}_{1}_dasyWIESMN.tif".format(city,attr_value,year),
                ROOT_DIR + "/Evaluation/{0}/aprf/dissever00/mae_{0}_dissever00WIESMN_2018_ams_Dasy_aprf_p[1]_12AIL12_1IL_it10_{1}.tif".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/aprf/dissever01/mae_{0}_dissever01WIESMN_100_2018_ams_DasyA_aprf_p[1]_12AIL12_13IL_it10_{1}.tif".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/apcatbr/mae_{0}_dissever01WIESMN_100_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{1}.tif".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/apcatbr/mae_{0}_dissever01WIESMN_250_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{1}.tif".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/apcatbr/mae_{0}_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{1}.tif".format(city,attr_value)]
    evalFilesPEbp = [ROOT_DIR + "/Evaluation/{0}/Pycno/div_{0}_{2}_{0}_{1}_pycno.tif".format(city,attr_value,year),
                ROOT_DIR + "/Evaluation/{0}/Dasy/div_{0}_{2}_{0}_{1}_dasyWIESMN.tif".format(city,attr_value,year),
                ROOT_DIR + "/Evaluation/{0}/aprf/dissever00/div_{0}_dissever00WIESMN_2018_ams_Dasy_aprf_p[1]_12AIL12_1IL_it10_{1}.tif".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/aprf/dissever01/div_{0}_dissever01WIESMN_100_2018_ams_DasyA_aprf_p[1]_12AIL12_13IL_it10_{1}.tif".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/apcatbr/div_{0}_dissever01WIESMN_100_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{1}.tif".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/apcatbr/div_{0}_dissever01WIESMN_250_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{1}.tif".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/apcatbr/div_{0}_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{1}.tif".format(city,attr_value)]
    directory0 ="C:/FUME/DasymetricMapping/SDis_Self-Training/Evaluation/ams/BoxPlots/BP_{}.png".format(attr_value)
    all_arr = []
    for i in evalFilesPEbp:
        src= rasterio.open(i)
        pop = src.read(1)
        pop[(np.where(pop <= -9999))] = 0     
        mean = np.mean(pop)
        standard_deviation = np.std(pop)
        distance_from_mean = abs(pop - mean)
        max_deviations = 2
        not_outlier = distance_from_mean < max_deviations * standard_deviation
        no_outliers = pop[not_outlier]

        print(pop.min(), pop.max(), pop.mean())
        all_arr.append(no_outliers)


    ax=sns.boxplot(data=all_arr, linewidth=1.0, fliersize=0.2)
    ax.set_xticklabels(["PI","WI","RFs","RFm" ,"GB100", "GB250","GB500"])
    plt.title('Percentage Error: {}'.format(attr_value), fontsize=11)
    # Set x-axis label
    plt.xlabel('Models', fontsize=11)
    # Set y-axis label
    plt.ylabel('Percentage Error (%)', fontsize=11)
    plt.ylim((-110,200))
    #plt.show()
    plt.savefig(directory0, dpi=300) 
    plt.cla()
    plt.close()

for i in attr_value:
    plotBoxPlot(i)