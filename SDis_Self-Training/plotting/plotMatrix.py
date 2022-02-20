import sys, os, seaborn as sns, rasterio, pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.definitions import ROOT_DIR, ancillary_path, city,year
from plotting.plotBoxPlot import BoxPlot, BoxPlotNoOutliers
from plotting.createScatterPlot import plotMatrix
import glob
attr_value ="mobadults"
   
#ROOT_DIR + "/Evaluation/{0}/aprf/dissever01WIESMN_100_2018_ams_DasyA_aprf_p[1]_3AIL5_13IL_it10_{1}.png".format(city,attr_value), 
#ROOT_DIR + "/Evaluation/{0}/aprf/dissever00WIESMN_2018_ams_Dasy_aprf_p[1]_3AIL5_1IL_it10_{1}.png".format(city,attr_value)
evalFiles = [ROOT_DIR + "/Evaluation/{0}/{0}_GT_{1}.png".format(city,attr_value,year),
                ROOT_DIR + "/Evaluation/{0}/aprf/dissever00/{0}_dissever00WIESMN_2018_ams_Dasy_aprf_p[1]_12AIL12_1IL_it10_{1}.png".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/aprf/dissever01/{0}_dissever01WIESMN_100_2018_ams_DasyA_aprf_p[1]_12AIL12_13IL_it10_{1}.png".format(city,attr_value),
                
                ROOT_DIR + "/Evaluation/{0}/apcatbr/{0}_dissever01WIESMN_250_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{1}.png".format(city,attr_value)
                ]
evalFilesPE = [ROOT_DIR + "/Evaluation/{0}/{0}_GT_{1}.png".format(city,attr_value,year), ROOT_DIR + "/Evaluation/{0}/Pycno/div_{2}_{0}_{1}_pycno_Grid.png".format(city,attr_value,year),
                ROOT_DIR + "/Evaluation/{0}/Dasy/div_{2}_{0}_{1}_dasyWIESMN_Grid.png".format(city,attr_value,year),
                ROOT_DIR + "/Evaluation/{0}/aprf/dissever00/div_dissever00WIESMN_2018_ams_Dasy_aprf_p[1]_12AIL12_1IL_it10_{1}_Grid.png".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/aprf/dissever01/div_dissever01WIESMN_100_2018_ams_DasyA_aprf_p[1]_12AIL12_13IL_it10_{1}_Grid.png".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/apcatbr/div_dissever01WIESMN_100_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_ag_{1}_Grid.png".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/apcatbr/div_dissever01WIESMN_250_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_ag_{1}_Grid.png".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/apcatbr/div_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_ag_{1}_Grid.png".format(city,attr_value)]
evalFilesMAE = [ROOT_DIR + "/Evaluation/{0}/{0}_GT_{1}.png".format(city,attr_value,year), ROOT_DIR + "/Evaluation/{0}/Pycno/mae_{2}_{0}_{1}_pycno_Grid.png".format(city,attr_value,year),
                ROOT_DIR + "/Evaluation/{0}/Dasy/mae_{2}_{0}_{1}_dasyWIESMN_Grid.png".format(city,attr_value,year),
                ROOT_DIR + "/Evaluation/{0}/aprf/dissever00/mae_dissever00WIESMN_2018_ams_Dasy_aprf_p[1]_12AIL12_1IL_it10_{1}_Grid.png".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/aprf/dissever01/mae_dissever01WIESMN_100_2018_ams_DasyA_aprf_p[1]_12AIL12_13IL_it10_{1}_Grid.png".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/apcatbr/mae_dissever01WIESMN_100_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_ag_{1}_Grid.png".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/apcatbr/mae_dissever01WIESMN_250_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_ag_{1}_Grid.png".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/apcatbr/mae_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_ag_{1}_Grid.png".format(city,attr_value)]

###############################################################################################
evalFilesMAEbp = [ROOT_DIR + "/Evaluation/{0}/Pycno/mae_{2}_{0}_{1}_pycno_Grid.png".format(city,attr_value,year),
                ROOT_DIR + "/Evaluation/{0}/Dasy/mae_{2}_{0}_{1}_dasyWIESMN_Grid.png".format(city,attr_value,year),
                ROOT_DIR + "/Evaluation/{0}/aprf/dissever00/mae_dissever00WIESMN_2018_ams_Dasy_aprf_p[1]_12AIL12_1IL_it10_{1}_Grid.png".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/aprf/dissever01/mae_dissever01WIESMN_100_2018_ams_DasyA_aprf_p[1]_12AIL12_13IL_it10_{1}_Grid.png".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/apcatbr/mae_dissever01WIESMN_100_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_ag_{1}_Grid.png".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/apcatbr/mae_dissever01WIESMN_250_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_ag_{1}_Grid.png".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/apcatbr/mae_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_ag_{1}_Grid.png".format(city,attr_value)]
evalFilesPEbp = [ROOT_DIR + "/Evaluation/{0}/Pycno/div_{2}_{0}_{1}_pycno_Grid.png".format(city,attr_value,year),
                ROOT_DIR + "/Evaluation/{0}/Dasy/div_{2}_{0}_{1}_dasyWIESMN_Grid.png".format(city,attr_value,year),
                ROOT_DIR + "/Evaluation/{0}/aprf/dissever00/div_dissever00WIESMN_2018_ams_Dasy_aprf_p[1]_12AIL12_1IL_it10_{1}_Grid.png".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/aprf/dissever01/div_dissever01WIESMN_100_2018_ams_DasyA_aprf_p[1]_12AIL12_13IL_it10_{1}_Grid.png".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/apcatbr/div_dissever01WIESMN_100_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_ag_{1}_Grid.png".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/apcatbr/div_dissever01WIESMN_250_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_ag_{1}_Grid.png".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/apcatbr/div_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_ag_{1}_Grid.png".format(city,attr_value)]
evalFiles = [ROOT_DIR + "/Evaluation/{0}/{0}_GT_{1}.png".format(city,attr_value,year),
             ROOT_DIR + "/Evaluation/{0}/aprf/dissever00/div_dissever00WIESMN_2018_ams_Dasy_aprf_p[1]_12AIL12_1IL_it10_{1}_Grid.png".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/aprf/dissever01/div_dissever01WIESMN_100_2018_ams_DasyA_aprf_p[1]_12AIL12_13IL_it10_{1}_Grid.png".format(city,attr_value), 
                ROOT_DIR + "/Evaluation/{0}/apcatbr/div_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_{1}_Grid.png".format(city,attr_value)]

a = [ROOT_DIR + "/Evaluation/ams_final/ams_GT_mobadults.png",ROOT_DIR + "/Evaluation/ams_final/apcatbr/ams_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_mobadults.png",
     ROOT_DIR + "/Evaluation/ams_final/ams_GT_nonwestern.png",ROOT_DIR + "/Evaluation/ams_final/apcatbr/ams_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_nonwestern.png",
     ROOT_DIR + "/Evaluation/cph_final/cph_GT_mobadults.png",ROOT_DIR + "/Evaluation/cph_final/cph_dissever01WIN_500_2018_cph_DasyB_apcatbr_p[1]_3AIL8_27IL_it10_mobadults.png",
     ROOT_DIR + "/Evaluation/cph_final/cph_GT_notEU.png",ROOT_DIR + "/Evaluation/cph_final/cph_dissever01WIN_500_2018_cph_DasyB_apcatbr_p[1]_3AIL8_27IL_it10_notEU.png"]
b = [ROOT_DIR + "/Evaluation/ams_final/ams_GT_totalpop.png",ROOT_DIR + "/Evaluation/ams_final/aprf/dissever00/ams_dissever00WIESMN_2018_ams_Dasy_aprf_p[1]_12AIL12_1IL_it10_totalpop.png",
     ROOT_DIR + "/Evaluation/ams_final/aprf/dissever01/ams_dissever01WIESMN_100_2018_ams_DasyA_aprf_p[1]_12AIL12_13IL_it10_totalpop.png",ROOT_DIR + "/Evaluation/ams_final/apcatbr/ams_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_ag_totalpop.png"]
c = [ROOT_DIR + "/Evaluation/cph_final/cph_GT_totalpop.png",ROOT_DIR + "/Evaluation/cph_final/cph_dissever00WIN_2018_cph_Dasy_aprf_p[1]_3AIL8_1IL_it10_totalpop.png",
     ROOT_DIR + "/Evaluation/cph_final/cph_dissever01WIN_100_2018_cph_DasyB_aprf_p[1]_3AIL8_28IL_it10_totalpop.png",ROOT_DIR + "/Evaluation/cph_final/cph_dissever01WIN_500_2018_cph_DasyB_apcatbr_p[1]_3AIL8_27IL_it10_ag_totalpop.png"]
d = [ROOT_DIR + "/Evaluation/ams_final/aprf/dissever00/div_dissever00WIESMN_2018_ams_Dasy_aprf_p[1]_12AIL12_1IL_it10_totalpop_Grid.png",
     ROOT_DIR + "/Evaluation/ams_final/apcatbr/div_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_ag_totalpop_Grid.png"]

e= [ROOT_DIR + "/Evaluation/ams_final/aprf/dissever00/div_dissever00WIESMN_2018_ams_Dasy_aprf_p[1]_12AIL12_1IL_it10_totalpop_Grid.png",
    ROOT_DIR + "/Evaluation/ams_final/apcatbr/div_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_ag_totalpop_Grid.png",
    ROOT_DIR + "/Evaluation/cph_final/div_dissever00WIN_2018_cph_Dasy_aprf_p[1]_3AIL8_1IL_it10_totalpop_Grid.png",
    ROOT_DIR + "/Evaluation/cph_final/div_dissever01WIN_500_2018_cph_DasyB_apcatbr_p[1]_3AIL8_27IL_it10_ag_totalpop_Grid.png"]
path="C:/FUME/DasymetricMapping/SDis_Self-Training/Evaluation/Images/ape.png".format(attr_value)
title = "Percentage Error: Total Population"
plotMatrix(e, path, title) 
