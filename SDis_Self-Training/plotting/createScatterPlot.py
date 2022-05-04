import sys, os, seaborn as sns, rasterio, pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.definitions import ROOT_DIR, ancillary_path, city,year


attr_value ="totalpop"    
gtP = ROOT_DIR + "/Evaluation/{0}_groundTruth/{2}_{0}_{1}.tif".format(city,attr_value,year)
srcGT= rasterio.open(gtP)
popGT = srcGT.read(1)
print(popGT.min(),popGT.max(), popGT.mean())

#prP = ROOT_DIR + "/Evaluation/{0}/apcatbr/div_{0}_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_ag_{1}.tif".format(city,attr_value)
def scatterplot(prP):
    cp = "C:/Users/NM12LQ/OneDrive - Aalborg Universitet/PopNetV2_backup/data_prep/ams_ProjectData/temp_tif/ams_CLC_2012_2018Reclas3.tif"
    srcC= rasterio.open(cp)
    corine = srcC.read(1)
    name = prP.split(".tif")[0].split("/")[-1]
    print(name)
    gtP = ROOT_DIR + "/Evaluation/{0}_groundTruth/{2}_{0}_{1}.tif".format(city,attr_value,year)
    srcGT= rasterio.open(gtP)
    popGT = srcGT.read(1)
    print(popGT.min(),popGT.max(), popGT.mean())
    srcPR= rasterio.open(prP)
    popPR = srcPR.read(1)
    popPR[(np.where(popPR <= -9999))] = 0 
    print(popPR.min(),popPR.max(), popPR.mean())
    cr=corine.flatten()
    x=popGT.flatten()
    y=popPR.flatten()
    df = pd.DataFrame(data={"gt": x, "predictions":y, "cr":cr})

    plt.figure(figsize=(20,20))

    g = sns.lmplot(data=df, x="gt", y="predictions", hue="cr", palette=["#0d2dc1","#ff9c1c","#71b951","#24f33d","#90308f", "#a8a8a8"],ci = None, order=2, scatter_kws={"s":0.5, "alpha": 0.5}, line_kws={"lw":2, "alpha": 0.5}, legend=False) 
    plt.legend(title= "Land Cover", labels= ['Water','Urban Fabric', 'Agriculture', 'Green Spaces','Industry','Transportation' ], loc='lower right', fontsize=5)
    plt.title('{0}'.format( name), fontsize=11)
    # Set x-axis label
    plt.xlabel('Ground Truth (persons)', fontsize=11)
    # Set y-axis label
    plt.ylabel('Predictions (persons)', fontsize=11)
    #total pop
    #plt.xscale('log')
    #plt.yscale('log')
    
    #mobile Adults
    #plt.xlim((0,200))
    #plt.ylim((-100,500))pl
    plt.axis('square')
    plt.xlim((0,400))
    plt.ylim((0,350))
    plt.tight_layout()
    #plt.show()
    plt.savefig(ROOT_DIR + "/Evaluation/{0}/ScatterPlots/SP4_{2}.png".format(city,attr_value, name),format='png',dpi=300)
    
evalFiles = [#gtP, 
                #ROOT_DIR + "/Evaluation/{0}/aprf/dissever00/{0}_dissever00WIESMN_2018_ams_Dasy_aprf_p[1]_12AIL12_1IL_it10_{1}.tif".format(city,attr_value),
                #ROOT_DIR + "/Evaluation/{0}/aprf/dissever01/{0}_dissever01WIESMN_100_2018_ams_DasyA_aprf_p[1]_12AIL12_13IL_it10_{1}.tif".format(city,attr_value),
                #ROOT_DIR + "/Evaluation/{0}/apcatbr/{0}_dissever01WIESMN_100_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_ag_{1}.tif".format(city,attr_value),
                #ROOT_DIR + "/Evaluation/{0}/apcatbr/{0}_dissever01WIESMN_250_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_ag_{1}.tif".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/apcatbr/{0}_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_ag_{1}.tif".format(city,attr_value),
                ]
    
evalFilesMAEbp = [ROOT_DIR + "/Evaluation/{0}/Pycno/mae_{0}_{2}_{0}_{1}_pycno.tif".format(city,attr_value,year),
                ROOT_DIR + "/Evaluation/{0}/Dasy/mae_{0}_{2}_{0}_{1}_dasyWIESMN.tif".format(city,attr_value,year),
                ROOT_DIR + "/Evaluation/{0}/aprf/dissever00/mae_{0}_dissever00WIESMN_2018_ams_Dasy_aprf_p[1]_12AIL12_1IL_it10_{1}.tif".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/aprf/dissever01/mae_{0}_dissever01WIESMN_100_2018_ams_DasyA_aprf_p[1]_12AIL12_13IL_it10_{1}.tif".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/apcatbr/mae_{0}_dissever01WIESMN_100_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_ag_{1}.tif".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/apcatbr/mae_{0}_dissever01WIESMN_250_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_ag_{1}.tif".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/apcatbr/mae_{0}_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_ag_{1}.tif".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/apcatbr/mae_{0}_dissever01WIESMN_250_2018_ams_DasyA_apcatbr_p[1]_3AIL5_12IL_it10_ag_{1}.tif".format(city,attr_value)]
evalFilesPEbp = [ROOT_DIR + "/Evaluation/{0}/Pycno/div_{0}_{2}_{0}_{1}_pycno.tif".format(city,attr_value,year),
                ROOT_DIR + "/Evaluation/{0}/Dasy/div_{0}_{2}_{0}_{1}_dasyWIESMN.tif".format(city,attr_value,year),
                ROOT_DIR + "/Evaluation/{0}/aprf/dissever00/div_{0}_dissever00WIESMN_2018_ams_Dasy_aprf_p[1]_12AIL12_1IL_it10_{1}.tif".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/aprf/dissever01/div_{0}_dissever01WIESMN_100_2018_ams_DasyA_aprf_p[1]_12AIL12_13IL_it10_{1}.tif".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/apcatbr/div_{0}_dissever01WIESMN_100_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_ag_{1}.tif".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/apcatbr/div_{0}_dissever01WIESMN_250_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_ag_{1}.tif".format(city,attr_value),
                ROOT_DIR + "/Evaluation/{0}/apcatbr/div_{0}_dissever01WIESMN_500_2018_ams_DasyA_apcatbr_p[1]_12AIL12_12IL_it10_ag_{1}.tif".format(city,attr_value)]

for i in evalFiles:
    scatterplot(i)
