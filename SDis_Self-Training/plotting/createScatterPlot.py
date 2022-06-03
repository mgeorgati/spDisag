import sys, os, seaborn as sns, rasterio, pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def scatterplot(xPath, yPath, outPath, xlab, ylab, title, x1, x2, y1, y2):
    cp = "C:/Users/NM12LQ/OneDrive - Aalborg Universitet/PopNetV2_backup/data_prep/ams_ProjectData/temp_tif/ams_CLC_2012_2018Reclas3.tif"
    srcC= rasterio.open(cp)
    corine = srcC.read(1)
    
    name = yPath.stem
    x_src= rasterio.open(xPath)
    
    x_arr = x_src.read(1)
    srcPR= rasterio.open(yPath)
    y_arr = srcPR.read(1)
    y_arr[(np.where(y_arr <= -9999))] = 0 
    
    cr = corine.flatten()
    x = x_arr.flatten()
    y = y_arr.flatten()
    df = pd.DataFrame(data={"x": x, "y":y, "cr":cr})
    
    print(np.min(y), np.min(x))
    #plt.figure(figsize=(20,20))
    #g = sns.lmplot(data=df, x="x", y="y", hue="cr", palette=["#0d2dc1","#ff9c1c","#71b951","#24f33d","#90308f", "#a8a8a8"],ci = None, order=2, scatter_kws={"s":0.5, "alpha": 0.5}, line_kws={"lw":2, "alpha": 0.5}, legend=False) 
    # Two subplots
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # ax1.plot(df.x, df.y)
    ax1.set_title('Sharing Y axis')
    # ax2.scatter(df.x, df.y)

    sns.regplot(data=df, x='x', y='y', ax=ax1)
    sns.histplot(data=df, x='x', ax=ax2)
    
    plt.legend(title= "Land Cover", labels= ['Water','Urban Fabric', 'Agriculture', 'Green Spaces','Industry','Transportation' ], loc='lower right', fontsize=5)
    plt.title('{0}\n{1}'.format(title, name) , fontsize=8)
    # adding vertical line in data co-ordinates
    plt.axvline(0, c='black', ls=':', lw=0.2)
    
    # adding horizontal line in data co-ordinates
    plt.axhline(0, c='black', ls='--', lw=0.2)
    # Set x-axis label
    plt.xlabel(xlab, fontsize=11)
    # Set y-axis label
    plt.ylabel(ylab, fontsize=11)
    #plt.xscale('log')
    #plt.yscale('log')
    
    #plt.axis('square')
    if x1!=None and x2!=None and y1!=None and y2!=None:
        plt.xlim((x1, x2))
        plt.ylim((y1, y2))
    plt.tight_layout()
    #plt.show()
    plt.savefig(outPath, format='png',dpi=300)
    
def histplot(xPath, yPath, outPath, xlab, ylab, title, x1, x2, y1, y2):
    cp = "C:/Users/NM12LQ/OneDrive - Aalborg Universitet/PopNetV2_backup/data_prep/ams_ProjectData/temp_tif/ams_CLC_2012_2018Reclas3.tif"
    srcC= rasterio.open(cp)
    corine = srcC.read(1)
    
    name = yPath.stem
    x_src= rasterio.open(xPath)
    
    x_arr = x_src.read(1)
    x_arr[(np.where(x_arr < 1))] = np.nan 
    
    cr = corine.flatten()
    x = x_arr.flatten()

    df = pd.DataFrame(data={"x": x, "cr":cr})
    

    plt.figure(figsize=(20,20))
    
    sns.histplot(data=df, x='x', hue="cr", palette=["#0d2dc1","#ff9c1c","#71b951","#24f33d","#90308f", "#a8a8a8"], common_norm=False)
    
    plt.legend(title= "Land Cover", labels= ['Water','Urban Fabric', 'Agriculture', 'Green Spaces','Industry','Transportation' ], loc='lower right', fontsize=5)
    plt.title('{0}\n{1}'.format(title, name) , fontsize=8)
    # adding vertical line in data co-ordinates
    plt.axvline(0, c='black', ls=':', lw=0.2)
    
    # adding horizontal line in data co-ordinates
    plt.axhline(0, c='black', ls='--', lw=0.2)
    # Set x-axis label
    plt.xlabel(xlab, fontsize=11)
    # Set y-axis label
    plt.ylabel(ylab, fontsize=11)
    #plt.xscale('log')
    #plt.yscale('log')
    
    #plt.axis('square')
    if x1!=None and x2!=None and y1!=None and y2!=None:
        plt.xlim((x1, x2))
        plt.ylim((y1, y2))
    plt.tight_layout()
    #plt.show()
    plt.savefig(outPath, format='png',dpi=300)

"""
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
"""

    
