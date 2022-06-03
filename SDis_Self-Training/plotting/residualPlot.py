#import necessary libraries 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import rasterio

#define figure size
fig = plt.figure(figsize=(12,8))

#produce regression plots
fig = sm.graphics.plot_regress_exog(model, 'points', fig=fig)

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
    