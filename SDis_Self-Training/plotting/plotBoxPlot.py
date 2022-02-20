import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
def compareCounts(fileList, column):
    df = pd.DataFrame()
    for i in fileList:
        path =Path(i)
        name = path.stem
        src = gpd.read_file(i)
        #print(src)
        src = src.loc[src['BU_CODE'].str.contains('BU0363')]
        if 'BU_CODE' not in df.columns:
            df['BU_CODE'] = src['BU_CODE'] 
            df['{}'.format(name)] = src['{}'.format(column)]
            #df = df.join(src.set_index('BU_CODE'), lsuffix='_l') 
        else:
            df['{}'.format(name)] = src['{}'.format(column)]
    print(df.head(3))
    for col in df.columns:      
        if col!="ams_l1_totalpop_2018" and col!= "BU_CODE":
            print(col)
            df['{}'.format(col)] = df['{}'.format(col)].fillna(0).astype(np.int64)
            df["dif_{}".format(col)] = df[col] - df["ams_l1_totalpop_2018"] 
            
            df["Error_{}".format(col)] = (df[col] - df["ams_l1_totalpop_2018"]) / df["ams_l1_totalpop_2018"].sum() * 100
            df["Accuracy_{}".format(col)] = (df[col] / df["ams_l1_totalpop_2018"]) * 100
            df["PrE_{}".format(col)] = (df[col] - df["ams_l1_totalpop_2018"]) * (df["ams_l1_totalpop_2018"] / df["ams_l1_totalpop_2018"].sum()) * 100#

    #frame['dif_sum_2018_l1_totalpop'][frame.sum_2018_l1_totalpop == 0] = 0
    #frame['Error_sum_2018_l1_totalpop'][frame.sum_2018_l1_totalpop == 0] = 0
    return df

def BoxPlotCBS(directory, df, xLegend):
    sns.boxplot(x=xLegend, y="Type", data=pd.melt(df, var_name='Type', value_name=xLegend), linewidth=1.0)
    #plt.show()
    plt.savefig(directory, dpi=300, bbox_inches='tight',) 
    plt.cla()
    plt.close()
    
def BoxPlotCBS_NO(directory, df, xLegend):
    for column in df.columns:
        Q1 = df['{}'.format(column)].quantile(0.25)
        Q3 = df['{}'.format(column)].quantile(0.75)
        IQR = Q3 - Q1    #IQR is interquartile range. 

        filter = (df['{}'.format(column)] >= Q1 - 1.5 * IQR) & (df['{}'.format(column)] <= Q3 + 1.5 *IQR)
        df =df.loc[filter]
        
    sns.boxplot(x=xLegend, y="Type", data=pd.melt(df, var_name='Type', value_name=xLegend), linewidth=1.0)
    #sns.swarmplot(x="Accuracy", y="Type", data=pd.melt(df, var_name='Type', value_name='Accuracy'), color=".15")
    #plt.show()
    plt.savefig(directory, dpi=300, bbox_inches='tight',) 
    plt.cla()
    plt.close()
    
def BoxPlot(directory, fileList, column):
    df = pd.DataFrame()
    for i in fileList:
        path =Path(i)
        name = path.stem
        src = gpd.read_file(i)
        df['{}'.format(name)] = src['{}'.format(column)] 
    print(df.head(3))
    sns.boxplot(x="Accuracy", y="Type", data=pd.melt(df, var_name='Type', value_name='Accuracy'), linewidth=1.0)
    #plt.show()
    plt.savefig(directory, dpi=300, bbox_inches='tight',) 
    plt.cla()
    plt.close()
    
def BoxPlotNoOutliers(directory, fileList, column):
    df = pd.DataFrame()
    for i in fileList:
        path =Path(i)
        name = path.stem
        src = gpd.read_file(i)
        Q1 = src['{}'.format(column)].quantile(0.25)
        Q3 = src['{}'.format(column)].quantile(0.75)
        IQR = Q3 - Q1    #IQR is interquartile range. 

        filter = (src['{}'.format(column)] >= Q1 - 1.5 * IQR) & (src['{}'.format(column)] <= Q3 + 1.5 *IQR)
        src= src.loc[filter]
        df['{}'.format(name)] = src['{}'.format(column)] 
    print(df.head(3))
    sns.boxplot(x="Accuracy", y="Type", data=pd.melt(df, var_name='Type', value_name='Accuracy'), linewidth=1.0)
    #sns.swarmplot(x="Accuracy", y="Type", data=pd.melt(df, var_name='Type', value_name='Accuracy'), color=".15")
    #plt.show()
    plt.savefig(directory, dpi=300, bbox_inches='tight',) 
    plt.cla()
    plt.close()   


