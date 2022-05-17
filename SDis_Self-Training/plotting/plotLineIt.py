import subprocess
import sys,os
import numpy as np
import rasterio
import glob
import pandas as pd, geopandas as gpd
import seaborn as sns
from config.definitions import ROOT_DIR, pop_path, ancillary_path, python_scripts_folder_path
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('TKAgg')

fileName = ['pcounts_2018_cph_Dasy_aprf_p[1]_14AIL7_1IL_it10_2e']
#'pcounts_2018_cph_Dasy_aprf_p[1]_8AIL0_1IL_it10_2e', 'pcounts_2018_cph_Dasy_aprf_p[1]_7AIL1_1IL_it10_2e','pcounts_2018_cph_Dasy_aprf_p[1]_11AIL2_1IL_it10_2e','pcounts_2018_cph_Dasy_aprf_p[1]_4AIL3_1IL_it10_2e','pcounts_2018_cph_Dasy_aprf_p[1]_11AIL4_1IL_it10_2e','pcounts_2018_cph_Dasy_aprf_p[1]_7AIL5_1IL_it10_2e','pcounts_2018_cph_Dasy_aprf_p[1]_10AIL6_1IL_it10_2e'
srcPath = 'K:/FUME/DasymetricMapping/SDis_Self-Training/TempCSV'
for i in os.listdir(srcPath):
    if i.endswith('.csv'):
        print(i)
        path = Path(i)
        name = path.stem
        print(name)
        if '_27IL_' in name:
            df = pd.read_csv(srcPath + '/' + i,delimiter=';')
            
            df =df.loc[(df['Val']== 'children') ] #& (df['Val']!= 'autoch') & (df['Val']!= 'sur') & (df['Val']!= 'mar') & (df['Val']!= 'ant') & (df['Val']!= 'tur')
            print(df.head(2))
            #sns.lineplot(data=df, x="IT", y="MAE")
            df_wide = df.pivot('Val', 'IT', 'MAE')
            #df_wide = df[['MAE','RMSE']]
            sns.lineplot(data=df_wide)
            #print(df_wide.head(2))
            #plt.show()
            #Add chart labels
            plt.title('Convergence')
            plt.xlabel('Iterations')
            plt.ylabel('Value')
            plt.savefig(srcPath + '/{}.png'.format(name), dpi=300, bbox_inches='tight')
            plt.cla()
            plt.close()
            del df
            del df_wide