import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import rasterio 
import rasterio.plot
import numpy as np
from .plotVectors import defineBinsRaster

def plot_map(city, evalType, src, exportPath, title,LegendTitle, districtPath, neighPath, waterPath, invertArea, addLabels=True):
    pop = src.read(1)
    pop= np.nan_to_num(pop, nan=0)
    pop = pop.astype(np.int64)
    pop[(np.where(pop == -9999))] = 0
    """    
    #neighborhoods = src.read(2)
    #neighborhood = neighborhoods == 1 
                      
    #pop = pop[neighborhood]
    """
    valMax = np.round(np.max(pop), 2)
    valMin = np.round(np.min(pop), 2)
    mean = np.round(np.mean(pop), 2)
    print(np.max(pop), np.min(pop))
    fig, ax = plt.subplots(figsize=(20, 20),facecolor='white') #50, 50
    cmap, norm, legend_labels = defineBinsRaster(evalType, valMin, valMax, mean)
    #hist=np.asarray(np.histogram(no_outliers, bins=bins, density=True))
    #values = hist[1]
    #print(values)
    #b= np.round(values, decimals=2)
    if districtPath:
        srcfile = gpd.read_file(districtPath)
        srcfile.plot(ax=ax, facecolor='none', edgecolor='#000000', linewidth=0.8,  zorder=17 ) #alpha=0.8,
        if addLabels:
            srcfile['coords']= srcfile['geometry'].apply(lambda x: x.representative_point().coords[:])
            srcfile['coords'] = [coords[0] for coords in srcfile['coords']]
            for idx, row in srcfile.iterrows():
                if 'Stadsdeel' in srcfile.columns:
                    plt.annotate(text = row['Stadsdeel'], xy=row['coords'], horizontalalignment= 'center', fontsize=12) 
                elif 'gm_naam' in srcfile.columns:
                    plt.annotate(text = row['gm_naam'], xy=row['coords'], horizontalalignment= 'center', fontsize=12)
                else:
                    print("There is no column for labels")
        xlim = ([srcfile.total_bounds[0],  srcfile.total_bounds[2]])
        ylim = ([srcfile.total_bounds[1],  srcfile.total_bounds[3]])
    else:
        xlim = ([src.bounds[0],  src.bounds[2]])
        ylim = ([src.bounds[1],  src.bounds[3]])
    if invertArea:
        outArea= gpd.read_file(invertArea)
        outArea.plot(ax=ax, facecolor='#FFFFFF', edgecolor='#000000', linewidth=.3, zorder=3 )
    if neighPath:
        neighborhoods = gpd.read_file(neighPath)
        neighborhoods.plot(ax=ax, facecolor='none', edgecolor='#000000', linewidth=.3, alpha=0.6, zorder=4 )
    if waterPath:
        waterTif = rasterio.open(waterPath)
        # colors for water layer
        cmapWater = ListedColormap(["#00000000","#7fa9b060" ])
        rasterio.plot.show(waterTif, ax=ax, cmap=cmapWater, zorder=5)
    
    # Plot the data                       
    rasterio.plot.show(waterTif, ax=ax, cmap=cmapWater, zorder=5)
    rasterio.plot.show(src, ax=ax, cmap=cmap, norm=norm, extent= [src.bounds[0],src.bounds[1], src.bounds[2], src.bounds[3]], zorder=1)               
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Add a title
    ax.set_title(title, color="black", fontsize=18)
    # Connect labels and colors, create legend and save the image
    patches = [mpatches.Patch(color=color, label=label)
                for color, label in legend_labels.items()]     
    if city == 'ams':
        ax.legend(handles=patches,  loc='lower left', facecolor="white", fontsize=12, title = LegendTitle, title_fontsize=14).set_zorder(6)
    else:        
        ax.legend(handles=patches,  loc='lower right', facecolor="white", fontsize=12, title = LegendTitle, title_fontsize=14).set_zorder(6)
    print("-----PLOTTING IMAGE {}-----".format(title))
    plt.savefig(exportPath, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(),transparent=True)
    
    plt.cla()
    plt.close()