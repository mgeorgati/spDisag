import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import mapclassify
import rasterio 
import rasterio.plot
import numpy as np

def plot_mapVectorPolygons(city, evalType, src, exportPath, title,LegendTitle, column, districtPath, neighPath, waterPath, invertArea, addLabels=True):
    """Plot data from vector polygonal datasets using an auxiliary raster to represent water with labels.

    Parameters
    ----------
    src : geopandas.GeoDataFrame
        layer of lower administrative division to be plotted [grid-neighborhood-district-municipality-nuts3]
    districtPath : str
        layer of higher administrative division 
    neigh : geopandas.GeoDataFrame
        layer of lower administrative division [grid-neighborhood-district-municipality-nuts3]
    column : str
        name of column in src_path to be plotted
    evalType : str
        type of evaluation ['mae', 'div']
    title : str
        title of the map
    LegendTitle : str
        titlr of the legend
    waterPath : str
        path to raster layer for water representation
    invertArea : str
        Area outside case study area to represent sth of hide sth
    exportPath : str
         Path to save the map

    Returns
    -------
    image .png

    """
    print(city)
    src['{}'.format(column)].replace(np.inf, 0, inplace=True)
    mean = np.round(src['{}'.format(column)].mean(),2)
    valMax =  np.round(src['{}'.format(column)].max(),2)
    valMin = src['{}'.format(column)].min()
    print(valMin, mean, valMax)
    fig, ax = plt.subplots(figsize=(20, 20),facecolor='white') #50, 50
    """cmap = mpl.cm.OrRd
    bounds = [0,75,95,105,125,valMax]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
    #sm = plt.cm.ScalarMappable(cmap='OrRd', norm = plt.Normalize(vmin=valMin,vmax=valMax))
    sm = plt.cm.ScalarMappable(cmap='OrRd', norm = norm)
    fig.colorbar(sm, orientation="horizontal", fraction=0.036, pad =0.1, aspect=10)"""
    cmap, norm, legend_labels = defineBinsVector(evalType, src, column, valMin, valMax, mean)
    
    # Plot the data                       
    src.plot(column='{}'.format(column), ax=ax, cmap=cmap, norm=norm, edgecolor='#000000',linewidth= 0.1, zorder=1) #norm=norm,
    
    if districtPath:
        srcfile = gpd.read_file(districtPath)
        srcfile.plot(ax=ax, facecolor='none', edgecolor='#000000', linewidth=0.8,  zorder=17 ) #alpha=0.8,
        if addLabels:
            srcfile['coords']= srcfile['geometry'].apply(lambda x: x.representative_point().coords[:])
            srcfile['coords'] = [coords[0] for coords in srcfile['coords']]
            for idx, row in srcfile.iterrows():
                if 'Stadsdeel' in srcfile:
                    plt.annotate(text = row['Stadsdeel'], xy=row['coords'], horizontalalignment= 'center', fontsize=12) 
                elif 'gm_naam' in srcfile.columns:
                    plt.annotate(text = row['gm_naam'], xy=row['coords'], horizontalalignment= 'center', fontsize=12)
                elif 'KOMNAME' in srcfile.columns:
                    plt.annotate(text = row['KOMNAME'], xy=row['coords'], horizontalalignment= 'center', fontsize=12)
                else:
                    print("There is no column for labels")
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
      
    # Set the plot extent                
    xlim = ([src.total_bounds[0],  src.total_bounds[2]])
    ylim = ([src.total_bounds[1],  src.total_bounds[3]])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Add a title
    ax.set_title(title, color="black", fontsize=18)
    # Connect labels and colors, create legend and save the image
    patches = [mpatches.Patch(color=color, label=label)
                for color, label in legend_labels.items()]   
    if city == "ams" :         
        ax.legend(handles=patches,  loc='lower left', facecolor="white", fontsize=12, title = LegendTitle, title_fontsize=14).set_zorder(6)
    else:
        ax.legend(handles=patches,  loc='lower right', facecolor="white", fontsize=12, title = LegendTitle, title_fontsize=14).set_zorder(6)
  
    print("-----PLOTTING IMAGE {}-----".format(title))
    plt.savefig(exportPath, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(),transparent=True) #
    #ax.axis('off')
    #plt.show()
    plt.cla()
    plt.close()

def defineBinsVector(evalType, src, column, valMin, valMax, mean): #
    if evalType == "div":
        bins = 7
        if 0<=valMin<=300 and 0<=valMax<=300:
            print(1,'edo')
            a=[0, 50, 75, 95, 105, 125, 200, 300]
        elif 50 <= valMin <= valMax :
            print(2)
            a=[0, 50, 75, 95, 105, 125, 200, valMax]
        else:
            print(3, 'edo2')
            a=[0, 50, 75, 95, 105, 125, 200, 300]
        cmap = ListedColormap(["#d7191c","#f17c4a","#fec980", "#ffffbf", "#c7e9ad", "#80bfac","#2b83ba"])
        norm = colors.BoundaryNorm(a, bins)  
        # Add a legend for labels
        legend_labels = { "#d7191c": "<{0}".format(int(a[1])), "#f17c4a": "{0}-{1}".format(int(a[1]), int(a[2])), "#fec980": "{0}-{1}".format(int(a[2]),int(a[3])), 
        "#ffffbf": "{0}-{1}".format(int(a[3]),int(a[4])),
        "#c7e9ad": "{0}-{1}".format(int(a[4]),int(a[5])), "#80bfac": "{0}-{1}".format(int(a[5]),int(a[6])), 
        "#2b83ba": "{0}-{1}".format(int(a[6]),valMax), "#2b83ba00": "mean total:{}".format(mean)}
    elif evalType == "mae":
        bins = 7
        if -30<=valMax<=30 and -30<=valMin<=30:
            print(1)
            a = [-30,-15, -10, -5, 5, 10, 15, 30]
        elif -30<=valMin<=30:
            a = [-30,-15, -10, -5, 5, 10, 15, valMax]
        elif -30<=valMax<=30:
            a = [valMin,-15, -10, -5, 5, 10, 15, 30]
        else:
            a=[valMin,-15, -10, -5, 5, 10, 15, valMax] #a=[valMin,-15, -10, -5, 5, 10, 15,valMax]
        cmap = ListedColormap(["#d7191c","#f17c4a","#fec980", "#ffffbf", "#c7e9ad", "#80bfac","#2b83ba"])
        norm = colors.BoundaryNorm(a, bins)  
        # Add a legend for labels
        legend_labels = { "#d7191c": "{0}-{1}".format(int(a[0]), int(a[1])), "#f17c4a": "{0}-{1}".format(int(a[1]), int(a[2])), "#fec980": "{0}-{1}".format(int(a[2]),int(a[3])), 
        "#ffffbf": "{0}-{1}".format(int(a[3]),int(a[4])),
        "#c7e9ad": "{0}-{1}".format(int(a[4]),int(a[5])), "#80bfac": "{0}-{1}".format(int(a[5]),int(a[6])), 
        "#2b83ba": "{0}-{1}".format(int(a[6]),int(a[7])), "#2b83ba00": "mean total:{}".format(mean)}
    elif evalType == "PrE":
        bins = 7
        if valMax in range(-160,160) :
            a=[-160,-80, -30, -10, 10, 30, 80,160]
        else:
            a=[valMin,-80, -30, -10, 10, 30, 80,valMax]
        cmap = ListedColormap(["#d7191c","#f17c4a","#fec980", "#ffffbf", "#c7e9ad", "#80bfac","#2b83ba"])
        norm = colors.BoundaryNorm(a, bins)  
        # Add a legend for labels
        legend_labels = { "#d7191c": "{0}-{1}".format(valMin, int(a[1])), "#f17c4a": "{0}-{1}".format(int(a[1]), int(a[2])), "#fec980": "{0}-{1}".format(int(a[2]),int(a[3])), 
        "#ffffbf": "{0}-{1}".format(int(a[3]),int(a[4])),
        "#c7e9ad": "{0}-{1}".format(int(a[4]),int(a[5])), "#80bfac": "{0}-{1}".format(int(a[5]),int(a[6])), 
        "#2b83ba": "{0}-{1}".format(int(a[6]),valMax), "#2b83ba00": "mean total:{}".format(mean)}
    elif evalType == "popdistribution":
        bins = 7
        
        if valMax in range(200,500):
            a=[valMin,1, 50, 100, 200, 300, 400, 500]
        else:
            a=[valMin,1, 50, 100, 200, 300, 400, valMax]
        cmap = ListedColormap(["#f1eef600","#e0c8e2","#da9acb", "#df65b0", "#de348a", "#c61266","#980043"])
        norm = colors.BoundaryNorm(a, bins)  
        # Add a legend for labels
        legend_labels = { "#e0c8e2": "{0}-{1}".format(int(a[1]), int(a[2])), "#da9acb": "{0}-{1}".format(int(a[2]),int(a[3])), 
        "#df65b0": "{0}-{1}".format(int(a[3]),int(a[4])),
        "#de348a": "{0}-{1}".format(int(a[4]),int(a[5])), "#c61266": "{0}-{1}".format(int(a[5]),int(a[6])), 
        "#980043": "{0}-{1}".format(int(a[6]),valMax), "#2b83ba00": "mean total:{}".format(mean)}
    elif evalType == "popdistributionGrid":
        bins = 7
        if valMax in range(int(valMin),400):
            a=[valMin,1, 25, 50, 100, 200, 300, 400]
        else:
            a=[valMin,1, 25, 50, 100, 200, 300, valMax]
        cmap = ListedColormap(["#f1eef600","#e0c8e2","#da9acb", "#df65b0", "#de348a", "#c61266","#980043"])
        norm = colors.BoundaryNorm(a, bins)  
        # Add a legend for labels
        legend_labels = { "#e0c8e2": "{0}-{1}".format(int(a[1]), int(a[2])), "#da9acb": "{0}-{1}".format(int(a[2]),int(a[3])), 
        "#df65b0": "{0}-{1}".format(int(a[3]),int(a[4])),
        "#de348a": "{0}-{1}".format(int(a[4]),int(a[5])), "#c61266": "{0}-{1}".format(int(a[5]),int(a[6])), 
        "#980043": "{0}-{1}".format(int(a[6]),valMax), "#2b83ba00": "mean total:{}".format(mean)}
    elif evalType == "popdistributionPolyg":
        bins = 7
        if valMax <= 100000:
            print('edo2')
            a=[valMin, 1, 5500, 11000, 22500, 45000, 67500, 100000]
        elif 100000 < valMax <= 300000:
            print('edo3')
            a=[valMin, 20000, 40000, 60000, 80000, 100000, 200000, 300000]
        else:
            print('edo1')
            a=[valMin, 20000, 40000, 60000, 80000, 100000, 300000, valMax]
        cmap = ListedColormap(["#f1eef600","#e0c8e2","#da9acb", "#df65b0", "#de348a", "#c61266","#980043"])
        norm = colors.BoundaryNorm(a, bins)  
        # Add a legend for labels
        legend_labels = { "#e0c8e2": "{0}-{1}".format(int(a[1]), int(a[2])), "#da9acb": "{0}-{1}".format(int(a[2]),int(a[3])), 
        "#df65b0": "{0}-{1}".format(int(a[3]),int(a[4])),
        "#de348a": "{0}-{1}".format(int(a[4]),int(a[5])), "#c61266": "{0}-{1}".format(int(a[5]),int(a[6])), 
        "#980043": "{0}-{1}".format(int(a[6]),valMax), "#2b83ba00": "mean total:{}".format(mean)}
    else:
        bins = 7
        hist=np.asarray(np.histogram([valMin,valMax], bins=bins, density=True))
        a = hist[1]
        print(a)
        cmap = ListedColormap(["#d7191c","#f17c4a","#fec980", "#ffffbf", "#c7e9ad", "#80bfac","#2b83ba"])
        norm = colors.BoundaryNorm(a, bins)  
        # Add a legend for labels
        legend_labels = { "#d7191c": "{0}-{1}".format(valMin, round(a[1],2)), "#f17c4a": "{0}-{1}".format(round(a[1],2), round(a[2],2)), "#fec980": "{0}-{1}".format(round(a[2],2),round(a[3],2)), 
        "#ffffbf": "{0}-{1}".format(round(a[3],2),round(a[4],2)),
        "#c7e9ad": "{0}-{1}".format(round(a[4],2),round(a[5],2)), "#80bfac": "{0}-{1}".format(round(a[5],2),round(a[6],2)), 
        "#2b83ba": "{0}-{1}".format(round(a[6],2),valMax), "#2b83ba00": "mean total:{}".format(mean)}
        
    return cmap, norm, legend_labels

def defineBinsRaster(evalType, valMin, valMax, mean): #
    if evalType == "div":
        bins = 7
        if valMin <= valMax <= 300:
            a=[valMin, 50, 70, 85, 115, 130, 200, 300]
        else:
            a=[valMin, 50, 70, 85, 115, 200, 500, valMax] #75, 95, 105, 200,
        cmap = ListedColormap(["#d7191c","#f17c4a","#fec980", "#ffffbf", "#c7e9ad", "#80bfac","#2b83ba"])
        norm = colors.BoundaryNorm(a, bins)  
        # Add a legend for labels
        legend_labels = { "#d7191c": "<{0}".format(int(a[1])), "#f17c4a": "{0}-{1}".format(int(a[1]), int(a[2])), "#fec980": "{0}-{1}".format(int(a[2]),int(a[3])), 
        "#ffffbf": "{0}-{1}".format(int(a[3]),int(a[4])),
        "#c7e9ad": "{0}-{1}".format(int(a[4]),int(a[5])), "#80bfac": "{0}-{1}".format(int(a[5]),int(a[6])), 
        "#2b83ba": "{0}-{1}".format(int(a[6]),valMax), "#2b83ba00": "mean total:{}".format(mean)}
    elif evalType == "div007":
        bins = 7
        if valMin <= -100 and valMax >= 100:
            a=[valMin, -80, -60, -20, 20, 60, 80, valMax]
        else:
            a=[-100, -80, -60, -20, 20, 60, 80, 100] #75, 95, 105, 200,
        cmap = ListedColormap(["#d7191c","#f17c4a","#fec980", "#ffffbf", "#c7e9ad", "#80bfac","#2b83ba"])
        norm = colors.BoundaryNorm(a, bins)  
        # Add a legend for labels
        legend_labels = { "#d7191c": "<{0}".format(int(a[1])), "#f17c4a": "{0}-{1}".format(int(a[1]), int(a[2])), "#fec980": "{0}-{1}".format(int(a[2]),int(a[3])), 
        "#ffffbf": "{0}-{1}".format(int(a[3]),int(a[4])),
        "#c7e9ad": "{0}-{1}".format(int(a[4]),int(a[5])), "#80bfac": "{0}-{1}".format(int(a[5]),int(a[6])), 
        "#2b83ba": "{0}-{1}".format(int(a[6]),valMax), "#2b83ba00": "mean total:{}".format(mean)}
    elif evalType == "mae":
        bins = 7
        if valMin <= valMax <= 30:
            a=[valMin,-15, -10, -5, 5, 10, 15, 30]
        elif -30<=valMin<=30:
            a=[-30,-15, -10, -3, 3, 10, 15, 30]
        else:
            a=[valMin,-15, -10, -3, 3, 10, 15, valMax]
        cmap = ListedColormap(["#d7191c","#f17c4a","#fec980", "#ffffbf", "#c7e9ad", "#80bfac","#2b83ba"])
        norm = colors.BoundaryNorm(a, bins)  
        # Add a legend for labels
        legend_labels = { "#d7191c": "{0}-{1}".format(valMin, int(a[1])), "#f17c4a": "{0}-{1}".format(int(a[1]), int(a[2])), "#fec980": "{0}-{1}".format(int(a[2]),int(a[3])), 
        "#ffffbf": "{0}-{1}".format(int(a[3]),int(a[4])),
        "#c7e9ad": "{0}-{1}".format(int(a[4]),int(a[5])), "#80bfac": "{0}-{1}".format(int(a[5]),int(a[6])), 
        "#2b83ba": "{0}-{1}".format(int(a[6]),valMax), "#2b83ba00": "mean total:{}".format(mean)}
    elif evalType == "PrE":
        bins = 7
        if valMin <= valMax <= 160:
            a=[valMin,-80, -30, -10, 10, 30, 80,160]
        else:
            a=[valMin,-80, -30, -10, 10, 30, 80,valMax]
        cmap = ListedColormap(["#d7191c","#f17c4a","#fec980", "#ffffbf", "#c7e9ad", "#80bfac","#2b83ba"])
        norm = colors.BoundaryNorm(a, bins)  
        # Add a legend for labels
        legend_labels = { "#d7191c": "{0}-{1}".format(valMin, int(a[1])), "#f17c4a": "{0}-{1}".format(int(a[1]), int(a[2])), "#fec980": "{0}-{1}".format(int(a[2]),int(a[3])), 
        "#ffffbf": "{0}-{1}".format(int(a[3]),int(a[4])),
        "#c7e9ad": "{0}-{1}".format(int(a[4]),int(a[5])), "#80bfac": "{0}-{1}".format(int(a[5]),int(a[6])), 
        "#2b83ba": "{0}-{1}".format(int(a[6]),valMax), "#2b83ba00": "mean total:{}".format(mean)}
    elif evalType == "popdistribution":
        bins = 7
        if valMax < 200:
            a=[valMin,1, 25, 50, 75, 100, 150, 200]
        elif 200<= valMax <= 500:
            a=[0, 1, 50, 100, 200, 300, 400, 500]
        else:
            a=[valMin,1, 50, 100, 200, 300, 400, valMax]
            #a=[valMin,1, 25, 50, 75, 100, 150, valMax]
        cmap = ListedColormap(["#f1eef600","#e0c8e2","#da9acb", "#df65b0", "#de348a", "#c61266","#980043"])
        norm = colors.BoundaryNorm(a, bins)  
        # Add a legend for labels
        legend_labels = { "#e0c8e2": "{0}-{1}".format(int(a[1]), int(a[2])), "#da9acb": "{0}-{1}".format(int(a[2]),int(a[3])), 
        "#df65b0": "{0}-{1}".format(int(a[3]),int(a[4])),
        "#de348a": "{0}-{1}".format(int(a[4]),int(a[5])), "#c61266": "{0}-{1}".format(int(a[5]),int(a[6])), 
        "#980043": "{0}-{1}".format(int(a[6]),valMax), "#2b83ba00": "mean total:{}".format(mean)}
    elif evalType == "popdistributionPolyg":
        bins = 7
        if valMin <= valMax <= 100000:
            a=[valMin, 1, 5500, 11000, 22500, 45000, 67500, 90000]
        else:
            a=[valMin,1, 500, 1500, 3000, 4500, 6000, valMax]
        cmap = ListedColormap(["#f1eef600","#e0c8e2","#da9acb", "#df65b0", "#de348a", "#c61266","#980043"])
        norm = colors.BoundaryNorm(a, bins)  
        # Add a legend for labels
        legend_labels = { "#e0c8e2": "{0}-{1}".format(int(a[1]), int(a[2])), "#da9acb": "{0}-{1}".format(int(a[2]),int(a[3])), 
        "#df65b0": "{0}-{1}".format(int(a[3]),int(a[4])),
        "#de348a": "{0}-{1}".format(int(a[4]),int(a[5])), "#c61266": "{0}-{1}".format(int(a[5]),int(a[6])), 
        "#980043": "{0}-{1}".format(int(a[6]),valMax), "#2b83ba00": "mean total:{}".format(mean)}
    else:
        bins = 7
        hist=np.asarray(np.histogram([valMin,valMax], bins=bins, density=True))
        a = hist[1]
        print(a)
        cmap = ListedColormap(["#d7191c","#f17c4a","#fec980", "#ffffbf", "#c7e9ad", "#80bfac","#2b83ba"])
        norm = colors.BoundaryNorm(a, bins)  
        # Add a legend for labels
        legend_labels = { "#d7191c": "{0}-{1}".format(valMin, round(a[1],2)), "#f17c4a": "{0}-{1}".format(round(a[1],2), round(a[2],2)), "#fec980": "{0}-{1}".format(round(a[2],2),round(a[3],2)), 
        "#ffffbf": "{0}-{1}".format(round(a[3],2),round(a[4],2)),
        "#c7e9ad": "{0}-{1}".format(round(a[4],2),round(a[5],2)), "#80bfac": "{0}-{1}".format(round(a[5],2),round(a[6],2)), 
        "#2b83ba": "{0}-{1}".format(round(a[6],2),valMax), "#2b83ba00": "mean total:{}".format(mean)}
        
    return cmap, norm, legend_labels

