import numpy as np
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
def binsNotUsed(evalType, valMin, valMax, mean):
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
            a=[-99, -80, -60, -20, 20, 60, 80, 99]
        else:
            a=[-100, -80, -60, -20, 20, 60, 80, 100] #75, 95, 105, 200,
        cmap = ListedColormap(["#d7191c00","#d7191c","#f17c4a","#fec980", "#ffffbf", "#c7e9ad", "#80bfac","#2b83ba"])
        norm = colors.BoundaryNorm(a, bins)  
        # Add a legend for labels
        legend_labels = { "#d7191c": "<{0}".format(int(a[1])), "#f17c4a": "{0}-{1}".format(int(a[1]), int(a[2])), "#fec980": "{0}-{1}".format(int(a[2]),int(a[3])), 
        "#ffffbf": "{0}-{1}".format(int(a[3]),int(a[4])),
        "#c7e9ad": "{0}-{1}".format(int(a[4]),int(a[5])), "#80bfac": "{0}-{1}".format(int(a[5]),int(a[6])), 
        "#2b83ba": ">{0}".format(int(a[6]),valMax), "#2b83ba00": "mean total:{}".format(mean)}
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
    elif evalType == "diss":
        # Light blue (vivid sky blue) to flickr pink 
        bins = 7
        colors_palette = ["#4CC9F0","#4361EE","#3A0CA3", "#560BAD", "#7209B7", "#B5179E","#F72585"]
        a=[0.00, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14] #a=[valMin,-15, -10, -5, 5, 10, 15,valMax]
        print(a)
        cmap = ListedColormap(colors_palette)
        norm = colors.BoundaryNorm(a, bins)  
        # Add a legend for labels
        
        legend_labels = { colors_palette[0]: "{0}-{1}".format(a[0], a[1]), colors_palette[1]: "{0}-{1}".format(a[1], a[2]), 
        colors_palette[2]: "{0}-{1}".format(a[2],a[3]), colors_palette[3]: "{0}-{1}".format(a[3],a[4]),
        colors_palette[4]: "{0}-{1}".format(a[4],a[5]), colors_palette[5]: "{0}-{1}".format(a[5],a[6]), 
        colors_palette[6]: "{0}-{1}".format(a[6],valMax)}
    elif evalType == "iso":
        # Blue yo red palette
        bins = 7
        colors_palette = ["#277da1", "#43aa8b", "#90be6d", "#f9c74f", "#f9c74f", "#f3722c","#f94144"]
        if valMax < 1.20: 
            a=[0, 0.02, 0.04, 0.06, 0.08, 1.00, 1.20, 1.50]
            cmap = ListedColormap(colors_palette)
            norm = colors.BoundaryNorm(a, bins)  
            # Add a legend for labels
            legend_labels = { colors_palette[0]: "{0}-{1}".format(a[0],a[1]),colors_palette[1]: "{0}-{1}".format(a[1], a[2]), 
            colors_palette[2]: "{0}-{1}".format(a[2],a[3]), colors_palette[3]: "{0}-{1}".format(a[3],a[4]),
            colors_palette[4]: "{0}-{1}".format(a[4],a[5]), colors_palette[5]: "{0}-{1}".format(a[5],a[6]), 
            colors_palette[6]: "{0}-{1}".format(a[6],a[7])}
        else:
            a=[0, 0.02, 0.04, 0.06, 0.08, 1.00, 1.20, 1.50]
            cmap = ListedColormap(colors_palette)
            norm = colors.BoundaryNorm(a, bins)  
            # Add a legend for labels
            legend_labels = { colors_palette[0]: "{0}-{1}".format(a[0],a[1]),colors_palette[1]: "{0}-{1}".format(a[1], a[2]), 
            colors_palette[2]: "{0}-{1}".format(a[2],a[3]), colors_palette[3]: "{0}-{1}".format(a[3],a[4]),
            colors_palette[4]: "{0}-{1}".format(a[4],a[5]), colors_palette[5]: "{0}-{1}".format(a[5],a[6]), 
            colors_palette[6]: "{0}-{1}".format(a[6],valMax)}
    elif evalType == "aaa":
        # skobeloff to palatinate purple
        colors_palette = ["#006466", "#065A60", "#144552", "#1B3A4B", "#272640", "#3E1F47","#4D194D"]
        bins = 7
        a=[0, 1, 5, 10, 25, 50, 75, 100]
        cmap = ListedColormap(colors_palette)
        norm = colors.BoundaryNorm(a, bins)  
        # Add a legend for labels
        legend_labels = { colors_palette[0]: "{0}-{1}".format(a[0],a[1]), colors_palette[1]: "{0}-{1}".format(a[1], a[2]), 
        colors_palette[2]: "{0}-{1}".format(a[2],a[3]), colors_palette[3]: "{0}-{1}".format(a[3],a[4]),
        colors_palette[4]: "{0}-{1}".format(a[4],a[5]), colors_palette[5]: "{0}-{1}".format(a[5],a[6]), 
        colors_palette[6]: "{0}-{1}".format(a[6],a[7])}
    elif evalType == "bbb":
        # nyanza (light green) to dark jungle green
        colors_palette = ["#D8F3DC", "#95D5B2", "#74C69D", "#52B788", "#2D6A4F", "#1B4332", "#081C15"]
        bins = 7
        a=[0, 0.02, 0.04, 0.06, 0.08, 1.00, 1.20, valMax]
        cmap = ListedColormap(colors_palette)
        norm = colors.BoundaryNorm(a, bins)  
        # Add a legend for labels
        legend_labels = { colors_palette[0]: "{0}-{1}".format(a[0],a[1]), colors_palette[1]: "{0}-{1}".format(a[1], a[2]), 
        colors_palette[2]: "{0}-{1}".format(a[2],a[3]), colors_palette[3]: "{0}-{1}".format(a[3],a[4]),
        colors_palette[4]: "{0}-{1}".format(a[4],a[5]), colors_palette[5]: "{0}-{1}".format(a[5],a[6]), 
        colors_palette[6]: "{0}-{1}".format(a[6],valMax)}
    elif evalType == "rdi":
        # Light green to purple 
        bins = 7
        colors_palette = ["#13f644","#2ccf63","#45a782", "#5e80a2", "#7759c1", "#9031e0","#a90aff"]
        a=[0, 0.10, 0.25, 0.5, 0.75, 0.90, 1.00,  valMax] #a=[valMin,-15, -10, -5, 5, 10, 15,valMax]
        print(a)
        cmap = ListedColormap(colors_palette)
        norm = colors.BoundaryNorm(a, bins)  
        # Add a legend for labels
        
        legend_labels = { colors_palette[0]: "{0}-{1}".format(a[0], a[1]), colors_palette[1]: "{0}-{1}".format(a[1], a[2]), 
        colors_palette[2]: "{0}-{1}".format(a[2],a[3]), colors_palette[3]: "{0}-{1}".format(a[3],a[4]),
        colors_palette[4]: "{0}-{1}".format(a[4],a[5]), colors_palette[5]: "{0}-{1}".format(a[5],a[6]), 
        colors_palette[6]: "{0}-{1}".format(a[6],a[7])}
    return cmap, norm, legend_labels

def defineBinsRaster(evalType, valMin, valMax, mean): #
    if evalType == "popdistribution":
        bins = 7
        colors_palette = ["#f1eef600","#ffffb2","#ffd76d", "#fea649", "#f86c30", "#e62f21", "#bd0026"] #graduate of orange
        if valMax < 200:
            a=[valMin, 1, 25, 50, 75, 100, 150, 200]
        else:
            a=[valMin, 1, 25, 50, 75, 100, 150, valMax]
        cmap = ListedColormap(colors_palette)
        norm = colors.BoundaryNorm(a, bins) 
          
        # Add a legend for labels
        legend_labels = { colors_palette[1]: "{0}-{1}".format(a[1], a[2]), 
        colors_palette[2]: "{0}-{1}".format(a[2],a[3]), colors_palette[3]: "{0}-{1}".format(a[3],a[4]),
        colors_palette[4]: "{0}-{1}".format(a[4],a[5]), colors_palette[5]: "{0}-{1}".format(a[5],a[6]), 
        colors_palette[6]: "{0}-{1}".format(a[6],a[7])}
    elif evalType == "popdistributionPred":
        bins = 7
        colors_palette = ["#f1eef600","#ffffb2","#ffd76d", "#fea649", "#f86c30", "#e62f21","#bd0026"] #graduate of orange
        if valMax < 250:
            a=[valMin,1, 25, 50, 75, 100, 150, 250]
        else:
            a=[valMin, 1, 50, 100, 200, 300, 400, 500]
            #a=[valMin,1, 25, 50, 75, 100, 150, valMax]
        cmap = ListedColormap(colors_palette)
        norm = colors.BoundaryNorm(a, bins)  
        # Add a legend for labels
        legend_labels = { colors_palette[1]: "{0}-{1}".format(a[1], a[2]), 
        colors_palette[2]: "{0}-{1}".format(a[2],a[3]), colors_palette[3]: "{0}-{1}".format(a[3],a[4]),
        colors_palette[4]: "{0}-{1}".format(a[4],a[5]), colors_palette[5]: "{0}-{1}".format(a[5],a[6]), 
        colors_palette[6]: "{0}-{1}".format(a[6],a[7])}
    elif evalType == "popdistributionPolyg":
        bins = 7
        if valMin <= valMax <= 7500:
            a=[valMin, 1, 500, 1500, 3000, 4500, 6000, 7500]
        else:
            a=[valMin, 1, 500, 1500, 3000, 4500, 6000, valMax]
        cmap = ListedColormap(["#f1eef600","#e0c8e2","#da9acb", "#df65b0", "#de348a", "#c61266","#980043"])
        norm = colors.BoundaryNorm(a, bins)  
        # Add a legend for labels
        legend_labels = { "#e0c8e2": "{0}-{1}".format(int(a[1]), int(a[2])), "#da9acb": "{0}-{1}".format(int(a[2]),int(a[3])), 
        "#df65b0": "{0}-{1}".format(int(a[3]),int(a[4])),
        "#de348a": "{0}-{1}".format(int(a[4]),int(a[5])), "#c61266": "{0}-{1}".format(int(a[5]),int(a[6])), 
        "#980043": "{0}-{1}".format(int(a[6]),valMax), "#2b83ba00": "mean total:{}".format(mean)}
    elif evalType == "mae":
        bins = 7
        colors_palette = ["#d7191c","#f17c4a","#fec980", "#ffffbf", "#c7e9ad", "#80bfac","#2b83ba"]
        if valMin <= valMax <= 30:
            a=[valMin,-15, -10, -5, 5, 10, 15, 30]
        elif -30<=valMin<=30:
            a=[-30,-15, -10, -3, 3, 10, 15, 30]
        else:
            a=[valMin,-15, -10, -3, 3, 10, 15, valMax]
        cmap = ListedColormap(colors_palette)
        norm = colors.BoundaryNorm(a, bins)  
        # Add a legend for labels
        legend_labels = { colors_palette[0]: "{0}-{1}".format(a[0],a[1]), colors_palette[1]: "{0}-{1}".format(a[1], a[2]), 
        colors_palette[2]: "{0}-{1}".format(a[2],a[3]), colors_palette[3]: "{0}-{1}".format(a[3],a[4]),
        colors_palette[4]: "{0}-{1}".format(a[4],a[5]), colors_palette[5]: "{0}-{1}".format(a[5],a[6]), 
        colors_palette[6]: "{0}-{1}".format(a[6],a[7])}
         
    elif evalType == "pe":
        bins = 7
        colors_palette = ["#d7191c00","#fff5f0", "#fdccb8", "#fc8f6f", "#f44d37","#c5161b","#67000d"]
        a=[0, 1, 20, 40, 60, 80, 100]
        cmap = ListedColormap(colors_palette)
        norm = colors.BoundaryNorm(a, bins)  
        # Add a legend for labels
        legend_labels = { colors_palette[0]: "{0}-{1}".format(a[0],a[1]), colors_palette[1]: "{0}-{1}".format(a[1], a[2]), 
        colors_palette[2]: "{0}-{1}".format(a[2],a[3]), colors_palette[3]: "{0}-{1}".format(a[3],a[4]),
        colors_palette[4]: "{0}-{1}".format(a[4],a[5]), colors_palette[5]: "{0}-{1}".format(a[5],a[6]), 
        colors_palette[6]: "{0}<".format(a[6])}
    
    elif evalType == "dif_KNN":
        # Light green to purple 
        print('----', valMin, valMax)
        bins = 7
        colors_palette = ["#386641","#6a994e","#a7c957", "#f2e8cf50", "#ff6f59", "#db504a","#bc4749"]
        a=[-8, -5, -3, -1, 1, 3, 5, 8] #a=[valMin,-15, -10, -5, 5, 10, 15,valMax]
        print(a)
        cmap = ListedColormap(colors_palette)
        norm = colors.BoundaryNorm(a, bins)  
        # Add a legend for labels
        legend_labels = { colors_palette[0]: "{0}-{1}".format(a[0], a[1]), colors_palette[1]: "{0}-{1}".format(a[1], a[2]), 
        colors_palette[2]: "{0}-{1}".format(a[2],a[3]), colors_palette[3]: "{0}-{1}".format(a[3],a[4]),
        colors_palette[4]: "{0}-{1}".format(a[4],a[5]), colors_palette[5]: "{0}-{1}".format(a[5],a[6]), 
        colors_palette[6]: "{0}-{1}".format(a[6],a[7])}
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