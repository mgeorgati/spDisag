import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import rasterio, pandas as pd, seaborn as sns, numpy as np
from pathlib import Path

def plotMatrix(fileList, exportPath, title):
    rows = 2
    if len(fileList) % rows==0:
        cols = int(len(fileList)/rows)
        print(cols)
    else:
        cols = int(round(len(fileList)/rows) )
        print(cols)   
    fig = plt.figure(figsize=(20, 10))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(rows, cols),  # creates 2x2 grid of axes
                    axes_pad=0.3,  # pad between axes in inch.
                    )

    for ax, im in zip(grid, fileList):
        image = plt.imread(im)
        # Iterating over the grid returns the Axes.
        ax.imshow(image)
        ax.set_axis_off()
        # Add a title
    #ax.set_title(title, color="black", fontsize=18)
        plt.axis('off')
    ax.set_axis_off()
    plt.axis('off')
    print("-----PLOTTING IMAGE {}-----".format(title))
    plt.savefig(exportPath, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(),transparent=True)
    
    plt.cla()
    plt.close()


