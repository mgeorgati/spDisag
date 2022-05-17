import numpy as np
import itertools

num_targets = 12
width = 16
height = 16
stride = 1
image_size = ( 20, 20 )
num_patches = ( image_size[0] - width + stride ) * ( image_size[1] - height + stride )
patches = np.ones( ( num_patches, width, height, num_targets ) )

def reconstructpatches(patches, image_size, stride):    
    if len(patches.shape) == 4:
      aux = [ reconstructpatches(patches[:,:,:,a], image_size, stride) for a in range(patches.shape[3]) ]
      return np.moveaxis( np.array(aux), 0, 2)
    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    mean = np.zeros(image_size)
    patch_count = np.zeros(image_size)
    n_h = int((i_h - p_h) / stride + 1)
    n_w = int((i_w - p_w) / stride + 1)
    for p, (i, j) in zip(patches, itertools.product(range(n_h), range(n_w))):
        patch_count[i * stride:i * stride + p_h, j * stride:j * stride + p_w] += ~np.isnan(p)
        ctignore = np.isnan(p)
        p[ctignore] = 0
        mean[i * stride:i * stride + p_h, j * stride:j * stride + p_w] += p
        p[ctignore] = np.nan
    mean = np.divide(mean, patch_count, out=np.zeros_like(mean), where=patch_count != 0)
    return mean

aux = reconstructpatches( patches, image_size, stride )
print(aux.shape)