import matplotlib
import numpy as np
import pycocotools
def DepthMap(depth):
    NormedDepth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    CMap = matplotlib.cm.get_cmap('plasma')
    depthMap = CMap(NormedDepth)
    depthMap = depthMap[:,:,0:3] * 255
    depthMap = depthMap.astype('uint8')
    return depthMap