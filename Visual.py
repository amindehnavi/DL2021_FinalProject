import matplotlib
import numpy as np

def visualize(RawImage, Depth, BoundingBox, DepthThreshold):
    NormedDepth = (Depth - np.min(Depth)) / (np.max(Depth) - np.min(Depth))
    CMap = matplotlib.cm.get_cmap('plasma')
    depthMap = CMap(NormedDepth)
    depthMap = depthMap[:,:,0:3] * 255
    depthMap = depthMap.astype('uint8')
    return depthMap
