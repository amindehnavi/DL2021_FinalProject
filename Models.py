from PIL import Image
import numpy as np
from loguru import logger
import cv2
from AdaBins.infer import InferenceHelper

def LoadAdabins():
    return InferenceHelper(dataset='nyu', device='cpu')

def PredictDepth(Model, Image):
    # predict depth of a single pillow image
    _, predicted_depth = Model.predict_pil(Image)

    return predicted_depth[0,0,:,:]
