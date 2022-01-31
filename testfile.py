import sys
#from PIL import Image
import numpy as np
import Models
import Visual
import matplotlib.pyplot as plt
import cv2
import Visual

DepthModel = Models.LoadAdabins()
YOLOModel, YOLOModel_exp = Models.LoadYOLOX()

img_path = '/content/test.jpg'

RawImage = cv2.imread(img_path)

Depth = Models.PredictDepth(DepthModel, RawImage)
YOLO_Out, Img_Info = Models.PredictBoundingBox(
    YOLOModel, YOLOModel_exp, RawImage, fp16=False, device='cpu')

DepthThreshold = 10

Image = Visual.visualize(RawImage, Depth,
                         YOLO_Out, DepthThreshold, Img_Info, Depth_check=True,
                         BoundingBox_Check=False,
                         DepthInfo_Check=False)
plt.imshow(Image)
plt.show()
