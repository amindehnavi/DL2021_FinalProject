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

DepthThreshold = 4

Image = Visual.visualize(RawImage, Depth,
                         YOLO_Out, DepthThreshold, Img_Info, Conf=0.3, Depth_Check=False,
                         BoundingBox_Check=True,
                         DepthInfo_Check=True)

cv2.imwrite('/content/out.jpg',Image)

'''
out = YOLO_Out[0].numpy()
print('YOLO_Out',out[:,6])
font = cv2.FONT_HERSHEY_SIMPLEX

for i in range(len(YOLO_Out[0])):
  score = YOLO_Out[0][i][4]*YOLO_Out[0][i][5]
  if score < 0.35:
    continue
  bbox = YOLO_Out[0][i][0:4].numpy()
  x1, y1, x2, y2 = bbox
  dis = np.median(Depth[int(y1):int(y2),int(x1):int(x2)])
  text = '{0:.4f}'.format(dis)
  cv2.rectangle(Depth,(x1,y1),(x2,y2),(0,0,0),2)
  cv2.putText(Depth, text, (int(x1), int(y1)),font,1,color=(0, 0, 0), thickness=1)
  cv2.imwrite('/content/out.jpg',Depth)
'''


