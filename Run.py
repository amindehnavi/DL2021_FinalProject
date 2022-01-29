from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import argparse
import os
import time
from loguru import logger

import cv2

import torch


def vis2(img, outs, predicted_depth):

    font = cv2.FONT_HERSHEY_SIMPLEX
    depth = np.copy(predicted_depth[0][0])
    for i in range(len(outs[0][0])):

        score = outs[0][0][i][4]*outs[0][0][i][5]
        if score < 0.35:
            continue
        bbox = outs[0][0][i][0:4].numpy()
        x1, y1, x2, y2 = bbox
        #dis = np.sum(predicted_depth[0,0,int(x1):int(x2),int(y1):int(y2)])/((x2-x1)*(y2-y1))
        dis = np.median(
            predicted_depth[0, 0, int(y1):int(y2), int(x1):int(x2)])
        # print(predicted_depth[0,0,int(y1):int(y2),int(x1):int(x2)])
        text = '{0:.4f}'.format(dis)
        cv2.rectangle(depth, (x1, y1), (x2, y2), (0, 0, 0), 2)
        # cv2.rectangle
        cv2.putText(depth, text, (int(x1), int(y1)),
                    font, 1, color=(0, 0, 0), thickness=2)
    plt.imshow(depth, cmap='jet')
    plt.show()


def run(img_path):

    os.chdir('/content/drive/MyDrive/DL_2021/FinalProject/AdaBins')
    from AdaBins.infer import InferenceHelper
    # Create a InferenceHelper object
    infer_helper = InferenceHelper(dataset='nyu', device='cpu')
    # predict depth of a single pillow image
    img = Image.open(img_path)  # any rgb pillow image
    bin_centers, predicted_depth = infer_helper.predict_pil(img)

    os.chdir('/content/drive/MyDrive/DL_2021/FinalProject/YOLOX')
    from tools.demo2 import predict_beta
    # object Detection Prediction
    outs, infos = predict_beta(path=img_path)

    os.chdir('/content/drive/MyDrive/DL_2021/FinalProject')

    vis2(img, outs, predicted_depth)
