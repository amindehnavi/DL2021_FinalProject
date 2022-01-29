from AdaBins.infer import InferenceHelper
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import argparse
import os
import time
from loguru import logger

import cv2

import torch

from YOLOX.yolox.data.data_augment import ValTransform
from YOLOX.yolox.data.datasets import COCO_CLASSES
from YOLOX.yolox.exp import get_exp
from YOLOX.yolox.utils import fuse_model, get_model_info, postprocess, vis
from YOLOX.yolox.tools.demo2 import predict_beta


def vis2(img, outs, predicted_depth):

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(len(outs[0][0])):
        depth = np.copy(predicted_depth[0][0])
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

    # Create a InferenceHelper object
    infer_helper = InferenceHelper()
    # predict depth of a single pillow image
    img = Image.open(img_path)  # any rgb pillow image
    bin_centers, predicted_depth = infer_helper.predict_pil(img)

    # object Detection Prediction
    outs, infos = predict_beta(img_path)

    vis2(img, outs, predicted_depth)
