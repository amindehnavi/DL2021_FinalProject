from PIL import Image
import numpy as np
from loguru import logger
import cv2
from AdaBins.infer import InferenceHelper
from YOLOX.tools.demo2 import predict_beta
from YOLOX.yolox.exp import get_exp
from YOLOX.yolox.utils import get_model_info
import torch


def LoadAdabins():
    return InferenceHelper(dataset='nyu', device='cpu')


def PredictDepth(Model, Image):
    # predict depth of a single pillow image
    _, predicted_depth = Model.predict_pil(Image)

    return predicted_depth[0, 0, :, :]


def LoadYOLOX(ckpt_path='./YOLOX/Pretrained/yolox_m.pth'):
    exp = get_exp(None, 'yolox-m')
    model = exp.get_model()
    logger.info("Model Summary: {}".format(
        get_model_info(model, exp.test_size)))

    model.eval()
    ckpt_file = ckpt_path
    logger.info("loading YOLOX model checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info("Loading the YOLOX-m model is done!")

    return model, exp


def PredictBoundingBox(model, exp, img, fp16=False, device='cpu'):
    return predict_beta(model, exp, img, fp16=fp16, device=device)
