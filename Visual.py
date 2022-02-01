import matplotlib
import numpy as np
from YOLOX.yolox.data.datasets import COCO_CLASSES
import cv2


def Bbox_Drawer(raw_image, img, predicted_depth, boxes, scores, cls_ids, conf=0.5, depth_thr=1, class_names=COCO_CLASSES):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        mask = Mask(raw_image[int(y0):int(y1), int(x0):int(x1)])/255
        dis1 = ( np.sum(predicted_depth[int(y0):int(y1), int(x0):int(x1)]*mask) )/np.sum(mask)
        mask = -1*(mask-1)
        dis2 = ( np.sum(predicted_depth[int(y0):int(y1), int(x0):int(x1)]*mask) )/np.sum(mask)
        dis = min(dis1, dis2)

        if dis > depth_thr:
            continue
        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text1 = '{}: {:.1f}%'.format(
            class_names[cls_id], score * 100)
        text2 = 'depth:{:.1f}m'.format(dis)
        txt_color = (0, 0, 0) if np.mean(
        _COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text1, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + 2*int(1.5*txt_size[1]) + 1),
            txt_bk_color,
            -1
        )
        cv2.putText(
            img, text1, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
        cv2.putText(
            img, text2, (x0, y0 + 2*txt_size[1]+6), font, 0.4, txt_color, thickness=1)

    return img

def Depth2Img (Depth):
    NormedDepth = (Depth - np.min(Depth)) / (np.max(Depth) - np.min(Depth))
    CMap = matplotlib.cm.get_cmap('plasma_r')
    CMap = CMap.reversed()
    depthMap = CMap(NormedDepth)
    depthMap = depthMap[:, :, 0:3] * 255
    depthMap = depthMap.astype('uint8')
    return depthMap

def Mask(img):  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def visualize(RawImage, Depth, YOLO_Out, Img_Info, DepthThreshold,  Conf=0.5, RadioButton = 'Raw Image'):

    if RadioButton == 'Depth Image':
        return Depth2Img (Depth)
    
    elif RadioButton == 'Raw Image':
        return RawImage

    else:
        ratio = Img_Info["ratio"]
        if YOLO_Out is None:
            
            if RadioButton == 'Raw Image + Bounding Boxes':
                return RawImage

            elif RadioButton == 'Depth Image + Bounding Boxes':
                return Depth2Img (Depth)

        Output = YOLO_Out[0].numpy()

        Bboxes = Output[:,0:4]

        # preprocessing: resize
        Bboxes /= ratio

        Cls = Output[:,6]
        Scores = Output[:,4] * Output[:,5]

        # Draw corresponding bounding boxes on depth map
        if RadioButton == 'Raw Image + Bounding Boxes':
            res_img = Bbox_Drawer(
                RawImage, RawImage, Depth, Bboxes, Scores, Cls, conf=Conf, depth_thr=DepthThreshold)
            return res_img

        elif RadioButton == 'Depth Image + Bounding Boxes':
            res_img = Bbox_Drawer(
                RawImage, Depth2Img(Depth), Depth, Bboxes, Scores, Cls, conf=Conf, depth_thr=DepthThreshold)
            return res_img
    
    return RawImage

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
