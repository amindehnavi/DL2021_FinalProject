import matplotlib
import numpy as np
from YOLOX.yolox.data.datasets import COCO_CLASSES
import cv2


def Bbox_Drawer(img, predicted_depth, boxes, scores, cls_ids, conf=0.5, depth_thr=1, class_names=COCO_CLASSES):

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

        dis = np.median(
            predicted_depth[0, 0, int(y0):int(y1), int(x0):int(x1)])

        if dis > depth_thr:
            continue

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}|S:{:.1f}|D:{:.1f}%'.format(
            class_names[cls_id], score * 100, dis)
        txt_color = (0, 0, 0) if np.mean(
            _COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(
            img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def visualize(RawImage, Depth, YOLO_Out, DepthThreshold, Img_Info, Conf=0.5, Depth_check=False, BoundingBox_check=False, Depth_Info_check=False):

    if Depth_check and ~BoundingBox_check and ~Depth_Info_check:
        NormedDepth = (Depth - np.min(Depth)) / (np.max(Depth) - np.min(Depth))
        CMap = matplotlib.cm.get_cmap('plasma')
        depthMap = CMap(NormedDepth)
        depthMap = depthMap[:, :, 0:3] * 255
        depthMap = depthMap.astype('uint8')
        return depthMap

    else:
        ratio = Img_Info["ratio"]
        if YOLO_Out is None:
            return RawImage
        Output = YOLO_Out.cpu()

        Bboxes = Output[:, 0:4]

        # preprocessing: resize
        Bboxes /= ratio

        Cls = Output[:, 6]
        Scores = Output[:, 4] * Output[:, 5]

        # Draw corresponding bounding boxes on depth map
        if Depth_check and BoundingBox_check and ~Depth_Info_check:
            res_img = Bbox_Drawer(
                Depth, Depth, Bboxes, Scores, Cls, conf=Conf, depth_thr=DepthThreshold)
        elif Depth_Info_check and ~Depth_check:
            res_img = Bbox_Drawer(
                RawImage, Depth, Bboxes, Scores, Cls, conf=Conf, depth_thr=DepthThreshold)
        else:
            res_img = RawImage

        return res_img


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
