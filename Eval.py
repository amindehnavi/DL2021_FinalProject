import numpy as np

# Compute Metrics for Join Networks

# For Single Image


def JointNets(det_pred, det_gt, iou_thr, dep_pred, dep_gt, dep_thr, AP_det):

    depth_recall, depth_precision = TruePR(dep_pred, dep_gt, dep_thr)
    # det_recall, det_precision = ObjectDetectionMetrics(
    #    det_pred, det_gt, iou_thr)

    #recalls = depth_recall * det_recall
    #precisions = depth_precision * det_precision

    # Calculate the F1-Score
    #F1 = 2 * (recalls * precisions) / (recalls + precisions)

    AP_joint = AP_det * depth_recall * depth_precision

    return AP_joint

# Depth Estimation Metrics  #####################################################


def DepthPR(pred, gt, threshold):

    max_depth = np.maximum(pred.max(), gt.max())
    print('pre_depth', max_depth)

    thresh = np.maximum((gt / pred), (pred / gt))
    delta2 = (thresh < 1.25 ** 2).mean()

    #ar = (1-2/3*delta2)/np.log(max_depth)
    #recall = ar*np.log(threshold)+(2/3)*delta2

    #ar = (0.9-2/3*delta2)/(np.exp(max_depth)-np.exp(1))
    #br = 2/3*delta2 - ar + 0.1
    #recall = ar*np.exp(threshold)+br

    #ap = 1/((max_depth/2)**2) - (2/(3*(max_depth/2)**2))*delta2
    #bp = (4/(3*(max_depth/2)))*delta2 - 2/(max_depth/2)
    # precision = ap*(threshold**2) + \
    #    bp*threshold + 1

    ar = (0.9-2/3*delta2)/np.log(max_depth)
    br = 2/3*delta2 + 0.1
    recall = ar*np.log(threshold)+br

    ap = (0.9-2/3*delta2)/np.log(max_depth)
    bp = 2/3*delta2 + 0.1
    precision = ap*np.log(threshold)+bp

    return recall, precision


def TruePR(pred, gt, threshold):

    p = pred <= threshold
    g = gt <= threshold

    TP = np.sum(p*g)
    FP = np.sum(p*(1-g))
    FN = np.sum((1-p)*g)
    TN = np.sum((1-p)*(1-g))

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)

    print('TP:', TP)
    print('FP:', FP)
    print('FN:', FN)
    print('TN:', TN)

    return recall, precision


def compute_depth_metrics(gt, pred):

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)


# Object Detection Metrics #######################################################

def ObjectDetectionMetrics(YOLO_Out, GTs, IoUThreshold=0.5):

    Output = YOLO_Out[0].numpy()

    pred_boxes = []
    gt_boxes = []
    for i in range(len(Output)):

        Bboxes = Output[i, 0:4]
        pred_boxes.append(Bboxes)

    for i in range(len(GTs)):

        Bboxes = GTs[i, 0:4]
        gt_boxes.append(Bboxes)

    tp, fp, fn = get_single_image_results(
        gt_boxes, pred_boxes, iou_thr=IoUThreshold)

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    return precision, recall


def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """
    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = 0
        return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}
    if len(all_gt_indices) == 0:
        tp = 0
        fp = 0
        fn = 0
        return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = calc_iou(gt_box, pred_box)

            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)
    iou_sort = np.argsort(ious)[::1]
    if len(iou_sort) == 0:
        tp = 0
        fp = 0
        fn = 0
        return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in iou_sort:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if(gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)
    return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}


def calc_iou(gt_bbox, pred_bbox):
    '''
    This function takes the predicted bounding box and ground truth bounding box and 
    return the IoU ratio
    '''
    x_topleft_gt, y_topleft_gt, x_bottomright_gt, y_bottomright_gt = gt_bbox
    x_topleft_p, y_topleft_p, x_bottomright_p, y_bottomright_p = pred_bbox

    if (x_topleft_gt > x_bottomright_gt) or (y_topleft_gt > y_bottomright_gt):
        raise AssertionError("Ground Truth Bounding Box is not correct")
    if (x_topleft_p > x_bottomright_p) or (y_topleft_p > y_bottomright_p):
        raise AssertionError("Predicted Bounding Box is not correct",
                             x_topleft_p, x_bottomright_p, y_topleft_p, y_bottomright_gt)

    # if the GT bbox and predcited BBox do not overlap then iou=0
    if(x_bottomright_gt < x_topleft_p):
        # If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox

        return 0.0
    if(y_bottomright_gt < y_topleft_p):  # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox

        return 0.0
    if(x_topleft_gt > x_bottomright_p):  # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox

        return 0.0
    if(y_topleft_gt > y_bottomright_p):  # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox

        return 0.0

    GT_bbox_area = (x_bottomright_gt - x_topleft_gt + 1) * \
        (y_bottomright_gt - y_topleft_gt + 1)
    Pred_bbox_area = (x_bottomright_p - x_topleft_p + 1) * \
        (y_bottomright_p - y_topleft_p + 1)

    x_top_left = np.max([x_topleft_gt, x_topleft_p])
    y_top_left = np.max([y_topleft_gt, y_topleft_p])
    x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
    y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])

    intersection_area = (x_bottom_right - x_top_left + 1) * \
        (y_bottom_right-y_top_left + 1)

    union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)

    return intersection_area/union_area
