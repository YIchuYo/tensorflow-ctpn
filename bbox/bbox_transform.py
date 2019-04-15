import numpy as np

def bbox_transform(ex_rois, gt_rois):
    """
    计算gt框到anchor的距离，并归一化尺寸
    :param ex_rois: n * 4 numpy array
    :param gt_rois: n * 4 numpy array
    :return: deltas: n * 4 numpy array (偏移修正)
    """

    # 计算ex的heights和widths,中心点x,y
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1
    ex_center_xs = ex_rois[:, 0] + 0.5 * ex_widths
    ex_center_ys = ex_rois[:, 1] + 0.5 * ex_heights

    # 计算gt的heights和widths,中心点x,y
    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1
    gt_center_xs = gt_rois[:, 0] + 0.5 * gt_widths
    gt_center_ys = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_center_xs - ex_center_xs) / ex_widths
    targets_dy = (gt_center_ys - ex_center_ys) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)).transpose()

    return targets

def bbox_transform_inv(boxes, deltas):
    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4] # 这里解释一下，dw, dh都是通过预测,对anchor中心y和h的偏移量
                         # 训练中使用，gt和anchor的偏移量作为标签
    # 多增加一个维度，将数据恢复了列的结构
    pred_ctr_x = ctr_x[:, np.newaxis] # 不关注
    pred_str_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis] # Vc=(Cy-Cy')/h' 逆过程 Cy=h'*Vc+Cy'
    pred_w = widths[:, np.newaxis]    # 不关注
    pred_h = np.exp(dh) * heights[:, np.newaxis] # Vh = log(h/h') 逆过程 h=exp(Vh)*h'


    # ready! fill the boxes
    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)

    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1::4] = pred_str_y - 0.5 * pred_h
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3::4] = pred_str_y + 0.5 * pred_h

    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    将所有的Boxes进行修正，有部分在图片外的anchor，使其约束在图片内
    :param boxes:
    :param im_shape:(height, width)
    :return:
    """
    # x1>=0 and x1<width
    boxes[:,0::4] = np.maximum(np.minimum(boxes[:,0::4], im_shape[1] -1 ), 0)
    # y1>=0 and y1<height
    boxes[:,1::4] = np.maximum(np.minimum(boxes[:,1::4], im_shape[0] -1 ), 0)
    # x2>=0 and x2<width
    boxes[:,2::4] = np.maximum(np.minimum(boxes[:,2::4], im_shape[1] -1 ), 0)
    # y2>=0 and y2<height
    boxes[:,3::4] = np.maximum(np.minimum(boxes[:,3::4], im_shape[0] -1 ), 0)

    return boxes