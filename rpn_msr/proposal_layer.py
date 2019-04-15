
import numpy as np
from bbox.nms import nms
from bbox.bbox_transform import bbox_transform_inv, clip_boxes
from rpn_msr.config import Config as cfg
from rpn_msr.generate_anchors import generate_anchors

DEBUG = False

def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, _feat_stride=[16,], anchor_sales=[16,]):
    """

    :param rpn_cls_prob_reshape: (1, H, W×A, 2) outputs of RPN, prob of bg or fg
    :param rpn_bbox_pred: (1, H, W, A×4), rgs boxes output of RPN
    :param im_info: a list of [image_height, image_width, scale_ratios]
            actually, we just input a image, so the len(im_info) = 1
    :param _feat_stride:
    :param anchor_sales:
    :return: rpn_rois: (1×H×W×A,5) e.g. [0,x1,y1,x2,y2]
    """
    # Algorithm:
    # for each (H, W) location i
    #   generate A anchor boxes deltas
    #   apply predicted bbox deltas at cell i to each of the A anchors
    # clip predicted boxes to image
    # remove predicted boxes with either height or width < threshold
    # 从高到低对所有的(proposal, score)对按score进行排序
    # 选取最高几个proposals
    # 对剩下的proposals应用NMS(threshold 0.7)
    # 选取最高几个proposals
    # return 最高的几个proposals

    _anchors = generate_anchors() # gen 10 basic anchors(左上右下表示法)
    _num_anchors = _anchors.shape[0] # 10

    im_info = im_info[0] # 原始图像的高宽和缩放尺度
    bbox_deltas = rpn_bbox_pred # 模型输出的pred是相对值，从全连接层就可以看出来啦，输出在[-1,1]之间

    assert rpn_cls_prob_reshape.shape[0] == 1, "The input of proposal layer must be one image"

    pre_nms_topN = cfg.RPN_PRE_NMS_TOP_N
    post_nms_topN = cfg.RPN_POST_NMS_TOP_N
    nms_threshold = cfg.RPN_NMS_THRESH
    box_min_size = cfg.RPN_BOX_MIN_SIZE

    height, width = rpn_bbox_pred.shape[1:3]

    # 提取object score, no object score不关心
    # => (1, H, W, A)
    # print(rpn_cls_prob_reshape)
    scores = np.reshape(np.reshape(rpn_cls_prob_reshape, [1, height, width, _num_anchors, 2])[:, :, :, :, 1],\
                        [1, height, width, _num_anchors])

    # 1. generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    # 生成所有anchors
    A = _num_anchors
    K = shifts.shape[0]
    # =>(K,A,4) K=H*W
    anchors = _anchors.reshape((1,A,4)) + shifts.reshape((1,K,4)).transpose((1,0,2))
    # =>(K*A, 4), 得到所有的anchors
    anchors = anchors.reshape((K*A, 4))

    # 根据rpn_bbox_pred和anchors得到真正的anchors！
    #   bbox deltas: (1, H, W, 4*A)=>(H*W*A, 4)
    bbox_deltas = bbox_deltas.reshape((-1,4))

    # 新鲜出炉的调整好的候选框（不是偏移量，而是正确的坐标值哦）
    proposals = bbox_transform_inv(anchors, bbox_deltas)

    # 2. 修正所有的proposal box，变换到图片内部
    proposals = clip_boxes(proposals, im_info[:2])

    scores = scores.reshape((-1,1))
    # 3. 移除proposal box中 height或width小于阈值的
    keep = remove_boxes_smaller_threshold(proposals, box_min_size)
    proposals = proposals[keep,:]
    scores = scores[keep]
    bbox_deltas = bbox_deltas[keep,:]

    # 4. 排序
    order = scores.ravel().argsort()[::-1] # 由高到低排序
    # 5. 取12000个proposals
    if pre_nms_topN > 0: # 保留12000个proposals
        order = order[:pre_nms_topN]
    proposals = proposals[order,:]
    scores = scores[order]
    bbox_deltas = bbox_deltas[order,:]

    # 6. 对剩下的proposals应用NMS(threshold 0.7)
    keep = nms(np.hstack((proposals, scores)), nms_threshold) # 进行nms操作，保留2000个proposal

    if post_nms_topN>0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep,:]
    scores = scores[keep]
    bbox_deltas = bbox_deltas[keep,:]

    blob = np.hstack((scores.astype(np.float32, copy=False), proposals.astype(np.float32, copy=False)))
    return blob, bbox_deltas





def remove_boxes_smaller_threshold(boxes, box_min_size):
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws>=box_min_size)&(hs>box_min_size))[0]
    return keep