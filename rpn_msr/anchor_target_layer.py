import os
import sys

import numpy as np
import numpy.random as nprandom
# sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))
sys.path.append(os.path.join(os.path.abspath(os.getcwd()), 'ctpn'))
# 注意模块和包的区别，from对应于包
from bbox.bbox import bbox_overlaps
from bbox.bbox_transform import bbox_transform

from rpn_msr.generate_anchors import generate_anchors
from rpn_msr.config import Config as cfg

DEBUG = False

def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride=[16], anchor_scales=[16]):
    """
    由GT的框，生成多个anchors。依次来作为GT真实的anchors
    :param rpn_cls_score: (1, H, W, A×2)
    :param gt_boxes: (G, 5)
    :param im_info: [image_height, image_width, scale_ratios]
    :param _feat_stride: 下采样比例（从特征图放大到原图）
    :param anchor_scales: basic anchor的尺度范围
    :return:
    rpn_labels: (HxWxA, 1), 对于每个anchor 0表示bg 1表示fg -1表示dont care
    rpn_bbox_targets: (HxWxA, 4), anchor和gt_boxes之间的距离，are the regression objectives
    rpn_bbox_inside_weights: (HxWxA, 4) 每个boxes的权重
    rpn_bbox_outside_weights: (HxWxA, 4) 用于平衡fg和bg的数量，显然，bg更多啦
    """

    # 之后可以进行丰富，现在做的就直接10个基本anchors，尺寸，height范围都是使用默认参数。
    _anchors = generate_anchors() # gen 10 basic anchors(左上右下表示法)
    _num_anchors = _anchors.shape[0] # 10

    im_info = im_info[0]
    # print("im_info: ", im_info)

    # 在feature-map上定位anchor
    # Alg:
    # for each (H, W) location i
    #  generate 9 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 9 anchors
    # filter out-of-image anchors
    # measure GT overlap

    assert rpn_cls_score.shape[0] == 1, "The input of anchor_target_layer must be one image " # 只支持一张图片哦

    height, width = rpn_cls_score.shape[1:3]

    # 1. gen proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    # 生成坐标矩阵
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # gen K is H * W, 4
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                       shift_x.ravel(), shift_y.ravel())).transpose()
    A = _num_anchors

    # anchors=(10,4)
    # shifts = (H*W,4)
    # -> (H*W*A, 4)
    K = shifts.shape[0] # H*W
    all_shifted_anchors = (_anchors.reshape((1,A,4))+
                            shifts.reshape((1,K,4)).transpose((1,0,2)))
    # [√]
    all_shifted_anchors = all_shifted_anchors.reshape((K*A, 4))
    _num_total_anchors = K*A


    # remove out of bounds
    keep_indexs = np.where(
        (all_shifted_anchors[:, 0] >= 0) &
        (all_shifted_anchors[:, 2] <= im_info[1]) &
        (all_shifted_anchors[:, 1] >= 0) &
        (all_shifted_anchors[:, 3] <= im_info[0])
    )[0]

    # 保留的anchors都是正常在图片的中anchor
    keep_first_anchors = all_shifted_anchors[keep_indexs,:]

    # labels (1 dim)
    labels = np.empty((len(keep_indexs), ), dtype=np.float32)
    labels.fill(-1) # 初始化label, 均为-1

    # overlaps between the anchors and the gt boxes
    # anchors:(H*W*A, 4) gt boxes:(?, 4)
    # 计算出每个anchor和gt_box之间的overlap矩阵
    """
    import numpy as np
    A = np.array([[1,1,2,2], [2,2,3,3],[3,3,4,4]])
    B = np.array([[1.5,1.5,3.5,3.5], [2,2,3,3]])
    bbox_overlaps(np.ascontiguousarray(A, dtype=np.float),np.ascontiguousarray(B, dtype=np.float))
    Out[7]: 
    array([[0.20930233, 0.14285714],
           [0.44444444, 1.        ],
           [0.20930233, 0.14285714]])
    """
    overlaps = bbox_overlaps(
        np.ascontiguousarray(keep_first_anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float)
    )


    # 计算回归目标
    # 给每个anchor一个对应的回归目标，还有计算损失函数时两个权重inside和outside
    box_argmax_overlaps = overlaps.argmax(axis=1) # 每个anchor对应最大overlap的索引
    # 每个anchor所对应的最大overlap值
    box_max_overlaps = overlaps[np.arange(len(keep_indexs)), box_argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0) # 每个gt对应最大的overlap的索引
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                            np.arange(overlaps.shape[1])] # gt个数

    # print('box_max_overlaps:',box_max_overlaps)
    # print('gt_max_overlaps:',gt_max_overlaps)
    # overlap小于阈值的认为是背景
    labels[box_max_overlaps < cfg.RPN_OVERLAP_IS_NEGATIVE_THRESHOLD] = 0
    # gt中overlap最大的对应的anchor认为是前景
    labels[gt_argmax_overlaps] = 1
    # overlap大于阈值的认为是前景（正样本）
    labels[box_max_overlaps >= cfg.RPN_OVERLAP_IS_POSITIVE_THRESHOLD] = 1

    # 如果正样本太多，则对正样本进行采样
    # 限制正样本的数量不超过128个
    num_fg_recommend = cfg.RPN_POSITIVE_SAMPLE_SIZE
    fg_inds = np.where(labels==1)[0]
    if len(fg_inds) > num_fg_recommend:
        # 采样，随机选择移除一部分
        disable_inds = nprandom.choice(
            fg_inds, size=(len(fg_inds) - num_fg_recommend), replace=False)
        labels[disable_inds] = -1

    # 如果负样本太多，则对负样本进行采样
    # 正样本最多128个，总样本数为256个样本
    num_bg_recommend = cfg.RPN_SAMPLE_SIZE - np.sum(labels==1)
    bg_inds = np.where(labels==0)[0]
    if len(bg_inds) > num_bg_recommend:
        # 采样，随机选择移除一部分
        disable_inds = nprandom.choice(
            bg_inds, size=(len(bg_inds) - num_bg_recommend), replace=False)
        labels[disable_inds] = -1
    # print(labels)
    # 以上，所有的label完成标记

    if DEBUG:
        print('anchors:', keep_first_anchors)
        print('gt_box:', gt_boxes)
        print(gt_boxes[box_argmax_overlaps, :])

    # box_argmax_overlaps得到的是每一个anchor对应的gtbox向量
    # 所以keep_first_anchors和gt_boxes[box_argmax_overlaps, :] shape[0]相同
    # [无实际意义] bbox_targets = np.zeros((len(keep_indexs), 4), dtype=np.float32)
    bbox_targets = _compute_targets(keep_first_anchors, gt_boxes[box_argmax_overlaps, :])

    bbox_inside_weights = np.zeros((len(keep_indexs), 4), dtype=np.float32)
    bbox_inside_weights[labels==1, :] = np.array(cfg.RPN_BBOX_INSIDE_WEIGHTS) # 内部权重，前景赋值为1

    bbox_outside_weights = np.zeros((len(keep_indexs), 4), dtype=np.float32)
    # 使正样本为1，负样本为0
    positive_weights = np.ones((1,4))
    negative_weights = np.zeros((1,4))
    bbox_outside_weights[labels==1, :] = positive_weights # 外部权重，前景是1
    bbox_outside_weights[labels==0, :] = negative_weights # 背景是0

    labels = _umap(labels, _num_total_anchors, keep_indexs, fill=-1)
    bbox_targets = _umap(bbox_targets, _num_total_anchors, keep_indexs, fill=0)
    bbox_inside_weights = _umap(bbox_inside_weights, _num_total_anchors, keep_indexs, fill=0)
    bbox_outside_weights = _umap(bbox_outside_weights, _num_total_anchors, keep_indexs, fill=0)

    # labels
    labels = labels.reshape((1, height, width, A))
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets.reshape((1, height, width, A*4))
    rpn_bbox_targets = bbox_targets

    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A*4))
    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, A*4))
    rpn_bbox_outside_weights = bbox_outside_weights

    if DEBUG:
        print('rpn_labels: ',rpn_labels)
        print('rpn_bbox_targets: ',rpn_bbox_targets)
        print('rpn_bbox_inside_weights: ',rpn_bbox_inside_weights)
        print('rpn_bbox_outside_weights: ',rpn_bbox_outside_weights)

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights



def t_overlap():
    import numpy as np
    anchors = np.array([[1,1,2,2], [2.5,2.5,3,3],[3,3,4,4], [5,5,6,6]])
    gt_boxes = np.array([[1.5,1.5,2.5,2.5], [2,2,3,3]])
    overlaps = bbox_overlaps(np.ascontiguousarray(anchors, dtype=np.float),np.ascontiguousarray(gt_boxes, dtype=np.float))
    # print('overlaps:', overlaps)
    keep_indexs= [0,1,2,3]
    # labels (1 dim)
    labels = np.empty((len(keep_indexs), ), dtype=np.float32)
    labels.fill(-1) # 初始化label, 均为-1
    # print(labels)
    box_argmax_overlaps = overlaps.argmax(axis=1) # 每个anchor对应最大overlap的索引
    # 每个anchor所对应的最大overlap值
    box_max_overlaps = overlaps[np.arange(len(keep_indexs)), box_argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0) # 每个gt对应最大的overlap的索引
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                            np.arange(overlaps.shape[1])] # gt个数
    # print('box_max_overlaps:',box_max_overlaps)
    # print('gt_max_overlaps:',gt_max_overlaps)
    # overlap小于阈值的认为是背景
    labels[box_max_overlaps < cfg.RPN_OVERLAP_IS_NEGATIVE_THRESHOLD] = 0
    # gt中overlap最大的对应的anchor认为是前景
    labels[gt_argmax_overlaps] = 1
    # overlap大于阈值的认为是前景（正样本）
    labels[box_max_overlaps >= cfg.RPN_OVERLAP_IS_POSITIVE_THRESHOLD] = 1

    # 如果正样本太多，则对正样本进行采样
    # 限制正样本的数量不超过128个
    num_fg_recommend = cfg.RPN_POSITIVE_SAMPLE_SIZE
    fg_inds = np.where(labels==1)[0]
    if len(fg_inds) > num_fg_recommend:
        # 采样，随机选择移除一部分
        disable_inds = nprandom.choice(
            fg_inds, size=(len(fg_inds) - num_fg_recommend), replace=False)
        labels[disable_inds] = -1

    # 如果负样本太多，则对负样本进行采样
    # 正样本最多128个，总样本数为256个样本
    num_bg_recommend = cfg.RPN_SAMPLE_SIZE - np.sum(labels==1)
    bg_inds = np.where(labels==0)[0]
    if len(bg_inds) > num_bg_recommend:
        # 采样，随机选择移除一部分
        disable_inds = nprandom.choice(
            bg_inds, size=(len(bg_inds) - num_bg_recommend), replace=False)
        labels[disable_inds] = -1
    # 所有的label完成标记
    # // bbox_targets = np.zeros((len(keep_indexs), 4), dtype=np.float32)

    keep_first_anchors = anchors
    gt_boxes = gt_boxes

    bbox_targets = _compute_targets(keep_first_anchors, gt_boxes[box_argmax_overlaps, :])

    bbox_inside_weights = np.zeros((len(keep_indexs), 4), dtype=np.float32)
    bbox_inside_weights[labels==1, :] = np.array(cfg.RPN_BBOX_INSIDE_WEIGHTS) # 内部权重，前景赋值为1

    bbox_outside_weights = np.zeros((len(keep_indexs), 4), dtype=np.float32)
    # 使正样本为1，负样本为0
    positive_weights = np.ones((1,4))
    negative_weights = np.zeros((1,4))
    bbox_outside_weights[labels==1, :] = positive_weights # 外部权重，前景是1
    bbox_outside_weights[labels==0, :] = negative_weights # 背景是0

    if DEBUG:
        print('anchors: ',anchors)
        print('gt_box: ',gt_boxes)
        print('keep_first_anchors: ', keep_first_anchors)
        print('choose_anchors: ',gt_boxes[box_argmax_overlaps, :])
        print('label: ', labels)
        print('bbox_targets: ', bbox_targets)
        print('bbox_inside_weights: ', bbox_inside_weights)
        print('bbox_outside_weights', bbox_outside_weights)

    _num_total_anchors = 6
    A = 1 # 基本anchor个数
    height = 2
    width = 2

    labels = _umap(labels, _num_total_anchors, keep_indexs, fill=-1)
    bbox_targets = _umap(bbox_targets, _num_total_anchors, keep_indexs, fill=0)
    bbox_inside_weights = _umap(bbox_inside_weights, _num_total_anchors, keep_indexs, fill=0)
    bbox_outside_weights = _umap(bbox_outside_weights, _num_total_anchors, keep_indexs, fill=0)


    # labels
    labels = labels.reshape((1, height, width, A))
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets.reshape((1, height, width, A*4))
    rpn_bbox_targets = bbox_targets

    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A*4))
    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, A*4))
    rpn_bbox_outside_weights = bbox_outside_weights

    print('rpn_labels: ',rpn_labels)
    print('rpn_bbox_targets: ',rpn_bbox_targets)
    print('rpn_bbox_inside_weights: ',rpn_bbox_inside_weights)
    print('rpn_bbox_outside_weights: ',rpn_bbox_outside_weights)

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights





def _umap(data, count, index, fill=0):
    """ 将保留的anchor(data)，补充之前移除的anchor，返回所有的anchor """
    if len(data.shape) == 1:
        ret = np.empty((count), dtype=np.float32)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[index, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """计算一张图片的bounding-box 回归目标"""
    # A = np.array([[1, 1, 2, 2], [2.5, 2.5, 3, 3], [3, 3, 4, 4]])
    # B = np.array([[1, 1, 2.5, 2.5], [2, 2, 3, 3], [3, 3, 3.5, 3.5]])

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    # assert gt_rois.shape[1] == 5
    return bbox_transform(ex_rois,gt_rois[:,:4]).astype(np.float32, copy=False)


if __name__ == '__main__':
    t_overlap()