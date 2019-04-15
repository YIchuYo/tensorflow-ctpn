import numpy as np

def bbox_overlaps(anchors, gt_boxes):
    # anchors = (A, 4)
    # gt_boxes = (B, 4)
    # 3 steps:
    #   1. 计算两块区域的交
    #   2. 计算两块区域的并
    #   3. 使交除并
    pass