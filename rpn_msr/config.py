class Config:
    RPN_OVERLAP_IS_NEGATIVE_THRESHOLD = 0.3
    RPN_OVERLAP_IS_POSITIVE_THRESHOLD = 0.7

    RPN_POSITIVE_SAMPLE_SIZE = 128 # 限制正样本最多只有128个
    RPN_SAMPLE_SIZE = 256 # 限制总样本为256个

    RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

    # proposal_layer config (for predict)
    RPN_PRE_NMS_TOP_N = 12000 # before nms, remain 12000 boxes mostly
    RPN_POST_NMS_TOP_N = 500 # after nms, remain 2000 boxes mostly
    RPN_NMS_THRESH = 0.7      # param for nms threshold
    RPN_BOX_MIN_SIZE = 16     # the min size of proposal box
                              # (required height and width > 16)
