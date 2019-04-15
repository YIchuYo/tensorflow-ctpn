import numpy as np

# gen A=10 according to height
def generate_anchors(fix_width = [16], base_height=[11, 283],
                     anchor_num=10): # [8, 16, 32]
    # 源码里面使用[11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    # 不规则尺寸，也有一定道理吧。
    # 我按固定尺寸生成anchor_num个anchor
    sizes = []

    for w in fix_width:
        h = base_height[0]
        while h < base_height[1]:
            sizes.append((h,w))
            h += (base_height[1] - base_height[0]) // (anchor_num - 1)
    # print(sizes)
    return generate_basic_anchors(sizes)

# gen A=10
def generate_basic_anchors(sizes, base_size=16):
    """
    根据sizes数组(height,width)，返回anchors
    :param sizes: A个anchors的(height,width) list
    :param base_size:
    :return:
    """
    basic_anchor = np.array([0, 0, base_size-1, base_size-1], np.int32) # 一个基anchor
    anchors = np.zeros((len(sizes), 4), np.int32) # anchors
    index = 0
    for h, w in sizes:
        # 这里我觉得。。弄的函数太多了把，虽然分工很明确。
        anchors[index] = custom_based_basic_anchor(basic_anchor, h, w)
        index += 1
    # print(anchors)
    return anchors

# gen 1
def custom_based_basic_anchor(anchor, h, w):
    x_ctr= (anchor[0] + anchor[2]) // 2
    y_ctr= (anchor[1] + anchor[3]) // 2

    custom_anchor = anchor.copy()
    custom_anchor[0] = x_ctr - w // 2
    custom_anchor[2] = x_ctr + w // 2
    custom_anchor[1] = y_ctr - h // 2
    custom_anchor[3] = y_ctr + h // 2

    return custom_anchor


