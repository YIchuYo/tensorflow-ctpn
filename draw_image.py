import cv2

def draw_box_4pt(img, pt, color=(0, 255, 0), thickness=1):
    if not isinstance(pt[0], int):
        pt = [int(pt[i]) for i in range(8)]
    img = cv2.line(img, (pt[0], pt[1]), (pt[2], pt[3]), color, thickness)
    img = cv2.line(img, (pt[2], pt[3]), (pt[4], pt[5]), color, thickness)
    img = cv2.line(img, (pt[4], pt[5]), (pt[6], pt[7]), color, thickness)
    img = cv2.line(img, (pt[6], pt[7]), (pt[0], pt[1]), color, thickness)
    return img

def draw_box_2pt(img, pt, color=(0, 255, 0), thickness=1):
    if not isinstance(pt[0], int):
        pt = [int(pt[i]) for i in range(4)]
    img = cv2.rectangle(img, (pt[0], pt[1]), (pt[2], pt[3]), color, thickness=thickness)
    return img

def draw_box_h_and_c(img, position, cy, h, anchor_width=16, color=(0, 255, 0), thickness=1):
    x_left = position * anchor_width
    x_right = (position + 1) * anchor_width - 1
    y_top = int(cy - (float(h) - 1) / 2.0)
    y_bottom = int(cy + (float(h) - 1) / 2.0)
    pt = [x_left, y_top, x_right, y_bottom]
    return draw_box_2pt(img, pt, color=color, thickness=thickness)