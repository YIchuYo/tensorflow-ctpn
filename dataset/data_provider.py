import os
import sys
import time
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

TRAIN_FOLDER = "data/train/"
TEST_FOLDER = "data/test"
PROJECT_DIR = os.path.dirname(os.path.abspath(os.getcwd()))
TRAIN_FOLDER = os.path.join(PROJECT_DIR, TRAIN_FOLDER)
TEST_FOLDER = os.path.join(PROJECT_DIR, TEST_FOLDER)


class Data_provider(object):
    index = 0
    mode = "train"
    def __init__(self, valid_size=0.05):
        self.train_image_paths, self.test_image_paths = self.get_all_image_path()
        random.shuffle(self.train_image_paths)
        valid_num = int(len(self.train_image_paths) * valid_size)
        self.valid_image_paths = self.train_image_paths[0:valid_num]
        self.train_image_paths = self.train_image_paths[valid_num:]
        # print(len(self.train_image_paths))
        # print(len(self.valid_image_paths))
        # print(len(self.test_image_paths))

        self.train_txt_paths = self.get_all_txt_path(self.train_image_paths)
        self.valid_txt_paths = self.get_all_txt_path(self.valid_image_paths)
        self.test_txt_paths = self.get_all_txt_path(self.test_image_paths)

        # self.all_num = len(self.all_image_paths)


    def clear_index(self):
        self.index = 0


    # def get_one_data(self, ind):
    #     if ind < self.all_num:
    #         im_fn = self.all_image_paths[ind]
    #         im = cv2.imread(im_fn)
    #         h, w, c = im.shape
    #         im = im.reshape([1, h, w, c])
    #         # print(h,w,c)
    #         im_info = np.array([h,w,c]).reshape([1,3])
    #         print("im_info",im_info)
    #         if not os.path.exists(self.all_image_paths[ind]):
    #             print("Ground truth for image {} not exist!".format(im_fn))
    #             return
    #         # bbox = self.load_annotation(self.all_txt_paths[ind])
    #         # if len(bbox) == 0:
    #         #     print("Ground truth for image {} empty!".format(im_fn))
    #         #     return -1
    #         bbox = None
    #         print(im_info)
    #         return [im, bbox, im_info]
    #     else:
    #         return None

    def get_next_data(self, mode="train"):
        if self.mode!=mode:
            self.clear_index()
            self.mode=mode

        if self.mode=="train":
            if self.index < len(self.train_image_paths):
                im_fn = self.train_image_paths[self.index]
                im = cv2.imread(im_fn)
                h, w, c = im.shape
                im = im.reshape([1, h, w, c])
                im_info = np.array([h,w,c]).reshape([1,3])
                # print("im_info",im_info)
                if not os.path.exists(self.train_txt_paths[self.index]):
                    print("Ground truth for train image {} not exist!".format(im_fn))
                    return
                bbox = self.load_annotation(self.train_txt_paths[self.index])
                if len(bbox) == 0:
                    print("Ground truth for image {} empty!".format(im_fn))
                    return -1

                self.index = self.index + 1
                return [im, bbox, im_info]
            else:
                return None
        elif self.mode=="valid":
            if self.index < len(self.valid_image_paths):
                im_fn = self.valid_image_paths[self.index]
                im = cv2.imread(im_fn)
                h, w, c = im.shape
                im = im.reshape([1, h, w, c])
                im_info = np.array([h,w,c]).reshape([1,3])
                # print("im_info",im_info)
                if not os.path.exists(self.valid_txt_paths[self.index]):
                    print("Ground truth for train image {} not exist!".format(im_fn))
                    return
                bbox = self.load_annotation(self.valid_txt_paths[self.index])
                if len(bbox) == 0:
                    print("Ground truth for image {} empty!".format(im_fn))
                    return -1

                self.index = self.index + 1
                return [im, bbox, im_info]
            else:
                return None


    def get_all_image_path(self):
        train_image_paths = []
        test_image_paths = []
        extens = ['jpg', 'png', 'jpeg']
        train_path = os.listdir(TRAIN_FOLDER)
        for p in train_path:
            if p.split('.')[-1] in extens:
                train_image_paths.append(os.path.join(TRAIN_FOLDER, p))

        test_path = os.listdir(TEST_FOLDER)
        for p in test_path:
            if p.split('.')[-1] in extens:
                test_image_paths.append(os.path.join(TEST_FOLDER, p))

        return train_image_paths, test_image_paths

    def get_all_txt_path(self, image_paths):
        txt_paths = []
        for p in image_paths:
            txt_paths.append(self._transfer_pic_txt_path(p))
        return txt_paths

    def load_annotation(self, path):
        bbox = []
        with open(path, "r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split(',')
            # print(line)
            x_min, y_min, x_max, y_max = map(int, line)
            bbox.append([x_min, y_min, x_max, y_max, 1])

        return bbox


    def _transfer_pic_txt_path(self, pic_path):
        _, filename = os.path.split(pic_path)
        filename, _ = os.path.splitext(filename)
        if self.mode=="train" or self.mode=="valid":
            txt_path = os.path.join(TRAIN_FOLDER, filename +'.txt')
        else :
            txt_path = os.path.join(TEST_FOLDER, filename +'.txt')

        return txt_path


# def get_both_data(vis=False):
#     image_list = np.array(get_training_data())
#     index_list = np.arange(0, image_list.shape[0])
#
#     im_fn = image_list[0]
#     im = cv2.imread(im_fn)
#     h, w, c = im.shape
#     im = im.reshape([h, w, c])
#     # print(h,w,c)
#     im_info = np.array([h,w,c]).reshape([1,3])
#     txt_path = _get_txt_path(im_fn)
#     # print(txt_path)
#     if not os.path.exists(txt_path):
#         print("Ground truth for image {} not exist!".format(im_fn))
#         return
#
#     bbox = load_annotation(txt_path)
#     if len(bbox) == 0:
#         print("Ground truth for image {} empty!".format(im_fn))
#
#     if vis:
#         bbox = [bbox[0][0:4]]
#         print(bbox)
#         for p in bbox:
#             print(p[0], p[1], p[2], p[3])
#             cv2.rectangle(im, (p[0],p[1]), (p[2], p[3]), color=(25,25,15), thickness=5)
#         fig, axs = plt.subplots(1, 1, figsize=(30,30))
#
#         axs.imshow(im[:,:,::-1])
#         plt.tight_layout()
#         plt.show()
#     im=im.reshape([1, h, w, c])
#     im = np.array(im)
#     bbox = np.array(bbox)
#     im_info = np.array(im_info)
#     # print('im: ', im.shape)
#     # print('bbox: ', bbox.shape)
#     # print('im_info: ', im_info.shape)
#     return [im, bbox, im_info]

# if __name__ == '__main__':
#     dp = Data_provider()
#     print("**************************************")
#     for i in range(0,3):
#         dp.clear_index()
#         next_data = dp.get_next_train_data()
#         while next_data is not None:
#             print(dp.index, next_data[2])
#             next_data = dp.get_next_train_data()
#         print("*******************",i,"*******************")