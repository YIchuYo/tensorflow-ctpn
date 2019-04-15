import os
import sys
import time
import shutil
import cv2
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))
import nets.model_train as model
from dataset import data_provider
from rpn_msr.proposal_layer import proposal_layer
from text_connector.detectors import TextDetector

PROJECT_PATH = os.path.dirname(os.path.abspath(os.getcwd()))
tf.app.flags.DEFINE_string('test_data_path', 'data/demo/', '')
tf.app.flags.DEFINE_string('output_path', 'data/res/', '')
# tf.app.flags.DEFINE_string('gpu', '3', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints_mlt/', '')

FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def main(argv=None):
    if os.path.exists(FLAGS.output_path):
        shutil.rmtree(FLAGS.output_path)
    os.makedirs(FLAGS.output_path)
    # os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    with tf.get_default_graph().as_default():
        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
        input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        bbox_pred, cls_pred, cls_prob = model.model(input_image)

        saver = tf.train.Saver()

        with tf.Session(config=tf.ConfigProto()) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)
            dp = data_provider.Data_provider()
            dp.clear_index()
            im_fn = dp.get_one_data(15)

            print('===============')
            # print(dp.index)
            start = time.time()
            im = im_fn[0].reshape([1,im_fn[2][0][0],im_fn[2][0][1],3])
            # print(im.shape)
            # h, w, c = im.shape
            im_info = im_fn[2]
            bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                                   feed_dict={input_image: im,
                                                              input_im_info: im_info})
            textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
            scores = textsegs[:, 0]
            textsegs = textsegs[:, 1:5] # proposals[x1,y1,x2,y2]

            textdetector = TextDetector()
            boxes = textdetector.detect(textsegs, scores[:, np.newaxis], im.shape[1:3])
            boxes = np.array(boxes, dtype=np.int)


            cost_time = (time.time() - start)
            print("cost time: {:.2f}s".format(cost_time))
            im = im.reshape([im_fn[2][0][0],im_fn[2][0][1],3])

            for i, box in enumerate(boxes):
                print(i)
                cv2.polylines(im, [box[:8].astype(np.int32).reshape((-1,1,2))], True, color=(255,0,0), thickness=2)
            # img = cv2.resize(im, None, None, interpolation=cv2.INTER_LINEAR)
            fig, axs = plt.subplots(1, 1, figsize=(30, 30))
            axs.imshow(im[:, :, ::-1])
            plt.tight_layout()
            plt.show()
            cv2.imwrite(os.path.join(FLAGS.output_path, 'test.jpg'), im)


if __name__ == '__main__':
    tf.app.run()