import os
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
PROJECT_PATH = os.path.dirname(os.path.abspath(os.getcwd()))
tf.app.flags.DEFINE_string('pretrained_model_path',  os.path.join(PROJECT_PATH, 'data/vgg_16.ckpt'), '')
FLAGS = tf.app.flags.FLAGS

def vgg_arg_scope(weight_decay=0.1):
  """Defines the VGG arg scope.
  Args:
    weight_decay: The l2 regularization coefficient.
  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc

def vgg16(inputs,scope='vgg_16'):
    with tf.variable_scope(None, 'vgg_16', [inputs]) as sc:
        # end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],):
                            # outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            # net = slim.max_pool2d(net, [2, 2], scope='pool5')
            # net = slim.fully_connected(net, 4096, scope='fc6')
            # net = slim.dropout(net, 0.5, scope='dropout6')
            # net = slim.fully_connected(net, 4096, scope='fc7')
            # net = slim.dropout(net, 0.5, scope='dropout7')
            # net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
        return net

def net():
    input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
    # with slim.arg_scope(vgg_arg_scope()):
    conv5_3 = vgg16(input_image)      # 核心代码，直接调用即可

    init = tf.global_variables_initializer()
    if FLAGS.pretrained_model_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path,
                                                             slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)
    with tf.Session() as sess:
        sess.run(init)
        if FLAGS.pretrained_model_path is not None:
            variable_restore_op(sess)
        a = sess.run([conv5_3],feed_dict={input_image:np.arange(54).reshape(1,3,6,3)})

if __name__ == '__main__':
    net()
    print(tf.trainable_variables())
    # [<tf.Variable 'vgg_16/conv1/conv1_1/weights:0' shape=(3, 3, 3, 64) dtype=float32_ref>,
    # <tf.Variable 'vgg_16/conv1/conv1_1/biases:0' shape=(64,) dtype=float32_ref>,
    # <tf.Variable 'vgg_16/conv1/conv1_2/weights:0' shape=(3, 3, 64, 64) dtype=float32_ref>,
    # <tf.Variable 'vgg_16/conv1/conv1_2/biases:0' shape=(64,) dtype=float32_ref>,
    # <tf.Variable 'vgg_16/conv2/conv2_1/weights:0' shape=(3, 3, 64, 128) dtype=float32_ref>,
    # <tf.Variable 'vgg_16/conv2/conv2_1/biases:0' shape=(128,) dtype=float32_ref>,
    # <tf.Variable 'vgg_16/conv2/conv2_2/weights:0' shape=(3, 3, 128, 128) dtype=float32_ref>,
    # <tf.Variable 'vgg_16/conv2/conv2_2/biases:0' shape=(128,) dtype=float32_ref>,
    # <tf.Variable 'vgg_16/conv3/conv3_1/weights:0' shape=(3, 3, 128, 256) dtype=float32_ref>,
    # <tf.Variable 'vgg_16/conv3/conv3_1/biases:0' shape=(256,) dtype=float32_ref>,
    # <tf.Variable 'vgg_16/conv3/conv3_2/weights:0' shape=(3, 3, 256, 256) dtype=float32_ref>,
    # <tf.Variable 'vgg_16/conv3/conv3_2/biases:0' shape=(256,) dtype=float32_ref>,
    # <tf.Variable 'vgg_16/conv3/conv3_3/weights:0' shape=(3, 3, 256, 256) dtype=float32_ref>,
    # <tf.Variable 'vgg_16/conv3/conv3_3/biases:0' shape=(256,) dtype=float32_ref>,
    # <tf.Variable 'vgg_16/conv4/conv4_1/weights:0' shape=(3, 3, 256, 512) dtype=float32_ref>,
    # <tf.Variable 'vgg_16/conv4/conv4_1/biases:0' shape=(512,) dtype=float32_ref>,
    # <tf.Variable 'vgg_16/conv4/conv4_2/weights:0' shape=(3, 3, 512, 512) dtype=float32_ref>,
    # <tf.Variable 'vgg_16/conv4/conv4_2/biases:0' shape=(512,) dtype=float32_ref>,
    # <tf.Variable 'vgg_16/conv4/conv4_3/weights:0' shape=(3, 3, 512, 512) dtype=float32_ref>,
    # <tf.Variable 'vgg_16/conv4/conv4_3/biases:0' shape=(512,) dtype=float32_ref>,
    # <tf.Variable 'vgg_16/conv5/conv5_1/weights:0' shape=(3, 3, 512, 512) dtype=float32_ref>,
    # <tf.Variable 'vgg_16/conv5/conv5_1/biases:0' shape=(512,) dtype=float32_ref>,
    # <tf.Variable 'vgg_16/conv5/conv5_2/weights:0' shape=(3, 3, 512, 512) dtype=float32_ref>,
    # <tf.Variable 'vgg_16/conv5/conv5_2/biases:0' shape=(512,) dtype=float32_ref>,
    # <tf.Variable 'vgg_16/conv5/conv5_3/weights:0' shape=(3, 3, 512, 512) dtype=float32_ref>,
    # <tf.Variable 'vgg_16/conv5/conv5_3/biases:0' shape=(512,) dtype=float32_ref>]