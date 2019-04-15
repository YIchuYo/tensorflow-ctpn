import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def vgg_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer(),
                        padding='SAME') as arg_sc: # initializer好像可以自己赋值，[kw]
        return arg_sc


def vgg_16(inputs=None, scope='vgg_16', istraining=False):
    with tf.variable_scope(scope, 'vgg_16',[inputs]) as sc:
        with slim.arg_scope([slim.conv2d, slim.max_pool2d]):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3,3], scope='conv1')
            net = slim.max_pool2d(net, [2,2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3,3], scope='conv2')
            net = slim.max_pool2d(net, [2,2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3,3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')

    return net




