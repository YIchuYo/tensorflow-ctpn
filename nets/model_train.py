import tensorflow as tf
from tensorflow.contrib import slim

from nets import vgg
from rpn_msr.anchor_target_layer import  anchor_target_layer as anchor_target_layer_py


def make_var(name, shape, initializer=None):
    return tf.get_variable(name, shape, initializer=initializer)

def Bilstm(net, input_channel, hidden_unit_num, output_channel, scope_name):
    # width--->time step
    with tf.variable_scope(scope_name) as scope:
        shape = tf.shape(net)
        N, H, W, C = shape[0], shape[1], shape[2], shape[3]
        net = tf.reshape(net, [N * H, W, C]) # 之前不怎么理解，为什么将N和H合并。
                                             # 现在知道了LSTM是一行传入LSTM
                                             # 第二次预测的时候传入第二行
        net.set_shape([None, None, input_channel]) # 将通道数修改？

        # hidden_unit_num: LSTM Cell中的单元数量，隐藏层神经元数量
        lstm_fw_cell = tf.contrib.rnn.LSTMCell(hidden_unit_num, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.LSTMCell(hidden_unit_num, state_is_tuple=True)

        # 这个函数已被弃用
        lstm_out, last_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, net , dtype=tf.float32)

        lstm_out = tf.concat(lstm_out, axis=-1) # 将两个正反lstm得到的值合并起来
        print(lstm_out)
        lstm_out = tf.reshape(lstm_out, [N*H*W,2*hidden_unit_num])

        init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
        init_biases = tf.constant_initializer(0.0)
        weights = make_var('weights', [2 * hidden_unit_num, output_channel], init_weights)
        biases = make_var('biases', [output_channel], init_biases)

        outputs = tf.matmul(lstm_out, weights) + biases # 矩阵相乘，得到输出尺寸
        outputs = tf.reshape(outputs, [N,H,W,output_channel])
        return outputs

def lstm_fc(net, input_channel, output_channel, scope_name):
    with tf.variable_scope(scope_name) as scope:
        shape = tf.shape(net)
        N, H, W, C = shape[0], shape[1], shape[2], shape[3]
        net = tf.reshape(net, [N*H*W,C])

        init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
        init_biases = tf.constant_initializer(0.0)
        weights = make_var('weights', [input_channel, output_channel], init_weights)
        biases = make_var('biases', [output_channel], init_biases)

        output = tf.matmul(net, weights) + biases
        output = tf.reshape(output, [N, H, W, output_channel])

    return output

def model(image):
    AnchorNum = 10
    # 很好奇，即使是在函数外，这个也可以作用到函数吗？
    with slim.arg_scope(vgg.vgg_arg_scope()):
        conv5_3 = vgg.vgg_16(image)
        print(conv5_3)

    # 不知道这边在论文里对应的步骤?
    conv6 = slim.conv2d(conv5_3, 512, 3) # kernel_size=512;stride=3
    print(conv6)

    lstm_output = Bilstm(conv6, 512, 128, 512, scope_name='BiLSTM')
    print(lstm_output)
    # 1:[kw] 这边原作者的代码是[lstm_output, 512, 10 * 4]
    #   之后想做一个这里的改动，现在继续按照原来的论文理解来弄
    # 2:其实卷积层也是可以的，不用fc层
    bbox_pred = lstm_fc(lstm_output, 512, AnchorNum * 4, scope_name='bbox_pred')# 2 :中心点的x值和 height
    cls_pred = lstm_fc(lstm_output, 512, AnchorNum * 2, scope_name='cls_pred') # text和非text分数
    print(bbox_pred)
    print(cls_pred)

    # 目标:得到分类结果,先进行转置,将分数那一维的(10*2)变为(2),放到前一维度,width维度
    cls_pred_shape = tf.shape(cls_pred)
    cls_pred_reshape = tf.reshape(cls_pred, [cls_pred_shape[0],
                                             cls_pred_shape[1],
                                             cls_pred_shape[2] * AnchorNum,
                                             2], name='cls_pred_reshape')

    cls_pred_reshape_shape = tf.shape(cls_pred_reshape)
    # text/no text 用softmax概率表示
    cls_prob = tf.reshape(tf.nn.softmax(tf.reshape(cls_pred_reshape, [-1, cls_pred_reshape_shape[3]])),
                          [-1, cls_pred_reshape_shape[1], cls_pred_reshape_shape[2], cls_pred_reshape_shape[3]],
                          name='cls_prob')
    print(cls_prob)

    return bbox_pred, cls_pred, cls_prob


def anchor_target_layer(cls_pred, gt_boxes, im_info, scope_name):
    with tf.variable_scope(scope_name) as scope:
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
            tf.py_func(anchor_target_layer_py,
                       [cls_pred, gt_boxes, im_info, [16,], [16]],
                       [tf.float32, tf.float32, tf.float32, tf.float32])
        rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32),
                                          name='rpn_labels')
        rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets,
                                            name='rpn_bbox_targets')
        rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights,
                                            name='rpn_bbox_inside_weights')
        rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights,
                                            name='rpn_bbox_outside_weights')

        return [rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights]


def smooth_l1_dist(deltas, sigma=9.0, name='smooth_l1_dist'):
    with tf.name_scope(name=name) as scope:
        deltas_abs = tf.abs(deltas)
        # 很厉害的代码，但是这边的sigma不是特别明白
        smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0/sigma), tf.float32) # 输出0或1，真表示1
        return tf.square(deltas) * 0.5 * sigma * smoothL1_sign + \
               (deltas_abs - 0.5 /sigma) * tf.abs(smoothL1_sign - 1)

def loss(bbox_pred, cls_pred, gt_boxes, im_info):
    """

    :param bbox_pred:
    :param cls_pred:
    :param gt_boxes:
    :param im_info:
    :return:
    """

    # rpn_labels: (HxWxA, 1), 对于每个anchor 0表示bg 1表示fg -1表示dont care
    # rpn_bbox_targets: (HxWxA, 4), anchor和gt_boxes之间的距离，are the regression objectives
    # rpn_bbox_inside_weights: (HxWxA, 4) 每个boxes的权重
    # rpn_bbox_outside_weights: (HxWxA, 4) 用于平衡fg和bg的数量，显然，bg更多啦
    rpn_data = anchor_target_layer(cls_pred, gt_boxes, im_info, "anchor_target_layer")
    
    # 1. classification loss
    # transpose: (1, H, W, A*2) -> (1, H, W*A, 2)
    cls_pred_shape = tf.shape(cls_pred)
    cls_pred_reshape = tf.reshape(cls_pred, [cls_pred_shape[0], cls_pred_shape[1], -1, 2])
    rpn_cls_score = tf.reshape(cls_pred_reshape, [-1, 2])
    rpn_label = tf.reshape(rpn_data[0], [-1]) # 变成一维

    # 忽视label(-1)
    fg_keep = tf.equal(rpn_label, 1) # 前景保留的label索引
    rpn_keep = tf.where(tf.not_equal(rpn_label, -1))

    rpn_cls_score = tf.gather(rpn_cls_score, rpn_keep) # 根据indices获取指定的元素
    rpn_label = tf.gather(rpn_label, rpn_keep)
    rpn_cross_entropy_axis1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rpn_label, logits=rpn_cls_score)

    # 2. box loss
    rpn_bbox_pred = bbox_pred # 预测的K个框
    rpn_bbox_targets = rpn_data[1]
    rpn_bbox_inside_weights = rpn_data[2]
    rpn_bbox_outside_weights = rpn_data[3]

    rpn_bbox_pred = tf.gather(tf.reshape(rpn_bbox_pred, [-1, 4]), rpn_keep)
    rpn_bbox_targets = tf.gather(tf.reshape(rpn_bbox_targets, [-1, 4]), rpn_keep)
    rpn_bbox_inside_weights = tf.gather(tf.reshape(rpn_bbox_inside_weights, [-1, 4]), rpn_keep)
    rpn_bbox_outside_weights = tf.gather(tf.reshape(rpn_bbox_outside_weights, [-1, 4]), rpn_keep)

    rpn_loss_box_axis1 = tf.reduce_sum(rpn_bbox_outside_weights * smooth_l1_dist(
        rpn_bbox_inside_weights * (rpn_bbox_pred - rpn_bbox_targets)), axis=1)

    rpn_loss_box = tf.reduce_sum(rpn_loss_box_axis1) / (tf.reduce_sum(tf.cast(fg_keep, tf.float32)) + 1)
    rpn_cross_entropy = tf.reduce_mean(rpn_cross_entropy_axis1)

    model_loss = rpn_cross_entropy + rpn_loss_box
    # 得到所有使用正则化的变量
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(regularization_losses) + model_loss
    #
    tf.summary.scalar('model_loss', model_loss)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('rpn_cross_entropy', rpn_cross_entropy)
    tf.summary.scalar('rpn_loss_box', rpn_loss_box)

    return total_loss, model_loss, rpn_cross_entropy, rpn_loss_box
