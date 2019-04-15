import os
import sys
import time
import datetime
import tensorflow as tf
from tensorflow.contrib import slim
# sys.path.append(os.getcwd())

sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))
import nets.model_train as model
from dataset import data_provider as dp
PROJECT_PATH = os.path.dirname(os.path.abspath(os.getcwd()))


tf.app.flags.DEFINE_float('learning_rate', 1e-5, '')
tf.app.flags.DEFINE_integer('max_steps', 500000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('gpu', '3', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints_mlt/', '')
tf.app.flags.DEFINE_string('logs_path', 'logs_mlt/', '')
tf.app.flags.DEFINE_string('pretrained_model_path',  os.path.join(PROJECT_PATH, 'data/vgg_16.ckpt'), '')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 500, '')
tf.app.flags.DEFINE_boolean('restore', True, '')

FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# def main(argv=None):
#     # 一张大小为600×900的图片
#     input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
#     # 有3个gtbox
#     input_bbox = tf.placeholder(tf.float32, shape=[None, 5], name='input_bbox')
#     # 一张图片的信息
#     input_im_info = tf.placeholder(tf.float32, shape=[None,3], name='input_im_info')
#
#     # 超参数
#     global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
#     learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
#     tf.summary.scalar('learning_rate', learning_rate)
#     opt = tf.train.AdamOptimizer(learning_rate)
#
#     # cpu train
#     with tf.name_scope('model') as scope:
#         bbox_pred, cls_pred, cls_prob = model.model(input_image)
#         total_loss, model_loss, rpn_cross_entropy, rpn_loss_box = model.loss(bbox_pred, cls_pred, input_bbox, input_im_info)
#         # batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
#         grads = opt.compute_gradients(total_loss)
#     # **************************************
#     apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
#
#     summary_op = tf.summary.merge_all()
#     variable_averages = tf.train.ExponentialMovingAverage(
#         FLAGS.moving_average_decay, global_step)
#     variables_averages_op = variable_averages.apply(tf.trainable_variables())
#     with tf.control_dependencies([variables_averages_op, apply_gradient_op]):
#     # with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
#         train_op = tf.no_op(name='train_op')
#
#     # saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
#     # summary_writer = tf.summary.FileWriter(FLAGS.logs_path + StyleTime, tf.get_default_graph())
#
#     init = tf.global_variables_initializer()
#
#     # **************************************
#
#     #
#     if FLAGS.pretrained_model_path is not None:
#         variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path,
#                                                              slim.get_trainable_variables(),
#                                                              ignore_missing_vars=True)
#
#
#     config = tf.ConfigProto()
#     with tf.Session(config=config) as sess:
#         # 第一次运行，有点激动
#         sess.run(init)
#         restore_step = 0
#         if FLAGS.pretrained_model_path is not None:
#             variable_restore_op(sess)
#
#         start = time.time()
#
#         for step in range(restore_step, FLAGS.max_steps):
#             data = dp.get_both_data()
#             # ml, tl, _, summary_str = sess.run([model_loss, total_loss, train_op, summary_op],
#             ml, tl, _, summary_str = sess.run([model_loss, total_loss, train_op, summary_op],
#                                               feed_dict={input_image:data[0],
#                                                          input_bbox:data[1],
#                                                          input_im_info:data[2]})
#
#             # summary_writer.add_summary(summary_str, global_step=step)
#
#             if step % 10 == 0:
#                 avg_time_per_step = (time.time() - start) / 10
#                 start = time.time()
#                 print('Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, LR: {:.6f}'.format(
#                     step, ml, tl, avg_time_per_step, learning_rate.eval()))
#
#     print('total_loss: ', total_loss)
#     print('model_loss: ', model_loss)
#     print('rpn_cross_entropy: ', rpn_cross_entropy)
#     print('rpn_loss_box: ', rpn_loss_box)


def main(argv=None):
    now = datetime.datetime.now()
    StyleTime = now.strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(FLAGS.logs_path + StyleTime)

    # 一张大小为600×900的图片
    input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
    # 有3个gtbox
    input_bbox = tf.placeholder(tf.float32, shape=[None, 5], name='input_bbox')
    # 一张图片的信息
    input_im_info = tf.placeholder(tf.float32, shape=[None,3], name='input_im_info')
    # 超参数
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
    tf.summary.scalar('learning_rate', learning_rate)

    opt = tf.train.AdamOptimizer(learning_rate)
    # cpu train
    gpu_id = int(FLAGS.gpu)
    # with tf.device('/gpu:%d' % gpu_id):
    with tf.name_scope('model_%d' % gpu_id) as scope:
        bbox_pred, cls_pred, cls_prob = model.model(input_image)
        total_loss, model_loss, rpn_cross_entropy, rpn_loss_box = model.loss(bbox_pred, cls_pred, input_bbox, input_im_info)

        grads = opt.compute_gradients(total_loss)


    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    summary_writer = tf.summary.FileWriter(FLAGS.logs_path + StyleTime, tf.get_default_graph())

    if FLAGS.pretrained_model_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path,
                                                             slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)



    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # config.gpu_options.allow_growth = True
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    with tf.Session(config=config) as sess:
        # 第一次运行，有点激动
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            restore_step = int(ckpt.split('.')[0].split('_')[-1])

            # ckpt = os.path.join(os.path.abspath(os.getcwd()), ckpt)
            print("continue training from previous checkpoint {}".format(restore_step))
            saver.restore(sess, ckpt)
            print("restore finish")
        else:
            sess.run(init)
            print('init!')
            restore_step = 0
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)
        print("_______________")
        start = time.time()
        data_provider = dp.Data_provider()
        for step in range(restore_step, FLAGS.max_steps):
            data_provider.clear_index()
            data = data_provider.get_next_data()
            while data is not None:
                # print(data[0].shape)
                # ml, tl, _, summary_str = sess.run([model_loss, total_loss, train_op, summary_op],
                ml, tl, _, summary_str= sess.run([model_loss, total_loss, train_op, summary_op],
                                 feed_dict={input_image:data[0],
                                            input_bbox:data[1],
                                            input_im_info:data[2]})
                summary_writer.add_summary(summary_str, global_step=step)

                # print("ml: ", ml)
                # print("tl: ", tl)

                avg_time_per_step = (time.time() - start) / 10
                start = time.time()
                p = step*data_provider.all_num+data_provider.index
                print('Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, LR: {:.6f}'.format(
                    p, ml, tl, avg_time_per_step, learning_rate.eval()))

                if p % FLAGS.save_checkpoint_steps == 0:
                    filename = ('ctpn_{:d}'.format(p) + '.ckpt')
                    filename = os.path.join(FLAGS.checkpoint_path, filename)
                    saver.save(sess, filename)
                    print('Write model to: {:s}'.format(filename))

                data = data_provider.get_next_data()
                while data==-1:
                    data_provider.index = data_provider.index + 1
                    data = data_provider.get_next_data()

if __name__ == '__main__':
    main()