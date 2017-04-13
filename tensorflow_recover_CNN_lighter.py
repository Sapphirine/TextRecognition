import tensorflow as tf
import numpy as np
from image_load_save import load_data, check_path, batch_recover


class conv_layer(object):
    def __init__(self, input_x, in_size, out_size, kernal_shape, seed, index=""):
        with tf.name_scope('conv_layer' + str(index)):
            with tf.name_scope('kernel'):
                w_shape = [kernal_shape, kernal_shape, in_size, out_size]
                weight = tf.Variable(tf.truncated_normal(shape=w_shape, stddev=0.1, seed=seed),
                                     name="kernel" + str(index))
                self.weight = weight

            with tf.name_scope('bias'):
                b_shape = [out_size]
                bias = tf.Variable(tf.constant(0.1, shape=b_shape), name="bias" + str(index))
                self.bias = bias

            # strides [1, x_movement, y_movement, 1]
            conv_out = tf.nn.conv2d(input_x, weight,
                                    strides=[1, 1, 1, 1], padding="SAME")
            cell_out = tf.nn.relu(conv_out + bias)
            tf.summary.histogram("conv_layer" + str(index) + '/output', cell_out)
            self.cell_out = cell_out
            # return cell_out


def avg_pooling_layer(input_x, k_size):
    # strides [1, k_size, k_size, 1]
    with tf.name_scope('avg_pooling'):
        pooling_shape = [1, k_size, k_size, 1]
        return tf.nn.avg_pool(input_x, strides=pooling_shape, ksize=pooling_shape, padding="SAME")
        # return tf.nn.max_pool(input_x, strides=pooling_shape, ksize=pooling_shape, padding="SAME")


def upsampling_layer(input_x, k_size):
    with tf.name_scope('upsampling'):
        height = tf.shape(input_x)[1]
        width = tf.shape(input_x)[2]

        new_size = [height * k_size, width * k_size]
        result = tf.image.resize_nearest_neighbor(
            images=input_x,
            size=new_size)
        return result


# class My_CNN(object):
#     def __init__(self, input_size, seed, feature_map_size=[64, 128], learning_rate=1e-4):
#         ####################
#         # Building Network #
#         ####################
#
#         print("Building nets...")
#         # check the input size and creat the place holder.
#         # The input size we have is 16*16, and the batch size we have is 128
#         with tf.name_scope("Input"):
#             xs = tf.placeholder(tf.float32, shape=[None, input_size * input_size], name="corrupted_image")
#             ys = tf.placeholder(tf.float32, shape=[None, input_size * input_size], name="truth_image")
#
#             self.xs = xs
#             self.ys = ys
#
#             x = tf.reshape(xs, [-1, input_size, input_size, 1])
#
#         # layer 1, input(batch_size * 16 * 16 * 1) -> output(batch_size * 16 * 16 * 64)
#         conv_1_1 = conv_layer(input_x=x,
#                               in_size=1,
#                               out_size=feature_map_size[0],
#                               kernal_shape=3,
#                               seed=seed,
#                               index="1_1")
#
#         # layer 2, input(batch_size * 16 * 16 * 64) -> output(batch_size * 8 * 8 * 128)
#         avg_pooling_2 = avg_pooling_layer(input_x=conv_1_1.cell_out, k_size=2)
#
#         conv_2_1 = conv_layer(input_x=avg_pooling_2,
#                               in_size=feature_map_size[0],
#                               out_size=feature_map_size[1],
#                               kernal_shape=3,
#                               seed=seed,
#                               index="2_1")
#
#         # layer 3, input(batch_size * 8 * 8 * 128) -> output(batch_size * 16 * 16 * 64)
#         upsampling_2 = upsampling_layer(input_x=conv_2_1.cell_out, k_size=2)
#
#         conv_3_1 = conv_layer(input_x=upsampling_2,
#                               in_size=feature_map_size[1],
#                               out_size=feature_map_size[0],
#                               kernal_shape=3,
#                               seed=seed,
#                               index="3_1")
#
#         # layer 4, input(batch_size * 16 * 16 * 64) -> output(batch_size * 16 * 16 * 1)
#         adding = conv_3_1.cell_out + conv_1_1.cell_out
#
#         conv_4_1 = conv_layer(input_x=adding,
#                               in_size=feature_map_size[0],
#                               out_size=1,
#                               kernal_shape=3,
#                               seed=seed,
#                               index="4_1")
#
#         with tf.name_scope("prediction"):
#             y_pred = tf.reshape(conv_4_1.cell_out, [-1, input_size * input_size], name="y_pred")
#             self.y_pred = y_pred
#
#         with tf.name_scope("loss"):
#             # the loss of prediction result
#             loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_pred - ys), reduction_indices=[1]))
#             self.loss = loss
#
#         with tf.name_scope("training"):
#             # training optimizer and train_step
#             optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#             self.train_step = optimizer.minimize(loss)
#
#     def load_params(self, param_path="my_params/save_net_light.ckpt"):
#         saver = tf.train.Saver()
#         with tf.Session() as sess:
#             saver.restore(sess, param_path)
#
#
# def training(training_input, training_truth,
#              validate_input, validate_truth,
#              test_input, test_truth,
#              batch_size=128, epoch=5000,
#              learning_rate=1e-4, seed=123):
#     ############
#     # Training #
#     ############
#     check_path("my_params")
#     check_path("log/")
#
#     image_size = int(np.sqrt(training_input.shape[1]))
#
#     # with tf.name_scope('CNN'):
#     my_cnn = My_CNN(
#         input_size=image_size,
#         seed=seed,
#         learning_rate=learning_rate
#     )
#
#     for v in tf.global_variables():
#         tf.add_to_collection('vars', v)
#
#     saver = tf.train.Saver()
#     best_valid = np.infty
#
#     with tf.Session() as sess:
#         merged = tf.summary.merge_all()
#
#         writer = tf.summary.FileWriter("log/", sess.graph)
#
#         sess.run(tf.global_variables_initializer())
#
#         iter_total = 0
#         for i in range(2):
#             iters = int(training_input.shape[0] / batch_size)
#
#             for iter in range(2):
#                 iter_total += 1
#
#                 training_batch = training_input[iter * batch_size: (1 + iter) * batch_size]
#                 training_batch_truth = training_truth[iter * batch_size: (1 + iter) * batch_size]
#
#                 sess.run(my_cnn.train_step, feed_dict={my_cnn.xs: training_batch,
#                                                        my_cnn.ys: training_batch_truth})
#
#                 if iter_total % 100 == 1:
#                     valid_loss = sess.run(my_cnn.loss, feed_dict={my_cnn.xs: validate_input,
#                                                                   my_cnn.ys: validate_truth})
#                     print("iter: %d, valid RMES: %4f" % (iter_total, valid_loss))
#                     merged_result = sess.run(merged, feed_dict={my_cnn.xs: validate_input,
#                                                                 my_cnn.ys: validate_truth})
#                     writer.add_summary(merged_result, iter_total)
#
#                     if valid_loss < best_valid:
#                         best_valid = valid_loss
#                         test_loss = sess.run(my_cnn.loss, feed_dict={my_cnn.xs: test_input,
#                                                                      my_cnn.ys: test_truth})
#
#                         print("======Best validation, test RMES: %4f" % test_loss)
#
#                         saver.save(sess, "my_params/light-model")
#
#
# def load_CNN(pred_input, param_path="my_params/light-model.meta"):
#     image_size = int(np.sqrt(pred_input.shape[1]))
#
#     my_cnn = My_CNN(
#         input_size=image_size,
#         seed=123
#     )
#
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#
#         saver2 = tf.train.import_meta_graph(param_path)
#         saver2.restore(sess, tf.train.latest_checkpoint('./'))
#         # all_vars = tf.get_collection('vars')
#
#         result = sess.run(my_cnn.y_pred, feed_dict={my_cnn.xs: pred_input,
#                                                     my_cnn.ys: pred_input})
#
#         batch_recover(result)


rval = load_data()
train_input, train_output = rval[0]
valid_input, valid_output = rval[1]
test_input, test_output = rval[2]

input_size = 16
seed = 123
feature_map_size = [64, 128]
learning_rate = 1e-4

batch_size = 128
epoch = 80

####################
# Building Network #
####################

print("Building nets...")
# check the input size and creat the place holder.
# The input size we have is 16*16, and the batch size we have is 128
with tf.name_scope("Input"):
    xs = tf.placeholder(tf.float32, shape=[None, input_size * input_size], name="corrupted_image")
    ys = tf.placeholder(tf.float32, shape=[None, input_size * input_size], name="truth_image")

    x = tf.reshape(xs, [-1, input_size, input_size, 1])

# layer 1, input(batch_size * 16 * 16 * 1) -> output(batch_size * 16 * 16 * 64)
conv_1_1 = conv_layer(input_x=x,
                      in_size=1,
                      out_size=feature_map_size[0],
                      kernal_shape=3,
                      seed=seed,
                      index="1_1")

# layer 2, input(batch_size * 16 * 16 * 64) -> output(batch_size * 8 * 8 * 128)
avg_pooling_2 = avg_pooling_layer(input_x=conv_1_1.cell_out, k_size=2)

conv_2_1 = conv_layer(input_x=avg_pooling_2,
                      in_size=feature_map_size[0],
                      out_size=feature_map_size[1],
                      kernal_shape=3,
                      seed=seed,
                      index="2_1")

# layer 3, input(batch_size * 8 * 8 * 128) -> output(batch_size * 16 * 16 * 64)
upsampling_2 = upsampling_layer(input_x=conv_2_1.cell_out, k_size=2)

conv_3_1 = conv_layer(input_x=upsampling_2,
                      in_size=feature_map_size[1],
                      out_size=feature_map_size[0],
                      kernal_shape=3,
                      seed=seed,
                      index="3_1")

# layer 4, input(batch_size * 16 * 16 * 64) -> output(batch_size * 16 * 16 * 1)
adding = conv_3_1.cell_out + conv_1_1.cell_out

conv_4_1 = conv_layer(input_x=adding,
                      in_size=feature_map_size[0],
                      out_size=1,
                      kernal_shape=3,
                      seed=seed,
                      index="4_1")

with tf.name_scope("prediction"):
    y_pred = tf.reshape(conv_4_1.cell_out, [-1, input_size * input_size], name="y_pred")

with tf.name_scope("loss"):
    # the loss of prediction result
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_pred - ys), reduction_indices=[1]))

with tf.name_scope("training"):
    # training optimizer and train_step
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = optimizer.minimize(loss)

############
# Training #
############

check_path("my_params")
check_path("log/")

image_size = int(np.sqrt(train_input.shape[1]))

for v in tf.global_variables():
    tf.add_to_collection('vars', v)

saver = tf.train.Saver()
best_valid = np.infty

with tf.Session() as sess:
    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter("log/", sess.graph)

    sess.run(tf.global_variables_initializer())

    # load the previous training result if it is possible
    try:
        saver.restore(sess, "my_params/light-model")
    except Exception:
        pass

    iter_total = 0
    for i in range(epoch):
        iters = int(train_input.shape[0] / batch_size)

        for iter in range(iters):
            iter_total += 1

            training_batch = train_input[iter * batch_size: (1 + iter) * batch_size]
            training_batch_truth = train_output[iter * batch_size: (1 + iter) * batch_size]

            sess.run(train_step, feed_dict={xs: training_batch,
                                            ys: training_batch_truth})

            if iter_total % 100 == 1:
                valid_loss = sess.run(loss, feed_dict={xs: valid_input,
                                                       ys: valid_output})
                print("iter: %d, valid RMES: %4f" % (iter_total, valid_loss))
                merged_result = sess.run(merged, feed_dict={xs: valid_input,
                                                            ys: valid_output})
                writer.add_summary(merged_result, iter_total)

                if valid_loss < best_valid:
                    best_valid = valid_loss
                    test_loss = sess.run(loss, feed_dict={xs: test_input,
                                                          ys: test_output})

                    print("======Best validation, test RMES: %4f" % test_loss)

                    saver.save(sess, "my_params/light-model")
