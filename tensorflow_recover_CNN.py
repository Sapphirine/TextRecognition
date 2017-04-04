import tensorflow as tf
import numpy as np
from image_load_save import load_data


def conv_layer(input_x, in_size, out_size, kernal_shape, seed, index=""):
    with tf.name_scope('conv_layer' + str(index)):
        with tf.name_scope('kernel'):
            w_shape = [kernal_shape, kernal_shape, in_size, out_size]
            weight = tf.Variable(tf.truncated_normal(shape=w_shape, stddev=0.1, seed=seed))

        with tf.name_scope('bias'):
            b_shape = [out_size]
            bias = tf.Variable(tf.constant(0.1, shape=b_shape))

        # strides [1, x_movement, y_movement, 1]
        conv_out = tf.nn.conv2d(input_x, weight,
                                strides=[1, 1, 1, 1], padding="SAME")
        cell_out = tf.nn.relu(conv_out + bias)
        tf.summary.histogram("conv_layer" + str(index) + '/output', cell_out)
        return cell_out


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


class My_CNN(object):
    def __init__(self, input_size, seed, feature_map_size=[64, 128], learning_rate=1e-4):
        ####################
        # Building Network #
        ####################

        print("Building nets...")
        # check the input size and creat the place holder.
        # The input size we have is 16*16, and the batch size we have is 128
        with tf.name_scope("Input"):
            xs = tf.placeholder(tf.float32, shape=[None, input_size * input_size], name="corrupted_image")
            ys = tf.placeholder(tf.float32, shape=[None, input_size * input_size], name="truth_image")

            self.xs = xs
            self.ys = ys

            x = tf.reshape(xs, [-1, input_size, input_size, 1])

        # layer 1, input(batch_size * 16 * 16 * 1) -> output(batch_size * 16 * 16 * 64）
        conv_1_1 = conv_layer(input_x=x,
                              in_size=1,
                              out_size=feature_map_size[0],
                              kernal_shape=3,
                              seed=seed,
                              index="1_1")
        conv_1_2 = conv_layer(input_x=conv_1_1,
                              in_size=feature_map_size[0],
                              out_size=feature_map_size[0],
                              kernal_shape=3,
                              seed=seed,
                              index="1_2")

        # layer 2, input(batch_size * 16 * 16 * 64) -> output(batch_size * 8 * 8 * 128）
        avg_pooling_2 = avg_pooling_layer(input_x=conv_1_2, k_size=2)

        conv_2_1 = conv_layer(input_x=avg_pooling_2,
                              in_size=feature_map_size[0],
                              out_size=feature_map_size[1],
                              kernal_shape=3,
                              seed=seed,
                              index="2_1")
        conv_2_2 = conv_layer(input_x=conv_2_1,
                              in_size=feature_map_size[1],
                              out_size=feature_map_size[1],
                              kernal_shape=3,
                              seed=seed,
                              index="2_2")

        # layer 3, input(batch_size * 8 * 8 * 128) -> output(batch_size * 16 * 16 * 64)
        upsampling_2 = upsampling_layer(input_x=conv_2_2, k_size=2)

        conv_3_1 = conv_layer(input_x=upsampling_2,
                              in_size=feature_map_size[1],
                              out_size=feature_map_size[0],
                              kernal_shape=3,
                              seed=seed,
                              index="3_1")
        conv_3_2 = conv_layer(input_x=conv_3_1,
                              in_size=feature_map_size[0],
                              out_size=feature_map_size[0],
                              kernal_shape=3,
                              seed=seed,
                              index="3_2")

        # layer 4, input(batch_size * 16 * 16 * 64) -> output(batch_size * 16 * 16 * 1）
        adding = conv_3_2 + conv_1_2

        conv_4_1 = conv_layer(input_x=adding,
                              in_size=feature_map_size[0],
                              out_size=1,
                              kernal_shape=3,
                              seed=seed,
                              index="4_1")

        with tf.name_scope("prediction"):
            y_pred = tf.reshape(conv_4_1, [-1, input_size * input_size], name="y_pred")

        with tf.name_scope("loss"):
            # the loss of prediction result
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_pred - ys), reduction_indices=[1]))
            self.loss = loss

        with tf.name_scope("training"):
            # training optimizer and train_step
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_step = optimizer.minimize(loss)


def training(training_input, training_truth,
             validate_input, validate_truth,
             test_input, test_truth,
             batch_size=128, epoch=5000,
             learning_rate=1e-4, seed=123):
    ############
    # Training #
    ############
    image_size = int(np.sqrt(training_input.shape[1]))

    with tf.name_scope('CNN'):
        my_cnn = My_CNN(
            input_size=image_size,
            seed=seed,
            learning_rate=learning_rate
        )

    best_valid = np.infty
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log/", sess.graph)

        sess.run(tf.global_variables_initializer())

        iter_total = 0
        for i in range(epoch):
            iters = int(training_input.shape[0] / batch_size)

            for iter in range(iters):
                iter_total += 1

                training_batch = training_input[iter * batch_size: (1 + iter) * batch_size]
                training_batch_truth = training_truth[iter * batch_size: (1 + iter) * batch_size]

                sess.run(my_cnn.train_step, feed_dict={my_cnn.xs: training_batch,
                                                       my_cnn.ys: training_batch_truth})

                train_loss = sess.run(my_cnn.loss, feed_dict={my_cnn.xs: training_batch,
                                                              my_cnn.ys: training_batch_truth})

                if iter_total % 100 == 1:
                    print("iter: %d, training RMES: %.4f" % (iter_total, train_loss))
                    merged_result = sess.run(merged, feed_dict={my_cnn.xs: training_batch,
                                                                my_cnn.ys: training_batch_truth})
                    writer.add_summary(merged_result, iter_total)

            valid_loss = sess.run(my_cnn.loss, feed_dict={my_cnn.xs: validate_input,
                                                          my_cnn.ys: validate_truth})
            if valid_loss < best_valid:
                best_valid = valid_loss


if __name__ == '__main__':
    rval = load_data()
    train_input, train_output = rval[0]
    valid_input, valid_output = rval[1]
    test_input, test_output = rval[2]

    training(train_input, train_output,
             valid_input, valid_output,
             test_input, test_output,
             batch_size=128, epoch=1,
             learning_rate=1e-4, seed=123)
