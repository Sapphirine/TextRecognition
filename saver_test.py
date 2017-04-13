import tensorflow as tf
from image_load_save import load_data, batch_recover


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
        results = tf.image.resize_nearest_neighbor(
            images=input_x,
            size=new_size)
        return results


input_size = 16
seed = 123
feature_map_size = [64, 128]
learning_rate = 1e-4

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

##################
# Restore Status #
##################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "my_params/light-model")

    test_input, test_output = load_data()[2]
    result = sess.run(y_pred, feed_dict={xs: test_input,
                                         ys: test_input})

    new_result = 0.2*result+0.8*test_output
    batch_recover(new_result)
