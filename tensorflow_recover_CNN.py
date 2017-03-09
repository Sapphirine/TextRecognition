import tensorflow as tf
import os

#
# IMAGE_SIZE=(800, 30)
#
# def load_data(path):
#     if not os.path.exists(path):
#         print("No such file!")
#         return
#

# a = tf.constant(3.0, tf.float32)
# b = tf.constant(4.0)
#
# # print(a, b)
#
sess = tf.Session()
#
# # print(sess.run([a,b]))
#
# c = tf.add(a, b)
# print(c)
# print(sess.run(c))

# node_a = tf.placeholder(tf.float32)
# node_b = tf.placeholder(tf.float32)
# adder_node = node_a + node_b
#
# print(sess.run(adder_node, {node_a: 3, node_b:4.5}))

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

init = tf.global_variables_initializer()
# fixW = tf.assign(W, [-1])
# fixb = tf.assign(b, [1])

optimizer = tf.train.GradientDescentOptimizer(0.01)  # learning rate 0.01
train = optimizer.minimize(loss)

# sess.run([fixW, fixb])
sess.run(init)  # reset values to incorrect defaults.
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))
