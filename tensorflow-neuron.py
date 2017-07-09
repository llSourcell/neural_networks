# Arda Mavi
import tensorflow as tf

# Inputs:
x = tf.placeholder(tf.float32, [None, 2])
y = tf.placeholder(tf.float32, [None, 1])

# Weight:
w = tf.Variable(tf.random_normal([2,1]), dtype=tf.float32)

# Bias:
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32, trainable=False)

# Relu:
pred = tf.nn.relu(tf.add(tf.matmul(x,w), b))

# Loss:
loss = tf.reduce_mean(tf.contrib.losses.mean_squared_error(pred, y))

# Optimizer:
learning_rate = 0.001
opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()


X_train, Y_train = # TODO: Data

epochs = 100
bach_size = 5

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        for (bach_x, bach_y) in zip(X_train[epoch*bach_size:(epoch+1)*bach_size], Y_train[epoch*bach_size:(epoch+1)*bach_size]):
            sess.run(opt, feed_dict={x:[bach_x], y:[bach_y]})
        c = sess.run(loss, feed_dict={x:X_train, y:Y_train})
        print("Epoch:", '%04d' % (epoch+1), "Loss=", "{:.4f}".format(c))
