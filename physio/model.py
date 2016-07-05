import tensorflow as tf
import numpy as np
import os


class CNN:
    def __init__(self, pcg, nclasses=2, learning_rate=0.001,
                 epochs=5, batch_size=100, dropout=0.75, model_name="cnn.tnfl"):
        self.pcg = pcg
        self.nclasses = nclasses
        self.d_input = self.pcg.train.X.shape[1]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.nbatches = int(self.pcg.train.X.shape[0] / float(self.batch_size))
        self.model_path = os.path.join("/Users/jisaacso/projects/physio/data/", model_name)

    def train(self):

        saver = tf.train.Saver()

        X = tf.placeholder(tf.float32, [None, self.d_input])
        y = tf.placeholder(tf.float32, [None, self.nclasses])
        do_drop = tf.placeholder(tf.float32)

        weights = {
            'wc1': tf.Variable(tf.random_normal([5, 1, 1, 32])),
            'wc2': tf.Variable(tf.random_normal([5, 1, 32, 64])),
            # 2 Max pools have taken original 10612 signal down to
            # 5306 --> 2653. Each max pool has a ksize=2.
            'wd1': tf.Variable(tf.random_normal([2653 * 1 * 64, 1024])),
            'out': tf.Variable(tf.random_normal([1024, self.nclasses]))
        }
        biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([self.nclasses]))
        }


        pred = self.model1D(X, weights, biases, do_drop)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y, name='cost'))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        is_correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.epochs):
                avg_cost = 0
                for batch in range(self.nbatches):
                    batch_x, batch_y = self.pcg.get_mini_batch(self.batch_size)
                    _, c = sess.run([optimizer, cost], feed_dict={X: batch_x, y: batch_y,
                                                                  do_drop: self.dropout})
                    avg_cost += c
                avg_cost /= float(self.nbatches * 9)
                print 'Epoch %s\tcost %s' % (epoch, avg_cost)

            print 'Accuracy %s' % (sess.run(accuracy, {X: self.pcg.test.X,
                                                       y: self.pcg.test.y,
                                                       do_drop: 1.}))
            saver.save(sess, self.model_path)

    def conv2d(self, x, w, b, strides=1):
        x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding="SAME")
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def model1D(self, x, weights, biases, dropout):
        x = tf.reshape(x, shape=[-1, 10611, 1, 1])  # [n_images, width, height, n_channels]

        conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding="SAME")

        d_layer1 = weights['wd1'].get_shape().as_list()[0]
        fc1 = tf.reshape(conv2, [-1, d_layer1])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, dropout)

        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out


    def model1DSplits(self, x, weights, biases, dropout):
        """
                            splits = np.linspace(0, batch_x.shape[1], 10)
                    for s_idx in range(len(splits)-1):
                        split = batch_x[:, splits[s_idx]:splits[s_idx+1]]

        """

        x = tf.reshape(x, shape=[-1, 10611, 1, 1])  # [n_images, width, height, n_channels]

        conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding="SAME")

        d_layer1 = weights['wd1'].get_shape().as_list()[0]
        fc1 = tf.reshape(conv2, [-1, d_layer1])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, dropout)

        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out
