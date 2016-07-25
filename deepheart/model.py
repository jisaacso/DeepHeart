import tensorflow as tf
import os
from datetime import datetime


class CNN:
    def __init__(self, pcg, nclasses=2, learning_rate=0.001,
                 epochs=5, batch_size=100, dropout=0.75, base_dir="/tmp",
                 model_name="cnn"):
        self.pcg = pcg
        self.nclasses = nclasses
        self.d_input = self.pcg.train.X.shape[1]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.nbatches = int(self.pcg.train.X.shape[0] / float(self.batch_size))
        self.model_name = model_name
        self.base_dir = base_dir

    def train(self):
        """
        Train a convolutional neural network over the input PCG dataset.
        This method is beefy: it is responsible for defining tensorflow
        variables, defining the training objective function, defining summary
        statistics creating the tensorflow session, running gradient
        descent and, ultimately, writing statistics

        In the future this will be refactored into more easily tested
        training segments.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        print('begin train')
        print(self.__get_output_name())

        with tf.name_scope('input'):
            X = tf.placeholder(tf.float32, [None, self.d_input], name='X')
            y = tf.placeholder(tf.float32, [None, self.nclasses], name='y')
            do_drop = tf.placeholder(tf.float32, name='drop')

        with tf.name_scope('weights'):
            weights = {
                'wc1': tf.Variable(tf.random_normal([5, 1, 1, 32]), name='wc1'),
                'wc2': tf.Variable(tf.random_normal([5, 1, 32, 64]), name='wc2'),
                # 2 Max pools have taken original 10612 signal down to
                # 5306 --> 2653. Each max pool has a ksize=2.
                # 'wd1': tf.Variable(tf.random_normal([2653 * 1 * 64, 1024])),
                'wd1': tf.Variable(tf.random_normal([int(self.d_input / 4) * 1 * 64, 1024]), name='wd1'),
                'out': tf.Variable(tf.random_normal([1024, self.nclasses]), name='outW')
            }
        with tf.name_scope('biases'):
            biases = {
                'bc1': tf.Variable(tf.random_normal([32]), name='bc1'),
                'bc2': tf.Variable(tf.random_normal([64]), name='bc2'),
                'bd1': tf.Variable(tf.random_normal([1024]), name='bd1'),
                'out': tf.Variable(tf.random_normal([self.nclasses]), name='outB')
            }

        with tf.name_scope('pred'):
            pred = self.model1D(X, weights, biases, do_drop)

        with tf.name_scope('cost'):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y, name='cost'))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        dim = tf.shape(y)[0]

        with tf.name_scope('sensitivity'):
            # sensitivity = correctly predicted abnormal / total number of actual abnormal
            abnormal_idxs = tf.cast(tf.equal(tf.argmax(pred, 1), 1), tf.float32)
            pred1d = tf.reshape(tf.slice(y, [0, 1], [dim, 1]), [-1])
            abn = tf.mul(pred1d, abnormal_idxs)
            sensitivity = tf.reduce_sum(abn) / tf.reduce_sum(pred1d)
            tf.scalar_summary('sensitivity', sensitivity)

        with tf.name_scope('specificity'):
            # specificity = correctly predicted normal / total number of actual normal
            normal_idxs = tf.cast(tf.equal(tf.argmax(pred, 1), 0), tf.float32)
            pred1d_n = tf.reshape(tf.slice(y, [0, 0], [dim, 1]), [-1])
            normal = tf.mul(pred1d_n, normal_idxs)
            specificity = tf.reduce_sum(normal) / tf.reduce_sum(pred1d_n)
            tf.scalar_summary('specificity', sensitivity)

        # Physionet score is the mean of sensitivity and specificity
        score = (sensitivity + specificity) / 2.0
        tf.scalar_summary('score', score)

        init = tf.initialize_all_variables()

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)

            merged = tf.merge_all_summaries()
            train_writer = tf.train.SummaryWriter(os.path.join(self.base_dir, 'train'), sess.graph)

            for epoch in range(self.epochs):
                avg_cost = 0
                for batch in range(self.nbatches):
                    batch_x, batch_y = self.pcg.get_mini_batch(self.batch_size)
                    summary, _, c = sess.run([merged, optimizer, cost],
                                             feed_dict={X: batch_x,
                                                        y: batch_y,
                                                        do_drop: self.dropout})
                    train_writer.add_summary(summary, epoch*batch)
                    avg_cost += c
                avg_cost /= float(self.nbatches)
                print('Epoch %s\tcost %s' % (epoch, avg_cost))

                if epoch % 10 == 0:
                    acc, sens, spec = sess.run([score, sensitivity, specificity],
                                                feed_dict={X: self.pcg.test.X,
                                                           y: self.pcg.test.y,
                                                           do_drop: 1.})
                    print('Score %s\tSensitivity %s\tSpecificity %s' % (acc, sens, spec))

                    saver.save(sess, self.__get_output_name())
                    print('Epoch written')

    def __get_output_name(self):
        now = datetime.now()
        time_str = "-%s" % (now.date())  # now.hour, now.minute, now.second)
        model_path = os.path.join(self.base_dir, self.model_name + time_str + '.tnfl')
        return model_path

    def conv2d(self, x, w, b, strides=1):
        """
        A small helper function for calcualting a 1D convolution
        from tensorflow's conv2d method

        Parameters
        ----------
        x: tensorflow.placeholder
            The feature vector
        w: tensorflow.Variable
            The unknown weights to learn
        b: tensorflow.Variable
            The unknown biases to learn
        strides: int
            The length of the stride to use for convolution

        Returns
        -------
        tensorflow.Variable
            A convolution over the input feature vector

        """


        x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding="SAME")
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def model1D(self, x, weights, biases, dropout):
        """
        A Wrapper to chain several TensorFlow convolutional units together. This 1D model
        ultimately calls TensorFlow's conv2d, mapping a 1D feature vector to a collapsed
        2D convolution

        Parameters
        ----------
        x: tensorflow.placeholder
            A feature vector of size [None, no_features]

        weights: dict<str, tensorflow.Variable>
            Dictionary of Unknown weights to learn

        biases: dict<str, tensorflow.Variable>
            Dictionary of unknown biases to learn

        dropout: float
            the dropout fraction for convolutional units

        Returns
        -------
        out: tensorflow.Variable
            The result of applying multiple convolutional layers and
            a fully connected unit to the input feature vector

        """

        with tf.name_scope('reshape'):
            x = tf.reshape(x, shape=[-1, self.d_input, 1, 1])  # [n_images, width, height, n_channels]

        with tf.name_scope('conv1'):
            conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
            conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')
            conv1 = tf.nn.relu(conv1)

        with tf.name_scope('conv2'):
            conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
            conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding="SAME")
            conv2 = tf.nn.relu(conv2)

        with tf.name_scope('fullyConnected'):
            d_layer1 = weights['wd1'].get_shape().as_list()[0]
            fc1 = tf.reshape(conv2, [-1, d_layer1])
            fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
            fc1 = tf.nn.relu(fc1)
            fc1 = tf.nn.dropout(fc1, dropout)

            out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out
