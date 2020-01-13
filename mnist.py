import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math
import time
# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.logging.set_verbosity(tf.logging.ERROR)

# Check if gpu working
# import tensorflow as tf
# with tf.device('/gpu:0'):
#     a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#     b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#     c = tf.matmul(a, b)
# with tf.Session() as sess:
#     print (sess.run(c))

class Model:

    def __init__(self, image, label):
        """
        A Model class contains a computational graph that classifies images
        to predictions. Each of its methods builds part of the graph
        on Model initialization. Do not modify the constructor, as doing so
        would break the autograder. You may, however, add class variables
        to use in your graph-building. e.g. learning rate,

        image: the input image to the computational graph as a tensor
        label: the correct label of an image as a tensor
        prediction: the output prediction of the computational graph,
                    produced by self.forward_pass()
        optimize: the model's optimizing tensor produced by self.optimizer()
        loss: the model's loss produced by computing self.loss_function()
        accuracy: the model's prediction accuracy
        """
        self.image = image
        self.label = label

        # TO-DO: Add any class variables you want to use.

        self.prediction = self.forward_pass()
        self.loss = self.loss_function()
        self.optimize = self.optimizer()
        self.accuracy = self.accuracy_function()

    def forward_pass(self):
        """
        Predicts a label given an image using convolution layers

        :return: the prediction as a tensor
        """
        filter_1 = tf.Variable(tf.truncated_normal([3, 3, 1, 8], stddev=0.1))
        conv_1 = tf.nn.conv2d(self.image, filter_1, [1, 1, 1, 1], "SAME")

        reshaped = tf.reshape(conv_1, shape=[50, -1])

        L1 = reshaped.shape[1].value
        L2 = 500
        W1 = tf.Variable(tf.random_normal([L1, L2], mean=0, stddev=0.01))
        b1 = tf.Variable(tf.random_normal([L2], mean=0, stddev=0.01))
        relu_1 = tf.nn.relu(tf.matmul(reshaped, W1) + b1)

        W2 = tf.Variable(tf.random_normal([L2, 10], mean=0, stddev=0.01))
        b2 = tf.Variable(tf.random_normal([10], mean=0, stddev=0.01))
        logits = tf.nn.relu(tf.matmul(relu_1, W2) + b2)
        return logits

    def loss_function(self):
        """
        Calculates the model cross-entropy loss

        :return: the loss of the model as a tensor
        """
        loss = tf.losses.softmax_cross_entropy(onehot_labels=self.label, logits=self.prediction)
        return loss

    def optimizer(self):
        """
        Optimizes the model loss using an Adam Optimizer

        :return: the optimizer as a tensor
        """
        learning_rate = 0.1
        sgd = tf.train.GradientDescentOptimizer(learning_rate)
        train = sgd.minimize(self.loss)
        return train

    def accuracy_function(self):
        """
        Calculates the model's prediction accuracy by comparing
        predictions to correct labels – no need to modify this

        :return: the accuracy of the model as a tensor
        """
        correct_prediction = tf.equal(tf.argmax(self.prediction, 1),
                                      tf.argmax(self.label, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def main():
    t_start = time.time()

    mnist = input_data.read_data_sets("data/mnist/", one_hot=True)
    batch_sz = 50
    batch = 2000

    inputs = tf.placeholder(shape=[batch_sz, 28, 28, 1], dtype=tf.float32)
    labels = tf.placeholder(shape=[batch_sz, 10], dtype=tf.float32)

    model = Model(inputs, labels)

    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=session_config)

    # sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    for i in range(batch):
        next_image, next_label = mnist.train.next_batch(batch_sz)
        next_image = next_image.reshape((batch_sz, 28, 28, 1))
        sess.run(model.optimize, feed_dict={inputs: next_image, labels: next_label})

    acc, test_images, test_labels = 0, mnist.test.images, mnist.test.labels
    test_batch = math.ceil(len(test_images) / batch_sz)
    for i in range(test_batch):
        batch_images = test_images[i * batch_sz: (i + 1) * batch_sz]
        batch_images = batch_images.reshape((batch_sz, 28, 28, 1))
        batch_labes = test_labels[i * batch_sz: (i + 1) * batch_sz]
        acc += sess.run(model.accuracy, feed_dict={inputs: batch_images, labels: batch_labes})
    acc /= test_batch
    print(acc)

    print(time.time() - t_start, 'seconds')

    return


if __name__ == '__main__':
    main()