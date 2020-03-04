import numpy as np
import os

import tensorflow as tf


def _bilinear_initializer(n_channels, kernel_size, cross_channel=False):
    """
    Creates a weight matrix that performs bilinear interpolation.
    :param n_channels: The number of channels, one per semantic class.
    :param kernel_size: The filter size, which is 2x the up-sampling factor,
        eg. a kernel/filter size of 4 up-samples 2x.
    :param cross_channel: Add contribution from all other channels to each channel.
        Defaults to False, meaning that each channel is up-sampled separately without
        contribution from the other channels.
    :return: A tf.constant_initializer with the weight initialized to bilinear interpolation.
    """
    # Make a 2D bilinear kernel suitable for up-sampling of the given (h, w) size.
    upscale_factor = (kernel_size+1)//2
    if kernel_size % 2 == 1:
        center = upscale_factor - 1
    else:
        center = upscale_factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    bilinear = (1-abs(og[0]-center)/upscale_factor) * (1-abs(og[1]-center)/upscale_factor)

    # The kernel filter needs to have shape [kernel_height, kernel_width, in_channels, num_filters]
    weights = np.zeros([kernel_size, kernel_size, n_channels, n_channels])
    if cross_channel:
        for i in range(n_channels):
            for j in range(n_channels):
                weights[:, :, i, j] = bilinear
    else:
        for i in range(n_channels):
            weights[:, :, i, i] = bilinear

    return tf.constant_initializer(value=weights)


class FCN(object):
    def __init__(self, image_shape, n_classes, vgg16_weights_path):
        """
        :param image_shape:
        :param n_classes: The number of semantic classes, excluding the void/ignore class.
        :param vgg16_weights_path:
        """
        self.image_shape = image_shape
        self.n_classes = n_classes
        self.vgg16_weights_path = vgg16_weights_path

    def build_from_vgg(self):
        model_name = 'FCN-8s'
        print('Building {} model...'.format(model_name))
        self.inputs = tf.placeholder(tf.float32, [None, *self.image_shape, 3], name='inputs')
        self.labels = tf.placeholder(tf.float32, [None, *self.image_shape], name='labels')
        self.dropout_rate = tf.placeholder(tf.float32, shape=[], name='rate')
        self._fcn_base()
        self._fcn_32()
        self._fcn_16()
        self.outputs = self._fcn_8()

    def _weight(self, layer_name):
        """
        :param layer_id: get weight from vgg base by layer_name 「conv_1_1, conv_1_2 ...」
        :return:
        """
        W = self.weights[layer_name + '_W']
        b = self.weights[layer_name + '_b']
        return W, b

    def conv2d_relu(self, prev_layer, layer_name, shape=None):
        """
        :param prev_layer:
        :param layer_id:
        :param layer_name:
        :param shape: in fc6 + fc7 layer, must reshape to conv6, conv7 shape
        :return:
        """
        with tf.name_scope(layer_name) as scope:
            W, b = self._weight(layer_name)
            if shape:
                W = W.reshape(shape)
            W = tf.Variable(W, name="weights", dtype=tf.float32)
            b = tf.Variable(b, name="bias", dtype=tf.float32)

            conv2d = tf.nn.conv2d(input=prev_layer,
                                  filter=W,
                                  strides=[1, 1, 1, 1],
                                  padding='SAME')
            out = tf.nn.relu(conv2d + b, name=scope)
        return out

    def maxpool(self, prev_layer, layer_name):
        """ Return the average pooling layer. The paper suggests that
        average pooling works better than max pooling.
        Input:
            prev_layer: the output tensor from the previous layer
            layer_name: the string that you want to name the layer.
                        It's used to specify variable_scope.
        Hint for choosing strides and kszie: choose what you feel appropriate
        """
        with tf.variable_scope(layer_name):
            out = tf.nn.max_pool(prev_layer,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME')
        return out

    def _fcn_base(self):
        """

        :return:
        """
        self.weights = np.load(self.vgg16_weights_path)
        with tf.name_scope('preprocess') as scope:
            # RGB mean pixel
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32,
                               shape=[1, 1, 1, 3], name='img_mean')
            images = self.inputs - mean

        # -------------- Block 1 -----------------
        self.conv1_1 = self.conv2d_relu(self.inputs, 'conv1_1')
        self.conv1_2 = self.conv2d_relu(self.conv1_1, 'conv1_2')
        self.pool1 = self.maxpool(self.conv1_2, 'pool1')

        # -------------- Block 2 -----------------
        self.conv2_1 = self.conv2d_relu(self.pool1, 'conv2_1')
        self.conv2_2 = self.conv2d_relu(self.conv2_1, 'conv2_2')
        self.pool2 = self.maxpool(self.conv2_2, 'pool2')

        # -------------- Block 3 -----------------
        self.conv3_1 = self.conv2d_relu(self.pool2, 'conv3_1')
        self.conv3_2 = self.conv2d_relu(self.conv3_1, 'conv3_2')
        self.conv3_3 = self.conv2d_relu(self.conv3_2, 'conv3_3')
        self.pool3 = self.maxpool(self.conv3_3, 'pool3')
        self.layer3_out = tf.identity(self.pool3, name='layer3_out')

        # -------------- Block 4 -----------------
        self.conv4_1 = self.conv2d_relu(self.pool3, 'conv4_1')
        self.conv4_2 = self.conv2d_relu(self.conv4_1, 'conv4_2')
        self.conv4_3 = self.conv2d_relu(self.conv4_2, 'conv4_3')
        self.pool4 = self.maxpool(self.conv4_3, 'pool4')
        self.layer4_out = tf.identity(self.pool4, name='layer4_out')

        # -------------- Block 5 -----------------
        self.conv5_1 = self.conv2d_relu(self.pool4, 'conv5_1')
        self.conv5_2 = self.conv2d_relu(self.conv5_1, 'conv5_2')
        self.conv5_3 = self.conv2d_relu(self.conv5_2, 'conv5_3')
        self.pool5 = self.maxpool(self.conv5_3, 'pool5')

        # -------------- Block 6 -----------------
        self.conv6 = self.conv2d_relu(self.pool5, 'fc6', (7, 7, 512, 4096))
        self.drop1 = tf.nn.dropout(self.conv6, name='dropout1', rate=self.dropout_rate)

        # -------------- Block 7 -----------------
        self.conv7 = self.conv2d_relu(self.drop1, 'fc7', (1, 1, 4096, 4096))
        self.drop2 = tf.nn.dropout(self.conv7, name='dropout2', rate=self.dropout_rate)
        self.layer7_out = tf.identity(self.drop2, name='layer7_out')

    def _fcn_32(self):
        """
        :return:
        """
        # Apply 1x1 convolution to predict classes of layer 7 at stride 32
        self.conv7_classes = tf.layers.conv2d(self.layer7_out, filters=self.n_classes + 1, kernel_size=1,
                                              kernel_initializer=tf.zeros_initializer(), name="conv7_classes")
        self.fcn32_out = tf.image.resize(self.conv7_classes, self.image_shape, method=tf.image.ResizeMethod.BILINEAR)
        return self.fcn32_out

    def _fcn_16(self):
        """

        :return:
        """
        # Apply 1x1 convolution to predict classes of layer 4 at stride 16
        self.pool4_classes = tf.layers.conv2d(self.layer4_out, filters=self.n_classes + 1, kernel_size=1,
                                              kernel_initializer=tf.zeros_initializer(), name="pool4_classes")

        # Up-sample (2x) conv7 class predictions to match the size of layer 4
        self.fcn32_upsampled = tf.layers.conv2d_transpose(self.conv7_classes, filters=self.n_classes+1,
                                                          kernel_size=4, strides=2, padding='SAME',
                                                          use_bias=False,
                                                          kernel_initializer=_bilinear_initializer(self.n_classes+1, 4),
                                                          name="fcn32_upsampled")

        # Add a skip connection between class predictions of layer 4 and up-sampled class predictions of layer 7
        self.skip_1 = tf.add(self.pool4_classes, self.fcn32_upsampled, name="skip_cnx_1")

        # 16x bilinear interpolation
        self.fcn16_out = tf.image.resize(self.skip_1, self.image_shape, method=tf.image.ResizeMethod.BILINEAR)

        return self.fcn16_out

    def _fcn_8(self):
        """
        :return:
        """
        # Apply 1x1 convolution to predict classes of layer 3 at stride 8
        self.pool3_classes = tf.layers.conv2d(self.layer3_out, filters=self.n_classes + 1, kernel_size=1,
                                              kernel_initializer=tf.zeros_initializer(),
                                              name="pool3_classes")

        # Up-sample (2x) skip_1 class predictions to match the size of layer 3
        self.fcn16_upsampled = tf.layers.conv2d_transpose(self.skip_1, filters=self.n_classes + 1,
                                                          kernel_size=4, strides=2, padding='SAME',
                                                          use_bias=False,
                                                          kernel_initializer=_bilinear_initializer(self.n_classes + 1, 4),
                                                          name="fcn16_upsampled")

        # Add a skip connection between class predictions of layer 4 and up-sampled class predictions of layer 7
        self.skip_2 = tf.add(self.pool3_classes, self.fcn16_upsampled, name="skip_cnx_2")

        # 8x bilinear interpolation
        self.fcn8_out = tf.image.resize(self.skip_2, self.image_shape, method=tf.image.ResizeMethod.BILINEAR)
        return self.fcn8_out

