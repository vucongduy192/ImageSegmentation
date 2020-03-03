import tensorflow as tf


class FCNVGG:
    # ---------------------------------------------------------------------------
    def __init__(self, session, num_classes):
        self.session = session
        self.num_classes = num_classes

    def build_from_metagraph(self, metagraph_file, checkpoint_file):
        self.__load_fcnvgg(metagraph_file, checkpoint_file)

    def build_from_vgg(self, vgg_path):
        self.__load_vgg(vgg_path)
        self.__make_result_tensors()

    def __load_fcnvgg(self, metagraph_file, checkpoint_file):
        saver = tf.train.import_meta_graph(metagraph_file)
        saver.restore(self.session, checkpoint_file)

        self.image_input = self.session.graph.get_tensor_by_name('image_input:0')
        self.keep_prob = self.session.graph.get_tensor_by_name('keep_prob:0')
        self.logits = self.session.graph.get_tensor_by_name('sum/fcn_logits:0')

        self.softmax = self.session.graph.get_tensor_by_name('result/Softmax:0')
        self.classes = self.session.graph.get_tensor_by_name('result/ArgMax:0')

    def __load_vgg(self, vgg_path):
        model = tf.saved_model.loader.load(self.session, ['vgg16'], vgg_path)
        graph = tf.get_default_graph()

        self.image_input = graph.get_tensor_by_name('image_input:0')
        self.keep_prob = graph.get_tensor_by_name('keep_prob:0')
        self.layer3 = graph.get_tensor_by_name('layer3_out:0')
        self.layer4 = graph.get_tensor_by_name('layer4_out:0')
        self.layer7 = graph.get_tensor_by_name('layer7_out:0')

    def __make_result_tensors(self):
        """
        FCN layer 7 to CONV:(?, ?, ?, 2)
                            (?, ?, ?, 512)
                            (?, ?, ?, 512)
                            (?, ?, ?, 256)
                            (?, ?, ?, 256)
                            (?, ?, ?, 2)
        :param correct_label:
        :param num_classes:
        :return:
        """
        # Use a shorter variable name for simplicity
        layer3, layer4, layer7 = self.layer3, self.layer4, self.layer7
        # Convert FCN to CONV
        # Apply 1x1 convolution in place of fully connected layer
        fcn8 = tf.layers.conv2d(layer7, filters=self.num_classes, kernel_size=1, name="fcn8")
        # Up-sample fcn8 with size depth=(4096?) to match size of layer 4
        # so that we can add skip connection with 4th layer
        fcn9 = tf.layers.conv2d_transpose(fcn8, filters=layer4.get_shape().as_list()[-1],
                                          kernel_size=4, strides=(2, 2), padding='SAME', name="fcn9")
        # Add a skip connection between current final layer fcn8 and 4th layer
        fcn9_skip_connected = tf.add(fcn9, layer4, name="fcn9_plus_vgg_layer4")
        # print(fcn9_skip_connected.shape)
        # Up-sample again
        fcn10 = tf.layers.conv2d_transpose(fcn9_skip_connected, filters=layer3.get_shape().as_list()[-1],
                                           kernel_size=4, strides=(2, 2), padding='SAME', name="fcn10_conv2d")

        # Add skip connection
        fcn10_skip_connected = tf.add(fcn10, layer3, name="fcn10_plus_vgg_layer3")

        # Up-sample again
        # Final output: fcn11 = 8 * (4 * layer_out7 + 2 * layer_out4 + layer_out3)
        fcn11 = tf.layers.conv2d_transpose(fcn10_skip_connected, filters=self.num_classes,
                                           kernel_size=16, strides=(8, 8), padding='SAME', name="fcn11")

        with tf.variable_scope('sum'):
            self.logits = tf.reshape(fcn11, (-1, self.num_classes), name="fcn_logits")

        with tf.name_scope('result'):
            self.softmax = tf.nn.softmax(self.logits)
            self.classes = tf.argmax(self.softmax, axis=3)

    def get_optimizer(self, correct_label, learning_rate=0.0001):
        # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
        correct_label_reshaped = tf.reshape(correct_label, (-1, self.num_classes))

        # Calculate distance from actual labels using cross entropy
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                labels=correct_label_reshaped[:])
        # Take mean for total loss
        loss = tf.reduce_mean(cross_entropy, name="fcn_loss")

        # The model implements this operation to find the weights/parameters that would yield correct pixel labels
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        optimizer = optimizer.minimize(loss, name="fcn_train_op")

        return optimizer, loss
