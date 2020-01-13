import tensorflow as tf


def build_fcn_layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    # Use a shorter variable name for simplicity
    layer3, layer4, layer7 = vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

    # Apply 1x1 convolution in place of fully connected layer
    fcn8 = tf.layers.conv2d(layer7, filters=num_classes, kernel_size=1, name="fcn8")

    # Upsample fcn8 with size depth=(4096?) to match size of layer 4 so that we can add skip connection with 4th layer
    fcn9 = tf.layers.conv2d_transpose(fcn8, filters=layer4.get_shape().as_list()[-1],
                                      kernel_size=4, strides=(2, 2), padding='SAME', name="fcn9")

    # Add a skip connection between current final layer fcn8 and 4th layer
    fcn9_skip_connected = tf.add(fcn9, layer4, name="fcn9_plus_vgg_layer4")

    # Upsample again
    fcn10 = tf.layers.conv2d_transpose(fcn9_skip_connected, filters=layer3.get_shape().as_list()[-1],
                                       kernel_size=4, strides=(2, 2), padding='SAME', name="fcn10_conv2d")

    # Add skip connection
    fcn10_skip_connected = tf.add(fcn10, layer3, name="fcn10_plus_vgg_layer3")

    # Upsample again
    fcn11 = tf.layers.conv2d_transpose(fcn10_skip_connected, filters=num_classes,
                                       kernel_size=16, strides=(8, 8), padding='SAME', name="fcn11")

    return fcn11


def get_optimizer(nn_last_layer, correct_label, learning_rate, num_classes):
    # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="fcn_logits")
    correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))

    # Calculate distance from actual labels using cross entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label_reshaped[:])
    # Take mean for total loss
    loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss")

    # The model implements this operation to find the weights/parameters that would yield correct pixel labels
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="fcn_train_op")

    return logits, train_op, loss_op


class FCNVGG:
    # ---------------------------------------------------------------------------
    def __init__(self, session):
        self.session = session

    def build_from_vgg(self, vgg_path, correct_label, learning_rate, num_classes):
        self.__load_vgg(vgg_path)
        self.__make_result_tensors(correct_label, learning_rate, num_classes)

    def __load_vgg(self, vgg_path):
        model = tf.saved_model.loader.load(self.session, ['vgg16'], vgg_path)
        graph = tf.get_default_graph()

        self.image_input = graph.get_tensor_by_name('image_input:0')
        self.keep_prob = graph.get_tensor_by_name('keep_prob:0')
        self.layer3 = graph.get_tensor_by_name('layer3_out:0')
        self.layer4 = graph.get_tensor_by_name('layer4_out:0')
        self.layer7 = graph.get_tensor_by_name('layer7_out:0')

    def __make_result_tensors(self, correct_label, learning_rate, num_classes):
        self.output = build_fcn_layers(self.layer3, self.layer4, self.layer7, num_classes)
        self.logits, self.train_op, self.loss_op = get_optimizer(self.output, correct_label, learning_rate, num_classes)