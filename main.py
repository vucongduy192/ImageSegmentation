#!/usr/bin/env python3
from utils import *

import cv2
import numpy as np
from tqdm import tqdm
import os

import tensorflow as tf
print('Tensorflow version: {}'.format(tf.__version__))

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

# Specify all directory paths
data_dir = './data_road'
vgg_path = './vgg'


def main():
    source = init_data_source()
    source.load_data(data_dir, validation_size=0.2)
    train_generator = source.train_generator
    valid_generator = source.valid_generator

    epochs = 10
    batch_size = 4
    num_classes = 2
    image_shape = (160, 576)  # height, width

    # Declare some placeholder will use when training
    correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
    learning_rate = tf.placeholder(tf.float32)
    # keep_prob = tf.placeholder(tf.float32)

    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=session_config)

    # Load pre-trained VGG model
    model = init_fcnvgg(session)
    model.build_from_vgg(vgg_path, correct_label, learning_rate, num_classes)
    print("Model build successful, starting training")

    saver = tf.train.Saver(max_to_keep=10)
    session.run(tf.global_variables_initializer())
    keep_prob_value = 0.5
    learning_rate_value = 0.001

    for epoch in range(epochs):
        print("START EPOCH {} ...".format(epoch + 1))
        # Create function to get batches
        total_loss = 0
        generator = train_generator(batch_size)
        for X_batch, gt_batch in generator:
            loss, _ = session.run([model.cross_entropy_loss, model.train_op],
                                  feed_dict={model.image_input: X_batch, correct_label: gt_batch,
                                             model.keep_prob: keep_prob_value, model.learning_rate: learning_rate_value})

            total_loss += loss;

        print("EPOCH {} ...".format(epoch + 1))
        print("Loss = {:.3f}".format(total_loss))

        if (epoch + 1) % 5 == 0:
            checkpoint = './saved_model/{}/epoch{}.ckpt'.format('13012020UTC', epoch + 1)
            saver.save(session, checkpoint)
            print('Checkpoint saved:', checkpoint)


if __name__ == '__main__':
    main()
