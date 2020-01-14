#!/usr/bin/env python3
from utils import *

import cv2
from datetime import datetime
import numpy as np
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
    source = load_data_source()
    source.load_data(data_dir, validation_size=0.2)
    train_generator = source.train_generator
    valid_generator = source.valid_generator

    epochs = 10
    batch_size = 4
    num_classes = 2
    image_shape = (160, 576)  # height, width

    # Declare some placeholder will use when training
    correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
    # learning_rate = tf.placeholder(tf.float32)
    # keep_prob = tf.placeholder(tf.float32)

    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=session_config) as session:
        # Load pre-trained VGG model
        net = load_fcnvgg(session)
        net.build_from_vgg(vgg_path, correct_label, num_classes)
        print("Model build successful, starting training")

        # Save checkpoint
        saver = tf.train.Saver(max_to_keep=10)
        optimizer, loss = net.get_optimizer(correct_label, num_classes)

        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        for epoch in range(epochs):
            print("START EPOCH {} ...".format(epoch + 1))
            # Create function to get batches
            total_loss = 0
            generator = train_generator(batch_size)
            for X_batch, gt_batch in generator:
                _, loss_batch = session.run([optimizer, loss],
                                            feed_dict={net.image_input: X_batch, correct_label: gt_batch,
                                                       net.keep_prob: 0.5})

                total_loss += loss_batch

            print("EPOCH {} ...".format(epoch + 1))
            print("Loss = {:.3f}".format(total_loss))

            if (epoch + 1) % 5 == 0:
                checkpoint = './saved_model/{}/epoch{}.ckpt'.format(datetime.utcnow().strftime("%Y%m%d"), epoch + 1)
                saver.save(session, checkpoint)
                print('Checkpoint saved:', checkpoint)


if __name__ == '__main__':
    main()
