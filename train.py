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
vgg_dir = './vgg'
log_dir = './tensorboard'


def main():
    source = load_data_source()
    source.load_data(data_dir, validation_size=0.2)
    train_generator = source.train_generator
    valid_generator = source.valid_generator

    epochs = 40
    batch_size = 4
    num_classes = 2
    image_shape = (160, 576)  # height, width

    # Declare some placeholder will use when training
    correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])

    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=session_config) as session:
        # Load pre-trained VGG model
        net = load_fcnvgg(session, num_classes)
        net.build_from_vgg(vgg_dir)
        print("Model build successful, starting train")

        # Save checkpoint
        checkpoint_dir = os.path.join('./saved_model/', datetime.utcnow().strftime("%Y%m%d-%H%M%S"))
        writer = tf.summary.FileWriter(log_dir, session.graph)
        saver = tf.train.Saver(max_to_keep=10)

        optimizer_op, loss_op = net.get_optimizer(correct_label, learning_rate=0.001)
        loss_summary = tf.summary.scalar("Loss", loss_op)

        training_loss = tf.placeholder(tf.float32)
        training_loss_summary_op = tf.summary.scalar('training_loss',
                                                     training_loss)
        validation_loss = tf.placeholder(tf.float32)
        validation_loss_summary_op = tf.summary.scalar('validation_loss',
                                                       validation_loss)

        session.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            print("STARTING EPOCH {} ...".format(epoch + 1))
            # ---------------------------------- #
            # Training process
            training_loss_total = 0
            generator = train_generator(batch_size)
            for X_batch, gt_batch in generator:
                _, loss = session.run([optimizer_op, loss_op],
                                      feed_dict={net.image_input: X_batch, correct_label: gt_batch,
                                                 net.keep_prob: 0.5})

                training_loss_total += loss * X_batch.shape[0]
            training_loss_total /= source.num_training

            # ----------------------------------- #
            # Validation process
            valid_loss_total = 0
            generator = valid_generator(batch_size)
            for X_batch, gt_batch in generator:
                _, loss = session.run([optimizer_op, loss_op],
                                      feed_dict={net.image_input: X_batch, correct_label: gt_batch,
                                                 net.keep_prob: 1.})

                valid_loss_total += loss * X_batch.shape[0]
            valid_loss_total /= source.num_validation
            # ----------------------------------- #

            print("EPOCH {} ...".format(epoch + 1))
            print("Training loss = {:.3f}".format(training_loss_total))
            print("Validation loss = {:.3f}".format(valid_loss_total))

            # ----------------------------------- #
            # Write loss summary
            feed = {validation_loss: valid_loss_total,
                    training_loss: training_loss_total}
            loss_summary = session.run([training_loss_summary_op,
                                        validation_loss_summary_op],
                                       feed_dict=feed)
            summary_op = tf.summary.merge([loss_summary[0], loss_summary[1]])
            writer.add_summary(summary_op.eval(), epoch)

            if (epoch + 1) % 5 == 0:
                saver.save(session, checkpoint_dir, epoch)
            # ----------------------------------- #


if __name__ == '__main__':
    main()
