#!/usr/bin/env python3
from utils import *
from datetime import datetime

import numpy as np
import cv2
import sys
import os

import tensorflow as tf

print('Tensorflow version: {}'.format(tf.__version__))

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

# Specify all directory paths
data_dir = './VOC2012'
vgg_dir = './vgg'
log_dir = './tensorboard'
output_dir = './output'
model_dir = './saved_model/20200206-042657/'
test_txt = os.path.join(data_dir, 'ImageSets/Segmentation/val.txt')  # 1464 images name

batch_size = 8

if __name__ == '__main__':
    source = load_voc_source()
    source.load_data(data_dir, test_txt)
    print(source.num_testing)
    print(source.label_colors)
    test_generator = source.test_generator

    state = tf.train.get_checkpoint_state(model_dir)
    if state is None:
        print('[!] No network state found in ' + model_dir)
        sys.exit(1)

    checkpoint_file = './saved_model/20200206-042657/epoch10.ckpt'
    metagraph_file = checkpoint_file + '.meta'
    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=session_config) as session:
        session.run(tf.global_variables_initializer())

        net = load_fcnvgg(session, source.num_classes)
        net.build_from_metagraph(metagraph_file, checkpoint_file)
        print("Model load successful, starting test")

        generator = test_generator(batch_size)
        for X_batch, gt_batch, names_batch in generator:
            feed = {net.image_input: X_batch, net.keep_prob: 1}
            image = session.run(net.logits, feed_dict=feed)

            image_softmax = tf.nn.softmax(image)
            session.run(image_softmax)
            predict = image_softmax.eval()
            image_reshaped = image_softmax.eval().reshape(batch_size, source.image_size[1], source.image_size[0], 2)
            for idx in range(image_reshaped.shape[0]):
                image_src = X_batch[idx]
                image_labelled = np.argmax(image_reshaped[idx], axis=2)
                image_save = np.zeros((source.image_size[1], source.image_size[0], 3))

                for val, color in source.label_colors.items():
                    label_mask = image_labelled == val
                    if val == 0:  # background pixel
                        image_save[label_mask] = image_src[label_mask]
                    else:
                        image_save[label_mask] = color

                cv2.imwrite(output_dir + '/' + names_batch[idx], image_save)
            break
