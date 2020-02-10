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
model_dir = './saved_model/20200208-165121/'
test_txt = os.path.join(data_dir, 'ImageSets/Segmentation/val.txt')  # 1464 images name

batch_size = 8


def draw_labels(img, label, label_colors):
    img_save = np.zeros_like(img)
    for val, color_rgb in label_colors.items(): # diningtable => rgb(124, 124, 0)
        label_mask = label == list(label_colors.keys()).index(val)
        img_save[label_mask] = color_rgb
    return img_save


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

    checkpoint_file = state.all_model_checkpoint_paths[-1]
    checkpoint_file = checkpoint_file.replace('/content/drive/My Drive/dataset/FCN/', './')
    print(checkpoint_file)
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
            y_batch = session.run(net.classes, feed_dict=feed)
            y_batch = y_batch.reshape(batch_size, source.image_size[1], source.image_size[0])

            for idx in range(X_batch.shape[0]):
                img_label = draw_labels(X_batch[idx], y_batch[idx], source.label_colors)
                cv2.imwrite(output_dir + '/' + names_batch[idx], img_label)
                cv2.imwrite(output_dir + '/source_' + names_batch[idx], X_batch[idx])
                break
            break
