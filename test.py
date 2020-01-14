from utils import *

import cv2
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob

import tensorflow as tf

print('Tensorflow version: {}'.format(tf.__version__))

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)


# -------------------------------------------------------------------------------
def sample_generator(samples, img_size, batch):
    for offset in range(0, len(samples), batch):
        files = samples[offset:offset + batch]
        images = []
        names = []
        for image_file in files:
            image = cv2.resize(cv2.imread(image_file), img_size)
            images.append(image.astype(np.float32))
            names.append(os.path.basename(image_file))
        yield np.array(images), names


# Specify all directory paths
output_dir = './data_road/testing/output'
data_dir = './data_road'
vgg_path = './vgg'
model_dir = './saved_model/20200114'
samples_dir = './data_road/testing/image_2/*.png'
sample_paths = glob(samples_dir)

batch_size = 4
image_size = (576, 160)
label_colors = {0: np.array([0, 0, 0]), 1: np.array([255, 0, 255])}


def main():
    state = tf.train.get_checkpoint_state(model_dir)
    if state is None:
        print('[!] No network state found in ' + model_dir)
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    checkpoint_file = state.all_model_checkpoint_paths[-1]
    metagraph_file = checkpoint_file + '.meta'

    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=session_config) as session:
        net = load_fcnvgg(session)
        net.build_from_metagraph(metagraph_file, checkpoint_file)
        print("Model load successful, starting test")

        for X_batch, names in sample_generator(sample_paths, image_size, batch_size):
            feed = {net.image_input: X_batch, net.keep_prob: 1}
            image = session.run(net.logits, feed_dict=feed)

            image_softmax = tf.nn.softmax(image)
            session.run(image_softmax)
            predict = image_softmax.eval()

            image_reshaped = image_softmax.eval().reshape(batch_size, image_size[1], image_size[0], 2)
            for idx in range(image_reshaped.shape[0]):
                image_labelled = np.argmax(image_reshaped[idx], axis=2)
                image_save = np.zeros((image_size[1], image_size[0], 3))
                for val, color in label_colors.items():
                    label_mask = image_labelled == val
                    image_save[label_mask] = color
                cv2.imwrite(output_dir + '/' + names[idx], image_save)
            break


if __name__ == '__main__':
    main()
