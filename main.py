#!/usr/bin/env python3

import cv2
import numpy as np

import tensorflow as tf
print ('Tensorflow version: {}'.format(tf.__version__))

from utils import *

# Specify all directory paths
data_dir = './data_road'
training_dir = './data/data_road/training'
testing_dir = './data/data_road/testing'
vgg_path = './vgg'

source = load_data_source()
source.load_data(data_dir, validation_size=0.2)
train_generator = source.train_generator
valid_generator = source.valid_generator

batch_generator = train_generator(16)
for X_batch, y_batch in batch_generator:
    print (X_batch.shape)
    print (X_batch[0].shape)
    print (y_batch.shape)
    print (y_batch[0].shape)
    break
