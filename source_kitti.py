import random
import cv2
import os
from sklearn.model_selection import train_test_split

import numpy as np

from glob import glob


class KittiSource:
    # ---------------------------------------------------------------------------
    def __init__(self):
        """
            Declare dataset configuration
            Divide dataset to training && testing, load by batch size
        """
        self.num_classes = 2
        self.image_size = (576, 160)  # (width , height)
        self.label_colors = {0: np.array([0, 0, 0]), 1: np.array([255, 0, 255])}

        self.num_training = None
        self.num_validation = None
        self.train_generator = None
        self.valid_generator = None

    # -------------------------------------------------------------------------------
    def load_data(self, data_dir, validation_size=0.2):
        """
        Load all image_paths from data_dir, divide a part to validation set
        Using batch_generator make 2 set corresponding train_generator && validation_generator
        After that, send batch_size to use inner function in generator (batch_generator --> 「gen_batch」)

        :param data_dir: the directory where the dataset's file are stored
        :param validation_size:
        :return:
        """
        images = data_dir + '/training/image_2/*.png'
        labels = data_dir + '/training/gt_image_2/*_road_*.png'

        image_paths = glob(images)
        label_paths = {
            os.path.basename(path).replace('_road_', '_'): path
            for path in glob(labels)}

        random.shuffle(image_paths)

        num_images = len(image_paths)
        valid_images = image_paths[:int(validation_size * num_images)]
        train_images = image_paths[int(validation_size * num_images):]

        self.num_training = len(train_images)
        self.num_validation = len(valid_images)
        self.train_generator = self.batch_generator(train_images, label_paths)
        self.valid_generator = self.batch_generator(valid_images, label_paths)

    def batch_generator(self, image_paths, label_paths):
        def gen_batch(batch_size):
            """
            :param batch_size:
            :return: generator of a batch contain (images, labels)
            """
            road_color = np.array([255, 0, 255])
            random.shuffle(image_paths)
            for offset in range(0, len(image_paths), batch_size):
                files = image_paths[offset:(offset + batch_size)]

                images = []
                labels = []
                for image_file in files:
                    label_file = label_paths[os.path.basename(image_file)]  # base name get file name from path

                    image = cv2.resize(cv2.imread(image_file), self.image_size)
                    label = cv2.resize(cv2.imread(label_file), self.image_size)

                    # create label_all:  array of pixels (160, 576, 2),
                    # each pixel is (1, 0) or (0, 1) if is_road or is_background
                    label_road = np.all(label == road_color, axis=2)
                    label_bg = np.any(label != road_color, axis=2)
                    label_all = np.dstack([label_bg, label_road])
                    label_all = label_all.astype(np.float32)

                    images.append(image.astype(np.float32))
                    labels.append(label_all)
                yield np.array(images), np.array(labels)

        return gen_batch
