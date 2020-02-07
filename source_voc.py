import random
import cv2
import os
# from skimage.io import imshow
# import matplotlib.pyplot as plt

import numpy as np


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def color_map_dict():
    labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
              'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
              'train', 'tvmonitor', 'void']
    # print(len(labels))
    return dict(zip(labels[:-1], color_map()[:-1]))


class VOCSource(object):
    def __init__(self):
        self.num_training = None
        self.num_validation = None
        self.train_generator = None
        self.valid_generator = None

        self.num_testing = None
        self.test_generator = None

        self.label_colors = color_map_dict()
        self.num_classes = len(self.label_colors)
        self.image_size = (224, 224)

    # -------------------------------------------------------------------------------
    def load_data(self, data_dir, images_txt, validation_size=None):
        images_dir = os.path.join(data_dir, './JPEGImages/')   # very large
        labels_dir = os.path.join(data_dir, './SegmentationClass/')  # 2913

        # train.txt: 1464 , val.txt: 1449 , Segmentation: 2913
        with open(images_txt, 'r') as f:
            image_names = f.readlines()
            image_paths, label_paths = [], {}
            for image in image_names:
                image_path = images_dir + image[:-1] + '.jpg'
                image_paths.append(image_path)
                label_paths[os.path.basename(image_path)] = labels_dir + image[:-1] + '.png'

        random.shuffle(image_paths)
        if validation_size is None:
            test_images = image_paths
            self.num_testing = len(test_images)
            self.test_generator = self.batch_generator(test_images, label_paths)
        else:
            num_images = len(image_paths)
            valid_images = image_paths[:int(validation_size * num_images)]
            train_images = image_paths[int(validation_size * num_images):]

            self.num_training = len(train_images)
            self.num_validation = len(valid_images)
            self.train_generator = self.batch_generator(train_images, label_paths)
            self.valid_generator = self.batch_generator(valid_images, label_paths)

    # -------------------------------------------------------------------------------
    def batch_generator(self, image_paths, label_paths):
        def gen_batch(batch_size=8):
            """
            :param batch_size:
            :return: generator of a batch contain (images, labels)
            """
            random.shuffle(image_paths)
            for offset in range(0, len(image_paths), batch_size):
                files = image_paths[offset:(offset + batch_size)]

                images = []
                labels = []
                names = []
                for image_file in files:
                    label_file = label_paths[os.path.basename(image_file)]  # base name get file name from path
                    image = cv2.cvtColor(cv2.resize(cv2.imread(image_file), self.image_size), cv2.COLOR_BGR2RGB)
                    label = cv2.cvtColor(cv2.resize(cv2.imread(label_file), self.image_size), cv2.COLOR_BGR2RGB)

                    # create label_all:  array of pixels (160, 576, 2),
                    # each pixel is (1, 0) or (0, 1) if is_road or is_background
                    label_obj = []
                    for obj, color_rgb in self.label_colors.items():
                        obj = np.all(label == color_rgb, axis=2)
                        label_obj.append(obj)

                    label_all = np.dstack(label_obj)
                    label_all = label_all.astype(np.float32)

                    images.append(image.astype(np.float32))
                    labels.append(label_all)
                    names.append(os.path.basename(image_file))
                yield np.array(images), np.array(labels), np.array(names)

        return gen_batch

# if __name__ == '__main__':
#     source = VOCSource(training=False)
#     source.load_data()
#     train_generator = source.train_generator
#     valid_generator = source.valid_generator
#     generator = train_generator(4)
#
#     for X_batch, gt_batch in generator:
#         print(gt_batch[0].shape)
#         print(gt_batch[0][112, 112, :])
#         break

# print(os.path.basename(image_file))
# fig = plt.figure(figsize=(8, 8))
# fig.add_subplot(1, 2, 1)
# plt.imshow(image)
# fig.add_subplot(1, 2, 2)
# plt.imshow(label)
# print(label[0, 0, :])
# plt.show()
