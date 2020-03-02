import numpy as np
import cv2
import os
import random

import tensorflow as tf

print('Tensorflow version: {}'.format(tf.__version__))

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)


def color_map_value(n_classes=256, normalized=False):
    """
    Builds the PASCAL VOC color map for the specified number of classes.
    :param n_classes: the number of classes in the colormap
    :param normalized: normalize pixel intensities, default is False
    :return: a list of RGB colors
    """

    def _bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((n_classes, 3), dtype=dtype)
    for i in range(n_classes):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (_bitget(c, 0) << 7 - j)
            g = g | (_bitget(c, 1) << 7 - j)
            b = b | (_bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def color_map(n_classes):
    """
    Builds the standard 21 class PASCAL VOC color map, plus one additional
    void/ignore label appended to the end of the list.
    :return: A list of RGB values.
    """
    cmap = color_map_value()
    cmap = np.vstack([cmap[:n_classes], cmap[-1].reshape(1, 3)])
    return cmap


def colors2labels(img, cmap):
    labels = np.zeros(img.shape[:-1], dtype='uint8')
    for i, color in enumerate(cmap):
        labels += i * np.all(img == color, axis=2).astype(dtype='uint8')

    return labels


class VOCDataset():
    """Dataset class for PASCAL VOC 2012."""

    def __init__(self, augmentation_params):
        self.augmentation_params = augmentation_params
        self.image_shape = (512, 512)  # Image is padded to obtain a shape divisible by 32.
        self.n_classes = 21  # Excluding the ignore/void class
        self.class_labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                             'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                             'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void']
        self.n_images = {
            'train': 1464,
            'val': 1449
        }
        self.cmap = color_map(self.n_classes)
        assert len(self.cmap) == (self.n_classes + 1), 'Invalid number of colors in cmap'

    def get_basenames(self, filename, dataset_path):
        assert filename in ('train', 'val', 'test')
        filename = os.path.join(dataset_path, 'ImageSets/Segmentation/', filename + '.txt')
        return [line.rstrip() for line in open(filename)]

    def export_sparse_encoding(self, filename, dataset_path):
        """
        Converts ground truth images to sparse labels and saves them to disk in PNG format.
        Each pixel coressponding index in cmap and has format: [5, 5, 5]

        :param filename:
        :param dataset_path:
        :return:
        """
        print('Encoding images...')
        basenames = self.get_basenames(filename, dataset_path)

        gt_path = os.path.join(dataset_path, 'SegmentationClass')
        gt_sparse_path = os.path.join(dataset_path, 'SegmentationSparseClass')

        for basename in basenames:
            gt = cv2.cvtColor(cv2.imread(os.path.join(gt_path, basename + '.png')),
                              cv2.COLOR_BGR2RGB)
            gt = colors2labels(gt, self.cmap)
            gt = np.dstack([gt, np.copy(gt), np.copy(gt)])
            cv2.imwrite(os.path.join(gt_sparse_path, basename + '.png'),
                        cv2.cvtColor(gt, cv2.COLOR_RGB2BGR))

    def export_tfrecord(self, filename, dataset_path, tfrecord_filename):
        print('Loading images...')
        basenames = self.get_basenames(filename, dataset_path)

        # Create folder for TF records
        tfrecords_path = os.path.join(dataset_path, 'TFRecords')
        img_set, gt_set, shape_set = [], [], []
        for basename in basenames:
            img_path = os.path.join(dataset_path, 'JPEGImages', basename + '.jpg')
            gt_path = os.path.join(dataset_path, 'SegmentationClass', basename + '.png')

            with open(img_path, 'rb') as file:
                img = file.read()
            gt = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
            gt_shape = gt.shape
            gt = colors2labels(gt, self.cmap)

            shape_set.append(gt_shape)
            img_set.append(img)
            gt_set.append(gt)

        print('Saving to ' + tfrecord_filename)
        self._export(img_set, gt_set, shape_set, os.path.join(tfrecords_path, tfrecord_filename))

    def _export(self, im_set, gt_set, shape_set, filename):
        """
        Using type BytesList & Int64List
        https://github.com/vucongduy192/ML/blob/master/Week%207/TFRecords.ipynb

        :param im_set:
        :param gt_set:
        :param shape_set:
        :param filename:
        :return:
        """

        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        with tf.io.TFRecordWriter(filename) as writer:
            for img, gt, shape in list(zip(im_set, gt_set, shape_set)):
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'height': _int64_feature(shape[0]),
                            'width': _int64_feature(shape[1]),
                            'depth': _int64_feature(shape[2]),
                            'image_raw': _bytes_feature(img),
                            'label_raw': _bytes_feature(gt.tostring())
                        }))
                writer.write(example.SerializeToString())

    def parse_record(self, record_serialized):
        """
        Parses a sample proto containing a training or validation example of an image. The output of the
        pascal_voc_dataset.py image preprocessing script is a dataset containing serialized sample protocol
        buffers. Each sample proto contains the following fields (values are included as examples):
            height: 281
            width: 500
            channels: 3
            format: 'JPEG'
            filename: '2007_000032'
            image_raw: <JPEG encoded string>
            label_raw: <Numpy array encoded string>
        :param record_serialized: scalar Tensor tf.string containing a serialized sample protocol buffer.
        :return:
            image: Tensor tf.uint8 containing the decoded JPEG file.
            labels: Tensor tf.int32 containing the image's pixels' labels.
            shape: list of float Tensors describing the image shape: [height, width, channels].
        """
        keys_to_features = {
            'height': tf.io.FixedLenFeature([1], tf.int64),
            'width': tf.io.FixedLenFeature([1], tf.int64),
            'depth': tf.io.FixedLenFeature([1], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'label_raw': tf.io.FixedLenFeature([], tf.string)
        }
        parsed = tf.io.parse_single_example(serialized=record_serialized, features=keys_to_features)

        # Decode the raw data
        im = tf.image.decode_png(parsed['image_raw'])
        gt = tf.io.decode_raw(parsed['label_raw'], tf.uint8)
        gt = tf.reshape(gt, [tf.cast(parsed['height'][0], tf.int64), tf.cast(parsed['width'][0], tf.int64)])

        return im, gt, (parsed['height'][0], parsed['width'][0], parsed['depth'][0])

    def load_dataset(self, dataset_path, batch_size, is_training=False):
        """Returns a TFRecordDataset for the requested dataset."""
        data_path = os.path.join(dataset_path, 'TFRecords',
                                 'segmentation_{}.tfrecords'.format('train' if is_training else 'val'))

        dataset = tf.data.TFRecordDataset(data_path)
        dataset = dataset.prefetch(buffer_size=batch_size)
        dataset = dataset.map(self.parse_record)

        return dataset.batch(batch_size)


if __name__ == '__main__':
    root_path = '../../'
    dataset_path = os.path.join(root_path, 'VOC2012/')
    dataset = VOCDataset(augmentation_params=None)

    # # Load list basenames of images
    # train_basenames = dataset.get_basenames('val', dataset_path)
    # print('Found', len(train_basenames), 'val samples')
    #
    # # Load image from dataset directory and sparse encoding
    # # dataset.export_sparse_encoding('val', dataset_path)
    #
    # # Export sparse encoding ground truth to TFRecords
    # dataset.export_tfrecord('val', dataset_path, 'segmentation_val.tfrecords')
    # print('Finished exporting')

    # Reload saved TFRecords

    dataset_val = dataset.load_dataset(dataset_path, batch_size=8, is_training=False)

    iterator = tf.data.Iterator.from_structure(dataset_val.output_types,
                                               dataset_val.output_shapes)

    next_batch = iterator.get_next()
    val_init_op = iterator.make_initializer(dataset_val)

    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=session_config) as session:
        session.run(tf.global_variables_initializer())
        session.run(val_init_op)

        while True:
            try:
                im_batch, gt_batch = session.run(next_batch)
            except tf.errors.OutOfRangeError:
                break
            break
