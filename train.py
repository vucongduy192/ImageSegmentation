import numpy as np
import os

import tensorflow as tf

print('Tensorflow version: {}'.format(tf.__version__))

from dataset import VOCDataset
from fcn import FCN

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

if __name__ == '__main__':
    image_shape = (512, 512)
    n_classes = 21
    vgg16_weights_path = './vgg/vgg16_weights.npz'
    model = FCN(image_shape, n_classes, vgg16_weights_path)
    model.build_from_vgg()

    root_path = './'
    dataset_path = os.path.join(root_path, 'VOC2012/')
    dataset = VOCDataset(augmentation_params=None)

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
                print(im_batch.shape)
            except tf.errors.OutOfRangeError:
                break
            break
