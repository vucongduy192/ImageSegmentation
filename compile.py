from datetime import datetime
import numpy as np
import cv2
import os
import random

import tensorflow as tf
from dataset import VOCDataset
from fcn import FCN

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)


def learning_rate_with_exp_decay(batch_size, n_images, decay_epochs, decay_rate=0.95, base_lr=1e-5):
    """
    :param batch_size:
    :param n_images:
    :param decay_epochs: The number of epochs after which the learning rate is decayed by `decay_rate`.
    :param decay_rate:
    :param base_lr:
    :return:
    """
    """ Decay learning_rate after each epoch
    """

    global_step = tf.Variable(0, name='global_step', trainable=False)
    n_batches = int(n_images / batch_size)

    def learning_rate_fn():
        lr = tf.train.exponential_decay(base_lr, global_step, n_batches * decay_epochs, decay_rate,
                                        name='exp_decay')
        return lr

    return learning_rate_fn


def compile_model(model, learning_rate_fn):
    """
    :param learning_rate_fn:
    :param model:
    :param learning_rate:
    :param metrics:
    :return:
    """

    # ------------ Create loss fucntion and optimizer --------------
    # Vector of ground truth labels
    # Reshape 4D tensor (batch, row, column, channel) to 2D, each row represents a pixel, each column a class
    labels = tf.stop_gradient(tf.reshape(model.labels, (-1,)))
    logits = tf.reshape(model.outputs, (-1, model.n_classes + 1), name="logits")

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(labels, tf.int32), logits=logits)
    loss = tf.reduce_mean(input_tensor=cross_entropy, name="cross_entropy")

    learning_rate = learning_rate_fn()
    tf.identity(learning_rate, name='learning_rate')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step(), name="train_op")

    # --------------- Create metrics -----------------
    tf_metrics = {}
    predictions = tf.argmax(input=logits, axis=1, name="predictions")
    # Pixel accuracy/ Mean class pixel accuracy/ Mean Intersection-over-Union (mIoU)
    tf_metrics['acc'] = tf.metrics.accuracy(labels, predictions, name='acc')
    tf_metrics['mean_acc'] = tf.metrics.mean_per_class_accuracy(labels, predictions, model.n_classes + 1,
                                                                name='mean_acc')
    tf_metrics['mean_iou'] = tf.metrics.mean_iou(labels, predictions, model.n_classes + 1,
                                                 name='mean_iou')

    # --------------- Assign operator to model ---------------
    session = tf.get_default_session()
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())

    model.logits = logits
    model.loss = loss
    model.optimizer = optimizer
    model.train_op = train_op
    model.metrics = tf_metrics
    print("Model build successful")


def metrics2message(model, res, message, metrics):
    for metric in list(model.metrics.keys()):
        # Remove the void/ignore class accuracy in the mean calculation because its value is 0
        if metric == 'mean_acc':
            val = np.mean(res[metric][1][:model.n_classes])
        # Remove the void/ignore class IoU in the mean calculation because its value is NaN
        elif metric == 'mean_iou':
            mat = res[metric][1][:model.n_classes, :model.n_classes]
            val = np.mean((np.diag(mat) / (mat.sum(axis=0) + mat.sum(axis=1) - np.diag(mat))))
        # No need to adjust other metrics
        else:
            val = res[metric][0]
        metrics[metric].append(val)
        message += ', {} = {:.3f}'.format(metric, val)

    return message, metrics


def fit_model(model, epochs, batch_size, dataset_train, dataset_val, checkpoint_dir, dropout_rate=0.5):
    """
    :param model:
    :param epochs:
    :param batch_size:
    :param dataset_train:
    :param dataset_val:
    :param dropout_rate:
    :return:
    """
    session = tf.get_default_session()
    iterator = tf.data.Iterator.from_structure(dataset_train.output_types,
                                               dataset_train.output_shapes)
    next_batch = iterator.get_next()
    train_init_op = iterator.make_initializer(dataset_train)
    val_init_op = iterator.make_initializer(dataset_val)

    training_loss, validation_loss = [], []
    training_metrics, validation_metrics = {}, {}

    saver = tf.train.Saver(max_to_keep=10)
    print("Starting training FCN-8s")

    for epoch in range(0, epochs):
        # Initialize an iterator over the training dataset
        print("Epoch {}/{}, LR={}".format(epoch + 1, epochs, session.run(model.optimizer._lr)))
        training_loss.append(0)
        session.run(train_init_op)
        n_batches = 0

        # ------------------ Training process ---------------------
        while True:
            try:
                im_batch, gt_batch = session.run(next_batch)
                if len(im_batch) < batch_size:
                    continue
                res = session.run({**{"loss": [model.loss, model.train_op]}, **model.metrics},
                                  feed_dict={model.inputs: im_batch,
                                             model.labels: gt_batch,
                                             model.dropout_rate: dropout_rate})
                training_loss[-1] += res["loss"][0]
                n_batches += 1
            except tf.errors.OutOfRangeError:
                break

        # Update training loss and metrics
        training_loss[-1] /= n_batches
        message = 'loss = {:.3f}'.format(training_loss[-1])
        message, training_metrics = metrics2message(model, res, message, training_metrics)

        # ------------------ Validation process ---------------------
        if dataset_val is not None:
            validation_loss.append(0)
            n_batches = 0

            # Initialize an iterator over the validation dataset
            session.run(val_init_op)
            while True:
                try:
                    im_batch, gt_batch = session.run(next_batch)
                    if len(im_batch) < batch_size:
                        continue
                    res = session.run({**{"loss": model.loss}, **model.metrics},
                                      feed_dict={model.inputs: im_batch,
                                                 model.labels: gt_batch,
                                                 model.dropout_rate: 0.0})
                    validation_loss[-1] += res["loss"]
                    n_batches += 1
                except tf.errors.OutOfRangeError:
                    break

            training_loss[-1] /= n_batches
            message = 'loss = {:.3f}'.format(training_loss[-1])
            message, validation_metrics = metrics2message(model, res, message, validation_metrics)
            print(message + '\n' + datetime.now().isoformat())

        # ------------------ Save checkpoint ---------------------
        if epoch % 25 == 0:
            checkpoint = checkpoint_dir + '/epoch{}.ckpt'.format(epoch)
            saver.save(session, checkpoint)
            print('Checkpoint saved:', checkpoint)


def train():
    ROOT = './'
    VGG16_WEIGHT_PATH = './vgg/vgg16_weights.npz'
    DATASET_PATH = os.path.join(ROOT, 'VOC2012/')
    CHECKPOINT_DIR = os.path.join(DATASET_PATH, 'saved_model')

    IMAGE_SHAPE = (512, 512)
    N_CLASSES = 21
    N_EPOCHS = 100
    BATCH_SIZE = 1

    LEARNING_RATE = 1e-5
    DECAY_RATE = 0.95
    DECAY_EPOCH = 10
    DROPOUT_RATE = 0.5

    print('Starting end-to-end training FCN-8s')
    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.compat.v1.InteractiveSession(config=session_config)
    session.as_default()

    # ------------- Load VOC from TFRecord ---------------
    dataset = VOCDataset()
    dataset_train = dataset.load_dataset(DATASET_PATH, BATCH_SIZE, is_training=True)
    dataset_val = dataset.load_dataset(DATASET_PATH, BATCH_SIZE, is_training=False)

    # ------------- Build fcn model ------------
    fcn = FCN(IMAGE_SHAPE, N_CLASSES, VGG16_WEIGHT_PATH)
    fcn.build_from_vgg()

    learning_rate_fn = learning_rate_with_exp_decay(BATCH_SIZE, dataset.n_images['train'], DECAY_EPOCH,
                                                    DECAY_RATE, LEARNING_RATE)
    compile_model(fcn, learning_rate_fn)
    fit_model(fcn, N_EPOCHS, BATCH_SIZE, dataset_train, dataset_val, CHECKPOINT_DIR, DROPOUT_RATE)


if __name__ == '__main__':
    train()
