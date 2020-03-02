import numpy as np
import cv2
import os

def imread(filename):
    """
    Loads an image from disk
    :param filename: the path to the image file to load
    :param target_shape: optional resizing to the specified shape
    :param interpolation: interpolation method. Defaults to cv2.INTER_AREA which is recommended for downsizing.
    :return: the loaded image in RGB format
    """
    img = cv2.imread(filename)
    if img is None:
        print('Error loading image. Check path:', filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads images in BGR format
    print(img.shape)
    return img

def pad(im, target_shape, center=False, cval=0):
    """
    Pads an image to the specified shape. The image must be smaller than the target shape.
    Returns a copy, the input image remains unchanged.
    :param im: the image to pad
    :param target_shape: the shape of the image after padding
    :param center: center the image or append rows and columns to the image
    :param cval: constant value for the padded pixels
    :return:
    """
    h_pad, w_pad = np.asarray(target_shape) - im.shape[:2]
    assert h_pad >= 0, 'Height padding must be non-negative'
    assert w_pad >= 0, 'Width padding must be non-negative'

    if center:
        padding = ((h_pad//2, h_pad-h_pad//2), (w_pad//2, w_pad-w_pad//2))
        if len(im.shape) == 3:
            padding += ((0, 0),)
        im_padded = np.pad(im, padding, mode='constant', constant_values=cval)
    else:
        padding = ((0, h_pad), (0, w_pad))
        if len(im.shape) == 3:
            padding += ((0, 0),)
        im_padded = np.pad(im, padding, mode='constant', constant_values=cval)
    return im_padded


def colors2labels(im, cmap, one_hot=False):
    """
    Converts a RGB ground truth segmentation image into a labels matrix with optional one-hot encoding.
    """
    if one_hot:
        labels = np.zeros((*im.shape[:-1], len(cmap)-1), dtype='uint8')
        for i, color in enumerate(cmap):
            labels[:, :, i] = np.all(im == color, axis=2).astype('uint8')
    else:
        labels = np.zeros(im.shape[:-1], dtype='uint8')
        for i, color in enumerate(cmap):
            labels += i * np.all(im == color, axis=2).astype(dtype='uint8')
    return labels