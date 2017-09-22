"""Adjust colour balance of images"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def adjust_contrast(images, contrast_factor):
    # Average intensity across x, y, channels
    mu = tf.reduce_mean(
            tf.reduce_mean(
                tf.reduce_mean(images, axis=-1),
                axis=-1),
            axis=-1)
    return (images - mu) * contrast_factor + mu


def random_contrast(images, lower, upper, seed=None):
    if images.rank <= 3:
        shape = [1]
    else:
        shape = image.shape[:-3]
    contrast_factor = random_uniform(shape,
                                     minval=lower,
                                     maxval=uupper,
                                     seed=seed)
    return adjust_contrast(images, contrast_factor)
