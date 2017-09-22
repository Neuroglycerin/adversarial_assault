"""Adjust colour balance of images"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from slim.layers import colorspace_transform


def _uniform_random_per_image(input_shape, lower, upper, seed=None):
    if len(input_shape) <= 3:
        output_shape = [1]
    else:
        output_shape = input_shape[:-3]
    outputs = tf.random_uniform(output_shape,
                                minval=lower,
                                maxval=upper,
                                seed=seed)
    return outputs


def adjust_contrast(images, contrast_factor):
    # Average intensity across x, y, channels
    mu = tf.reduce_mean(
            tf.reduce_mean(
                tf.reduce_mean(images, axis=-1),
                axis=-1),
            axis=-1)
    return (images - mu) * contrast_factor + mu


def random_contrast(images, lower, upper, seed=None):
    contrast_factor = _uniform_random_per_image(images.shape,
                                                lower=lower,
                                                upper=upper,
                                                seed=seed)
    return adjust_contrast(images, contrast_factor)


def adjust_hue(image, theta, name=None):
    image = colorspace_transform.tf_rgb_to_flab(image)
    ones = tf.ones_like(theta)
    zeros = tf.zeros_like(theta)
    M = tf.stack(
            [tf.stack([ones, zeros, zeros], axis=-1),
             tf.stack([zeros, tf.cos(theta), -tf.sin(theta)], axis=-1),
             tf.stack([zeros, tf.sin(theta), tf.cos(theta)], axis=-1)],
            axis=-2)
    image = tf.matmul(image, M)
    image = colorspace_transform.tf_flab_to_rgb(image)
    return image


def random_hue(image, max_theta, seed=None):
    theta = _uniform_random_per_image(image.shape,
                                      lower=-max_theta,
                                      upper=max_theta,
                                      seed=seed)
    return adjust_hue(image, theta)


def adjust_saturation(image, saturation_factor, name=None):
    image = colorspace_transform.tf_rgb_to_flab(image)
    # Add extra dimensions to the factor, so it is the same across 2D space
    saturation_factor = tf.expand_dims(saturation_factor, axis=-1)
    if len(saturation_factor.shape) < len(image.shape) - 1:
        saturation_factor = tf.expand_dims(saturation_factor, axis=-1)
    ones = tf.ones_like(saturation_factor)
    M = tf.stack([ones, saturation_factor, saturation_factor], axis=-1)
    image *= M
    image = colorspace_transform.tf_flab_to_rgb(image)
    return image


def random_saturation(image, lower, upper, seed=None):
    saturation_factor = _uniform_random_per_image(image.shape,
                                                  lower=lower,
                                                  upper=upper,
                                                  seed=seed)
    return adjust_saturation(image, saturation_factor)
