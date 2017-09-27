"""Adjust colour balance of images"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

from slim.layers import colorspace_transform


def random_rotate(image, max_angle=10):
    angle = tf.random_uniform([], minval=-max_angle, maxval=max_angle)
    image = tf.contrib.image.rotate(
        image,
        angle * math.pi / 180,
        interpolation='BILINEAR'
        )
    return image


def distort_color(image,
                  color_ordering=0,
                  fast_mode=True,
                  scope=None,
                  brightness_max_delta=0.12,
                  contrast_max_ratio=1.2,
                  saturation_max_ratio=1.2,
                  hue_max_degree=15,
                  ):
  """Distort the color of a Tensor image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  brightness_max_delta = brightness_max_delta
  contrast_max_ratio = contrast_max_ratio
  saturation_max_ratio = saturation_max_ratio
  hue_max_delta = hue_max_degree * math.pi / 180
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
        image = image_utils.random_saturation(image,
                                              lower=(1. / saturation_max_ratio),
                                              upper=saturation_max_ratio)

      else:
        image = image_utils.random_saturation(image,
                                              lower=(1. / saturation_max_ratio),
                                              upper=saturation_max_ratio)
        image = tf.image.random_brightness(image, max_delta=brightness_max_delta)

    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
        # This part all in fLAB colorspace
        image = colorspace_transform.tf_rgb_to_flab(image)
        image = image_utils.random_saturation(image,
                                              lower=(1. / saturation_max_ratio),
                                              upper=saturation_max_ratio,
                                              source_space='flab')
        image = image_utils.random_hue(image,
                                       max_theta=hue_max_delta,
                                       source_space='flab')
        image = colorspace_transform.tf_flab_to_rgb(image)
        # Back to RGB colorspace
        image = image_utils.random_contrast(image,
                                            lower=(1. / contrast_max_ratio),
                                            upper=contrast_max_ratio)

      elif color_ordering == 1:
        image = image_utils.random_saturation(image,
                                              lower=(1. / saturation_max_ratio),
                                              upper=saturation_max_ratio)
        image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
        image = image_utils.random_contrast(image,
                                            lower=(1. / contrast_max_ratio),
                                            upper=contrast_max_ratio)
        image = image_utils.random_hue(image, max_theta=hue_max_delta)

      elif color_ordering == 2:
        image = image_utils.random_contrast(image,
                                            lower=(1. / contrast_max_ratio),
                                            upper=contrast_max_ratio)
        image = image_utils.random_hue(image, max_theta=hue_max_delta)
        image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
        image = image_utils.random_saturation(image, lower=0.5, upper=1.5)

      elif color_ordering == 3:
        # This part all in fLAB colorspace
        image = colorspace_transform.tf_rgb_to_flab(image)
        image = image_utils.random_hue(image,
                                       max_theta=hue_max_delta,
                                       source_space='flab')
        image = image_utils.random_saturation(image,
                                              lower=(1. / saturation_max_ratio),
                                              upper=saturation_max_ratio,
                                              source_space='flab')
        image = colorspace_transform.tf_flab_to_rgb(image)
        # This part all in RGB colorspace
        image = image_utils.random_contrast(image,
                                            lower=(1. / contrast_max_ratio),
                                            upper=contrast_max_ratio)
        image = tf.image.random_brightness(image, max_delta=brightness_max_delta)

      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # NB: The random_* ops do not necessarily clamp.
    return image


def _uniform_random_per_image(input_shape, lower, upper, dtype=tf.float32,
                              seed=None):
    output_shape = input_shape[:-3]
    outputs = tf.random_uniform(output_shape,
                                minval=lower,
                                maxval=upper,
                                dtype=dtype,
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


def adjust_hue(image, theta, name=None, source_space='rgb'):
    if source_space.lower() == 'rgb':
        image = colorspace_transform.tf_rgb_to_flab(image)

    # Add extra dimensions to theta, so it is the same across 2D space
    theta = tf.expand_dims(theta, axis=-1)
    if len(theta.shape) < len(image.shape) - 1:
        theta = tf.expand_dims(theta, axis=-1)

    # Prepend a dimension to image, incase there is more than one of them
    # This is because we might want to apply a different theta to each
    # in the batch.
    image = tf.expand_dims(image, axis=0)

    # Assemble the transformation matrix
    ones = tf.ones_like(theta)
    zeros = tf.zeros_like(theta)
    M = tf.stack(
            [tf.stack([ones, zeros, zeros], axis=-1),
             tf.stack([zeros, tf.cos(theta), -tf.sin(theta)], axis=-1),
             tf.stack([zeros, tf.sin(theta), tf.cos(theta)], axis=-1)],
            axis=-2)

    # Ensure theta has correct dimensionality
    n_dims = len(image.shape) - 2
    for i_dim in range(n_dims - len(M.shape) + 2):
        M = tf.expand_dims(M, axis=0)

    # Convolve with our rotation filter
    image = tf.nn.convolution (
        input=image,
        filter=M,
        padding='SAME',
        name='adjust_hue')

    # Remove the dimension we added as a precaution earlier
    image = tf.squeeze(image, 0)

    if source_space.lower() == 'rgb':
        # Convert back to RGB space
        image = colorspace_transform.tf_flab_to_rgb(image)
    return image


def random_hue(image, max_theta, seed=None, source_space='rgb'):
    theta = _uniform_random_per_image(image.shape,
                                      lower=-max_theta,
                                      upper=max_theta,
                                      seed=seed)
    return adjust_hue(image, theta, source_space=source_space)


def adjust_saturation(image, saturation_factor, name=None, source_space='rgb'):
    if source_space.lower() == 'rgb':
        image = colorspace_transform.tf_rgb_to_flab(image)
    # Add extra dimensions to the factor, so it is the same across 2D space
    saturation_factor = tf.expand_dims(saturation_factor, axis=-1)
    if len(saturation_factor.shape) < len(image.shape) - 1:
        saturation_factor = tf.expand_dims(saturation_factor, axis=-1)
    ones = tf.ones_like(saturation_factor)
    M = tf.stack([ones, saturation_factor, saturation_factor], axis=-1)
    image *= M
    if source_space.lower() == 'rgb':
        image = colorspace_transform.tf_flab_to_rgb(image)
    return image


def random_saturation(image, lower, upper, seed=None, source_space='rgb'):
    saturation_factor = _uniform_random_per_image(image.shape,
                                                  lower=lower,
                                                  upper=upper,
                                                  seed=seed)
    return adjust_saturation(image, saturation_factor,
                             source_space=source_space)
