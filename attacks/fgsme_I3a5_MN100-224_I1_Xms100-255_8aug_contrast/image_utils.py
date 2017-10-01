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


def random_crop(image):
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                       dtype=tf.float32,
                       shape=[1, 1, 4])
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].
    from slim.preprocessing import inception_preprocessing
    image, new_bbox = inception_preprocessing.distorted_bounding_box_crop(
        image,
        bbox,
        min_object_covered=0.75,
        aspect_ratio_range=(0.75, 1.33),
        area_range=(0.75, 1.0),
        max_attempts=100,
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
        image = random_saturation(image,
                                  lower=(1. / saturation_max_ratio),
                                  upper=saturation_max_ratio)

      else:
        image = random_saturation(image,
                                  lower=(1. / saturation_max_ratio),
                                  upper=saturation_max_ratio)
        image = tf.image.random_brightness(image, max_delta=brightness_max_delta)

    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
        # This part all in fLAB colorspace
        image = colorspace_transform.tf_rgb_to_flab(image)
        image = random_saturation(image,
                                  lower=(1. / saturation_max_ratio),
                                  upper=saturation_max_ratio,
                                  source_space='flab')
        image = random_hue(image,
                           max_theta=hue_max_delta,
                           source_space='flab')
        image = colorspace_transform.tf_flab_to_rgb(image)
        # Back to RGB colorspace
        image = random_contrast(image,
                                lower=(1. / contrast_max_ratio),
                                upper=contrast_max_ratio)

      elif color_ordering == 1:
        image = random_saturation(image,
                                  lower=(1. / saturation_max_ratio),
                                  upper=saturation_max_ratio)
        image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
        image = random_contrast(image,
                                lower=(1. / contrast_max_ratio),
                                upper=contrast_max_ratio)
        image = random_hue(image, max_theta=hue_max_delta)

      elif color_ordering == 2:
        image = random_contrast(image,
                                lower=(1. / contrast_max_ratio),
                                upper=contrast_max_ratio)
        image = random_hue(image, max_theta=hue_max_delta)
        image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
        image = random_saturation(image, lower=0.5, upper=1.5)

      elif color_ordering == 3:
        # This part all in fLAB colorspace
        image = colorspace_transform.tf_rgb_to_flab(image)
        image = random_hue(image,
                           max_theta=hue_max_delta,
                           source_space='flab')
        image = random_saturation(image,
                                  lower=(1. / saturation_max_ratio),
                                  upper=saturation_max_ratio,
                                  source_space='flab')
        image = colorspace_transform.tf_flab_to_rgb(image)
        # This part all in RGB colorspace
        image = random_contrast(image,
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


def adjust_brightness(images, delta, source_space='rgb'):
    # Add extra dimensions to the factor, so it is the same across 3D space
    for i in range(3):
        delta = tf.expand_dims(delta, axis=-1)
    if source_space.lower() == 'rgb':
        return images + delta
    elif source_space.lower() == 'flab':
        lightness, a_and_b_components = tf.split(images, [1, 2], -1)
        lightness += delta
        return tf.concat((lightness, a_and_b_components), axis=-1)
    else:
        raise ValueError('Unrecognised source space: {}'.format(source_space))


def random_brightness(images, max_delta, seed=None, source_space='rgb'):
    brightness_delta = _uniform_random_per_image(images.shape,
                                                 lower=max_delta,
                                                 upper=max_delta,
                                                 seed=seed)
    return adjust_brightness(images, brightness_delta,
                             source_space=source_space)


def adjust_contrast(images, contrast_factor):
    # Add extra dimensions to the factor, so it is the same across 3D space
    for i in range(3):
        contrast_factor = tf.expand_dims(contrast_factor, axis=-1)
    # Average intensity across x, y, channels
    mu = tf.reduce_mean(
            tf.reduce_mean(
                tf.reduce_mean(images, axis=-1, keep_dims=True),
                axis=-2, keep_dims=True),
            axis=-3, keep_dims=True)
    return (images - mu) * contrast_factor + mu


def random_contrast(images, lower, upper, seed=None):
    contrast_factor = _uniform_random_per_image(images.shape,
                                                lower=lower,
                                                upper=upper,
                                                seed=seed)
    return adjust_contrast(images, contrast_factor)


def adjust_saturation(image, saturation_factor, name=None, source_space='rgb'):
    if source_space.lower() == 'rgb':
        image = colorspace_transform.tf_rgb_to_flab(image)
    # Add extra dimensions to the factor, so it is the same across 2D space
    for i in range(2):
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


def adjust_saturation_and_contrast(images, saturation_factor, contrast_factor,
                                   source_space='rgb'):
    if contrast_factor.shape != saturation_factor.shape:
        raise ValueError('Shape mismatch. Inputs were {} and {}'
                         ''.format(contrast_factor.shape, saturation_factor.shape))

    if source_space.lower() == 'rgb':
        images = colorspace_transform.tf_rgb_to_flab(images)
    elif source_space.lower() != 'flab':
        raise ValueError('Unrecognised source space: {}'.format(source_space))

    # Add extra dimensions to the factors, so they are the same across 3D space
    for i in range(3):
        saturation_factor = tf.expand_dims(saturation_factor, axis=-1)
        contrast_factor = tf.expand_dims(contrast_factor, axis=-1)

    # Split up lightness and colour components
    lightness, a_and_b_components = tf.split(images, [1, 2], -1)

    # For lightness, we take the average across space and rescale the distance
    # from this average
    mu = tf.reduce_mean(
        tf.reduce_mean(lightness, axis=-2, keep_dims=True),
        axis=-3,
        keep_dims=True)
    lightness = (lightness - mu) * contrast_factor + mu

    # For A and B channels, we rescale by the product of saturation and
    # contrast
    a_and_b_components *= saturation_factor * contrast_factor

    # Now we join the L, A and B components back together
    images = tf.concat((lightness, a_and_b_components), axis=-1)

    if source_space.lower() == 'rgb':
        # Convert back to RGB space
        images = colorspace_transform.tf_flab_to_rgb(images)

    return images


def random_saturation_and_contrast(
        images,
        saturation_lower,
        saturation_upper,
        contrast_lower,
        contrast_upper,
        source_space='rgb',
        seed=None):
    saturation_factor = _uniform_random_per_image(images.shape,
                                                  lower=saturation_lower,
                                                  upper=saturation_upper,
                                                  seed=seed)
    contrast_factor = _uniform_random_per_image(images.shape,
                                                lower=contrast_lower,
                                                upper=contrast_upper,
                                                seed=seed)
    return adjust_saturation_and_contrast(images,
                                          saturation_factor,
                                          contrast_factor,
                                          source_space=source_space)


def adjust_hue(images, theta, name=None, source_space='rgb'):
    if source_space.lower() == 'rgb':
        images = colorspace_transform.tf_rgb_to_flab(images)

    # Add extra dimensions to theta, so it is the same across 2D space
    for i in range(2):
        theta = tf.expand_dims(theta, axis=-1)

    # Split up lightness and colour components
    lightness, ab_comp = tf.split(images, [1, 2], -1)

    # Prepend a dimension to image, incase there is more than one of them
    # This is because we might want to apply a different theta to each
    # in the batch and we can't do things differently across batch dimension
    # with the convolution function.
    ab_comp = tf.expand_dims(ab_comp, axis=0)

    # Assemble the transformation matrix
    M = tf.stack(
            [tf.stack([tf.cos(theta), -tf.sin(theta)], axis=-1),
             tf.stack([tf.sin(theta), tf.cos(theta)], axis=-1)],
            axis=-2)

    # Ensure theta has correct dimensionality
    n_dims = len(images.shape) - 2
    for i_dim in range(n_dims - len(M.shape) + 2):
        M = tf.expand_dims(M, axis=0)

    # Convolve with our rotation filter
    ab_comp = tf.nn.convolution (
        input=ab_comp,
        filter=M,
        padding='SAME',
        name='adjust_hue')

    # Remove the dimension we added as a precaution earlier
    ab_comp = tf.squeeze(ab_comp, 0)

    # Now we join the L, A and B components back together
    images = tf.concat((lightness, ab_comp), axis=-1)

    if source_space.lower() == 'rgb':
        # Convert back to RGB space
        images = colorspace_transform.tf_flab_to_rgb(images)
    return images


def random_hue(image, max_theta, seed=None, source_space='rgb'):
    theta = _uniform_random_per_image(image.shape,
                                      lower=-max_theta,
                                      upper=max_theta,
                                      seed=seed)
    return adjust_hue(image, theta, source_space=source_space)
