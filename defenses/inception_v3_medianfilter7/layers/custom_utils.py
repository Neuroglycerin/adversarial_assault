"""Utilities for custom layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope


def tfrepeat(a, repeats, axis, name='repeat'):
    """Repeat elements of a tensor.

    Parameters:
        a : array_like
            Input array.
        repeats : int or array of ints
            The number of repetitions for each element. repeats is
            broadcasted to fit the shape of the given axis.
        axis : int, optional
            The axis along which to repeat values.

    Returns:
        repeated_array : ndarray
            Output array which has the same shape as a, except along the given axis.
    """
    shp = a.shape
    if len(shp) < axis-1:
        num_el = 1
    else:
        num_el = shp[axis]
    if isinstance(repeats, int):
        repeats = [repeats] * num_el
    splits = tf.unstack(a, num_el, axis)
    joins = []
    for i, split in enumerate(splits):
        joins += [split] * repeats[i]
    return tf.stack(joins, axis, name)


def tf_soft_clip_by_value(x, clip_min=None, clip_max=None):
    if clip_min is not None:
        x = tf.nn.elu(x - clip_min) + clip_min
    if clip_max is not None:
        x = clip_max - tf.nn.elu(clip_max - x)
    return x


@add_arg_scope
def tf_median_pool(x, kernel, strides=[1,1], padding='SAME', keep_edges=None):
    """
    If keep_edges is non-zero, we don't use the median pool for the edges of the
    image. Instead, we keep the original value there.
    This prevents all the corners from going to 0.
    By default, we include the original for the edge with length kernel/2 if
    padding is 'SAME'.
    """

    if keep_edges and padding is 'VALID':
        raise ValueError('Shouldnt use keep_edges with padding VALID')

    if isinstance(strides, int):
        strides = [strides, strides]

    if keep_edges is None and padding is 'SAME':
        in_height = int(x.shape[1])
        in_width = int(x.shape[2])
        out_height = ceil(float(in_height) / float(strides[0]))
        out_width  = ceil(float(in_width) / float(strides[1]))
        if (in_height % strides[0] == 0):
            pad_along_height = max(kernel[0] - strides[0], 0)
        else:
            pad_along_height = max(kernel[0] - (in_height % strides[0]), 0)
        if (in_width % strides[1] == 0):
            pad_along_width = max(kernel[1] - strides[1], 0)
        else:
            pad_along_width = max(kernel[1] - (in_width % strides[1]), 0)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        keep_edges = [[int(ceil(pad_top / strides[0])),
                       int(ceil(pad_bottom / strides[0]))],
                      [int(ceil(pad_left / strides[1])),
                       int(ceil(pad_right / strides[0]))]]

    elif not keep_edges:
        keep_edges = []

    elif isinstance(keep_edges, int):
        keep_edges = [[keep_edges, keep_edges]] * 2

    batch_size, input_height, input_width, num_channels = x.shape

    # Input is shaped [batch_size, input_height, input_width, channels]

    num_kernel_elements = 1
    for k in kernel:
        num_kernel_elements *= k

    x_original = x

    # Unstack depth
    x_depths = tf.unstack(x, axis=-1)
    # For each depth, extract patches
    x_depths = [tf.extract_image_patches(
                    tf.expand_dims(x_depth, axis=-1),
                    [1] + kernel + [1],
                    strides=[1] + strides + [1],
                    rates=[1,1,1,1],
                    padding=padding)
                for x_depth in x_depths]
    # Stack depths again, into a new axis where channels should be
    x = tf.stack(x_depths, axis=-2)
    # Now shaped [batch_size, height, width, channels, kernel * kernel]

    # Next sort the values within the kernel. We only need the top half, and
    # don't need the indices.
    middle_idx = int((num_kernel_elements) / 2)
    x, _ = tf.nn.top_k(x, middle_idx + 1, sorted=True)
    # Now shaped [batch_size, height, width, channels, middle_idx + 1]

    # Next, we extract the median value from the set
    if (num_kernel_elements % 2) == 1:
        # With an odd number, it is just the smallest value from the top half
        x = x[:, :, :, :, -1]
    else:
        # With an even number, we average the bottom two values
        x = tf.reduce_mean(x[:, :, :, :, -2:], axis=-1)
    # We drop a dimension as we do this (in either case)
    # Now shaped [batch_size, height, width, channels]

    if keep_edges:
        # Cut off the edge
        x_top = x_original[:, 0:keep_edges[0][0]*strides[0]:strides[0], ::strides[1], :]
        if keep_edges[0][1]==0:
            x_bottom = x_original[:, 0:0:strides[0], ::strides[1], :]
        else:
            x_bottom = x_original[:, -keep_edges[0][1]*strides[0]::strides[0], ::strides[1], :]
        if keep_edges[0][1]==0:
            x = x[:, keep_edges[0][0]:, :, :]
        else:
            x = x[:, keep_edges[0][0]:-keep_edges[0][1], :, :]

        # Add back the original edge
        x = tf.concat([x_top, x, x_bottom], axis=1)

        # Cut off the edge
        x_left = x_original[:, ::strides[0], 0:keep_edges[1][0]*strides[1]:strides[1], :]
        if keep_edges[1][1]==0:
            x_right = x_original[:, ::strides[0], 0:0:strides[1], :]
        else:
            x_right = x_original[:, ::strides[0], -keep_edges[1][1]*strides[1]::strides[1], :]
        if keep_edges[1][1]==0:
            x = x[:, :, keep_edges[1][0]:, :]
        else:
            x = x[:, :, keep_edges[1][0]:-keep_edges[1][1], :]

        # Add back the original edge
        x = tf.concat([x_left, x, x_right], axis=2)

    return x
