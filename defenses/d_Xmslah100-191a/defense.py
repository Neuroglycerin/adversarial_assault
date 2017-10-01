"""Implementation of sample defense.

This defense loads inception v4 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
time_start_script = time.time()

import os
import math

import numpy as np
from PIL import Image
from scipy.misc import imread

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

from slim.preprocessing import inception_preprocessing
from slim.layers import colorspace_transform

import model_loader
import image_utils


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_file', '', 'Output file to save labels.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 1, 'How many images process at one time.')

tf.flags.DEFINE_integer(
    'num_aug', 1, 'Number of augmented repetitions of the image to each net.')

tf.flags.DEFINE_float(
    'time_limit_per_100_samples', 500., 'Time limit in seconds per 100 samples.')


FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.

    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath) as f:
            image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1]
        # interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def augment_single_pre_resize(image, source_space='rgb'):
    # Don't bother with colour augmentations any more

    # Crop
    # image = random_crop(image)

    # Parameters
    brightness_max_delta = 0.06 * 2
    contrast_max_ratio = 1.05
    saturation_max_ratio = 1.05
    hue_max_degree = 17

    # Randomly distort the colors. There are 4 ways to do it.
    # This part all in fLAB colorspace
    if source_space.lower() == 'rgb':
        image = colorspace_transform.tf_rgb_to_flab(image)
    elif source_space.lower() != 'flab':
        raise ValueError('Unrecognised colour space: {}'.format(source_space))
    image = image_utils.random_brightness(
        image,
        max_delta=brightness_max_delta,
        source_space='flab')
    image = image_utils.random_saturation_and_contrast(
        image,
        saturation_lower=(1. / saturation_max_ratio),
        saturation_upper=saturation_max_ratio,
        contrast_lower=(1. / contrast_max_ratio),
        contrast_upper=contrast_max_ratio,
        source_space='flab')
    image = image_utils.random_hue(image,
                                   max_theta=hue_max_degree,
                                   source_space='flab')
    if source_space.lower() == 'rgb':
        image = colorspace_transform.tf_flab_to_rgb(image)

    return image


def augment_single_post_resize(image, index=None):

    # Parameters
    rotation_max_angle = 5.8

    # Rotate
    image = image_utils.random_rotate(image, max_angle=rotation_max_angle)

    # Randomly flip the image horizontally
    if index is None:
        image = tf.image.random_flip_left_right(image)
    else:
        unflipped = lambda: image
        flipped = lambda: tf.image.flip_left_right(image)
        image = tf.cond(tf.less(tf.floormod(index, 2), 1), unflipped, flipped)

    return image


def augment_batch_pre_resize(x, source_space='rgb'):

    # First, do all the operations which have to occur on images one at a time
    images = tf.unstack(x, axis=0)
    augmented_images = []
    for image in images:
        for i in range(FLAGS.num_aug):
            augmented_images.append(augment_single_pre_resize(
                image,
                source_space=source_space))
    images = tf.stack(augmented_images, axis=0)

    return images


def augment_batch_post_resize(x):
    # First, do all the operations which have to occur on images one at a time
    images = tf.unstack(x, axis=0)
    augmented_images = []
    for image in images:
        for i in range(FLAGS.num_aug):
            if i % 2 == 0:
                i_offset = tf.random_uniform([], maxval=2, dtype=tf.int32)
            augmented_images.append(
                augment_single_post_resize(image, index=(i + i_offset))
                )
    images = tf.stack(augmented_images, axis=0)

    # Add some random noise to the image
    unit_change = 2.0 / 255.0  # this level is equivalent to 1 utf8 change
    dist = unit_change * 15.
    noise = tf.random_uniform(
        shape=tf.shape(images), minval=-dist, maxval=dist, dtype=images.dtype)
    images = images + noise

    # Clip pixels back to the appropriate range of values
    return tf.clip_by_value(images, -1.0, 1.0)


def main(_):
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        time_start_building_graph = time.time()

        # Fix seed for reproducibility
        tf.set_random_seed(9349008288)

        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        model_stack = model_loader.ModelLoaderStack(
            batch_size=FLAGS.batch_size,
            num_augmentations=FLAGS.num_aug)
        model_stack.add('xception_multiscale2lah', 'models/xception_multiscale2lah_100_191_adv', im_size=191)

        def update_logits(x):
            ## Now we generate new inputs with which to check whether we hit the
            ## target, and if not collect gradients again
            ## First, transform the image into flab space for preprocessing
            #x_adv_flab = colorspace_transform.tf_rgb_to_flab(x_adv)
            ## We will do all the pre-resize operations in this colorspace
            #pre_resize_fn = lambda x: augment_batch_pre_resize(x, source_space='flab')
            ## Then we transform back to RGB space before adding pixel-wise noise
            #post_resize_fn = lambda x: \
            #    augment_batch_post_resize(colorspace_transform.tf_flab_to_rgb(x))

            # Collect up the output logits from each model with each of its augs
            logits_list = []
            total_mass = 0
            for model in model_stack.models:
                model_logits = model.get_logits(
                    x,
                    pre_resize_fn=None,
                    post_resize_fn=None)
                num_logits_for_model = len(model_logits)
                if FLAGS.num_aug > 1:
                    assert FLAGS.batch_size == 1
                    model_logits = tf.stack(model_logits, axis=0)
                    model_logits = tf.reduce_mean(model_logits, axis=1, keep_dims=False)
                else:
                    model_logits = tf.concat(model_logits, axis=0)
                model_logits *= model.weight / num_logits_for_model
                logits_list.append(model_logits)
                total_mass += model.weight
            logits_stack = tf.concat(logits_list, axis=0) / total_mass
            return logits_stack

        # Collect logits
        logits_stack = update_logits(x_input)

        assert FLAGS.batch_size == 1
        avg_logits = tf.reduce_sum(logits_stack, axis=0, keep_dims=True)

        predicted_labels = tf.argmax(avg_logits, axis=-1)

        # Run computation
        with tf.Session() as sess:

            time_start_session = time.time()
            logging.info('Starting to load models after {} seconds'.format(
                time_start_session - time_start_script))

            time_start_loading_models = time.time()
            for model in model_stack.models:
                model.restore(sess)
            time_end_loading_models = time.time()
            logging.info('Loading models took {} seconds'.format(
                time_end_loading_models - time_start_loading_models))

            time_start_generating = time.time()

            with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
                for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                    labels = sess.run(predicted_labels, feed_dict={x_input: images})
                    for filename, label in zip(filenames, labels):
                        out_file.write('{0},{1}\n'.format(filename, label))


if __name__ == '__main__':
  tf.app.run()
