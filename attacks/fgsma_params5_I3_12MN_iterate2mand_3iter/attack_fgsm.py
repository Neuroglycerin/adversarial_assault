"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math

import numpy as np
from PIL import Image

import tensorflow as tf
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
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 1, 'How many images process at one time.')

tf.flags.DEFINE_float(
    'p_threshold_antitarget', 0.50, 'What split of probabilities to anti-target.')

tf.flags.DEFINE_integer(
    'num_aug', 1, 'Number of augmented repetitions of the image to each net.')

tf.flags.DEFINE_integer(
    'max_iter', 3, 'Maximum number of iterations.')

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


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
      images: array with minibatch of images
      filenames: list of filenames without path
        If number of file names in this list less than number of images in
        the minibatch then only first len(filenames) images will be saved.
      output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            img = np.round(((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(f, format='PNG')


def augment_single_pre_resize(image, rotation_max_angle=5, source_space='rgb'):

    # Parameters
    rotation_max_angle = 5.75
    brightness_max_delta = 0.06 * 2
    contrast_max_ratio = 1.05
    saturation_max_ratio = 1.05
    hue_max_degree = 17

    # Rotate
    image = image_utils.random_rotate(image, max_angle=rotation_max_angle)

    # Crop
    # image = random_crop(image)

    # Randomly flip the image horizontally
    image = tf.image.random_flip_left_right(image)

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


def augment_batch_post_resize(images):
    # Add some random noise to the image
    unit_change = 2.0 / 255.0  # this level is equivalent to 1 utf8 change
    dist = unit_change * 15.
    noise = tf.random_uniform(
        shape=tf.shape(images), minval=-dist, maxval=dist, dtype=images.dtype)
    images = images + noise

    # Clip pixels back to the appropriate range of values
    return tf.clip_by_value(images, -1.0, 1.0)


def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    eps = FLAGS.max_epsilon * 2.0 / 255.0
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        # Fix seed for reproducibility
        tf.set_random_seed(9349008288)

        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        model_stack = model_loader.ModelLoaderStack(
            batch_size=FLAGS.batch_size,
            num_augmentations=FLAGS.num_aug)
        model_stack.add('inception_v3', 'models/inception_v3', im_size=299)
        model_stack.add('mobilenet_v1_100', 'models/mobilenet_v1_100_224', im_size=224)
        model_stack.add('mobilenet_v1_100', 'models/mobilenet_v1_100_192', im_size=192)
        model_stack.add('mobilenet_v1_100', 'models/mobilenet_v1_100_160', im_size=160)
        model_stack.add('mobilenet_v1_100', 'models/mobilenet_v1_100_128', im_size=128)
        model_stack.add('mobilenet_v1_075', 'models/mobilenet_v1_075_224', im_size=224)
        model_stack.add('mobilenet_v1_075', 'models/mobilenet_v1_075_192', im_size=192)
        model_stack.add('mobilenet_v1_075', 'models/mobilenet_v1_075_160', im_size=160)
        model_stack.add('mobilenet_v1_075', 'models/mobilenet_v1_075_128', im_size=128)
        model_stack.add('mobilenet_v1_050', 'models/mobilenet_v1_050_224', im_size=224)
        model_stack.add('mobilenet_v1_050', 'models/mobilenet_v1_050_192', im_size=192)
        model_stack.add('mobilenet_v1_050', 'models/mobilenet_v1_050_160', im_size=160)
        model_stack.add('mobilenet_v1_050', 'models/mobilenet_v1_050_128', im_size=128)

        # First, put through the unadulterated input to determine the true class
        logits_list = []
        for model in model_stack.models:
            logits_list += model.get_logits(x_input)
        logits = tf.reduce_mean(tf.stack(logits_list, axis=-1), axis=-1)

        preds = tf.nn.softmax(logits)
        preds = preds / tf.reduce_sum(preds, axis=-1, keep_dims=True)

        preds_sorted, sort_indices = tf.nn.top_k(
            preds, k=(num_classes - 1), sorted=True)
        top_label_index = sort_indices[:, 0]
        sort_indices_offset = sort_indices + tf.expand_dims(tf.range(FLAGS.batch_size), 1)

        preds_sum = tf.cumsum(preds_sorted, axis=-1, exclusive=True)
        class_is_before_threshold = preds_sum < FLAGS.p_threshold_antitarget
        weights_sorted = preds_sorted * tf.cast(class_is_before_threshold, preds_sorted.dtype)
        weights_sorted = tf.square(weights_sorted)
        weights_sorted = weights_sorted / tf.reduce_sum(weights_sorted, axis=-1, keep_dims=True)
        # 1 x 1000
        weights = tf.scatter_nd(tf.expand_dims(tf.reshape(sort_indices_offset, [-1]), -1),
                                tf.reshape(weights_sorted, [-1]),
                                [FLAGS.batch_size * num_classes])
        weights = tf.reshape(weights, preds.shape)

        # Stop the gradients on these
        weights = tf.stop_gradient(weights)
        top_label_index = tf.stop_gradient(top_label_index)

        # Now, put through augmented inputs to determine the vector to move in
        def test_loop_continue(iter_count, x_adv, logits, weights, top_label_index, label_is_right):
            # We always do the first step, otherwise the image is unchanged
            is_first_iter = tf.equal(iter_count, 0)
            # Otherwise, we only continue if we have not hit the iteration
            # limit
            iter_limit_not_reached = tf.less(iter_count, FLAGS.max_iter)
            # And if the predicted label is still correct
            # Put this all together
            return tf.logical_or(is_first_iter,
                                 tf.logical_and(iter_limit_not_reached,
                                                label_is_right)
                                 )

        def test_accuracy(logits):
            predicted_label = tf.argmax(logits, axis=-1, output_type=tf.int32)
            label_is_right = tf.equal(top_label_index, predicted_label)
            label_is_right = tf.reduce_any(label_is_right, axis=0)
            return label_is_right

        def update_x(x_adv):
            # First, we manipulate the image based on the output from the last
            # input image
            cross_entropy = tf.losses.softmax_cross_entropy(weights, logits)
            # First, we manipulate the image based on the gradients of the
            # cross entropy we just derived
            scaled_signed_grad = eps * tf.sign(tf.gradients(cross_entropy, x_adv)[0])
            x_next = tf.stop_gradient(x_adv + scaled_signed_grad)
            x_next = tf.clip_by_value(x_next, x_min, x_max)
            return x_next

        def update_logits(x_adv):
            # Now we generate new inputs with which to check whether we hit the
            # target, and if not collect gradients again
            # First, transform the image into flab space for preprocessing
            x_adv_flab = colorspace_transform.tf_rgb_to_flab(x_adv)
            # We will do all the pre-resize operations in this colorspace
            pre_resize_fn = lambda x: augment_batch_pre_resize(x, source_space='flab')
            # Then we transform back to RGB space before adding pixel-wise noise
            post_resize_fn = lambda x: \
                augment_batch_post_resize(colorspace_transform.tf_flab_to_rgb(x))

            # Collect up the output logits from each model with each of its augs
            logits_list = []
            for model in model_stack.models:
                logits_list += model.get_logits(
                    x_adv_flab,
                    pre_resize_fn=pre_resize_fn,
                    post_resize_fn=post_resize_fn)
            logits = tf.reduce_mean(tf.stack(logits_list, axis=-1), axis=-1)
            if FLAGS.num_aug > 1:
                assert FLAGS.batch_size == 1
                logits = tf.reduce_mean(logits, axis=0, keep_dims=True)
            return logits

        # Initialise loop variables
        # We start with the true image
        x_adv = x_input
        label_is_right = True

        for iter_count in range(FLAGS.max_iter):
            # Stop the gradients! We must take the gradient within the loop.
            x_adv = tf.stop_gradient(x_adv)
            # Generate augmented versions of input and forward propogate
            logits = update_logits(x_adv)
            # Check whether the current prediction is accurate
            label_is_right = test_accuracy(logits)
            # We always do the first step, otherwise the image is unchanged
            is_first_iter = tf.equal(iter_count, 0)
            #needs_update = tf.logical_or(is_first_iter, label_is_right)
            needs_update = tf.ones([], dtype=bool)
            # Maybe update x_adv
            x_adv = tf.cond(needs_update,
                            lambda: update_x(x_adv),
                            lambda: x_adv)

        # Run computation
        with tf.Session() as sess:

            for model in model_stack.models:
                model.restore(sess)

            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                adv_images = sess.run(x_adv, feed_dict={x_input: images})
                save_images(adv_images, filenames, FLAGS.output_dir)


if __name__ == '__main__':
    tf.app.run()
