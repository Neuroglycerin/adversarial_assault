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
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(f, format='PNG')


def augment_single(image):
    # Randomly flip the image horizontally
    image = tf.image.random_flip_left_right(image)
    # Randomly distort the colors. There are 4 ways to do it.
    fast_mode = False
    image = image * 0.5 + 0.5
    image = inception_preprocessing.apply_with_random_selector(
        image,
        lambda x, ordering: distort_color(x, ordering, fast_mode),
        num_cases=4)
    image = image * 2.0 - 1.0
    return image


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
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
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        #image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        #image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        #image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        #image = tf.image.random_hue(image, max_delta=0.2)
        image = image_utils.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        #image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = image_utils.random_contrast(image, lower=0.5, upper=1.5)
        #image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = image_utils.random_contrast(image, lower=0.5, upper=1.5)
        #image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        #image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        #image = tf.image.random_hue(image, max_delta=0.2)
        #image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = image_utils.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)


def augment_batch(x):
    images = tf.unstack(x, axis=0)
    images = [augment_single(image) for image in images]
    return tf.stack(images, axis=0)


def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        model_stack = model_loader.ModelLoaderStack()
        model_stack.add('inception_v3', 'inception_v3.ckpt', im_size=299)
        model_stack.add('mobilenet_v1_100', 'mobilenet_v1_1.0_224.ckpt', im_size=224)
        model_stack.add('mobilenet_v1_100', 'mobilenet_v1_1.0_192.ckpt', im_size=192)
        model_stack.add('mobilenet_v1_100', 'mobilenet_v1_1.0_160.ckpt', im_size=160)
        model_stack.add('mobilenet_v1_100', 'mobilenet_v1_1.0_128.ckpt', im_size=128)
        model_stack.add('mobilenet_v1_075', 'mobilenet_v1_0.75_224.ckpt', im_size=224)
        model_stack.add('mobilenet_v1_075', 'mobilenet_v1_0.75_192.ckpt', im_size=192)
        model_stack.add('mobilenet_v1_075', 'mobilenet_v1_0.75_160.ckpt', im_size=160)
        model_stack.add('mobilenet_v1_075', 'mobilenet_v1_0.75_128.ckpt', im_size=128)
        model_stack.add('mobilenet_v1_050', 'mobilenet_v1_0.50_224.ckpt', im_size=224)
        model_stack.add('mobilenet_v1_050', 'mobilenet_v1_0.50_192.ckpt', im_size=192)
        model_stack.add('mobilenet_v1_050', 'mobilenet_v1_0.50_160.ckpt', im_size=160)
        model_stack.add('mobilenet_v1_050', 'mobilenet_v1_0.50_128.ckpt', im_size=128)

        x_adv = x_input

        logits_list = []
        for model in model_stack.models:
            logits_list += model.get_logits(x_adv,
                                            augmentation_fn=augment_batch)
        logits = tf.reduce_mean(tf.stack(logits_list, axis=-1), axis=-1)

        preds = tf.nn.softmax(logits)
        preds = preds / tf.reduce_sum(preds, axis=-1, keep_dims=True)

        preds_sorted, sort_indices = tf.nn.top_k(
            preds, k=(num_classes - 1), sorted=True)
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

        cross_entropy = tf.losses.softmax_cross_entropy(weights, logits)

        scaled_signed_grad = eps * tf.sign(tf.gradients(cross_entropy, x_input)[0])
        x_adv = tf.stop_gradient(x_input + scaled_signed_grad)
        x_adv = tf.clip_by_value(x_adv, x_min, x_max)

        # Run computation
        with tf.Session() as sess:

            for model in model_stack.models:
                model.restore(sess)

            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                adv_images = sess.run(x_adv, feed_dict={x_input: images})
                save_images(adv_images, filenames, FLAGS.output_dir)


if __name__ == '__main__':
    tf.app.run()
