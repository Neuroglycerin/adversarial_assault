"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave

import tensorflow as tf

import mobilenet_v1

slim = tf.contrib.slim


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

tf.flags.DEFINE_float(
    'iter_alpha', 1.0, 'Step size for one iteration.')

tf.flags.DEFINE_integer(
    'num_iter', 20, 'Number of iterations.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_target_class(input_dir):
  """Loads target classes."""
  with tf.gfile.Open(os.path.join(input_dir, 'target_class.csv')) as f:
    return {row[0]: int(row[1]) for row in csv.reader(f) if len(row) >= 2}


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
      image = imread(f, mode='RGB').astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
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
      imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')


def main(_):
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  alpha = 2.0 * FLAGS.iter_alpha / 255.0
  num_iter = FLAGS.num_iter
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  tf.logging.set_verbosity(tf.logging.INFO)

  all_images_taget_class = load_target_class(FLAGS.input_dir)

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
    x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

    with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
      mobilenet_v1.mobilenet_v1(
          x_input, num_classes=num_classes, is_training=False,
          spatial_squeeze=False)

    x_adv = x_input
    target_class_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
    one_hot_target_class = tf.one_hot(target_class_input, num_classes)

    for _ in range(num_iter):

      cross_entropy = 0

      x_reshaped = tf.image.resize_images(x_adv, (224, 224))

      with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
        logits, end_points = mobilenet_v1.mobilenet_v1(
            x_reshaped, num_classes=num_classes, is_training=False, reuse=True,
            spatial_squeeze=False)
        logits = tf.squeeze(logits, axis=(1, 2))
      cross_entropy += tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                      logits)

      x_next = x_adv - alpha * tf.sign(tf.gradients(cross_entropy, x_adv)[0])
      x_next = tf.clip_by_value(x_next, x_min, x_max)
      x_adv = x_next

    # Run computation
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        master=FLAGS.master)

    with tf.Session() as sess:
      saver0 = tf.train.Saver()
      saver0.restore(sess, 'mobilenet_v1_1.0_224.ckpt')

      for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        target_class_for_batch = (
            [all_images_taget_class[n] for n in filenames]
            + [0] * (FLAGS.batch_size - len(filenames)))
        adv_images = sess.run(x_adv,
                              feed_dict={
                                  x_input: images,
                                  target_class_input: target_class_for_batch
                              })
        save_images(adv_images, filenames, FLAGS.output_dir)


if __name__ == '__main__':
  tf.app.run()
