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
from tensorflow.contrib.slim.nets import inception

from cleverhans.attacks import BasicIterativeMethod

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

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

tf.flags.DEFINE_float(
    'nb_iter', 20, 'Number of iterations of adversarial perturbation.')

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
  print(input_dir)
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    print(filepath)
    #with tf.gfile.Open(filepath) as f:
    image = imread(filepath, mode='RGB').astype(np.float) / 255.0
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


class InceptionModel(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      logits, end_points = inception.inception_v3(
          x_input, num_classes=self.num_classes, is_training=False,
          reuse=reuse)
    self.built = True
    #output = end_points['Predictions']
    # Strip off the extra reshape op at the output
    #probs = output.op.inputs[0]
    return logits


def main(_):
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  tf.logging.set_verbosity(tf.logging.INFO)

  all_images_taget_class = load_target_class(FLAGS.input_dir)

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    model = InceptionModel(num_classes)

    target_class_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
    one_hot_target_class = tf.one_hot(target_class_input, num_classes)

    nb_iter = FLAGS.nb_iter
    eps_iter = 2 * eps / nb_iter

    bim = BasicIterativeMethod(model)
    x_adv = bim.generate(x_input, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter,
                         y_target=one_hot_target_class,
                         clip_min=-1., clip_max=1.)

    # Run computation
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=FLAGS.checkpoint_path,
        master=FLAGS.master)

    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
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
