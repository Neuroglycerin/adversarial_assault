"""Implementation of sample defense.

This defense loads inception v3 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from scipy.misc import imread

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim


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
    'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS
devset_dir = 'comparison_images'

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
    #with tf.gfile.Open(filepath) as f:
    #  image = imread(f, mode='RGB').astype(np.float) / 255.0
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


def main(_):
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    with slim.arg_scope(inception.inception_v3_arg_scope()):
      _, end_points = inception.inception_v3(
          x_input, num_classes=num_classes, is_training=False)

    predicted_labels = tf.argmax(end_points['Predictions'], 1)

    # Run computation
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=FLAGS.checkpoint_path,
        master=FLAGS.master)

    # load comparison images
    full_batch = (1000, batch_shape[1], batch_shape[2], batch_shape[3])
    paths = tf.gfile.Glob(os.path.join(devset_dir, '*.png'))
    comparison = np.zeros(full_batch)
    for idx, filepath in enumerate(paths):
      image = imread(filepath, mode='RGB').astype(np.float) / 255.0
      # Images for inception classifier are normalized to be in [-1, 1] interval.
      comparison[idx, :, :, :] = image * 2.0 - 1.0

    def comparison_swap(images):
      swapped = np.zeros_like(images)
      #l2_distance = np.square(comparison[np.newaxis,:,:,:,:] - images[:,np.newaxis,:,:,:])
      #closest_inds = np.argmin(np.sum(l2_distance, axis=[2,3,4]), axis=1)
      for i in range(images.shape[0]):
        l2_distance = np.sum(np.square(comparison - images[np.newaxis,i]), axis=(1,2,3))
        closest_ind = np.argmin(l2_distance)
        swapped[i,:,:,:] = comparison[closest_ind,:,:,:]
        #swapped[i,:,:,:] = comparison[closest_inds[i],:,:,:]
      return swapped

    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
      with open(FLAGS.output_file, 'w') as out_file:
        for filenames, images in load_images(FLAGS.input_dir, batch_shape):
          images = comparison_swap(images)
          labels = sess.run(predicted_labels, feed_dict={x_input: images})
          for filename, label in zip(filenames, labels):
            out_file.write('{0},{1}\n'.format(filename, label))


if __name__ == '__main__':
  tf.app.run()