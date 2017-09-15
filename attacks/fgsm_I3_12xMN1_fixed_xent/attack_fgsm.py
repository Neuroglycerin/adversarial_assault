"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
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

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 1, 'How many images process at one time.')

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
        x_224 = tf.image.resize_images(x_input, (224, 224))
        x_192 = tf.image.resize_images(x_input, (192, 192))
        x_160 = tf.image.resize_images(x_input, (160, 160))
        x_128 = tf.image.resize_images(x_input, (128, 128))

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            inception.inception_v3(
                x_input, num_classes=num_classes, is_training=False)


        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            mobilenet_v1.mobilenet_v1(
                x_224, num_classes=num_classes, is_training=False,
                spatial_squeeze=False, scope='MobilenetV1_100_244')

        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            mobilenet_v1.mobilenet_v1(
                x_192, num_classes=num_classes, is_training=False,
                spatial_squeeze=False, scope='MobilenetV1_100_192')

        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            mobilenet_v1.mobilenet_v1(
                x_160, num_classes=num_classes, is_training=False,
                spatial_squeeze=False, scope='MobilenetV1_100_160')

        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            mobilenet_v1.mobilenet_v1(
                x_128, num_classes=num_classes, is_training=False,
                spatial_squeeze=False, scope='MobilenetV1_100_128')


        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            mobilenet_v1.mobilenet_v1_075(
                x_224, num_classes=num_classes, is_training=False,
                spatial_squeeze=False, scope='MobilenetV1_075_244')

        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            mobilenet_v1.mobilenet_v1_075(
                x_192, num_classes=num_classes, is_training=False,
                spatial_squeeze=False, scope='MobilenetV1_075_192')

        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            mobilenet_v1.mobilenet_v1_075(
                x_160, num_classes=num_classes, is_training=False,
                spatial_squeeze=False, scope='MobilenetV1_075_160')

        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            mobilenet_v1.mobilenet_v1_075(
                x_128, num_classes=num_classes, is_training=False,
                spatial_squeeze=False, scope='MobilenetV1_075_128')


        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            mobilenet_v1.mobilenet_v1_050(
                x_224, num_classes=num_classes, is_training=False,
                spatial_squeeze=False, scope='MobilenetV1_050_244')

        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            mobilenet_v1.mobilenet_v1_050(
                x_192, num_classes=num_classes, is_training=False,
                spatial_squeeze=False, scope='MobilenetV1_050_192')

        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            mobilenet_v1.mobilenet_v1_050(
                x_160, num_classes=num_classes, is_training=False,
                spatial_squeeze=False, scope='MobilenetV1_050_160')

        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            mobilenet_v1.mobilenet_v1_050(
                x_128, num_classes=num_classes, is_training=False,
                spatial_squeeze=False, scope='MobilenetV1_050_128')


        x_adv = x_input
        all_logits = []
        all_preds = 0
        n_preds = 0

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, end_points = inception.inception_v3(
                x_adv, num_classes=num_classes, is_training=False, reuse=True)
        all_logits.append(logits)
        all_logits.append(end_points['AuxLogits'])
        all_preds += end_points['Predictions']
        n_preds += 1

        x_224 = tf.image.resize_images(x_adv, (224, 224))
        x_192 = tf.image.resize_images(x_adv, (192, 192))
        x_160 = tf.image.resize_images(x_adv, (160, 160))
        x_128 = tf.image.resize_images(x_adv, (128, 128))

        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            logits, end_points = mobilenet_v1.mobilenet_v1(
                x_224, num_classes=num_classes, is_training=False, reuse=True,
                spatial_squeeze=False, scope='MobilenetV1_100_244')
        logits = tf.squeeze(logits, axis=(1, 2))
        preds = tf.squeeze(end_points['Predictions'], axis=(1, 2))
        all_logits.append(logits)
        all_preds += preds
        n_preds += 1

        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            logits, end_points = mobilenet_v1.mobilenet_v1(
                x_192, num_classes=num_classes, is_training=False, reuse=True,
                spatial_squeeze=False, scope='MobilenetV1_100_192')
        logits = tf.squeeze(logits, axis=(1, 2))
        preds = tf.squeeze(end_points['Predictions'], axis=(1, 2))
        all_logits.append(logits)
        all_preds += preds
        n_preds += 1

        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            logits, end_points = mobilenet_v1.mobilenet_v1(
                x_160, num_classes=num_classes, is_training=False, reuse=True,
                spatial_squeeze=False, scope='MobilenetV1_100_160')
        logits = tf.squeeze(logits, axis=(1, 2))
        preds = tf.squeeze(end_points['Predictions'], axis=(1, 2))
        all_logits.append(logits)
        all_preds += preds
        n_preds += 1

        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            logits, end_points = mobilenet_v1.mobilenet_v1(
                x_128, num_classes=num_classes, is_training=False, reuse=True,
                spatial_squeeze=False, scope='MobilenetV1_100_128')
        logits = tf.squeeze(logits, axis=(1, 2))
        preds = tf.squeeze(end_points['Predictions'], axis=(1, 2))
        all_logits.append(logits)
        all_preds += preds
        n_preds += 1


        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            logits, end_points = mobilenet_v1.mobilenet_v1_075(
                x_224, num_classes=num_classes, is_training=False, reuse=True,
                spatial_squeeze=False, scope='MobilenetV1_075_244')
        logits = tf.squeeze(logits, axis=(1, 2))
        preds = tf.squeeze(end_points['Predictions'], axis=(1, 2))
        all_logits.append(logits)
        all_preds += preds
        n_preds += 1

        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            logits, end_points = mobilenet_v1.mobilenet_v1_075(
                x_192, num_classes=num_classes, is_training=False, reuse=True,
                spatial_squeeze=False, scope='MobilenetV1_075_192')
        logits = tf.squeeze(logits, axis=(1, 2))
        preds = tf.squeeze(end_points['Predictions'], axis=(1, 2))
        all_logits.append(logits)
        all_preds += preds
        n_preds += 1

        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            logits, end_points = mobilenet_v1.mobilenet_v1_075(
                x_160, num_classes=num_classes, is_training=False, reuse=True,
                spatial_squeeze=False, scope='MobilenetV1_075_160')
        logits = tf.squeeze(logits, axis=(1, 2))
        preds = tf.squeeze(end_points['Predictions'], axis=(1, 2))
        all_logits.append(logits)
        all_preds += preds
        n_preds += 1

        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            logits, end_points = mobilenet_v1.mobilenet_v1_075(
                x_128, num_classes=num_classes, is_training=False, reuse=True,
                spatial_squeeze=False, scope='MobilenetV1_075_128')
        logits = tf.squeeze(logits, axis=(1, 2))
        preds = tf.squeeze(end_points['Predictions'], axis=(1, 2))
        all_logits.append(logits)
        all_preds += preds
        n_preds += 1


        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            logits, end_points = mobilenet_v1.mobilenet_v1_050(
                x_224, num_classes=num_classes, is_training=False, reuse=True,
                spatial_squeeze=False, scope='MobilenetV1_050_244')
        logits = tf.squeeze(logits, axis=(1, 2))
        preds = tf.squeeze(end_points['Predictions'], axis=(1, 2))
        all_logits.append(logits)
        all_preds += preds
        n_preds += 1

        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            logits, end_points = mobilenet_v1.mobilenet_v1_050(
                x_192, num_classes=num_classes, is_training=False, reuse=True,
                spatial_squeeze=False, scope='MobilenetV1_050_192')
        logits = tf.squeeze(logits, axis=(1, 2))
        preds = tf.squeeze(end_points['Predictions'], axis=(1, 2))
        all_logits.append(logits)
        all_preds += preds
        n_preds += 1

        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            logits, end_points = mobilenet_v1.mobilenet_v1_050(
                x_160, num_classes=num_classes, is_training=False, reuse=True,
                spatial_squeeze=False, scope='MobilenetV1_050_160')
        logits = tf.squeeze(logits, axis=(1, 2))
        preds = tf.squeeze(end_points['Predictions'], axis=(1, 2))
        all_logits.append(logits)
        all_preds += preds
        n_preds += 1

        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            logits, end_points = mobilenet_v1.mobilenet_v1_050(
                x_128, num_classes=num_classes, is_training=False, reuse=True,
                spatial_squeeze=False, scope='MobilenetV1_050_128')
        logits = tf.squeeze(logits, axis=(1, 2))
        preds = tf.squeeze(end_points['Predictions'], axis=(1, 2))
        all_logits.append(logits)
        all_preds += preds
        n_preds += 1


        preds = all_preds / n_logits

        preds_max = tf.reduce_max(preds, 1, keep_dims=True)
        y = tf.to_float(tf.equal(preds, preds_max))
        y = y / tf.reduce_sum(y, 1, keep_dims=True)
        for logits in all_logits:
            cross_entropy += tf.losses.softmax_cross_entropy(y, logits)

        scaled_signed_grad = eps * tf.sign(tf.gradients(cross_entropy, x_input)[0])
        x_adv = tf.stop_gradient(x_input + scaled_signed_grad)
        x_adv = tf.clip_by_value(x_adv, x_min, x_max)

        # Run computation
        with tf.Session() as sess:

            saver = tf.train.Saver(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='InceptionV3'))
            saver.restore(sess, 'inception_v3.ckpt')


            var_dict = {('MobilenetV1' + v.op.name.lstrip('MobilenetV1_100_244')): v
                for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='MobilenetV1_100_244')}
            saver = tf.train.Saver(var_list=var_dict)
            saver.restore(sess, 'mobilenet_v1_1.0_224.ckpt')

            var_dict = {('MobilenetV1' + v.op.name.lstrip('MobilenetV1_100_192')): v
                for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='MobilenetV1_100_192')}
            saver = tf.train.Saver(var_list=var_dict)
            saver.restore(sess, 'mobilenet_v1_1.0_192.ckpt')

            var_dict = {('MobilenetV1' + v.op.name.lstrip('MobilenetV1_100_160')): v
                for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='MobilenetV1_100_160')}
            saver = tf.train.Saver(var_list=var_dict)
            saver.restore(sess, 'mobilenet_v1_1.0_160.ckpt')

            var_dict = {('MobilenetV1' + v.op.name.lstrip('MobilenetV1_100_128')): v
                for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='MobilenetV1_100_128')}
            saver = tf.train.Saver(var_list=var_dict)
            saver.restore(sess, 'mobilenet_v1_1.0_128.ckpt')


            var_dict = {('MobilenetV1' + v.op.name.lstrip('MobilenetV1_075_244')): v
                for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='MobilenetV1_075_244')}
            saver = tf.train.Saver(var_list=var_dict)
            saver.restore(sess, 'mobilenet_v1_0.75_224.ckpt')

            var_dict = {('MobilenetV1' + v.op.name.lstrip('MobilenetV1_075_192')): v
                for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='MobilenetV1_075_192')}
            saver = tf.train.Saver(var_list=var_dict)
            saver.restore(sess, 'mobilenet_v1_0.75_192.ckpt')

            var_dict = {('MobilenetV1' + v.op.name.lstrip('MobilenetV1_075_160')): v
                for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='MobilenetV1_075_160')}
            saver = tf.train.Saver(var_list=var_dict)
            saver.restore(sess, 'mobilenet_v1_0.75_160.ckpt')

            var_dict = {('MobilenetV1' + v.op.name.lstrip('MobilenetV1_075_128')): v
                for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='MobilenetV1_075_128')}
            saver = tf.train.Saver(var_list=var_dict)
            saver.restore(sess, 'mobilenet_v1_0.75_128.ckpt')


            var_dict = {('MobilenetV1' + v.op.name.lstrip('MobilenetV1_050_244')): v
                for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='MobilenetV1_050_244')}
            saver = tf.train.Saver(var_list=var_dict)
            saver.restore(sess, 'mobilenet_v1_0.50_224.ckpt')

            var_dict = {('MobilenetV1' + v.op.name.lstrip('MobilenetV1_050_192')): v
                for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='MobilenetV1_050_192')}
            saver = tf.train.Saver(var_list=var_dict)
            saver.restore(sess, 'mobilenet_v1_0.50_192.ckpt')

            var_dict = {('MobilenetV1' + v.op.name.lstrip('MobilenetV1_050_160')): v
                for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='MobilenetV1_050_160')}
            saver = tf.train.Saver(var_list=var_dict)
            saver.restore(sess, 'mobilenet_v1_0.50_160.ckpt')

            var_dict = {('MobilenetV1' + v.op.name.lstrip('MobilenetV1_050_128')): v
                for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='MobilenetV1_050_128')}
            saver = tf.train.Saver(var_list=var_dict)
            saver.restore(sess, 'mobilenet_v1_0.50_128.ckpt')


            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                adv_images = sess.run(x_adv, feed_dict={x_input: images})
                save_images(adv_images, filenames, FLAGS.output_dir)


if __name__ == '__main__':
    tf.app.run()
