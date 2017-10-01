"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
time_start_script = time.time()

import csv
import os
import math

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
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
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 1, 'How many images process at one time.')

tf.flags.DEFINE_integer(
    'num_aug', 1, 'Number of augmented repetitions of the image to each net.')

tf.flags.DEFINE_integer(
    'max_iter', 15, 'Maximum number of iterations.')

tf.flags.DEFINE_float(
    'time_limit_per_100_samples', 500., 'Time limit in seconds per 100 samples.')

tf.flags.DEFINE_float(
    'fraction_to_update', 0.68, 'Fraction of pixels to update initially.')

tf.flags.DEFINE_float(
    'initial_update_factor', 0.7, '')

tf.flags.DEFINE_float(
    'update_decay_rate', 0.92, '')

tf.flags.DEFINE_float(
    'rprop_increment', 1.1, '')

tf.flags.DEFINE_float(
    'rprop_decrement', 0.60, '')


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


def get_number_of_images(input_dir):
    tally = 0
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        tally += 1
    return tally


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
    image = tf.cond(
        tf.equal(index, 0),
        lambda: image,
        lambda: image_utils.random_rotate(image, max_angle=rotation_max_angle),
        )

    # Randomly flip the image horizontally
    if index is None:
        image = tf.image.random_flip_left_right(image)
    else:
        unflipped = lambda: image
        flipped = lambda: tf.image.flip_left_right(image)
        image = tf.cond(tf.less(tf.floormod(index, 2), 1), flipped, unflipped)

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
            if i == 0:
                i_offset = 0
            elif i % 2 == 0:
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
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    eps = FLAGS.max_epsilon * 2.0 / 255.0
    #initial_step_length = eps / math.sqrt(FLAGS.max_iter)
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)

    all_images_taget_class = load_target_class(FLAGS.input_dir)

    with tf.Graph().as_default():
        time_start_building_graph = time.time()

        # Fix seed for reproducibility
        tf.set_random_seed(9349008288)

        # Prepare graph
        iter_limit = tf.placeholder(tf.int32, shape=[])
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        target_class_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
        label_weights = tf.one_hot(target_class_input, num_classes)

        model_stack = model_loader.ModelLoaderStack(
            batch_size=FLAGS.batch_size,
            num_augmentations=FLAGS.num_aug)
        model_stack.add('inception_v3_5aux', 'models/inception_v3_5aux_299', im_size=299)
        model_stack.add('inception_resnet_v2_5aux', 'models/ens_adv_inception_resnet_v2_5aux', im_size=299)


        def update_logits(x):
            aux_weights = [1., .8, .6, .5, .4, .3, .3, .3]
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
                for i in range(num_logits_for_model):
                    model_logits[i] *= aux_weights[i] / sum(aux_weights[:num_logits_for_model])
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

        # Collect logits using true stimulus, with(!) augmentations
        logits_stack = update_logits(x_input)

        assert FLAGS.batch_size == 1
        avg_logits = tf.reduce_sum(logits_stack, axis=0, keep_dims=True)


        def update_x(x, local_logits, update_coefficients, prev_grad, iter_num=tf.constant(0)):
            # First, we manipulate the image based on the output from the last
            # input image
            cross_entropy = tf.losses.softmax_cross_entropy(label_weights, local_logits)
            # First, we manipulate the image based on the gradients of the
            # cross entropy we just derived
            initial_step_length = FLAGS.initial_update_factor * eps
            alpha = initial_step_length * tf.pow(FLAGS.update_decay_rate, tf.cast(iter_num, tf.float32))

            def get_initial_update_weights(grad):
                num_el = tf.size(grad)
                num_el_to_update = tf.cast(num_el, tf.float32) * tf.cast(FLAGS.fraction_to_update, tf.float32)
                num_el_to_update = tf.cast(tf.ceil(num_el_to_update), tf.int32)
                abs_grad = tf.abs(grad)
                _, sort_indices = tf.nn.top_k(tf.reshape(abs_grad, [-1]), num_el_to_update)
                unit_lengths = tf.cast(num_el - tf.range(num_el), dtype=tf.float32) \
                                / tf.cast(num_el, dtype=tf.float32)
                updates = tf.scatter_nd(tf.expand_dims(sort_indices, -1),
                                        tf.ones([num_el_to_update], dtype=grad.dtype),
                                        [num_el])
                updates = tf.reshape(updates, grad.shape)
                return updates

            def get_later_update_weights(signed_grad):
                multiplier = tf.where(tf.equal(signed_grad, prev_grad),
                                 FLAGS.rprop_increment * tf.ones_like(signed_grad),
                                 FLAGS.rprop_decrement * tf.ones_like(signed_grad))
                return multiplier


            grad = tf.gradients(cross_entropy, x)[0]
            signed_grad = tf.sign(grad)

            multiplier = tf.cond(
                tf.equal(iter_num, 0),
                lambda: get_initial_update_weights(grad),
                lambda: get_later_update_weights(signed_grad),
                )

            update_coefficients *= multiplier

            scaled_signed_grad = alpha * signed_grad * update_coefficients
            # Note we subtract the gradient here
            x_next = tf.stop_gradient(x - scaled_signed_grad)
            x_next = tf.clip_by_value(x_next, x_min, x_max)
            return x_next, update_coefficients, signed_grad

        # We definitely update here
        update_coefficients = tf.ones_like(x_input)
        prev_grad = tf.zeros_like(x_input)
        x_adv, update_coefficients, prev_grad = update_x(x_input, avg_logits, update_coefficients, prev_grad)

        def test_accuracy(stack_of_logits):
            predicted_label = tf.argmax(stack_of_logits, axis=-1, output_type=tf.int32)
            label_is_bad = tf.not_equal(target_class_input, predicted_label)
            any_label_is_bad = tf.reduce_any(label_is_bad, axis=0)
            return any_label_is_bad


        num_iter_used = tf.constant(1)
        prev_logits_stack = logits_stack
        prev_logits = avg_logits
        prev_x_adv = x_adv

        for iter_count in range(1, FLAGS.max_iter):
            # Generate augmented versions of input and forward propogate.

            logits_stack = update_logits(x_adv)

            # Maybe update x_adv
            logits = tf.reduce_sum(logits_stack, axis=0, keep_dims=True)

            # Maybe update x_adv
            x_adv, update_coefficients, prev_grad = update_x(
                x_adv, logits, update_coefficients, prev_grad, iter_count)

            prev_logits_stack = logits_stack
            prev_logits = logits
            prev_x_adv = x_adv

            num_iter_used = num_iter_used + 1

        # Project vector to extremities
        adv_vector = x_adv - x_input
        max_delta = tf.reduce_max(tf.abs(x_input - x_adv))
        # Now unit vector under inf norm
        adv_vector = adv_vector / max_delta
        # We project as far as possible in this direction
        x_next = x_input + adv_vector * eps
        x_next = tf.clip_by_value(x_next, x_min, x_max)
        x_adv = x_next


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
            num_samples_shown = 0
            num_iter_tally = []
            last_report_nsamp = 0
            logging.info('Starting to generate images after {} seconds'.format(
                time_start_generating - time_start_script))

            num_img = get_number_of_images(FLAGS.input_dir)
            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                target_class_for_batch = (
                    [all_images_taget_class[n] for n in filenames]
                    + [0] * (FLAGS.batch_size - len(filenames)))

                local_iter_limit = FLAGS.max_iter

                adv_images, batch_num_iter = sess.run(
                    [x_adv, num_iter_used],
                    feed_dict={x_input: images,
                               iter_limit: local_iter_limit,
                               target_class_input: target_class_for_batch,
                               }
                    )
                save_images(adv_images, filenames, FLAGS.output_dir)

                num_iter_tally.append(batch_num_iter)

                num_samples_shown += FLAGS.batch_size
                if num_samples_shown // 100 > last_report_nsamp // 100:
                    tally = np.array(num_iter_tally)
                    logging.info('After {} samples: Took {} sec/sample. '
                                 'Using {}+/-{} iterations.'.format(
                                    num_samples_shown,
                                    (time.time() - time_start_generating) / num_samples_shown,
                                    np.mean(tally),
                                    np.std(tally),
                                    )
                                 )
                    last_report_nsamp = num_samples_shown

            logging.info('Finished with internal duration of {} seconds'.format(
                time.time() - time_start_script))


if __name__ == '__main__':
    tf.app.run()
