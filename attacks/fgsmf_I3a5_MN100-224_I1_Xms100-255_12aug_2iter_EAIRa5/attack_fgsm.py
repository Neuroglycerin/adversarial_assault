"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
time_start_script = time.time()

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

tf.flags.DEFINE_float(
    'p_threshold_antitarget', 0.50, 'What split of probabilities to anti-target.')

tf.flags.DEFINE_integer(
    'num_aug', 12, 'Number of augmented repetitions of the image to each net.')

tf.flags.DEFINE_integer(
    'max_iter', 2, 'Maximum number of iterations.')

tf.flags.DEFINE_float(
    'time_limit_per_100_samples', 500., 'Time limit in seconds per 100 samples.')

tf.flags.DEFINE_float(
    'fraction_to_update', 0.68, 'Fraction of pixels to update.')


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
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    eps = FLAGS.max_epsilon * 2.0 / 255.0
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        time_start_building_graph = time.time()

        # Fix seed for reproducibility
        tf.set_random_seed(9349008288)

        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        model_stack = model_loader.ModelLoaderStack(
            batch_size=FLAGS.batch_size,
            num_augmentations=FLAGS.num_aug)
        model_stack.add('inception_v3_5aux', 'models/inception_v3_5aux_299', im_size=299)
        model_stack.add('mobilenet_v1_100', 'models/mobilenet_v1_100_224', im_size=224)
        model_stack.add('inception_v1', 'models/inception_v1', im_size=224)
        model_stack.add('xception_multiscale2_flab', 'models/xception_multiscale2_flab_255', im_size=255)
        model_stack.add('inception_resnet_v2_5aux', 'models/ens_adv_inception_resnet_v2_5aux', im_size=299)


        def maybe_average_augs(x):
            if FLAGS.num_aug > 1:
                assert FLAGS.batch_size == 1
                x = tf.reduce_mean(x, axis=0, keep_dims=True)
            return x


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
                    post_resize_fn=augment_batch_post_resize)
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

        # Collect logits using true stimulus, with(!) augmentations
        logits_stack = update_logits(x_input)

        assert FLAGS.batch_size == 1
        avg_logits = tf.reduce_sum(logits_stack, axis=0, keep_dims=True)

        # Determine the true label
        preds = tf.nn.softmax(avg_logits)
        preds = preds / tf.reduce_sum(preds, axis=-1, keep_dims=True)

        preds_sorted, sort_indices = tf.nn.top_k(
            preds, k=(num_classes - 1), sorted=True)
        top_label_index = sort_indices[:, 0]
        sort_indices_offset = sort_indices + tf.expand_dims(tf.range(FLAGS.batch_size), 1)

        preds_sum = tf.cumsum(preds_sorted, axis=-1, exclusive=True)
        class_is_before_threshold = preds_sum < FLAGS.p_threshold_antitarget
        label_weights_sorted = preds_sorted * tf.cast(class_is_before_threshold, preds_sorted.dtype)
        label_weights_sorted = tf.square(label_weights_sorted)
        label_weights_sorted = label_weights_sorted / tf.reduce_sum(label_weights_sorted, axis=-1, keep_dims=True)
        # 1 x 1000
        label_weights = tf.scatter_nd(tf.expand_dims(tf.reshape(sort_indices_offset, [-1]), -1),
                                      tf.reshape(label_weights_sorted, [-1]),
                                      [FLAGS.batch_size * num_classes])
        label_weights = tf.reshape(label_weights, preds.shape)

        # Stop the gradients on these?
        #label_weights = tf.stop_gradient(label_weights)
        top_label_index = tf.stop_gradient(top_label_index)

        def update_x(x, local_logits, iter_num=tf.constant(0)):
            # First, we manipulate the image based on the output from the last
            # input image
            cross_entropy = tf.losses.softmax_cross_entropy(label_weights, local_logits)
            # First, we manipulate the image based on the gradients of the
            # cross entropy we just derived
            grad = tf.gradients(cross_entropy, x)[0]
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
            alpha = eps / tf.sqrt(tf.cast(iter_num, dtype=tf.float32) + 1)
            scaled_signed_grad = alpha * updates * tf.sign(grad)
            x_next = tf.stop_gradient(x + scaled_signed_grad)
            x_next = tf.clip_by_value(x_next, x_min, x_max)
            return x_next

        # We definitely update here
        x_adv = update_x(x_input, avg_logits)

        def test_accuracy(stack_of_logits):
            predicted_label = tf.argmax(stack_of_logits, axis=-1, output_type=tf.int32)
            label_is_right = tf.equal(top_label_index, predicted_label)
            any_label_is_right = tf.reduce_any(label_is_right, axis=0)
            return any_label_is_right


        #iter_limit = FLAGS.max_iter
        iter_limit = tf.placeholder(tf.int32, shape=[])
        num_iter_used = tf.constant(1)
        should_run_update = tf.ones([], dtype=tf.bool)
        prev_logits_stack = logits_stack
        prev_logits = avg_logits
        prev_x_adv = x_adv

        for iter_count in range(1, FLAGS.max_iter):
            # Generate augmented versions of input and forward propogate.

            logits_stack_kept = control_flow_ops.switch(prev_logits_stack,
                                                        should_run_update)[0]
            logits_stack_changed = update_logits(
                control_flow_ops.switch(x_adv, should_run_update)[1])
            logits_stack = control_flow_ops.merge(
                [logits_stack_changed, logits_stack_kept]
                )[0]

            # Check whether the current prediction is accurate
            have_time_for_one_more = tf.less(iter_count, iter_limit)
            label_is_right = test_accuracy(logits_stack)
            should_run_update = tf.logical_and(have_time_for_one_more,
                                               label_is_right)

            # Maybe update x_adv
            logits = tf.reduce_sum(logits_stack, axis=0, keep_dims=True)

            # Maybe update x_adv
            x_adv = tf.cond(should_run_update,
                            lambda: update_x(x_adv, logits, iter_count),
                            lambda: x_adv)

            prev_logits_stack = logits_stack
            prev_logits = logits
            prev_x_adv = x_adv

            num_iter_used = tf.cond(should_run_update,
                                    lambda: num_iter_used + 1,
                                    lambda: num_iter_used)


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
            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                if num_samples_shown < 10:
                    local_iter_limit = FLAGS.max_iter
                else:
                    # More fancy code should go here!
                    local_iter_limit = FLAGS.max_iter

                adv_images, batch_num_iter = sess.run(
                    [x_adv, num_iter_used],
                    feed_dict={x_input: images, iter_limit: local_iter_limit})
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
