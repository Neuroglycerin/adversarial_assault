"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from slim.nets import nets_factory


model_name_to_default_scope_map = {
    'alexnet_v2': 'alexnet_v2',
    'cifarnet': 'CifarNet',
    'deadversiser_noop': 'noop',
    'deadversiser_v0': 'deadversarialiser',
    'deadversiser_v1': 'deadversarialiser',
    'deadversiser_v2': 'deadversarialiser',
    'overfeat': 'overfeat',
    'vgg_a': 'vgg_a',
    'vgg_16': 'vgg_16',
    'vgg_19': 'vgg_19',
    'inception_v1': 'InceptionV1',
    'inception_v2': 'InceptionV2',
    'inception_v3': 'InceptionV3',
    'inception_v4': 'InceptionV4',
    'inception_resnet_v2': 'InceptionResnetV2',
    'lenet': 'LeNet',
    'resnet_v1_50': 'resnet_v1_50',
    'resnet_v1_101': 'resnet_v1_101',
    'resnet_v1_152': 'resnet_v1_152',
    'resnet_v1_200': 'resnet_v1_200',
    'resnet_v2_50': 'resnet_v2_50',
    'resnet_v2_101': 'resnet_v2_101',
    'resnet_v2_152': 'resnet_v2_152',
    'resnet_v2_200': 'resnet_v2_200',
    'mobilenet_v1': 'MobilenetV1',
    'mobilenet_v1x': 'MobilenetV1',
    'mobilenet_v1_multiscale': 'MobilenetV1Multiscale',
    'mobilenet_v1_multiscale2': 'MobilenetV1',
    'mobilenet_v1_lahrelu': 'MobilenetV1',
    'mobilenet_v1_lahrelu2': 'MobilenetV1',
    'mobilenet_v1_multiscale2lah': 'MobilenetV1',
    'xception': 'Xception',
    }


model_groups_which_need_manual_squeeze = [
    'mobilenet',
    'resnet',
    ]

model_groups_which_take_1000_classes = [
    'resnet',
    'vgg',
    ]


def model_name_to_scope(model_name):
    if model_name in model_name_to_default_scope_map:
        return model_name_to_default_scope_map[model_name]
    parts = model_name.split('_')
    for i in range(1, len(parts)):
        partial_model_name = '_'.join(parts[:-i])
        if partial_model_name in model_name_to_default_scope_map:
            return model_name_to_default_scope_map[partial_model_name]
    raise ValueError('Model name (and its leading parital parts) {} not'
                     'found'.format(model_name))


class ModelLoader():

    _is_graph_prepared = False

    def __init__(self,
                 model_name,
                 checkpoint_path,
                 scope_to_use=None,
                 im_size=299,
                 batch_size=1,
                 num_augmentations=1,
                 num_channels=3,
                 weight=1.):

        self.model_name = model_name
        self.checkpoint_path = self._sanitise_checkpoint_path(checkpoint_path)
        self.im_size = im_size
        self.batch_size = batch_size
        self.num_augmentations = num_augmentations
        self.num_channels = num_channels
        self.weight = weight

        self._scope_in_checkpoint = model_name_to_scope(model_name)
        self._model_group = model_name.split('_')[0]

        if scope_to_use is None:
            self.scope_to_use = self._scope_in_checkpoint
        else:
            self.scope_to_use = scope_to_use

        self._takes_1000_classes = self._model_group in \
                model_groups_which_take_1000_classes
        if self._takes_1000_classes:
            self.num_classes = 1000
        else:
            self.num_classes = 1001


    def _sanitise_checkpoint_path(self, checkpoint_path):
        if os.path.isfile(checkpoint_path):
            return checkpoint_path
        if not os.path.isdir(checkpoint_path):
            print('Caution! The file {} does not exist!'
                  ''.format(checkpoint_path))
            return checkpoint_path
        # List directory contents, for files containing .ckpt
        files = os.listdir(checkpoint_path)
        files = [f for f in files if '.ckpt' in f]
        # If there is only one, that's it!
        if len(files) == 1:
            return os.path.join(checkpoint_path, files[0])
        if not files:
            raise EnvironmentError('No .ckpt file found in path {}'
                                   ''.format(checkpoint_path))
        # If there is multiple, look at - separated number
        ckpt_step_numbers = []
        # File names should look like this:
        # model.ckpt-26140.meta
        for fname in files:
            parts = os.path.splitext(fname)[0].split('-')
            if len(parts) < 2:
                ckpt_step_numbers.append(-1)
            else:
                ckpt_step_numbers.append(int(parts[1]))
        highest_step = max(ckpt_step_numbers)
        # Compare ints, select the highest
        for fname, ckpt_step_number in zip(files, ckpt_step_numbers):
            if ckpt_step_number == highest_step:
                break
        # Note that if there are multiple files which don't fit the naming
        # convension with a hyphen, we take the arbitrarily last file now
        # Cut off the junk
        fname = fname[0:fname.find('.ckpt')] + '.ckpt'
        if highest_step > -1:
            fname += '-' + str(highest_step)
        return os.path.join(checkpoint_path, fname)


    def prepare_graph(self):

        batch_shape = (self.batch_size * self.num_augmentations,
                       self.im_size,
                       self.im_size,
                       self.num_channels)
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        arg_scope = nets_factory.arg_scopes_map[self.model_name]
        network_fn = nets_factory.networks_map[self.model_name]

        kwargs = {}
        needs_manual_squeeze = self._model_group in \
                model_groups_which_need_manual_squeeze
        if needs_manual_squeeze:
            kwargs['spatial_squeeze'] = False

        with tf.contrib.slim.arg_scope(arg_scope()):
            network_fn(
                x_input,
                num_classes=self.num_classes,
                is_training=False,
                reuse=False,
                scope=self.scope_to_use,
                **kwargs)

        self._is_graph_prepared = True


    def get_logits(self, x,
                   pre_resize_fn=None,
                   post_resize_fn=None,
                   resize_mode=0):
        if not self._is_graph_prepared:
            self.prepare_graph()

        if pre_resize_fn is not None:
            x = pre_resize_fn(x)

        func = lambda y, method: tf.image.resize_images(
            y, (self.im_size, self.im_size), method)

        # Resize with potentially random resize_mode
        num_cases = 2
        x = control_flow_ops.merge([
            func(control_flow_ops.switch(x, tf.equal(resize_mode, case))[1], case)
            for case in range(num_cases)])[0]

        if post_resize_fn is not None:
            x = post_resize_fn(x)

        kwargs = {}
        needs_manual_squeeze = self._model_group in \
                model_groups_which_need_manual_squeeze
        if needs_manual_squeeze:
            kwargs['spatial_squeeze'] = False

        arg_scope = nets_factory.arg_scopes_map[self.model_name]
        network_fn = nets_factory.networks_map[self.model_name]

        with tf.contrib.slim.arg_scope(arg_scope()):
            logits, end_points = network_fn(
                x,
                num_classes=self.num_classes,
                is_training=False,
                reuse=True,
                scope=self.scope_to_use,
                **kwargs)

        outputs = [logits]

        if 'AuxLogits' in end_points:
            outputs.append(end_points['AuxLogits'])
        other_aux = []
        for key in end_points:
            if key.startswith('AuxLogits_'):
                other_aux.append(key)
        other_aux = sorted(other_aux)
        for key in other_aux:
            outputs.append(end_points[key])

        if needs_manual_squeeze:
            outputs = [tf.squeeze(logits, axis=(1, 2)) for logits in outputs]

        if self.num_classes == 1000:
            outputs = [
                tf.concat(
                    (tf.zeros((x.shape[0], 1), dtype=tf.float32), logits),
                    axis=-1)
                for logits in outputs
                ]

        return outputs


    def restore(self, sess):
        var_dict = {(self._scope_in_checkpoint + v.op.name.lstrip(self.scope_to_use)): v
            for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                       scope=self.scope_to_use)}
        saver = tf.train.Saver(var_list=var_dict)
        return saver.restore(sess, self.checkpoint_path)


class ModelLoaderStack():

    models = []
    num_models_added = 0

    def __init__(self, batch_size=1, num_augmentations=1):
        self.batch_size = batch_size
        self.num_augmentations = num_augmentations

    def add(self, model_name, checkpoint_path, scope_to_use=None,
            batch_size=None, num_augmentations=None, **kwargs):

        if scope_to_use is None:
            scope_to_use = 'Model{}_{}'.format(
                self.num_models_added, model_name_to_scope(model_name)
                )

        if batch_size is None:
            batch_size = self.batch_size
        if num_augmentations is None:
            num_augmentations = self.num_augmentations

        self.models.append(ModelLoader(model_name,
                                       checkpoint_path,
                                       scope_to_use=scope_to_use,
                                       batch_size=batch_size,
                                       num_augmentations=num_augmentations,
                                       **kwargs)
                                       )
        self.num_models_added += 1
