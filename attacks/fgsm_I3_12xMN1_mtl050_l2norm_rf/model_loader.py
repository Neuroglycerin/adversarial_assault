"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from slim import nets_factory


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
    for i in range(parts):
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
                 num_channels=3):

        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.im_size = im_size
        self.batch_size = batch_size
        self.num_augmentations = num_augmentations
        self.num_channels = num_channels

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


    def get_logits(self, x, augmentation_fn=None, resize_mode=0):
        if not self._is_graph_prepared:
            self.prepare_graph()

        if augmentation_fn is not None:
            x = augmentation_fn(x)
        x = tf.image.resize_images(x, (self.im_size, self.im_size), resize_mode)

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

        if 'AuxLogits' in end_points:
            logits += end_points['AuxLogits']

        if needs_manual_squeeze:
            logits = tf.squeeze(logits, axis=(1, 2))

        if self.num_classes == 1000:
            logits = tf.concat(
                (tf.zeros((x.shape[0], 1), dtype=tf.float32), logits),
                axis=-1)

        return logits


    def restore(self):
        var_dict = {(self._scope_in_checkpoint + v.op.name.lstrip(self.scope_to_use)): v
            for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                       scope=self.scope_to_use)}
        saver = tf.train.Saver(var_list=var_dict)
        return saver.restore(sess, self.checkpoint_path)


class ModelLoaderStack():

    models = []
    num_models_added = 0

    def add(self, model_name, checkpoint_path, scope_to_use=None, **kwargs):

        if scope_to_use is None:
            scope_to_use = 'Model{}/{}'.format(
                self.num_models_added, model_name_to_scope(model_name)
                )

        self.models.append(ModelLoader(model_name,
                                       checkpoint_path,
                                       scope_to_use=scope_to_use,
                                       **kwargs)
                                       )
        self.num_models_added += 1
