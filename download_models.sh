#!/bin/bash

# exit on first error
set -e

# directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODELS_DIR="${SCRIPT_DIR}/models"

mkdir -p "${MODELS_DIR}"

cd "${MODELS_DIR}"


###############################################################################
# Models provided by competiton hosts
###############################################################################

if [ ! -e "adv_inception_v3.ckpt.data-00000-of-00001" ]; then
    # Download ensemble adversarially trained inception resnet v2 checkpoint
    wget http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz
    tar -xvzf adv_inception_v3_2017_08_18.tar.gz
    rm adv_inception_v3_2017_08_18.tar.gz
fi
if [ ! -e "ens_adv_inception_resnet_v2.ckpt.data-00000-of-00001" ]; then
    # Download ensemble adversarially trained inception resnet v2 checkpoint
    wget http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz
    tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
    rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz
fi


###############################################################################
# Models from Keras
###############################################################################
if [ ! e 'xception_weights_tf_dim_ordering_tf_kernels.h5' ]; then
    wget https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5
fi


###############################################################################
# Models from tf-slim
###############################################################################

#------------------------------------------------------------------------------
# Inception
#------------------------------------------------------------------------------
if [ ! -e "inception_v1.ckpt" ]; then
    wget http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz
    tar -xvzf inception_v1_2016_08_28.tar.gz
    rm inception_v1_2016_08_28.tar.gz
fi
if [ ! -e "inception_v2.ckpt" ]; then
    wget http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz
    tar -xvzf inception_v2_2016_08_28.tar.gz
    rm inception_v2_2016_08_28.tar.gz
fi
if [ ! -e "inception_v3.ckpt" ]; then
    wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
    tar -xvzf inception_v3_2016_08_28.tar.gz
    rm inception_v3_2016_08_28.tar.gz
fi
if [ ! -e "inception_v4.ckpt" ]; then
    wget http://download.tensorflow.org/models/inception_v4_2016_08_28.tar.gz
    tar -xvzf inception_v4_2016_08_28.tar.gz
    rm inception_v4_2016_08_28.tar.gz
fi

#------------------------------------------------------------------------------
# Inception ResNet
#------------------------------------------------------------------------------
if [ ! -e "inception_resnet_v2_2016_08_30.ckpt" ]; then
    wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
    tar -xvzf inception_resnet_v2_2016_08_30.tar.gz
    rm inception_resnet_v2_2016_08_30.tar.gz
fi

#------------------------------------------------------------------------------
# ResNet
#------------------------------------------------------------------------------
if [ ! -e "resnet_v1_50.ckpt" ]; then
    wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
    tar -xvzf resnet_v1_50_2016_08_28.tar.gz
    rm resnet_v1_50_2016_08_28.tar.gz
fi
if [ ! -e "resnet_v1_101.ckpt" ]; then
    wget http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
    tar -xvzf resnet_v1_101_2016_08_28.tar.gz
    rm resnet_v1_101_2016_08_28.tar.gz
fi
if [ ! -e "resnet_v1_152.ckpt" ]; then
    wget http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz
    tar -xvzf resnet_v1_152_2016_08_28.tar.gz
    rm resnet_v1_152_2016_08_28.tar.gz
fi
if [ ! -e "resnet_v2_50.ckpt" ]; then
    wget http://download.tensorflow.org/models/resnet_v2_50_2016_08_28.tar.gz
    tar -xvzf resnet_v2_50_2016_08_28.tar.gz
    rm resnet_v2_50_2016_08_28.tar.gz
fi
if [ ! -e "resnet_v2_101.ckpt" ]; then
    wget http://download.tensorflow.org/models/resnet_v2_101_2016_08_28.tar.gz
    tar -xvzf resnet_v2_101_2016_08_28.tar.gz
    rm resnet_v2_101_2016_08_28.tar.gz
fi
if [ ! -e "resnet_v2_152.ckpt" ]; then
    wget http://download.tensorflow.org/models/resnet_v2_152_2016_08_28.tar.gz
    tar -xvzf resnet_v2_152_2016_08_28.tar.gz
    rm resnet_v2_152_2016_08_28.tar.gz
fi

#------------------------------------------------------------------------------
# VGG
#------------------------------------------------------------------------------
if [ ! -e "vgg_16.ckpt" ]; then
    wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
    tar -xvzf vgg_16_2016_08_28.tar.gz
    rm vgg_16_2016_08_28.tar.gz
fi
if [ ! -e "vgg_19.ckpt" ]; then
    wget http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz
    tar -xvzf vgg_19_2016_08_28.tar.gz
    rm vgg_19_2016_08_28.tar.gz
fi

#------------------------------------------------------------------------------
# MobileNet
#------------------------------------------------------------------------------
if [ ! -e "mobilenet_v1_1.0_224.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_1.0_224_2017_06_14.tar.gz
    rm mobilenet_v1_1.0_224_2017_06_14.tar.gz
fi
if [ ! -e "mobilenet_v1_1.0_192.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_1.0_192_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_1.0_192_2017_06_14.tar.gz
    rm mobilenet_v1_1.0_192_2017_06_14.tar.gz
fi
if [ ! -e "mobilenet_v1_1.0_160.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_1.0_160_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_1.0_160_2017_06_14.tar.gz
    rm mobilenet_v1_1.0_160_2017_06_14.tar.gz
fi
if [ ! -e "mobilenet_v1_1.0_128.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_1.0_128_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_1.0_128_2017_06_14.tar.gz
    rm mobilenet_v1_1.0_128_2017_06_14.tar.gz
fi

if [ ! -e "mobilenet_v1_0.75_224.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.75_224_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.75_224_2017_06_14.tar.gz
    rm mobilenet_v1_0.75_224_2017_06_14.tar.gz
fi
if [ ! -e "mobilenet_v1_0.75_192.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.75_192_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.75_192_2017_06_14.tar.gz
    rm mobilenet_v1_0.75_192_2017_06_14.tar.gz
fi
if [ ! -e "mobilenet_v1_0.75_160.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.75_160_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.75_160_2017_06_14.tar.gz
    rm mobilenet_v1_0.75_160_2017_06_14.tar.gz
fi
if [ ! -e "mobilenet_v1_0.75_128.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.75_128_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.75_128_2017_06_14.tar.gz
    rm mobilenet_v1_0.75_128_2017_06_14.tar.gz
fi

if [ ! -e "mobilenet_v1_0.5_224.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.5_224_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.5_224_2017_06_14.tar.gz
    rm mobilenet_v1_0.5_224_2017_06_14.tar.gz
fi
if [ ! -e "mobilenet_v1_0.5_192.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.5_192_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.5_192_2017_06_14.tar.gz
    rm mobilenet_v1_0.5_192_2017_06_14.tar.gz
fi
if [ ! -e "mobilenet_v1_0.5_160.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.5_160_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.5_160_2017_06_14.tar.gz
    rm mobilenet_v1_0.5_160_2017_06_14.tar.gz
fi
if [ ! -e "mobilenet_v1_0.5_128.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.5_128_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.5_128_2017_06_14.tar.gz
    rm mobilenet_v1_0.5_128_2017_06_14.tar.gz
fi

if [ ! -e "mobilenet_v1_0.25_224.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.25_224_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.25_224_2017_06_14.tar.gz
    rm mobilenet_v1_0.25_224_2017_06_14.tar.gz
fi
if [ ! -e "mobilenet_v1_0.25_192.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.25_192_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.25_192_2017_06_14.tar.gz
    rm mobilenet_v1_0.25_192_2017_06_14.tar.gz
fi
if [ ! -e "mobilenet_v1_0.25_160.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.25_160_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.25_160_2017_06_14.tar.gz
    rm mobilenet_v1_0.25_160_2017_06_14.tar.gz
fi
if [ ! -e "mobilenet_v1_0.25_128.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.25_128_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.25_128_2017_06_14.tar.gz
    rm mobilenet_v1_0.25_128_2017_06_14.tar.gz
fi
