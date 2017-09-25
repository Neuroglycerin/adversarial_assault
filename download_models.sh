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

mkdir -p adv_inception_v3
if [ ! -e "adv_inception_v3/adv_inception_v3.ckpt.data-00000-of-00001" ]; then
    # Download ensemble adversarially trained inception resnet v2 checkpoint
    wget http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz
    tar -xvzf adv_inception_v3_2017_08_18.tar.gz -C adv_inception_v3
    rm adv_inception_v3_2017_08_18.tar.gz
fi
mkdir -p ens_adv_inception_resnet_v2
if [ ! -e "ens_adv_inception_resnet_v2/ens_adv_inception_resnet_v2.ckpt.data-00000-of-00001" ]; then
    # Download ensemble adversarially trained inception resnet v2 checkpoint
    wget http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz
    tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz -C ens_adv_inception_resnet_v2
    rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz
fi


###############################################################################
# Models from Keras
###############################################################################
mkdir -p xception
if [ ! -e "xception/xception_weights_tf_dim_ordering_tf_kernels.h5" ]; then
    wget https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5
    mv xception_weights_tf_dim_ordering_tf_kernels.h5 xception
fi


###############################################################################
# Models from tf-slim
###############################################################################

#------------------------------------------------------------------------------
# Inception
#------------------------------------------------------------------------------
mkdir -p inception_v1
if [ ! -e "inception_v1/inception_v1.ckpt" ]; then
    wget http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz
    tar -xvzf inception_v1_2016_08_28.tar.gz -C inception_v1
    rm inception_v1_2016_08_28.tar.gz
fi
mkdir -p inception_v2
if [ ! -e "inception_v2/inception_v2.ckpt" ]; then
    wget http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz
    tar -xvzf inception_v2_2016_08_28.tar.gz -C inception_v2
    rm inception_v2_2016_08_28.tar.gz
fi
mkdir -p inception_v3
if [ ! -e "inception_v3/inception_v3.ckpt" ]; then
    wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
    tar -xvzf inception_v3_2016_08_28.tar.gz -C inception_v3
    rm inception_v3_2016_08_28.tar.gz
fi
mkdir -p inception_v4
if [ ! -e "inception_v4/inception_v4.ckpt" ]; then
    wget http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
    tar -xvzf inception_v4_2016_09_09.tar.gz -C inception_v4
    rm inception_v4_2016_09_09.tar.gz
fi

#------------------------------------------------------------------------------
# Inception ResNet
#------------------------------------------------------------------------------
mkdir -p inception_resnet_v2
if [ ! -e "inception_resnet_v2/inception_resnet_v2_2016_08_30.ckpt" ]; then
    wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
    tar -xvzf inception_resnet_v2_2016_08_30.tar.gz -C inception_resnet_v2
    rm inception_resnet_v2_2016_08_30.tar.gz
fi

#------------------------------------------------------------------------------
# ResNet
#------------------------------------------------------------------------------
mkdir -p resnet_v1_50
if [ ! -e "resnet_v1_50/resnet_v1_50.ckpt" ]; then
    wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
    tar -xvzf resnet_v1_50_2016_08_28.tar.gz -C resnet_v1_50
    rm resnet_v1_50_2016_08_28.tar.gz
fi
mkdir -p resnet_v1_101
if [ ! -e "resnet_v1_101/resnet_v1_101.ckpt" ]; then
    wget http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
    tar -xvzf resnet_v1_101_2016_08_28.tar.gz -C resnet_v1_101
    rm resnet_v1_101_2016_08_28.tar.gz
fi
mkdir -p resnet_v1_152
if [ ! -e "resnet_v1_152/resnet_v1_152.ckpt" ]; then
    wget http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz
    tar -xvzf resnet_v1_152_2016_08_28.tar.gz -C resnet_v1_152
    rm resnet_v1_152_2016_08_28.tar.gz
fi
mkdir -p resnet_v2_50
if [ ! -e "resnet_v2_50/resnet_v2_50.ckpt" ]; then
    wget http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz
    tar -xvzf resnet_v2_50_2017_04_14.tar.gz -C resnet_v2_50
    rm resnet_v2_50_2017_04_14.tar.gz
fi
mkdir -p resnet_v2_101
if [ ! -e "resnet_v2_101/resnet_v2_101.ckpt" ]; then
    wget http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz
    tar -xvzf resnet_v2_101_2017_04_14.tar.gz -C resnet_v2_101
    rm resnet_v2_101_2017_04_14.tar.gz
fi
mkdir -p resnet_v2_152
if [ ! -e "resnet_v2_152/resnet_v2_152.ckpt" ]; then
    wget http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz
    tar -xvzf resnet_v2_152_2017_04_14.tar.gz -C resnet_v2_152
    rm resnet_v2_152_2017_04_14.tar.gz
fi

#------------------------------------------------------------------------------
# VGG
#------------------------------------------------------------------------------
mkdir -p vgg_16
if [ ! -e "vgg_16/vgg_16.ckpt" ]; then
    wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
    tar -xvzf vgg_16_2016_08_28.tar.gz -C vgg_16
    rm vgg_16_2016_08_28.tar.gz
fi
mkdir -p vgg_19
if [ ! -e "vgg_19/vgg_19.ckpt" ]; then
    wget http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz
    tar -xvzf vgg_19_2016_08_28.tar.gz -C vgg_19
    rm vgg_19_2016_08_28.tar.gz
fi

#------------------------------------------------------------------------------
# MobileNet
#------------------------------------------------------------------------------
mkdir -p mobilenet_v1_100_224
if [ ! -e "mobilenet_v1_100_224/mobilenet_v1_1.0_224.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_1.0_224_2017_06_14.tar.gz -C mobilenet_v1_100_224
    rm mobilenet_v1_1.0_224_2017_06_14.tar.gz
fi
mkdir -p mobilenet_v1_100_192
if [ ! -e "mobilenet_v1_100_192/mobilenet_v1_1.0_192.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_1.0_192_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_1.0_192_2017_06_14.tar.gz -C mobilenet_v1_100_192
    rm mobilenet_v1_1.0_192_2017_06_14.tar.gz
fi
mkdir -p mobilenet_v1_100_160
if [ ! -e "mobilenet_v1_100_160/mobilenet_v1_1.0_160.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_1.0_160_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_1.0_160_2017_06_14.tar.gz -C mobilenet_v1_100_160
    rm mobilenet_v1_1.0_160_2017_06_14.tar.gz
fi
mkdir -p mobilenet_v1_100_128
if [ ! -e "mobilenet_v1_100_128/mobilenet_v1_1.0_128.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_1.0_128_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_1.0_128_2017_06_14.tar.gz -C mobilenet_v1_100_128
    rm mobilenet_v1_1.0_128_2017_06_14.tar.gz
fi

mkdir -p mobilenet_v1_075_224
if [ ! -e "mobilenet_v1_075_224/mobilenet_v1_0.75_224.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.75_224_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.75_224_2017_06_14.tar.gz -C mobilenet_v1_075_224
    rm mobilenet_v1_0.75_224_2017_06_14.tar.gz
fi
mkdir -p mobilenet_v1_075_192
if [ ! -e "mobilenet_v1_075_192/mobilenet_v1_0.75_192.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.75_192_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.75_192_2017_06_14.tar.gz -C mobilenet_v1_075_192
    rm mobilenet_v1_0.75_192_2017_06_14.tar.gz
fi
mkdir -p mobilenet_v1_075_160
if [ ! -e "mobilenet_v1_075_160/mobilenet_v1_0.75_160.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.75_160_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.75_160_2017_06_14.tar.gz -C mobilenet_v1_075_160
    rm mobilenet_v1_0.75_160_2017_06_14.tar.gz
fi
mkdir -p mobilenet_v1_075_128
if [ ! -e "mobilenet_v1_075_128/mobilenet_v1_0.75_128.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.75_128_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.75_128_2017_06_14.tar.gz -C mobilenet_v1_075_128
    rm mobilenet_v1_0.75_128_2017_06_14.tar.gz
fi

mkdir -p mobilenet_v1_050_224
if [ ! -e "mobilenet_v1_050_224/mobilenet_v1_0.50_224.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.50_224_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.50_224_2017_06_14.tar.gz -C mobilenet_v1_050_224
    rm mobilenet_v1_0.50_224_2017_06_14.tar.gz
fi
mkdir -p mobilenet_v1_050_192
if [ ! -e "mobilenet_v1_050_192/mobilenet_v1_0.50_192.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.50_192_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.50_192_2017_06_14.tar.gz -C mobilenet_v1_050_192
    rm mobilenet_v1_0.50_192_2017_06_14.tar.gz
fi
mkdir -p mobilenet_v1_050_160
if [ ! -e "mobilenet_v1_050_160/mobilenet_v1_0.50_160.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.50_160_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.50_160_2017_06_14.tar.gz -C mobilenet_v1_050_160
    rm mobilenet_v1_0.50_160_2017_06_14.tar.gz
fi
mkdir -p mobilenet_v1_050_128
if [ ! -e "mobilenet_v1_050_128/mobilenet_v1_0.50_128.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.50_128_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.50_128_2017_06_14.tar.gz -C mobilenet_v1_050_128
    rm mobilenet_v1_0.50_128_2017_06_14.tar.gz
fi

mkdir -p mobilenet_v1_025_224
if [ ! -e "mobilenet_v1_025_224/mobilenet_v1_0.25_224.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.25_224_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.25_224_2017_06_14.tar.gz -C mobilenet_v1_025_224
    rm mobilenet_v1_0.25_224_2017_06_14.tar.gz
fi
mkdir -p mobilenet_v1_025_192
if [ ! -e "mobilenet_v1_025_192/mobilenet_v1_0.25_192.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.25_192_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.25_192_2017_06_14.tar.gz -C mobilenet_v1_025_192
    rm mobilenet_v1_0.25_192_2017_06_14.tar.gz
fi
mkdir -p mobilenet_v1_025_160
if [ ! -e "mobilenet_v1_025_160/mobilenet_v1_0.25_160.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.25_160_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.25_160_2017_06_14.tar.gz -C mobilenet_v1_025_160
    rm mobilenet_v1_0.25_160_2017_06_14.tar.gz
fi
mkdir -p mobilenet_v1_025_128
if [ ! -e "mobilenet_v1_025_128/mobilenet_v1_0.25_128.ckpt.data-00000-of-00001" ]; then
    wget http://download.tensorflow.org/models/mobilenet_v1_0.25_128_2017_06_14.tar.gz
    tar -xvzf mobilenet_v1_0.25_128_2017_06_14.tar.gz -C mobilenet_v1_025_128
    rm mobilenet_v1_0.25_128_2017_06_14.tar.gz
fi
