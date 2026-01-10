# imports
import os, sys

# third party imports
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'


import voxelmorph as vxm
import neurite as ne




# 原始 unet
unet = vxm.networks.Unet(inshape=(64, 64), nb_features=[[32, 64], [64, 32]])

# 添加位移头
disp_tensor = tf.keras.layers.Conv2D(2, 3, padding='same')(unet.output)

# 构建新模型
def_model = tf.keras.Model(unet.input, disp_tensor)

# 查看结构
def_model.summary()