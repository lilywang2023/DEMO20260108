# This demo trains an anatomy-aware affine registration network with SynthMorph,
# a strategy for learning image registration with variable synthetic data only.
# 本演示使用 SynthMorph 训练一个解剖结构感知的仿射配准网络。
# SynthMorph 是一种仅利用可变合成数据来学习图像配准的策略。
# Further information is available at https://synthmorph.io.
#
# If you find the demo useful, please cite:
#
#     Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
#     Hoffmann M, Hoopes A, Fischl B*, Dalca AV* (*equal contribution)
#     SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
#     https://doi.org/10.1117/12.2653251
#     https://synthmorph.io/#papers (PDF)
#
# We distribute this notebook under the MIT License:
# https://choosealicense.com/licenses/mit

import numpy as np
import surfa as sf
import neurite as ne
import voxelmorph as vxm
import tensorflow as tf
import matplotlib.pyplot as plt