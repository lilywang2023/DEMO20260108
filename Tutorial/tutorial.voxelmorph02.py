import os
import urllib.request
import tarfile

# imports
import sys

# third party imports
import numpy as np
import tensorflow as tf

assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'

import voxelmorph as vxm
import neurite as ne

# add from 01
# configure unet features
nb_features = [
    [32, 32, 32, 32],  # encoder features
    [32, 32, 32, 32, 32, 16]  # decoder features
]


def vxm_data_generator(x_data, batch_size=32):
    vol_shape = x_data.shape[1:]  # extract data shape
    ndims = len(vol_shape)
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    while True:
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = x_data[idx1, ..., np.newaxis]

        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = x_data[idx2, ..., np.newaxis]
        inputs = [moving_images, fixed_images]
        outputs = [fixed_images, zero_phi]

        yield (inputs, outputs)


# Registration of Brain MRI
# #############################################################################################################################################################
print("---start registration of Brain MRI---")
# 现在我们将配准一些更接近真实情况的数据——大脑 MRI 图像。为了便于在本教程中进行训练和配准，我们将首先提取大脑扫描图像的中间切片。
# 请注意，由于此任务无法捕捉第三维度的形变，因此某些对应关系无法完全对应。尽管如此，此练习仍将演示如何使用更接近真实情况的复杂图像进行配准。
# 大脑已经过强度归一化仿射对齐，并使用 FreeSurfer 去除颅骨，以便能够专注于可变形配准。
# download MRI tutorial data
# !wget https://surfer.nmr.mgh.harvard.edu/pub/data/voxelmorph/tutorial_data.tar.gz -O data.tar.gz
# !tar -xzvf data.tar.gz


# 1. 设置下载 URL 和本地文件名
url = "https://surfer.nmr.mgh.harvard.edu/pub/data/voxelmorph/tutorial_data.tar.gz"
local_filename = "data.tar.gz"

# 2. 如果文件不存在，则下载
if not os.path.exists(local_filename):
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, local_filename)
    # urllib.request.urlretrieve(url, filename)
    # 调用标准库    urllib.request    模块中的    urlretrieve    函数。
    # 功能：           同步下载整个文件，并保存为本地文件。
    # 参数：           url：要下载的资源地址（str或Request对象）。
    # filename（可选）：本地保存路径；若未提供，则返回临时文件路径。

    print(f"Downloaded to {local_filename}")
else:
    print(f"{local_filename} already exists. Skipping download.")

# 3. 解压 tar.gz 文件
extract_to = "."  # 当前目录，可改为 "data/" 等
if not os.path.exists("tutorial_data.npz"):  # 如果不存在tutorial_data文件，则解压。
    print(f"Extracting {local_filename} to {extract_to} ...")
    with tarfile.open(local_filename, "r:gz") as tar:
        tar.extractall(path=extract_to)
        # tarfile.open(..., "r:gz")  以“读取 + gzip   解压”模式打开.tar.gz        文件
        # with ... as tar:    使用上下文管理器，确保文件正确关闭
        # tar.extractall(path=extract_to)     将所有内容解压到指定目录

    print("Extraction completed.")
else:
    print("tutorial_data/ already exists. Skipping extraction.")

npz = np.load('tutorial_data.npz')
# np	                NumPy 模块的别名（需先 import numpy as np）
# .load()	            NumPy 提供的顶级函数，用于从磁盘加载数组数据
# 'tutorial_data.npz'	位置参数（positional argument），指定要加载的文件路径（字符串）
# npz 不是普通数组，而是一个类似字典的只读容器，具有以下关键属性：
print("npz keys:", npz.keys(), "\nnpz files:", npz.files)

x_train = npz['train']
x_val = npz['validate']

# the 208 volumes are of size 160x192
vol_shape = x_train.shape[1:]
print('train shape:', x_train.shape)

# extract some brains
nb_vis = 5
idx = np.random.randint(0, x_train.shape[0], [5, ])

print('idx', idx)

example_digits = [f for f in x_train[idx, ...]]
# example_digits = x_train[idx, ...]

# visualize
ne.plot.slices(example_digits, cmaps=['gray'], do_colorbars=True)

# Model
# #############################################################################################################################################################
print("--- start to set model for Brain MRI ---")

# unet
vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0)

# losses and loss weights
losses = ['mse', vxm.losses.Grad('l2').loss]
loss_weights = [1, 0.01]

vxm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=losses, loss_weights=loss_weights)

train_generator = vxm_data_generator(x_train, batch_size=8)
in_sample, out_sample = next(train_generator)

# visualize
images = [img[0, :, :, 0] for img in in_sample + out_sample]
titles = ['moving', 'fixed', 'moved ground-truth (fixed)', 'zeros']
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

# hist = vxm_model.fit_generator(train_generator, epochs=4, steps_per_epoch=5, verbose=2)
hist = vxm_model.fit(train_generator, epochs=4, steps_per_epoch=5, verbose=2)

# load pretrained model weights
vxm_model.load_weights('brain_2d_smooth.h5')

# create the validation data generator
val_generator = vxm_data_generator(x_val, batch_size=1)
val_input, _ = next(val_generator)

# prediction
val_pred = vxm_model.predict(val_input)

# visualize registration
images = [img[0, :, :, 0] for img in val_input + list(val_pred)]
titles = ['moving', 'fixed', 'moved', 'flow']
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

# visualize flow
flow = val_pred[1].squeeze()[::3, ::3]
ne.plot.flow([flow], width=5)

# Evaluation
# #############################################################################################################################################################
print("--- start to Evaluation ---")

# prediction from model with MSE + smoothness loss
vxm_model.load_weights('brain_2d_smooth.h5')
our_val_pred = vxm_model.predict(val_input)

# prediction from model with just MSE loss
vxm_model.load_weights('brain_2d_no_smooth.h5')
mse_val_pred = vxm_model.predict(val_input)

# visualize MSE + smoothness model output
images = [img[0, ..., 0] for img in [val_input[1], *our_val_pred]]
titles = ['fixed', 'MSE + smoothness', 'flow']
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

# visualize MSE model output
images = [img[0, ..., 0] for img in [val_input[1], *mse_val_pred]]
titles = ['fixed', 'MSE only', 'flow']
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

ne.plot.flow([img[1].squeeze()[::3, ::3] for img in [our_val_pred, mse_val_pred]], width=10)

# 3D MRI brain scan registration
# #############################################################################################################################################################
print("--- start to 3D MRI brain scan registrate---")

# Model
# #############################################################################################################################################################
print("--- start to set model---")
# our data will be of shape 160 x 192 x 224
vol_shape = (160, 192, 224)
nb_features = [
    [16, 32, 32, 32],
    [32, 32, 32, 32, 32, 16, 16]
]

# build vxm network
vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0)

# Validation data
# #############################################################################################################################################################
print("--- start to prepare Validation data---")

subj1 = np.load('subj1.npz')
subj2 = np.load('subj2.npz')

print("subj1 keys:", subj1.keys(), "\nsubj2 keys:", subj2.keys())

val_volume_1 = np.load('subj1.npz')['vol']
seg_volume_1 = np.load('subj1.npz')['seg']
val_volume_2 = np.load('subj2.npz')['vol']
seg_volume_2 = np.load('subj2.npz')['seg']

print(type(val_volume_1))

if isinstance(val_volume_1, np.ndarray):
    print("✅ val_volume_1 is a NumPy array.")
    print(val_volume_1.shape)
    print(val_volume_1.dtype)
else:
    print("❌ Not a NumPy array!")

val_input = [
    val_volume_1[np.newaxis, ..., np.newaxis],
    val_volume_2[np.newaxis, ..., np.newaxis]
]

vxm_model.load_weights('brain_3d.h5')

# Now let's register.
# #############################################################################################################################################################
print("--- start to Registration---")
val_pred = vxm_model.predict(val_input)   # 使用前面下载的已训练模型 brain_3d.h5来进行预测，输入为val_input=[moving, fixed]

moved_pred = val_pred[0].squeeze()  # .squeeze()	NumPy 去冗余维度方法，常用于去除 batch/channel 维
pred_warp = val_pred[1]

mid_slices_fixed = [np.take(val_volume_2, vol_shape[d] // 2, axis=d) for d in range(3)]
# mid_slices_fixed 是一个包含 3 个 2D NumPy 数组的列表，分别代表三个正交中心切片。

mid_slices_fixed[1] = np.rot90(mid_slices_fixed[1], 1)
mid_slices_fixed[2] = np.rot90(mid_slices_fixed[2], -1)

mid_slices_pred = [np.take(moved_pred, vol_shape[d] // 2, axis=d) for d in range(3)]
mid_slices_pred[1] = np.rot90(mid_slices_pred[1], 1)
mid_slices_pred[2] = np.rot90(mid_slices_pred[2], -1)
ne.plot.slices(mid_slices_fixed + mid_slices_pred, cmaps=['gray'], do_colorbars=True, grid=[2, 3])

# Segmentation
# #############################################################################################################################################################
print("--- start to Segmentation---")
warp_model = vxm.networks.Transform(vol_shape, interp_method='nearest')

warped_seg = warp_model.predict([seg_volume_1[np.newaxis,...,np.newaxis], pred_warp])

from pystrum.pytools.plot import jitter
import matplotlib

[ccmap, scrambled_cmap] = jitter(255, nargout=2) #

print("scrambled_cmp[3]=", scrambled_cmap[3])
print("scrambled_cmp[0]=", scrambled_cmap[0])

print("scramble_cmpa.shape=", scrambled_cmap.shape)


scrambled_cmap[0, :] = np.array([0, 0, 0, 1])               # 将颜色映射表（colormap）中第 0 类的颜色强制设为纯黑色（不透明）。
ccmap = matplotlib.colors.ListedColormap(scrambled_cmap)    # 将一个自定义的离散颜色数组转换为 Matplotlib 可识别的颜色映射对象（colormap）

mid_slices_fixed = [np.take(seg_volume_1, vol_shape[d]//1.8, axis=d) for d in range(3)]
# 列表推导式（list comprehension），用于从一个三维医学图像（或分割体积）中沿三个正交方向（轴向、冠状、矢状）提取“近似中间”的二维切片

mid_slices_fixed[1] = np.rot90(mid_slices_fixed[1], 1)
mid_slices_fixed[2] = np.rot90(mid_slices_fixed[2], -1)

mid_slices_pred = [np.take(warped_seg.squeeze(), vol_shape[d]//1.8, axis=d) for d in range(3)]
mid_slices_pred[1] = np.rot90(mid_slices_pred[1], 1)
mid_slices_pred[2] = np.rot90(mid_slices_pred[2], -1)

slices = mid_slices_fixed + mid_slices_pred
print('slices.shape=', slices[0].shape)
print("length of slices=", len(slices))

for si, slc  in enumerate(slices):
    slices[si][0] = 255
ne.plot.slices(slices, cmaps = [ccmap], grid=[2,3])
