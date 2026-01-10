# imports
import os, sys

# third party imports
import numpy as np
import tensorflow as tf

assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'

import voxelmorph as vxm
import neurite as ne

# You should most often have this import together with all other imports at the top,
# but we include here here explicitly to show where data comes from
from tensorflow.keras.datasets import mnist

# load MNIST data.
# `mnist.load_data()` already splits our data into train and test.
(x_train_load, y_train_load), (x_test_load, y_test_load) = mnist.load_data()
# (x_train, y_train),  # 训练集：60,000 张图像 + 标签
# (x_test, y_test)  # 测试集：10,000 张图像 + 标签
# 变量	     形状（shape）	   数据类型	含义
# x_train	(60000, 28, 28)	    uint8	6 万张 28×28 像素的灰度图（像素值 0–255）
# y_train	(60000,)	        uint8	对应的标签（数字 0–9）
# x_test	(10000, 28, 28) 	uint8	1 万张测试图像
# y_test	(10000,)	        uint8	测试标签
# 第 1 步：函数返回一个嵌套元组
# result 的结构是： ( (array_x_train, array_y_train), (array_x_test,  array_y_test) )
# 第 2 步：用嵌套元组进行解包赋值
# (a, b), (c, d) = result

# Data
# ###########################################################################################################################################################################
print("---start to load data---")
digit_sel = 6

# extract only instances of the digit 5
x_train = x_train_load[y_train_load == digit_sel, ...]
y_train = y_train_load[y_train_load == digit_sel]
x_test = x_test_load[y_test_load == digit_sel, ...]
y_test = y_test_load[y_test_load == digit_sel]

# let's get some shapes to understand what we loaded.
print('shape of x_train: {}, y_train: {}'.format(x_train.shape, y_train.shape))

nb_val = 1000  # keep 1,000 subjects for validation
x_val = x_train[-nb_val:, ...]  # this indexing means "the last nb_val entries" of the zeroth axis
y_val = y_train[-nb_val:]
x_train = x_train[:-nb_val, ...]
y_train = y_train[:-nb_val]
print('shape of x_val: {}, y_val: {}'.format(x_val.shape, y_val.shape))
print('shape of x_train: {}, y_train: {}'.format(x_train.shape, y_train.shape))

# Visualize Data
# ###########################################################################################################################################################################
print("---start to visualize data---")
nb_vis = 5

# choose nb_vis sample indexes
idx = np.random.choice(x_train.shape[0], nb_vis, replace=False)
# a：可选值范围。= x_train.shape[0]
# 如果 a 是一个整数（如本例中的 x_train.shape[0]），则表示从 0 到 a-1 的整数中采样。
# 如果 a 是一个数组，则从该数组元素中采样。
# size（此处为 nb_vis）：要采样的样本数量。
# replace=False：表示无放回抽样（即不重复）。

example_digits = [f for f in x_train[idx, ...]]

print('idx=', idx)

# plot
# ne.plot.slices(example_digits, cmaps=['gray'], do_colorbars=True);

# fix data
x_train = x_train.astype('float') / 255
x_val = x_val.astype('float') / 255
x_test = x_test.astype('float') / 255

# verify
print('training maximum value', x_train.max())

# re-visualize
example_digits = [f for f in x_train[idx, ...]]

titles = [f for f in idx]
ne.plot.slices(example_digits, titles=titles, cmaps=['gray'], do_colorbars=True)

# 填充规格
pad_amount = ((0, 0), (2, 2), (2, 2))

# fix data
x_train = np.pad(x_train, pad_amount, 'constant')
x_val = np.pad(x_val, pad_amount, 'constant')
x_test = np.pad(x_test, pad_amount, 'constant')
# np.pad(array, pad_width, mode='constant', constant_values=0)
# 各参数说明：
# array	要填充的输入数组（如 x_train）
# pad_width	指定每个轴（维度）前后要填充多少元素（即 pad_amount）
# mode	填充方式，如 'constant'、'edge'、'reflect' 等
# constant_values（可选）	当 mode='constant' 时，指定填充的常数值，默认为 0

# verify
print('shape of training data', x_train.shape)

# re-visualize
example_digits = [f for f in x_train[idx, ...]]
ne.plot.slices(example_digits, cmaps=['gray'], do_colorbars=True);

# CNN Model
# ################################################################################################################################################################################
# configure unet input shape (concatenation of moving and fixed images)
print("---start to set CNN Model---")
ndim = 2
unet_input_features = 2
# 表示 U-Net 的输入通道数（input channels）为 2。
# 在 图像配准任务中，通常将 两幅图像拼接（concatenate）作为输入：
# 第 1 通道：固定图像（fixed image）
# 第 0 通道：浮动图像（moving image）
# 因此输入是一个 (H, W, 2) 的张量（2D 情况下）。
# 📌 这是 VoxelMorph 典型的“联合输入”策略：把 fixed + moving 图像叠在一起送入网络，预测它们之间的形变场。

inshape = (*x_train.shape[1:], unet_input_features)
# *是解包函数的意思
print('inshape:', inshape)

# configure unet features
nb_features = [
    [32, 32, 32, 32],  # encoder features
    [32, 32, 32, 32, 32, 16]  # decoder features
]

# build model
unet = vxm.networks.Unet(inshape=inshape, nb_features=nb_features)
# 创建了一个专为医学图像配准设计的 U-Net 网络，其作用是：
# 接收一对拼接的 2D/3D 医学图像，输出一个 dense displacement field（密集形变场），用于后续图像 warp。
# inshape：定义图像空间尺寸（如 (128, 128)）
# nb_features：定义网络每层的卷积通道数，控制模型复杂度
# 返回的是一个可直接用于构建完整配准模型的 Keras 子模块
# 这是 VoxelMorph 实现快速、无监督、端到端图像配准的关键组件之一。

print('input shape: ', unet.input.shape)
print('output shape:', unet.output.shape)

# transform the results into a flow field.
disp_tensor = tf.keras.layers.Conv2D(ndim, kernel_size=3, padding='same', name='disp')(unet.output)
# 在 U-Net 的输出（如 (None, 32, 32, 16)）上应用一个 1×1 或 3×3 卷积层，将其通道数压缩为 ndim。
# 输出即为 位移场（displacement field），表示每个像素需要移动的向量。

# 输入：unet.output，假设形状为 (None, 32, 32, 16)  None→ 表示 batch 大小不定，图像 32×32，16 个特征通道。
# 操作：用一个 3×3 卷积 将 16 通道压缩为 ndim 通道。
# 输出：disp_tensor，形状为 (None, 32, 32, ndim)
# 若 ndim=2（2D 图像），则输出每个像素的 (dx, dy) 位移向量
# 若 ndim=3（3D 图像），则输出 (dx, dy, dz)
# 💡 虽然注释说“1×1 或 3×3”，但这里明确用了 kernel_size=3。
# 实际上，3×3 卷积比 1×1 更好：能利用局部上下文信息平滑位移场，避免噪声。

# check tensor shape
print('displacement tensor:', disp_tensor.shape)

# using keras, we can easily form new models via tensor pointers
def_model = tf.keras.models.Model(unet.inputs, disp_tensor)
# 📌 含义：创建一个新的 Keras 模型 def_model
# 输入：unet.inputs（即原始拼接图像，如 (None, 32, 32, 2)）
# 输出：disp_tensor（即位移场 (None, 32, 32, ndim)）
# ✅ 为什么这样做？
# 原始 unet 输出的是中间特征（如 16 通道），而我们真正需要的是位移场。
# 通过 Model(inputs, outputs)，我们可以直接构建从输入图像到位移场的端到端映射。
# 这个 def_model 可以：
# 单独用于推理（预测形变场）
# 作为更大模型（如完整配准模型）的一部分
# 🔗 这体现了 Keras 的核心优势：通过张量指针（tensor pointers）灵活组装模型

# 系统解释
# tf.keras.models.Model(inputs, outputs, name=None)
# inputs：模型的输入张量（或张量列表）。    # 可以是单个 tf.Tensor（如 unet.input），也可以是多个张量组成的列表（如 [input1, input2]）。
# 在本例中，unet.inputs 是一个列表（即使只有一个输入，Keras 通常也以列表形式存储）。
# outputs：模型的输出张量（或张量列表）。   # 表示从输入经过一系列层计算后得到的最终结果。
# 本例中 disp_tensor 是一个 tf.Tensor，由 Conv2D 层作用于 unet.output 得到。
# name（可选）：为模型指定名称，便于调试或可视化。
# ✅ 该构造函数会自动追踪从 inputs 到 outputs 的所有计算路径，构建完整的计算图，并生成一个可训练、可保存、可调用的 tf.keras.Model 实例。

# 查看结构
# def_model.summary()

# Loss
# #############################################################################################################################################################
# build transformer layer
print("---start to set loss---")
spatial_transformer = vxm.layers.SpatialTransformer(name='transformer')
# ✅ 功能：# 创建一个 可微分的空间变换模块，能根据位移场对图像进行形变。
# 这是 VoxelMorph 自定义的 Keras 层，封装了 网格采样（grid sampling） 或 双线性插值（bilinear interpolation） 的实现。
# 关键特性： # 可微分（differentiable）：梯度可以反向传播到位移场，从而端到端训练整个网络。# 支持 2D/3D：自动根据输入张量维度选择实现。
# 输入要求： # [source_image, displacement_field]     # source_image: (B, H, W, 1)（2D）或 (B, D, H, W, 1)（3D）
# displacement_field: (B, H, W, 2) 或 (B, D, H, W, 3)
# 💡 这个层是 VoxelMorph 能实现 无监督配准 的核心技术之一——无需 ground truth 形变，仅靠图像相似性即可训练。

# extract the first frame (i.e. the "moving" image) from unet input tensor
moving_image = tf.expand_dims(unet.input[..., 0], axis=-1)
# 分步解析：
# 1. unet.input
# 这是 Keras 模型的输入张量（tf.Tensor 或 KerasTensor）。
# 形状如 (B, H, W, D, 2)

# 2. unet.input[..., 0]
# ... 表示“所有前面的维度”
# [..., 0] 相当于取最后一维（通道维）的第 0 个通道， 备注：unet.input 是两张图(0:fixed, 1:moving)的拼装。
# 结果形状：(B, H, W, D) ← 少了通道维
# ⚠️ 问题：大多数图像处理层（包括 SpatialTransformer）期望输入有显式的通道维（即使只有 1 个通道）。
#
# 3. tf.expand_dims(..., axis=-1)
# 在最后一个维度（即通道维）上增加一个大小为 1 的新维度
# 输入形状 (B, H, W, D) → 输出形状 (B, H, W, D, 1)
# ✅ 目的：恢复标准的图像张量格式，符合后续层的输入要求。

# warp the moving image with the transformer
moved_image_tensor = spatial_transformer([moving_image, disp_tensor])
# 是在模型构建阶段，将 SpatialTransformer 层“连接”到计算图中，生成一个代表形变结果的输出张量。
print("moved_image_tensor shape", moved_image_tensor.shape) # 注意：输出为一个2D的图像，是由moving image变换得到的Moved image

outputs = [moved_image_tensor, disp_tensor]
# 这里使用了 Python 的列表字面量表示法来创建一个包含两个元素的列表
# 定义模型的两个输出：配准结果 + 形变场

vxm_model = tf.keras.models.Model(inputs=unet.inputs, outputs=outputs)  ## 继续看 12-28
# ✅ 结果： # vxm_model 是一个端到端可训练的模型。
# 调用 vxm_model([fixed, moving]) 会同时返回：
# warped_img, displacement = vxm_model([fixed_batch, moving_batch])
# 之前第一次tf.keras.models.Model拼接实现了从unet.inputs（fixed, moving） 得到一个变形场disp_tensor
# 第二次拼接，实现了从unet.inputs 计算 得到 outputs = [moved_image_tensor, disp_tensor]，即一个变形后的图像，附带之前的变形场。

from tensorflow.keras.utils import plot_model

plot_model(
    vxm_model,
    to_file='vxm_model.png',
    show_shapes=True,      # 显示张量形状
    show_dtype=False,
    show_layer_names=True,
    rankdir='TB',          # 'TB' = top to bottom, 'LR' = left to right
    expand_nested=False,
    dpi=96
)

# build model using VxmDense
inshape = x_train.shape[1:]
print("x_train.shape[1:]", inshape)

vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)
# 这是调用 VoxelMorph库中的 VxmDense 类来构建一个完整的端到端配准网络。
# 🔸 什么是 VxmDense？
# 它是 VoxelMorph 提供的一个预定义模型类，封装了：# 一个 U-Net 编码器-解码器# 一个 可选的微分同胚积分模块（VecInt）# 一个 SpatialTransformer层
# 输入：一对图像 [fixed, moving] # 输出：[warped_moving, displacement_field]
# 🔹 参数详解
# ✅ 1. in-shape
# 类型：tuple，如 (160, 192, 224, 1)（3D）或 (256, 256, 1)（2D）
# 含义：单个输入图像的形状（不含 batch 维）
# 用途：用于初始化 U-Net 的输入层和 SpatialTransformer# 💡 如图像是 (128, 128, 128) 且单通道，inshape = (128, 128, 128, 1)
# ✅ 2. nb_features
# 类型：list of lists，定义 U-Net 每一层的特征图数量# 典型值：# nb_features = [[16, 32, 32, 32],  #encoder: 每层卷积的滤波器数
#     [32, 32, 32, 32, 32, 16, 16]  ] # decoder
# 结构：[encoder_filters, decoder_filters]  # 作用：控制模型容量和感受野
# ✅ 3. int_steps=0
# 含义：微分同胚积分（diffeomorphic integration）关键机制：
# 若 int_steps > 0：U-Net 预测的是速度场（velocity field），然后通过 scaling and squaring 积分得到形变场 φ = exp(v)
# 若 int_steps = 0：U-Net 直接预测位移场（displacement field）u，即 φ(x) = x + u(x)

print('input shape: ', ', '.join([str(t.shape) for t in vxm_model.inputs]))
print('output shape:', ', '.join([str(t.shape) for t in vxm_model.outputs]))

# voxelmorph has a variety of custom loss classes
losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
# 1. vxm.losses.MSE().loss
# 全称：Mean Squared Error（均方误差）
# 2. vxm.losses.Grad('l2').loss
# 全称：Displacement Field Gradient Regularization（位移场梯度正则化）
# 核心思想：惩罚位移场的空间剧烈变化（即鼓励平滑）

# usually, we have to balance the two losses by a hyper-parameter
lambda_param = 0.05
loss_weights = [1, lambda_param]
# 相似性损失（如 MSE）的值可能在 [0, 1] 范围
# 正则化损失（梯度平方和）可能非常大（如 1000）
# 如果直接相加，正则项会主导优化， # 因此引入超参数 λ 控制正则强度：

vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)  ## 12-29 继续
# 总结
# 维度	        说明
# 语法本质	    Keras 模型训练前的配置步骤
# 核心功能	    定义优化器、多任务损失及其权重
# 在配准中的作用	实现 相似性 + 正则化 的无监督学习目标
# 关键约定	    losses[i] 对应 model.outputs[i]
# 典型值	        losses = [MSE, Grad], loss_weights = [1.0, 0.01]
# 💡 一句话概括：compile() 将你的物理先验（平滑形变） 和任务目标（图像对齐） 转化为可优化的数学表达式，是连接模型结构与训练目标的桥梁。

# Train Model
# #############################################################################################################################################################
print("---start to train model---")
def vxm_data_generator(x_data, batch_size=32):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.
    该生成器接收大小为 [N, H, W] 的数据，并生成用于我们自定义 vxm 模型的数据。请注意，我们需要为每个输入和每个输出提供 numpy 数据。

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
    vol_shape = x_data.shape[1:]  # extract data shape
    ndims = len(vol_shape)
    # 提取图像形状和维度:    # vol_shape：单个图像的空间尺寸    # ndims：判断是 2D 还是3D数据（决定位移场通道数）

    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    # 2.    创建零位移场模板
    # 形状示例（3D）：(32, 128, 128, 128, 3)
    # 用途：作为模型第二个输出（位移场）的“假标签”
    # 为什么需要？
    # Keras要求多输出模型的outputs必须与loss列表长度一致
    # 虽然Grad损失不使用y_true，但仍需提供一个同形状的张量占位
    # 💡 这是一个接口兼容性设计，实际训练中zero_phi的值无关紧要。

    while True:
        # 使生成器可被model.fit(steps_per_epoch=...)无限调用
        # 避免epoch结束时停止（适合无监督自监督任务）
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        # x_data.shape[0] 表示数据集中的样本总数（即 batch 维度大小）
        # np.random.randint(0, N, size=k) 生成一个长度为 k 的一维 NumPy 数组 每个元素是 [0, N) 范围内的随机整数 允许重复（即有放回抽样）
        # 若需“无放回”，可用 np.random.choice(..., replace=False)，但要求 batch_size ≤ N

        moving_images = x_data[idx1, ..., np.newaxis]
        # 这行代码的作用是从x_data中根据索引idx1提取出一批样本，并通过np.newaxis增加一个新的维度。
        # 具体来说，如果原始x_data的形状是(N, H, W)，那么提取出的moving_images形状将会是(batch_size, H, W, 1)。这个新添加的维度通常用于表示通道数，在图像处理中特别有用。

        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = x_data[idx2, ..., np.newaxis]

        inputs = [moving_images, fixed_images]

        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare
        # the resulting moved image with the fixed image.
        # we also wish to penalize the deformation field.
        outputs = [fixed_images, zero_phi]

        yield (inputs, outputs)


# let's test it
train_generator = vxm_data_generator(x_train)
# vxm_data_generator是什么？：
# 你之前定义的生成器函数（generator function）
# x_train：训练数据，通常是一个 NumPy 数组，形状如 (N, H, W) 或 (N, H, W, D)，表示 N 个医学图像
# 调用时未指定 batch_size，使用默认值 32

# train_generator 是什么？
# 它是一个 惰性迭代器（lazy iterator）
# 内部保存了：# 对 x_train 的引用；# 默认 batch_size=32；# 函数执行的“挂起点”（初始为函数开头）
# 如何获取数据？ 通过迭代或 next()：

in_sample, out_sample = next(train_generator)
# next(train_generator)	从生成器获取下一个 (inputs, outputs) 元组
# in_sample, out_sample = ...	解包为输入和输出两部分，便于单独操作
# 目的	调试、可视化、自定义训练等需要显式访问单个 batch 的场景
# 前提	生成器必须按 Keras 约定 yield (inputs, outputs)

# visualize
comb = in_sample + out_sample

images = [img[0, :, :, 0] for img in comb]
# images是一个列表，用于从comb中存储的4个向量中的每一个中提取一个元素：img[0, :, :, 0]，这又是一个基本索引操作（整数+切片），结果是第一个样本的第一个通道切片。

titles = ['moving', 'fixed', 'moved ground-truth (fixed)', 'zeros']
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)
# 调用 ne.plot.slices 来绘制图像切片：
# images：一个包含四张 2D 或 3D 图像的列表或数组（如 [img1, img2, img3, img4]），顺序必须与 titles 对应。
# titles=titles：为每张图像设置对应的标题。
# cmaps=['gray']：指定所有图像使用灰度 colormap（医学图像常用）。注意这里虽然是列表形式，但只传了一个 'gray'，函数内部会自动广播到所有子图。
# do_colorbars=True：在每张子图旁边显示颜色条（colorbar），用于指示像素值范围。

nb_epochs = 3
steps_per_epoch = 100
hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2)
# 这是 Keras（或 TensorFlow < 2.1）中用于训练模型的函数（在 TF ≥ 2.1 后推荐使用 model.fit()，但 fit_generator 在旧版 VoxelMorph 中仍常见）。
# # 参数详解：
# 1-train_generator：一个 Python 生成器（generator），每次调用 next() 会返回一个 batch 的训练数据。
# 对于 VoxelMorph，通常返回格式为： ([moving_batch, fixed_batch], [fixed_batch, zero_disp_field])
# 其中：     输入是 (moving, fixed) 图像对；#输出包括重建图像（应接近 fixed）和形变场（监督信号可能为零，若使用无监督配准）。
# 2-epochs=nb_epochs：# 训练 10 个 epoch。
# 3-steps_per_epoch=steps_per_epoch：# 每个 epoch 跑 100 步。
# 4-verbose=2：# 控制训练日志输出的详细程度：# 0：静默；# 1：进度条（实时更新）；# 2：每个 epoch 结束后打印一行摘要（如 loss 值），适合日志记录。
# 5-hist：# 返回一个 History 对象，包含训练过程中每个 epoch 的 loss 和 metrics（如 loss, dec_loss, grad_loss 等），可用于后续绘图或分析。

import matplotlib.pyplot as plt


def plot_history(hist, loss_name='loss'):
    # Simple function to plot training history.
    plt.figure()
    plt.plot(hist.epoch, hist.history[loss_name], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

plot_history(hist)

# Registration
# #############################################################################################################################################################
# let's get some data
print("---start Registration---")
val_generator = vxm_data_generator(x_val, batch_size=1)
# 作用：创建一个每次返回 1 对验证图像的生成器

val_input, _ = next(val_generator)
# 作用：从生成器中取出一个 batch 的数据，只保留输入部分（inputs），忽略目标（targets）

import time

start = time.time()

val_pred = vxm_model.predict(val_input)
# 将 val_input 输入到已训练的 VoxelMorph 网络中，网络前向传播，生成两个输出：
# 配准后的图像（warped moving image） 形变场（deformation field / flow）# 返回结果并赋值给 val_pred

# %timeit is a 'jupyter magic' that times the given line over several runs
# %timeit vxm_model.predict(val_input)

elapsed = time.time() - start
print(f"Inference time: {elapsed:.2f} s")

print("Type of val_input:", type(val_input), "Shape of val_input:", val_input[0].shape)
print("Type of val_pred:", type(val_pred), "Shape of val_pred:", val_pred[0].shape)

# visualize
images = [img[0, :, :, 0] for img in val_input + list(val_pred)]
# inputs = [moving_images, fixed_images]，    outputs = [wrapped_images, flow]

titles = ['moving', 'fixed', 'moved', 'flow']  # 设置图像标题
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)
# ne 是 neurite 库（VoxelMorph 官方配套可视化工具）
# ne.plot.slices()：将多个 2D 图像并排显示
# 参数说明：
# cmaps=['gray']：所有图像用灰度 colormap（适合 MRI/US）
# do_colorbars=True：显示颜色条（便于观察 intensity 范围）
# ✅ 输出效果：一行 4 张图，标注清晰，带色标。

# 直接打印形状
print(val_pred[1].shape)
ne.plot.flow([val_pred[1].squeeze()], width=5)

# Generalization 泛化
# #############################################################################################################################################################
# extract only instances of the digit 7
print("---start Generalization---")
x_sevens = x_train_load[y_train_load == 7, ...].astype('float') / 255
# y_train_load == 7：布尔索引，返回一个与标签长度相同的布尔数组，值为 True 的位置对应标签为 7 的样本。
# x_train_load[...]：用该布尔数组索引 x_train_load，选出所有标签为 7 的图像。
# 假设 x_train_load shape 是 (N, H, W) 或 (N, H, W, 1)（MNIST 通常是 (60000, 28, 28)）
# 结果 x_sevens shape 可能是 (M, 28, 28)，其中 M 是数字 7 的样本数（约 6000+）
# .astype('float')：将整型像素值（0–255）转为浮点型，便于后续除法。
# / 255：归一化到 [0, 1] 范围，这是深度学习模型的标准输入要求。

x_sevens = np.pad(x_sevens, pad_amount, 'constant')
# pad_amount = ((0, 0), (2, 2), (2, 2))
# 该代码使用 NumPy 的 pad 函数，在数组 x_sevens 的边缘填充零值（zero-padding），
# 目的是将图像尺寸从 (N, 28, 28) 扩展为 (N, 32, 32)，以满足深度学习模型（如 VoxelMorph、U-Net）对输入尺寸为 2 的幂次的要求。

# predict
seven_generator = vxm_data_generator(x_sevens, batch_size=1)
# 创建一个数据生成器（generator），用于从数字“7”的图像集合 x_sevens 中动态地、成对地采样图像，以供 VoxelMorph 等无监督图像配准模型训练或推理使用。注意已指定atch_size=1


seven_sample, _ = next(seven_generator)
# 作用是从一个数据生成器（generator） seven_generator 中获取下一个 batch 的数据，并将其解包为输入和输出两部分，其中只保留输入部分 seven_sample，而忽略输出部分（用 _ 表示丢弃）。

seven_pred = vxm_model.predict(seven_sample)
# 使用已训练好的 VoxelMorph 模型对一对图像（moving 和 fixed）进行推理（inference），以获得配准结果（即形变后的图像和形变场）


# visualize
images = [img[0, :, :, 0] for img in seven_sample + list(seven_pred)]
titles = ['moving', 'fixed', 'moved', 'flow']
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)
# cmaps=['gray']：所有图像用灰度 colormap（适合 MRI/US）
# do_colorbars=True：显示颜色条（便于观察 intensity 范围）


# 我们来试试另一种变化。如果我们只修改（原始）数据集，但将像素强度乘以一个系数，结果会怎样？
factor = 5

print("shape of val_input = ", val_input[0].shape)
# val_input来源： val_generator = vxm_data_generator(x_val, batch_size=1)  val_input, _ = next(val_generator)
moving_image = val_input[0]
fixed_image = val_input[1]

print('moving image pixel range: ', moving_image.min(), moving_image.max())
print('fixed image pixel range: ', fixed_image.min(), fixed_image.max())


# 对输入图像进行放大
scaled_moving_image = moving_image * factor
scaled_fixed_image = fixed_image * factor

print('scaled_moving image pixel range: ', scaled_moving_image.min(), scaled_moving_image.max())
print('scaled_fixed image pixel range: ', scaled_fixed_image.min(), scaled_fixed_image.max())

val_pred = vxm_model.predict([f * factor for f in val_input])
# val_input 中的每个元素（即 moving 和 fixed 图像）都被乘以了 factor=5。传递给模型的是已经被放大的图像。

# 对输入图像进行放大
scaled_val_input = [f * factor for f in val_input]

# visualizeb
images = [img[0, :, :, 0] for img in scaled_val_input + list(val_pred)]
# val_input 是未被放大的原始图像，而 list(val_pred) 包含了由放大后的图像生成的 moved 图像和 flow 场。

titles = ['moving', 'fixed', 'moved', 'flow']
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

# Registration of Brain MRI
# #############################################################################################################################################################
print("start registration of Brain MRI")
# 现在我们将配准一些更接近真实情况的数据——大脑 MRI 图像。为了便于在本教程中进行训练和配准，我们将首先提取大脑扫描图像的中间切片。
# 请注意，由于此任务无法捕捉第三维度的形变，因此某些对应关系无法完全对应。尽管如此，此练习仍将演示如何使用更接近真实情况的复杂图像进行配准。
# 大脑已经过强度归一化仿射对齐，并使用 FreeSurfer 去除颅骨，以便能够专注于可变形配准。
# download MRI tutorial data
# !wget https://surfer.nmr.mgh.harvard.edu/pub/data/voxelmorph/tutorial_data.tar.gz -O data.tar.gz
# !tar -xzvf data.tar.gz

import os
import urllib.request
import tarfile

# 1. 设置下载 URL 和本地文件名
url = "https://surfer.nmr.mgh.harvard.edu/pub/data/voxelmorph/tutorial_data.tar.gz"
local_filename = "data.tar.gz"

# 2. 如果文件不存在，则下载
if not os.path.exists(local_filename):
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, local_filename)
    print(f"Downloaded to {local_filename}")
else:
    print(f"{local_filename} already exists. Skipping download.")

# 3. 解压 tar.gz 文件
extract_to = "."  # 当前目录，可改为 "data/" 等
if not os.path.exists("tutorial_data"):
    print(f"Extracting {local_filename} to {extract_to} ...")
    with tarfile.open(local_filename, "r:gz") as tar:
        tar.extractall(path=extract_to)
    print("Extraction completed.")
else:
    print("tutorial_data/ already exists. Skipping extraction.")

