
# import matplotlib
# matplotlib.use('Agg')


import tqdm
import numpy as np
import neurite as ne
import voxelmorph as vxm
import tensorflow as tf
import matplotlib.pyplot as plt


# 合成一批具有复杂几何结构的 2D 伪标签图（label maps），常用于医学图像配准、分割等任务的数据增强或合成数据生成
# Input shapes.
in_shape = (256,) * 2  # (1,) * 3  # → (1, 1, 1)
num_dim = len(in_shape)
num_label = 16
num_maps = 40

# Shape generation.
label_maps = []  # 创建一个空列表（empty list），用于后续存储生成的标签图（label maps）。
for _ in tqdm.tqdm(range(num_maps)):  # tqdm.tqdm(...)：显示进度条，提升用户体验。
    # Draw image and warp
    im = ne.utils.augment.draw_perlin(
        out_shape=(*in_shape, num_label),
        scales=(32, 64), max_std=1,
    )  # 返回的变量 im 是一个 NumPy 数组（numpy.ndarray），其具体结构由传入的 out_shape 参数决定。out_shape 的将是 (256, 256, 16)
    warp = ne.utils.augment.draw_perlin(
        out_shape=(*in_shape, num_label, num_dim),
        scales=(16, 32, 64), max_std=16,
    )

    # Transform and create label map.
    im = vxm.utils.transform(im, warp)  # 用于将图像 im 根据形变场 warp 进行空间变换（即“重采样”或“warping”）
    lab = tf.argmax(im, axis=-1)  # 对输入张量 im 的每一个位置，找出其在 类别维度（channel dimension） 上最大值所在的索引，作为预测类别。
    label_maps.append(np.uint8(lab))  # 将张量或数组转换为 8 位无符号整型并添加到列表

# Visualize shapes.
num_row = 2
per_row = 10
for i in range(0, num_row * per_row, per_row):
    ne.plot.slices(label_maps[i:i + per_row], cmaps=['tab20c'])

# Training-image generation. For accurate registration, the landscape of warps
# and image contrasts will need to include the target distribution.
prop = dict(  # 使用关键字参数风格构造字典（dictionary）
    in_shape=in_shape,
    # labels_in=range(num_label),
    in_label_list=range(num_label),
    # warp_max=4,
    # warp_std=(0.2, 0.8),
    # warp_blur_min=(8, 8),
    # warp_blur_max=(64, 64),
    # warp_res = (8, 64),
)
model_gen_1 = ne.models.labels_to_image(**prop, id=1)   # 就是“用 prop 字典里的所有合法设置，再加上 id=1 这个特化标识，来创建一个从标签生成逼真医学图像的模型”。
# 返回的 Keras 模型接受一批标签图为输入，并生成相应的一批合成图像和/或标签图，以及可选的形变场信息。
model_gen_2 = ne.models.labels_to_image(**prop, id=2)

# Test repeatedly on the same input.
num_gen = 10
input = np.expand_dims(label_maps[0], axis=(0, -1))
# axis=(0, -1) 表示在两个位置同时插入新维度：
# axis=0：在最前面插入一个维度 → 用于表示 batch 维度
# axis=-1：在最后面插入一个维度 → 通常用于表示 通道维度（channel）
slices = [model_gen_1.predict(input)[0] for _ in range(num_gen)]
# 每次调用 predict() 都会触发模型内部的随机增强机制（如随机形变、强度变化、偏置场等）；
# 即使输入 input 完全相同，每次输出的图像也各不相同；
# 最终 slices 是一个包含 num_gen (=10)个合成图像的列表。

ne.plot.slices(slices)

# Registration model.
# #############################################################################################################################
print("--- start to Register ---")
model_def = vxm.networks.VxmDense(
  inshape=in_shape,
  int_resolution=2,
  svf_resolution=2,
  nb_unet_features=([256] * 4, [256] * 8),
  reg_field='warp',
)

# Moved labels.
ima_1, map_1 = model_gen_1.outputs
ima_2, map_2 = model_gen_2.outputs

_, warp = model_def((ima_1, ima_2))   # ima_1是fixed image，ima_2是moving image
moved = vxm.layers.SpatialTransformer(fill_value=0)((map_1, warp))  # warp是预测的形变场

# Contrast invariance: MSE loss on probability maps.对比不变性：概率图上的均方误差损失
class AddLoss(tf.keras.layers.Layer):
  def call(self, x):
    moved, map_2, warp = x
    # const = tf.zeros((tf.shape(moved)[0], 1))
    # self.add_loss(vxm.losses.MSE().loss(moved, map_2) + const)

    # 先计算损失
    mse_loss = vxm.losses.MSE().loss(moved, map_2)
    grad_loss = vxm.losses.Grad('l2', loss_mult=0.05).loss(None, warp)

    mse_loss_scalar = tf.reduce_mean(mse_loss)
    grad_loss_scalar = tf.reduce_mean(grad_loss)

    # 再打印（仅用于调试！）
    # tf.print("MSE shape:", tf.shape(mse_loss))
    # tf.print("Grad shape:", tf.shape(grad_loss))

    self.add_loss(mse_loss_scalar)
    self.add_loss(grad_loss_scalar)
    return x



# Combined model: synthesis and registration.
inputs = (*model_gen_1.inputs, *model_gen_2.inputs)
model = tf.keras.Model(inputs, AddLoss()((moved, map_2, warp)))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

# Training. Re-running the cell will continue training.
# #############################################################################################################################
print("--- start to Training ---")
hist = model.fit(
  vxm.generators.synthmorph(label_maps, same_subj=True, flip=True),
  epochs=3,
  steps_per_epoch=100,
)

# Visualize loss.
# plt.plot(hist.epoch, hist.history['loss'], '.-')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')

# Skip training, download model weights.
# !gdown -cO weights.h5 19ayDE-otx2kTcGzEhz_pv9MjnJkpX638
model.load_weights('weight0103/weights.h5')

# Resize and normalize test images.
def conform(x, in_shape=in_shape):
  x = np.float32(x) # 将输入 x 转换为 32 位浮点数格式。这一步通常是为了确保数值计算的一致性和效率。
  x = np.squeeze(x) # 使用 np.squeeze 函数来移除数组 x 中所有长度为 1 的维度
  x = ne.utils.minmax_norm(x)   # 对 x 应用最小-最大规范化（min-max normalization）。这通常涉及到线性变换，使得 x 中的所有元素都位于 [0, 1] 区间内
  x = ne.utils.zoom(x, zoom_factor=[o / i for o, i in zip(in_shape, x.shape)])
  # 根据 in_shape 和当前 x 的形状计算缩放因子，并使用这些因子对 x 进行缩放。目的是使 x 的形状与指定的 in_shape 相匹配。zoom_factor 是通过计算目标尺寸 o 与当前尺寸 i 的比率得到的，对于每个维度都是如此。
  return x[None, ..., None]
  # 在返回之前，增加新的维度到 x。None 在索引中用于扩展新轴。这里的表达式在 x 的开头和结尾各添加一个新的维度，从而改变其形状。例如，如果 x 原本是一个二维数组，则经过此操作后它会变成四维数组。

def register(moving, fixed):
  # Conform and register.
  moving = conform(moving)
  fixed = conform(fixed)
  moved, warp = model_def.predict((moving, fixed), verbose=0)

  # Visualize.
  slices = (moving, fixed, moved, warp[..., 0])
  titles = ('Moving', 'Fixed', 'Moved', 'Warp (x-axis)')
  ne.plot.slices(slices, titles, do_colorbars=True)

# Test on MNIST.
print("--- Test on MNIST ---")
images, digits = tf.keras.datasets.mnist.load_data()[-1]
ind = np.flatnonzero(digits == 4)
register(moving=images[ind[6]], fixed=images[ind[9]])

# Test on OASIS-1.
print("--- Test on OASIS-1 ---")
images = ne.py.data.load_dataset('2D-OASIS-TUTORIAL')
register(moving=images[2], fixed=images[7])
