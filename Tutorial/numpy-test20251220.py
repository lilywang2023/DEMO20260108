import numpy as np

x = np.arange(32).reshape((2, 4, 2, 2))

print("原数组为：", x)

print("shape of x = ", x.shape)

y = x[0, :, :, 0]

print("y = ", y)
print("shape of y = ", y.shape)

arr = np.array([[1, 2, 3],
                [4, 5, 6]])
print("arr[0, 1] = ", arr[0, 1])  # → 2（等价于 arr[0][1]，但更高效）
print("arr[-1, -1] = ", arr[-1, -1])  # → 6

print("arr[0, :] = ", arr[0, :])  # 第0行 → array([1, 2, 3])
print("arr[:, 1] = ", arr[:, 1])  # 第1列 → array([2, 5])
print("arr[0:2, 1:3] = ", arr[0:2, 1:3])  # 子矩阵 → array([[2, 3], [5, 6]])


import numpy as np
arr = np.arange(10)  # [0,1,2,...,9]

# 切片对象 → 基本索引
sub1 = arr[1:5]          # slice(1,5,None)
print(type(sub1))        # <class 'numpy.ndarray'>
# sub1 是原数组的视图（view）

# 非切片对象（列表）→ 高级索引
sub2 = arr[[1, 2, 3, 4]] # list → 非切片对象
print(type(sub2))        # <class 'numpy.ndarray'>
# sub2 是副本（copy）


import numpy as np
arr = np.array([1, 2, 3, 4])

# 切片 → 视图
sub = arr[1:3] # 此为切片，属于基本索引，修改会改变原数值
sub[0] = 999
print(arr)  # → [1, 999, 3, 4]  原数组被修改！

# 高级索引 → 副本
sub2 = arr[[1, 2]]
sub2[0] = 888
print(arr)  # → [1, 999, 3, 4]  原数组不变
