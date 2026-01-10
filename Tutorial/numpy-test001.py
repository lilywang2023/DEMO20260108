
# https://www.bilibili.com/video/BV1dS4y1e7bg?spm_id_from=333.788.player.switch&vd_source=7f4dc3a917eeaf6b5ddb3b84e55845a6


import numpy as np

x = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
           ])

print(x.shape)

# 方式一：整数数组索引
############################################################################################
y = x[ [0,1,2],[0,1,0]]

print(y.shape)

print(y)


z = x[ [0,1,2],:]

print(z.shape)

print(z)



x = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])

print(x.shape)
print('\n')

rows = np.array([[0,0],[3,3]])
cols = np.array([[0,2],[0,2]])

y = x[rows,cols]

print(y)

# 方式一：布尔索引
############################################################################################

x = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])

print(x.shape)
print('\n')

rows = np.array([True,False,False,True])
cols = np.array([False,True,False])

y = x[rows,cols]

print("y=",y)

y1 = x[x>4]
print("y1=",y1)
# 方式一：花式索引
############################################################################################
x=np.arange(32).reshape((8,4))
print("原数组为：",x)

print("化式索引后的数组为：")

print(x[[4,2,1,7]])