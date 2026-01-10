import tensorflow as tf
x = tf.constant([1.0, 2.0, 3.0])
with tf.GradientTape() as tape:
    y = tf.reduce_sum(x ** 2)
grad = tape.gradient(y, x)  # 自动求导

print('grad',y)
