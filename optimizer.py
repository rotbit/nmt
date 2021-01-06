import tensorflow as tf

# 定义优化器， 优化器其实就是梯度下降算法的实现,Adam是其中一种实现的梯度下降算法
optimizer = tf.keras.optimizers.Adam()

# tf.keras.losses.SparseCategoricalCrossentropy 计算损失预测值和标签值的差
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

# 这里输入的是词语的index，如果是0，那么属于非法的句向量，mask的作用是去掉这些向量
def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)