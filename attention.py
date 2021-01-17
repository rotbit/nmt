import tensorflow as tf

class DotProductAttention(tf.keras.layers.Layer):
  def __init__(self):
    super(DotProductAttention, self).__init__()

  def call(self, query, value):
    # 32 * 512 * 1
    hidden = tf.expand_dims(query, -1)
    score = tf.matmul(value, hidden)
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * value
    # 求和
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector, attention_weights
