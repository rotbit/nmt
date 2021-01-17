import tensorflow as tf
import optimizer

# 单个样本的模型训练
def train_step(encoder, decoder, inp, targ, targ_lang, enc_hidden, BATCH_SIZE):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

    # 以文本长度为主，遍历所有词语
    for t in range(1, targ.shape[1]):
      # 将编码器输出 （enc_output） 传送至解码器
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      # 这里输入的是一个batch
      loss += optimizer.loss_function(targ[:, t], predictions)

      # 教师强制 - 将目标词作为下一个输入，一个batch的循环
      dec_input = tf.expand_dims(targ[:, t], 1)
  batch_loss = (loss / int(targ.shape[1]))
  variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.optimizer.apply_gradients(zip(gradients, variables))
  return batch_loss