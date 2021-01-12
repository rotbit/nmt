import tensorflow as tf
import decoder
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import attention
import optimizer
from sklearn.model_selection import train_test_split
import unicodedata
import re
import numpy as np
import os
import io
import time
import encoder
import preprocess


# 教师强制 Professor Forcing: A New Algorithm for Training Recurrent Networks
def train_step(inp, inp_lang,targ, targ_lang, enc_hidden,BATCH_SIZE):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

    # 教师强制 - 将目标词作为下一个输入，一个batch的循环
    #print("targ.shape={}".format(targ.shape))
    #print("targ.shape[0]={}".format(targ.shape[0]))

    # 以文本长度为主，便利所有词语
    for t in range(1, targ.shape[1]):
      # 将编码器输出 （enc_output） 传送至解码器
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      # 这里输入的是一个batch
      loss += optimizer.loss_function(targ[:, t], predictions)

      # 使用教师强制【论文】(取具体的单词)
      dec_input = tf.expand_dims(targ[:, t], 1)
      #print("dec_input={}".format(dec_input))s
  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss

def train(dataset, inp, inp_lang,targ, targ_lang, enc_hidden,BATCH_SIZE):
    EPOCHS = 20

    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        # dataset最多有steps_per_epoch个元素
        for (batch, (inp, targ)) in enumerate(dataset.take(len(inp))):
            batch_loss = train_step(inp, inp_lang,targ, targ_lang, enc_hidden,BATCH_SIZE)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))

#train()