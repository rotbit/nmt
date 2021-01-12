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
import train

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


input_tensor, target_tensor, inp_lang, targ_lang = preprocess.load_dataset("./cmn.txt", 30000)

#print(target_tensor[100])
#for t in target_tensor[100]:
#    if t != 0:
#        print("%d ---> %s" % (t, targ_lang.index_word[t]))


# 采用80-20的比例切分训练集和验证集
input_tensor_train, input_tensor_val, target_tensor_train, \
    target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

print(len(input_tensor_train), len(input_tensor_val))

# 创建一个 tf.data 数据集
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 32
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256 # 词向量维度
units = 512
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

# 看一下dataset
example_input_batch, example_target_batch = next(iter(dataset))
print(example_input_batch.shape, example_target_batch.shape)


encoder = encoder.Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print ('output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

attention_layer = attention.BahdanauAttention(512)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))


# 注意力
attention_layer = attention.DotProductAttention()
context_vector, attention_weights = attention_layer(sample_hidden, sample_output)
print ('context_vector shape:  {}'.format(context_vector.shape))
print ('attention_weights state: {}'.format(attention_weights.shape))
