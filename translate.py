import tensorflow as tf
import decoder
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import attention
from sklearn.model_selection import train_test_split
import jieba
import unicodedata
import re
import numpy as np
import os
import io
import time
import encoder

jieba.enable_paddle()

# 加载数据，文本预处理
# tensorflow的操作大多数希望输入的是utf-8编码，这里处理一下
def preprocess_sentence(w, type):
    if type == 0:
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
    if type == 1:
        seg_list = jieba.cut(w)
        w = " ".join(seg_list)
    w = '<start> ' + w + ' <end>'
    return w

# 加载文本
# Hi.	嗨。	CC-BY 2.0 (France) Attribution: tatoeba.org #538123 (CM) & #891077 (Martha)
# 第一个是英文，第二个是中文，后面的直接丢掉
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = []
    for l in lines[:num_examples]:
        pairs = []
        cnt = 0
        for w in l.split('\t'):
            # 英文
            if cnt == 0:
                word = preprocess_sentence(w, 0)
            # 中文
            elif cnt == 1:
                word = preprocess_sentence(w, 1)
            else:
                break
            pairs.append(word)
            cnt += 1
        word_pairs.append(pairs)
    return list(zip(*word_pairs))

def load_dataset(path, num_examples=None):
    inp_lang, targ_lang = create_dataset(path, num_examples)
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

# 文本内容标志化
def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')
    return tensor, lang_tokenizer

input_tensor, target_tensor, inp_lang, targ_lang = load_dataset("./cmn.txt", 30000)

#print(target_tensor[100])
#for t in target_tensor[100]:
#    if t != 0:
#        print("%d ---> %s" % (t, targ_lang.index_word[t]))


# 采用80-20的比例切分训练集和验证集
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

print(len(input_tensor_train), len(input_tensor_val))

# 创建一个 tf.data 数据集
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256 # 词向量维度
units = 1024
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
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))


# 注意力
attention_layer = attention.BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))


#  改下注释
decoder = decoder.Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((64, 1)),
                                      sample_hidden, sample_output)

print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))