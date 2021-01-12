import tensorflow as tf
import decoder
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import attention
import optimizer
from sklearn.model_selection import train_test_split
import jieba
import unicodedata
import re
import numpy as np
import os
import io
import time
import encoder

def seg_char(sent):
    """
    把句子按字分开，不破坏英文结构
    """
    # 首先分割 英文 以及英文和标点
    pattern_char_1 = re.compile(r'([\W])')
    parts = pattern_char_1.split(sent)
    parts = [p for p in parts if len(p.strip())>0]
    # 分割中文
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    chars = pattern.split(sent)
    chars = [w for w in chars if len(w.strip())>0]
    return chars

# 文本预处理
def preprocess_sentence(w, type):
    if type == 0:
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
    if type == 1:
        #seg_list = jieba.cut(w)
        seg_list = seg_char(w)
        w = " ".join(seg_list)
    w = '<start> ' + w + ' <end>'
    return w

# 加载数据集 例如: Hi.	嗨.  CC-BY 2.0 (France) Attribution: tatoeba.org #538123 (CM) & #891077 (Martha)
# 第一个是英文，第二个是中文，后面的直接丢掉
# 返回一个元组列表，内容为(英文，中文)

# path 数据存储路径
# num_examples 读入记录条数
# 加载文本
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    # 英文文本
    english_words = []
    # 中文文本
    chinese_words = []
    for l in lines[:num_examples]:
        word_arrs = l.split('\t')
        if len(word_arrs) < 2:
            continue
        english_w = preprocess_sentence(word_arrs[0], 0)
        chinese_w = preprocess_sentence(word_arrs[1], 1)
        english_words.append(english_w)
        chinese_words.append(chinese_w)
    return english_words, chinese_words

# 文本内容转向量
def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')
    return tensor, lang_tokenizer

def load_dataset(path, num_examples=None):
    inp_lang, targ_lang = create_dataset(path, num_examples)
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

#inp_lang, targ_lang = create_dataset('cmn.txt', 4)
#print("inp_lang={}, targ_lang={}".format(inp_lang, targ_lang))
input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer = load_dataset("cmn.txt", 4)
print("input_tensor={}, inp_lang_tokenizer={}".format(input_tensor, inp_lang_tokenizer.index_word))