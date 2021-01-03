import tensorflow as tf

# encoder
# 初始化: vocab_size-词汇表大小 embedding_dim-词嵌入维度 enc_uints-编码RNN节点数量 batch_sz-批大小
# batch_size: 深度学习里面，每一次参数的更新所计算的损失函数并不是仅仅由一个{data:label}所计算的，而是由一组{data:label}
# 加权得到的，这组数据的大小就是batch_size

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz # 批大小
        self.enc_units = enc_units # 编码单元个数(RNN单元个数)
        # Embedding 把一个整数转为一个固定长度的稠密向量
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn = tf.keras.layers.SimpleRNN(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    # x BATCH_SIZE * 文本特征长度 的矩阵（文本特征是每个词在单词表中的序号）, 输入的时候，都是输入一个BATCH_SIZE的文本向量
    # hidden BATCH_SIZE * enc_units 的矩阵 BATCH_SIZE个样本输入，所有会有BATCH_SIZE个输出结果，enc_uints是RNN输出神经元个数，所以输出结果是
    # BATCH_SIZE * enc_units 的矩阵
    # output BATCH_SIZE * 文本特征长度 * enc_uints
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.rnn(x, initial_state=hidden)
        return output, state

    # 张量的概念 tf.Tensor https://www.tensorflow.org/guide/tensor
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

