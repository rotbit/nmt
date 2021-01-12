import tensorflow as tf

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, attention):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn = tf.keras.layers.SimpleRNN(self.dec_units,
                                       return_sequences=True,
                                       return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = attention

    # x 是输出目标词语[教师强制](这儿是个整数，是单词在词表中的index)
    def call(self, x, hidden, enc_output):
        # 编码器输出 （enc_output） 的形状 == （批大小，最大长度，隐藏层大小）
        # context_vector 的shape == (批大小，隐藏层大小)
        # attention_weight == (批大小，最大长度, 1)
        context_vector, attention_weights = self.attention(hidden, enc_output)
        #print("context_vector.shape={}".format(context_vector.shape))
        # x 在通过嵌入层后的形状 == （批大小，1，嵌入维度）
        x = self.embedding(x)

        # x 在拼接 （concatenation） 后的形状 == （批大小，1，嵌入维度 + 隐藏层大小）[特征拼接]
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        #print("x.shape={}".format(x.shape))

        # 将合并后的向量传送到 RNN, rnn需要的shape是(batch_size, time_step, feature)
        output, state = self.rnn(x)
        #print("output 1.shape={}".format(output.shape))

        # 输出的形状 == （批大小 * 1，隐藏层大小）
        # 将合并后的向量传送到 RNN, rnn需要的shape是(batch_size, time_step, feature),time_step这个维度没什么意义，在全连接层可以去掉,
        # 这里去掉
        output = tf.reshape(output, (-1, output.shape[2]))
        #print("output 2.shape={}".format(output.shape))
        # 输出的形状 == （批大小，vocab）,输出所有单词概率
        x = self.fc(output)
        return x, state, attention_weights