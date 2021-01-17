import tensorflow as tf
import decoder
import attention
import encoder
import preprocess

# 加载、预处理数据
input_tensor, target_tensor, inp_lang, targ_lang = preprocess.load_dataset("./cmn.txt", 30000)


# 公共参数定义
BUFFER_SIZE = len(input_tensor)
BATCH_SIZE = 32
steps_per_epoch = len(input_tensor)//BATCH_SIZE
embedding_dim = 256 # 词向量维度
units = 512
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

# 数据集
dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

# 输出dataset样例
example_input_batch, example_target_batch = next(iter(dataset))
print(example_input_batch.shape, example_target_batch.shape)

# 定义encoder
encoder = encoder.Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print ('encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

# 定义注意力
attention_layer = attention.DotProductAttention()
context_vector, attention_weights = attention_layer(sample_hidden, sample_output)
print ('context_vector shape:  {}'.format(context_vector.shape))
print ('attention_weights state: {}'.format(attention_weights.shape))

# 定义decoder
dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)
decoder = decoder.Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE, attention_layer)
dec_output, dec_state, attention_weights = decoder(dec_input, sample_hidden, sample_output)
print ('decoder shape: (batch size, sequence length, units) {}'.format(dec_output.shape))
print ('decoder Hidden state shape: (batch size, units) {}'.format(dec_state.shape))