import tensorflow as tf
import attention
import decoder_impl
import encoder_impl
import preprocess
import train_function

# 优先GPU运行
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

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
max_length_targ, max_length_inp = preprocess.max_length(target_tensor), preprocess.max_length(input_tensor)

# 数据集
dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

# 定义encoder
encoder = encoder_impl.Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

# 定义注意力
attention_layer = attention.DotProductAttention()

# 定义decoder
decoder = decoder_impl.Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE, attention_layer)

# 模型训练
def train(epochs):
    EPOCHS = epochs

    for epoch in range(EPOCHS):
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        # dataset最多有steps_per_epoch个元素
        for (batch, (inp, targ)) in enumerate(dataset.take(len(input_tensor))):
            batch_loss = train_function.train_step(encoder, decoder, inp, targ, targ_lang, enc_hidden, BATCH_SIZE)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))

# 预测目标解码词语
def evaluate(sentence):

    sentence = preprocess.preprocess_sentence(sentence, 0)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    # max_length_targ 解码张量的最大长度
    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)
        tf.reshape(attention_weights, (-1, ))

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence

        # 预测的 ID 被输送回模型
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence

# 翻译
def translate(sentence):
    result, sentence = evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))


train(20)
translate("he is swimming in the river")
