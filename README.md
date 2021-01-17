# nmt
中英文神经网络机器翻译，用于学习attention模型的一个例子，原文请参考:   
目前已知的问题:  
1、数据量小，没有进行足够的训练，仅作为demo展示encoder-decoder、attention的实际流程  
2、迭代次数不够，拟合程度不够


## encoder_impl.py  
### Encoder模型实现  
- embeding
- rnn

## decoder_impl.py 
### Decoder模型实现  
- embeding
- rnn
- attention

## attention.py
- attention
## test.py 
文本预处理、encoder、decoder、attention调用方法样例

## main.py
主函数，训练、文本翻译
