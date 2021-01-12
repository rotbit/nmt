# nmt
中英文神经网络机器翻译  
待解决的问题

1、下一步，看具体执行的输入输出,优化代码  
2、取一个batch 只取一个batch, dataset.take怎么用?  
3、tf.math.logical_not  
4、tf.keras.optimizers.Adam()  
5、tf.math.equal  
6、tf.cast  
7、tf.expand_dims  
8、variables = encoder.trainable_variables + decoder.trainable_variables  
9、教师强制 
10、targ[:, t]
11、trainable_variables

----------------------------
0、熟悉dataset
#创建一个数据集，其元素是一个元组  
dataset = tf.data.Dataset.from_tensor_slices(([1,2,3,4], [7,8,9,0]))
# 将数据按照批划分，这里执行的结果是([1,2,3], [7,8,9]) ([4], [0])
dataset = dataset.batch(3)
# 便利数据集，通过迭代器遍历
v1, v2 = next(iter(dataset))


--------------------------------
7、tf.expand_dims  
