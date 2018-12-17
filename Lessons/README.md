## 实验记录:machine_translation_seq2seq

### Goal1：encode双向，decoder注意力
```
1. 1、2、3、4寻找一组稳定且效果好的结构
```
#### 结果分析：
```
1. 目前来看，2比1略好，3、4比1明显好，3、4很接近但是4不太稳定
2. 注意：以上采用BahdauanAttention，当采用LuongAttention的时候，效果比1还差（具体原因尚未调查）
```
#### 实验结果
```
1. machine_translation_seq2seq:
2. machine_translation_seq2seq_0: 设置batch_size=512，epoch=20，实例测试，BLEU计算模块
3. machine_translation_seq2seq_1: 设置batch_size=512，epoch=3，BLEU计算模块
4. machine_translation_seq2seq_2: 设置batch_size=512，epoch=3，BLEU计算模块，设置encoder为bidirectional LSTM
5. machine_translation_seq2seq_3: 设置batch_size=512，epoch=3，BLEU计算模块，decoder增加attention mechanism
6. machine_translation_seq2seq_4: 设置batch_size=512，epoch=3，BLEU计算模块，设置encoder为bidirectional LSTM，decoder增加attention mechanism
```

---
### Goal2：decoder逆序
```
1. 观察逆序是否能增加宾语的多样性
```
#### 结果分析：
```  
1. BLEU减少到和1一致的水平
2. 宾语多样性
```
#### 实验结果
```
7. machine_translation_seq2seq_5: 设置batch_size=512，epoch=3，BLEU计算模块，设置encoder为bidirectional LSTM，decoder增加attention mechanism，decoder逆序decoding
```

---
### Goal3、4、5、6、7调研方向
```
1. 训练过程：
        1. 不进行预训练，直接更新encoder、decoder1、decoder2
        2. 先预训练encoder、decoder1，正式训练decoder2，更新encoder、decoder1；
        3. (encoder、decoder1)和(encoder、decoder1、decoder2)交替更新
2. 具体参数：
        1. decoder2的state使用encoder的final state初始化，还是decoder1的final state初始化，还是两个final state的结合,结合分为两种：一种拼接，一种相加
        2. decoder1提供给attention mechanism的仅仅是state_ta，还是state_ta和output的结合
        3. encoder提供给attention mechanism的仅仅是state_ta，还是state_ta和input embedded的结合
        4. decoder1提供给attention mechanism的是通过greedysearch、sample、beamsearch(目前在训练阶段使用有困难，之后优化的时候可以进行探索)
```

---
### Goal3：polishing process构建
```
1. 对比4、7、8、9，验证polishing process是否可以提高BLEU；分析训练过程得出哪个提升更高、更稳定
```
#### 结果分析:
```
1. 经过测试，7、8、9相比4有明显的提升，近7个百分点
2. 提升值和速度：8>9>7
3. 稳定性和收敛值：9>8>7
```
#### 源码修改方案：
```
方案一：使用LSTM，修改源码，输出(h,c)，h和c都是TensorArray
方案二：使用GRU，修改源码，输出h，h是TensorArray
```
#### 实验结果
```
8. machine_translation_seq2seq_6: 设置batch_size=512，epoch=3，BLEU计算模块，设置encoder为bidirectional LSTM，修改了源码decoder.py使得decoder返回所有timestep的output和state，此处state是指返回LSTMStateTuple(h,c)中的h，h和c都是TensorArray
9. machine_translation_seq2seq_7: 设置batch_size=512，epoch=3，BLEU计算模块，设置encoder为bidirectional LSTM，修改了源码decoder.py使得decoder返回所有timestep的output和state，此处state是指返回LSTMStateTuple(h,c)中的h，h和c都是TensorArray。实现正序同向polishing process
    1. 训练阶段：直接训练(encoder1，decoder1，decoder2)，对比decoder1和decoder2的输出结果
10. machine_translation_seq2seq_8: 设置batch_size=512，epoch=3，BLEU计算模块，设置encoder为bidirectional LSTM，修改了源码decoder.py使得decoder返回所有timestep的output和state，此处state是指返回LSTMStateTuple(h,c)中的h，h和c都是TensorArray。实现正序同向polishing process
    1. 预训练阶段：encoder1和decoder1进行3个epoch的训练，然后利用encoder1的final state初始化decoder2的initial state，然后建立两个attention mechanism，分别是encoder1的h，decoder1的h
    2. 正式训练阶段：encoder1、decoder1、decoder2，对比decoder1和decoder2的输出结果
11. machine_translation_seq2seq_9: 设置batch_size=512，epoch=3，BLEU计算模块，设置encoder为bidirectional LSTM，修改了源码decoder.py使得decoder返回所有timestep的output和state，此处state是指返回LSTMStateTuple(h,c)中的h，h和c都是TensorArray。实现正序同向polishing process
    1. 训练阶段：(encoder1，decoder1)和(encoder1，decoder1，decoder2)交替训练，对比decoder1和decoder2的输出结果
```

---
### Goal4: decoder2的initial state提供源
```
1. 7、10、11、12，分析decoder2的initial state的从哪里提供
```
#### 结果分析:
```
1. 根据实验结果，11>12=7>10
```
#### 实验结果
```
12. machine_translation_seq2seq_10: 设置batch_size=512，epoch=3，BLEU计算模块，设置encoder为bidirectional LSTM，修改了源码decoder.py使得decoder返回所有timestep的output和state，此处state是指返回LSTMStateTuple(h,c)中的h，h和c都是TensorArray。实现正序同向polishing process
    1. 训练阶段：直接训练(encoder1，decoder1，decoder2)，对比decoder1和decoder2的输出结果
    2. decoder2的initial state使用decoder1的final state
13. machine_translation_seq2seq_11: 设置batch_size=512，epoch=3，BLEU计算模块，设置encoder为bidirectional LSTM，修改了源码decoder.py使得decoder返回所有timestep的output和state，此处state是指返回LSTMStateTuple(h,c)中的h，h和c都是TensorArray。实现正序同向polishing process
    1. 训练阶段：直接训练(encoder1，decoder1，decoder2)，对比decoder1和decoder2的输出结果
    2. decoder2的initial state使用encoder1和decoder1的final state的拼接
14. machine_translation_seq2seq_12: 设置batch_size=512，epoch=3，BLEU计算模块，设置encoder为bidirectional LSTM，修改了源码decoder.py使得decoder返回所有timestep的output和state，此处state是指返回LSTMStateTuple(h,c)中的h，h和c都是TensorArray。实现正序同向polishing process
    1. 训练阶段：直接训练(encoder1，decoder1，decoder2)，对比decoder1和decoder2的输出结果
    2. decoder2的initial state使用encoder1和decoder1的final state的相加
seq2seq_12:0.937 0.948
seq2seq_11:0.961 0.965
seq2seq_10:0.928 0.937
seq2seq_7:0.953 0.932
```

---
### Goal5：decoder1提供的memory
```
1. 验证decoder1提供给decoder2的memory是否包含decoder1的output、output采取哪种方式获得、output采取sample后的值的时候是否会更新之前的参数(13_2)
```
#### 结果分析：
```
1. 根据实验结果，13比11平均增加了4个百分点，说明memory包含output是有效的 
2. output使用transform后获得的值比transform+softmax+sample后的值平均高2.5个百分点
3. sample后的值是无法更新之前的参数，所有梯度都为None
```
#### 实验结果
```     
15. machine_translation_seq2seq_13: 设置batch_size=512，epoch=3，BLEU计算模块，设置encoder为bidirectional LSTM，修改了源码decoder.py使得decoder返回所有timestep的output和state，此处state是指返回LSTMStateTuple(h,c)中的h，h和c都是TensorArray。实现正序同向polishing process
    1. 训练阶段：直接训练(encoder1，decoder1，decoder2)，对比decoder1和decoder2的输出结果
    2. decoder2的initial state使用encoder1和decoder1的final state的拼接
    3. decoder1提供给decoder2的memory的组成(h,output)，output通过greedysearch获得，output分为两种情况：一种是transform后的值(13)，另外一种是transform+softmax+sample后的值(13_1)
    4. encoder1提供给decoder1的memory的组成h
    5. encoder1提供给decoder2的memory的组成h
seq2seq_13:0.938 0.930 0.936 0.966(6个epoch)
seq2seq_11:0.761 0.907 0.870 0.902 0.932(6个epoch)
seq2seq_13:   0.934 0.929 0.968
seq2seq_13_1: 0.913 0.908 0.938
```

---
### Goal6：encoder1提供的memory
```
1. 根据13、14、15，观察encoder1提供给decoder1、decoder2的memory是否需要包含input embedded
```
#### 结果分析：
```
1. 根据以下实验数据可得，15相对于14提升率0.5个百分点，15相对于13提升了1.3个百分点
13:0.926 0.9372 0.930
14:0.935 0.9448 0.938
15:0.944 0.945  0.941
```
#### 实验结果
```
16. machine_translation_seq2seq_14: 设置batch_size=512，epoch=3，BLEU计算模块，设置encoder为bidirectional LSTM，修改了源码decoder.py使得decoder返回所有timestep的output和state，此处state是指返回LSTMStateTuple(h,c)中的h，h和c都是TensorArray。实现正序同向polishing process
    1. 训练阶段：直接训练(encoder1，decoder1，decoder2)，对比decoder1和decoder2的输出结果
    2. decoder2的initial state使用encoder1和decoder1的final state的拼接
    3. decoder1提供给decoder2的memory的组成(h,output)，output通过greedysearch获得
    4. encoder1提供给decoder1的memory的组成h
    5. encoder1提供提供给decoder2的memory的组成(h,input embedded)
17. machine_translation_seq2seq_15: 设置batch_size=512，epoch=3，BLEU计算模块，设置encoder为bidirectional LSTM，修改了源码decoder.py使得decoder返回所有timestep的output和state，此处state是指返回LSTMStateTuple(h,c)中的h，h和c都是TensorArray。实现正序同向polishing process
    1. 训练阶段：直接训练(encoder1，decoder1，decoder2)，对比decoder1和decoder2的输出结果
    2. decoder2的initial state使用encoder1和decoder1的final state的拼接
    3. decoder1提供给decoder2的memory的组成(h,output)，output通过greedysearch获得
    4. encoder1提供给decoder1的memory的组成(h,input embedded)
    5. encoder1提供给decoder2的memory的组成(h,input embedded)
```

---
### Goal7: BeamSearchDecoder实现
```
1. 验证beamsearch相对于greedysearch是否会提升
```
#### 结果分析:
```
1. beam search相对于greedysearch提升了2个百分点
```
#### 方案调研
```
- 调研：RNN、LSTM、GRU、Attention的参数更新对象和过程
- 调研：推敲网络、异步反向更新网络的参数设置、更新过程
    - 理论上来讲应该用sample，但是直接多次采样计算量大，而且方差大不稳定(容易获得差的样本)，所以使用实践上来讲使用greedy search、beam search
    - greedy search：可以获取state、sample_id、sample_probability，可以采取(state，sample_id)、(state，sample_probability)，目前实验结果后者更好，初步分析是因为后者中sample_probability可以进行梯度传导，而前者中的sample_id不可以
    - beam search：使用方法同上，但难点在于如何获取state、sample_probability，因为前馈过程有剪枝操作，最后要构建树并回溯才可以获得对应sample_id的state、sample_probability
```
#### 实验结果
```    
18. machine_translation_seq2seq_16: 设置batch_size=512，epoch=3，BLEU计算模块，设置encoder为bidirectional LSTM，decoder增加attention mechanism
    1. encoder1提供给decoder1的memory的组成(h,input embedded)
    2. decoder1的inference过程使用BeamSearch
19. machine_translation_seq2seq_16_1: 设置batch_size=512，epoch=3，BLEU计算模块，设置encoder为bidirectional LSTM，decoder增加attention mechanism
    1. encoder1提供给decoder1的memory的组成(h,input embedded)
    2. decoder1的inference过程使用GreedySearch
```

---
### Goal8：对比正序和逆序的polishing process
```
1. polishing process相当于一种self attention
2. 对比两种顺序生成的实例，观察是否逆序的生成在宾语上更具多样性
```
#### 结果分析
```
两者的BLEU相差基本1个百分点，体现不出来两种模型的区别，无论在BLEU，还是实例上的差别，接下来的实验应该迁移到对话数据集上做对比
```
#### 数据预处理和BLEU计算方案注意事项
```
1. 数据预处理注意事项：首先将原句reverse()，然后对于target句子的处理，Padding放在EOS之后
2. BLEU计算方案：首先翻译过来的逆序句子必须再次正向回去，然后去掉EOS和Padding，然后是左侧的空格    
```
#### 实验结果
```
20. machine_translation_seq2seq_17: 设置batch_size=512，epoch=3，BLEU计算模块，设置encoder为bidirectional LSTM，修改了源码decoder.py使得decoder返回所有timestep的output和state，此处state是指返回LSTMStateTuple(h,c)中的h，h和c都是TensorArray。实现==正序同向polishing process==
    1. 训练阶段：(encoder1，decoder1)和(encoder1，decoder1，decoder2)交替训练，对比decoder1和decoder2的输出结果    
    2. decoder2的initial state使用encoder1和decoder1的final state的拼接
    3. decoder1提供给decoder2的memory的组成(h,output)，output通过greedysearch获得
    4. encoder1提供给decoder1的memory的组成(h,input embedded)
    5. encoder1提供给decoder2的memory的组成(h,input embedded)
21. machine_translation_seq2seq_18: 设置batch_size=512，epoch=3，BLEU计算模块，设置encoder为bidirectional LSTM，修改了源码decoder.py使得decoder返回所有timestep的output和state，此处state是指返回LSTMStateTuple(h,c)中的h，h和c都是TensorArray。实现==逆序同向polishing process==
    1. 训练阶段：(encoder1，decoder1)和(encoder1，decoder1，decoder2)交替训练，对比decoder1和decoder2的输出结果    
    2. decoder2的initial state使用encoder1和decoder1的final state的拼接
    3. decoder1提供给decoder2的memory的组成(h,output)，output通过greedysearch获得
    4. encoder1提供给decoder1的memory的组成(h,input embedded)
    5. encoder1提供给decoder2的memory的组成(h,input embedded)
```