# coding=utf-8
import numpy as np 
import tensorflow as tf 
import matplotlib as mpl 
mpl.use('Agg')
import matplotlib.pyplot as plt 
import tqdm
import os
import _pickle as cpickle # python3.x中cpickle改为_pickle

f = open('dataFile.txt', 'rb')
text = cpickle.load(f)
source_text = text[0]
target_text = text[1]
source = cpickle.load(f)
source_vocab_to_int = source[0]
source_int_to_vocab = source[1]
target = cpickle.load(f)
target_vocab_to_int = target[0]
target_int_to_vocab = target[1]
text_to_int = cpickle.load(f)
source_text_to_int = text_to_int[0]
target_text_to_int = text_to_int[1]
f.close()

X = np.array(source_text_to_int)
Y = np.array(target_text_to_int)

def model_inputs():
    """
    构造输入

    返回：inputs, targets, learning_rate, source_sequence_len, target_sequence_len, max_target_sequence_len，类型为tensor
    """
    inputs = tf.placeholder(tf.int32, [None, None], name="inputs")
    targets = tf.placeholder(tf.int32, [None, None], name="targets")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    source_sequence_len = tf.placeholder(tf.int32, (None,), name="source_sequence_len") # TODO 这里和[None]是不是一模一样
    target_sequence_len = tf.placeholder(tf.int32, (None,), name="target_sequence_len")
    max_target_sequence_len = tf.placeholder(tf.int32, (None,), name="max_target_sequence_len")

    return inputs, targets, learning_rate, source_sequence_len, target_sequence_len, max_target_sequence_len


def encoder_layer(rnn_inputs, rnn_size, rnn_num_layers,
                  source_sequence_len, source_vocab_size, encoder_embedding_size=100):
    """
    构造Encoder端

    @param rnn_inputs: rnn的输入
    @param rnn_size: rnn的隐层结点数
    @param rnn_num_layers: rnn的堆叠层数
    @param source_sequence_len: 英文句子序列的长度
    @param source_vocab_size: 英文词典的大小
    @param encoder_embedding_size: Encoder层中对单词进行词向量嵌入后的维度
    """
    # 对输入的单词进行词向量嵌入
    encoder_embed = tf.contrib.layers.embed_sequence(rnn_inputs, source_vocab_size, encoder_embedding_size)

    with tf.variable_scope("birnn_encoder"):
        encoder_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
        encoder_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
        encoder_fw_multi_cell = tf.nn.rnn_cell.MultiRNNCell(
            [encoder_fw_cell for _ in range(rnn_num_layers)])
        encoder_bw_multi_cell = tf.nn.rnn_cell.MultiRNNCell(
            [encoder_bw_cell for _ in range(rnn_num_layers)])

        # The following four tensor shape：B*T*D，B*T*D，B*D，B*D
        # outputs: (batch_size, time_steps, rnn_size)
        # state: [rnn_num_layers, 2]，即rnn_num_layers个LSTMStateTuple: (c, h)。c、h形状: (batch_size, rnn_size)
        (encoder_fw_all_outputs, encoder_bw_all_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_fw_multi_cell,
                                            cell_bw=encoder_bw_multi_cell,
                                            inputs=encoder_embed,
                                            sequence_length=source_sequence_len,
                                            dtype=tf.float32, time_major=False)

        # encoder_outputs.shape: (batch_size, time_steps, rnn_size*2=256)
        encoder_outputs = tf.concat((encoder_fw_all_outputs, encoder_bw_all_outputs), 2)
        encoder_final_state_c = tf.concat(
            (encoder_fw_final_state[-1].c, encoder_bw_final_state[0].c), 1) # (batch_size, rnn_size*2)
        encoder_final_state_h = tf.concat(
            (encoder_fw_final_state[-1].h, encoder_bw_final_state[0].h), 1) # (batch_size, rnn_size*2)
        encoder_states = tf.nn.rnn_cell.LSTMStateTuple(
            c=encoder_final_state_c,
            h=encoder_final_state_h)

    return encoder_outputs, (encoder_states,)


def decoder_layer_inputs(target_data, target_vocab_to_int, batch_size):
    """
    对Decoder端的输入进行处理

    @param target_data: 法语数据的tensor
    @param target_vocab_to_int: 法语数据的词典到索引的映射
    @param batch_size: batch size
    """
    # 去掉batch中每个序列句子的最后一个单词
    # tf.strided_slice(input, start, end, 切片长度). end是开区间
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    # 在batch中每个序列句子的前面添加”<GO>"
    decoder_inputs = tf.concat([tf.fill([batch_size, 1], target_vocab_to_int["<GO>"]),
                                ending], 1)

    return decoder_inputs

def decoder_layer_train(encoder_outputs, encoder_states, decoder_attn_multi_cell, decoder_embed,
                        target_sequence_len, max_target_sequence_len, output_layer):

    """
    Decoder端的训练

    @param encoder_outputs: Encoder端编码得到的output
    @param encoder_states: Encoder端编码得到的Context Vector
    @param decoder_attn_multi_cell: Decoder端
    @param decoder_embed: Decoder端词向量嵌入后的输入
    @type decoder_embed: list [target_vocab_size, embedding_size]
    @param target_sequence_len: 法语文本的长度
    @type target_sequence_len: list
    @param max_target_sequence_len: 法语文本的最大长度
    @param output_layer: 输出层
    """

    # 生成helper对象
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed,
                                                       sequence_length=target_sequence_len,
                                                       time_major=False)
    # 生成decoder端initial hidden state
    # TODO 为什么要clone？
    decoder_initial_state = decoder_attn_multi_cell.zero_state(batch_size, 
                                                               tf.float32).clone(cell_state=encoder_states)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_attn_multi_cell,
                                                      training_helper,
                                                      decoder_initial_state,
                                                      output_layer)
    training_decoder_outputs, training_decoder_final_state, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                  impute_finished=True,
                                                                  maximum_iterations=max_target_sequence_len)

    return training_decoder_outputs 

def decoder_layer_infer(encoder_outputs, encoder_states, decoder_attn_multi_cell, decoder_embed, start_id, end_id,
                        max_target_sequence_len, output_layer, batch_size):
    """
    Decoder端的预测/推断

    @param encoder_outputs: Encoder端编码得到的output
    @param encoder_states: Encoder端编码得到的Context Vector
    @param decoder_attn_multi_cell: Decoder端
    @param decoder_embed: Decoder端词向量嵌入后的输入
    @type decoder_embed: list [target_vocab_size, embedding_size]
    @param start_id: 句子起始单词的token id， 即"<GO>"的编码
    @param end_id: 句子结束的token id，即"<EOS>"的编码
    @param max_target_sequence_len: 法语文本的最大长度
    @param output_layer: 输出层
    @batch_size: batch size
    """

    # batch_size 个 start_id 组成的数组
    start_tokens = tf.tile(tf.constant([start_id], dtype=tf.int32), [batch_size], name="start_tokens")

    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embed,
                                                                start_tokens,
                                                                end_id)
    # 生成decoder端initial hidden state
    # TODO 为什么要 clone
    decoder_initial_state = decoder_attn_multi_cell.zero_state(batch_size, 
                                                               tf.float32).clone(cell_state=encoder_states)

    inference_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_attn_multi_cell,
                                                       inference_helper,
                                                       decoder_initial_state,
                                                       output_layer)

    inference_decoder_outputs, inference_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                      impute_finished=True,
                                                                      maximum_iterations=max_target_sequence_len)

    return inference_decoder_outputs 

def decoder_layer(encoder_outputs, encoder_states, decoder_inputs, target_sequence_len,
                   max_target_sequence_len, rnn_size, rnn_num_layers,
                   target_vocab_to_int, target_vocab_size, decoder_embedding_size, batch_size):
    """
    构造Decoder端

    @param encoder_outputs: Encoder端编码得到的output
    @param encoder_states: Encoder端编码得到的Context Vector
    @param decoder_inputs: Decoder端的输入
    @param target_sequence_len: 法语文本的长度
    @param max_target_sequence_len: 法语文本的最大长度
    @param rnn_size: rnn隐层结点数
    @param rnn_num_layers: rnn堆叠层数
    @param target_vocab_to_int: 法语单词到token id的映射
    @param target_vocab_size: 法语词典的大小
    @param decoder_embedding_size: Decoder端词向量嵌入的大小
    @param batch_size: batch size
    """
    # embedding layer 
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoder_embedding_size]))
    decoder_embed = tf.nn.embedding_lookup(decoder_embeddings, decoder_inputs)

    # lstm layer 
    def get_lstm(rnn_size):
        lstm = tf.contrib.rnn.LSTMCell(rnn_size*2, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=456))
        return lstm 
    decoder_cell = tf.contrib.rnn.MultiRNNCell([get_lstm(rnn_size) for _ in range(rnn_num_layers)])
    
    # attention layer 
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(  # .LuongAttention(
        num_units=rnn_size*2,
        memory=encoder_outputs,
        memory_sequence_length=target_sequence_len)
    decoder_attn_multi_cell = tf.contrib.seq2seq.AttentionWrapper(
        decoder_cell, 
        attention_mechanism,
        attention_layer_size=rnn_size*2,
        alignment_history=True) # TODO 什么是 alignment_history ？

    # output_layer logits
    output_layer = tf.layers.Dense(target_vocab_size)

    with tf.variable_scope("attn_decoder"):
        training_logits = decoder_layer_train(encoder_outputs,
                                              encoder_states,
                                              decoder_attn_multi_cell,
                                              decoder_embed,
                                              target_sequence_len,
                                              max_target_sequence_len,
                                              output_layer)

    with tf.variable_scope("attn_decoder", reuse=True):
        inference_logits = decoder_layer_infer(encoder_outputs,
                                               encoder_states,
                                               decoder_attn_multi_cell,
                                               decoder_embeddings,
                                               target_vocab_to_int["<GO>"],
                                               target_vocab_to_int["<EOS>"],
                                               max_target_sequence_len,
                                               output_layer,
                                               batch_size)

    return training_logits, inference_logits

def seq2seq_model(input_data, target_data, batch_size,
                 source_sequence_len, target_sequence_len, max_target_sentence_len,
                 source_vocab_size, target_vocab_size,
                 encoder_embedding_size, decoder_embeding_size,
                 rnn_size, rnn_num_layers, target_vocab_to_int):

    """
    构造Seq2Seq模型

    @param input_data: tensor of input data
    @param target_data: tensor of target data
    @param batch_size: batch size
    @param source_sequence_len: 英文语料的长度
    @param target_sequence_len: 法语语料的长度
    @param max_target_sentence_len: 法语的最大句子长度
    @param source_vocab_size: 英文词典的大小
    @param target_vocab_size: 法语词典的大小
    @param encoder_embedding_size: Encoder端词嵌入向量大小
    @param decoder_embedding_size: Decoder端词嵌入向量大小
    @param rnn_size: rnn隐层结点数
    @param rnn_num_layers: rnn堆叠层数
    @param target_vocab_to_int: 法语单词到token id的映射
    """
    encoder_outputs, encoder_states = encoder_layer(input_data, rnn_size, rnn_num_layers, source_sequence_len,
                                      source_vocab_size, encoder_embedding_size)

    decoder_inputs = decoder_layer_inputs(target_data, target_vocab_to_int, batch_size)

    training_decoder_outputs, inference_decoder_outputs = decoder_layer(encoder_outputs, 
                                                                        encoder_states,
                                                                        decoder_inputs,
                                                                        target_sequence_len,
                                                                        max_target_sentence_len,
                                                                        rnn_size,
                                                                        rnn_num_layers,
                                                                        target_vocab_to_int,
                                                                        target_vocab_size,
                                                                        decoder_embeding_size,
                                                                        batch_size)
    return training_decoder_outputs, inference_decoder_outputs

# Number of Epochs
epochs = 4
# Batch Size
batch_size = 512
# RNN Size
rnn_size = 128
# Number of Layers
rnn_num_layers = 1
# Embedding Size
encoder_embedding_size = 100
decoder_embedding_size = 100
# Learning Rate
lr = 0.001
# 每50轮打一次结果
display_step = 50

train_graph = tf.Graph()

with train_graph.as_default():
    inputs, targets, learning_rate, source_sequence_len, target_sequence_len, _ = model_inputs()

    max_target_sequence_len = 25
    train_logits, inference_logits = seq2seq_model(tf.reverse(inputs, [-1]),
                                                  targets,
                                                  batch_size,
                                                  source_sequence_len,
                                                  target_sequence_len,
                                                  max_target_sequence_len,
                                                  len(source_vocab_to_int),
                                                  len(target_vocab_to_int),
                                                  encoder_embedding_size,
                                                  decoder_embedding_size,
                                                  rnn_size,
                                                  rnn_num_layers,
                                                  target_vocab_to_int)

    training_logits = tf.identity(train_logits.rnn_output, name="logits")
    inference_logits = tf.identity(inference_logits.sample_id, name="predictions")

    masks = tf.sequence_mask(target_sequence_len, max_target_sequence_len, dtype=tf.float32, name="masks")

    with tf.name_scope("optimization"):
        cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)

        optimizer = tf.train.AdamOptimizer(learning_rate)

        gradients = optimizer.compute_gradients(cost)
        clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(clipped_gradients)

def get_batches(sources, targets, batch_size):
    """
    获取batch
    """
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        
        # Need the lengths for the _lengths parameters
        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))

        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))

        yield sources_batch, targets_batch, source_lengths, targets_lengths 

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                get_batches(source_text_to_int, target_text_to_int, batch_size)):
    
            _, loss = sess.run(
                [train_op, cost],
                {inputs: source_batch,
                 targets: target_batch,
                 learning_rate: lr,
                 source_sequence_len: sources_lengths,
                 target_sequence_len: targets_lengths})


            if batch_i % display_step == 0 and batch_i > 0:


                batch_train_logits = sess.run(
                    inference_logits,
                    {inputs: source_batch,
                     source_sequence_len: sources_lengths,
                     target_sequence_len: targets_lengths})

                print('Epoch {:>3} Batch {:>4}/{} - Loss: {:>6.4f}'
                      .format(epoch_i, batch_i, len(source_text_to_int) // batch_size, loss))
        
    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, "checkpoints/dev")
    print('Model Trained and Saved')
