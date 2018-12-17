# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
from six.moves import xrange


class HierRNNModel(object):
    def __init__(self, config, name_scope, dtype=tf.float32):
        self._init_config(config)
        self._init_placeholders()

        encoder_emb, context_multi = self._create_cell()

        self._build_graph(encoder_emb, context_multi, name_scope)

        self._saver(name_scope)

    def _init_config(self, config):
        """
        使用config初始化模型的参数
        """
        self.emb_dim = config.embed_dim
        self.num_layers = config.num_layers
        self.vocab_size = config.vocab_size
        self.num_class = config.num_class  # 二分类问题
        self.buckets = config.buckets
        self.lr = config.lr
        self.max_grad_norm=config.max_grad_norm

    def _init_placeholders(self):
        """
        初始化模型的placeholder
        self.query：time-major
        self.answer：time-major
        """
        self.global_step = tf.Variable(initial_value=0, trainable=False) # 记录全局训练步数
        self.query = []
        self.answer = []
        for i in range(self.buckets[-1][0]):
            self.query.append(tf.placeholder(dtype=tf.int32, shape=[None], name="query{0}".format(i)))
        for i in xrange(self.buckets[-1][1]):
            self.answer.append(tf.placeholder(dtype=tf.int32, shape=[None], name="answer{0}".format(i)))

        self.target = tf.placeholder(dtype=tf.int64, shape=[None], name="target")

    def _create_cell(self):
        """
        创建encoder层与context层的RNN单元
        """
        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.emb_dim)
        encoder_mutil = tf.contrib.rnn.MultiRNNCell([encoder_cell] * self.num_layers)
        # TODO(Zhu) 判别器需要使用自己的embedding吗？可不可以和生成器使用同一份word_embedding？
        encoder_emb = tf.contrib.rnn.EmbeddingWrapper(encoder_mutil, embedding_classes=self.vocab_size,
                                                      embedding_size=self.emb_dim)

        context_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.emb_dim)
        context_multi = tf.contrib.rnn.MultiRNNCell([context_cell] * self.num_layers)

        return encoder_emb, context_multi

    def _build_graph(self, encoder_emb, context_multi, name_scope):
        """
        构建图
        :param encoder_emb: 一个 Multi-LSTM 单元
        :param context_multi: 一个 Multi-LSTM 单元
        :param name_scope
        """
        self.b_query_state = []
        self.b_answer_state = []
        self.b_state = []
        self.b_logits = []
        self.b_loss = []
        self.b_train_op = []
        for i, bucket in enumerate(self.buckets):
            with tf.variable_scope(name_or_scope="Hier_RNN_encoder", reuse=True if i > 0 else None) as var_scope:
                # query_state 形状： [num_layer, 2, batch_size, emb_dim]
                # num_layer 是 MultiRNNCell 的层数，2 表示每一层 LSTM 的输出是一个元组(c_t, h_t）
                # 第二维的每个元组的形状是[batch_size, emb_dim]
                query_output, query_state = tf.contrib.rnn.static_rnn(encoder_emb, inputs=self.query[:bucket[0]],
                                                                      dtype=tf.float32)
                var_scope.reuse_variables()  # TODO(Zhu) 这里reuse表示answer和query的RNN参数是同一套吗？
                answer_output, answer_state = tf.contrib.rnn.static_rnn(encoder_emb, inputs=self.answer[:bucket[1]],
                                                                        dtype=tf.float32)
                self.b_query_state.append(query_state)
                self.b_answer_state.append(answer_state)
                context_input = [query_state[-1][1], answer_state[-1][1]]

            with tf.variable_scope(name_or_scope="Hier_RNN_context", reuse=True if i > 0 else None):
                output, state = tf.contrib.rnn.static_rnn(context_multi, context_input, dtype=tf.float32)
                self.b_state.append(state)
                top_state = state[-1][1]  # [batch_size, emb_dim]

            with tf.variable_scope("Softmax_layer_and_output", reuse=True if i > 0 else None):
                softmax_w = tf.get_variable("softmax_w", [self.emb_dim, self.num_class], dtype=tf.float32)
                softmax_b = tf.get_variable("softmax_b", [self.num_class], dtype=tf.float32)
                logits = tf.matmul(top_state, softmax_w) + softmax_b
                self.b_logits.append(logits)

            with tf.name_scope("loss"):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.target)
                mean_loss = tf.reduce_mean(loss)
                self.b_loss.append(mean_loss)

            with tf.name_scope("gradient_descent"):
                disc_params = [var for var in tf.trainable_variables() if name_scope in var.name]
                grads, norm = tf.clip_by_global_norm(tf.gradients(mean_loss, disc_params), self.max_grad_norm)
                optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
                train_op = optimizer.apply_gradients(zip(grads, disc_params), global_step=self.global_step)
                self.b_train_op.append(train_op)

    def _saver(self, name_scope):
        """
        保存变量
        """
        all_variables = [v for v in tf.global_variables() if name_scope in v.name]
        self.saver = tf.train.Saver(all_variables)