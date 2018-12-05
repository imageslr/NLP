# -*- coding: UTF-8 -*-

import random
import sys
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import gen.seq2seq as rl_seq2seq


class GenModel(object):
    def __init__(self, config, name_scope, forward_only=False, num_samples=256, dtype=tf.float32):
        """
        :param config:
        :param name_scope:
        :param forward_only: 是否是测试阶段。TODO(Zhu) 这个和 self.forward_only 的值应该是一样的吧
        :param num_samples: 候选采样的个数
        :param dtype:
        """
        # 初始化配置参数
        self.source_vocab_size = config.vocab_size
        self.target_vocab_size = config.vocab_size
        self.emb_dim = config.emb_dim
        self.buckets = config.buckets
        self.learning_rate = tf.Variable(float(config.learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * config.learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.batch_size = config.batch_size
        self.num_layers = config.num_layers
        self.max_gradient_norm = config.max_gradient_norm

        self.mc_search = tf.placeholder(tf.bool, name="mc_search")
        self.forward_only = tf.placeholder(tf.bool, name="forward_only")
        self.up_reward = tf.placeholder(tf.bool, name="up_reward") # 是否将loss乘上对应的奖励 TODO(Zhu) 表示是否使用强化学习方式？
        self.reward_bias = tf.get_variable("reward_bias", [1], dtype=tf.float32)

        self.dtype = dtype
        self.num_samples = num_samples

        # 初始化输入变量的 placeholder
        self._init_input_placeholders()

        # 创建 Multi-RNN cell
        cell = self._create_cell()

        # 构建图
        self._build_graph(cell, name_scope, forward_only)

        # 保存变量
        self._saver(name_scope)

    def _create_cell(self):
        single_cell = tf.contrib.rnn.GRUCell(self.emb_dim)
        cell = single_cell
        if self.num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell] * self.num_layers)
        return cell

    def _init_input_placeholders(self):
        """
        设置输入的placeholder
        :return: encoder_inputs, decoder_inputs, target_weights, reward, targets
        """
        encoder_inputs = []
        decoder_inputs = []
        target_weights = []

        for i in xrange(self.buckets[-1][0]):  # Last bucket is the biggest one.
            # encoder_inputs 这个列表对象中的每一个元素名字分别为encoder0, encoder1,…,encoder39，encoder{i}，
            # 几何意义是 batch 中所有数据下标为 i 的元素组成的列表
            encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))

        # 长度加一表示在所有序列最后额外添加一个 last_target（用0表示，PAD_ID，只作占位？）
        # TODO(Zhu) 其他地方我描述这个 last_target 是 EOS，可能是错误的。这里应该只是一个占位的PAD，防止target移进的时候下标越界
        for i in xrange(self.buckets[-1][1] + 1):
            decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))

            # target_weights 是一个与 decoder_outputs 大小一样的 0-1 矩阵 (decoder_size + 1)
            # 该矩阵将目标序列长度以外的其他位置填充为标量值 0
            target_weights.append(tf.placeholder(self.dtype, shape=[None], name="weight{0}".format(i)))

        reward = [tf.placeholder(tf.float32, name="reward_%i" % i) for i in range(len(self.buckets))]

        # target 是 decoder_inputs 移进一个元素 (长度为 decoder_size)
        targets = [decoder_inputs[i + 1] for i in xrange(len(decoder_inputs) - 1)]

        self.encoder_inputs, \
        self.decoder_inputs, \
        self.target_weights, \
        self.reward, \
        self.targets = encoder_inputs, decoder_inputs, target_weights, reward, targets

        # return encoder_inputs, decoder_inputs, target_weights, reward, targets

    def _build_graph(self, cell, name_scope, forward_only):
        """
        构建图，定义训练方式。是一个seq2seq模型
        :param cell:
        :param name_scope:
        :return:
        """

        # （输入数据的前向传播）构建增强学习seq2seq模型
        self.outputs, self.losses, self.encoder_state = self._rl_seq2seq_model(cell)

        # （误差信息的反向传播）如果不是测试阶段，需要使用策略梯度下降法来更新参数
        if not forward_only:
            with tf.name_scope("gradient_descent"):
                self.gradient_norms = []  # 梯度的范数
                self.updates = []  # 用后向传播更新参数，列表中第i个元素表示graph{i}的后向传播梯度更新操作
                self.aj_losses = []
                self.gen_params = [p for p in tf.trainable_variables() if name_scope in p.name]  # 生成器要训练的参数
                opt = tf.train.AdamOptimizer()

                for b in xrange(len(self.buckets)):  # 依次使用每个桶的数据进行训练
                    # 由之前代码可知 self.reward 是一个类型为 float 的一维数组，每个桶有一个奖励值
                    R = tf.subtract(self.reward[b], self.reward_bias)  # 计算奖励
                    # TODO(Zhu) 这个adjusted_loss什么意思？使用策略调整的损失？
                    adjusted_loss = tf.cond(self.up_reward,
                                            lambda: tf.multiply(self.losses[b], self.reward[b]),
                                            lambda: self.losses[b])
                    self.aj_losses.append(adjusted_loss)  # 保存每一步的损失
                    gradients = tf.gradients(adjusted_loss, self.gen_params)  # 计算损失函数关于参数的梯度
                    clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                     self.max_gradient_norm)  # 防止梯度爆炸
                    self.gradient_norms.append(norm)
                    self.updates.append(opt.apply_gradients(
                        zip(clipped_gradients, self.gen_params), global_step=self.global_step))  # 更新参数
                    # ↑ 当前定义了length(buckets)个graph，故self.updates是一个列表对象，尺寸为length(buckets)，
                    # ↑ 列表中第i个元素表示graph{i}的梯度更新操作

    def _rl_seq2seq_model(self, cell):
        """
        构建seq2seq模型，返回模型的输出
        如果是在测试阶段，需要使用候选采样，通过投影层投影输出以解码
        :return: outputs, losses, encoder_state
        """
        output_projection, softmax_loss_function = self._get_sample_loss_fn()

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return rl_seq2seq.embedding_attention_seq2seq(
                encoder_inputs,
                decoder_inputs,
                cell,
                num_encoder_symbols=self.source_vocab_size,
                num_decoder_symbols=self.target_vocab_size,
                embedding_size=self.emb_dim,
                output_projection=output_projection,
                feed_previous=do_decode,  # 是否把上一轮的预测作为这一轮的输入 || 是否在测试
                mc_search=self.mc_search,  # TODO(Zhu) 文件位置：seq2seq._argmax_or_mcsearch 什么意思？
                dtype=self.dtype)


        outputs, losses, encoder_state = rl_seq2seq.model_with_buckets(
            self.encoder_inputs, self.decoder_inputs, self.targets, self.target_weights,
            self.buckets, self.source_vocab_size, self.batch_size,
            lambda x, y: seq2seq_f(x, y, tf.where(self.forward_only, True, False)),
            output_projection=output_projection, softmax_loss_function=softmax_loss_function)

        # 如果使用了 output_protection，需要投影输出以解码
        # 如果forward_only为true的话，outputs 一开始的形状是[bucket_num, num_steps, batch_size, emb_dim或rnn_size]
        # 如果forward_only为false的话，或者经过投影层转换后
        # outputs 的形状是[bucket_num, num_steps或decoder_size, batch_size, target_vocab_size]，也就是所有时间步的预测概率
        for b in xrange(len(self.buckets)):
            outputs[b] = [
                tf.cond(
                    self.forward_only,
                    lambda: tf.matmul(output, output_projection[0]) + output_projection[1],
                    lambda: output
                )
                for output in outputs[b]
            ]

        return outputs, losses, encoder_state

    def _get_sample_loss_fn(self):
        """
        如果使用采样softmax，需要设置投影层 output_projection
        只有在输出词表大于采样数量的时候才能使用采样softmax
        """
        output_projection = None
        softmax_loss_function = None

        if self.num_samples > 0 and self.num_samples < self.target_vocab_size:
            w_t = tf.get_variable("proj_w", [self.target_vocab_size, self.emb_dim], dtype=self.dtype)
            w = tf.transpose(w_t)  # [emb_dim, target_vocab_size]
            b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=self.dtype)
            output_projection = (w, b)  # XW+B 注意这里w是转置后的

            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])  # 全部展开为二维数组，第二维是正确的类别，对于每个预测单词来说正确的只有一个
                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(inputs, tf.float32)
                return tf.cast(
                    tf.nn.sampled_softmax_loss(local_w_t, local_b, labels, local_inputs,
                                               self.num_samples, self.target_vocab_size), self.dtype)

            softmax_loss_function = sampled_loss

        return output_projection, softmax_loss_function

    def _saver(self, name_scope):
        """
        保存变量
        """
        self.gen_variables = [k for k in tf.global_variables() if name_scope in k.name]
        self.saver = tf.train.Saver(self.gen_variables)