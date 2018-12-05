# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
import os
import time
import copy
import sys
import random
from six.moves import xrange
from utils.conf import disc_config
from utils.utils import just_message as just, softmax
from .hier_rnn_data import get_dataset
from .hier_rnn_model import HierRNNModel


class HierRNNTrain(object):
    def __init__(self):
        self.config_disc = disc_config
        self.config_evl = copy.deepcopy(disc_config) # TODO(Zhu) 重构前这里是通过引用直接赋值，不知道有什么影响
        self.config_evl.keep_prob = 1.0 # TODO(Zhu) config_evl这个参数好像没有使用到，不知道有什么用。所以上面好像没有什么影响？

    @staticmethod
    def create_model(sess, config, name_scope, initializer=None):
        """
        创建判别模型：如果已经有训练好的，读入；否则，初始化
        :param sess:
        :param config:
        :param name_scope: 也就是config.name_model
        :param initializer:
        :return:
        """
        print(just("Creating disc model"))
        with tf.variable_scope(name_or_scope=name_scope, initializer=initializer):
            model = HierRNNModel(config=config, name_scope=name_scope)
            disc_ckpt_dir = os.path.abspath(os.path.join(config.train_dir, "checkpoints"))
            ckpt = tf.train.get_checkpoint_state(disc_ckpt_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print(just("Reading Hier Disc model parameters from %s" % ckpt.model_checkpoint_path))
                model.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print(just("Created Hier Disc model with fresh parameters."))
                disc_global_variables = [gv for gv in tf.global_variables() if name_scope in gv.name]
                sess.run(tf.variables_initializer(disc_global_variables))
            return model

    def pre_train(self):
        """
        预训练判别器
        :return:
        """
        print(just("Begin training"))
        with tf.Session() as session:
            # ① 创建模型
            model = self.create_model(session, self.config_disc, name_scope=self.config_disc.name_model)

            # ② 获取数据集
            self.query_set, \
            self.answer_set, \
            self.gen_set, \
            self.train_buckets_scale = self._get_dataset()

            # [Ignore]... log相关
            step_time, loss = 0.0, 0.0
            current_step = 0
            step_loss_summary = tf.Summary()
            disc_writer = tf.summary.FileWriter(self.config_disc.tensorboard_dir, session.graph)

            while current_step <= self.config_disc.max_pre_train_step:
                start_time = time.time() # [Ignore]... log相关：开始时间

                # ③ 获取一个batch的训练数据
                bucket_id = self._get_random_bid()
                train_query, train_answer, train_labels = self._get_batch(bucket_id)

                # ④ 获取处理后的输入数据
                feed_dict = self._get_feed_dict(model, bucket_id, train_query, train_answer, train_labels)

                # ⑤ 选择训练OP，进行训练
                fetches = [model.b_train_op[bucket_id], model.b_logits[bucket_id], model.b_loss[bucket_id], model.target]
                train_op, logits, step_loss, target = session.run(fetches, feed_dict)

                # ================================ [Ignore]... log相关: 记录日志、保存变量 ================================ #

                # log相关：运行时间
                step_time += (time.time() - start_time) / self.config_disc.steps_per_checkpoint
                loss += step_loss / self.config_disc.steps_per_checkpoint
                current_step += 1

                # 每运行 config_disc.steps_per_checkpoint 次记录一下
                if current_step % self.config_disc.steps_per_checkpoint == 0:
                    # log相关
                    disc_loss_value = step_loss_summary.value.add()
                    disc_loss_value.tag = self.config_disc.name_loss
                    disc_loss_value.simple_value = float(loss)
                    disc_writer.add_summary(step_loss_summary, int(session.run(model.global_step)))

                    # softmax operation
                    logits = np.transpose(softmax(np.transpose(logits)))
                    reward = 0.0
                    for logit, label in zip(logits, train_labels):  # ([1, 0], 1)
                        reward += logit[1]  # only for true probility
                    reward = reward / len(train_labels)
                    print("reward: ", reward)

                    print("current_step: %d, step_loss: %.4f" % (current_step, step_loss))
                    if current_step % (self.config_disc.steps_per_checkpoint * 3) == 0:
                        print("current_step: %d, save_model" % (current_step))
                        disc_ckpt_dir = os.path.abspath(os.path.join(self.config_disc.train_dir, "checkpoints"))
                        if not os.path.exists(disc_ckpt_dir):
                            os.makedirs(disc_ckpt_dir)
                        disc_model_path = os.path.join(disc_ckpt_dir, "disc.model")
                        model.saver.save(session, disc_model_path, global_step=model.global_step)

                    step_time, loss = 0.0, 0.0
                    sys.stdout.flush()

    def step(self, sess, bucket_id, disc_model, train_query, train_answer, train_labels, forward_only=False):
        """
        使用包含正反例的一批数据训练判别器
        :param sess:
        :param bucket_id:
        :param disc_model:
        :param train_query: time-major
        :param train_answer: time-major
        :param train_labels:
        :param forward_only:
        :return:
        """
        feed_dict = {}

        for i in xrange(len(train_query)):
            feed_dict[disc_model.query[i].name] = train_query[i]

        for i in xrange(len(train_answer)):
            feed_dict[disc_model.answer[i].name] = train_answer[i]

        feed_dict[disc_model.target.name] = train_labels

        loss = 0.0
        if forward_only:
            fetches = [disc_model.b_logits[bucket_id]]  # 测试生成器：只需要获取最终结果
            logits = sess.run(fetches, feed_dict)
            logits = logits[0]
        else:
            # 训练判别器：需要执行训练、求损失等操作
            fetches = [disc_model.b_train_op[bucket_id], disc_model.b_loss[bucket_id], disc_model.b_logits[bucket_id]]
            train_op, loss, logits = sess.run(fetches, feed_dict)

        # softmax operation
        logits = np.transpose(softmax(np.transpose(logits)))

        reward, gen_num = 0.0, 0
        for logit, label in zip(logits, train_labels):
            if int(label) == 0:
                reward += logit[1]
                gen_num += 1
        reward = reward / gen_num  # 最终奖励是这一批数据中所有负例的奖励的平均值

        return reward, loss

    def _get_dataset(self):
        print(just("Prepare_data"))

        query_set, answer_set, gen_set = get_dataset(self.config_disc)

        train_bucket_sizes = [len(query_set[b]) for b in xrange(len(self.config_disc.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        return query_set, answer_set, gen_set, train_buckets_scale

    def _get_random_bid(self):
        """
        随机获取一个训练用的桶id
        :return:
        """
        random_number_01 = np.random.random_sample()
        return min([i for i in xrange(len(self.train_buckets_scale))
                         if self.train_buckets_scale[i] > random_number_01])

    def _get_batch(self, bucket_id):
        """
        获取一个batch的数据
        :param bucket_id:
        :return: train_query, train_answer, train_labels
        """
        batch_size = self.config_disc.batch_size

        if batch_size % 2 == 1:
            return IOError("Error")

        query_set, answer_set, gen_set = self.query_set[bucket_id], self.answer_set[bucket_id], self.gen_set[bucket_id]
        max_set = len(query_set) - 1

        train_query = []
        train_answer = []
        train_labels = []
        half_size = int(batch_size / 2)

        # 循环一半次数，每次得到两个训练数据：真实回答（标记为1），生成回答（标记为0）
        for _ in range(half_size):
            index = random.randint(0, max_set)
            train_query.append(query_set[index])
            train_answer.append(answer_set[index])
            train_labels.append(1)
            train_query.append(query_set[index])
            train_answer.append(gen_set[index])
            train_labels.append(0)

        return train_query, train_answer, train_labels

    def _get_feed_dict(self, model, bucket_id, train_query, train_answer, train_labels):
        """
        获取feed_dict，feed_dict为time-major形式
        :param model:
        :param bucket_id:
        :param train_query:
        :param train_answer:
        :param train_labels:
        :return:
        """
        train_query = np.transpose(train_query)
        train_answer = np.transpose(train_answer) # time-major

        feed_dict = {}

        for i in xrange(self.config_disc.buckets[bucket_id][0]):
            feed_dict[model.query[i].name] = train_query[i] # train_query[i] 表示这个 batch 中所有元素的第 i 个位置
        for i in xrange(self.config_disc.buckets[bucket_id][1]):
            feed_dict[model.answer[i].name] = train_answer[i]
        feed_dict[model.target.name] = train_labels

        return feed_dict