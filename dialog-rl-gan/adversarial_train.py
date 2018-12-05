# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
import sys
import os
import time
from six.moves import xrange
import utils.conf as conf
from gen.gen_train import GenTrain
from gen.gen_data import get_dataset, get_batch
from disc.hier_rnn_model import HierRNNModel
from disc.hier_rnn_train import HierRNNTrain
from utils.utils import just_message as just
import utils.data_utils as data_utils

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Adversarial Learning for Neural Dialogue Generation
def al_train():

    gen_config = conf.gen_config
    disc_config = conf.disc_config
    adver_config = conf.adver_config

    with tf.Session() as sess:
        # ① 获取数据集
        vocab, rev_vocab, dev_set, train_set = get_dataset(gen_config)
        for set in train_set:
            print("al train len: ", len(set))

        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(gen_config.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # ② 创建模型
        disc_model = HierRNNTrain.create_model(sess, disc_config, disc_config.name_model)
        gen_model = GenTrain.create_model(sess, gen_config, forward_only=False, name_scope=gen_config.name_model)

        # [Ignore]... log相关
        current_step = 0
        step_time, disc_loss, gen_loss, t_loss, batch_reward = 0.0, 0.0, 0.0, 0.0, 0.0
        gen_loss_summary = tf.Summary()
        disc_loss_summary = tf.Summary()
        gen_writer = tf.summary.FileWriter(gen_config.tensorboard_dir, sess.graph)
        disc_writer = tf.summary.FileWriter(disc_config.tensorboard_dir, sess.graph)

        # ③ 开始对抗训练
        while current_step <= adver_config.max_train_step:
            start_time = time.time() # [Ignore]... log相关：开始时间
            current_step += 1

            bucket_id = get_random_bid(train_buckets_scale)

            # =========================================== ③.① 训练判别器 =========================================== #
            print(just("Update Discriminator: %d" % current_step))

            # 1. 获取一个batch的真实数据
            encoder_inputs, decoder_inputs, target_weights, source_inputs, source_outputs = get_batch(
                gen_model, train_set, bucket_id, gen_config.batch_size)

            # 2. 使用生成器生成负例数据，补充数据集 Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X)
            train_query, train_answer, train_labels = disc_train_data(sess, gen_model, vocab, source_inputs,
                                                                      source_outputs,
                                                                      encoder_inputs, decoder_inputs, target_weights,
                                                                      bucket_id, mc_search=False)

            # [Ignore]... message
            print(just("mc_search: False"))
            if current_step % 200 == 0:
                print("train_query: ", len(train_query))
                print("train_answer: ", len(train_answer))
                print("train_labels: ", len(train_labels))
                for i in xrange(len(train_query)):
                    print("label: ", train_labels[i])
                    print("train_answer_sentence: ", train_answer[i])
                    print(" ".join([tf.compat.as_str(rev_vocab[output]) for output in train_answer[i]]))

            # 转置为 time-major
            train_query = np.transpose(train_query)
            train_answer = np.transpose(train_answer)

            # 3. 使用正反例训练判别器 Update D using (X, Y ) as positive examples and(X, ^Y) as negative examples
            _, disc_step_loss = HierRNNTrain().step(sess, bucket_id, disc_model, train_query, train_answer, train_labels,
                                                    forward_only=False)
            disc_loss += disc_step_loss / disc_config.steps_per_checkpoint

            # =========================================== ③.② 训练生成器 =========================================== #
            print(just("Update Generator: %d" % current_step))

            # 1. 获取一批真实数据 Sample (X,Y) from real disc_data
            update_gen_data = get_batch(gen_model, train_set, bucket_id, gen_config.batch_size)
            encoder, decoder, weights, source_inputs, source_outputs = update_gen_data

            # 2. 生成一批训练数据，包含自己生成的负例。生成负例时采用蒙特卡洛方法
            # Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X) with Monte Carlo search
            train_query, train_answer, train_labels = disc_train_data(sess, gen_model, vocab, source_inputs,
                                                                      source_outputs,
                                                                      encoder, decoder, weights, bucket_id,
                                                                      mc_search=True)

            # [Ignore]... message
            print(just("mc_search: True"))
            if current_step % 200 == 0:
                for i in xrange(len(train_query)):
                    print("label: ", train_labels[i])
                    print(" ".join([tf.compat.as_str(rev_vocab[output]) for output in train_answer[i]]))

            train_query = np.transpose(train_query)
            train_answer = np.transpose(train_answer)

            # 3. 计算生成器（基于蒙特卡洛方法）生成的数据的奖励值：最终奖励是这一批数据中所有负例的奖励的平均值
            # Compute Reward r for (X, ^Y ) using D.---based on Monte Carlo search
            reward, _ = HierRNNTrain().step(sess, bucket_id, disc_model, train_query, train_answer, train_labels,
                                            forward_only=True)
            batch_reward += reward / gen_config.steps_per_checkpoint
            print("step_reward: ", reward)

            # 4. 强化学习：使用奖励值r更新生成器 Update G on (X, ^Y ) using reward r
            # TODO(Zhu) 如何使用奖励值reward实现强化学习？
            gan_adjusted_loss, gen_step_loss, _ = GenTrain().step(gen_model, sess, encoder, decoder, weights, bucket_id,
                                                                  forward_only=False, # forward_only=False 训练模型
                                                                  reward=reward, up_reward=True) # up_reward：使用reward
            gen_loss += gen_step_loss / gen_config.steps_per_checkpoint

            print("gen_step_loss: ", gen_step_loss)
            print("gen_step_adjusted_loss: ", gan_adjusted_loss)

            # 5. Teacher-Forcing: Update G on (X, Y ) 这时候就不需要设置up_reward为true了
            t_adjusted_loss, t_step_loss, a = GenTrain().step(gen_model, sess, encoder, decoder, weights, bucket_id,
                                                              forward_only=False)  # forward_only=False 训练模型
            t_loss += t_step_loss / gen_config.steps_per_checkpoint

            print("t_step_loss: ", t_step_loss)
            print("t_adjusted_loss", t_adjusted_loss)

            # ================================ [Ignore]... log相关: 记录日志、保存变量 ================================ #

            if current_step % gen_config.steps_per_checkpoint == 0:
                step_time += (time.time() - start_time) / gen_config.steps_per_checkpoint

                print("current_steps: %d, step time: %.4f, disc_loss: %.3f, gen_loss: %.3f, t_loss: %.3f, reward: %.3f"
                      % (current_step, step_time, disc_loss, gen_loss, t_loss, batch_reward))

                disc_loss_value = disc_loss_summary.value.add()
                disc_loss_value.tag = disc_config.name_loss
                disc_loss_value.simple_value = float(disc_loss)
                disc_writer.add_summary(disc_loss_summary, int(sess.run(disc_model.global_step)))

                gen_global_steps = sess.run(gen_model.global_step)
                gen_loss_value = gen_loss_summary.value.add()
                gen_loss_value.tag = gen_config.name_loss
                gen_loss_value.simple_value = float(gen_loss)
                t_loss_value = gen_loss_summary.value.add()
                t_loss_value.tag = gen_config.teacher_loss
                t_loss_value.simple_value = float(t_loss)
                batch_reward_value = gen_loss_summary.value.add()
                batch_reward_value.tag = gen_config.reward_name
                batch_reward_value.simple_value = float(batch_reward)
                gen_writer.add_summary(gen_loss_summary, int(gen_global_steps))

                if current_step % (gen_config.steps_per_checkpoint * 2) == 0:
                    print("current_steps: %d, save disc model" % current_step)
                    disc_ckpt_dir = os.path.abspath(os.path.join(disc_config.train_dir, "checkpoints"))
                    if not os.path.exists(disc_ckpt_dir):
                        os.makedirs(disc_ckpt_dir)
                    disc_model_path = os.path.join(disc_ckpt_dir, "disc.model")
                    disc_model.saver.save(sess, disc_model_path, global_step=disc_model.global_step)

                    print("current_steps: %d, save gen model" % current_step)
                    gen_ckpt_dir = os.path.abspath(os.path.join(gen_config.train_dir, "checkpoints"))
                    if not os.path.exists(gen_ckpt_dir):
                        os.makedirs(gen_ckpt_dir)
                    gen_model_path = os.path.join(gen_ckpt_dir, "gen.model")
                    gen_model.saver.save(sess, gen_model_path, global_step=gen_model.global_step)

                step_time, disc_loss, gen_loss, t_loss, batch_reward = 0.0, 0.0, 0.0, 0.0, 0.0
                sys.stdout.flush()


def get_random_bid(train_buckets_scale):
    """
    随机获取一个训练用的桶id
    :return:
    """
    random_number_01 = np.random.random_sample()
    return min([i for i in xrange(len(train_buckets_scale))
                     if train_buckets_scale[i] > random_number_01])


def disc_train_data(sess, gen_model, vocab, source_inputs, source_outputs,
                    encoder_inputs, decoder_inputs, target_weights, bucket_id, mc_search=False):
    """
    生成判别器的训练数据：对一批问题，使用生成器生成一堆负例回答
    如果使用mc_search，需要重复beam_size次
    :param sess:
    :param gen_model:
    :param vocab:
    :param source_inputs: batch-major
    :param source_outputs: batch-major
    :param encoder_inputs: time-major
    :param decoder_inputs: time-major
    :param target_weights:
    :param bucket_id:
    :param mc_search:
    :return: train_query, train_answer, train_labels 都是batch-major
    """
    gen_config = conf.gen_config
    train_query, train_answer = [], []
    query_len = gen_config.buckets[bucket_id][0]
    answer_len = gen_config.buckets[bucket_id][1]

    # set queries and answers to a respectively identical length with pad
    for query, answer in zip(source_inputs, source_outputs): # [q1, q2], [a1, a2] => [(q1, a1), (q2, a2)]
        query = query[:query_len] + [int(data_utils.PAD_ID)] * (query_len - len(query) if query_len > len(query) else 0)
        train_query.append(query)
        answer = answer[:-1] # del tag EOS
        answer = answer[:answer_len] + [int(data_utils.PAD_ID)] * (answer_len - len(answer) if answer_len > len(answer) else 0)
        train_answer.append(answer)
        train_labels = [1 for _ in source_inputs] # 1 if positive, 0 if negative

    def decoder(num_roll):
        """
        使用生成器生成负例
        :param num_roll:
        :return:
        """
        for _ in xrange(num_roll):
            # output_logits：[batch_size, target_vocab_size]
            _, _, output_logits = GenTrain().step(gen_model, sess, encoder_inputs, decoder_inputs, target_weights, bucket_id,
                                                 forward_only=True, mc_search=mc_search) # TODO：用mc_search有什么区别

            seq_tokens = []
            resps = []
            for seq in output_logits: # TODO: output_logits shape: [num_step, batch_size, target_vocab_size]？
                row_token = []
                for t in seq:
                    # t是一个一维数组，下标表示单词id，值是概率。np.argmax(t, axis=0)返回的是值最大的下标
                    row_token.append(int(np.argmax(t, axis=0)))
                seq_tokens.append(row_token)

            seq_tokens_t = [] # 转置，每一行是完整的一句话
            for col in range(len(seq_tokens[0])):
                seq_tokens_t.append([seq_tokens[row][col] for row in range(len(seq_tokens))])

            # 如果这句话有结束符，就取到结束符。最后都要截取为最长的桶长度。
            for seq in seq_tokens_t:
                if data_utils.EOS_ID in seq:
                    resps.append(seq[:seq.index(data_utils.EOS_ID)][:gen_config.buckets[bucket_id][1]])
                else:
                    resps.append(seq[:gen_config.buckets[bucket_id][1]])

            for i, output in enumerate(resps):
                # 如果output长度不够answer_len，用pad补齐
                output = output[:answer_len] + [data_utils.PAD_ID] * (answer_len - len(output) if answer_len > len(output) else 0)
                train_query.append(train_query[i])
                train_answer.append(output)
                train_labels.append(0) # 标记为负例

        return train_query, train_answer, train_labels

    if mc_search: # TODO：有没有mc_search有什么区别？为什么使用mc_search需要重复beam_size次？
        train_query, train_answer, train_labels = decoder(gen_config.beam_size)
    else:
        train_query, train_answer, train_labels = decoder(1)

    return train_query, train_answer, train_labels


def gen_pre_train():
    GenTrain().pre_train(conf.gen_config)


def gen_test():
    GenTrain().test_model(conf.gen_config)


def gen_disc():
    GenTrain().decoder(conf.gen_config)


def disc_pre_train():
    HierRNNTrain().pre_train()

def main(_):
    # step_1 training gen model
    # gen_pre_train()

    # model test
    # gen_test()

    # step_2 gen training data for disc
    # gen_disc()

    # step_3 training disc model
    # disc_pre_train()

    # step_4 training al model
    al_train()

    # model test
    # gen_test()


if __name__ == "__main__":
    tf.app.run()
