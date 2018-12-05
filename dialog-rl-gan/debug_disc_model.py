# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import os
from hier_rnn_model import HierRNNModel


os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def main(_):
    """
    测试判别器模型：只构建图，不作真实训练，方便打断点查看图的结构
    """
    with tf.Session() as sess:
        query = [[1],[2],[3],[4],[5]]
        answer = [[6],[7],[8],[9],[0],[0],[0],[0],[0],[0]]
        target = [1]
        config = Config
        initializer = tf.random_uniform_initializer(-1 * config.init_scale, 1 * config.init_scale)

        with tf.variable_scope(name_or_scope=config.name_model, initializer=initializer):
            model = HierRNNModel(config, name_scope=config.name_model)
            sess.run(tf.global_variables_initializer())
        input_feed = {}
        for i in range(config.buckets[0][0]):
            input_feed[model.query[i].name] = query[i]
        for i in range(config.buckets[0][1]):
            input_feed[model.answer[i].name] = answer[i]
        input_feed[model.target.name] = target

        fetches = [model.b_train_op[0], model.b_query_state[0],  model.b_state[0], model.b_logits[0]]
        train_op, query, state, logits = sess.run(fetches=fetches, feed_dict=input_feed)
        print("query: ", np.shape(query))

    pass


class Config(object):
    embed_dim = 12
    lr = 0.1
    num_class = 2
    train_dir = './disc_data/'
    name_model = "disc_model_temp" # 使用一个临时的name_scope进行测试，防止训练已有模型中的参
    tensorboard_dir = "./tensorboard/disc_log/"
    name_loss = "disc_loss"
    num_layers = 3
    vocab_size = 10
    max_len = 50
    batch_size = 1
    init_scale = 0.1
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50), (50, 50)]
    max_grad_norm = 5


if __name__ == '__main__':
    tf.app.run()
