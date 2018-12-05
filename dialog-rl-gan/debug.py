# -*- coding: UTF-8 -*-
# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
import os
from six.moves import xrange
import utils.conf as conf
from gen.gen_train import GenTrain

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def main(_):
    """
    测试生成器的预训练
    """
    train_obj = GenTrain()

    train_obj.pre_train(gen_config)

    pass


class gen_config(object):
    max_pre_train_step = 300 # 预训练的时候最多训练多少次
    beam_size = 2
    learning_rate = 0.5
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    batch_size = 1
    emb_dim = 12
    num_layers = 2
    vocab_size = 300 # 这里一定要比 num_samples=256 大
    train_dir = "./gen_data/"
    name_model = "gen_model_temp" # 使用一个临时的name_scope进行测试，防止训练已有模型中的参数
    tensorboard_dir = "./tensorboard/gen_log/"
    name_loss = "gen_loss"
    teacher_loss = "teacher_loss"
    reward_name = "reward"
    max_train_data_size = 0
    steps_per_checkpoint = 100
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
    buckets_concat = [(5, 10), (10, 15), (20, 25), (40, 50), (100, 50)]


if __name__ == '__main__':
    tf.app.run()
