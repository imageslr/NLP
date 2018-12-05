# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
import os
from gen_model import GenModel
from six.moves import xrange


os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def main(_):
    """
    测试生成器模型：只构建图，不作真实训练，方便打断点查看图的结构
    """
    with tf.Session() as sess:
        query = [[1],[2],[3],[4],[5]]
        answer = [[6],[7],[8],[9],[0],[0],[0],[0],[0],[0]]
        target = [1]
        reward = 1
        bucket_id = 0
        config = Config
        encoder_size, decoder_size = config.buckets[bucket_id]
        initializer = tf.random_uniform_initializer()

        with tf.variable_scope(name_or_scope=config.name_model, initializer=initializer):
            model = GenModel(config, name_scope=config.name_model, num_samples=1)
            sess.run(tf.global_variables_initializer())

        input_feed = {
            model.forward_only.name: False,
            model.up_reward.name: False,
            model.mc_search.name: False
        }


        for l in xrange(len(model.buckets)):
            input_feed[model.reward[l].name] = reward
        for l in xrange(encoder_size):
            input_feed[model.encoder_inputs[l].name] = query[l]
        for l in xrange(decoder_size):
            input_feed[model.decoder_inputs[l].name] = answer[l]
            input_feed[model.target_weights[l].name] = 1
        last_target = model.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([model.batch_size], dtype=np.int32)


        # Output feed: depends on whether we do a backward step or not.
        # normal training
        output_feed = [model.updates[bucket_id],  # Update Op that does SGD.
                       model.aj_losses[bucket_id],  # Gradient norm.
                       model.losses[bucket_id]]  # Loss for this batch.

        outputs = sess.run(output_feed, input_feed)
        return outputs[1], outputs[2], outputs[0]  # Gradient norm, loss, no outputs.

    pass


class Config(object):
    emb_dim = 1
    train_dir = './gen_data/'
    name_model = "gen_model_temp" # 使用一个临时的name_scope进行测试，防止训练已有模型中的参数
    tensorboard_dir = "./tensorboard/gen_log/"
    name_loss = "gen_loss"
    num_layers = 2
    vocab_size = 9
    batch_size = 1
    beam_size = 3
    learning_rate = 0.5
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    teacher_loss = "teacher_loss"
    reward_name = "reward"
    max_train_data_size = 0
    steps_per_checkpoint = 100
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


if __name__ == '__main__':
    tf.app.run()
