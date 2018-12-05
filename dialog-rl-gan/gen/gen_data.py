# -*- coding: UTF-8 -*-
import os
import random
import sys
from six.moves import xrange
import numpy as np
from tensorflow.python.platform import gfile
import utils.data_utils as data_utils
from utils.utils import just_message as just


def get_dataset(gen_config):
    """
    获取训练数据
    :return: vocab, rev_vocab, dev_set, train_set
    """
    train_path = os.path.join(gen_config.train_dir, "chitchat.train")
    voc_file_path = [train_path + ".answer", train_path + ".query"]
    vocab_path = os.path.join(gen_config.train_dir, "vocab%d.all" % gen_config.vocab_size)
    data_utils.create_vocabulary(vocab_path, voc_file_path, gen_config.vocab_size)
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)  # {dog: 0, cat: 1} [dog, cat]

    print(just("Preparing Chitchat gen_data in %s" % gen_config.train_dir))
    train_query, train_answer, dev_query, dev_answer = data_utils.prepare_chitchat_data(
        gen_config.train_dir, vocab, gen_config.vocab_size)

    # Read disc_data into buckets and compute their sizes.
    print(just("Reading development and training gen_data (limit: %d)."
          % gen_config.max_train_data_size))
    dev_set = read_data(gen_config, dev_query, dev_answer)
    train_set = read_data(gen_config, train_query, train_answer, gen_config.max_train_data_size)

    return vocab, rev_vocab, dev_set, train_set


def read_data(config, source_path, target_path, max_size=None):
    """
    读取数据，读的时候按照源序列与目标序列的长度分桶
    返回值：dataset: [bucket1, bucket2, ...]，每个bucket是一个list，list中每个元素是一个[seq_ids, target_ids]对
    :param config:
    :param source_path:
    :param target_path:
    :param max_size:
    :return:
    """
    data_set = [[] for _ in config.buckets]
    with gfile.GFile(source_path, mode="r") as source_file:
        with gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading disc_data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(config.buckets): #[bucket_id, (source_size, target_size)]
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set


def get_batch(model, train_data, bucket_id, batch_size, type=0):
    """
    从一个指定的桶中获取一个随机的batch，用于step(..)的训练。step(..)接受的数据是time-major的
    Args:
      train_data: 大小是分桶个数的列表，每个元素是一个列表，由(Q, A)对组成
      bucket_id: integer, which bucket to get the batch for.
      type: - 0：正常的获取预训练用的数据
            - 1：TODO 好像没用
            - 2：对抗训练中判别器的训练数据，这个也好像没用
    Returns:
      (batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_source_encoder, batch_source_decoder)
      依次是(time-major，time-major，time-major, batch-major, batch-major)
    """

    encoder_size, decoder_size = model.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data, 用到的type都是0，都是从桶里随机挑一组
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    batch_source_encoder, batch_source_decoder = [], []

    if type == 1:
        batch_size = 1
    for batch_i in xrange(batch_size):
        if type == 1:  # 返回桶内所有数据
            encoder_input, decoder_input = train_data[bucket_id]
        elif type == 2:  # 取桶内第一组，encoder_input是第batch_i个单词，encoder只有一个单词 # TODO 但下面是把它当数组用的，这里就报错了
            # print("disc_data[bucket_id]: ", disc_data[bucket_id][0])
            encoder_input_a, decoder_input = train_data[bucket_id][0]
            encoder_input = encoder_input_a[batch_i]
        elif type == 0:  # 桶内随机挑一组
            encoder_input, decoder_input = random.choice(train_data[bucket_id])
            # print("train en: %s, de: %s" %(encoder_input, decoder_input))

        batch_source_encoder.append(encoder_input)
        batch_source_decoder.append(decoder_input)
        # Encoder inputs are padded and then reversed.
        encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
        encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

        # Decoder inputs get an extra "GO" symbol, and are padded then.
        decoder_pad_size = decoder_size - len(decoder_input) - 1
        decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                              [data_utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the disc_data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
        batch_encoder_inputs.append(
            np.array([encoder_inputs[batch_idx][length_idx]
                      for batch_idx in xrange(batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
        batch_decoder_inputs.append(
            np.array([decoder_inputs[batch_idx][length_idx]
                      for batch_idx in xrange(batch_size)], dtype=np.int32))

        # Create target_weights to be 0 for targets that are padding.
        batch_weight = np.ones(batch_size, dtype=np.float32)
        for batch_idx in xrange(batch_size):
            # We set weight to 0 if the corresponding target is a PAD symbol. 如果是PAD，设置它的权值为0
            # The corresponding target is decoder_input shifted by 1 forward.
            if length_idx < decoder_size - 1:
                target = decoder_inputs[batch_idx][length_idx + 1]
            if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                batch_weight[batch_idx] = 0.0
        batch_weights.append(batch_weight)

    return (batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_source_encoder, batch_source_decoder)