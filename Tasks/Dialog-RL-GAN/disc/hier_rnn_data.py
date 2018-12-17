# -*- coding: UTF-8 -*-

import os
import utils.data_utils as data_utils
from tensorflow.python.platform import gfile

def get_dataset(config):
    """
    获取训练数据
    :return: query_set, answer_set, gen_set
    """
    train_path = os.path.join(config.train_dir, "train")
    voc_file_path = [train_path + ".query", train_path + ".answer", train_path + ".gen"]
    vocab_path = os.path.join(config.train_dir, "vocab%d.all" % config.vocab_size)
    data_utils.create_vocabulary(vocab_path, voc_file_path, config.vocab_size)
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

    print("Preparing train disc_data in %s" % config.train_dir)
    train_query_path, train_answer_path, train_gen_path, dev_query_path, dev_answer_path, dev_gen_path = \
        data_utils.hier_prepare_disc_data(config.train_dir, vocab, config.vocab_size)
    query_set, answer_set, gen_set = hier_read_data(config, train_query_path, train_answer_path, train_gen_path)
    return query_set, answer_set, gen_set


def hier_read_data(config, query_path, answer_path, gen_path):
    query_set = [[] for _ in config.buckets]
    answer_set = [[] for _ in config.buckets]
    gen_set = [[] for _ in config.buckets]
    with gfile.GFile(query_path, mode="r") as query_file:
        with gfile.GFile(answer_path, mode="r") as answer_file:
            with gfile.GFile(gen_path, mode="r") as gen_file:
                query, answer, gen = query_file.readline(), answer_file.readline(), gen_file.readline()
                counter = 0
                while query and answer and gen:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  reading disc_data line %d" % counter)
                    query = [int(id) for id in query.strip().split()]
                    answer = [int(id) for id in answer.strip().split()]
                    gen = [int(id) for id in gen.strip().split()]
                    for i, (query_size, answer_size) in enumerate(config.buckets):
                        if len(query) <= query_size and len(answer) <= answer_size and len(gen) <= answer_size:
                            query = query[:query_size] + [data_utils.PAD_ID] * (query_size - len(query) if query_size > len(query) else 0)
                            query_set[i].append(query)
                            answer = answer[:answer_size] + [data_utils.PAD_ID] * (answer_size - len(answer) if answer_size > len(answer) else 0)
                            answer_set[i].append(answer)
                            gen = gen[:answer_size] + [data_utils.PAD_ID] * (answer_size - len(gen) if answer_size > len(gen) else 0)
                            gen_set[i].append(gen)
                    query, answer, gen = query_file.readline(), answer_file.readline(), gen_file.readline()

    return query_set, answer_set, gen_set
