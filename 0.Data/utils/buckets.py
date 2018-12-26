import os
import sys
from tensorflow.python.platform import gfile
import utils
from utils.data_process import create_vocabulary
from utils.data_process import initialize_vocabulary
from utils.data_process import prepare_chitchat_data
from utils.conf import gen_config as conf



def read_data(config, source_path, target_path, max_size=None):
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
                target_ids.append(utils.data_process.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(config.buckets): #[bucket_id, (source_size, target_size)]
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set

def prepare_data(gen_config):
    train_path = os.path.join(gen_config.train_dir, "train")
    voc_file_path = [train_path+".dec", train_path+".enc"]
    vocab_path = os.path.join(gen_config.train_dir, "vocab%d.all" % gen_config.vocab_size)
    create_vocabulary(vocab_path, voc_file_path, gen_config.vocab_size)
    vocab, rev_vocab = initialize_vocabulary(vocab_path)

    print("Preparing Chitchat gen_data in %s" % gen_config.train_dir)
    train_query, train_answer = prepare_chitchat_data(
        gen_config.train_dir, vocab, gen_config.vocab_size)

    # Read disc_data into buckets and compute their sizes.
    print ("Reading development and training gen_data (limit: %d)."
               % gen_config.max_train_data_size)
    # dev_set = read_data(gen_config, dev_query, dev_answer)
    train_set = read_data(gen_config, train_query, train_answer, gen_config.max_train_data_size)

    return vocab, rev_vocab, train_set

if __name__ == "__main__":
    vocab, rev_vocab, train_set = prepare_data(conf)
    print(train_set[3][1])
    print(train_set[1][1])
