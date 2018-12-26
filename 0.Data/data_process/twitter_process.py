# coding:utf-8
EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz,.?!\' '  # space is included in whitelist
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''

FILENAME = 'data/twitter/chat.txt'

limit = {
    'maxq': 30,
    'minq': 0,
    'maxa': 30,
    'mina': 3
}

UNK = 'unk'
VOCAB_SIZE = 6000

import os
from tqdm import tqdm
import random

def read_lines(filename):
    return open(filename, encoding='utf-8').read().split('\n')[:-1]

#判断空行，是空行为true
def judge(line1, line2):
    if (len(line1) < 2 or len(line2) < 2):
        return True
    else:
        return False


def filter_line(line, whitelist):
    line_pre= ''.join([ch for ch in line if ch in whitelist])
    line = line_pre
    replace_list = ["..", "..."]
    for replace_string in replace_list:
        line = line.replace(replace_string, " ")
    line = line.replace(".", " . ")
    line = line.replace(",", " , ")
    line = line.replace("!", " !")
    line = line.replace("?", " ?")
    line = line.replace('"', '')
    line_pro = " ".join(line.lower().split())
    line_pro += '\n'
    return line_pro


def output_file(questions, answers, output_directory='dataset/twitter/v1.0/', test_set_size= 3000):
    isExists = os.path.exists(output_directory)
    if not isExists:
        os.mkdir(output_directory)
        print('Created directory successfully: ', '//', output_directory)
    else:
        print('the directory:', '//', output_directory, 'has already exited!')

    #读写文件
    train_enc_filepath = os.path.join(output_directory, 'train.enc')
    train_dec_filepath = os.path.join(output_directory, 'train.dec')
    test_enc_filepath = os.path.join(output_directory, 'test.enc')
    test_dec_filepath = os.path.join(output_directory, 'test.dec')
    train_enc = open(train_enc_filepath, 'w', encoding='utf-8')
    train_dec = open(train_dec_filepath, 'w', encoding='utf-8')
    test_enc = open(test_enc_filepath, 'w', encoding='utf-8')
    test_dec = open(test_dec_filepath, 'w', encoding='utf-8')

    test_ids = random.sample(range(len(questions)), test_set_size)
    print('Outputting train/test enc/dec files...')
    for i in tqdm(range(len(questions))):
        if i in test_ids:
            test_enc.write(questions[i])
            test_dec.write(answers[i])
        else:
            train_enc.write(questions[i])
            train_dec.write(answers[i])
    # close files
    train_enc.close()
    train_dec.close()
    test_enc.close()
    test_dec.close()
    return train_enc_filepath, train_dec_filepath


def punctuation_processing(line):
    """
    1\在',', '.', '?', '!'符号前加入空格
    2\去除'[',']','...','-','<i>','</i>','<u>','</u>'
    3\全部转换为小写字母
    :param line: 原始句子
    :return: line_pro 处理后的句子
    """
    replace_list = ["..", "...", "-", "[", "]", "<i>", "</i>", "<u>", "</u>"]
    for replace_string in replace_list:
        line = line.replace(replace_string, " ")
    line = line.replace(".", " . ")
    line = line.replace(",", " , ")
    line = line.replace("!", " !")
    line = line.replace("?", " ?")
    line = line.replace('"', '')
    line_pro = " ".join(line.lower().split())
    return line_pro


def filter_data(sequences):
    filtered_q, filtered_a = [], []
    raw_data_len = len(sequences) // 2

    for i in range(0, len(sequences), 2):
        qlen, alen = len(sequences[i].split(' ')), len(sequences[i + 1].split(' '))
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa']:
                if judge(sequences[i], sequences[i+1]):
                    continue
                filtered_q.append(sequences[i])
                filtered_a.append(sequences[i + 1])

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len) * 100 / raw_data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a


def process_data():
    print('\n>> Read lines from file')
    lines = read_lines(filename=FILENAME)

    # 全部变成小写
    lines = [line.lower() for line in lines]

    # 去掉非英文字符
    lines = [filter_line(line, EN_WHITELIST) for line in lines]
    #拆分成对话段
    questions, answers = filter_data(lines)
    #输出文件
    output_file(questions, answers)
    print('Done!')

if __name__ == '__main__':
    process_data()

