# coding:utf-8
"""
A corpus parser for preparing data for a tensorflow chatbot
"""
import os
import random
from ast import literal_eval
from tqdm import tqdm

DELIM = ' +++$+++ '
"""分隔符"""
SIZE = 3000
movie_lines_filepath = 'data/cornell-movie-dialogs-corpus/movie_lines.txt'
movie_conversations = 'data/cornell-movie-dialogs-corpus/movie_conversations.txt'

def punctuation_processing(line):
    """
    1\在',', '.', '?', '!'符号前加入空格
    2\去除'[',']','...','-','<i>','</i>','<u>','</u>'
    3\全部转换为小写字母
    :param line: 原始句子
    :return: line_pro 处理后的句子
    """
    replace_list = ["..", "...", "-", "[", "]", "<i>", "</i>", "<u>", "</u>",
                    "<b>", "</b>", "<U>", "</U>", "<", ">", "{", "}"]
    for replace_string in replace_list:
        line = line.replace(replace_string, " ")
        line = line.replace(".", " . ")
        line = line.replace(",", " , ")
        line = line.replace("!", " !")
        line = line.replace("?", " ?")
        line = line.replace('"', '')
        line_pro = " ".join(line.lower().split()) + "\n"
    return line_pro


def get_id2line():
    """
    1. 读取 'movie-lines.txt'
    2. 构建 line_id 和 text对应关系的词典( key = line_id, value = text )
    :return: (dict) {line-id: text, ...}
    """
    id2line = {}
    id_index = 0
    text_index = 4
    with open(movie_lines_filepath, 'r', encoding= 'utf-8') as f:
        for line in f:
            items = line.split(DELIM)
            if len(items) == 5:
                line_id = items[id_index]
                dialog_text = punctuation_processing(items[text_index])
                id2line[line_id] = dialog_text
    return id2line

def get_conversations():
    """
    1. 读取'movie_conversations.txt'
    2. 生成对话列表[list of line_id's]
    :return: [list of line_id's]
    """
    conversation_ids_index = -1
    conversations = []
    with open(movie_conversations, 'r') as f:
        for line in f:
            items = line.split(DELIM)
            conversation_ids_field = items[conversation_ids_index]
            conversation_ids = literal_eval(conversation_ids_field)  # evaluate as a python list
            conversations.append(conversation_ids)
    return conversations


def count_linestokens(line):
    """
    计算句子token个数（包括符号）
    :param line: 句子
    :return: line_num
    """
    line_num = len(line.split(' '))
    return line_num


def judge(line1, line2):
    """
    判断对话中的行是否为空
    :param line1:
    :param line2:
    :return:
    """
    if (len(line1) < 2 or len(line2) < 2):
        return True
    else:
        return False


def generate_double(id2line, conversations, output_directory='tmp', test_set_size=3000):
    """
    生成二元组对话文件
    :param conversations: (list) Collection line ids consisting of a single conversation
    :param id2line: (dict) mapping of line-ids to actual line text
    :param output_directory: (str) Directory to write files
    :param test_set_size: (int) number of samples to use for test data set 测试集大小 默认为30000
    :return:train_enc_filepath, train_dec_filepath, test_enc_filepath, test_dec_filepath 路径
    """
    questions = []
    answers = []
    for conversation in conversations:

        if len(conversation) % 2 != 0:
            conversation = conversation[:-1]  # remove last item
        for idx, line_id in enumerate(conversation):
            if idx % 2 == 0:
                questions.append(id2line[line_id])
            else:
                answers.append(id2line[line_id])
        """
        for idx, line_id in enumerate(conversation):
            if idx == 0:
                questions.append(id2line[line_id])
            elif idx == len(conversation)-1:
                answers.append(id2line[line_id])
            else:
                questions.append(id2line[line_id])
                answers.append(id2line[line_id])
        """
    questions_tmp = []
    answers_tmp = []
    print('Processing replace blank line')
    for i in range(len(questions)):
        if judge(questions[i], answers[i]):
            continue
        questions_tmp.append(questions[i])
        answers_tmp.append(answers[i])

    questions = questions_tmp
    answers = answers_tmp
    #创建文件目录
    output_directory = 'dataset/cornell/' + output_directory
    isExists = os.path.exists(output_directory)
    if not isExists:
        os.mkdir(output_directory)
        print('Created directory successfully ', output_directory)
    else:
        print('the directory:', '//', output_directory, 'has already exited!')

    #输出
    train_enc_filepath = os.path.join(output_directory, 'train.enc')
    train_dec_filepath = os.path.join(output_directory, 'train.dec')
    test_enc_filepath = os.path.join(output_directory, 'test.enc')
    test_dec_filepath = os.path.join(output_directory, 'test.dec')
    train_enc = open(train_enc_filepath, 'w', encoding='utf-8')
    train_dec = open(train_dec_filepath, 'w', encoding='utf-8')
    test_enc = open(test_enc_filepath, 'w', encoding='utf-8')
    test_dec = open(test_dec_filepath, 'w', encoding='utf-8')

    # choose test_set_size number of items to put into testset
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

    return train_enc_filepath, train_dec_filepath, test_enc_filepath, test_dec_filepath

def generate_triple(id2line, conversations, output_directory='triple', test_set_size=3000):
    """
    生成三元组对话文件
    :param conversations: (list) Collection line ids consisting of a single conversation
    :param id2line: (dict) mapping of line-ids to actual line text
    :param output_directory: (str) Directory to write files
    :param test_set_size: (int) number of samples to use for test data set 测试集大小 默认为3000
    :return:train_enc1_filepath, train_enc2_filepath, train_dec_filepath, test_enc1_filepath,
    test_enc2_filepath, test_dec_filepath 路径
    """
    first = []
    second = []
    third = []
    #
    for conversation in conversations:
        ConversationLength = len(conversation)
        if ConversationLength >= 3:
            for idx, line_id in enumerate(conversation):
                if idx == 0:
                    first.append(id2line[line_id])
                elif idx == ConversationLength - 1:
                    third.append(id2line[line_id])
                elif ConversationLength == 3:
                    second.append(id2line[line_id])
                elif idx == 1:
                    first.append(id2line[line_id])
                    second.append(id2line[line_id])
                elif idx == ConversationLength - 2:
                    second.append(id2line[line_id])
                    third.append(id2line[line_id])
                else:
                    first.append(id2line[line_id])
                    second.append(id2line[line_id])
                    third.append(id2line[line_id])

    isExists = os.path.exists(output_directory)
    if not isExists:
        os.mkdirs(output_directory)
        print('Created directory successfully: ', '//' , output_directory)
    else:
        print('the directory:','//', output_directory, 'has already exited!')
    train_enc1_filepath = os.path.join(output_directory, 'train.enc1')
    train_enc2_filepath = os.path.join(output_directory, 'train.enc2')
    train_dec_filepath = os.path.join(output_directory, 'train.dec')
    test_enc1_filepath = os.path.join(output_directory, 'test.enc1')
    test_enc2_filepath = os.path.join(output_directory, 'test.enc2')
    test_dec_filepath = os.path.join(output_directory, 'test.dec')

    train_enc1 = open(train_enc1_filepath, 'w', encoding='utf-8')
    train_enc2 = open(train_enc2_filepath, 'w', encoding='utf-8')
    train_dec = open(train_dec_filepath, 'w', encoding='utf-8')
    test_enc1 = open(test_enc1_filepath, 'w', encoding='utf-8')
    test_enc2 = open(test_enc2_filepath, 'w', encoding='utf-8')
    test_dec = open(test_dec_filepath, 'w', encoding='utf-8')

    # choose test_set_size number of items to put into testset
    test_ids = random.sample(range(len(first)), test_set_size)
    print('Outputting train/test enc/dec files...')
    for i in tqdm(range(len(first))):
        if i in test_ids:
            test_enc1.write(first[i])
            test_enc2.write(second[i])
            test_dec.write(third[i])
        else:
            train_enc1.write(first[i])
            train_enc2.write(second[i])
            train_dec.write(third[i])

    # close files
    train_enc1.close()
    train_enc2.close()
    train_dec.close()
    test_enc1.close()
    test_enc2.close()
    test_dec.close()

    return train_enc1_filepath, train_enc2_filepath, train_dec_filepath, \
           test_enc1_filepath, test_enc2_filepath, test_dec_filepath


if __name__ == '__main__':
    import argparse
    """
    parser = argparse.ArgumentParser(description=__doc__)
    DEFAULT_OUTPUT_DIRECTORY = 'cornell'
    parser.add_argument('-l', '--lines',
                        default='data/cornell-movie-dialogs-corpus/movie_lines.txt',
                        help='Path to Cornell Corpus, "movie_lines.txt"')
    parser.add_argument('-c', '--conversations',
                        default='data/cornell-movie-dialogs-corpus/movie_conversations.txt',
                        help='Path to Cornell Corpus, "movie_conversations.txt"')
    parser.add_argument('-o', '--output_directory',
                        dest='output_directory',
                        default=DEFAULT_OUTPUT_DIRECTORY,
                        help='Output directory for train/test data [DEFAULT={}]'.format(DEFAULT_OUTPUT_DIRECTORY))
    parser.add_argument('-s', '--size',
                        default=3000,
                        help='Size of test set')
    args = parser.parse_args()
    """


    print('Collection line-ids...')
    id2lines = get_id2line()
    print('Collection conversations...')
    conversations = get_conversations()

    result_filepaths = generate_double(id2lines, conversations, 'v1.1', SIZE)
    print('Done')


