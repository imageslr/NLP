# coding:utf-8
import os
from tqdm import tqdm
import re
import random

_emoji_RE = re.compile(u"[\U00010000-\U0010ffff]")
_theme_RE = re.compile("#.*?#")
_char_RE = re.compile("[^\u4e00-\u9fa5a-zA-Z\d,\,.,。，？！]+")



def filter_line(line):
    """
    清洗符号
    :param line:
    :return: line_pro
    """
    l = _emoji_RE.sub("", line)
    l = _theme_RE.sub("", l)
    l = _char_RE.split(l)
    line_pro = ' '.join(l)
    line_pro = line_pro + '\n'
    return line_pro

def get_line(filepath="data/weibo/"):
    que_path = os.path.join(filepath,"stc_weibo_train_post")
    ans_path = os.path.join(filepath,"stc_weibo_train_response")
    questions = []
    answers = []
    q_file = open(que_path, 'r', encoding='utf-8')
    a_file = open(ans_path, 'r', encoding='utf-8')
    for que,ans in zip (q_file,a_file):
        questions.append(filter_line(que))
        answers.append(filter_line(ans))
    return questions,answers

def output_file(questions, answers, output_directory='twitter/twitter_v1.0', test_set_size=30000):
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
    print('Outputting train/test enc/dec files...')
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


def process_data():
    file_path = "data/weibo/"
    print('Read lines from file')
    questions, answers = get_line(file_path)
    output_file(questions, answers, output_directory='dataset/weibo/v0.0')
    print('Done!')

if __name__ == '__main__':
    process_data()

