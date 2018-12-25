# coding:utf-8

import os
import random
from ast import literal_eval
from tqdm import tqdm

DELIM = '|'
"""分隔符"""

class OpensubtitlesProcessor:
    def __init__(self, short='s_given_t_dialogue_length2_3.txt', long='s_given_t_dialogue_length2_6.txt', dict ='dictionarie.txt'):
        self.short_filepath = short
        self.long_filepath = long
        self.dic_path = dict


    def get_convercations(self):
        questions_id = []
        answers_id = []
        with open(self.short_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                items = line.split(DELIM)
                questions_id.append(items[0])
                answers_id.append(items[1])
        with open(self.long_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                items = line.split(DELIM)
                questions_id.append(items[0])
                answers_id.append(items[1])
        return questions_id, answers_id


    def creatdic(self):
        dic = {}
        lineid = 1
        with open(self.dic_path, 'r', encoding='utf-8') as f:
            for line in f:
                dic[lineid] = ''.join(line.split())
                lineid+=1
        return dic

    def id2word(self, questions_id, answers_id):
        questions = []
        answers = []
        dic = self.creatdic()
        for line in questions_id:
            items = line.split(' ')
            words = ""
            for i in items:
                if i != '\n':
                    words += (dic[int(i)]+' ')

            words+='\n'
            questions.append(words)
        for line in answers_id:
            items = line.split(' ')
            words = ""
            for i in items:
                if i != '\n':
                    words += (dic[int(i)]+' ')

            words += '\n'
            answers.append(words)
        return questions, answers

    def output_file(self, questions,answers, output_directory='dataset/opensubtitle/v1.0', test_set_size=30000 ):
        isExists = os.path.exists(output_directory)
        if not isExists:
            os.mkdir(output_directory)
            print('Created directory successfully: ', '//', output_directory)
        else:
            print('the directory:', '//', output_directory, 'has already exited!')
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
        train_enc.close()
        train_dec.close()
        return train_enc_filepath, train_dec_filepath


if __name__ == '__main__':
    argsshort = 'data/OpenSubtitles/s_given_t_dialogue_length2_3.txt'
    argslong = 'data/OpenSubtitles/s_given_t_dialogue_length2_6.txt'
    argsdic = 'data/OpenSubtitles/dictionarie.txt'
    processor = OpensubtitlesProcessor(argsshort, argslong, argsdic)
    print('Get convercations Processing...')
    que_id,ans_id = processor.get_convercations()
    print('Id to Words Processing...')
    que, ans = processor.id2word(que_id, ans_id)
    processor.output_file(que, ans, output_directory='dataset/opensubtitle/v1.0', test_set_size=30000 )
