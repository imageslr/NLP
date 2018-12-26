# coding:utf-8
"""
A corpus parser for preparing data for a tensorflow chatbot
"""
import os
import random
from ast import literal_eval
from tqdm import tqdm

DELIM = '__eou__'
"""分隔符"""

class CornellMovieCorpusProcessor:

    def __init__(self, conversations='dialogues_text.txt'):
        self.movie_conversations = conversations

    def punctuation_processing(self, line):
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
        line_pro = " ".join(line.lower().split()) + "\n"
        return line_pro

    #备用版替换字符函数
    def punctuation_processing_standby_application(self, line):
        """
        1\在',', '.', '?', '!'符号前加入空格
        2\去除'[',']','...','-','<i>','</i>','<u>','</u>'
        :param line: 原始句子
        :return: line_pro 处理后的句子
        """
        punctuationlist = [',', '.', '?', '!']
        pset = set(punctuationlist)
        line_pro = ''
        for i in range(len(line)):
            if line[i] == '"':
                line_pro += (' ' + line[i] + ' ')
            elif (line[i] in pset) and i>0 and line[i-1]!='.':
                line_pro += (' ' + line[i])
            else:
                line_pro += line[i]
        return line_pro



    def get_conversations(self):
        """
        1. 读取'movie_conversations.txt'
        2. 生成对话列表[list of line_id's]
        :return: [list of line_id's]
        """
        conversation_ids_index = -1
        conversations = []
        with open(self.movie_conversations, 'r', encoding='utf-8') as f:
            for line in f:
                conversation = []
                items = line.split(DELIM)
                for item in items:
                    if item == '\n':
                        break
                    it = self.punctuation_processing(item)
                    conversation.append(it)
                conversations.append(conversation)
        return conversations


    def count_linestokens(self, line):
        """
        计算句子token个数（包括符号）
        :param line: 句子
        :return: line_num
        """
        line_num = len(line.split(' '))
        return line_num


    def generate_double_org(self, conversations, output_directory='tmp', test_set_size=3000, value = 6):
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


            for idx, item in enumerate(conversation):
                if idx % 2 == 0:
                    questions.append(item)
                else:
                    answers.append(item)
                """
                if idx == 0:
                    questions.append(item)
                elif idx == len(conversation)-1:
                    answers.append(item)
                else:
                    questions.append(item)
                    answers.append(item)
                """
        output_directory = 'dataset/dailydialog/' + output_directory
        isExists = os.path.exists(output_directory)
        if not isExists:
            os.mkdir(output_directory)
            print('Created directory successfully ', output_directory)
        else:
            print('the directory:', '/', output_directory, 'has already exited!')

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


if __name__ == '__main__':
    argsconversations = 'data/dailydialog/dialogues_text.txt'
    processor = CornellMovieCorpusProcessor(argsconversations)

    print('Collection conversations...')
    conversations = processor.get_conversations()

    result_filepaths = processor.generate_double_org(conversations, 'v1.0')
    print('Done')


