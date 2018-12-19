# -*- coding: UTF-8 -*-
__author__ = 'zhangxuri'
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
class TestUtils():
    def __init__(self,answers,gens):
        '''
        初始化，读入标准answers与生成的gens
        :param answers: 标准回答的list
        :param gens: 生成回答的list
        '''
        self.answers=answers
        self.gens=gens
        self.test=zip(answers,gens)
        return

    def get_bleu_score(self):
        '''
        根据answer与gen逐条对比计算bleu_score
        :return: bleu_score(float类型)
        '''
        totalscore=0
        for answer,gen in self.test:
            score=sentence_bleu([answer], gen, smoothing_function=SmoothingFunction().method4, auto_reweigh=True)
            totalscore=totalscore+score
        return totalscore/len(self.gens)

    def get_diversity_score(self):
        '''
        计算gens的diversity_score
        :return: diversity_score
        '''
        gen=[t.split() for t in self.gens]
        gen_flatten=[y for x in gen for y in x]
        return str(len(set(gen_flatten))/len(gen_flatten))

if __name__ == '__main__':
    #测试根目录
    train_dir="../test_data/"
    #你的测试数据的标准答案
    test_answer_dir = train_dir + "test.answer"
    #生成的答案
    gen_answer_dir = train_dir + "test.gen"
    with open(test_answer_dir) as test_answer:
        with open(gen_answer_dir) as test_gen:
            answers = [t.split('\n')[0] for t in test_answer.readlines()]
            gens = [t.split('\n')[0] for t in test_gen.readlines()]
    model = TestUtils(answers,gens)
    print(model.get_bleu_score())
    print(model.get_diversity_score())
