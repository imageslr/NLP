# coding=utf-8

import numpy as np
import tqdm
import _pickle as cpickle # python3.x中cpickle改为_pickle

# English source data 
with open('data/small_vocab_en', 'r', encoding='utf-8') as f:
    source_text = f.read()
# French target data 
with open('data/small_vocab_fr', 'r', encoding='utf-8') as f:
    target_text = f.read()

# 下面是对原始文本按照空格分开，这样就可以查看原始文本到底包含了多少个单词
view_sentence_range = (0, 10)
print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word : None for word in source_text.split()})))

# 按照换行符将原始文本分割成句子
print('-'*5 + 'English Text' + '-'*5)
sentences = source_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]
print('Number of sentences: {}'.format(len(sentences)))
print('Average number of words in a sentence: {}'.format(np.average(word_counts)))
print('Max number of words in a sentence: {}'.format(np.max(word_counts)))

print()
print("-"*5 + "French Text" + "-"*5)
sentences = target_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]
print('Number of sentences: {}'.format(len(sentences)))
print('Average number of words in a sentence: {}'.format(np.average(word_counts)))
print('Max number of words in a sentence: {}'.format(np.max(word_counts)))

print()
print('English sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(source_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
print()
print('French sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(target_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))

# 构造英文词典
source_vocab = list(set(source_text.lower().split()))
# 构造法文词典
target_vocab = list(set(target_text.lower().split()))

print("The size of English vocab is : {}".format(len(source_vocab)))
print("The size of French vocab is : {}".format(len(target_vocab)))

# 特殊字符
SOURCE_CODES = ['<PAD>', '<UNK>']
TARGET_CODES = ['<PAD>', '<EOS>', '<UNK>', '<GO>']  # 在target中，需要增加<GO>与<EOS>特殊字符

# 构造英文映射字典
source_vocab_to_int = {word: idx for idx, word in enumerate(SOURCE_CODES + source_vocab)}
source_int_to_vocab = {idx: word for idx, word in enumerate(SOURCE_CODES + source_vocab)}

# 构造法语映射词典
target_vocab_to_int = {word: idx for idx, word in enumerate(TARGET_CODES + target_vocab)}
target_int_to_vocab = {idx: word for idx, word in enumerate(TARGET_CODES + target_vocab)}

print("The size of English Map is : {}".format(len(source_vocab_to_int)))
print("The size of French Map is : {}".format(len(target_vocab_to_int)))

def text_to_int(sentence, map_dict, max_length=20, is_target=False):
    """
    对文本句子进行数字编码

    @param sentence: 一个完整的句子，str类型
    @param map_dict: 单词到数字的映射，dict
    @param max_length: 句子的最大长度
    @param is_target: 是否为目标语句。在这里要区分目标句子与源句子，因为对于目标句子（即翻译后的句子）我们需要在句子最后增加<EOS>
    """

    # 用<PAD>填充整个序列
    text_to_idx = []
    # unk index
    unk_idx = map_dict.get("<UNK>")
    pad_idx = map_dict.get("<PAD>")
    eos_idx = map_dict.get("<EOS>")

    # 如果是输入源文本
    if not is_target:
        for word in sentence.lower().split():
            text_to_idx.append(map_dict.get(word, unk_idx))

    # 否则，对于输出目标文本需要做<EOS>的填充最后
    else:
        for word in sentence.lower().split():
            text_to_idx.append(map_dict.get(word, unk_idx))
        text_to_idx.reverse()

    # 如果超长需要截断
    if len(text_to_idx) > max_length:
        return text_to_idx[:max_length]
    # 如果不够则增加<PAD>
    else:
        text_to_idx = text_to_idx + [pad_idx] * (max_length - len(text_to_idx))
        return text_to_idx

# 对源句子进行转换 Tx = 20
source_text_to_int = []
for sentence in tqdm.tqdm(source_text.split("\n")):
    source_text_to_int.append(text_to_int(sentence, source_vocab_to_int, 20, 
                                          is_target=False))

# 对目标句子进行转换  Ty = 25
target_text_to_int = []
for sentence in tqdm.tqdm(target_text.split("\n")):
    target_text_to_int.append(text_to_int(sentence, target_vocab_to_int, 25, 
                                          is_target=True))

random_index = 77

print("-"*5 + "English example" + "-"*5)
print(source_text.split("\n")[random_index])
print(source_text_to_int[random_index])

print()
print("-"*5 + "French example" + "-"*5)
print(target_text.split("\n")[random_index])
print(target_text_to_int[random_index])

# source_text target_text 
# source_vocab_to_int source_int_to_vocab 
# target_vocab_to_int target_int_to_vocab 
# source_text_to_int target_text_to_int 
f = open('dataFile.txt', 'wb')
cpickle.dump([source_text, target_text], f, True)
cpickle.dump([source_vocab_to_int, source_int_to_vocab], f, True)
cpickle.dump([target_vocab_to_int, target_int_to_vocab], f, True)
cpickle.dump([source_text_to_int, target_text_to_int], f, True)
f.close()

f = open('dataFile.txt', 'rb')
text = cpickle.load(f)
source_text = text[0]
target_text = text[1]
source = cpickle.load(f)
source_vocab_to_int = source[0]
source_int_to_vocab = source[1]
target = cpickle.load(f)
target_vocab_to_int = target[0]
target_int_to_vocab = target[1]
text_to_int = cpickle.load(f)
source_text_to_int = text_to_int[0]
target_text_to_int = text_to_int[1]
f.close()
# coding=utf-8
