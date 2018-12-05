# coding=utf-8
import numpy as np 
import tensorflow as tf 
import matplotlib as mpl 
mpl.use('Agg')
import matplotlib.pyplot as plt 
import tqdm
import os
import _pickle as cpickle # python3.x中cpickle改为_pickle
from nltk.translate.bleu_score import sentence_bleu


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

##########hyper-parameters##########
# Batch Size
batch_size = 512
max_source_sentence_length = 20
max_target_sentence_length = 25

##########BLEU###########

# 翻译结果
loaded_graph = tf.Graph()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph('checkpoints/dev.meta')
    loader.restore(sess, tf.train.latest_checkpoint('./checkpoints'))

    input_data = loaded_graph.get_tensor_by_name('inputs:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_len:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_len:0')

    # 存储每个句子的模型翻译结果
    fr_preds = []

    # 对样本中的每个英文进行翻译
    for i in tqdm.tqdm(range(len(source_text_to_int)//batch_size)):
        batch_sentences = source_text_to_int[batch_size*i : batch_size*(i+1)]
        translate_logits = sess.run(logits, {input_data: batch_sentences,
				             target_sequence_length: [max_target_sentence_length]*batch_size,
				             source_sequence_length: [max_source_sentence_length]*batch_size})
        for j in range(batch_size):
            fr_preds.append(" ".join([target_int_to_vocab[word_idx] for word_idx in translate_logits[j][::-1]]))
 
    # 存储每个句子的BLEU分数
    bleu_score = []
    references = target_text.split('\n')
    for k in tqdm.tqdm(range(len(fr_preds))):
        # 去掉特殊字符
        pred = fr_preds[k].replace("<EOS>", "").replace("<PAD>", "").lstrip()
        reference = references[k].lower()
        if k%10000 == 0:
            print("pred:", pred)
            print("reference:", reference)
        # 计算BLEU分数
        score = sentence_bleu([reference.split()], pred.split(), weights=[0.25,0.25,0.25,0.25])
        bleu_score.append(score)
    print("The BLEU score on our corpus is about {}".format(sum(bleu_score) / len(bleu_score)))
