# -*- coding: utf-8 -*-
# @Time    : 2019/1/22 14:33
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : data_helper.py
# @Software: PyCharm


import sys
import numpy as np
import random
from collections import namedtuple
import pickle

random.seed(1337)
np.random.seed(1337)

def load_embedding(dstPath):
    with open(dstPath, 'rb') as fin:
        _embeddings = pickle.load(fin)
    print("load embedding finish! embedding shape:{}".format(np.shape(_embeddings)))
    return _embeddings

def load_data(f):
    data = pickle.load(open(f, 'rb'))
    question = data[0]
    answer = data[1]
    label = data[2]
    question_mask = data[3]
    answer_mask = data[4]
    return question, answer, label, question_mask, answer_mask



class Batch:
    # batch类，里面包含了encoder输入，decoder输入，decoder标签，decoder样本长度mask
    def __init__(self):
        self.quest = []
        self.ans = []
        self.quest_mask = []
        self.ans_mask = []
        self.label = []


def get_train_Batches(quest, ans, quest_mask, ans_mask, label, batch_size):
    # 每个epoch之前都要进行样本的shuffle
    data_len = len(quest)
    batch_num = int((data_len-1) / batch_size) + 1
    shuffle_idx = np.random.permutation(np.arange(data_len))
    quest = np.array(quest)[shuffle_idx]
    quest_mask = np.array(quest_mask)[shuffle_idx]
    ans = np.array(ans)[shuffle_idx]
    ans_mask = np.array(ans_mask)[shuffle_idx]
    label = np.array(label)[shuffle_idx]
    batches = []
    for batch in range(batch_num):
        min_batch = Batch()
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, data_len)
        min_batch.quest.extend(quest[start_idx: end_idx])
        min_batch.quest_mask.extend(quest_mask[start_idx: end_idx])
        min_batch.ans.extend(ans[start_idx: end_idx])
        min_batch.ans_mask.extend(ans_mask[start_idx: end_idx])
        min_batch.label.extend(label[start_idx: end_idx])
        batches.append(min_batch)
    # print("batches_len:{}".format(len(batches)))
    return batches

def get_test_Batches(quest, ans, label, quest_mask, ans_mask, batch_size):
    data_len = len(quest)
    batch_num = int((data_len-1) / batch_size) + 1
    for batch in range(batch_num):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, data_len)
        yield (quest[start_idx:end_idx], ans[start_idx:end_idx], label[start_idx:end_idx],
               quest_mask[start_idx:end_idx], ans_mask[start_idx:end_idx])



if __name__ == '__main__':
    train_file = './data/wikiqa/self/raw/pre/pre_train.pkl'
    dev_file = './data/wikiqa/self/raw/pre/pre_dev.pkl'
    test_file = './data/wikiqa/self/raw/pre/pre_test.pkl'
    embed = './data/wikiqa/self/raw/wiki_embedding.pkl'
    load_embedding(embed)

