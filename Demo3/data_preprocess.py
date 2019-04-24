# -*- coding: utf-8 -*-
# @Time    : 2019/1/23 14:35
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : data_preprocess.py
# @Software: PyCharm


import sys
import numpy as np
import random
from collections import namedtuple
import pickle
import time

random.seed(1337)
np.random.seed(1337)

ModelParam = namedtuple("ModelParam",
                        "hidden_dim,enc_timesteps,dec_timesteps,batch_size,random_size,k_value_ques,k_value_ans,lr")

UNKNOWN_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'

class DataGenerator(object):
    """Dataset class
    """

    def __init__(self, vocab, model_param, answer_file=""):
        self.vocab = vocab
        self.param = model_param
        self.corpus_amount = 0
        if answer_file != "":
            self.answers = pickle.load(open(answer_file, 'rb'))  # Bad Answer, Type: dic
            print(self.answers.values())

    def padq(self, data):
        return self.pad_q(data, self.param.ques_len)

    def pada(self, data):
        return self.pad_a(data, self.param.ans_len)

    def pad_q(self, data, lens=None):
        def _pad(seq, lens):
            if (len(seq) >= lens):
                return seq[:lens], lens
            else:
                return seq + [0 for i in range(lens - len(seq))], len(seq)
        # return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)
        return map(lambda x: _pad(x, lens), data)

    def pad_a(self, data, lens=None):
        def _pad(seq, lens):
            if (len(seq) >= lens):
                return seq[:lens]
            else:
                return seq + [0 for i in range(lens - len(seq))]
        # return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)
        return map(lambda x: _pad(x, lens), data)

    def cut(self, x, y):
        if x > y:
            return y
        else:
            return x

    def wikiQaGenerate(self, filename, out_file, flag="basic", verbose=True):
        # data Type: [([511, 18], [64, 23637, 11818, 44, 23638, 30, 19447, 4486, 28657, 14], 0),(..),..]
        data = pickle.load(open(filename, 'rb'))
        print(data[:2])
        print('raw_data:{}'.format(len(data)))
        question_dic = {}
        question = list()
        question_mask = list()
        answer = list()
        answer_mask = []
        label = list()
        for item in data:
            # setdefault: 如果字典中包含有给定键，则返回该键对应的值，否则返回为该键设置的值。
            # 将同一问题的答案和标签收集到一起
            question_dic.setdefault(str(item[0]), {})
            question_dic[str(item[0])].setdefault("question", [])
            question_dic[str(item[0])].setdefault("answer", [])
            question_dic[str(item[0])].setdefault("label", [])
            question_dic[str(item[0])]["question"].append(item[0])
            question_dic[str(item[0])]["answer"].append(item[1])
            question_dic[str(item[0])]["label"].append(item[2])
        delCount = 0
        for key in list(question_dic.keys()):
            question_dic[key]["question"] = question_dic[key]["question"]
            question_dic[key]["answer"] = question_dic[key]["answer"]
            if sum(question_dic[key]["label"]) == 0:
                delCount += 1
                del (question_dic[key])
        bad_answers = []
        for item in question_dic.values():
            bad_ans = [item["answer"][i] for i in range(len(item["question"])) if item["label"][i] == 0]
            bad_answers.extend(bad_ans)
        # 进行listwise填充
        for item in question_dic.values():
            good_answer = [item["answer"][i] for i in range(len(item["question"])) if item["label"][i] == 1]
            good_length = len(good_answer)
            bad_answer = [item["answer"][i] for i in range(len(item["question"])) if item["label"][i] == 0]
            bad_answers.extend(bad_answer)
            if len(item["answer"]) >= self.param.list_size:
                good_answer.extend(random.sample(bad_answer, self.param.list_size - good_length))
                temp_answer = good_answer
                temp_label = [1 / float(sum(item["label"]))  for i in range(good_length)]
                # temp_label = [1.0  for i in range(good_length)]
                temp_label.extend([0.0 for i in range(self.param.list_size - good_length)])
            else:
                temp_answer = item["answer"]
                temp_answer.extend(random.sample(bad_answers, self.param.list_size - len(item["question"])))
                temp_label = [i / float(sum(item["label"])) for i in item["label"]]
                # temp_label = item["label"]
                temp_label.extend([0.0 for i in range(self.param.list_size - len(item["question"]))])
            label.append(temp_label)
            answer.append(list(self.pada(temp_answer)))
            answer_mask.append([self.cut(len(single_ans), self.param.ans_len) for single_ans in temp_answer])
            # 得出每个问题的长度
            question += [[list(self.padq([item["question"][0]]))[0][0]] * self.param.list_size]
            question_mask += [[list(self.padq([item["question"][0]]))[0][1]] * self.param.list_size]

        question = np.array(question)
        answer = np.array(answer)
        label = np.array(label)
        question_mask = np.array(question_mask)
        answer_mask = np.array(answer_mask)

        if (verbose):
            print(question[23])
            print(question.shape)
            print(question_mask[23])
            print(question_mask.shape)
            print(answer[23])
            print(answer.shape)
            print(answer_mask[23])
            print(answer_mask.shape)
            print(label[23])
            print(label.shape)

        if flag == "size":
            return question, answer, label, question_mask, answer_mask
        all_out = (question, answer, label, question_mask, answer_mask)
        with open(out_file, 'wb') as f:
            pickle.dump(all_out, f)
        return question, answer, label, question_mask, answer_mask

    # -------------------------------------------------------------------------------------------------
    def trecQaGenerate(self, filename, flag="basic", verbose=False):
        data = pickle.load(open(filename, 'r'))
        question_dic = {}
        question = list()
        answer = list()
        label = list()
        question_len = list()
        answer_len = list()
        answer_size = list()
        for item in data:
            question_dic.setdefault(str(item[0]), {})
            question_dic[str(item[0])].setdefault("question", [])
            question_dic[str(item[0])].setdefault("answer", [])
            question_dic[str(item[0])].setdefault("label", [])
            question_dic[str(item[0])]["question"].append(item[0])
            question_dic[str(item[0])]["answer"].append(item[1])
            question_dic[str(item[0])]["label"].append(item[2])
        delCount = 0
        for key in question_dic.keys():
            question_dic[key]["question"] = question_dic[key]["question"]
            question_dic[key]["answer"] = question_dic[key]["answer"]
            if sum(question_dic[key]["label"]) == 0:
                delCount += 1
                del (question_dic[key])
        for item in question_dic.values():
            good_answer = [item["answer"][i] for i in range(len(item["question"])) if item["label"][i] == 1]
            good_length = len(good_answer)
            if good_length >= self.param.random_size / 2:
                good_answer = random.sample(good_answer, self.param.random_size / 2)
                good_length = len(good_answer)
            bad_answer = [item["answer"][i] for i in range(len(item["question"])) if item["label"][i] == 0]
            trash_sample = self.param.random_size
            if len(bad_answer) >= self.param.random_size - good_length:
                good_answer.extend(random.sample(bad_answer, self.param.random_size - good_length))
                temp_answer = good_answer
                temp_label = [1 / float(good_length) for i in range(good_length)]
                temp_label.extend([0.0 for i in range(self.param.random_size - good_length)])
            else:
                temp_answer = good_answer + bad_answer
                trash_sample = len(temp_answer)
                temp_answer.extend(random.sample(self.answers.values(), self.param.random_size - len(temp_answer)))
                temp_label = [1 / float(len(good_answer)) for i in range(len(good_answer))]
                temp_label.extend([0.0 for i in range(self.param.random_size - len(good_answer))])

            label.append(temp_label)
            answer.append(self.pada(temp_answer))
            length = [1 for i in range(len(item["question"][0]))]

            ans_length = [[1 for i in range(len(single_ans))] for single_ans in temp_answer]
            answer_len.append(self.pada(ans_length))
            question_len += [[list(self.padq([length]))[0]] * self.param.list_size]
            question += [[list(self.padq([item["question"][0]]))[0]] * self.param.list_size]
            answer_size += [[1 for i in range(self.param.random_size) if i < trash_sample] + [0 for i in range(
                self.param.random_size - trash_sample)]]

        question = np.array(question)
        answer = np.array(answer)
        label = np.array(label)
        question_len = np.array(question_len)
        answer_len = np.array(answer_len)
        answer_size = np.array(answer_size)
        if (verbose):
            print(question.shape)
            print(question_len.shape)
            print(answer.shape)
            print(answer_len.shape)
            print(label.shape)
        if flag == "size":
            return question, answer, label, question_len, answer_len, answer_size
        return question, answer, question_len, answer_len, label

    def EvaluateGenerate(self, filename):
        data = pickle.load(open(filename, 'r'))
        question_dic = {}
        for item in data:
            question_dic.setdefault(str(item[0]), {})
            question_dic[str(item[0])].setdefault("question", [])
            question_dic[str(item[0])].setdefault("answer", [])
            question_dic[str(item[0])].setdefault("label", [])
            question_dic[str(item[0])]["question"].append(item[0])
            question_dic[str(item[0])]["answer"].append(item[1])
            question_dic[str(item[0])]["label"].append(item[2])
        delCount = 0
        for key in question_dic.keys():
            question_dic[key]["question"] = self.padq(question_dic[key]["question"])
            question_dic[key]["answer"] = self.pada(question_dic[key]["answer"])
            question_dic[key]["ques_len"] = self.padq(
                [[1 for i in range(len(single_que))] for single_que in question_dic[key]["question"]])
            question_dic[key]["ans_len"] = self.pada(
                [[1 for i in range(len(single_ans))] for single_ans in question_dic[key]["answer"]])

            if sum(question_dic[key]["label"]) == 0:
                delCount += 1
                del (question_dic[key])
        print(delCount)
        print(len(question_dic))
        return question_dic


class DataGenerator2(object):
    def __init__(self,params):
        self.params = params

    def padseq(self,seq_to_pad,pad_to_len):
        if(len(seq_to_pad)>=pad_to_len):
            return seq_to_pad[:pad_to_len]
        else:
            seq_to_pad.extend([0 for i in range(pad_to_len-len(seq_to_pad))])
            return seq_to_pad

    def test_listwise_clean(self, test_file, padding=True):
        list_size = 30
        test_f = pickle.load(open(test_file, 'rb'))
        test_size = len(test_f)
        question, answer, label = zip(*test_f)
        print('raw questions:{}'.format(np.shape(question)))
        print('raw answers:{}'.format(np.shape(answer)))
        print('raw labels:{}'.format(np.shape(label)))
        question_len= list(map(lambda x: [1 for _ in range(len(x))],question))
        answer_len= list(map(lambda x: [1 for _ in range(len(x))],answer))
        test_dic = dict()
        for i, ques in enumerate(question):
            test_dic.setdefault(str(ques), [])
            test_dic[str(ques)].append([ques, answer[i], question_len[i], answer_len[i], label[i]])
        print("size of test_dic: ", len(test_dic))
        questions = []
        answers = []
        questions_len = []
        answers_len = []
        labels = []
        answers_size = []
        for k, v in test_dic.items():
            ques, ans, ques_len, ans_len, label = zip(*v)
            if (np.sum(label)==0): continue
            ques = list(map(lambda x: self.padseq(x, self.params.ques_len), ques))
            ans = list(map(lambda x: self.padseq(x, self.params.ans_len), ans))
            ques_len = list(map(lambda x: self.padseq(x, self.params.ques_len), ques_len))
            ans_len = list(map(lambda x: self.padseq(x, self.params.ans_len), ans_len))
            # 用0来padding剩下的list里面的句子
            if(padding):
                if list_size-len(label) < 0:
                    print(label)
                ques_pad = [[0]*self.params.ques_len]*(list_size-len(label))
                ans_pad = [[0]*self.params.ans_len]*(list_size-len(label))
                ques.extend(ques_pad)
                ans.extend(ans_pad)
                ques_len.extend(ques_pad)
                ans_len.extend(ans_pad)
                label_pad = [0]*(list_size-len(label))
                label = list(label)
                label.extend(label_pad)
                answer_size = [1]*len(label)+[0]*(list_size-len(label))
                answers_size.append(answer_size)
            questions.append(np.array(ques))
            answers.append(np.array(ans))
            questions_len.append(np.array(ques_len))
            answers_len.append(np.array(ans_len))
            labels.append(np.array(label))
        questions = np.array(questions)
        answers = np.array(answers)
        labels = np.array(labels)
        questions_len = np.array(questions_len)
        answers_len = np.array(answers_len)
        answers_size = np.array(answers_size)
        #print np.array(questions[100]).shape

        print ("questions: ",questions.shape)
        print ("questions_len: ",questions_len.shape)
        print ("answers: ",answers.shape)
        print ("answers_len: ",answers_len.shape)
        print ("labels: ",labels.shape)
        print ("answers_size: ",answers_size.shape)
        all_out = (questions, answers, labels)
        with open('./data/wikiqa/self/raw/pre/pre_dev.pkl', 'wb') as f:
            pickle.dump(all_out, f)
        return (questions, answers, questions_len, answers_len, labels, answers_size) if padding else \
                (questions, answers, questions_len, answers_len, labels)


# 根据词表生成对应的embedding
def data_transform(embedding_size):
    words = []
    with open('data/wikiqa/self/raw/wiki_vocab.txt', 'r', encoding='utf-8') as f1:
        for line in f1:
            word = line.strip().split('\t')[1].lower()
            words.append(word)
    print(len(words))

    raw_glove = 'D:/NLP_shiyan/MAN-WikiQA-V1-3/glove/glove.6B.300d.txt'
    embedding_dic = {}
    count = 1
    rng = np.random.RandomState(None)
    pad_embedding = rng.uniform(-0.25, 0.25, size=(1, embedding_size))
    unk_embedding = rng.uniform(-0.25, 0.25, size=(1, embedding_size))
    embeddings = []
    clean_words = ['<PAD>', '<UNK>']
    embeddings.append(pad_embedding.reshape(-1).tolist())
    embeddings.append(unk_embedding.reshape(-1).tolist())
    print('uniform_init...')
    with open(raw_glove, 'r', encoding='utf-8') as fin:
        for line in fin:
            try:
                line_info = line.strip().split()
                word = line_info[0]
                embedding = [float(val) for val in line_info[1:]]
                embedding_dic[word] = embedding
                if word in words:
                    count += 1
                    clean_words.append(word)
                    embeddings.append(embedding)
            except:
                print('Error while loading line: {}'.format(line.strip()))
    print(count)
    print(len(clean_words))
    print(len(embeddings))
    print(np.shape(embeddings))
    with open('data/wikiqa/self/raw/wiki_clean_vocab.txt', 'w', encoding='utf-8') as f:
        for i, j in enumerate(clean_words):
            f.write('{}\t{}\n'.format(i, j))
    with open('./data/wikiqa/self/raw/wiki_embedding.pkl', 'wb') as f2:
        pickle.dump(embeddings, f2)


# 获得train、dev、test中所有的词
def gen_vocab():
    words = []
    data_sets = ['train', 'dev', 'test']
    for set_name in data_sets:
        fin_path = 'data/wikiqa/self/raw/WikiQA-{}.tsv'.format(set_name)
        with open(fin_path, 'r', encoding='utf-8') as fin:
            fin.readline()
            for line in fin:
                line_in = line.strip().split('\t')
                question = line_in[1].split(' ')
                answer = line_in[3].split(' ')
                for r1 in question:
                    if r1 not in words:
                        words.append(r1)
                for r2 in answer:
                    if r2 not in words:
                        words.append(r2)
    with open('data/wikiqa/self/raw/wiki_vocab.txt', 'w', encoding='utf-8') as f:
        for i, j in enumerate(words):
            f.write('{}\t{}\n'.format(i, j))


# 将train、dev、test替换成数字
def gen_data():
    clean_vocab = {}
    with open('data/wikiqa/self/raw/wiki_clean_vocab.txt', 'r', encoding='utf-8') as f1:
        for w in f1:
            w_in = w.strip().split('\t')
            clean_vocab[w_in[1]] = int(w_in[0])
    # print(clean_vocab)
    def trans(x, y):
        tran = []
        for i in x:
            if i.lower() in y:
                t = y[i.lower()]
            else:
                t = y['<UNK>']
            tran.append(t)
        return tran
    data_sets = ['train', 'dev', 'test']
    for loc,set_name in enumerate(data_sets):
        all_trans = []
        fin_path = 'data/wikiqa/self/raw/WikiQA-{}.tsv'.format(set_name)
        with open(fin_path, 'r', encoding='utf-8') as fin:
            fin.readline()
            for line in fin:
                line_in = line.strip().split('\t')
                question = line_in[1].split(' ')
                question_transed = trans(question,clean_vocab)
                answer = line_in[3].split(' ')
                answer_transed = trans(answer, clean_vocab)
                lable = int(line_in[4].split(' ')[0])
                res = (question_transed, answer_transed, lable)
                all_trans.append(res)
        print(all_trans)
        if loc == 0:
            with open('./data/wikiqa/self/raw/WikiQA_train.pkl', 'wb') as f:
                pickle.dump(all_trans, f)
        elif loc == 1:
            with open('./data/wikiqa/self/raw/WikiQA_dev.pkl', 'wb') as f:
                pickle.dump(all_trans, f)
        elif loc == 2:
            with open('./data/wikiqa/self/raw/WikiQA_test.pkl', 'wb') as f:
                pickle.dump(all_trans, f)


if __name__ == '__main__':
    # 分别生成所有词的词表 / 生成embedding和clean词表/ 根据clean词表生成数字列表
    # gen_vocab()
    # data_transform(300)
    # gen_data()

    # 数据扩充：生成listwise数据格式并进行padding
    class M_P():
        random_size = 15
        list_size = 15
        ans_len = 100
        ques_len = 25
    m_p = M_P()
    dg = DataGenerator(1, m_p, '')
    infile = './data/wikiqa/self/raw/WikiQA_train.pkl'
    outfile = './data/wikiqa/self/raw/pre/15_15/float/pre_train.pkl'
    train_data = dg.wikiQaGenerate(infile, outfile)

# raw_train: 20359
# raw_dev: 1129
# raw_test: 6164