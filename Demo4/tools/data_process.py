#-*-coding:utf-8 -*-
# from __future__ import print_function
import sys
import numpy as np
import random
from collections import namedtuple
import pickle
random.seed(1337)
np.random.seed(1337)

ModelParam = namedtuple("ModelParam","hidden_dim,enc_timesteps,dec_timesteps,batch_size,random_size,k_value_ques,k_value_ans,lr")

UNKNOWN_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
class Vocab(object):
    def __init__(self, vocab_file, max_size):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0
        before_list = [PAD_TOKEN]
        for word in before_list:
                self.CreateWord(word)
        with open(vocab_file, 'r') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    sys.stderr.write('Bad line: %s\n' % line)
                    continue
                if pieces[1] in self._word_to_id:
                    raise ValueError('Duplicated word: %s.' % pieces[1])
                self._word_to_id[pieces[1]] = self._count
                self._id_to_word[self._count] = pieces[1]
                self._count += 1
                if self._count > max_size-1:
                    sys.stderr.write('Too many words: >%d.' % max_size)
                    break
    def WordToId(self, word):
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def IdToWord(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError('id not found in vocab: %d.' % word_id)
        return self._id_to_word[word_id]

    def NumIds(self):
        return self._count

    def CreateWord(self,word):
        if word not in self._word_to_id:
            self._word_to_id[word] = self._count
            self._id_to_word[self._count] = word
            self._count += 1
    def Revert(self,indices):
        vocab = self._id_to_word
        return [vocab.get(i, 'X') for i in indices]
    def Encode(self,indices):
        vocab = self._word_to_id
        return [vocab.get(i, 'nonum') for i in indices]


class DataGenerator(object):
    """Dataset class
    """
    def __init__(self,vocab,model_param,answer_file = ""):
        self.vocab = vocab
        self.param = model_param
        self.batch_size = self.param.batch_size
        self.corpus_amount = 0
        if answer_file != "":
            self.answers = pickle.load(open(answer_file,'rb'))
    
    def padq(self, data):
        return self.pad(data, self.param.ques_len)
    
    def pada(self, data):
        return self.pad(data, self.param.ans_len)
    
    def pad(self, data, lens=None):
        def _pad(seq,lens):
            if(len(seq)>=lens):
                return seq[:lens]
            else:
                return seq + [0 for i in range(lens-len(seq))]
        #return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)
        return map(lambda x: _pad(x,lens),data)
    
    def wikiQaGenerate(self,filename,flag="basic",verbose=False):
        data = pickle.load(open(filename,'rb'))
        question_dic = {}
        question = list()
        answer = list()
        label = list()
        question_len = list()
        answer_len = list()
        answer_size = list()
        for item in data:
            question_dic.setdefault(str(item[0]),{})
            question_dic[str(item[0])].setdefault("question",[])
            question_dic[str(item[0])].setdefault("answer",[])
            question_dic[str(item[0])].setdefault("label",[])
            question_dic[str(item[0])]["question"].append(item[0])
            question_dic[str(item[0])]["answer"].append(item[1])
            question_dic[str(item[0])]["label"].append(item[2])
        delCount = 0
        for key in list(question_dic.keys()):
            question_dic[key]["question"] = question_dic[key]["question"]
            question_dic[key]["answer"] = question_dic[key]["answer"]
            if sum(question_dic[key]["label"]) == 0:
                delCount += 1
                del(question_dic[key])
        bad_answers = []
        for item in question_dic.values():
            bad_ans = [item["answer"][i] for i in range(len(item["question"])) if item["label"][i] == 0]
            bad_answers.extend(bad_ans)
        for item in question_dic.values():
            good_answer = [item["answer"][i] for i in range(len(item["question"])) if item["label"][i] == 1] 
            good_length = len(good_answer)
            bad_answer = [item["answer"][i] for i in range(len(item["question"])) if item["label"][i] == 0] 
            trash_sample = self.param.list_size
            if len(item["answer"]) >= self.param.list_size:
                good_answer.extend(random.sample(bad_answer,self.param.list_size - good_length))
                temp_answer = good_answer
                temp_label = [1 / float(sum(item["label"])) for i in range(good_length)]
                temp_label.extend([0.0 for i in range(self.param.list_size-good_length)])
            else:
                temp_answer = item["answer"]
                temp_answer.extend(random.sample(bad_answers, self.param.list_size-len(item["question"])))
                temp_label = [ll / float(sum(item["label"])) for ll in item["label"]]
                temp_label.extend([0.0 for i in range(self.param.list_size-len(item["question"]))])
                # trash_sample记录一个问题对应多少个答案
                trash_sample = len(item["question"])
            label.append(temp_label)
            answer.append(list(self.pada(temp_answer)))
            # item["question"][0]: [102, 15, 191, 9805, 1035, 562, 23, 460, 782],即每个问题的长度
            # length对应问题的mask向量
            length = [1 for i in range(len(item["question"][0]))]
            # print(item["question"][0])

            ans_length = [[1 for i in range(len(single_ans))] for single_ans in temp_answer]
            # # answer_len生成list_size个答案mask向量
            answer_len.append(list(self.pada(ans_length)))
            # list(self.padq([length])): [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
            # print(list(self.padq([length])))
            # question_len生成list_size个问题mask向量
            question_len += [[list(self.padq([length]))[0]]*self.param.list_size]
            question += [[list(self.padq([item["question"][0]]))[0]]*self.param.list_size]
            # answer_size用1和0表示listwise中原始的答案和后面扩充的negative答案
            answer_size += [[1 for i in range(self.param.list_size) if i < trash_sample] + [0 for i in range(self.param.list_size-trash_sample)]]
    
        question = np.array(question)
        answer = np.array(answer)
        label = np.array(label)
        question_len = np.array(question_len)
        answer_len = np.array(answer_len)
        answer_size = np.array(answer_size)
    
        if(verbose):
            print (question[23])
            print (question.shape)
            print (answer[23])
            print (question_len.shape)
            print (answer.shape)
            print (answer_len.shape)
            print (label.shape)
    
        if flag == "size":
            return question,answer,label,question_len,answer_len,answer_size
        return question,answer,question_len,answer_len,label


    def EvaluateGenerate(self,filename):
        data = pickle.load(open(filename,'r'))
        question_dic = {}
        for item in data:
            question_dic.setdefault(str(item[0]),{})
            question_dic[str(item[0])].setdefault("question",[])
            question_dic[str(item[0])].setdefault("answer",[])
            question_dic[str(item[0])].setdefault("label",[])
            question_dic[str(item[0])]["question"].append(item[0])
            question_dic[str(item[0])]["answer"].append(item[1])
            question_dic[str(item[0])]["label"].append(item[2])
        delCount = 0
        for key in question_dic.keys():
            question_dic[key]["question"] = self.padq(question_dic[key]["question"])
            question_dic[key]["answer"] = self.pada(question_dic[key]["answer"])
            question_dic[key]["ques_len"] = self.padq([[1 for i in range(len(single_que))] for single_que in question_dic[key]["question"] ])
            question_dic[key]["ans_len"] = self.pada([[1 for i in range(len(single_ans))] for single_ans in question_dic[key]["answer"] ])
    
            if sum(question_dic[key]["label"]) == 0:
                delCount += 1
                del(question_dic[key])
        print (delCount)
        print (len(question_dic))
        return question_dic

if __name__ == '__main__':
    class M_P():
        batch_size = 10
        enc_timesteps = 20
        dec_timesteps = 30
        random_size = 50
        list_size = 15
        ans_len = 30
        ques_len = 20
    m_p = M_P()
    infile = '../data/wikiqa/self/raw/WikiQA_train.pkl'
    dg = DataGenerator(1,m_p,'')
    dg.wikiQaGenerate(infile)
    # dg = DataGenerator(1,m_p,'../data/trecqa/trec_answer_train.pkl')
    #dg.EvaluateGenerate('../data/wikiqa/wiki_dev.pkl')



