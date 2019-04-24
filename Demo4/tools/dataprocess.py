#!/usr/bin/env python
# -*- coding=utf-8 -*-

import tensorflow as tf
import numpy as np
from collections import namedtuple
import pickle
import random
random.seed(1337)
np.random.seed(1337)

class DataGenerator(object):
    def __init__(self,params):
        self.params = params

    def padseq(self,seq_to_pad,pad_to_len):
        if(len(seq_to_pad)>=pad_to_len):
            return seq_to_pad[:pad_to_len]
        else:
            seq_to_pad.extend([0 for i in range(pad_to_len-len(seq_to_pad))])
            return seq_to_pad

    def data_pointwise(self,train_file):
        train_f = pickle.load(open(train_file,'r'))
        train_size = len(train_f)
        question, answer, label = zip(*train_f)
        question_len= map(lambda x: len(x),question)
        answer_len = map(lambda x: len(x),answer)
        question = map(lambda x: self.padseq(x,self.params.ques_len),question)
        answer = map(lambda x: self.padseq(x,self.params.ques_len),answer)

        question = np.array(question)
        answer = np.array(answer)
        label = np.array(label)
        question_len = np.array(question_len)
        answer_len = np.array(answer_len)
        print (question.shape)
        print (question_len.shape)
        print (answer.shape)
        print (answer_len.shape)
        print (label.shape)

        return question, question_len, answer, answer_len, label

    def data_listwise_clean_internal_sample(self,train_file,answer_file):
        train_f = pickle.load(open(train_file,'r'))
        answer_f = pickle.load(open(answer_file,'r'))
        train_size = len(train_f)
        question,answer,label = zip(*train_f)
        assert len(question)==len(answer)==len(label), "Invalid train data with vary size among question,answer,label!"
        question_len = map(lambda x: [1 for _ in range(len(x))], question)
        answer_len = map(lambda x: [1 for _ in range(len(x))], answer)
        train_dic = dict()
        for i,ques in enumerate(question):
            train_dic.setdefault(str(ques),[])
            train_dic[str(ques)].append([ques,answer[i],question_len[i],answer_len[i],label[i]])
        print ("size of train_dic:",len(train_dic))

        questions = []
        answers = []
        questions_len = []
        answers_len = []
        p_labels = []
        l_labels = []
        list_size = self.params.list_size
        for k,v in train_dic.items():
            if(len(v)>=list_size):
                false_sample = [i for i in range(len(v)) if v[i][-1]==0]
                filtered_false_sample = set(random.sample(false_sample,len(v)-list_size))
                train_dic[k] = [v[i] for i in range(len(v)) if i not in filtered_false_sample]
            else:
                pad_size = list_size - len(v)
                false_sample = [i for i in range(len(v)) if v[i][-1]==0]*pad_size
                if(len(false_sample)):
                    pad_answer = [v[i][1] for i in random.sample(false_sample,pad_size)]
                else:
                    pad_answer = random.sample(answer_f.values(),pad_size)
                pad_answer_len = [[1 for _ in range(len(x))] for x in pad_answer]
                pad_sample = [[v[0][0],pad_answer[i],v[0][2],pad_answer_len[i],0] for i in range(len(pad_answer))]
                train_dic[k].extend(pad_sample)
            ques, ans, ques_len, ans_len, p_label = zip(*train_dic[k])
            if(np.sum(p_label)==0): continue
            ques = map(lambda x: self.padseq(x,self.params.ques_len),ques)
            ans = map(lambda x: self.padseq(x,self.params.ans_len),ans)
            ques_len = map(lambda x: self.padseq(x,self.params.ques_len),ques_len)
            ans_len = map(lambda x: self.padseq(x,self.params.ans_len),ans_len)
            l_label = [float(t)/np.sum(p_label) if np.sum(p_label) else float(t) for t in p_label]
            questions.append(ques)
            answers.append(ans)
            questions_len.append(ques_len)
            answers_len.append(ans_len)
            p_labels.append(p_label)
            l_labels.append(l_label)
        questions = np.array(questions)
        answers = np.array(answers)
        questions_len = np.array(questions_len)
        answers_len = np.array(answers_len)
        p_labels = np.array(p_labels)
        l_labels = np.array(l_labels)

        print ('questions:',questions.shape)
        print ('answers:',answers.shape)
        print ('questions_len:',questions_len.shape)
        print ('answers_len',answers_len.shape)
        print ('p_labels:',p_labels.shape)
        print ('l_labels:',l_labels.shape)

        return questions, answers, questions_len, answers_len, p_labels, l_labels

    def data_listwise_clean(self,train_file,answer_file):
        train_f = pickle.load(open(train_file,'r'))
        answer_f = pickle.load(open(answer_file,'r'))
        train_size = len(train_f)
        question,answer,label = zip(*train_f)
        assert len(question)==len(answer)==len(label), "Invalid train data with vary size among question,answer,label!"
        question_len = map(lambda x: [1 for _ in range(len(x))], question)
        answer_len = map(lambda x: [1 for _ in range(len(x))], answer)
        train_dic = dict()
        for i,ques in enumerate(question):
            train_dic.setdefault(str(ques),[])
            train_dic[str(ques)].append([ques,answer[i],question_len[i],answer_len[i],label[i]])
        print ("size of train_dic:",len(train_dic))

        questions = []
        answers = []
        questions_len = []
        answers_len = []
        p_labels = []
        l_labels = []
        list_size = self.params.list_size
        for k,v in train_dic.items():
            if(len(v)>=list_size):
                false_sample = [i for i in range(len(v)) if v[i][-1]==0]
                filtered_false_sample = set(random.sample(false_sample,len(v)-list_size))
                train_dic[k] = [v[i] for i in range(len(v)) if i not in filtered_false_sample]
            else:
                pad_size = list_size - len(v)
                pad_answer = random.sample(answer_f.values(),pad_size)
                pad_answer_len = [[1 for _ in range(len(x))] for x in pad_answer]
                pad_sample = [[v[0][0],pad_answer[i],v[0][2],pad_answer_len[i],0] for i in range(len(pad_answer))]
                train_dic[k].extend(pad_sample)
            ques, ans, ques_len, ans_len, p_label = zip(*train_dic[k])
            if(np.sum(p_label)==0): continue
            ques = map(lambda x: self.padseq(x,self.params.ques_len),ques)
            ans = map(lambda x: self.padseq(x,self.params.ans_len),ans)
            ques_len = map(lambda x: self.padseq(x,self.params.ques_len),ques_len)
            ans_len = map(lambda x: self.padseq(x,self.params.ans_len),ans_len)
            l_label = [float(t)/np.sum(p_label) if np.sum(p_label) else float(t) for t in p_label]
            questions.append(ques)
            answers.append(ans)
            questions_len.append(ques_len)
            answers_len.append(ans_len)
            p_labels.append(p_label)
            l_labels.append(l_label)
        questions = np.array(questions)
        answers = np.array(answers)
        questions_len = np.array(questions_len)
        answers_len = np.array(answers_len)
        p_labels = np.array(p_labels)
        l_labels = np.array(l_labels)

        #print 'q',questions[99]
        print ('questions:',questions.shape)
        #print 'a',answers[99]
        print ('answers:',answers.shape)
        #print 'q_l',questions_len[99]
        print ('questions_len:',questions_len.shape)
        #print 'a_l',answers_len[99]
        print ('answers_len',answers_len.shape)
        #print 'p_labels:',p_labels.shape
        print ('l_labels:',l_labels.shape)
        print (np.sum(l_labels))

        return questions, answers, questions_len, answers_len, l_labels

    def test_listwise_clean(self,test_file,padding=False):
        list_size = 30
        test_f = pickle.load(open(test_file,'r'))
        test_size = len(test_f)
        question, answer, label = zip(*test_f)
        question_len= map(lambda x: [1 for _ in range(len(x))],question)
        answer_len= map(lambda x: [1 for _ in range(len(x))],answer)
        test_dic = dict()
        for i,ques in enumerate(question):
            test_dic.setdefault(str(ques),[])
            test_dic[str(ques)].append([ques,answer[i],question_len[i],answer_len[i],label[i]])
        print ("size of test_dic: ",len(test_dic))
        questions = []
        answers = []
        questions_len = []
        answers_len = []
        labels = []
        answers_size = []
        for k,v in test_dic.items():
            ques, ans, ques_len, ans_len, label = zip(*v)
            if(np.sum(label)==0): continue
            ques = map(lambda x: self.padseq(x,self.params.ques_len),ques)
            ans = map(lambda x: self.padseq(x,self.params.ans_len),ans)
            ques_len = map(lambda x: self.padseq(x,self.params.ques_len),ques_len)
            ans_len = map(lambda x: self.padseq(x,self.params.ans_len),ans_len)
            if(padding):
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

        return (questions, answers, questions_len, answers_len, labels, answers_size) if padding else \
                (questions, answers, questions_len, answers_len, labels)

def evaluate_map_mrr(pred,label):
    assert len(pred)==len(label), "Invalid Input! pred:%s label:%s" % (len(pred),len(label))
    mAp, mrr = 0.0, 0.0
    for i in range(len(pred)):
        count = 0.0
        t_map, t_mrr = 0.0, 0.0
        rank_index = np.argsort(pred[i])[::-1]
        for j in range(len(pred[i])):
            if(label[i][rank_index[j]]==1):
                if(count==0):
                    t_mrr = 1/float((j+1))
                count += 1
                t_map += count/(j+1)
        t_map = t_map/(count+ 1e-6)
        mAp += t_map
        mrr += t_mrr
    mAp /= len(pred)
    mrr /= len(pred)

    return mAp, mrr

def evaluate_score(sess,model,dev_data):
    dev_ques = dev_data[0]
    dev_ans = dev_data[1]
    dev_ques_l = dev_data[2]
    dev_ans_l = dev_data[3]

    score_list = []
    for i in range(len(dev_ques)):
        # np.newaxis增加一维为batch_size
        score = sess.run(model.logit_score,feed_dict={model.r_ques:dev_ques[i][np.newaxis,:],
                                                        model.r_ans:dev_ans[i][np.newaxis,:],
                                                        model.r_ques_len:dev_ques_l[i][np.newaxis,:],
                                                        model.r_ans_len:dev_ans_l[i][np.newaxis,:],
                                                      model.is_train:False,
                                                      model.dc:True
                                                 })
        score_list.append(score[0])
    return score_list

if __name__ == '__main__':
    class Param():
        def __init__(self):
            self.ques_len = 10
            self.ans_len = 20
            self.list_size = 15
    param = Param()
    datag = DataGenerator(param)
    print ('DataGenerator avaliable...')
    datag.data_listwise_clean_internal_sample('../data/wikiqa/wiki_train.pkl','../data/wikiqa/wiki_answer_train.pkl')
    #datag.data_listwise_clean('../data/wikiqa/wiki_train.pkl','../data/wikiqa/wiki_answer_train.pkl')
    #datag.test_listwise_clean('../data/wikiqa/wiki_test.pkl')
    #datag.test_listwise_clean('../data/wikiqa/wiki_dev.pkl')
    #datag.test_listwise('../data/wikiqa/wiki_train.pkl')
    #print(evaluate_map_mrr(np.array([[0.8,0.1,0.1],[0.4,0.3,0.3]]),np.array([[0,0,1],[0,0,1]])))
