# -*- coding: utf-8 -*-
# @Time    : 2019/1/22 19:41
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : train.py
# @Software: PyCharm
import tensorflow as tf
from model2 import CAM
import os
import argparse
from tqdm import tqdm
from Utils import *
import sys
from datetime import datetime

import time
from data_helper import *
import logging
import numpy as np

parser = argparse.ArgumentParser(description="QA2018")
parser.add_argument('-m','--model',type=str,default='b',help="base line model")
parser.add_argument('-t','--task',type=str,default='wiki',help="task name")
parser.add_argument('-b','--batch_size',type=int,default=25,help="batch size")  # 5
parser.add_argument('-ls','--list_size',type=int,default=50,help="list-wise size")
parser.add_argument('-qu','--ques_len',type=int,default=25,help="question len")
parser.add_argument('-an','--ans_len',type=int,default=100,help="answer len")
parser.add_argument('-cl','--clip_value',type=int,default=None,help="clip_value")  #15
parser.add_argument('-lr','--learning_rate',type=float,default=0.001,help="learning rate")
parser.add_argument('-em','--embedding_size',type=int,default=300,help="embedding_size")  # 300
parser.add_argument('-ph','--preprocess_hidden_size',type=int,default=100,help="preprocess_hidden_size")  # 300
parser.add_argument('-ch','--classification_hidden_size',type=int,default=300,help="classification_hidden_size")
parser.add_argument('-nf','--n_filters',type=int,default=100,help="n_filters")  # 300
parser.add_argument('-gid','--gpu_id',type=str,default='0,1',help="gpu id")
parser.add_argument('-cv','--cv',type=int,default=3,help="cv num")
parser.add_argument('-kp','--keep_prob',type=float,default=0.5,help="keep prob")  # 0.8
parser.add_argument('-lrdc','--lr_decay',type=float,default=0.99,help="learning rate decay")
parser.add_argument('-lrsr','--lr_shrink',type=float,default=0.5,help="learning rate shrink")
parser.add_argument('-opt','--optimizer',type=str,default='adam',help="optimizer")
parser.add_argument('-emb','--embedding_file',type=str,default=' ',help="optimizer")
parser.add_argument('-pre','--pre_train',type=int,default=1,help="pre-train epochs")
parser.add_argument('-num','--num_epochs',type=int,default=100,help="num_epochs")
parser.add_argument('-ev','--eval_batch',type=int,default=50,help="num_epochs")
parser.add_argument('-sa','--save_path',type=str,default='./model/checkpoint',help="save_path")
parser.add_argument('-be','--best_path',type=str,default='./model/bestval',help="best_path")
arg = parser.parse_args()
window_sizes = [1, 2, 3, 4, 5]

current_path = os.path.abspath(".")
wiki_train_file = current_path+"/data/wikiqa/wiki_train.pkl"
wiki_dev_file = current_path+"/data/wikiqa/wiki_dev.pkl"
wiki_test_file = current_path+"/data/wikiqa/wiki_test.pkl"
wiki_answer_file = current_path+"/data/wikiqa/wiki_answer_train.pkl"
wiki_embedding_file = current_path+"/data/wikiqa/wikiqa_glovec.txt"

trec_train_file = current_path+"/data/trecqa/trec_train.pkl"
trec_dev_file = current_path+"/data/trecqa/trec_dev.pkl"
trec_test_file = current_path+"/data/trecqa/trec_test.pkl"
trec_answer_file = current_path+"/data/trecqa/trec_answer_train.pkl"
trec_embedding_file = current_path+"/data/trecqa/trecqa_glovec.txt"

tfboard_path = "./tensorboard"
save_path = "./model/checkpoint"
best_path = "./model/bestval"
ckpt_name = "MAN.ckpt"

# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

# 创建一个handler，用于写入日志文件
timestamp = str(int(time.time()))
fh = logging.FileHandler('./log/log_' + timestamp +'.txt')
fh.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
# ch.setFormatter(formatter)

# 给logger添加handler
logger.addHandler(fh)
# logger.addHandler(ch)

# (873, 30, 20)
train_file = './data/wikiqa/self/raw/pre/15_15/float/pre_train.pkl'
dev_file = './data/wikiqa/self/raw/pre/15_15/float/pre_dev.pkl'
test_file = './data/wikiqa/self/raw/pre/15_15/float/pre_test.pkl'
train_question, train_answer, train_label, train_question_mask, train_answer_mask = load_data(train_file)
dev_question, dev_answer, dev_label, dev_question_mask, dev_answer_mask = load_data(dev_file)
test_question, test_answer, test_label, test_question_mask, test_answer_mask = load_data(test_file)
_embeddings = load_embedding('./data/wikiqa/self/raw/wiki_embedding.pkl')


def get_learning_rate(lr, model_params ,dev_map,last_dev_map,epoch):
    if epoch < 10:
        return lr
    if dev_map>=last_dev_map:
        return lr*model_params.lr_decay
    else:
        return lr*model_params.lr_shrink


def dev_model(sess, model, dev_question, dev_answer, dev_label, dev_question_mask, dev_answer_mask, batch_size, lr):
    batch_num = 0
    dev_scores = []
    dev_losses = []
    dev_labels = []
    for dev_ques, dev_ans, dev_lab, dev_ques_mask, dev_ans_mask in get_test_Batches(dev_question, dev_answer, dev_label,
                                                                                    dev_question_mask, dev_answer_mask,
                                                                                    batch_size):
        batch_num +=1
        dev_loss, _, dev_score = model.eval(sess, dev_ques, dev_ans, dev_lab, dev_ques_mask, dev_ans_mask, lr)
        # print('=======================================')
        # # print(las2)
        # print(dev_score)
        # print(dev_lab)
        dev_scores.extend(dev_score)
        dev_losses.append(dev_loss)
        dev_labels.extend(dev_lab)
        # print(dev_lab)
    Dev_loss = float(np.sum(dev_losses) / batch_num)
    dev_map = new_map_score(dev_labels, dev_scores)
    dev_mrr = new_mean_reciprocal_rank(dev_labels, dev_scores)
    dev_p_k = new_precision_at_k(dev_labels, dev_scores)
    return dev_map, dev_mrr, dev_p_k, Dev_loss

def test_model(sess, model, test_question, test_answer, test_label, test_question_mask, test_answer_mask, batch_size, lr):
    batch_num = 0
    test_scores = []
    test_labels = []
    for test_ques, test_ans, test_lab, test_ques_mask, test_ans_mask in get_test_Batches(test_question, test_answer,
                                                                                         test_label, test_question_mask,
                                                                                         test_answer_mask, batch_size):
        batch_num +=1
        test_score = model.infer(sess, test_ques, test_ans, test_lab, test_ques_mask, test_ans_mask, lr)
        test_scores.extend(np.reshape(test_score, [-1, 15]).tolist())
        test_labels.extend(test_lab)
    test_map = new_map_score(test_labels, test_scores)
    test_mrr = new_mean_reciprocal_rank(test_labels, test_scores)
    test_p_k = new_precision_at_k(test_labels, test_scores)
    return test_map, test_mrr, test_p_k


with tf.Session() as sess:
    # training
    print('Start training and evaluating ...')
    start_time = time.time()

    #embedding_size, preprocess_hidden_size, ques_len, ans_len, embedding_file, optimizer, learning_rate, clip_value, window_sizes, n_filters, classification_hidden_size

    model = CAM(arg.embedding_size,
                arg.preprocess_hidden_size,
                arg.ques_len,
                arg.ans_len,
                _embeddings,
                arg.optimizer,
                arg.clip_value,
                window_sizes,
                arg.n_filters,
                arg.classification_hidden_size,
                )
    if not os.path.exists(arg.save_path):
        os.makedirs(arg.save_path)
    if not os.path.exists(arg.best_path):
        os.makedirs(arg.best_path)
    ckpt = tf.train.get_checkpoint_state(arg.save_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('Created new model parameters..')
        sess.run(tf.global_variables_initializer())
        # count trainable parameters
    total_parameters = count_parameters()
    print('Total trainable parameters : {}'.format(total_parameters))

    current_step = 0
    best_acc_val = 0.0
    best_map_val = 0.0
    best_mrr_val = 0.0
    best_pk_val = 0.0
    last_improved_batch = 0
    isEarlyStop = False
    last_dev_acc = 0.0
    last_dev_map = 0.0
    last_dev_mrr = 0.0
    last_dev_pk = 0.0
    learning_rate = arg.learning_rate
    best_epoch = 0
    prior_map = 0.0
    print('Configuring TensorBoard and Saver ...')
    summary_writer = tf.summary.FileWriter(arg.save_path, graph=sess.graph)
    for e in range(arg.num_epochs):
        count = 0
        print("----- Epoch {}/{} -----".format(e + 1, arg.num_epochs))
        print("learning_rate :{}".format(learning_rate))
        train_batches = get_train_Batches(train_question, train_answer, train_question_mask, train_answer_mask,
                                          train_label, arg.batch_size)
        for nextBatch in tqdm(train_batches, desc='Training'):
            train_loss, summary = model.train(sess, nextBatch, arg.keep_prob, learning_rate)
            current_step +=1
            count += 1
            if count % 20 == 0:
                print('[epoch {}, batch {}]Loss:{}'.format(e, count, train_loss))
            summary_writer.add_summary(summary, current_step)

        dev_map, dev_mrr, dev_p_k, Dev_loss = dev_model(sess, model, dev_question, dev_answer, dev_label,
                                                        dev_question_mask, dev_answer_mask, arg.batch_size, learning_rate)
        test_map, test_mrr, test_p_k = test_model(sess, model, test_question, test_answer, test_label,
                                                  test_question_mask, test_answer_mask, arg.batch_size, learning_rate)
        # update lr
        learning_rate = get_learning_rate(learning_rate, arg, test_map, prior_map, e)

        logger.info("\nEvaluation:")
        logger.info(
            "--- Step %d -- Dev_Loss %.5f -- P_K %.5f -- MAP %.5f -- MRR %.5f" % (
                current_step, Dev_loss, dev_p_k, dev_map, dev_mrr))
        tqdm.write(
            "--- Step %d -- Dev_Loss %.5f -- P_K %.5f -- MAP %.5f -- MRR %.5f" % (
                current_step, Dev_loss, dev_p_k, dev_map, dev_mrr))

        logger.info("\nTest:")
        logger.info("--- test_P_K %.5f -- test_MAP %.5f -- test_MRR %.5f" % (test_p_k, test_map, test_mrr))
        print("--- test_P_K %.5f -- test_MAP %.5f -- test_MRR %.5f" % (test_p_k, test_map, test_mrr))

        prior_map = test_map

        checkpoint_path = os.path.join(arg.save_path, 'map{:.4f}_{}.ckpt'.format(test_map, current_step))
        bestcheck_path = os.path.join(arg.best_path, 'map{:.4f}_{}.ckpt'.format(test_map, current_step))
        model.saver.save(sess, checkpoint_path, global_step=current_step)
        if test_map > best_map_val or test_mrr > best_mrr_val:
            best_map_val = test_map
            best_mrr_val = test_mrr
            last_improved_batch = current_step
            best_epoch = e+1
            model.saver.save(sess, bestcheck_path, global_step=last_improved_batch)
    logger.info("\nBest and Last:")
    logger.info('--- best_MAP %.5f -- best_MRR %.5f -- last_MAP %.5f -- last_MRR %.5f -- best-epoch %.1f' % (
        best_map_val, best_mrr_val, last_dev_map, last_dev_mrr, best_epoch))
    print('--- best_MAP %.5f -- best_MRR %.5f -- last_MAP %.5f -- last_MRR %.5f -- best-epoch %.1f' % (
        best_map_val, best_mrr_val, last_dev_map, last_dev_mrr, best_epoch))