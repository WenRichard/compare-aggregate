# -*- coding: utf-8 -*-
# @Time    : 2018/12/21 11:12
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : Utils.py
# @Software: PyCharm

import tensorflow as tf
import time
from datetime import timedelta
import numpy as np
from collections import defaultdict
import pickle
import os
from collections import Counter

UNKNOWN = '<UNK>'
PADDING = '<PAD>'


# print tensor shape
def print_shape(varname, var):
    """
    :param varname: tensor name
    :param var: tensor variable
    """
    try:
        print('{0} : {1}'.format(varname, var.get_shape()))
    except:
        print('{0} : {1}'.format(varname, np.shape(var)))


# print log info on SCREEN and LOG file simultaneously
def print_log(*args, **kwargs):
    print(*args)
    if len(kwargs) > 0:
        print(*args, **kwargs)
    return None


# print all used hyper-parameters on both SCREEN an LOG file
def print_args(args, log_file):
    """
    :Param args: all used hyper-parameters
    :Param log_f: the log life
    """
    argsDict = vars(args)
    argsList = sorted(argsDict.items())
    print_log("------------- HYPER PARAMETERS -------------", file = log_file)
    for a in argsList:
        print_log("%s: %s" % (a[0], str(a[1])), file = log_file)
    print("-----------------------------------------", file = log_file)
    return None


# time cost
def get_time_diff(startTime):
    endTime = time.time()
    diff = endTime - startTime
    return timedelta(seconds = int(round(diff)))


# count the number of trainable parameters in model
def count_parameters():
    totalParams = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variableParams = 1
        for dim in shape:
            variableParams *= dim.value
        totalParams += variableParams
    return totalParams


# 余弦相似度计算
def feature2cos_sim(feat_q, feat_a):
    # feat_q: 2D:(bz, hz)
    norm_q = tf.sqrt(tf.reduce_sum(tf.multiply(feat_q, feat_q), 1))
    norm_a = tf.sqrt(tf.reduce_sum(tf.multiply(feat_a, feat_a), 1))
    mul_q_a = tf.reduce_sum(tf.multiply(feat_q, feat_a), 1)
    cos_sim_q_a = tf.div(mul_q_a, tf.multiply(norm_q, norm_a))
    return tf.clip_by_value(cos_sim_q_a, 1e-5, 0.99999)


# margin Loss
def cal_loss_and_acc(ori_cand, ori_neg, M):
    # the target function
    zero = tf.fill(tf.shape(ori_cand), 0.0)
    margin = tf.fill(tf.shape(ori_cand), M)  # 0.2
    # 采用margin loss作为损失函数，希望正样本分数越高越好，负样本分数越低越好
    with tf.name_scope("loss"):
        losses = tf.maximum(zero, tf.subtract(margin, tf.subtract(ori_cand, ori_neg)))
        loss = tf.reduce_sum(losses)
    # cal accurancy
    # 该acc计算方式有待考量
    with tf.name_scope("acc"):
        correct = tf.equal(zero, losses)
        acc = tf.reduce_mean(tf.cast(correct, "float"), name="acc")
    return loss, acc

def map_score(qids, labels, preds):
    """"""
    labels = [int(i) for i in labels]
    qid2cand = defaultdict(list)
    for qid, label, pred in zip(qids, labels, preds):
        qid2cand[qid].append((pred, label))

    average_precs = []
    for qid, candidates in qid2cand.items():
        average_prec = 0
        running_correct_count = 0
        for i, (score, label) in enumerate(sorted(candidates, key=lambda asd: asd[0], reverse=True), 1):
            if label > 0:
                running_correct_count += 1
                average_prec += float(running_correct_count) / i
        average_precs.append(average_prec / (running_correct_count + 1e-6))
    map_score = sum(average_precs) / len(average_precs)
    return map_score


def mean_reciprocal_rank(qids, labels, preds):
    """"""
    labels = [int(i) for i in labels]
    qid2cand = defaultdict(list)
    for qid, label, pred in zip(qids, labels, preds):
        qid2cand[qid].append((pred, label))

    mrr = 0.
    for qid, candidates in qid2cand.items():
        for i, (score, label) in enumerate(sorted(candidates, key=lambda asd: asd[0], reverse=True), 1):
            if label > 0:
                mrr += 1. / i
                break

    mrr /= len(qid2cand)
    return mrr


def precision_at_k(qids, labels, preds, k=1):
    """"""
    labels = [int(i) for i in labels]
    qid2cand = defaultdict(list)
    for qid, label, pred in zip(qids, labels, preds):
        qid2cand[qid].append((pred, label))

    p_at_k = 0.0
    good_qids = []
    for qid, candidates in qid2cand.items():
        for i, (score, label) in enumerate(sorted(candidates, key=lambda asd: asd[0], reverse=True), 1):
            if i == k and label > 0:
                p_at_k += 1.
                good_qids.append(qid)
            if i > k:
                break

    p_at_k /= len(qid2cand)
    return p_at_k

def new_map_score(labels, preds):
    """
    :param labels: (questions_classes, list_wise) ----> (873, 15)
    :param preds: (questions_classes, list_wise) ----> (873, 15)
    :return:
    """

    assert len(preds) == len(labels), "Invalid Input! pred:%s label:%s" % (len(preds), len(labels))
    qid2cand = defaultdict(list)
    for qid in range(len(labels)):
        qid2cand[qid] = zip(preds[qid], labels[qid])
    average_precs = []
    for qid, candidates in qid2cand.items():
        average_prec = 0
        running_correct_count = 0
        for i, (score, label) in enumerate(sorted(candidates, key=lambda asd: asd[0], reverse=True), 1):
            if label > 0:
                running_correct_count += 1
                average_prec += float(running_correct_count) / i
        average_precs.append(average_prec / (running_correct_count + 1e-6))
    map_score = sum(average_precs) / len(average_precs)
    return map_score


def new_mean_reciprocal_rank(labels, preds):
    """
    :param labels: (questions_classes, list_wise) ----> (873, 15)
    :param preds: (questions_classes, list_wise) ----> (873, 15)
    :return:
    """

    assert len(preds) == len(labels), "Invalid Input! pred:%s label:%s" % (len(preds), len(labels))
    qid2cand = defaultdict(list)
    for qid in range(len(labels)):
        qid2cand[qid] = zip(preds[qid], labels[qid])
    mrr = 0.
    for qid, candidates in qid2cand.items():
        for i, (score, label) in enumerate(sorted(candidates, key=lambda asd: asd[0], reverse=True), 1):
            if label > 0:
                mrr += 1. / i
                break

    mrr /= len(qid2cand)
    return mrr


def new_precision_at_k(labels, preds, k=1):
    """
    :param labels: (questions_classes, list_wise) ----> (873, 15)
    :param preds: (questions_classes, list_wise) ----> (873, 15)
    :return:
    """
    assert len(preds) == len(labels), "Invalid Input! pred:%s label:%s" % (len(preds), len(labels))
    qid2cand = defaultdict(list)
    for qid in range(len(labels)):
        qid2cand[qid] = zip(preds[qid], labels[qid])

    p_at_k = 0.0
    good_qids = []
    for qid, candidates in qid2cand.items():
        for i, (score, label) in enumerate(sorted(candidates, key=lambda asd: asd[0], reverse=True), 1):
            if i == k and label > 0:
                p_at_k += 1.
                good_qids.append(qid)
            if i > k:
                break

    p_at_k /= len(qid2cand)
    return p_at_k

if __name__ == '__main__':
    print(new_map_score(np.array([[0, 0, 1], [0, 0, 1]]), np.array([[0.8, 0.1, 0.1], [0.4, 0.3, 0.3]])))
    print(new_map_score(np.array([[0, 0, 1], [0, 0, 1]]), np.array([[-0.8, -0.1, -0.1], [-0.4, -0.2, -0.2]])))
    print(new_map_score(np.array([[0, 0, 1], [0, 0, 1]]), np.array([[-0.8, 0.1, -0.1], [0.4, 0.3, 0.3]])))

