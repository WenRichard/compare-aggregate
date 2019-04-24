# -*- coding: utf-8 -*-
# @Time    : 2019/3/19 20:19
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : train.py
# @Software: PyCharm


import time
import logging
import numpy as np
import tensorflow as tf
import os
import tqdm
import sys
from copy import deepcopy
stdout = sys.stdout

from data_helper import *
from model import CAM
from model_utils import *

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


class NNConfig(object):
    def __init__(self, embeddings=None):
        # 输入问题(句子)长度
        self.ques_length = 25
        # 输入答案长度
        self.ans_length = 90
        # 循环数
        self.num_epochs = 100
        # batch大小
        self.batch_size = 25
        # 不同类型的filter，对应不同的尺寸
        self.window_sizes = [1, 2, 3, 4, 5]
        # 隐层大小
        self.hidden_size = 100
        self.output_size = 128
        self.keep_prob = 0.5
        # 每种filter的数量
        self.n_filters = 100
        # RNN单元类型和大小与堆叠层数
        self.rnn_type = 'lstm'
        self.rnn_size = 300
        self.layer_size = 1
        # 词向量大小
        self.embeddings = np.array(embeddings).astype(np.float32)
        # 学习率
        self.learning_rate = 0.001
        # contrasive loss 中的 positive loss部分的权重
        self.pos_weight = 0.25
        # 优化器
        self.optimizer = 'adam'
        self.clip_value = None
        self.l2_lambda = 0.00001
        # 评测
        self.eval_batch = 100

        # self.cf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # self.cf.gpu_options.per_process_gpu_memory_fraction = 0.2


def evaluate(sess, model, corpus, config):
    iterator = Iterator(corpus)

    count = 0
    total_qids = []
    total_aids = []
    total_pred = []
    total_labels = []
    total_loss = 0.
    for batch_x in iterator.next(config.batch_size, shuffle=False):
        batch_qids, batch_q, batch_aids, batch_a, batch_qmask, batch_amask, labels = zip(*batch_x)
        batch_q = np.asarray(batch_q)
        batch_a = np.asarray(batch_a)
        batch_qmask = np.asarray(batch_qmask)
        batch_amask = np.asarray(batch_amask)
        y_hat, loss = sess.run([model.y_hat, model.total_loss],
                                     feed_dict={model._ques: batch_q,
                                                model._ans: batch_a,
                                                model._y: labels,
                                                model._ques_mask: batch_qmask,
                                                model._ans_mask: batch_amask,
                                                model.dropout_keep_prob: 1.0})
        y_hat = np.argmax(y_hat, axis=-1)
        total_loss += loss
        count += 1
        total_qids.append(batch_qids)
        total_aids.append(batch_aids)
        total_pred.append(y_hat)
        total_labels.append(labels)

        # print(batch_qids[0], [id2word[_] for _ in batch_q[0]],
        #     batch_aids[0], [id2word[_] for _ in batch_ap[0]])
    total_qids = np.concatenate(total_qids, axis=0)
    total_aids = np.concatenate(total_aids, axis=0)
    total_pred = np.concatenate(total_pred, axis=0)
    total_labels = np.concatenate(total_labels, axis=0)
    MAP, MRR = eval_map_mrr(total_qids, total_aids, total_pred, total_labels)
    # print('Eval loss:{}'.format(total_loss / count))
    return MAP, MRR, total_loss


def test(corpus, config):
    with tf.Session() as sess:
        model = CAM(config)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(best_path))
        test_MAP, test_MRR, _ = evaluate(sess, model, corpus, config)
        print('start test...............')
        print("-- test MAP %.5f -- test MRR %.5f" % (test_MAP, test_MRR))


def train(train_corpus, val_corpus, test_corpus, config, eval_train_corpus=None):
    iterator = Iterator(train_corpus)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(best_path):
        os.makedirs(best_path)

    with tf.Session() as sess:
        # training
        print('Start training and evaluating ...')
        start_time = time.time()

        model = CAM(config)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        ckpt = tf.train.get_checkpoint_state(save_path)
        print('Configuring TensorBoard and Saver ...')
        summary_writer = tf.summary.FileWriter(save_path, graph=sess.graph)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Created new model parameters..')
            sess.run(tf.global_variables_initializer())

        # count trainable parameters
        total_parameters = count_parameters()
        print('Total trainable parameters : {}'.format(total_parameters))

        current_step = 0
        best_map_val = 0.0
        best_mrr_val = 0.0
        last_dev_map = 0.0
        last_dev_mrr = 0.0
        for epoch in range(config.num_epochs):
            print("----- Epoch {}/{} -----".format(epoch + 1, config.num_epochs))
            count = 0
            for batch_x in iterator.next(config.batch_size, shuffle=True):

                batch_qids, batch_q, batch_aids, batch_a, batch_qmask, batch_amask, labels = zip(*batch_x)
                batch_q = np.asarray(batch_q)
                batch_a = np.asarray(batch_a)
                batch_qmask = np.asarray(batch_qmask)
                batch_amask = np.asarray(batch_amask)
                _, loss, summary = sess.run([model.train_op, model.total_loss, model.summary_op],
                                   feed_dict={model._ques: batch_q,
                                              model._ans: batch_a,
                                              model._y: labels,
                                              model._ques_mask: batch_qmask,
                                              model._ans_mask: batch_amask,
                                              model.dropout_keep_prob: config.keep_prob})
                count += 1
                current_step += 1
                if count % 20 == 0:
                    print('[epoch {}, batch {}]Loss:{}'.format(epoch, count, loss))
                summary_writer.add_summary(summary, current_step)
            if eval_train_corpus is not None:
                train_MAP, train_MRR, train_Loss = evaluate(sess, model, eval_train_corpus, config)
                print("--- epoch %d  -- train Loss %.5f -- train MAP %.5f -- train MRR %.5f" % (
                        epoch+1, train_Loss, train_MAP, train_MRR))
            if val_corpus is not None:
                dev_MAP, dev_MRR, dev_Loss = evaluate(sess, model, val_corpus, config)
                print("--- epoch %d  -- dev Loss %.5f -- dev MAP %.5f -- dev MRR %.5f" % (
                    epoch + 1, dev_Loss, dev_MAP, dev_MRR))
                logger.info("\nEvaluation:")
                logger.info("--- epoch %d  -- dev Loss %.5f -- dev MAP %.5f -- dev MRR %.5f" % (
                    epoch + 1, dev_Loss, dev_MAP, dev_MRR))

                test_MAP, test_MRR, test_Loss= evaluate(sess, model, test_corpus, config)
                print("--- epoch %d  -- test Loss %.5f -- test MAP %.5f -- test MRR %.5f" % (
                    epoch + 1, test_Loss, test_MAP, test_MRR))
                logger.info("\nTest:")
                logger.info("--- epoch %d  -- test Loss %.5f -- test MAP %.5f -- test MRR %.5f" % (
                    epoch + 1, test_Loss, test_MAP, test_MRR))

                checkpoint_path = os.path.join(save_path, 'map{:.5f}_{}.ckpt'.format(test_MAP, current_step))
                bestcheck_path = os.path.join(best_path, 'map{:.5f}_{}.ckpt'.format(test_MAP, current_step))
                saver.save(sess, checkpoint_path, global_step=epoch)
                if test_MAP > best_map_val or test_MRR > best_mrr_val:
                    best_map_val = test_MAP
                    best_mrr_val = test_MRR
                    best_saver.save(sess, bestcheck_path, global_step=epoch)
                last_dev_map = test_MAP
                last_dev_mrr = test_MRR
        logger.info("\nBest and Last:")
        logger.info('--- best_MAP %.4f -- best_MRR %.4f -- last_MAP %.4f -- last_MRR %.4f'% (
            best_map_val, best_mrr_val, last_dev_map, last_dev_mrr))
        print('--- best_MAP %.4f -- best_MRR %.4f -- last_MAP %.4f -- last_MRR %.4f' % (
            best_map_val, best_mrr_val, last_dev_map, last_dev_mrr))


def main(args):
    max_q_length = 25
    max_a_length = 90
    processed_data_path_pointwise = '../data/WikiQA/processed/pointwise'
    train_file = os.path.join(processed_data_path_pointwise, 'WikiQA-train.tsv')
    dev_file = os.path.join(processed_data_path_pointwise, 'WikiQA-dev.tsv')
    test_file = os.path.join(processed_data_path_pointwise, 'WikiQA-test.tsv')
    vocab = os.path.join(processed_data_path_pointwise, 'wiki_clean_vocab.txt')
    embeddings_file = os.path.join(processed_data_path_pointwise, 'wiki_embedding.pkl')
    _embeddings = load_embedding(embeddings_file)
    train_transform = transform(train_file, vocab)
    dev_transform = transform(dev_file, vocab)
    test_transform = transform(test_file, vocab)
    train_corpus = load_data(train_transform, max_q_length, max_a_length, keep_ids=True)
    dev_corpus = load_data(dev_transform, max_q_length, max_a_length, keep_ids=True)
    test_corpus = load_data(test_transform, max_q_length, max_a_length, keep_ids=True)

    config = NNConfig(embeddings=_embeddings)
    config.ques_length = max_q_length
    config.ans_length = max_a_length
    if args.train:
        train(deepcopy(train_corpus), dev_corpus, test_corpus, config)
    elif args.test:
        test(test_corpus, config)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="whether to train", action='store_true')
    parser.add_argument("--test", help="whether to test", action='store_true')
    args = parser.parse_args()

    save_path = "./model/checkpoint"
    best_path = "./model/bestval"
    main(args)
