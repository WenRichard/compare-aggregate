# -*- coding: utf-8 -*-
# @Time    : 2019/2/26 14:57
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : layers.py
# @Software: PyCharm

import tensorflow as tf

def cal_algin_w(sent1, sent2, s1_m, s2_m, w, g):
    # sent1: bz x q_len x hz
    # sent2: bz x a_len x hz
    # s1_m: bz x q_len
    # s2_m: bz x a_len
    # w: bz x hz x hz

    # cross mask
    m1_m2 = tf.multiply(tf.expand_dims(s1_m, 2), tf.expand_dims(s2_m, 1))  # bz x q_len x a_len

    # compute the unnormalized attention for all word pairs
    sent2 = tf.transpose(sent2, [0, 2, 1])
    raw_atten = tf.matmul(sent1, w) + g # bz x q_len x hz
    raw_atten = tf.matmul(raw_atten, sent2)  # bz x q_len x a_len
    raw_atten = tf.multiply(raw_atten, m1_m2)  # bz x q_len x a_len

    # weighted attention,
    # using Softmax at two directions axis=-1 and axis=-2, for alpha and beta respectively
    atten1 = tf.exp(raw_atten - tf.reduce_max(raw_atten, axis=2, keep_dims=True))  # bz x q_len x a_len
    atten2 = tf.exp(raw_atten - tf.reduce_max(raw_atten, axis=1, keep_dims=True))  # bz x q_len x a_len
    # mask
    atten1 = tf.multiply(atten1, tf.expand_dims(s2_m, 1))  # bz x q_len x a_len
    atten2 = tf.multiply(atten2, tf.expand_dims(s1_m, 2))  # bz x q_len x a_len
    # get softmax value
    atten1 = tf.divide(atten1, tf.reduce_sum(atten1, axis=2, keep_dims=True))  # bz x q_len x 1
    atten2 = tf.divide(atten2, tf.reduce_sum(atten2, axis=1, keep_dims=True))  # bz x 1 x a_len
    # mask
    atten1 = tf.multiply(atten1, m1_m2)  # bz x q_len x a_len
    atten2 = tf.multiply(atten2, m1_m2)  # bz x q_len x a_len

    q_repres = tf.matmul(atten1, sent2, name='alpha')  # bz x q_len x hz
    a_repres = tf.matmul(tf.transpose(atten2, [0, 2, 1]), sent1, name='beta')  # bz x a_len x hz
    return q_repres, a_repres
