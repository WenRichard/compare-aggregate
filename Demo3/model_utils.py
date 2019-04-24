# -*- coding: utf-8 -*-
# @Time    : 2019/3/7 18:30
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : model_utils.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer, xavier_initializer

def attention_softmax_3d_align(attention, indices_non_zero, dim = 1):
    """Softmax that ignores values of zero token indices (zero-padding)

    :param attention: (bz, q_len, a_len)
    :param indices_non_zero: (bz, q_len, 1)或者(bz, 1, a_len)
    :return:
    """
    ex = tf.multiply(tf.exp(attention), indices_non_zero)
    sum = tf.reduce_sum(ex, [dim], keep_dims=True)
    softmax = tf.divide(ex, sum)
    return softmax

def attention_softmax_3d(attention, indices_non_zero):
    """Softmax that ignores values of zero token indices (zero-padding)

    :param attention:
    :param raw_indices:
    :return:
    """
    ex = tf.multiply(tf.exp(attention), tf.expand_dims(indices_non_zero, 1))
    sum = tf.reduce_sum(ex, [2], keep_dims=True)
    softmax = tf.divide(ex, sum)
    return softmax


def maxpool(item):
    return tf.reduce_max(item, [1], keep_dims=False)


def weight_variable(name, shape, reg=None):
    if reg is not None:
        reg = tf.contrib.layers.l2_regularizer(scale=reg)
    return tf.get_variable(name, shape=shape, initializer=xavier_initializer(), regularizer=reg)


def bias_variable(name, shape, value=0.0):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(value))


def multiply_3_2(x, y, n_items=None, n_values=None, n_output_values=None):
    """Matmuls each 2d matrix in a 3d tensor with a 2d mulitplicator
    :param x: 3d input
    :param y: 2d input
    :param n_items: you can explicitly set the shape of the input to enable better debugging in tensorflow
    :return:
    """
    shape_x = tf.shape(x)
    shape_y = tf.shape(y)

    n_items = shape_x[1] if n_items is None else n_items
    n_values = shape_x[2] if n_values is None else n_values
    n_output_values = shape_y[1] if n_output_values is None else n_output_values

    x_2d = tf.reshape(x, [-1, n_values])
    result_2d = tf.matmul(x_2d, y)
    result_3d = tf.reshape(result_2d, [-1, n_items, n_output_values])
    return result_3d


def non_zero_tokens(tokens):
    """Receives a batch of vectors of tokens (float) which are zero-padded. Returns a vector of the same size, which has
    the value 1.0 in positions with actual tokens and 0.0 in positions with zero-padding.
    :param tokens:
    :return:
    """
    return tf.ceil(tokens / tf.reduce_max(tokens, [1], keep_dims=True))

