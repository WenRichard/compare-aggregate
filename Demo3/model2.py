# -*- coding: utf-8 -*-
# @Time    : 2019/3/7 18:32
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : model2.py
# @Software: PyCharm

import tensorflow as tf
from model_utils import bias_variable, weight_variable, multiply_3_2, attention_softmax_3d, maxpool
from Utils import *

class CAM():
    def __init__(self, embedding_size, preprocess_hidden_size, ques_len, ans_len, embedding_file, optimizer,
                 clip_value, window_sizes, n_filters, classification_hidden_size):
        self.embedding_size = embedding_size
        self.preprocess_hidden_size = preprocess_hidden_size
        self.ques_len = ques_len
        self.ans_len = ans_len
        self.embedding_file = embedding_file
        self.optimizer = optimizer
        self.clip_value = clip_value
        self.window_sizes = window_sizes
        self.n_filters = n_filters
        self.classification_hidden_size = classification_hidden_size
        self.normalization = True

        self._placeholder_init_pointwise()
        self.initialize_weights()
        self.build(self.embedding_file)
        self._loss_op()
        self._train_op()

    def _placeholder_init_pointwise(self):
        self._ques = tf.placeholder(tf.int32, [None, None, self.ques_len], name='ques_point')
        self._ans = tf.placeholder(tf.int32, [None, None, self.ans_len], name='ans_point')
        self._ques_mask = tf.placeholder(tf.int32, [None, None], 'ques_mask')
        self._ans_mask = tf.placeholder(tf.int32, [None, None], 'ans_mask')
        self.p_label = tf.placeholder(tf.float32, [None, None])
        self.learning_rate = tf.placeholder(tf.float32)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size, self.list_size = tf.shape(self._ans)[0], tf.shape(self._ans)[1]
        self.ques = tf.reshape(self._ques, [-1, self.ques_len])
        self.ans = tf.reshape(self._ans, [-1, self.ans_len])
        self.ques_mask = tf.reshape(self._ques_mask, [-1])
        self.ans_mask = tf.reshape(self._ans_mask, [-1])
        # self.label = tf.reshape(self.p_label, [-1])

    def initialize_weights(self):
        """Global initialization of weights for the representation layer

        """
        # preprocessing
        self.W_i = weight_variable('W_i', [self.embedding_size, self.preprocess_hidden_size])
        self.b_i = bias_variable('b_i', [self.preprocess_hidden_size])
        self.W_u = weight_variable('W_u', [self.embedding_size, self.preprocess_hidden_size])
        self.b_u = bias_variable('b_u', [self.preprocess_hidden_size])

        # attention
        self.W_g = weight_variable('W_g', [self.preprocess_hidden_size, self.preprocess_hidden_size])
        self.b_g = bias_variable('b_g', [self.preprocess_hidden_size])

        # aggregation
        self.W_windows = dict()
        self.b_windows = dict()
        for size in self.window_sizes:
            self.W_windows[size] = weight_variable('W_conv_{}'.format(size),
                                                   [size, self.preprocess_hidden_size, self.n_filters])
            self.b_windows[size] = bias_variable('b_conv_{}'.format(size), [self.n_filters])


        # classification
        self.W_2 = weight_variable('W_dense_1', [self.n_filters * len(self.window_sizes), self.classification_hidden_size])
        self.b_2 = bias_variable('b_dense_1', [self.classification_hidden_size])

        self.W_3 = weight_variable('W_dense_out', [self.classification_hidden_size])
        self.b_3 = bias_variable('b_dense_out', [1])

    def _Preprocess_layer(self, item):
        # 对应于原文公式（1）,得到Q,A
        item_sig = tf.nn.sigmoid(tf.nn.bias_add(multiply_3_2(item, self.W_i), self.b_i))
        item_tanh = tf.nn.tanh(tf.nn.bias_add(multiply_3_2(item, self.W_u), self.b_u))
        if self.normalization:
            item_tanh = tf.nn.l2_normalize(item_tanh, dim=2)
        item_h = tf.multiply(item_sig, item_tanh)  # (bz*ls, q_len, hz)
        return item_h

    def _Attention_layer(self, question, answer, question_mask, answer_mask):
        WQ = tf.nn.bias_add(multiply_3_2(question, self.W_g), self.b_g)

        G = attention_softmax_3d(tf.matmul(answer, WQ, transpose_b=True), question_mask)
        # we set all items to zero that correspond to zero-padded positions of the answer
        G_zero = tf.multiply(G, tf.expand_dims(answer_mask, -1))
        result = tf.matmul(G_zero, question)
        return result

    def _Cnn_layer(self, input, ans_mask):
        '''
        :param input: (bz, len, hz)
        :return: (bz, 1, 4hz)
        '''
        convolutions = []
        for size in self.window_sizes:
            convoluted = tf.nn.bias_add(
                tf.nn.conv1d(
                    input,
                    self.W_windows[size],
                    stride=1,
                    padding='SAME'
                ),
                self.b_windows[size]
            )
            convolutions.append(convoluted)

        all_convoluted = tf.concat(convolutions, axis=2)
        all_convoluted_padded = tf.multiply(
            all_convoluted,
            tf.tile(
                tf.expand_dims(ans_mask, -1),
                [1, 1, self.n_filters * len(self.window_sizes)]
            )
        )
        R = maxpool(tf.nn.relu(all_convoluted_padded))
        return R

    def _cnn_layer(self, input, ans_mask):
        '''
        :param input: (bz, len, hz)
        :return: (bz, 1, 4hz)
        '''
        # tf.layers.Conv1D(inputs, filters, kernel_size, strides=1)
        # self.n_filters一般指词嵌入的维度
        all = []
        for i in range(len(self.window_sizes)):
            cnn_out = tf.layers.conv1d(input, self.n_filters, self.window_sizes[i], padding='same',
                                       activation=tf.nn.relu, name='q_conv_' + str(i))
            all.append(cnn_out)
        cnn_outs = tf.concat(all, axis=-1)  # (bz*ls, len, 4hz)
        cnn_outs_padded = tf.multiply(cnn_outs, tf.tile(tf.expand_dims(ans_mask, -1),
                          [1, 1, self.n_filters * len(self.window_sizes)]))
        R = maxpool(cnn_outs_padded)  # (bz*ls, 4hz)
        print_shape('R', R)
        return R

    def _Classification_layer(self, R):
        """

        :param R: |answers| x filters
        :return:
        """
        dense = tf.nn.tanh(tf.nn.xw_plus_b(R, self.W_2, self.b_2))
        predict = tf.nn.xw_plus_b(dense, tf.expand_dims(self.W_3, -1), self.b_3)
        predict = tf.reshape(predict, [-1, self.list_size])
        return predict

    def build(self, embeddings):
        self.Embedding = tf.Variable(tf.to_float(embeddings), trainable=True, name='Embedding')
        self.embeded_left = tf.nn.embedding_lookup(self.Embedding, self.ques)
        self.embeded_right = tf.nn.embedding_lookup(self.Embedding, self.ans)
        print_shape('embeded_left', self.embeded_left)
        print_shape('embeded_right', self.embeded_right)

        # ~~ Preprocessing
        question = self._Preprocess_layer(self.embeded_left)
        answer = self._Preprocess_layer(self.embeded_right)
        question_mask = tf.sequence_mask(self.ques_mask, self.ques_len, dtype=tf.float32)
        answer_mask = tf.sequence_mask(self.ans_mask, self.ans_len, dtype=tf.float32)

        print_shape('question', question)
        print_shape('answer', answer)
        print_shape('question_mask', question_mask)
        print_shape('answer_mask', answer_mask)

        # ~~ Attention and Comparison
        H = self._Attention_layer(question, answer, question_mask, answer_mask)
        T = tf.multiply(answer, H)

        # ~~ Aggregation
        R = self._Cnn_layer(T, answer_mask)

        # ~~ Predict
        self.predict = self._Classification_layer(R)
        self.logit_score = tf.nn.softmax(self.predict, dim=-1)  # (bz, list_size)
        # self.logit_score = tf.nn.sigmoid(self.predict)
        # self.logit_score = tf.squeeze(final_, -1)
        print_shape('logit_score:', self.logit_score)

    def _loss_op(self, l2_lambda=0.0001):
        self.loss_listwise = tf.reduce_mean(
            self.p_label * (tf.log(tf.clip_by_value(self.p_label, 1e-5, 1.0)) - tf.log(self.logit_score)))
        # self.loss_listwise = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.p_label, logits=self.logit_score))
        # self.loss_listwise = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.p_label, logits=self.predict))
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        l2_loss = sum(reg_losses) * l2_lambda
        self.loss_listwise = self.loss_listwise + l2_loss
        print_shape('loss_listwise', self.loss_listwise)

        tf.summary.scalar('loss', self.loss_listwise)
        self.summary_op = tf.summary.merge_all()

    def _train_op(self):
        # Training summary for the current batch_loss
        optimizer = None
        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999)
        elif self.optimizer == 'adamax':
            optimizer = AdamaxOptimizer(self.learning_rate, 0.9, 0.999)
        elif self.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
        elif self.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
        elif self.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        else:
            ValueError('Unknown optimizer : {0}'.format(self.optimizer))

        if self.clip_value is not None:
            gradients, v = zip(*optimizer.compute_gradients(self.loss_listwise))
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)
            self.train_op = optimizer.apply_gradients(zip(clip_gradients, v))
        else:
            self.train_op = optimizer.minimize(self.loss_listwise)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)

    # 对于训练阶段，需要执行self.train_op, self.loss, self.summary_op三个op，并传入相应的数据
    def train(self, sess, batch, dropout_keep_prob, lr):
        feed_dict = {self._ques: batch.quest,
                     self._ans: batch.ans,
                     self._ques_mask: batch.quest_mask,
                     self._ans_mask: batch.ans_mask,
                     self.p_label: batch.label,
                     self.dropout_keep_prob: dropout_keep_prob,
                     self.learning_rate: lr}
        _, loss, summary = sess.run([self.train_op, self.loss_listwise, self.summary_op], feed_dict=feed_dict)
        return loss, summary

    # 对于eval阶段，不需要反向传播，所以只执行self.loss, self.summary_op两个op，并传入相应的数据
    def eval(self, sess, quest, ans, label, quest_mask, ans_mask, lr):
        feed_dict = {self._ques: quest,
                     self._ans: ans,
                     self._ques_mask: quest_mask,
                     self._ans_mask: ans_mask,
                     self.p_label: label,
                     self.dropout_keep_prob: 1.0,
                     self.learning_rate: lr}
        loss, summary, score = sess.run([self.loss_listwise, self.summary_op, self.logit_score],
                                                       feed_dict=feed_dict)
        return loss, summary, score

    def infer(self, sess, quest, ans, label, quest_mask, ans_mask, lr):
        feed_dict = {self._ques: quest,
                     self._ans: ans,
                     self._ques_mask: quest_mask,
                     self._ans_mask: ans_mask,
                     self.p_label: label,
                     self.dropout_keep_prob: 1.0,
                     self.learning_rate: lr}
        score = sess.run([self.logit_score], feed_dict=feed_dict)
        return score


from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer


class AdamaxOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Adamax algorithm.
    See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980)
    ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
    @@__init__
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, use_locking=False, name="Adamax"):
        super(AdamaxOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        if var.dtype.base_dtype == tf.float16:
            eps = 1e-7  # Can't use 1e-8 due to underflow -- not sure if it makes a big difference.
        else:
            eps = 1e-8

        v = self.get_slot(var, "v")
        v_t = v.assign(beta1_t * v + (1. - beta1_t) * grad)
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = v_t / m_t

        var_update = state_ops.assign_sub(var, lr_t * g_t)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")