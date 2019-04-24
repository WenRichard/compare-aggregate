# -*- coding: utf-8 -*-
# @Time    : 2019/3/19 16:21
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : model.py
# @Software: PyCharm

import tensorflow as tf
from model_utils import *
from tensorflow.contrib.layers import l2_regularizer, xavier_initializer
from tensorflow.contrib.rnn import LSTMCell, GRUCell, DropoutWrapper


class CAM(object):
    def __init__(self, config):
        self.ques_len = config.ques_length
        self.ans_len = config.ans_length
        self.hidden_size = config.hidden_size
        self.output_size = config.output_size
        self.pos_weight = config.pos_weight
        self.learning_rate = config.learning_rate
        self.optimizer = config.optimizer
        self.l2_lambda = config.l2_lambda
        self.clip_value = config.clip_value
        self.embeddings = config.embeddings
        self.window_sizes = config.window_sizes
        self.n_filters = config.n_filters
        self.rnn_size = config.rnn_size

        self._placeholder_init_pointwise()
        self.initialize_weights()
        pred = self._build(self.embeddings)
        # 损失和精确度
        self.y_hat, self.total_loss= self._add_loss_op(pred)
        # 训练节点
        self.train_op = self._add_train_op(self.total_loss)

    def _placeholder_init_pointwise(self):
        self._ques = tf.placeholder(tf.int32, [None, self.ques_len], name='ques_point')
        self._ans = tf.placeholder(tf.int32, [None, self.ans_len], name='ans_point')
        self._ques_mask = tf.placeholder(tf.int32, [None], 'ques_mask')
        self._ans_mask = tf.placeholder(tf.int32, [None], 'ans_mask')
        self._y = tf.placeholder(tf.int32, [None])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size, self.list_size = tf.shape(self._ans)[0], tf.shape(self._ans)[1]

    def initialize_weights(self):
        """Global initialization of weights for the representation layer

        """
        # preprocessing
        self.W_i = weight_variable('W_i', [self.hidden_size, self.hidden_size])
        self.W_l1 = weight_variable('W_l1', [5 * self.hidden_size, self.hidden_size])
        self.W_l2 = weight_variable('W_l2', [self.hidden_size, 2])

        self.b_i = bias_variable('b_i', [self.hidden_size])
        self.b_l1 = bias_variable('b_l1', [self.hidden_size])
        self.b_l2 = bias_variable('b_l2', [2])


    def proj_layer(self, seq, out_size, name, reuse=None):
        """
        投影层
        """
        assert len(seq.get_shape()) == 3
        out1 = self.mlp(seq, out_size, 1,
                        tf.nn.sigmoid, name + '_sigmoid', reuse=reuse)
        out2 = self.mlp(seq, out_size, 1,
                        tf.nn.tanh, name + '_tanh', reuse=reuse)
        out = out1 * out2
        return out

    def biLSTMBlock(self, inputs, num_units, scope, rnn_type, dropout_keep_prob, seq_len=None, isReuse=None):
        with tf.variable_scope(scope, reuse=isReuse):
            if rnn_type == 'LSTM':
                lstmCell = LSTMCell(num_units=num_units)
            elif rnn_type == 'GRU':
                lstmCell = GRUCell(num_units=num_units)
            dropLSTMCell = lambda: DropoutWrapper(lstmCell, output_keep_prob=dropout_keep_prob)
            fwLSTMCell, bwLSTMCell = dropLSTMCell(), dropLSTMCell()
            output = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwLSTMCell,
                                                     cell_bw=bwLSTMCell,
                                                     inputs=inputs,
                                                     sequence_length=seq_len,
                                                     dtype=tf.float32)
            return output

    def _preprocess_layer(self, question, answer, out_size):
        # 对应于原文公式（1）,得到Q,A
        with tf.variable_scope('context_encoding') as scope:
            q_encode = self.proj_layer(question, out_size, 'proj_layer', reuse=None)
            a_encode = self.proj_layer(answer, out_size, 'proj_layer', reuse=True)
        return q_encode, a_encode

    def _preprocess_layer2(self, question, answer, out_size, reuse=None):
        # 不共享参数
        out1 = self.mlp(question, out_size, 1,
                        tf.nn.relu, 'proj_layer' + 'ques_relu', reuse=reuse)
        out2 = self.mlp(answer, out_size, 1,
                        tf.nn.relu, 'proj_layer' + 'ans_relu', reuse=reuse)
        return out1, out2

    def _preprocess_layer3(self, question, answer, out_size):
        # 共享参数
        out1 = self.mlp(question, out_size, 1,
                        tf.nn.relu, 'proj_layer' + 'relu', reuse=None)
        out2 = self.mlp(answer, out_size, 1,
                        tf.nn.relu, 'proj_layer' + 'relu', reuse=True)
        return out1, out2

    def _preprocess_layer4(self, question, answer, out_size):
        # 共享参数
        # bilstm
        rnn_outputs_left, final_state_left = self.biLSTMBlock(question, out_size, 'R', 'LSTM',
                                                         self.dropout_keep_prob, self._ques_mask)
        rnn_outputs_right, final_state_right =self.biLSTMBlock(answer, out_size, 'R', 'LSTM',
                                                           self.dropout_keep_prob, self._ans_mask, isReuse=True)
        rnn_q = tf.concat(rnn_outputs_left, axis=2)  # (bz*ls, q_len, 2rz)
        rnn_a = tf.concat(rnn_outputs_right, axis=2)  # (bz*ls, a_len, 2rz)
        print_shape('rnn_q', rnn_q)
        print_shape('rnn_a', rnn_a)
        return rnn_q, rnn_a

    def _attention_layer(self, question, answer, question_mask, answer_mask):
        """
       q: [batch_size, q_length, represent_dim]
       a: [batch_size, a_length, represent_dim]
       q_mask : [bz, q_len] -> 3d
       a_mask : [bz, a_len] -> 3d
       """
        question = tf.reshape(question, [-1, self.hidden_size])
        question = tf.nn.xw_plus_b(question, self.W_i, self.b_i)
        question = tf.reshape(question, [-1, self.ques_len, self.hidden_size])
        att_inner_product = tf.matmul(question, tf.transpose(answer, (0, 2, 1)))  # [batch_size, q_length, a_length]
        question_mask = tf.expand_dims(question_mask, axis=-1)
        answer_mask = tf.expand_dims(answer_mask, axis=1)
        q_softmax = attention_softmax_3d_align(att_inner_product, question_mask, dim=1)
        # we set all items to zero that correspond to zero-padded positions of the answer
        G_zero = tf.multiply(q_softmax, answer_mask)
        output_a = tf.matmul(tf.transpose(G_zero, [0, 2, 1]), question)

        a_softmax = attention_softmax_3d_align(att_inner_product, answer_mask, dim=-1)
        G_zero_ = tf.multiply(a_softmax, question_mask)
        output_q = tf.matmul(G_zero_, answer)  # [batch_size, q_length, 2hz]
        return output_a, output_q

    def _attention_layer2(self, question, answer, question_mask, answer_mask):
        """
       q: [batch_size, q_length, represent_dim]
       a: [batch_size, a_length, represent_dim]
       q_mask : [bz, q_len] -> 3d
       a_mask : [bz, a_len] -> 3d
       """
        question = tf.nn.xw_plus_b(question, self.W_i, self.b_i)
        # question = tf.multiply(question, tf.expand_dims(question_mask, axis=-1))
        att_inner_product = tf.matmul(question, tf.transpose(answer, (0, 2, 1)))  # [batch_size, q_length, a_length]
        question_mask = tf.expand_dims(question_mask, axis=-1)
        answer_mask = tf.expand_dims(answer_mask, axis=1)
        q_softmax = attention_softmax_3d_align(att_inner_product, question_mask, dim=1)
        # we set all items to zero that correspond to zero-padded positions of the answer
        G_zero = tf.multiply(q_softmax, answer_mask)
        output_a = tf.matmul(tf.transpose(G_zero, [0, 2, 1]), question)

        a_softmax = attention_softmax_3d_align(att_inner_product, answer_mask, dim=-1)
        G_zero_ = tf.multiply(a_softmax, question_mask)
        output_q = tf.matmul(G_zero_, answer)  # [batch_size, q_length, 2hz]
        return output_a, output_q

    def _compare_layer(self, q, q_align, a, a_align, comp_type):
        """
        a: [batch_size, a_length, 2hz]
        a_att: [batch_size, a_length, 2hz]
        fuse_A: [batch_size, a_length, 2hz]
        fuse_Q: [batch_size, q_length, 2hz]
        """
        size = q.get_shape()[-1]
        if comp_type == 'Gate_fuse':
            fuse_a = tf.concat([a, a_align, a * a_align, a - a_align], axis=2)
            fuse_q = tf.concat([q, q_align, q * q_align, q - q_align], axis=2)
            fuse_a_sigmoid = self.mlp(fuse_a, size, 1, tf.nn.sigmoid, 'fuse_a_sigmoid',
                                      use_dropout=False, bias=True)
            fuse_q_sigmoid = self.mlp(fuse_q, size, 1, tf.nn.sigmoid, 'fuse_q_sigmoid',
                                      use_dropout=False, bias=True)
            fuse_a_tanh = self.mlp(fuse_a, size, 1, tf.nn.tanh, 'fuse_a_tanh',
                                      use_dropout=False, bias=True)
            fuse_q_tanh = self.mlp(fuse_q, size, 1, tf.nn.tanh, 'fuse_q_tanh',
                                      use_dropout=False, bias=True)
            fuse_A = fuse_a_sigmoid * fuse_a_tanh + a - fuse_a_sigmoid*a
            fuse_Q = fuse_q_sigmoid * fuse_q_tanh + q - fuse_q_sigmoid*q
        elif comp_type == 'simple_fuse':
            fuse_A = tf.concat([a, a_align, a * a_align, a - a_align], axis=2)
            fuse_Q = tf.concat([q, q_align, q * q_align, q - q_align], axis=2)
        elif comp_type == 'mul':
            fuse_A = a * a_align
            fuse_Q = q * q_align
        else:
            raise ValueError('{} method is not implemented!'.format(comp_type))
        return fuse_A, fuse_Q

    def _cnn_layer(self, input, mask, name, isreuse=False, dim = -1):
        """
        :param
        :return:
        """
        # tf.layers.Conv1D(inputs, filters, kernel_size, strides=1)
        # self.n_filters一般指词嵌入的维度
        with tf.variable_scope(name, reuse=isreuse) as scope1:
            all = []
            for i in range(len(self.window_sizes)):
                cnn_out = tf.layers.conv1d(input, self.n_filters, self.window_sizes[i], padding='same',
                                           activation=tf.nn.relu, name='q_conv_' + str(i))
                all.append(cnn_out)
            cnn_outs = tf.concat(all, axis=-1)
            cnn_outs_padded = tf.multiply(cnn_outs, tf.expand_dims(mask, axis=dim))
            R_max = maxpool(cnn_outs_padded)
            R_men = meanpool(cnn_outs_padded)
            print_shape('R', R_max)
        return R_max, R_men

    def _build(self, embeddings, encoder_type = 'sigmoid'):
        self.Embedding = tf.Variable(tf.to_float(embeddings), trainable=False, name='Embedding')
        self.q_embed = tf.nn.embedding_lookup(self.Embedding, self._ques)
        self.a_embed = tf.nn.embedding_lookup(self.Embedding, self._ans)
        q_mask = tf.sequence_mask(self._ques_mask, self.ques_len, dtype=tf.float32)
        a_mask = tf.sequence_mask(self._ans_mask, self.ans_len, dtype=tf.float32)

        # ~~ Preprocessing
        if encoder_type == 'lstm':
            question, answer = self._preprocess_layer4(self.q_embed, self.a_embed, self.rnn_size)  # (bz, len, 2hz)
        elif encoder_type == 'relu_share':
            question, answer = self._preprocess_layer3(self.q_embed, self.a_embed, self.rnn_size)  # (bz, len, hz)
        elif encoder_type == 'relu_noshare':
            question, answer = self._preprocess_layer2(self.q_embed, self.a_embed, self.rnn_size)  # (bz, len, hz)
        elif encoder_type == 'sigmoid':
            question, answer = self._preprocess_layer(self.q_embed, self.a_embed, self.hidden_size)  # (bz, len, hz)

        # ~~ Attention and Comparison
        algin_a, algin_q = self._attention_layer(question, answer, q_mask, a_mask)
        fuse_A, fuse_Q = self._compare_layer(question, algin_q, answer, algin_a, comp_type='mul')

        # ~~ Aggregation
        # self, input, mask, name, isreuse=False, dim = 1
        R_max_a, R_men_a = self._cnn_layer(fuse_A, a_mask, 'cnn_answer')
        # R_max_q, R_men_q = self._cnn_layer(fuse_Q, q_mask, 'cnn_question')

        # ~~ Predict
        # M = tf.concat([R_men_q, R_max_q, R_men_a, R_max_a], axis=-1)  # (?, 1, 16*2hz)
        M = R_max_a

        fc1 = tf.nn.xw_plus_b(M, self.W_l1, self.b_l1)
        fc1 = tf.nn.tanh(fc1)
        fc2 = tf.nn.xw_plus_b(fc1, self.W_l2, self.b_l2)
        predict = fc2
        print_shape('predict:', predict)
        return predict

    def _add_loss_op(self, pred, l2_lambda=0.00001):
        """
        损失节点
        """
        y_hat = tf.nn.softmax(pred, dim=-1)
        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(self._y, pred))
        tf.add_to_collection('total_loss', loss)
        total_loss = tf.add_n(tf.get_collection('total_loss'))
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        l2_loss = sum(reg_losses) * l2_lambda
        pointwise_loss = total_loss + l2_loss
        tf.summary.scalar('pointwise_loss', pointwise_loss)
        self.summary_op = tf.summary.merge_all()
        return y_hat, pointwise_loss

    def _add_train_op(self, loss):
        """
        训练节点
        """
        with tf.name_scope('train_op'):
            # 记录训练步骤
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            opt = tf.train.AdamOptimizer(self.learning_rate)
            # train_op = opt.minimize(loss, self.global_step)
            gradients, v = zip(*opt.compute_gradients(loss))
            clip_gradients = gradients
            if self.clip_value is not None:
                clip_gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)
            train_op = opt.apply_gradients(zip(clip_gradients, v), global_step=self.global_step)
            return train_op

    def mlp(self, bottom, size, layer_num, activation, name, use_dropout=True, reuse=None, bias = True):
        """
        bottom: 上层输入
        size: 神经元大小
        layer_num: 神经网络层数
        name: mlp的名称
        reuse: 是否复用层
        initializer: w和b的初始化均采用xavier_initializer()
        """

        now = bottom
        if use_dropout:
            now = tf.nn.dropout(now, keep_prob=self.dropout_keep_prob)
        for i in range(layer_num):
            now = tf.layers.dense(now, size,
                                  activation=activation,
                                  name=name + '_{}'.format(i),
                                  reuse=reuse,
                                  use_bias=bias,
                                  kernel_initializer=xavier_initializer(),
                                  bias_initializer=xavier_initializer()
                                  )
        return now

