#!/usr/bin/env python
#-*-coding:utf-8-*-

from __future__ import print_function
import numpy as np
import tensorflow as tf
import pickle
#np.random.seed(1337)
#tf.set_random_seed(1337)

class Base_Line():
    def __init__(self,model_params):
        self.hidden_dim = model_params.hidden_dim
        self.ques_len = model_params.ques_len
        self.ans_len = model_params.ans_len
        self.embedding_file = model_params.embedding_file
        self.keep_prob = model_params.keep_prob
        #self._build_base_line_pointwise()

    def _build_base_line_pointwise(self):
        with tf.variable_scope('input') as input_l:
            self._ques = tf.placeholder(tf.int32,[None,self.ques_len],name='ques_point')
            self._ques_len = tf.placeholder(tf.float32,[None,self.ques_len],name='ques_len_point')
            self._ans = tf.placeholder(tf.int32,[None,self.ans_len],name='ans_point')
            self._ans_len = tf.placeholder(tf.float32,[None,self.ans_len],name='ans_len_point')

            self._ques_filter_len = tf.tile(tf.reshape(self._ans_len,[-1,1,self.ans_len]),[1,self.ques_len,1])  # (-1, ques_len, ans_len)
            self._ans_filter_len = tf.tile(tf.reshape(self._ques_len,[-1,1,self.ques_len]),[1,self.ans_len,1])  # (-1, ans_len, ques_len, )

            self._ques_align_len = tf.tile(tf.reshape(self._ques_len,[-1,self.ques_len,1]),[1,1,self.hidden_dim])  # (-1, ques_len, hd)
            self._ans_align_len = tf.tile(tf.reshape(self._ans_len,[-1,self.ans_len,1]),[1,1,self.hidden_dim])  # (-1, ans_len, hd)

            self.p_label = tf.placeholder(tf.float32,[None,])
            #self.l_label = tf.placeholder(tf.float32,[None,self.list_size])
        with tf.name_scope('list_wise'):
            with tf.variable_scope('embedding_layer') as embedding_l:
                weights = np.load(self.embedding_file)
                weights[0] = np.zeros((weights.shape[1]))
                embeddings = tf.Variable(weights,dtype=tf.float32)

                ques_emb = tf.nn.embedding_lookup(embeddings,self._ques)
                ans_emb = tf.nn.embedding_lookup(embeddings,self._ans)
                print('ques_emb:',ques_emb.shape)
                print('ans_emb',ans_emb.shape)
            with tf.variable_scope('preprocess_layer') as prep_l:
                sig_den = tf.layers.Dense(self.hidden_dim,activation=tf.sigmoid,name='sigmoid_dense')
                tan_den = tf.layers.Dense(self.hidden_dim,activation=tf.tanh,name='tanh_dense')

                ques_sig = sig_den(ques_emb)
                ques_tan = tan_den(ques_emb)
                ques_h = tf.multiply(ques_sig,ques_tan)

                ans_sig = sig_den(ans_emb)
                ans_tan = tan_den(ans_emb)
                ans_h = tf.multiply(ans_sig,ans_tan)
            with tf.variable_scope('attention_softalign') as att_align_l:
                ques_att_matrix = self.getAttMat(ques_h,ans_h)  # (bz, q_len, a_len)
                ans_att_matrix = self.getAttMat(ans_h,ques_h)  # (bz, a_len, q_len)
                print('ques_att_matrix:',ques_att_matrix.shape)
                ques_align = self.getAlign(ans_h,ques_att_matrix,self._ques_filter_len)
                ans_align = self.getAlign(ques_h,ans_att_matrix,self._ans_filter_len)
                print('ques_align:',ques_align.shape)

                ques_aligned = tf.multiply(tf.multiply(ques_align,ques_h),self._ques_align_len)
                ans_aligned = tf.multiply(tf.multiply(ans_align,ans_h),self._ans_align_len)
            with tf.variable_scope('cnn_feature') as cnn_l:
                cnn_ques = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='q_conv_'+str(i)) for i in range(1,6)]
                cnn_ans = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='a_conv_'+str(i)) for i in range(1,6)]

                ques_cnn = self.conv1d_listwise(cnn_ques,ques_aligned)
                ans_cnn = self.conv1d_listwise(cnn_ans,ans_aligned)
                print('ques_cnn:',ques_cnn.shape)
            with tf.variable_scope('output_layer') as out_l:
                ques_o1 = tf.layers.dense(ques_cnn,self.hidden_dim,activation=tf.tanh,name='q_out1')
                ans_o1 = tf.layers.dense(ans_cnn,self.hidden_dim,activation=tf.tanh,name='a_out1')

                finalo1 = tf.concat([ques_o1,ans_o1],axis=-1)

                finalo2 = tf.layers.dense(finalo1,self.hidden_dim,activation=tf.tanh,name='finalout')
                self.score = tf.layers.dense(finalo2,1,name='score')
                print('score:',self.score.shape)
            #with tf.variable_scope('loss') as loss_l:
            #    self.loss_pointwise = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.p_label,logits=tf.squeeze(self.score,-1)))
            #    self.loss_listwise = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.l_label,logits=tf.squeeze(self.score,-1)))

            #    self.total_loss = self.loss_pointwise + self.loss_listwise


    # -----------------------------------------------------------------------------------------------------------------


    def _build_base_line_listwise(self):
        with tf.variable_scope('inputs') as inputs:
            self.r_ques = tf.placeholder(tf.int32,[None,None,self.ques_len],name='ques_point')
            self.r_ques_len = tf.placeholder(tf.float32,[None,None,self.ques_len],name='ques_len_point')
            self.r_ans = tf.placeholder(tf.int32,[None,None,self.ans_len],name='ans_point')
            self.r_ans_len = tf.placeholder(tf.float32,[None,None,self.ans_len],name='ans_len_point')

            self._ques_filter_len = tf.tile(tf.expand_dims(self.r_ans_len,2),[1,1,self.ques_len,1])  # (bz, ls, q_len, a_len)
            self._ans_filter_len = tf.tile(tf.expand_dims(self.r_ques_len,2),[1,1,self.ans_len,1])

            self._ques_align_len = tf.tile(tf.expand_dims(self.r_ques_len,3),[1,1,1,self.hidden_dim])
            self._ans_align_len = tf.tile(tf.expand_dims(self.r_ans_len,3),[1,1,1,self.hidden_dim])

            #self.p_label = tf.placeholder(tf.float32,[None,None])
            self.l_label = tf.placeholder(tf.float32,[None,None])

            self.is_train = tf.placeholder(tf.bool)

            self._ques = self.r_ques
            self._ques_len = self.r_ques_len
            self._ans = self.r_ans
            self._ans_len = self.r_ans_len

            batch_size, list_size = tf.shape(self._ans)[0], tf.shape(self._ans)[1]
            self.dc = tf.placeholder(tf.bool)

        with tf.name_scope('list_wise'):
            with tf.variable_scope('embedding_layer') as embedding_l:
                with open(self.embedding_file, 'rb') as fin:
                    weights = pickle.load(fin)
                embeddings = tf.Variable(weights,dtype=tf.float32,trainable=False)

                ques_emb = tf.nn.embedding_lookup(embeddings,self._ques)
                ans_emb = tf.nn.embedding_lookup(embeddings,self._ans)
                print('ques_emb:',ques_emb.shape)
                print('ans_emb',ans_emb.shape)
            with tf.variable_scope('preprocess_layer') as prep_l:
                sig_den = tf.layers.Dense(self.hidden_dim,name='q_sigmoid_dense')
                tan_den = tf.layers.Dense(self.hidden_dim,name='q_tanh_dense')

                #a_sig_den = tf.layers.Dense(self.hidden_dim,name='a_sigmoid_dense')
                #a_tan_den = tf.layers.Dense(self.hidden_dim,name='a_tanh_dense')

                ques_sig = sig_den(ques_emb)
                ques_tan = tan_den(ques_emb)
                #ques_sig = tf.layers.batch_normalization(ques_sig,training=self.is_train)
                #ques_tan = tf.layers.batch_normalization(ques_tan,training=self.is_train)
                ques_sig = tf.sigmoid(ques_sig)
                ques_tan = tf.tanh(ques_tan)
                ques_h = tf.multiply(ques_sig,ques_tan)
                #ques_h = ques_emb

                ans_sig = sig_den(ans_emb)
                ans_tan = tan_den(ans_emb)
                #ans_sig = tf.layers.batch_normalization(ans_sig,training=self.is_train)
                #ans_tan = tf.layers.batch_normalization(ans_tan,training=self.is_train)
                ans_sig = tf.sigmoid(ans_sig)
                ans_tan = tf.tanh(ans_tan)
                ans_h = tf.multiply(ans_sig,ans_tan)
                #ans_h = ans_emb
            with tf.variable_scope('attention_softalign') as att_align_l:
                ques_att_matrix = self.getAttMat(ques_h,ans_h)
                ans_att_matrix = self.getAttMat(ans_h,ques_h)
                print('ques_att_matrix:',ques_att_matrix.shape)
                ques_align = self.getAlign(ans_h,ques_att_matrix,self._ques_filter_len)  # (bz, ls, _q_len, hd)
                ans_align = self.getAlign(ques_h,ans_att_matrix,self._ans_filter_len)
                print('ques_align:',ques_align.shape)

                ques_aligned = tf.multiply(tf.multiply(ques_align,ques_h),self._ques_align_len)
                ans_aligned = tf.multiply(tf.multiply(ans_align,ans_h),self._ans_align_len)
            with tf.variable_scope('cnn_feature') as cnn_l:
                self.cnn_ques = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='q_conv_'+str(i)) for i in range(1,6)]
                self.cnn_ans = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='a_conv_'+str(i)) for i in range(1,6)]

                ques_aligned = tf.reshape(ques_aligned,shape=(-1,self.ques_len,self.hidden_dim))
                ans_aligned = tf.reshape(ans_aligned,shape=(-1,self.ans_len,self.hidden_dim))

                ques_cnn_filter = tf.reshape(self._ques_align_len,shape=(-1,self.ques_len,self.hidden_dim))
                ans_cnn_filter = tf.reshape(self._ans_align_len,shape=(-1,self.ans_len,self.hidden_dim))

                ques_cnn = self.conv1d_listwise(self.cnn_ques,ques_aligned,ques_cnn_filter)
                ans_cnn = self.conv1d_listwise(self.cnn_ans,ans_aligned,ans_cnn_filter)

                ques_cnn = tf.reshape(ques_cnn,shape=(batch_size,list_size,self.hidden_dim*len(self.cnn_ques)))
                ans_cnn = tf.reshape(ans_cnn,shape=(batch_size,list_size,self.hidden_dim*len(self.cnn_ans)))
                #ques_cnn = tf.concat([self.conv1d_listwise(self.cnn_ques,ques_aligned[:,i,:,:],keep_dims=True) for i in range(ques_aligned.shape[1])],axis=1)
                #ans_cnn = tf.concat([self.conv1d_listwise(self.cnn_ans,ans_aligned[:,i,:,:],keep_dims=True) for i in range(ques_aligned.shape[1])],axis=1)
                #def _conv1d_listwise(step,sent_cnn,sent_aligned,signal):
                #    conv1dfn = self.cnn_ques if tf.equal(signal,tf.constant(1)) is not None else self.cnn_ans
                #    sent_cnn = tf.concat([sent_cnn,self.conv1d_listwise(conv1dfn,sent_aligned[:,step,:,:],True)],1)
                #    return step+1,sent_cnn,sent_aligned,signal
                #ques_cnn = tf.zeros([tf.shape(ques_aligned)[0],1,self.hidden_dim*len(self.cnn_ques)],dtype=tf.float32)
                #ans_cnn = tf.zeros([tf.shape(ques_aligned)[0],1,self.hidden_dim*len(self.cnn_ans)],dtype=tf.float32)
                #step = tf.constant(0)
                #signal = tf.constant(1)
                #_,ques_cnn,_,_ = tf.while_loop(cond=lambda step,*_: step<tf.shape(ques_aligned)[1],
                #                        body=_conv1d_listwise,
                #                        loop_vars=[step,ques_cnn,ques_aligned,signal],
                #                               shape_invariants=[step.get_shape(),tf.TensorShape([ques_cnn.shape[0],None,ques_cnn.shape[2]]),ques_aligned.get_shape(),signal.get_shape()])
                #step = tf.constant(0)
                #signal = tf.constant(0)
                #_,ans_cnn,_,_ = tf.while_loop(cond=lambda step,*_: step<tf.shape(ans_aligned)[1],
                #                        body=_conv1d_listwise,
                #                        loop_vars=[step,ans_cnn,ans_aligned,signal],
                #                               shape_invariants=[step.get_shape(),tf.TensorShape([ans_cnn.shape[0],None,ans_cnn.shape[2]]),ans_aligned.get_shape(),signal.get_shape()])
                #ques_cnn = ques_cnn[:,1:,:]
                #ans_cnn = ans_cnn[:,1:,:]
                print('ques_cnn:',ques_cnn.shape)
                print('ans_cnn:',ans_cnn.shape )
            with tf.variable_scope('output_layer') as out_l:
                ques_o1 = tf.layers.dense(ques_cnn,self.hidden_dim,activation=tf.tanh,name='q_out1')
                ans_o1 = tf.layers.dense(ans_cnn,self.hidden_dim,activation=tf.tanh,name='a_out1')

                finalo1 = tf.concat([ques_o1,ans_o1],axis=-1)

                finalo2 = tf.layers.dense(finalo1,self.hidden_dim,activation=tf.tanh,name='finalout')
                self.score = tf.layers.dense(finalo2,1,name='score')
                print('score:',self.score.shape)
                self.logit_score = tf.nn.log_softmax(tf.squeeze(self.score,-1),dim=-1)
                print('logit_score:',self.logit_score.shape)
            with tf.variable_scope('loss') as loss_l:
                #self.loss_pointwise = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.p_label,logits=tf.squeeze(self.score,-1)))
                self.loss_listwise = tf.reduce_mean(self.l_label*(tf.log(tf.clip_by_value(self.l_label,1e-5,1.0))-self.logit_score))

                #self.total_loss = self.loss_pointwise + self.loss_listwise

    @staticmethod
    def getAttMat(sent1,sent2):
        return tf.matmul(sent1,sent2,transpose_b=True)

    @staticmethod
    #  self.getAlign(ans_h,ques_att_matrix,self._ques_filter_len)
    #  ans_h: (bz, ls, a_len, hd)
    #  ques_filter_len: (bz, ls, _q_len, a_len)
    #  return: (bz, ls, _q_len, hd)
    def getAlign(sent,matrix,sent_len):
        matrix_e = tf.exp(matrix-tf.reduce_max(matrix,-1,keep_dims=True))
        matrix_e_true = tf.multiply(matrix_e,sent_len)  # (bz, ls, _q_len, a_len)
        matrix_s = tf.reduce_sum(matrix_e_true,-1,keep_dims=True)  # (bz, ls, _q_len, 1)
        matrix_sm = matrix_e_true/matrix_s  # (bz, ls, _q_len, a_len)
        return tf.matmul(matrix_sm,sent)

    @staticmethod

    # ques_cnn = self.conv1d_listwise(self.cnn_ques, ques_aligned, ques_cnn_filter)
    def conv1d_listwise(conv1dfn,sent,sent_len,keep_dims=False):
        cnn_out = tf.concat([conv1dfn[i](sent)*sent_len for i in range(len(conv1dfn))],axis=-1)
        maxpool_out = tf.reduce_max(cnn_out,1,keep_dims=keep_dims)
        return maxpool_out



if __name__ == '__main__':
    class Model_Param():
        batch_size = 10
        hidden_dim = 200
        list_size = 15
        ques_len = 30
        ans_len = 40
        keep_prob = 0.5
        embedding_file = '/data/wikiqa/self/raw/wiki_embedding.pkl'
    m_p = Model_Param()
    base_line = Base_Line(m_p)
    base_line._build_base_line_listwise()
