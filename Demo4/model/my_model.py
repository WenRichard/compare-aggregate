#!/usr/bin/env python
#-*-coding:utf-8-*-

from __future__ import print_function
import numpy as np
import tensorflow as tf
#np.random.seed(1337)
#tf.set_random_seed(1337)

class My_Model():
    def __init__(self,model_params):
        self.hidden_dim = model_params.hidden_dim
        self.ques_len = model_params.ques_len
        self.ans_len = model_params.ans_len
        self.embedding_file = model_params.embedding_file
        #self._build_base_line_pointwise()

    def _build_my_model_pointwise(self):
        with tf.variable_scope('input') as input_l:
            self._ques = tf.placeholder(tf.int32,[None,self.ques_len],name='ques_point')
            self._ques_len = tf.placeholder(tf.float32,[None,self.ques_len],name='ques_len_point')
            self._ans = tf.placeholder(tf.int32,[None,self.ans_len],name='ans_point')
            self._ans_len = tf.placeholder(tf.float32,[None,self.ans_len],name='ans_len_point')

            self._ques_filter_len = tf.tile(tf.reshape(self._ans_len,[-1,1,self.ans_len]),[1,self.ques_len,1])
            self._ans_filter_len = tf.tile(tf.reshape(self._ques_len,[-1,1,self.ques_len]),[1,self.ans_len,1])

            self._ques_align_len = tf.tile(tf.reshape(self._ques_len,[-1,self.ques_len,1]),[1,1,self.hidden_dim])
            self._ans_align_len = tf.tile(tf.reshape(self._ans_len,[-1,self.ans_len,1]),[1,1,self.hidden_dim])

            self.p_label = tf.placeholder(tf.float32,[None,])
            #self.l_label = tf.placeholder(tf.float32,[None,self.list_size])

            self.is_train = tf.placeholder(tf.bool)
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
                ques_att_matrix = self.getAttMat(ques_h,ans_h)
                ans_att_matrix = self.getAttMat(ans_h,ques_h)
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

    def _build_syn_ext_listwise(self):
        with tf.variable_scope('inputs') as inputs:
            self.r_ques = tf.placeholder(tf.int32,[None,None,self.ques_len],name='ques_point')
            self.r_ques_len = tf.placeholder(tf.float32,[None,None,self.ques_len],name='ques_len_point')
            self.r_ans = tf.placeholder(tf.int32,[None,None,self.ans_len],name='ans_point')
            self.r_ans_len = tf.placeholder(tf.float32,[None,None,self.ans_len],name='ans_len_point')

            self._ques_filter_len = tf.tile(tf.expand_dims(self.r_ans_len,2),[1,1,self.ques_len,1])
            self._ans_filter_len = tf.tile(tf.expand_dims(self.r_ques_len,2),[1,1,self.ans_len,1])

            self._ques_align_len = tf.expand_dims(self.r_ques_len,3)
            self._ans_align_len = tf.expand_dims(self.r_ans_len,3)

            #self.p_label = tf.placeholder(tf.float32,[None,None])
            self.l_label = tf.placeholder(tf.float32,[None,None])

            self.batch_size, self.list_size = tf.shape(self.r_ans)[0], tf.shape(self.r_ans)[1]

            self._ques = tf.reshape(self.r_ques,shape=(-1,self.ques_len))
            self._ans = tf.reshape(self.r_ans,shape=(-1,self.ans_len))
            self._ques_filter_len = tf.reshape(self._ques_filter_len,shape=(-1,self.ques_len,self.ans_len))
            self._ans_filter_len = tf.reshape(self._ans_filter_len,shape=(-1,self.ans_len,self.ques_len))
            self._ques_align_len = tf.reshape(self._ques_align_len,shape=(-1,self.ques_len,1))
            self._ans_align_len = tf.reshape(self._ans_align_len,shape=(-1,self.ans_len,1))

            self.is_train = tf.placeholder(tf.bool)
            self.dc = tf.placeholder(tf.bool)
        with tf.name_scope('list_wise'):
            with tf.variable_scope('embedding_layer') as embedding_l:
                weights = np.load(self.embedding_file)
                weights[0] = np.zeros((weights.shape[1]))
                embeddings = tf.Variable(weights,dtype=tf.float32,trainable=False)

                ques_emb = tf.nn.embedding_lookup(embeddings,self._ques)
                ans_emb = tf.nn.embedding_lookup(embeddings,self._ans)
                print('ques_emb:',ques_emb.shape)
                print('ans_emb',ans_emb.shape)
            with tf.variable_scope('preprocess_layer') as prep_l:
                sig_den = tf.layers.Dense(self.hidden_dim,name='sigmoid_dense')
                tan_den = tf.layers.Dense(self.hidden_dim,name='tanh_dense')

                ques_sig = sig_den(ques_emb)
                ques_sig = tf.sigmoid(ques_sig)
                ques_tan = tan_den(ques_emb)
                ques_tan = tf.tanh(ques_tan)
                ques_h = tf.multiply(ques_sig,ques_tan)

                ans_sig = sig_den(ans_emb)
                ans_sig = tf.sigmoid(ans_sig)
                ans_tan = tan_den(ans_emb)
                ans_tan = tf.tanh(ans_tan)
                ans_h = tf.multiply(ans_sig,ans_tan)
            with tf.variable_scope('synatic_extract') as syn_ext_l:
                q_syn_extractor = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='q_extrator_'+str(i)) for i in range(2,6)]
                a_syn_extractor = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='q_extrator_'+str(i)) for i in range(2,6)]

                ques_h = tf.multiply(ques_h,self._ques_align_len)
                ans_h = tf.multiply(ans_h,self._ans_align_len)

                ques_syn = self.convXd_listwise(q_syn_extractor,ques_h,self._ques_align_len)
                ans_syn = self.convXd_listwise(a_syn_extractor,ans_h,self._ans_align_len)

                print('ques_syn:',ques_syn.shape)
                ques_syn = tf.reshape(ques_syn,shape=(-1,self.ques_len,len(q_syn_extractor),self.hidden_dim))
                ans_syn = tf.reshape(ans_syn,shape=(-1,self.ans_len,len(a_syn_extractor),self.hidden_dim))

                ques_syn = tf.reduce_max(ques_syn,axis=1)
                ans_syn = tf.reduce_max(ans_syn,axis=1)
                print('ques_syn:',ques_syn.shape)

                #q_a_syn = tf.concat([ques_syn,ans_syn],axis=-1)

                #q_a_syn = tf.layers.dropout(q_a_syn,rate=0.5,training=self.is_train)

                #q_a_syn = tf.layers.dense(q_a_syn,self.hidden_dim,name='q_a_syn')
                #q_a_syn = tf.nn.tanh(q_a_syn)

            with tf.variable_scope('thresh_net') as thresh_l:
                Thresh = tf.layers.Dense(1,kernel_initializer=tf.initializers.zeros,name='threshold')

                ques_att_thresh = Thresh(ques_h)*self._ques_align_len
                ans_att_thresh = Thresh(ans_h)*self._ans_align_len

            with tf.variable_scope('attention_softalign') as att_align_l:
                ques_att_matrix1 = self.getAttMat(ques_h,ans_h)
                ques_att_matrix2 = self.getAttMat(ques_h,ans_syn)
                ans_att_matrix1 = self.getAttMat(ans_h,ques_h)
                ans_att_matrix2 = self.getAttMat(ans_h,ques_syn)
                print('ques_att_matrix1:',ques_att_matrix1.shape)
                print('ques_att_matrix2:',ques_att_matrix2.shape)
                ques_align1 = self.getAlign(ans_h,ques_att_matrix1,self._ques_filter_len)
                ques_align2 = self.getAlign(ans_syn,ques_att_matrix2)
                ans_align1 = self.getAlign(ques_h,ans_att_matrix1,self._ans_filter_len)
                ans_align2 = self.getAlign(ques_syn,ans_att_matrix2)

                ques_align = tf.multiply(ques_align1,ques_align2)
                ans_align = tf.multiply(ans_align1,ans_align2)
                print('ques_align:',ques_align.shape)

                ques_aligned = tf.multiply(tf.multiply(ques_align,ques_h),self._ques_align_len)
                ans_aligned = tf.multiply(tf.multiply(ans_align,ans_h),self._ans_align_len)
            with tf.variable_scope('cnn_feature') as cnn_l:
                self.cnn_ques = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='q_conv_'+str(i)) for i in range(1,6)]
                self.cnn_ans = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='a_conv_'+str(i)) for i in range(1,6)]

                ques_cnn = self.convXd_listwise(self.cnn_ques,ques_aligned,self._ques_align_len,1)
                ans_cnn = self.convXd_listwise(self.cnn_ans,ans_aligned,self._ans_align_len,1)

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

                #finalo1 = tf.concat([ques_o1,q_a_syn,ans_o1],axis=-1)
                finalo1 = tf.concat([ques_o1,ans_o1],axis=-1)

                finalo2 = tf.layers.dense(finalo1,self.hidden_dim,activation=tf.tanh,name='finalout')
                self.score = tf.layers.dense(finalo2,1,name='score')
                print('score:',self.score.shape)
                self.score = tf.reshape(self.score,shape=(self.batch_size,self.list_size))
                self.logit_score = tf.nn.log_softmax(self.score,dim=-1)
                print('logit_score:',self.logit_score.shape)
            with tf.variable_scope('loss') as loss_l:
                #self.loss_pointwise = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.p_label,logits=tf.squeeze(self.score,-1)))
                self.loss_listwise = tf.reduce_mean(self.l_label*(tf.log(tf.clip_by_value(self.l_label,1e-5,1.0))-self.logit_score))

                #self.total_loss = self.loss_pointwise + self.loss_listwise

    def _build_syn_ext2_listwise(self):
        with tf.variable_scope('inputs') as inputs:
            self.r_ques = tf.placeholder(tf.int32,[None,None,self.ques_len],name='ques_point')
            self.r_ques_len = tf.placeholder(tf.float32,[None,None,self.ques_len],name='ques_len_point')
            self.r_ans = tf.placeholder(tf.int32,[None,None,self.ans_len],name='ans_point')
            self.r_ans_len = tf.placeholder(tf.float32,[None,None,self.ans_len],name='ans_len_point')

            self._ques_filter_len = tf.tile(tf.expand_dims(self.r_ans_len,2),[1,1,self.ques_len,1])
            self._ans_filter_len = tf.tile(tf.expand_dims(self.r_ques_len,2),[1,1,self.ans_len,1])

            self._ques_align_len = tf.expand_dims(self.r_ques_len,3)
            self._ans_align_len = tf.expand_dims(self.r_ans_len,3)

            #self.p_label = tf.placeholder(tf.float32,[None,None])
            self.l_label = tf.placeholder(tf.float32,[None,None])

            self.batch_size, self.list_size = tf.shape(self.r_ans)[0], tf.shape(self.r_ans)[1]

            self._ques = tf.reshape(self.r_ques,shape=(-1,self.ques_len))
            self._ans = tf.reshape(self.r_ans,shape=(-1,self.ans_len))
            self._ques_filter_len = tf.reshape(self._ques_filter_len,shape=(-1,self.ques_len,self.ans_len))
            self._ans_filter_len = tf.reshape(self._ans_filter_len,shape=(-1,self.ans_len,self.ques_len))
            self._ques_align_len = tf.reshape(self._ques_align_len,shape=(-1,self.ques_len,1))
            self._ans_align_len = tf.reshape(self._ans_align_len,shape=(-1,self.ans_len,1))

            self.is_train = tf.placeholder(tf.bool)
            self.dc = tf.placeholder(tf.bool)
        with tf.name_scope('list_wise'):
            with tf.variable_scope('embedding_layer') as embedding_l:
                weights = np.load(self.embedding_file)
                weights[0] = np.zeros((weights.shape[1]))
                embeddings = tf.Variable(weights,dtype=tf.float32,trainable=False)

                ques_emb = tf.nn.embedding_lookup(embeddings,self._ques)
                ans_emb = tf.nn.embedding_lookup(embeddings,self._ans)
                print('ques_emb:',ques_emb.shape)
                print('ans_emb',ans_emb.shape)
            with tf.variable_scope('preprocess_layer') as prep_l:
                sig_den = tf.layers.Dense(self.hidden_dim,name='sigmoid_dense')
                tan_den = tf.layers.Dense(self.hidden_dim,name='tanh_dense')

                ques_sig = sig_den(ques_emb)
                ques_sig = tf.sigmoid(ques_sig)
                ques_tan = tan_den(ques_emb)
                ques_tan = tf.tanh(ques_tan)
                ques_h = tf.multiply(ques_sig,ques_tan)

                ans_sig = sig_den(ans_emb)
                ans_sig = tf.sigmoid(ans_sig)
                ans_tan = tan_den(ans_emb)
                ans_tan = tf.tanh(ans_tan)
                ans_h = tf.multiply(ans_sig,ans_tan)
            with tf.variable_scope('synatic_extract') as syn_ext_l:
                q_syn_extractor = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='q_extrator_'+str(i)) for i in range(2,6)]
                a_syn_extractor = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='q_extrator_'+str(i)) for i in range(2,6)]

                self.ques_h = ques_h = tf.multiply(ques_h,self._ques_align_len)
                ans_h = tf.multiply(ans_h,self._ans_align_len)

                self.ques_syn = ques_syn = self.convXd_listwise(q_syn_extractor,ques_h,self._ques_align_len)
                ans_syn = self.convXd_listwise(a_syn_extractor,ans_h,self._ans_align_len)
                print('ques_syn:',ques_syn.shape)

                q_a_syn = tf.concat([ques_syn,ans_syn],axis=-1)

                q_a_syn = tf.layers.dropout(q_a_syn,rate=0.5,training=self.is_train)

                q_a_syn = tf.layers.dense(q_a_syn,self.hidden_dim,name='q_a_syn')
                q_a_syn = tf.nn.tanh(q_a_syn)

            #with tf.variable_scope('thresh_net') as thresh_l:
            #    Thresh = tf.layers.Dense(1,kernel_initializer=tf.initializers.zeros,name='threshold')

            #    ques_att_thresh = Thresh(ques_h)*self._ques_align_len
            #    ans_att_thresh = Thresh(ans_h)*self._ans_align_len

            with tf.variable_scope('attention_softalign') as att_align_l:
                ques_att_matrix = self.getAttMat(ques_h,ans_h)
                ans_att_matrix = self.getAttMat(ans_h,ques_h)
                print('ques_att_matrix:',ques_att_matrix.shape)
                ques_align = self.getAlign(ans_h,ques_att_matrix,self._ques_filter_len)
                ans_align = self.getAlign(ques_h,ans_att_matrix,self._ans_filter_len)
                print('ques_align:',ques_align.shape)

                ques_aligned = tf.multiply(tf.multiply(ques_align,ques_h),self._ques_align_len)
                ans_aligned = tf.multiply(tf.multiply(ans_align,ans_h),self._ans_align_len)
            with tf.variable_scope('cnn_feature') as cnn_l:
                self.cnn_ques = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='q_conv_'+str(i)) for i in range(1,6)]
                self.cnn_ans = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='a_conv_'+str(i)) for i in range(1,6)]

                ques_cnn = self.convXd_listwise(self.cnn_ques,ques_aligned,self._ques_align_len,1)
                ans_cnn = self.convXd_listwise(self.cnn_ans,ans_aligned,self._ans_align_len,1)

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

                finalo1 = tf.concat([ques_o1,q_a_syn,ans_o1],axis=-1)
                #finalo1 = tf.concat([ques_o1,ans_o1],axis=-1)

                finalo2 = tf.layers.dense(finalo1,self.hidden_dim,activation=tf.tanh,name='finalout')
                self.score = tf.layers.dense(finalo2,1,name='score')
                print('score:',self.score.shape)
                self.score = tf.reshape(self.score,shape=(self.batch_size,self.list_size))
                self.logit_score = tf.nn.log_softmax(self.score,dim=-1)
                print('logit_score:',self.logit_score.shape)
            with tf.variable_scope('loss') as loss_l:
                #self.loss_pointwise = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.p_label,logits=tf.squeeze(self.score,-1)))
                self.loss_listwise = tf.reduce_mean(self.l_label*(tf.log(tf.clip_by_value(self.l_label,1e-5,1.0))-self.logit_score))

                #self.total_loss = self.loss_pointwise + self.loss_listwise

    def _build_syn_ext3_listwise(self):
        with tf.variable_scope('inputs') as inputs:
            self.r_ques = tf.placeholder(tf.int32,[None,None,self.ques_len],name='ques_point')
            self.r_ques_len = tf.placeholder(tf.float32,[None,None,self.ques_len],name='ques_len_point')
            self.r_ans = tf.placeholder(tf.int32,[None,None,self.ans_len],name='ans_point')
            self.r_ans_len = tf.placeholder(tf.float32,[None,None,self.ans_len],name='ans_len_point')

            self._ques_filter_len = tf.tile(tf.expand_dims(self.r_ans_len,2),[1,1,self.ques_len,1])
            self._ans_filter_len = tf.tile(tf.expand_dims(self.r_ques_len,2),[1,1,self.ans_len,1])

            self._ques_align_len = tf.expand_dims(self.r_ques_len,3)
            self._ans_align_len = tf.expand_dims(self.r_ans_len,3)

            #self.p_label = tf.placeholder(tf.float32,[None,None])
            self.l_label = tf.placeholder(tf.float32,[None,None])

            self.batch_size, self.list_size = tf.shape(self.r_ans)[0], tf.shape(self.r_ans)[1]

            self._ques = tf.reshape(self.r_ques,shape=(-1,self.ques_len))
            self._ans = tf.reshape(self.r_ans,shape=(-1,self.ans_len))
            self._ques_filter_len = tf.reshape(self._ques_filter_len,shape=(-1,self.ques_len,self.ans_len))
            self._ans_filter_len = tf.reshape(self._ans_filter_len,shape=(-1,self.ans_len,self.ques_len))
            self._ques_align_len = tf.reshape(self._ques_align_len,shape=(-1,self.ques_len,1))
            self._ans_align_len = tf.reshape(self._ans_align_len,shape=(-1,self.ans_len,1))

            self.is_train = tf.placeholder(tf.bool)
            self.dc = tf.placeholder(tf.bool)
        with tf.name_scope('list_wise'):
            with tf.variable_scope('embedding_layer') as embedding_l:
                weights = np.load(self.embedding_file)
                weights[0] = np.zeros((weights.shape[1]))
                embeddings = tf.Variable(weights,dtype=tf.float32,trainable=False)

                ques_emb = tf.nn.embedding_lookup(embeddings,self._ques)
                ans_emb = tf.nn.embedding_lookup(embeddings,self._ans)
                print('ques_emb:',ques_emb.shape)
                print('ans_emb',ans_emb.shape)
            with tf.variable_scope('preprocess_layer') as prep_l:
                sig_den = tf.layers.Dense(self.hidden_dim,name='sigmoid_dense')
                tan_den = tf.layers.Dense(self.hidden_dim,name='tanh_dense')

                ques_sig = sig_den(ques_emb)
                ques_sig = tf.sigmoid(ques_sig)
                ques_tan = tan_den(ques_emb)
                ques_tan = tf.tanh(ques_tan)
                ques_h = tf.multiply(ques_sig,ques_tan)

                ans_sig = sig_den(ans_emb)
                ans_sig = tf.sigmoid(ans_sig)
                ans_tan = tan_den(ans_emb)
                ans_tan = tf.tanh(ans_tan)
                ans_h = tf.multiply(ans_sig,ans_tan)
            with tf.variable_scope('synatic_extract') as syn_ext_l:
                q_kernel_size = [2]
                a_kernel_size = [2]
                #q_syn_extractor = [tf.layers.Conv2D(1,(i,1),strides=(i,1),padding='valid',kernel_initializer=tf.initializers.ones,trainable=False,name='q_extrator_'+str(i)) for i in q_kernel_size]
                #a_syn_extractor = [tf.layers.Conv2D(1,(i,1),strides=(i,1),padding='valid',kernel_initializer=tf.initializers.ones,trainable=False,name='a_extrator_'+str(i)) for i in a_kernel_size]
                q_syn_extractor = [tf.layers.Conv2D(1,(i,1),strides=(1,1),padding='same',kernel_initializer=tf.initializers.ones,trainable=False,name='q_extrator_'+str(i)) for i in q_kernel_size]
                a_syn_extractor = [tf.layers.Conv2D(1,(i,1),strides=(1,1),padding='same',kernel_initializer=tf.initializers.ones,trainable=False,name='a_extrator_'+str(i)) for i in a_kernel_size]
                #q_syn_extractor = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='q_extrator_'+str(i)) for i in q_kernel_size]
                #a_syn_extractor = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='q_extrator_'+str(i)) for i in a_kernel_size]

                self.ques_h = ques_h = tf.multiply(ques_h,self._ques_align_len)
                ans_h = tf.multiply(ans_h,self._ans_align_len)

                self.ques_syn = ques_syn = self.local_summarize(self,q_syn_extractor,tf.expand_dims(ques_h,3),self.ques_len,q_kernel_size,-1,self._ques_align_len,Serial=True)
                ans_syn = self.local_summarize(self,a_syn_extractor,tf.expand_dims(ans_h,3),self.ans_len,a_kernel_size,-1,self._ans_align_len,Serial=True)
                #self.ques_syn = ques_syn = self.convXd_listwise(q_syn_extractor,ques_h,self._ques_align_len)
                #ans_syn = self.convXd_listwise(a_syn_extractor,ans_h,self._ans_align_len)
                print('ans_syn: ',ans_syn.shape)

                #ques_syn = tf.layers.dense(ques_syn,self.hidden_dim,activation=tf.tanh,name='q_syn_dense')
                #ans_syn = tf.layers.dense(ans_syn,self.hidden_dim,activation=tf.tanh,name='a_syn_dense')

                #q_a_syn = tf.concat([ques_syn,ans_syn],axis=-1)

                ##q_a_syn = tf.layers.dropout(q_a_syn,rate=0.5,training=self.is_train)

                #q_a_syn = tf.layers.dense(q_a_syn,self.hidden_dim,name='q_a_syn')
                #q_a_syn = tf.nn.tanh(q_a_syn)

            #with tf.variable_scope('thresh_net') as thresh_l:
            #    Thresh = tf.layers.Dense(1,kernel_initializer=tf.initializers.zeros,name='threshold')

            #    ques_att_thresh = Thresh(ques_h)*self._ques_align_len
            #    ans_att_thresh = Thresh(ans_h)*self._ans_align_len

            with tf.variable_scope('attention_softalign') as att_align_l:

                #ques_h = tf.concat([ques_h,ques_syn],axis=1)
                #ans_h = tf.concat([ans_h,ans_syn],axis=1)

                ques_att_matrix = self.getAttMat(ques_h,ans_h)
                ans_att_matrix = self.getAttMat(ans_h,ques_h)
                ques_syn_matrix = self.getAttMat(ques_syn,ans_syn)
                ans_syn_matrix = self.getAttMat(ans_syn,ques_syn)
                #print('ques_att_matrix:',ques_att_matrix.shape)
                #print('ques_syn_matrix:',ques_syn_matrix.shape)
                #double_ques_filter_len = tf.tile(self._ques_filter_len,[1,2,2])
                #double_ans_filter_len = tf.tile(self._ans_filter_len,[1,2,2])
                #double_ques_align_len = tf.tile(self._ques_align_len,[1,2,1])
                #double_ans_align_len = tf.tile(self._ans_align_len,[1,2,1])
                ques_align, self.ques_att = self.getAlign(ans_h,ques_att_matrix,self._ques_filter_len)
                ans_align, _ = self.getAlign(ques_h,ans_att_matrix,self._ans_filter_len)
                #ques_align = self.getAlign(ans_h,ques_att_matrix,double_ques_filter_len)
                #ans_align = self.getAlign(ques_h,ans_att_matrix,double_ans_filter_len)
                ques_syn_align, self.ques_syn_att = self.getAlign(ans_syn,ques_syn_matrix,self._ques_filter_len)
                ans_syn_align, _ = self.getAlign(ques_syn,ans_syn_matrix,self._ans_filter_len)
                print('ques_align:',ques_align.shape)
                #print('ques_syn_align:',ques_syn_align.shape)

                ques_aligned = tf.multiply(tf.multiply(ques_align,ques_h),self._ques_align_len)
                ans_aligned = tf.multiply(tf.multiply(ans_align,ans_h),self._ans_align_len)
                #ques_aligned = tf.multiply(tf.multiply(ques_align,ques_h),double_ques_align_len)
                #ans_aligned = tf.multiply(tf.multiply(ans_align,ans_h),double_ans_align_len)
                ques_syn_aligned = tf.multiply(tf.multiply(ques_syn_align,ques_syn),self._ques_align_len)
                ans_syn_aligned = tf.multiply(tf.multiply(ans_syn_align,ans_syn),self._ans_align_len)
            with tf.variable_scope('cnn_feature') as cnn_l:
                self.cnn_ques = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='q_conv_'+str(i)) for i in range(1,6)]
                self.cnn_ans = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='a_conv_'+str(i)) for i in range(1,6)]
                #self.cnn_syn_ques = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='q_syn_conv_'+str(i)) for i in range(1,6)]
                #self.cnn_syn_ans = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='a_syn_conv_'+str(i)) for i in range(1,6)]

                ques_cnn = self.convXd_listwise(self.cnn_ques,ques_aligned,self._ques_align_len,1)
                ans_cnn = self.convXd_listwise(self.cnn_ans,ans_aligned,self._ans_align_len,1)
                #ques_cnn = self.convXd_listwise(self.cnn_ques,ques_aligned,double_ques_align_len,1)
                #ans_cnn = self.convXd_listwise(self.cnn_ans,ans_aligned,double_ans_align_len,1)
                ques_syn_cnn = self.convXd_listwise(self.cnn_ques,ques_syn_aligned,self._ques_align_len,1)
                ans_syn_cnn = self.convXd_listwise(self.cnn_ans,ans_syn_aligned,self._ans_align_len,1)

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
                ques_den = tf.layers.Dense(self.hidden_dim,activation=tf.tanh,name='q_dense')
                ans_den = tf.layers.Dense(self.hidden_dim,activation=tf.tanh,name='a_dense')

                #ques_syn_den = tf.layers.Dense(self.hidden_dim,activation=tf.tanh,name='q_syn_dense')
                #ans_syn_den = tf.layers.Dense(self.hidden_dim,activation=tf.tanh,name='a_syn_dense')

                ques_o1 = ques_den(ques_cnn)
                ans_o1 = ans_den(ans_cnn)
                #ques_syn_cnn = tf.layers.dropout(ques_syn_cnn,rate=0.2,training=self.is_train)
                #ans_syn_cnn = tf.layers.dropout(ans_syn_cnn,rate=0.2,training=self.is_train)
                #ques_syn_o1 = ques_syn_den(ques_syn_cnn)
                #ans_syn_o1 = ans_syn_den(ans_syn_cnn)
                q_a_syn = tf.concat([ques_syn_cnn,ans_syn_cnn],axis=-1)
                q_a_syn = tf.layers.dropout(q_a_syn,rate=0.5,training=self.is_train)
                q_a_syn = tf.layers.dense(q_a_syn,self.hidden_dim,activation=tf.nn.tanh,name='a_s_out1')

                #finalo1 = tf.concat([ques_o1,ans_o1],axis=-1)
                finalo1 = tf.concat([ques_o1,q_a_syn,ans_o1],axis=-1)
                #finalo1 = tf.concat([ques_syn_o1,ans_syn_o1],axis=-1)
                #finalo1 = tf.concat([ques_o1,ans_o1,ques_syn_o1,ans_syn_o1],axis=-1)

                finalo2 = tf.layers.dense(finalo1,self.hidden_dim,activation=tf.tanh,name='finalout')
                self.score = tf.layers.dense(finalo2,1,name='score')
                print('score:',self.score.shape)
                self.score = tf.reshape(self.score,shape=(self.batch_size,self.list_size))
                self.logit_score = tf.nn.log_softmax(self.score,dim=-1)
                print('logit_score:',self.logit_score.shape)
            with tf.variable_scope('loss') as loss_l:
                #self.loss_pointwise = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.p_label,logits=tf.squeeze(self.score,-1)))
                self.loss_listwise = tf.reduce_mean(self.l_label*(tf.log(tf.clip_by_value(self.l_label,1e-5,1.0))-self.logit_score))

                #self.total_loss = self.loss_pointwise + self.loss_listwise

    def _build_syn_ext4_listwise(self):
        with tf.variable_scope('inputs') as inputs:
            self.r_ques = tf.placeholder(tf.int32,[None,None,self.ques_len],name='ques_point')
            self.r_ques_len = tf.placeholder(tf.float32,[None,None,self.ques_len],name='ques_len_point')
            self.r_ans = tf.placeholder(tf.int32,[None,None,self.ans_len],name='ans_point')
            self.r_ans_len = tf.placeholder(tf.float32,[None,None,self.ans_len],name='ans_len_point')

            self._ques_filter_len = tf.tile(tf.expand_dims(self.r_ans_len,2),[1,1,self.ques_len,1])
            self._ans_filter_len = tf.tile(tf.expand_dims(self.r_ques_len,2),[1,1,self.ans_len,1])

            self._ques_align_len = tf.expand_dims(self.r_ques_len,3)
            self._ans_align_len = tf.expand_dims(self.r_ans_len,3)

            #self.p_label = tf.placeholder(tf.float32,[None,None])
            self.l_label = tf.placeholder(tf.float32,[None,None])

            self.batch_size, self.list_size = tf.shape(self.r_ans)[0], tf.shape(self.r_ans)[1]

            self._ques = tf.reshape(self.r_ques,shape=(-1,self.ques_len))
            self._ans = tf.reshape(self.r_ans,shape=(-1,self.ans_len))
            self._ques_filter_len = tf.reshape(self._ques_filter_len,shape=(-1,self.ques_len,self.ans_len))
            self._ans_filter_len = tf.reshape(self._ans_filter_len,shape=(-1,self.ans_len,self.ques_len))
            self._ques_align_len = tf.reshape(self._ques_align_len,shape=(-1,self.ques_len,1))
            self._ans_align_len = tf.reshape(self._ans_align_len,shape=(-1,self.ans_len,1))

            self.is_train = tf.placeholder(tf.bool)
            self.dc = tf.placeholder(tf.bool)
        with tf.name_scope('list_wise'):
            with tf.variable_scope('embedding_layer') as embedding_l:
                weights = np.load(self.embedding_file)
                weights[0] = np.zeros((weights.shape[1]))
                embeddings = tf.Variable(weights,dtype=tf.float32,trainable=False)

                ques_emb = tf.nn.embedding_lookup(embeddings,self._ques)
                ans_emb = tf.nn.embedding_lookup(embeddings,self._ans)
                print('ques_emb:',ques_emb.shape)
                print('ans_emb',ans_emb.shape)
            with tf.variable_scope('preprocess_layer') as prep_l:
                sig_den = tf.layers.Dense(self.hidden_dim,name='sigmoid_dense')
                tan_den = tf.layers.Dense(self.hidden_dim,name='tanh_dense')

                ques_sig = sig_den(ques_emb)
                ques_sig = tf.sigmoid(ques_sig)
                ques_tan = tan_den(ques_emb)
                ques_tan = tf.tanh(ques_tan)
                ques_h = tf.multiply(ques_sig,ques_tan)

                ans_sig = sig_den(ans_emb)
                ans_sig = tf.sigmoid(ans_sig)
                ans_tan = tan_den(ans_emb)
                ans_tan = tf.tanh(ans_tan)
                ans_h = tf.multiply(ans_sig,ans_tan)
            with tf.variable_scope('synatic_extract') as syn_ext_l:
                q_kernel_size = [2]
                a_kernel_size = [2]
                q_syn_extractor = [tf.layers.Conv2D(1,(i,1),strides=(i,1),padding='valid',kernel_initializer=tf.initializers.ones,trainable=False,name='q_extrator_'+str(i)) for i in q_kernel_size]
                a_syn_extractor = [tf.layers.Conv2D(1,(i,1),strides=(i,1),padding='valid',kernel_initializer=tf.initializers.ones,trainable=False,name='a_extrator_'+str(i)) for i in a_kernel_size]
                #q_syn_extractor = [tf.layers.Conv2D(1,(i,1),strides=(1,1),padding='same',kernel_initializer=tf.initializers.ones,trainable=False,name='q_extrator_'+str(i)) for i in q_kernel_size]
                #a_syn_extractor = [tf.layers.Conv2D(1,(i,1),strides=(1,1),padding='same',kernel_initializer=tf.initializers.ones,trainable=False,name='a_extrator_'+str(i)) for i in a_kernel_size]
                #q_syn_extractor = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='q_extrator_'+str(i)) for i in q_kernel_size]
                #a_syn_extractor = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='q_extrator_'+str(i)) for i in a_kernel_size]

                self.ques_h = ques_h = tf.multiply(ques_h,self._ques_align_len)
                ans_h = tf.multiply(ans_h,self._ans_align_len)

                self.ques_syn = ques_syn = self.local_summarize(self,q_syn_extractor,tf.expand_dims(ques_h,3),self.ques_len,q_kernel_size,-1,self._ques_align_len,Serial=False)
                ans_syn = self.local_summarize(self,a_syn_extractor,tf.expand_dims(ans_h,3),self.ans_len,a_kernel_size,-1,self._ans_align_len,Serial=False)
                #self.ques_syn = ques_syn = self.convXd_listwise(q_syn_extractor,ques_h,self._ques_align_len)
                #ans_syn = self.convXd_listwise(a_syn_extractor,ans_h,self._ans_align_len)
                print('ans_syn: ',ans_syn.shape)

                #ques_syn = tf.layers.dense(ques_syn,self.hidden_dim,activation=tf.tanh,name='q_syn_dense')
                #ans_syn = tf.layers.dense(ans_syn,self.hidden_dim,activation=tf.tanh,name='a_syn_dense')

                #q_a_syn = tf.concat([ques_syn,ans_syn],axis=-1)

                ##q_a_syn = tf.layers.dropout(q_a_syn,rate=0.5,training=self.is_train)

                #q_a_syn = tf.layers.dense(q_a_syn,self.hidden_dim,name='q_a_syn')
                #q_a_syn = tf.nn.tanh(q_a_syn)

            #with tf.variable_scope('thresh_net') as thresh_l:
            #    Thresh = tf.layers.Dense(1,kernel_initializer=tf.initializers.zeros,name='threshold')

            #    ques_att_thresh = Thresh(ques_h)*self._ques_align_len
            #    ans_att_thresh = Thresh(ans_h)*self._ans_align_len

            with tf.variable_scope('attention_softalign') as att_align_l:

                #ques_h = tf.concat([ques_h,ques_syn],axis=1)
                #ans_h = tf.concat([ans_h,ans_syn],axis=1)

                ques_att_matrix = self.getAttMat(ques_h,ans_h)
                ans_att_matrix = self.getAttMat(ans_h,ques_h)
                ques_syn_matrix = self.getAttMat(ques_syn,ans_syn)
                ans_syn_matrix = self.getAttMat(ans_syn,ques_syn)
                print('ques_att_matrix:',ques_att_matrix.shape)
                print('ques_syn_matrix:',ques_syn_matrix.shape)
                #double_ques_filter_len = tf.tile(self._ques_filter_len,[1,2,2])
                #double_ans_filter_len = tf.tile(self._ans_filter_len,[1,2,2])
                #double_ques_align_len = tf.tile(self._ques_align_len,[1,2,1])
                #double_ans_align_len = tf.tile(self._ans_align_len,[1,2,1])
                ques_align = self.getAlign(ans_h,ques_att_matrix,self._ques_filter_len)
                ans_align = self.getAlign(ques_h,ans_att_matrix,self._ans_filter_len)
                ques_syn_align = self.getAlign(ans_syn,ques_syn_matrix,self._ques_filter_len)
                ans_syn_align = self.getAlign(ques_syn,ans_syn_matrix,self._ans_filter_len)
                print('ques_align:',ques_align.shape)
                #print('ques_syn_align:',ques_syn_align.shape)

                ques_aligned = tf.multiply(tf.multiply(ques_align,ques_h),self._ques_align_len)
                ans_aligned = tf.multiply(tf.multiply(ans_align,ans_h),self._ans_align_len)
                ques_syn_aligned = tf.multiply(tf.multiply(ques_syn_align,ques_syn),self._ques_align_len)
                ans_syn_aligned = tf.multiply(tf.multiply(ans_syn_align,ans_syn),self._ans_align_len)
            with tf.variable_scope('cnn_feature') as cnn_l:
                self.cnn_ques = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='q_conv_'+str(i)) for i in range(1,6)]
                self.cnn_ans = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='a_conv_'+str(i)) for i in range(1,6)]
                #self.cnn_syn_ques = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='q_syn_conv_'+str(i)) for i in range(1,6)]
                #self.cnn_syn_ans = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='a_syn_conv_'+str(i)) for i in range(1,6)]

                ques_cnn = self.convXd_listwise(self.cnn_ques,ques_aligned,self._ques_align_len,1)
                ans_cnn = self.convXd_listwise(self.cnn_ans,ans_aligned,self._ans_align_len,1)
                ques_syn_cnn = self.convXd_listwise(self.cnn_ques,ques_syn_aligned,self._ques_align_len,1)
                ans_syn_cnn = self.convXd_listwise(self.cnn_ans,ans_syn_aligned,self._ans_align_len,1)

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
                ques_den = tf.layers.Dense(self.hidden_dim,activation=tf.tanh,name='q_dense')
                ans_den = tf.layers.Dense(self.hidden_dim,activation=tf.tanh,name='a_dense')

                #ques_syn_den = tf.layers.Dense(self.hidden_dim,activation=tf.tanh,name='q_syn_dense')
                #ans_syn_den = tf.layers.Dense(self.hidden_dim,activation=tf.tanh,name='a_syn_dense')

                ques_o1 = ques_den(ques_cnn)
                ans_o1 = ans_den(ans_cnn)
                #ques_syn_cnn = tf.layers.dropout(ques_syn_cnn,rate=0.2,training=self.is_train)
                #ans_syn_cnn = tf.layers.dropout(ans_syn_cnn,rate=0.2,training=self.is_train)
                #ques_syn_o1 = ques_syn_den(ques_syn_cnn)
                #ans_syn_o1 = ans_syn_den(ans_syn_cnn)
                q_a_syn = tf.concat([ques_syn_cnn,ans_syn_cnn],axis=-1)
                q_a_syn = tf.layers.dropout(q_a_syn,rate=0.5,training=self.is_train)
                q_a_syn = tf.layers.dense(q_a_syn,self.hidden_dim,activation=tf.nn.tanh,name='a_s_out1')

                #finalo1 = tf.concat([ques_o1,ans_o1],axis=-1)
                finalo1 = tf.concat([ques_o1,q_a_syn,ans_o1],axis=-1)
                #finalo1 = tf.concat([ques_syn_o1,ans_syn_o1],axis=-1)
                #finalo1 = tf.concat([ques_o1,ans_o1,ques_syn_o1,ans_syn_o1],axis=-1)

                finalo2 = tf.layers.dense(finalo1,self.hidden_dim,activation=tf.tanh,name='finalout')
                self.score = tf.layers.dense(finalo2,1,name='score')
                print('score:',self.score.shape)
                self.score = tf.reshape(self.score,shape=(self.batch_size,self.list_size))
                self.logit_score = tf.nn.log_softmax(self.score,dim=-1)
                print('logit_score:',self.logit_score.shape)
            with tf.variable_scope('loss') as loss_l:
                #self.loss_pointwise = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.p_label,logits=tf.squeeze(self.score,-1)))
                self.loss_listwise = tf.reduce_mean(self.l_label*(tf.log(tf.clip_by_value(self.l_label,1e-5,1.0))-self.logit_score))

                #self.total_loss = self.loss_pointwise + self.loss_listwise

    def _build_syn_ext5_listwise(self):
        with tf.variable_scope('inputs') as inputs:
            self.r_ques = tf.placeholder(tf.int32,[None,None,self.ques_len],name='ques_point')
            self.r_ques_len = tf.placeholder(tf.float32,[None,None,self.ques_len],name='ques_len_point')
            self.r_ans = tf.placeholder(tf.int32,[None,None,self.ans_len],name='ans_point')
            self.r_ans_len = tf.placeholder(tf.float32,[None,None,self.ans_len],name='ans_len_point')

            self._ques_filter_len = tf.tile(tf.expand_dims(self.r_ans_len,2),[1,1,self.ques_len,1])
            self._ans_filter_len = tf.tile(tf.expand_dims(self.r_ques_len,2),[1,1,self.ans_len,1])

            self._ques_align_len = tf.expand_dims(self.r_ques_len,3)
            self._ans_align_len = tf.expand_dims(self.r_ans_len,3)

            #self.p_label = tf.placeholder(tf.float32,[None,None])
            self.l_label = tf.placeholder(tf.float32,[None,None])

            self.batch_size, self.list_size = tf.shape(self.r_ans)[0], tf.shape(self.r_ans)[1]

            self._ques = tf.reshape(self.r_ques,shape=(-1,self.ques_len))
            self._ans = tf.reshape(self.r_ans,shape=(-1,self.ans_len))
            self._ques_filter_len = tf.reshape(self._ques_filter_len,shape=(-1,self.ques_len,self.ans_len))
            self._ans_filter_len = tf.reshape(self._ans_filter_len,shape=(-1,self.ans_len,self.ques_len))
            self._ques_align_len = tf.reshape(self._ques_align_len,shape=(-1,self.ques_len,1))
            self._ans_align_len = tf.reshape(self._ans_align_len,shape=(-1,self.ans_len,1))

            self.is_train = tf.placeholder(tf.bool)
            self.dc = tf.placeholder(tf.bool)
        with tf.name_scope('list_wise'):
            with tf.variable_scope('embedding_layer') as embedding_l:
                weights = np.load(self.embedding_file)
                weights[0] = np.zeros((weights.shape[1]))
                embeddings = tf.Variable(weights,dtype=tf.float32,trainable=False)

                ques_emb = tf.nn.embedding_lookup(embeddings,self._ques)
                ans_emb = tf.nn.embedding_lookup(embeddings,self._ans)
                print('ques_emb:',ques_emb.shape)
                print('ans_emb',ans_emb.shape)
            with tf.variable_scope('preprocess_layer') as prep_l:
                sig_den = tf.layers.Dense(self.hidden_dim,name='sigmoid_dense')
                tan_den = tf.layers.Dense(self.hidden_dim,name='tanh_dense')

                ques_sig = sig_den(ques_emb)
                ques_sig = tf.sigmoid(ques_sig)
                ques_tan = tan_den(ques_emb)
                ques_tan = tf.tanh(ques_tan)
                ques_h = tf.multiply(ques_sig,ques_tan)

                ans_sig = sig_den(ans_emb)
                ans_sig = tf.sigmoid(ans_sig)
                ans_tan = tan_den(ans_emb)
                ans_tan = tf.tanh(ans_tan)
                ans_h = tf.multiply(ans_sig,ans_tan)
            with tf.variable_scope('synatic_extract') as syn_ext_l:
                q_kernel_size = [2]
                a_kernel_size = [2]
                #q_syn_extractor = [tf.layers.Conv2D(1,(i,1),strides=(i,1),padding='valid',kernel_initializer=tf.initializers.ones,trainable=False,name='q_extrator_'+str(i)) for i in q_kernel_size]
                #a_syn_extractor = [tf.layers.Conv2D(1,(i,1),strides=(i,1),padding='valid',kernel_initializer=tf.initializers.ones,trainable=False,name='a_extrator_'+str(i)) for i in a_kernel_size]
                #q_syn_extractor = [tf.layers.Conv2D(1,(i,1),strides=(1,1),padding='same',kernel_initializer=tf.initializers.ones,trainable=False,name='q_extrator_'+str(i)) for i in q_kernel_size]
                #a_syn_extractor = [tf.layers.Conv2D(1,(i,1),strides=(1,1),padding='same',kernel_initializer=tf.initializers.ones,trainable=False,name='a_extrator_'+str(i)) for i in a_kernel_size]
                q_syn_extractor = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='q_extrator_'+str(i)) for i in q_kernel_size]
                a_syn_extractor = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='q_extrator_'+str(i)) for i in a_kernel_size]

                self.ques_h = ques_h = tf.multiply(ques_h,self._ques_align_len)
                ans_h = tf.multiply(ans_h,self._ans_align_len)

                #self.ques_syn = ques_syn = self.local_summarize(self,q_syn_extractor,tf.expand_dims(ques_h,3),self.ques_len,q_kernel_size,-1,self._ques_align_len)
                #ans_syn = self.local_summarize(self,a_syn_extractor,tf.expand_dims(ans_h,3),self.ans_len,a_kernel_size,-1,self._ans_align_len)
                self.ques_syn = ques_syn = self.convXd_listwise(q_syn_extractor,ques_h,self._ques_align_len)
                ans_syn = self.convXd_listwise(a_syn_extractor,ans_h,self._ans_align_len)
                print('ans_syn: ',ans_syn.shape)

                #ques_syn = tf.layers.dense(ques_syn,self.hidden_dim,activation=tf.tanh,name='q_syn_dense')
                #ans_syn = tf.layers.dense(ans_syn,self.hidden_dim,activation=tf.tanh,name='a_syn_dense')

                #q_a_syn = tf.concat([ques_syn,ans_syn],axis=-1)

                ##q_a_syn = tf.layers.dropout(q_a_syn,rate=0.5,training=self.is_train)

                #q_a_syn = tf.layers.dense(q_a_syn,self.hidden_dim,name='q_a_syn')
                #q_a_syn = tf.nn.tanh(q_a_syn)

            #with tf.variable_scope('thresh_net') as thresh_l:
            #    Thresh = tf.layers.Dense(1,kernel_initializer=tf.initializers.zeros,name='threshold')

            #    ques_att_thresh = Thresh(ques_h)*self._ques_align_len
            #    ans_att_thresh = Thresh(ans_h)*self._ans_align_len

            with tf.variable_scope('attention_softalign') as att_align_l:

                #ques_h = tf.concat([ques_h,ques_syn],axis=1)
                #ans_h = tf.concat([ans_h,ans_syn],axis=1)

                ques_att_matrix = self.getAttMat(ques_h,ans_h)
                ans_att_matrix = self.getAttMat(ans_h,ques_h)
                ques_syn_matrix = self.getAttMat(ques_syn,ans_syn)
                ans_syn_matrix = self.getAttMat(ans_syn,ques_syn)
                print('ques_att_matrix:',ques_att_matrix.shape)
                print('ques_syn_matrix:',ques_syn_matrix.shape)
                #double_ques_filter_len = tf.tile(self._ques_filter_len,[1,2,2])
                #double_ans_filter_len = tf.tile(self._ans_filter_len,[1,2,2])
                #double_ques_align_len = tf.tile(self._ques_align_len,[1,2,1])
                #double_ans_align_len = tf.tile(self._ans_align_len,[1,2,1])
                ques_align = self.getAlign(ans_h,ques_att_matrix,self._ques_filter_len)
                ans_align = self.getAlign(ques_h,ans_att_matrix,self._ans_filter_len)
                ques_syn_align = self.getAlign(ans_syn,ques_syn_matrix,self._ques_filter_len)
                ans_syn_align = self.getAlign(ques_syn,ans_syn_matrix,self._ans_filter_len)
                print('ques_align:',ques_align.shape)
                #print('ques_syn_align:',ques_syn_align.shape)

                ques_aligned = tf.multiply(tf.multiply(ques_align,ques_h),self._ques_align_len)
                ans_aligned = tf.multiply(tf.multiply(ans_align,ans_h),self._ans_align_len)
                ques_syn_aligned = tf.multiply(tf.multiply(ques_syn_align,ques_syn),self._ques_align_len)
                ans_syn_aligned = tf.multiply(tf.multiply(ans_syn_align,ans_syn),self._ans_align_len)
            with tf.variable_scope('cnn_feature') as cnn_l:
                self.cnn_ques = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='q_conv_'+str(i)) for i in range(1,6)]
                self.cnn_ans = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='a_conv_'+str(i)) for i in range(1,6)]
                #self.cnn_syn_ques = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='q_syn_conv_'+str(i)) for i in range(1,6)]
                #self.cnn_syn_ans = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='a_syn_conv_'+str(i)) for i in range(1,6)]

                ques_cnn = self.convXd_listwise(self.cnn_ques,ques_aligned,self._ques_align_len,1)
                ans_cnn = self.convXd_listwise(self.cnn_ans,ans_aligned,self._ans_align_len,1)
                ques_syn_cnn = self.convXd_listwise(self.cnn_ques,ques_syn_aligned,self._ques_align_len,1)
                ans_syn_cnn = self.convXd_listwise(self.cnn_ans,ans_syn_aligned,self._ans_align_len,1)

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
                ques_den = tf.layers.Dense(self.hidden_dim,activation=tf.tanh,name='q_dense')
                ans_den = tf.layers.Dense(self.hidden_dim,activation=tf.tanh,name='a_dense')

                #ques_syn_den = tf.layers.Dense(self.hidden_dim,activation=tf.tanh,name='q_syn_dense')
                #ans_syn_den = tf.layers.Dense(self.hidden_dim,activation=tf.tanh,name='a_syn_dense')

                ques_o1 = ques_den(ques_cnn)
                ans_o1 = ans_den(ans_cnn)
                #ques_syn_cnn = tf.layers.dropout(ques_syn_cnn,rate=0.2,training=self.is_train)
                #ans_syn_cnn = tf.layers.dropout(ans_syn_cnn,rate=0.2,training=self.is_train)
                #ques_syn_o1 = ques_syn_den(ques_syn_cnn)
                #ans_syn_o1 = ans_syn_den(ans_syn_cnn)
                q_a_syn = tf.concat([ques_syn_cnn,ans_syn_cnn],axis=-1)
                q_a_syn = tf.layers.dropout(q_a_syn,rate=0.5,training=self.is_train)
                q_a_syn = tf.layers.dense(q_a_syn,self.hidden_dim,activation=tf.nn.tanh,name='a_s_out1')

                #finalo1 = tf.concat([ques_o1,ans_o1],axis=-1)
                finalo1 = tf.concat([ques_o1,q_a_syn,ans_o1],axis=-1)
                #finalo1 = tf.concat([ques_syn_o1,ans_syn_o1],axis=-1)
                #finalo1 = tf.concat([ques_o1,ans_o1,ques_syn_o1,ans_syn_o1],axis=-1)

                finalo2 = tf.layers.dense(finalo1,self.hidden_dim,activation=tf.tanh,name='finalout')
                self.score = tf.layers.dense(finalo2,1,name='score')
                print('score:',self.score.shape)
                self.score = tf.reshape(self.score,shape=(self.batch_size,self.list_size))
                self.logit_score = tf.nn.log_softmax(self.score,dim=-1)
                print('logit_score:',self.logit_score.shape)
            with tf.variable_scope('loss') as loss_l:
                #self.loss_pointwise = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.p_label,logits=tf.squeeze(self.score,-1)))
                self.loss_listwise = tf.reduce_mean(self.l_label*(tf.log(tf.clip_by_value(self.l_label,1e-5,1.0))-self.logit_score))

                #self.total_loss = self.loss_pointwise + self.loss_listwise

    def _build_syn_ext6_listwise(self):
        with tf.variable_scope('inputs') as inputs:
            self.r_ques = tf.placeholder(tf.int32,[None,None,self.ques_len],name='ques_point')
            self.r_ques_len = tf.placeholder(tf.float32,[None,None,self.ques_len],name='ques_len_point')
            self.r_ans = tf.placeholder(tf.int32,[None,None,self.ans_len],name='ans_point')
            self.r_ans_len = tf.placeholder(tf.float32,[None,None,self.ans_len],name='ans_len_point')

            self._ques_filter_len = tf.tile(tf.expand_dims(self.r_ans_len,2),[1,1,self.ques_len,1])
            self._ans_filter_len = tf.tile(tf.expand_dims(self.r_ques_len,2),[1,1,self.ans_len,1])

            self._ques_align_len = tf.expand_dims(self.r_ques_len,3)
            self._ans_align_len = tf.expand_dims(self.r_ans_len,3)

            #self.p_label = tf.placeholder(tf.float32,[None,None])
            self.l_label = tf.placeholder(tf.float32,[None,None])

            self.batch_size, self.list_size = tf.shape(self.r_ans)[0], tf.shape(self.r_ans)[1]

            self._ques = tf.reshape(self.r_ques,shape=(-1,self.ques_len))
            self._ans = tf.reshape(self.r_ans,shape=(-1,self.ans_len))
            self._ques_filter_len = tf.reshape(self._ques_filter_len,shape=(-1,self.ques_len,self.ans_len))
            self._ans_filter_len = tf.reshape(self._ans_filter_len,shape=(-1,self.ans_len,self.ques_len))
            self._ques_align_len = tf.reshape(self._ques_align_len,shape=(-1,self.ques_len,1))
            self._ans_align_len = tf.reshape(self._ans_align_len,shape=(-1,self.ans_len,1))

            self.is_train = tf.placeholder(tf.bool)
            self.dc = tf.placeholder(tf.bool)
        with tf.name_scope('list_wise'):
            with tf.variable_scope('embedding_layer') as embedding_l:
                weights = np.load(self.embedding_file)
                weights[0] = np.zeros((weights.shape[1]))
                embeddings = tf.Variable(weights,dtype=tf.float32,trainable=False)

                ques_emb = tf.nn.embedding_lookup(embeddings,self._ques)
                ans_emb = tf.nn.embedding_lookup(embeddings,self._ans)
                print('ques_emb:',ques_emb.shape)
                print('ans_emb',ans_emb.shape)
            with tf.variable_scope('preprocess_layer') as prep_l:
                sig_den = tf.layers.Dense(self.hidden_dim,name='sigmoid_dense')
                tan_den = tf.layers.Dense(self.hidden_dim,name='tanh_dense')

                ques_sig = sig_den(ques_emb)
                ques_sig = tf.sigmoid(ques_sig)
                ques_tan = tan_den(ques_emb)
                ques_tan = tf.tanh(ques_tan)
                ques_h = tf.multiply(ques_sig,ques_tan)

                ans_sig = sig_den(ans_emb)
                ans_sig = tf.sigmoid(ans_sig)
                ans_tan = tan_den(ans_emb)
                ans_tan = tf.tanh(ans_tan)
                ans_h = tf.multiply(ans_sig,ans_tan)
            with tf.variable_scope('synatic_extract') as syn_ext_l:
                q_kernel_size = [2]
                a_kernel_size = [2]
                #q_syn_extractor = [tf.layers.Conv2D(1,(i,1),strides=(i,1),padding='valid',kernel_initializer=tf.initializers.ones,trainable=False,name='q_extrator_'+str(i)) for i in q_kernel_size]
                #a_syn_extractor = [tf.layers.Conv2D(1,(i,1),strides=(i,1),padding='valid',kernel_initializer=tf.initializers.ones,trainable=False,name='a_extrator_'+str(i)) for i in a_kernel_size]
                q_syn_extractor = [tf.layers.Conv2D(1,(i,1),strides=(1,1),padding='same',kernel_initializer=tf.initializers.ones,trainable=False,name='q_extrator_'+str(i)) for i in q_kernel_size]
                a_syn_extractor = [tf.layers.Conv2D(1,(i,1),strides=(1,1),padding='same',kernel_initializer=tf.initializers.ones,trainable=False,name='a_extrator_'+str(i)) for i in a_kernel_size]
                #q_syn_extractor = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='q_extrator_'+str(i)) for i in q_kernel_size]
                #a_syn_extractor = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='q_extrator_'+str(i)) for i in a_kernel_size]

                self.ques_h = ques_h = tf.multiply(ques_h,self._ques_align_len)
                ans_h = tf.multiply(ans_h,self._ans_align_len)

                self.ques_syn = ques_syn = self.local_summarize(self,q_syn_extractor,tf.expand_dims(ques_h,3),self.ques_len,q_kernel_size,-1,self._ques_align_len,Serial=True)
                ans_syn = self.local_summarize(self,a_syn_extractor,tf.expand_dims(ans_h,3),self.ans_len,a_kernel_size,-1,self._ans_align_len,Serial=True)
                #self.ques_syn = ques_syn = self.convXd_listwise(q_syn_extractor,ques_h,self._ques_align_len)
                #ans_syn = self.convXd_listwise(a_syn_extractor,ans_h,self._ans_align_len)
                print('ans_syn: ',ans_syn.shape)

                #ques_syn = tf.layers.dense(ques_syn,self.hidden_dim,activation=tf.tanh,name='q_syn_dense')
                #ans_syn = tf.layers.dense(ans_syn,self.hidden_dim,activation=tf.tanh,name='a_syn_dense')

                #q_a_syn = tf.concat([ques_syn,ans_syn],axis=-1)

                ##q_a_syn = tf.layers.dropout(q_a_syn,rate=0.5,training=self.is_train)

                #q_a_syn = tf.layers.dense(q_a_syn,self.hidden_dim,name='q_a_syn')
                #q_a_syn = tf.nn.tanh(q_a_syn)

            #with tf.variable_scope('thresh_net') as thresh_l:
            #    Thresh = tf.layers.Dense(1,kernel_initializer=tf.initializers.zeros,name='threshold')

            #    ques_att_thresh = Thresh(ques_h)*self._ques_align_len
            #    ans_att_thresh = Thresh(ans_h)*self._ans_align_len

            with tf.variable_scope('attention_softalign') as att_align_l:

                ques_h = tf.concat([ques_h,ques_syn],axis=1)
                ans_h = tf.concat([ans_h,ans_syn],axis=1)

                ques_att_matrix = self.getAttMat(ques_h,ans_h)
                ans_att_matrix = self.getAttMat(ans_h,ques_h)
                #ques_syn_matrix = self.getAttMat(ques_syn,ans_syn)
                #ans_syn_matrix = self.getAttMat(ans_syn,ques_syn)
                #print('ques_att_matrix:',ques_att_matrix.shape)
                #print('ques_syn_matrix:',ques_syn_matrix.shape)
                double_ques_filter_len = tf.tile(self._ques_filter_len,[1,2,2])
                double_ans_filter_len = tf.tile(self._ans_filter_len,[1,2,2])
                double_ques_align_len = tf.tile(self._ques_align_len,[1,2,1])
                double_ans_align_len = tf.tile(self._ans_align_len,[1,2,1])
                #ques_align = self.getAlign(ans_h,ques_att_matrix,self._ques_filter_len)
                #ans_align = self.getAlign(ques_h,ans_att_matrix,self._ans_filter_len)
                ans_align = self.getAlign(self,ques_h,ans_att_matrix,double_ans_filter_len)
                ques_align = self.getAlign(self,ans_h,ques_att_matrix,double_ques_filter_len)
                #ques_syn_align = self.getAlign(ans_syn,ques_syn_matrix,self._ques_filter_len)
                #ans_syn_align = self.getAlign(ques_syn,ans_syn_matrix,self._ans_filter_len)
                print('ques_align:',ques_align.shape)
                #print('ques_syn_align:',ques_syn_align.shape)

                #ques_aligned = tf.multiply(tf.multiply(ques_align,ques_h),self._ques_align_len)
                #ans_aligned = tf.multiply(tf.multiply(ans_align,ans_h),self._ans_align_len)
                ques_aligned = tf.multiply(tf.multiply(ques_align,ques_h),double_ques_align_len)
                ans_aligned = tf.multiply(tf.multiply(ans_align,ans_h),double_ans_align_len)
                #ques_syn_aligned = tf.multiply(tf.multiply(ques_syn_align,ques_syn),self._ques_align_len)
                #ans_syn_aligned = tf.multiply(tf.multiply(ans_syn_align,ans_syn),self._ans_align_len)
            with tf.variable_scope('cnn_feature') as cnn_l:
                self.cnn_ques = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='q_conv_'+str(i)) for i in range(1,6)]
                self.cnn_ans = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='a_conv_'+str(i)) for i in range(1,6)]
                #self.cnn_syn_ques = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='q_syn_conv_'+str(i)) for i in range(1,6)]
                #self.cnn_syn_ans = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='a_syn_conv_'+str(i)) for i in range(1,6)]

                #ques_cnn = self.convXd_listwise(self.cnn_ques,ques_aligned,self._ques_align_len,1)
                #ans_cnn = self.convXd_listwise(self.cnn_ans,ans_aligned,self._ans_align_len,1)
                ques_cnn = self.convXd_listwise(self.cnn_ques,ques_aligned,double_ques_align_len,1)
                ans_cnn = self.convXd_listwise(self.cnn_ans,ans_aligned,double_ans_align_len,1)
                #ques_syn_cnn = self.convXd_listwise(self.cnn_ques,ques_syn_aligned,self._ques_align_len,1)
                #ans_syn_cnn = self.convXd_listwise(self.cnn_ans,ans_syn_aligned,self._ans_align_len,1)

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
                ques_den = tf.layers.Dense(self.hidden_dim,activation=tf.tanh,name='q_dense')
                ans_den = tf.layers.Dense(self.hidden_dim,activation=tf.tanh,name='a_dense')

                #ques_syn_den = tf.layers.Dense(self.hidden_dim,activation=tf.tanh,name='q_syn_dense')
                #ans_syn_den = tf.layers.Dense(self.hidden_dim,activation=tf.tanh,name='a_syn_dense')

                ques_o1 = ques_den(ques_cnn)
                ans_o1 = ans_den(ans_cnn)
                #ques_syn_cnn = tf.layers.dropout(ques_syn_cnn,rate=0.2,training=self.is_train)
                #ans_syn_cnn = tf.layers.dropout(ans_syn_cnn,rate=0.2,training=self.is_train)
                #ques_syn_o1 = ques_syn_den(ques_syn_cnn)
                #ans_syn_o1 = ans_syn_den(ans_syn_cnn)
                #q_a_syn = tf.concat([ques_syn_cnn,ans_syn_cnn],axis=-1)
                #q_a_syn = tf.layers.dropout(q_a_syn,rate=0.5,training=self.is_train)
                #q_a_syn = tf.layers.dense(q_a_syn,self.hidden_dim,activation=tf.nn.tanh,name='a_s_out1')

                finalo1 = tf.concat([ques_o1,ans_o1],axis=-1)
                #finalo1 = tf.concat([ques_o1,q_a_syn,ans_o1],axis=-1)
                #finalo1 = tf.concat([ques_syn_o1,ans_syn_o1],axis=-1)
                #finalo1 = tf.concat([ques_o1,ans_o1,ques_syn_o1,ans_syn_o1],axis=-1)

                finalo2 = tf.layers.dense(finalo1,self.hidden_dim,activation=tf.tanh,name='finalout')
                self.score = tf.layers.dense(finalo2,1,name='score')
                print('score:',self.score.shape)
                self.score = tf.reshape(self.score,shape=(self.batch_size,self.list_size))
                self.logit_score = tf.nn.log_softmax(self.score,dim=-1)
                print('logit_score:',self.logit_score.shape)
            with tf.variable_scope('loss') as loss_l:
                #self.loss_pointwise = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.p_label,logits=tf.squeeze(self.score,-1)))
                self.loss_listwise = tf.reduce_mean(self.l_label*(tf.log(tf.clip_by_value(self.l_label,1e-5,1.0))-self.logit_score))

                #self.total_loss = self.loss_pointwise + self.loss_listwise

    def _build_att_ext_listwise(self):
        with tf.variable_scope('inputs') as inputs:
            self.r_ques = tf.placeholder(tf.int32,[None,None,self.ques_len],name='ques_point')
            self.r_ques_len = tf.placeholder(tf.float32,[None,None,self.ques_len],name='ques_len_point')
            self.r_ans = tf.placeholder(tf.int32,[None,None,self.ans_len],name='ans_point')
            self.r_ans_len = tf.placeholder(tf.float32,[None,None,self.ans_len],name='ans_len_point')

            self._ques_filter_len = tf.tile(tf.expand_dims(self.r_ans_len,2),[1,1,self.ques_len,1])
            self._ans_filter_len = tf.tile(tf.expand_dims(self.r_ques_len,2),[1,1,self.ans_len,1])

            self._ques_align_len = tf.tile(tf.expand_dims(self.r_ques_len,3),[1,1,1,self.hidden_dim])
            self._ans_align_len = tf.tile(tf.expand_dims(self.r_ans_len,3),[1,1,1,self.hidden_dim])

            #self.p_label = tf.placeholder(tf.float32,[None,None])
            self.l_label = tf.placeholder(tf.float32,[None,None])

            self.batch_size, self.list_size = tf.shape(self.r_ans)[0], tf.shape(self.r_ans)[1]

            self._ques = tf.reshape(self.r_ques,shape=(-1,self.ques_len))
            self._ans = tf.reshape(self.r_ans,shape=(-1,self.ans_len))
            self._ques_filter_len = tf.reshape(self._ques_filter_len,shape=(-1,self.ques_len,self.ans_len))
            self._ans_filter_len = tf.reshape(self._ans_filter_len,shape=(-1,self.ans_len,self.ques_len))
            self._ques_align_len = tf.reshape(self._ques_align_len,shape=(-1,self.ques_len,self.hidden_dim))
            self._ans_align_len = tf.reshape(self._ans_align_len,shape=(-1,self.ans_len,self.hidden_dim))

            self.is_train = tf.placeholder(tf.bool)
            self.dc = tf.placeholder(tf.bool)
        with tf.name_scope('list_wise'):
            with tf.variable_scope('embedding_layer') as embedding_l:
                weights = np.load(self.embedding_file)
                weights[0] = np.zeros((weights.shape[1]))
                embeddings = tf.Variable(weights,dtype=tf.float32,trainable=False)

                ques_emb = tf.nn.embedding_lookup(embeddings,self._ques)
                ans_emb = tf.nn.embedding_lookup(embeddings,self._ans)
                print('ques_emb:',ques_emb.shape)
                print('ans_emb',ans_emb.shape)
            with tf.variable_scope('preprocess_layer') as prep_l:
                sig_den = tf.layers.Dense(self.hidden_dim,name='sigmoid_dense')
                tan_den = tf.layers.Dense(self.hidden_dim,name='tanh_dense')

                ques_sig = sig_den(ques_emb)
                ques_sig = tf.sigmoid(ques_sig)
                ques_tan = tan_den(ques_emb)
                ques_tan = tf.tanh(ques_tan)
                ques_h = tf.multiply(ques_sig,ques_tan)

                ans_sig = sig_den(ans_emb)
                ans_sig = tf.sigmoid(ans_sig)
                ans_tan = tan_den(ans_emb)
                ans_tan = tf.tanh(ans_tan)
                ans_h = tf.multiply(ans_sig,ans_tan)
            with tf.variable_scope('attention_softalign') as att_align_l:
                ques_att_matrix = self.getAttMat(ques_h,ans_h)
                ans_att_matrix = self.getAttMat(ans_h,ques_h)
                print('ques_att_matrix:',ques_att_matrix.shape)
                ques_align = self.getAlign(ans_h,ques_att_matrix,self._ques_filter_len)
                ans_align = self.getAlign(ques_h,ans_att_matrix,self._ans_filter_len)
                print('ques_align:',ques_align.shape)

                ques_aligned = tf.multiply(tf.multiply(ques_align,ques_h),self._ques_align_len)
                ans_aligned = tf.multiply(tf.multiply(ans_align,ans_h),self._ans_align_len)
            with tf.variable_scope('cnn_feature') as cnn_l:
                self.cnn_ques = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='q_conv_'+str(i)) for i in range(1,6)]
                self.cnn_ans = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='a_conv_'+str(i)) for i in range(1,6)]

                ques_cnn = self.convXd_listwise(self.cnn_ques,ques_aligned,self._ques_align_len,1)
                ans_cnn = self.convXd_listwise(self.cnn_ans,ans_aligned,self._ans_align_len,1)

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
                self.score = tf.reshape(self.score,shape=(self.batch_size,self.list_size))
                self.logit_score = tf.nn.log_softmax(self.score,dim=-1)
                print('logit_score:',self.logit_score.shape)
            with tf.variable_scope('loss') as loss_l:
                #self.loss_pointwise = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.p_label,logits=tf.squeeze(self.score,-1)))
                self.loss_listwise = tf.reduce_mean(self.l_label*(tf.log(tf.clip_by_value(self.l_label,1e-5,1.0))-self.logit_score))

                #self.total_loss = self.loss_pointwise + self.loss_listwise

    def _build_thresh_listwise(self):
        with tf.variable_scope('inputs') as inputs:
            self.r_ques = tf.placeholder(tf.int32,[None,None,self.ques_len],name='ques_point')
            self.r_ques_len = tf.placeholder(tf.float32,[None,None,self.ques_len],name='ques_len_point')
            self.r_ans = tf.placeholder(tf.int32,[None,None,self.ans_len],name='ans_point')
            self.r_ans_len = tf.placeholder(tf.float32,[None,None,self.ans_len],name='ans_len_point')

            self._ques_filter_len = tf.tile(tf.expand_dims(self.r_ans_len,2),[1,1,self.ques_len,1])
            self._ans_filter_len = tf.tile(tf.expand_dims(self.r_ques_len,2),[1,1,self.ans_len,1])

            self._ques_align_len = tf.expand_dims(self.r_ques_len,3)
            self._ans_align_len = tf.expand_dims(self.r_ans_len,3)

            #self.p_label = tf.placeholder(tf.float32,[None,None])
            self.l_label = tf.placeholder(tf.float32,[None,None])

            self.batch_size, self.list_size = tf.shape(self.r_ans)[0], tf.shape(self.r_ans)[1]

            self._ques = tf.reshape(self.r_ques,shape=(-1,self.ques_len))
            self._ans = tf.reshape(self.r_ans,shape=(-1,self.ans_len))
            self._ques_filter_len = tf.reshape(self._ques_filter_len,shape=(-1,self.ques_len,self.ans_len))
            self._ans_filter_len = tf.reshape(self._ans_filter_len,shape=(-1,self.ans_len,self.ques_len))
            self._ques_align_len = tf.reshape(self._ques_align_len,shape=(-1,self.ques_len,1))
            self._ans_align_len = tf.reshape(self._ans_align_len,shape=(-1,self.ans_len,1))

            self.is_train = tf.placeholder(tf.bool)
            self.dc = tf.placeholder(tf.bool)
        with tf.name_scope('list_wise'):
            with tf.variable_scope('embedding_layer') as embedding_l:
                weights = np.load(self.embedding_file)
                weights[0] = np.zeros((weights.shape[1]))
                embeddings = tf.Variable(weights,dtype=tf.float32,trainable=False)

                ques_emb = tf.nn.embedding_lookup(embeddings,self._ques)
                ans_emb = tf.nn.embedding_lookup(embeddings,self._ans)
                print('ques_emb:',ques_emb.shape)
                print('ans_emb',ans_emb.shape)
            with tf.variable_scope('preprocess_layer') as prep_l:
                sig_den = tf.layers.Dense(self.hidden_dim,name='sigmoid_dense')
                tan_den = tf.layers.Dense(self.hidden_dim,name='tanh_dense')

                ques_sig = sig_den(ques_emb)
                ques_sig = tf.sigmoid(ques_sig)
                ques_tan = tan_den(ques_emb)
                ques_tan = tf.tanh(ques_tan)
                ques_h = tf.multiply(ques_sig,ques_tan)

                ans_sig = sig_den(ans_emb)
                ans_sig = tf.sigmoid(ans_sig)
                ans_tan = tan_den(ans_emb)
                ans_tan = tf.tanh(ans_tan)
                ans_h = tf.multiply(ans_sig,ans_tan)

            with tf.variable_scope('thresh_net') as thresh_l:
                #Thresh_sig = tf.layers.Dense(1,activation=tf.tanh,kernel_initializer=tf.initializers.zeros,name='threshold')
                Thresh_tan = tf.layers.Dense(1,activation=tf.tanh,kernel_initializer=tf.initializers.zeros,name='threshold_tan')
                #Thresh_sig = tf.layers.Dense(1,activation=tf.sigmoid,kernel_initializer=tf.initializers.zeros,name='threshold_sig')

                ques_mean = tf.tile(tf.reduce_mean(ques_h*self._ques_align_len,axis=1,keep_dims=True),[1,self.ans_len,1])
                ans_mean = tf.tile(tf.reduce_mean(ans_h*self._ans_align_len,axis=1,keep_dims=True),[1,self.ques_len,1])
                #ques_att_thresh_sig = Thresh_sig(ques_h)*self._ques_align_len
                ques_att_thresh = Thresh_tan(tf.concat([ques_h,ans_mean],axis=-1))*self._ques_align_len
                #ques_att_thresh = tf.multiply(ques_att_thresh_sig,ques_att_thresh_tan)

                #ans_att_thresh_sig = Thresh_sig(ans_h)*self._ans_align_len
                ans_att_thresh = Thresh_tan(tf.concat([ans_h,ques_mean],axis=-1))*self._ans_align_len
                #ans_att_thresh = tf.multiply(ans_att_thresh_sig,ans_att_thresh_tan)

            with tf.variable_scope('attention_softalign') as att_align_l:
                ques_att_matrix = self.getAttMat(ques_h,ans_h)
                ans_att_matrix = self.getAttMat(ans_h,ques_h)
                print('ques_att_matrix:',ques_att_matrix.shape)
                ques_align = self.getAlign(ans_h,ques_att_matrix,self._ques_filter_len,ques_att_thresh,clip_flag=self.dc)
                ans_align = self.getAlign(ques_h,ans_att_matrix,self._ans_filter_len,ans_att_thresh,clip_flag=self.dc)
                print('ques_align:',ques_align.shape)

                ques_aligned = tf.multiply(tf.multiply(ques_align,ques_h),self._ques_align_len)
                ans_aligned = tf.multiply(tf.multiply(ans_align,ans_h),self._ans_align_len)
            with tf.variable_scope('cnn_feature') as cnn_l:
                self.cnn_ques = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='q_conv_'+str(i)) for i in range(1,6)]
                self.cnn_ans = [tf.layers.Conv1D(self.hidden_dim,i,padding='same',activation=tf.nn.relu,name='a_conv_'+str(i)) for i in range(1,6)]

                ques_cnn = self.convXd_listwise(self.cnn_ques,ques_aligned,self._ques_align_len,1)
                ans_cnn = self.convXd_listwise(self.cnn_ans,ans_aligned,self._ans_align_len,1)

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

                #finalo1 = tf.concat([ques_o1,q_a_syn,ans_o1],axis=-1)
                finalo1 = tf.concat([ques_o1,ans_o1],axis=-1)

                finalo2 = tf.layers.dense(finalo1,self.hidden_dim,activation=tf.tanh,name='finalout')
                self.score = tf.layers.dense(finalo2,1,name='score')
                print('score:',self.score.shape)
                self.score = tf.reshape(self.score,shape=(self.batch_size,self.list_size))
                self.logit_score = tf.nn.log_softmax(self.score,dim=-1)
                print('logit_score:',self.logit_score.shape)
            with tf.variable_scope('loss') as loss_l:
                #self.loss_pointwise = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.p_label,logits=tf.squeeze(self.score,-1)))
                self.loss_listwise = tf.reduce_mean(self.l_label*(tf.log(tf.clip_by_value(self.l_label,1e-5,1.0))-self.logit_score))

                #self.total_loss = self.loss_pointwise + self.loss_listwise

    @staticmethod
    def getAttMat(sent1,sent2):
        return tf.matmul(sent1,sent2,transpose_b=True)

    @staticmethod
    def getAlign(sent,matrix,sent_len=None,thresh=None,clip_flag=None):
        matrix_e = tf.exp(matrix-tf.reduce_max(matrix,-1,keep_dims=True))
        if(sent_len is not None):
            matrix_e = tf.multiply(matrix_e,sent_len)
        matrix_s = tf.reduce_sum(matrix_e,-1,keep_dims=True)
        matrix_sm = matrix_e/matrix_s
        #self.matrix_sm = matrix_sm
        def thresh_clip():
            matrix_clip_e = tf.where(tf.less_equal(matrix_sm,thresh),tf.zeros_like(matrix_sm),matrix_sm)
            matrix_clip_s = tf.reduce_sum(matrix_clip_e,-1,keep_dims=True)
            matrix_clip_sm = matrix_clip_e/matrix_clip_s
            return matrix_clip_sm
        if(clip_flag is not None and thresh is not None):
            matrix_sm = tf.cond(clip_flag,thresh_clip,lambda: matrix_sm)
        return tf.matmul(matrix_sm,sent), matrix_sm

    @staticmethod
    def convXd_listwise(convXdfn,sent,sent_len,pooling_dim=None,keep_dims=False):
        cnn_out = tf.concat([convXdfn[i](sent)*sent_len for i in range(len(convXdfn))],axis=-1)
        if(pooling_dim is not None):
            cnn_out = tf.reduce_max(cnn_out,pooling_dim,keep_dims=keep_dims)
        return cnn_out
    @staticmethod
    def local_summarize(self,localfn,sent,sentl,kernel_size,merge_dim,sent_len=None,Serial=True):
        if(sent_len is None):
            local_out = tf.concat([tf.reshape(tf.tile(tf.expand_dims(localfn[i](sent),2),[1,1,kernel_size[i],1,1]),(-1,sentl,self.hidden_dim)) for i in range(len(localfn))],axis=merge_dim)
        elif(Serial):
            #print(tf.tile(tf.expand_dims(localfn[0](sent),2),[1,1,kernel_size[0],1,1]).shape)
            #print(tf.tile(tf.expand_dims(localfn[1](sent),2),[1,1,kernel_size[1],1,1]).shape)
            #print(tf.tile(tf.expand_dims(localfn[2](sent),2),[1,1,kernel_size[2],1,1]).shape)
            #print(tf.tile(tf.expand_dims(localfn[3](sent),2),[1,1,kernel_size[3],1,1]).shape)
            #local_out = tf.concat([tf.reshape(tf.tile(tf.expand_dims(localfn[i](sent)/float(kernel_size[i]),2),[1,1,kernel_size[i],1,1]),(-1,sentl,self.hidden_dim))*sent_len for i in range(len(localfn))],axis=merge_dim)
            #print((tf.squeeze(localfn[0](sent)/float(kernel_size[0]),-1)*sent_len).shape)
            local_out = tf.concat([tf.squeeze(localfn[i](sent)/float(kernel_size[i]),-1)*sent_len for i in range(len(localfn))],axis=merge_dim)
        else:
            local_out = tf.concat([tf.reshape(tf.tile(tf.expand_dims(localfn[i](sent)/float(kernel_size[i]),2),[1,1,kernel_size[i],1,1]),(-1,sentl,self.hidden_dim))*sent_len for i in range(len(localfn))],axis=merge_dim)
        print(local_out.shape)
        return local_out

    @staticmethod
    def att_matrix_reweight(matrix,filt):
        matrix_copy = matrix
        matrix = matrix*tf.transpose(filt,(0,2,1))
        matrix_e = tf.exp(matrix-tf.reduce_max(matrix,1,keep_dims=True))
        matrix_e = matrix*tf.transpose(filt,(0,2,1))
        matrix_s = tf.reduce_sum(matrix_e,1,keep_dims=True)
        matrix_sm = matrix_e/matrix_s
        return matrix_sm



if __name__ == '__main__':
    class Model_Param():
        batch_size = 64
        hidden_dim = 300
        list_size = 15
        ques_len = 25
        ans_len = 90
        embedding_file = '../data/wikiqa/wikiqa_glovec.txt'
    m_p = Model_Param()
    base_line = My_Model(m_p)
    #base_line._build_my_model_listwise()
    #base_line._build_syn_ext_listwise()
    #base_line._build_att_ext_listwise()
    #base_line._build_thresh_listwise()
    #base_line._build_syn_ext2_listwise()
    base_line._build_syn_ext3_listwise()
    #base_line._build_syn_ext4_listwise()
    #base_line._build_syn_ext5_listwise()
    #base_line._build_syn_ext6_listwise()
