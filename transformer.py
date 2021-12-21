#!/usr/bin/env python
# coding: utf-8

import sys
import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn, layers

def normalize(inputs,
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta

    return outputs


def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              scope="embedding",
              reuse=tf.AUTO_REUSE,
              ptr_embeddings=None,
              padding_index=-1):

    with tf.variable_scope(scope, reuse=reuse):
        if ptr_embeddings is not None:
            ptr_embeddings = tf.convert_to_tensor(ptr_embeddings)
            lookup_table = tf.get_variable('lookup_table',
                                           dtype=tf.float32,
                                           initializer=ptr_embeddings)
        else:
            lookup_table = tf.get_variable('lookup_table',
                                           dtype=tf.float32,
                                           shape=[vocab_size, num_units],
                                           initializer=tf.contrib.layers.xavier_initializer())

            # lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
            #                           lookup_table[1:, :]), 0)
            if zero_pad:
                if padding_index == -1:
                    raise ValueError('padding index != -1 expected')
                tmp_list = tf.unstack(lookup_table)
                tmp_list[padding_index] = tf.zeros(num_units)
                lookup_table = tf.stack(tmp_list)

        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if outputs.get_shape().as_list()[-1] != num_units:
            outputs = tf.layers.dense(outputs, num_units, tf.nn.sigmoid)

        if scale:
            outputs = outputs * (num_units ** 0.5)

    return outputs


def multihead_attention(key_emb,
                        que_emb,
                        queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None,
                        save_attns_for_visualization=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]

        ### 输入embedding reshape 成 num_units 维
        if keys.get_shape().as_list()[-1] != num_units:
            keys = tf.layers.dense(keys, num_units, activation=tf.nn.sigmoid)
        if queries.get_shape().as_list()[-1] != num_units:
            queries = tf.layers.dense(queries, num_units, activation=tf.nn.sigmoid)

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.sigmoid)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.sigmoid)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.sigmoid)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(key_emb, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(que_emb, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Attention Visualization
        if save_attns_for_visualization is not None:
            if isinstance(save_attns_for_visualization, list):
                save_attns_for_visualization.append(tf.transpose(tf.convert_to_tensor(
                    tf.split(outputs, num_heads, axis=0)), [1,0,2,3])) ### (N, h, T_q, T_k)
            else:
                raise NotImplementedError

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs)  # (N, T_q, C)

    return outputs


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                reuse=None,
                residual=True,
                layer_norm=True,
                layer_residual=False):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # # Inner layer
        # params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
        #           "activation": tf.nn.relu, "use_bias": True}
        # outputs = tf.layers.conv1d(**params)
        #
        # # Readout layer
        # params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
        #           "activation": None, "use_bias": True}
        # outputs = tf.layers.conv1d(**params)

        outputs = inputs
        for hid_units in num_units[:-1]:
            params = {"inputs": outputs, "filters": hid_units, "kernel_size": 1,
                                "activation": tf.nn.sigmoid, "use_bias": True}
            outputs_prev = outputs
            outputs = tf.layers.conv1d(**params)
            if residual and layer_residual:
                outputs += outputs_prev

        params = {"inputs": outputs, "filters": num_units[-1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs_prev = outputs
        outputs = tf.layers.conv1d(**params)
        if residual and layer_residual:
            outputs += outputs_prev

        # Residual connection
        if residual and not layer_residual:
            assert num_units[-1] == inputs.get_shape().as_list()[-1]
            outputs += inputs

        # Normalize
        if layer_norm:
            outputs = normalize(outputs)

    return outputs


def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(
        (tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


class Transformer():
    def __init__(self, args):
        tf.reset_default_graph()
        with tf.variable_scope('input'):
            if args.is_variable_len:
                self.x = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_max_len+1, args.feat_num])  ### 注
            else:
                self.x = tf.placeholder(dtype=tf.float32, shape=[None, args.act_seq_num+1, args.feat_num])  ### 注
            if args.auto_regressive:
                self.y = tf.placeholder(dtype=tf.float32, shape=[None, args.feat_num])
                self.lw = tf.placeholder(dtype=tf.float32, shape=[args.feat_num])
            else:
                self.y = tf.placeholder(dtype=tf.float32, shape=[None, args.label_num])
                self.lw = tf.placeholder(dtype=tf.float32, shape=[args.label_num])
            self.seq_lens = tf.placeholder(dtype=tf.int32, shape=[None])

            # truncate path text length
            if self.x.get_shape()[-1] > args.pos_max_length:
                self.x = tf.slice(self.x, [0, 0, 0], [args.batch_size, args.pos_max_length, args.feat_num])
            else:
                self.x = self.x[:, :args.pos_max_length] # ???

            input_x = self.x
            if input_x.get_shape()[-1] != args.d_model:
                input_x = tf.layers.dense(input_x, args.d_model, activation=tf.nn.sigmoid, name='transform')

            ### postition embedding
            h = input_x + embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(input_x)[1]), 0), [tf.shape(input_x)[0], 1]),
                vocab_size=args.pos_max_length, num_units=args.d_model, zero_pad=False, scale=False,
                scope="position_emb")

            ## Dropout
            h = tf.layers.dropout(h, rate=1 - args.trm_keep_prop,
                                         training=tf.convert_to_tensor(args.istrain))

        # Encoder
        with tf.variable_scope("encoder"):

            ## Blocks
            self.enc_attns = []
            for block_ind in range(args.enc_block_num):
                with tf.variable_scope("block_{}".format(block_ind)):
                    ### Multihead Attention
                    h = multihead_attention(key_emb=self.x,
                                           que_emb=self.x,
                                           queries=h,
                                           keys=h,
                                           num_units=args.d_model,
                                           num_heads=args.trm_head_num,
                                           dropout_rate=1-args.trm_keep_prop,
                                           is_training=args.istrain,
                                           causality=False,
                                           scope='self_attention',
                                           save_attns_for_visualization=self.enc_attns)

                    ### Feed Forward
                    h = feedforward(h, num_units=[4 * args.d_model, args.d_model], scope='ffn')

            self.enc_attns = tf.convert_to_tensor(self.enc_attns)

        with tf.variable_scope("pooler"):
            # We "pool" the model by simply taking the hidden state corresponding
            # to the first token. We assume that this has been pre-trained
            sentence_cls = tf.squeeze(h[:, 0:1, :], axis=1)
            sentence_first = tf.squeeze(h[:, 1:2, :], axis=1)
            sentence_sum = tf.reduce_sum(h[:, 1:, :], axis=1)
            sentence_mean = tf.reduce_mean(h[:, 1:, :], axis=1)
            sentence_max = tf.reduce_max(h[:, 1:, :], axis=1)
            sentence_conv = tf.contrib.layers.conv1d(h[:, 1:, :], h.shape.as_list()[2],
                                                     kernel_size=h.shape.as_list()[1]-1,
                                                     activation_fn=tf.nn.relu, padding='valid')
            sentence_conv = tf.squeeze(sentence_conv, [1])
            if args.SE_select == 'cls':
                encode_vector = sentence_cls
            elif args.SE_select == 'first':
                encode_vector = sentence_first
            elif args.SE_select == 'sum':
                encode_vector = sentence_sum
            elif args.SE_select == 'mean':
                encode_vector = sentence_mean
            elif args.SE_select == 'max':
                encode_vector = sentence_max
            elif args.SE_select == 'conv':
                encode_vector = sentence_conv
            else:
                print("Not implemented error: args.model_select only allows ['cls', 'first', 'sum', 'mean', 'max']")
                sys.exit(1)
            pooled_output = tf.layers.dense(
                encode_vector,
                args.d_model,
                activation=tf.tanh,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.2))

        with tf.variable_scope('predictor'):
            for pred_layer_ind in range(args.trm_pred_layer_num):
                with tf.variable_scope('ffn_{}'.format(pred_layer_ind)):
                    pooled_output = tf.layers.dense(pooled_output, args.d_model, activation=tf.nn.sigmoid,
                                                    bias_initializer=tf.constant_initializer(0.1))
        with tf.variable_scope('final_output'):
            if args.auto_regressive:
                self.preds = tf.layers.dense(pooled_output, args.feat_num, activation=None,
                                             bias_initializer=tf.constant_initializer(0.1))
                if args.ensemble_naive:
                    self.naive_preds = self.x[:, -1, :]
            else:
                self.preds = tf.layers.dense(pooled_output, args.label_num, activation=None,
                                             bias_initializer=tf.constant_initializer(0.1))
                if args.ensemble_naive:
                    self.naive_preds = tf.tile(self.x[:, -1, :], (1, args.label_num)) # 有问题，不过用不上了
            if args.ensemble_naive:
                self.alpha = tf.Variable(0.5, name='ensemble_w')
                self.preds = self.alpha * self.preds + (1 - self.alpha) * self.naive_preds

        self.model_mer = tf.abs((self.preds - self.y) / self.y)

        # tv = tf.trainable_variables()
        # reg_loss = 0.001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv ])
        self.model_mse = tf.square(self.preds - self.y)  # 平均平方误差
        self.model_weighted_mse = tf.reduce_mean(self.lw * self.model_mse)
        self.model_weighted_mer = tf.reduce_mean(self.lw * self.model_mer)  # 平均误差率
        # self.model_mer = tf.reduce_mean(tf.abs((self.preds - self.y) / self.y)) # 平均误差率
        # self.loss = tf.reduce_mean(tf.square((self.output - self.y) / self.y), axis=0)
        # self.loss = self.model_weighted_mse
        self.loss = self.model_weighted_mer

        # Training Scheme
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-8)
        self.train_step = optimizer.minimize(self.loss, global_step=global_step)