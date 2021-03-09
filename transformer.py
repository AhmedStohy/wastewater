#!/usr/bin/env python
# coding: utf-8

import sys
import tensorflow as tf
import numpy as np

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
    '''Embeds a given tensor.
    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.

    For example,

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]
     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]
     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]
    ```
    '''
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
            outputs = tf.layers.dense(outputs, num_units, tf.nn.relu)

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
            keys = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
        if queries.get_shape().as_list()[-1] != num_units:
            queries = tf.layers.dense(queries, num_units, activation=tf.nn.relu)

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

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
                                "activation": tf.nn.relu, "use_bias": True}
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


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.

    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],
      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],
       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    '''
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)


def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(
        (tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

class Graph():
    def __init__(self, arg, embeddings=None):
        tf.reset_default_graph()
        self.is_training = arg.is_training
        self.use_pretrain = arg.use_pretrain
        self.hidden_units = arg.hidden_units
        self.vocab_size_path = arg.vocab_size_path
        self.vocab_size_pos = arg.vocab_size_pos
        self.num_heads = arg.num_heads
        self.num_blocks = arg.num_blocks
        self.max_length = arg.max_length
        self.lr = arg.lr
        self.dropout_rate = arg.dropout_rate
        self.ptr_embeddings = embeddings
        self.padding_index_path = arg.padding_index_path
        self.padding_index_pos = arg.padding_index_pos
        self.model_select = arg.model_select
        self.class_num = arg.class_num
        self.stat_hid_units = arg.stat_hid_units
        self.stat_dims = arg.stat_dims
        self.SE_select = arg.SE_select

        # input placeholder
        self.x_path = tf.placeholder(tf.int32, shape=(None, None))
        self.x_pos = tf.placeholder(tf.int32, shape=(None, None))
        self.x_stat = tf.placeholder(tf.float32, shape=(None, self.stat_dims))
        self.y = tf.placeholder(tf.int32, shape=(None,))

        # truncate path text length
        if self.x_path.get_shape()[-1] > self.max_length:
            self.x_path = tf.slice(self.x_path, [0, 0], [arg.batch_size, self.max_length])
            self.x_pos = tf.slice(self.x_pos, [0, 0], [arg.batch_size, self.max_length])
        else:
            self.x_path = self.x_path[:, :self.max_length]
            self.x_pos = self.x_pos[:, :self.max_length]

        # Encoder
        with tf.variable_scope("encoder"):

            # embedding
            if self.use_pretrain and self.is_training:
                if self.ptr_embeddings is None:
                    raise ValueError('Pretrain embeddings are None')
                self.enc_emb_path = embedding(self.x_path, vocab_size=self.vocab_size_path, num_units=self.hidden_units,
                                            scale=True, scope="embedding_path", ptr_embeddings=self.ptr_embeddings)
            else:
                self.enc_emb_path = embedding(self.x_path, vocab_size=self.vocab_size_path, num_units=self.hidden_units,
                                            scale=True, scope="embedding_path", padding_index=self.padding_index_path)

            self.enc_emb_pos = embedding(self.x_pos, vocab_size=self.vocab_size_pos, num_units=self.hidden_units,
                                            scale=True, scope="embedding_pos", padding_index=self.padding_index_pos)

            ### postition embedding
            self.enc_path = self.enc_emb_path + embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.x_path)[1]), 0), [tf.shape(self.x_path)[0], 1]),
                vocab_size=self.max_length, num_units=self.hidden_units, zero_pad=False, scale=False,
                scope="embedding_position")

            self.enc_pos = self.enc_emb_pos + embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.x_pos)[1]), 0), [tf.shape(self.x_pos)[0], 1]),
                vocab_size=self.max_length, num_units=self.hidden_units, zero_pad=False, scale=False,
                scope="embedding_position")

            if self.model_select == 'bert':
                self.enc = self.enc_path
            elif self.model_select in ['bertSIE', 'bertSIE_stat', 'stat']:
                self.enc = self.enc_path + self.enc_pos
            else:
                print("Not implemented error: arg.model_select only allows 'bert' or 'bertSIE' or 'bertSIE_stat'")
                sys.exit(1)

            ## Dropout
            self.enc = tf.layers.dropout(self.enc,
                                         rate=self.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))

            ## Blocks
            self.enc_attns = []
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    self.enc = multihead_attention(key_emb=self.enc_emb_path,
                                                   que_emb=self.enc_emb_path,
                                                   queries=self.enc,
                                                   keys=self.enc,
                                                   num_units=self.hidden_units,
                                                   num_heads=self.num_heads,
                                                   dropout_rate=self.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=False,
                                                   scope='self_attention',
                                                   save_attns_for_visualization=self.enc_attns)

                    ### Feed Forward
                    self.enc = feedforward(self.enc, num_units=[4 * self.hidden_units, self.hidden_units], scope='output')

            self.enc_attns = tf.convert_to_tensor(self.enc_attns)

        with tf.variable_scope("pooler"):
            # We "pool" the model by simply taking the hidden state corresponding
            # to the first token. We assume that this has been pre-trained
            sentence_cls = tf.squeeze(self.enc[:, 0:1, :], axis=1)
            sentence_first = tf.squeeze(self.enc[:, 1:2, :], axis=1)
            sentence_sum = tf.reduce_sum(self.enc[:, 1:, :], axis=1)
            sentence_mean = tf.reduce_mean(self.enc[:, 1:, :], axis=1)
            sentence_max = tf.reduce_max(self.enc[:, 1:, :], axis=1)
            if self.SE_select == 'cls':
                encode_vector = sentence_cls
            elif self.SE_select == 'first':
                encode_vector = sentence_first
            elif self.SE_select == 'sum':
                encode_vector = sentence_sum
            elif self.SE_select == 'mean':
                encode_vector = sentence_mean
            elif self.SE_select == 'max':
                encode_vector = sentence_max
            else:
                print("Not implemented error: arg.model_select only allows ['cls', 'first', 'sum', 'mean', 'max']")
                sys.exit(1)
            self.pooled_output = tf.layers.dense(
                encode_vector,
                arg.hidden_units,
                activation=tf.tanh,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.2))

        if self.model_select in ['bertSIE_stat', 'stat']:
            with tf.variable_scope("stat_encoder"):
                self.stat_feats = feedforward(tf.expand_dims(self.x_stat, axis=1), self.stat_hid_units, scope='feed_foward', residual=True)
                self.stat_feats = tf.squeeze(self.stat_feats, axis=1)

                # self.stat_feats = self.x_stat

            if self.model_select == 'bertSIE_stat':
                self.final_output = tf.concat([self.pooled_output, self.stat_feats], axis=-1)
            elif self.model_select == 'stat':
                self.final_output = self.stat_feats
        else:
            self.final_output = self.pooled_output

        # Final linear projection
        self.logits = tf.layers.dense(self.final_output, self.class_num)
        self.probs = tf.nn.softmax(self.logits)
        self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
        # self.istarget = tf.to_float(tf.not_equal(self.y, self.padding_index_out)) ### shape=[batch_size, batch_max+length]
        self.batch_correct_pred = tf.to_float(tf.equal(self.preds, self.y))
        self.batch_total_acc = tf.reduce_sum(self.batch_correct_pred)
        self.batch_TP = tf.reduce_sum(self.batch_correct_pred * tf.to_float(self.y))
        self.batch_TPFN = tf.to_float(tf.reduce_sum(self.y))
        self.batch_TPFP = tf.to_float(tf.reduce_sum(self.preds))
        # tf.summary.scalar('acc', tf.reduce_mean(self.batch_total_acc))

        # Loss
        self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=self.class_num))
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y_smoothed)  ### shape=[batch_size, batch_max+length]
        self.batch_total_loss = tf.reduce_sum(self.loss)

        # Training Scheme
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
        self.train_op = self.optimizer.minimize(self.batch_total_loss, global_step=self.global_step)

        # # Summary
        # tf.summary.scalar('mean_loss', self.batch_total_loss)
        # self.merged = tf.summary.merge_all()