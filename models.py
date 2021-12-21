import tensorflow as tf

from tensorflow.contrib import rnn, layers
from tensorflow.contrib.layers import conv1d, conv2d
from tensorflow.contrib.layers.python.layers import batch_norm


class MLP(object):

    def __init__(self, args):
        tf.reset_default_graph()
        # self.x = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_num, args.feat_num])
        with tf.variable_scope('input'):
            if args.is_variable_len:
                self.x = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_max_len, args.feat_num])  ### 注
            else:
                self.x = tf.placeholder(dtype=tf.float32, shape=[None, args.act_seq_num, args.feat_num])  ### 注
            if args.auto_regressive:
                self.y = tf.placeholder(dtype=tf.float32, shape=[None, args.feat_num])
                self.lw = tf.placeholder(dtype=tf.float32, shape=[args.feat_num])
            else:
                self.y = tf.placeholder(dtype=tf.float32, shape=[None, args.label_num])
                self.lw = tf.placeholder(dtype=tf.float32, shape=[args.label_num])
            self.seq_lens = tf.placeholder(dtype=tf.int32, shape=[None])

            input_x = self.x
            if args.is_variable_len:
                input_x = tf.reshape(input_x, (-1, args.seq_max_len * args.feat_num))
            else:
                input_x = tf.reshape(input_x, (-1, args.act_seq_num * args.feat_num))
            if args.use_last_n_step > 0:
                input_x = input_x[:, -args.use_last_n_step * args.feat_num:]

        with tf.variable_scope('encoder'):
            h = input_x
            for layer_ind, layer_dim in enumerate(args.mlp_hid_dims):
                with tf.variable_scope('layer_{}'.format(layer_ind)):
                    # h = Dense(layer, activation='relu')(h)
                    h = tf.layers.dense(inputs=h, units=layer_dim, activation=tf.nn.tanh,
                                        # kernel_initializer=tf.glorot_uniform_initializer(),
                                        kernel_initializer=tf.truncated_normal_initializer(),
                                        kernel_regularizer=layers.l2_regularizer(0.003),
                                        bias_initializer=tf.constant_initializer(0.1))

        with tf.variable_scope('predictor'):
            if args.auto_regressive:
                self.preds = tf.layers.dense(inputs=h, units=args.feat_num, activation=None,
                                             # kernel_initializer=tf.glorot_uniform_initializer(),
                                             kernel_initializer=tf.truncated_normal_initializer(),
                                             kernel_regularizer=layers.l2_regularizer(0.003),
                                             bias_initializer=tf.constant_initializer(0.1))
            else:
                self.preds = tf.layers.dense(inputs=h, units=args.label_num, activation=None,
                                             # kernel_initializer=tf.glorot_uniform_initializer(),
                                             kernel_initializer=tf.truncated_normal_initializer(),
                                             kernel_regularizer=layers.l2_regularizer(0.003),
                                             bias_initializer=tf.constant_initializer(0.1))
        self.model_mse = tf.square(self.preds - self.y)  # 平均平方误差
        self.model_mer = tf.abs((self.preds - self.y) / self.y)

        model_weighted_mse = tf.reduce_mean(self.lw * self.model_mse)
        model_weighted_mer = tf.reduce_mean(self.lw * self.model_mer)  # 平均误差率

        if args.loss_function == 'mse':
            loss = model_weighted_mse
        elif args.loss_function == 'mer':
            loss = model_weighted_mer
        else:
            raise NotImplementedError

        global_step = tf.Variable(0, trainable=False)
        self.train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=global_step)


def layer_normalize(inputs, epsilon = 1e-8):

    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]

    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    beta= tf.Variable(tf.zeros(params_shape) + 0.1)
    gamma = tf.Variable(tf.ones(params_shape))
    normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
    outputs = gamma * normalized + beta

    return outputs


def feed_forward_linear(inputs, hid_dims, use_layer_norm=True, use_residual=True):
    assert len(hid_dims) == 2
    assert inputs.get_shape()[-1] == hid_dims[1]

    if use_layer_norm:
        hidden = layer_normalize(inputs)
    else:
        hidden = inputs

    hidden = layers.fully_connected(hidden, hid_dims[0], tf.nn.sigmoid)
    hidden = layers.fully_connected(hidden, hid_dims[1], None)

    if use_residual:
        hidden += inputs

    return hidden


class LSTM(object):

    def lstm_block(self, raw_x, hid_dim, head_num, istrain, dropout_keep_prop=1.0, use_layer_norm=True,
                   use_residual=True, is_variable_len=False, sequence_length=None):
        assert hid_dim % head_num == 0

        ### transform
        if raw_x.get_shape()[-1] != hid_dim:
            raw_x = tf.layers.dense(raw_x, hid_dim, activation=tf.nn.sigmoid, name='input_transform')

        ### multi head
        raw_x_reshape = tf.concat(tf.split(raw_x, head_num, axis=2), axis=0)

        if use_layer_norm:
            input_x = layer_normalize(raw_x_reshape)
        else:
            input_x = raw_x_reshape

        stacked_lstm = []
        lstm_fw = rnn.LSTMCell(hid_dim, use_peepholes=True, activation=tf.nn.softsign)
        # lstm_fw = rnn.SRUCell(hid_dim, activation=tf.nn.softsign)
        # lstm_fw = rnn.GRUCell(hid_dim, activation=tf.nn.softsign)
        # lstm_fw = rnn.BasicRNNCell(hid_dim, activation=tf.nn.softsign)
        stacked_lstm.append(lstm_fw)
        multilayer_lstm = rnn.MultiRNNCell(stacked_lstm)
        if not is_variable_len:
            output_matrix, output_states = tf.nn.dynamic_rnn(multilayer_lstm, input_x, dtype=tf.float32)
        else:
            output_matrix, output_states = tf.nn.dynamic_rnn(multilayer_lstm, input_x, dtype=tf.float32,
                                                             sequence_length=sequence_length)

        # lstm_fw = rnn.MultiRNNCell([rnn.SRUCell(hid_dim, activation=tf.nn.softsign)])
        # lstm_bw = rnn.MultiRNNCell([rnn.SRUCell(hid_dim, activation=tf.nn.softsign)])
        # output_matrix, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw, input_x, dtype=tf.float32)
        # output_matrix = tf.concat(output_matrix, -1)
        # output_states = tf.concat(output_states, -1)


        output_matrix = tf.layers.dropout(output_matrix, rate=1.0-dropout_keep_prop, training=istrain)
        output_matrix = tf.concat(tf.split(output_matrix, head_num, axis=0), axis=2)

        if use_residual:
            output_matrix += raw_x

        return output_matrix, output_states


    def __init__(self, args):
        tf.reset_default_graph()
        with tf.variable_scope('input'):
            # self.x = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_num, args.feat_num])
            if args.is_variable_len:
                self.x = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_max_len, args.feat_num])  ### 注
            else:
                self.x = tf.placeholder(dtype=tf.float32, shape=[None, args.act_seq_num, args.feat_num])  ### 注
            if args.auto_regressive:
                self.y = tf.placeholder(dtype=tf.float32, shape=[None, args.feat_num])
                self.lw = tf.placeholder(dtype=tf.float32, shape=[args.feat_num])
            else:
                self.y = tf.placeholder(dtype=tf.float32, shape=[None, args.label_num])
                self.lw = tf.placeholder(dtype=tf.float32, shape=[args.label_num])
            self.seq_lens = tf.placeholder(dtype=tf.int32, shape=[None])

        self.hidden = self.x
        # if args.use_last_n_step > 0:
        #     self.hidden = self.hidden[:, -args.use_last_n_step:, :]
        with tf.variable_scope('encoder'):
            for block_ind in range(args.rnn_layer_num):
                with tf.variable_scope('lstm_block_{}'.format(block_ind)):
                    self.hidden, self.output_states = self.lstm_block(self.hidden, args.rnn_hid_dim, args.rnn_head_num,
                                                                      args.istrain, dropout_keep_prop=args.rnn_keep_prop,
                                                                      use_layer_norm=True, use_residual=False,
                                                                      is_variable_len=args.is_variable_len,
                                                                      sequence_length=self.seq_lens)
                    # self.hidden = feed_forward_linear(self.hidden, [hidden_dim*4, hidden_dim],
                    #                                   use_layer_norm=True, use_residual=True)

            with tf.variable_scope('pooler'):
                if args.is_variable_len:
                    self.preds = self.output_states[0][1]
                else:
                    if args.rnn_pool == 'last':
                        outputs = tf.transpose(self.hidden, [1, 0, 2])
                        self.preds = outputs[-1] # last pool
                    elif args.rnn_pool == 'mean':
                        self.preds = tf.reduce_mean(self.hidden, axis=1) # mean pool
                    elif args.rnn_pool == 'max':
                        self.preds = tf.reduce_max(self.hidden, axis=1) # mean pool
                    elif args.rnn_pool == 'conv1d':
                        self.preds = conv1d(self.hidden, args.rnn_hid_dim, kernel_size=self.hidden.shape.as_list()[1],
                                        activation_fn=tf.nn.relu, padding='valid')
                if args.rnn_concat_c:
                    self.preds = tf.concat((self.preds, self.output_states[0][0]), axis=-1)

                # self.preds = tf.concat([outputs[0], outputs[-1]], axis=1) # firstlast pool
                # self.preds = outputs[-3]
                # self.preds = self.output_states[0][1]
                # self.preds = tf.concat([outputs[0], self.output_states[0][1]], axis=1)
                # self.preds = tf.reduce_mean(self.output_matrix, axis=1)

        with tf.variable_scope('predictor'):
            for block_ind, hidden_dim in enumerate(args.rnn_pred_hid_dims):
                with tf.variable_scope('feedforward_{}'.format(block_ind)):
                    self.preds = tf.layers.dense(inputs=self.preds, units=hidden_dim, activation=tf.nn.sigmoid)
            with tf.variable_scope('final_output'):
                if args.auto_regressive:
                    self.preds = tf.layers.dense(inputs=self.preds, units=args.feat_num, activation=None)
                    if args.ensemble_naive:
                        self.naive_preds = self.x[:, -1, :]
                else:
                    self.preds = tf.layers.dense(inputs=self.preds, units=args.label_num, activation=None)
                    if args.ensemble_naive:
                        self.naive_preds = tf.tile(self.x[:, -1, :], (1, args.label_num))
                if args.ensemble_naive:
                    self.alpha = tf.Variable(0.5, name='ensemble_w')
                    self.preds = self.alpha * self.preds + (1 - self.alpha) * self.naive_preds

        self.model_mer = tf.abs((self.preds - self.y) / self.y)

        # tv = tf.trainable_variables()
        # reg_loss = 0.001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv ])
        self.model_mse = tf.square(self.preds - self.y)  # 平均平方误差
        model_weighted_mse = tf.reduce_mean(self.lw * self.model_mse)
        model_weighted_mer = tf.reduce_mean(self.lw * self.model_mer)  # 平均误差率

        if args.loss_function == 'mse':
            loss = model_weighted_mse
        elif args.loss_function == 'mer':
            loss = model_weighted_mer
        else:
            raise NotImplementedError

        global_step = tf.Variable(0, trainable=False)
        self.train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=global_step)


class CNN(object):

    def __init__(self, args):
        tf.reset_default_graph()
        with tf.variable_scope('input'):
            if args.is_variable_len:
                self.x = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_max_len, args.feat_num])  ### 注
            else:
                self.x = tf.placeholder(dtype=tf.float32, shape=[None, args.act_seq_num, args.feat_num])  ### 注
            if args.auto_regressive:
                self.y = tf.placeholder(dtype=tf.float32, shape=[None, args.feat_num])
                self.lw = tf.placeholder(dtype=tf.float32, shape=[args.feat_num])
            else:
                self.y = tf.placeholder(dtype=tf.float32, shape=[None, args.label_num])
                self.lw = tf.placeholder(dtype=tf.float32, shape=[args.label_num])
            self.seq_lens = tf.placeholder(dtype=tf.int32, shape=[None])

        assert args.conv_kernel_size.__len__() >= 1

        with tf.variable_scope('encoder'):
            out_lst = []
            for kernel_ind, kd in enumerate(args.conv_kernel_size):
                with tf.variable_scope('kernel_{}'.format(kernel_ind)):
                    h_conv = self.x
                    # if args.use_last_n_step > 0:
                    #     h_conv = h_conv[:, -args.use_last_n_step:, :]
                    for block_ind, ld in enumerate(args.feature_map_num[:-1]):
                        with tf.variable_scope('cnn_block_{}'.format(block_ind)):
                            h_conv = conv1d(h_conv, ld, kernel_size=kd, stride=args.conv_stride[kernel_ind],
                                            activation_fn=tf.nn.relu,
                                            padding='same')  # 注意stride, padding
                            h_conv = batch_norm(h_conv, decay=0.9, updates_collections=None, is_training=args.istrain)

                    with tf.variable_scope('pooler'):
                        if args.last_max_pool:
                            h_conv = tf.nn.pool(h_conv, [3], 'MAX', padding='VALID')
                        h_conv = conv1d(h_conv, args.feature_map_num[-1], kernel_size=h_conv.shape.as_list()[1],
                                        activation_fn=tf.nn.relu, padding='valid')
                        h_conv = tf.squeeze(h_conv, [1])
                        out_lst.append(h_conv)

            out_conv = tf.concat(out_lst, axis=-1)

        with tf.variable_scope('predictor'):
            preds = out_conv
            for l in args.cnn_pred_hid_dims:
                preds = tf.layers.dense(preds, l, activation=tf.nn.sigmoid)

            with tf.variable_scope('final_output'):
                if args.auto_regressive:
                    self.preds = tf.layers.dense(inputs=preds, units=args.feat_num, activation=None)
                else:
                    self.preds = tf.layers.dense(inputs=preds, units=args.label_num, activation=None)

        self.model_mse = tf.square(self.preds - self.y)  # 均方误差
        self.model_mer = tf.abs((self.preds - self.y) / self.y)  # 平均误差率

        # tv = tf.trainable_variables()
        # reg_loss = 0.001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv ])
        model_weighted_mse = tf.reduce_mean(self.lw * self.model_mse)
        model_weighted_mer = tf.reduce_mean(self.lw * self.model_mer)  # 加权平均误差率

        if args.loss_function == 'mse':
            loss = model_weighted_mse
        elif args.loss_function == 'mer':
            loss = model_weighted_mer
        else:
            raise NotImplementedError

        global_step = tf.Variable(0, trainable=False)
        self.train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=global_step)

        # optimizer = tf.train.AdamOptimizer(learning_rate)
        # gvs = optimizer.compute_gradients(self.loss) # 计算梯度
        # clipped_gvs = [(tf.clip_by_value(grad, -5, 5), var) for grad, var in gvs] # 梯度裁剪
        # train_step = optimizer.apply_gradients(clipped_gvs) # 梯度下降


class MLP_origin(object):

    def __init__(self, args):

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, args.total_feat_num])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, args.label_num])
        self.lw = tf.placeholder(dtype=tf.float32, shape=[args.label_num])

        h = self.x
        for layer_dim in args.hid_dims:
            # h = Dense(layer, activation='relu')(h)
            h = tf.layers.dense(inputs=h, units=layer_dim, activation=tf.nn.tanh,
                                # kernel_initializer=tf.glorot_uniform_initializer(),
                                kernel_initializer=tf.truncated_normal_initializer(),
                                kernel_regularizer=layers.l2_regularizer(0.003),
                                bias_initializer=tf.constant_initializer(0.1))
            # h = batch_norm(h, decay=0.9, updates_collections=None, is_training=istrain)
            # if istrain:
            #     h = tf.nn.dropout(h, keep_prop)

        self.output = tf.layers.dense(inputs=h, units=args.label_num, activation=None,
                                 # kernel_initializer=tf.glorot_uniform_initializer(),
                                 kernel_initializer=tf.truncated_normal_initializer(),
                                 kernel_regularizer=layers.l2_regularizer(0.003),
                                 bias_initializer=tf.constant_initializer(0.1))
        self.model_mse = tf.square(self.output - self.y)  # 平均平方误差
        self.model_mer = tf.abs((self.output - self.y) / self.y)

        model_weighted_mse = tf.reduce_mean(self.lw * self.model_mse)
        model_weighted_mer = tf.reduce_mean(self.lw * self.model_mer)  # 平均误差率

        # loss = tf.reduce_mean(tf.square((output - y) / y), axis=0)
        # loss = model_weighted_mse
        loss = model_weighted_mer

        global_step = tf.Variable(0, trainable=False)
        self.train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=global_step)


class LSTM_origin(object):

    def __init__(self, args):
        tf.reset_default_graph()

        # self.x = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_num, args.feat_num])
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, None, args.feat_num]) ### 注
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, args.label_num])
        self.lw = tf.placeholder(dtype=tf.float32, shape=[args.label_num])
        self.seq_lens = tf.placeholder(dtype=tf.int32, shape=[None])

        with tf.variable_scope('encoder'):
            self.stacked_lstm = []

            for layer_ind, layer_hid_dim in enumerate(args.hid_dims):
                with tf.variable_scope('lstm_layer_{}'.format(layer_ind)):
                    self.lstm_fw = rnn.LSTMCell(layer_hid_dim, use_peepholes=True, activation=tf.nn.softsign)
                    # lstm_bw = rnn.LSTMCell(layer_hid_dim, activation=tf.nn.softsign)

                    self.stacked_lstm.append(self.lstm_fw)

            self.multilayer_lstm = rnn.MultiRNNCell(self.stacked_lstm)

            # outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw, x, dtype=tf.float32)
            # outputs = tf.concat(outputs, 2)
            if not args.is_variable_len:
                self.output_matrix, self.output_states = tf.nn.dynamic_rnn(self.multilayer_lstm, self.x, dtype=tf.float32)
            else:
                self.output_matrix, self.output_states = tf.nn.dynamic_rnn(self.multilayer_lstm, self.x, dtype=tf.float32, sequence_length=self.seq_lens)

            self.output_matrix = tf.layers.dropout(self.output_matrix, rate=1.0 - args.keep_prop, training=args.istrain)

        with tf.variable_scope('pooler'):
            # outputs = tf.transpose(outputs, [1, 0, 2])
            # preds = outputs[-1]
            self.preds = self.output_states[0][1]
            for l in args.rnn_pred_hid_dims:
                self.preds = layers.fully_connected(self.preds, l, activation_fn=tf.nn.sigmoid)
            self.preds = layers.fully_connected(self.preds, args.label_num, activation_fn=None)

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

        self.train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)


class CNN_origin(object):

    def __init__(self, args):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, args.act_seq_num, args.feat_num])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, args.label_num])
        self.lw = tf.placeholder(dtype=tf.float32, shape=[args.label_num])

        assert args.kernel_size.__len__() >= 1

        out_lst = []
        for kd in args.kernel_size:
            h_conv = self.x
            h_pool = None
            # h_conv = None
            # h_pool = x
            for ld in args.feature_map_num[:-1]:
                h_conv = conv1d(h_conv, ld, kernel_size=kd, stride=kd, activation_fn=tf.nn.relu,
                                padding='same')  # 注意stride, padding
                # h_conv = conv1d(h_pool, ld, kernel_size=kd, stride=kd, activation_fn=tf.nn.relu, padding='valid') # 注意stride
                h_conv = batch_norm(h_conv, decay=0.9, updates_collections=None, is_training=args.istrain)
                # h_pool = tf.nn.pool(h_conv, [3], 'MAX', padding='VALID')
                # h_conv = h_pool
                h_pool = h_conv
            h_conv = conv1d(h_pool, args.feature_map_num[-1], kernel_size=h_pool.shape.as_list()[1],
                            activation_fn=tf.nn.relu, padding='valid')
            h_conv = tf.squeeze(h_conv, [1])
            out_lst.append(h_conv)

        out_conv = tf.concat(out_lst, axis=-1)

        preds = out_conv
        for l in args.cnn_pred_hid_dims:
            preds = layers.fully_connected(preds, l, activation_fn=tf.nn.sigmoid)
        self.preds = layers.fully_connected(preds, args.label_num, activation_fn=None)

        self.model_mse = tf.square(self.preds - self.y)  # 均方误差
        self.model_mer = tf.abs((self.preds - self.y) / self.y)  # 平均误差率

        # tv = tf.trainable_variables()
        # reg_loss = 0.001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv ])
        model_weighted_mse = tf.reduce_mean(self.lw * self.model_mse)
        model_weighted_mer = tf.reduce_mean(self.lw * self.model_mer)  # 加权平均误差率

        # self.loss = model_weighted_mse
        self.loss = model_weighted_mer

        self.train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)

        # optimizer = tf.train.AdamOptimizer(learning_rate)
        # gvs = optimizer.compute_gradients(self.loss) # 计算梯度
        # clipped_gvs = [(tf.clip_by_value(grad, -5, 5), var) for grad, var in gvs] # 梯度裁剪
        # train_step = optimizer.apply_gradients(clipped_gvs) # 梯度下降