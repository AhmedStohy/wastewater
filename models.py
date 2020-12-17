import tensorflow as tf

from tensorflow.contrib import rnn, layers
from tensorflow.contrib.layers import conv1d, conv2d
from tensorflow.contrib.layers.python.layers import batch_norm


class MLP(object):

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


class LSTM(object):

    def __init__(self, arg):

        # self.x = tf.placeholder(dtype=tf.float32, shape=[None, arg.seq_num, arg.feat_num])
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, None, arg.feat_num]) ### 注
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, arg.label_num])
        self.lw = tf.placeholder(dtype=tf.float32, shape=[arg.label_num])
        self.seq_lens = tf.placeholder(dtype=tf.int32, shape=[None])

        self.stacked_lstm = []

        for layer_hid_dim in arg.hid_dims:
            # dropout
            if arg.istrain:
                self.lstm_fw = rnn.DropoutWrapper(rnn.LSTMCell(layer_hid_dim, activation=tf.nn.softsign),
                                             input_keep_prob=arg.keep_prop)
                # lstm_bw = rnn.DropoutWrapper(rnn.LSTMCell(layer_hid_dim, activation=tf.nn.softsign), input_keep_prob=keep_prop)
            else:
                self.lstm_fw = rnn.LSTMCell(layer_hid_dim, activation=tf.nn.softsign)
                # lstm_bw = rnn.LSTMCell(layer_hid_dim, activation=tf.nn.softsign)

            # lstm_fw = rnn.LSTMCell(layer_hid_dim, activation=tf.nn.softsign)
            # lstm_bw = rnn.LSTMCell(layer_hid_dim, activation=tf.nn.softsign)

            self.stacked_lstm.append(self.lstm_fw)

            self.multilayer_lstm = rnn.MultiRNNCell(self.stacked_lstm)

        # outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw, x, dtype=tf.float32)
        # outputs = tf.concat(outputs, 2)
        if not arg.isVariableLen:
            _, self.output_states = tf.nn.dynamic_rnn(self.multilayer_lstm, self.x, dtype=tf.float32)
        else:
            _, self.output_states = tf.nn.dynamic_rnn(self.multilayer_lstm, self.x, dtype=tf.float32, sequence_length=self.seq_lens)

        # outputs = tf.transpose(outputs, [1, 0, 2])
        # preds = outputs[-1]
        self.preds = self.output_states[0][1]
        for l in arg.out_hid_dims:
            self.preds = layers.fully_connected(self.preds, l, activation_fn=tf.nn.sigmoid)
        self.preds = layers.fully_connected(self.preds, arg.label_num, activation_fn=None)

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

        self.train_step = tf.train.AdamOptimizer(arg.learning_rate).minimize(self.loss)


class CNN(object):

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
        for l in args.out_hid_dims:
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