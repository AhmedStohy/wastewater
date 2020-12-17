import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import time

from tensorflow.contrib import layers
from tensorflow.contrib.layers import conv1d, conv2d
from tensorflow.contrib.layers.python.layers import batch_norm
from data_utils import dataset_split, batch_generator

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

raw_data = 'data/20190516data.txt'

label_num = 4

hidden_layer_num = 5
learning_rate = 0.0003
batch_size = 128
istrain = True


all_data = np.genfromtxt(raw_data, dtype=np.float32, delimiter='\t', skip_header=1)
all_label = all_data[:, -label_num:]

x = tf.placeholder(dtype=tf.float32, shape=[None, label_num - 1])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

h = x
for i in range(hidden_layer_num):
    h = layers.fully_connected(h, label_num - 1, activation_fn=tf.nn.tanh)

preds = layers.fully_connected(h, 1, activation_fn=None)

model_mse = tf.reduce_mean(tf.square(preds - y)) # 平均平方误差
model_mer = tf.reduce_mean(tf.abs((preds - y) / y)) # 平均误差率
# model_mer = tf.reduce_mean(tf.abs((preds - y) / y)) # 平均误差率
# loss = tf.reduce_mean(tf.square((output - y) / y), axis=0)
loss = model_mer

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

save_dir = 'l_rel/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
display_step = 5
early_stop = 25
max_epoch = 2000

if istrain:

    label_loss = []
    fold_id = 0
    for i in range(label_num):
        label = all_label[:, i][:, np.newaxis]
        data = np.delete(all_label, i, axis=-1)

        train_d, train_l, val_d, val_l, test_d, test_l = dataset_split(data, label)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)

            ### train
            min_val_loss = float('inf')
            best_epoch = -1
            es_count = 0  # 早停计数
            for epo in range(max_epoch):

                batch_num = 0
                for batch_xs, batch_ys in batch_generator(train_d, train_l, batch_size):
                    _, l = sess.run([train_step, loss], feed_dict={x: batch_xs, y: batch_ys})

                    batch_num += 1
                    if batch_num % display_step == 0:
                        print('Epoch:{} step:{} loss={:.6f}'.format(epo, batch_num, l))

                val_loss = sess.run(loss, feed_dict={x: val_d, y: val_l})
                print('Epoch:{} val_loss={:.6f}'.format(epo, val_loss))
                if min_val_loss > val_loss:
                    min_val_loss = val_loss
                    best_epoch = epo
                    saver.save(sess, save_path=save_dir + 'epoch{}'.format(epo) + '.ckpt')
                    print('save model at Epoch:{}\n'.format(epo))
                    es_count = 0
                else:
                    es_count += 1

                if es_count == early_stop:
                    print(
                        'val_loss have no improvement for {} epochs, early stop. best_epoch:{} val_loss:{:.6f}'
                        .format(early_stop, best_epoch, min_val_loss))
                    break

            ### test and save features
            ckpt = tf.train.latest_checkpoint(save_dir)
            assert ckpt
            saver.restore(sess, ckpt)
            test_loss = sess.run(loss, feed_dict={x: test_d, y: test_l})
            label_loss.append(test_loss)
            print('test_loss={:.6f}'.format(test_loss))
    print(label_loss)