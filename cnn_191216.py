import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import time

from tensorflow.contrib import layers
from tensorflow.contrib.layers import conv1d, conv2d
from tensorflow.contrib.layers.python.layers import batch_norm
from data_utils import process_raw_data, tenfold_generator, batch_generator

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

raw_data = 'data/20191219data.txt' # ！！注意
proc_data = raw_data[:raw_data.rfind('.')] + '_ok.txt'

seq_num = int(60 / 5 * 24)
act_seq_num = int(seq_num / 1) # 注意
feat_num = 13 # 注意
total_feat_num = feat_num * act_seq_num # 注意
label_num = 3

feature_map_num = [128, 128, 128]
kernel_size = [3]
out_hid_dims = [128]
keep_prop = 0.5
learning_rate = 0.0003
max_epoch = 2000
batch_size = 128
istrain = False
# loss_weight = np.array([0.04, 0.04, 1.6, 2.32])
loss_weight = np.array([0.04, 1.6, 2.32])

data = []
label = []

if not os.path.exists(proc_data):
    data, label = process_raw_data(raw_data, proc_data, seq_num, label_num)
else:
    with open(proc_data) as fr:
        for line in fr.readlines():
            line = line.strip()
            # splt = line.split('\t')
            splt = line.split(' ') # 注意
            data.append(splt[:total_feat_num])
            label.append(splt[-label_num:])

data = np.array(data, dtype=np.float)
label = np.array(label, dtype=np.float)

## 乱序
if istrain:
    m, _ = data.shape
    inds = np.arange(m)
    np.random.shuffle(inds)
    data = data[inds]
    label = label[inds]
    all = np.c_[data, label]
    np.savetxt(proc_data, all, fmt='%f')

x = tf.placeholder(dtype=tf.float32, shape=[None, act_seq_num, feat_num])
y = tf.placeholder(dtype=tf.float32, shape=[None, label_num])

assert kernel_size.__len__() >= 1

out_lst = []
for kd in kernel_size:
    h_conv = x
    h_pool = None
    # h_conv = None
    # h_pool = x
    for ld in feature_map_num[:-1]:
        h_conv = conv1d(h_conv, ld, kernel_size=kd, stride=kd, activation_fn=tf.nn.relu, padding='valid') # 注意stride
        # h_conv = conv1d(h_pool, ld, kernel_size=kd, stride=kd, activation_fn=tf.nn.relu, padding='valid') # 注意stride
        h_conv = batch_norm(h_conv, decay=0.9, updates_collections=None, is_training=istrain)
        h_pool = tf.nn.pool(h_conv, [3], 'MAX', padding='VALID')
    h_conv = conv1d(h_pool, feature_map_num[-1], kernel_size=h_pool.shape.as_list()[1], activation_fn=tf.nn.relu, padding='valid')
    h_conv = tf.squeeze(h_conv, [1])
    out_lst.append(h_conv)

out_conv = tf.concat(out_lst, axis=-1)

preds = out_conv
for l in out_hid_dims:
    preds = layers.fully_connected(preds, l, activation_fn=tf.nn.tanh)
preds = layers.fully_connected(preds, label_num, activation_fn=None)

tv = tf.trainable_variables()
# reg_loss = 0.001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv ])
model_mse = tf.reduce_mean(tf.square(preds - y)) # 平均平方误差
model_mer = tf.reduce_mean(loss_weight * tf.abs((preds - y) / y)) # 平均误差率
# model_mer = tf.reduce_mean(tf.abs((preds - y) / y)) # 平均误差率
# loss = tf.reduce_mean(tf.square((output - y) / y), axis=0)
loss = model_mer

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

savedir = 'cnn/'
display_step = 5
early_stop = 25

if istrain:

    tenfold_mse = []
    tenfold_mer = []
    tenfold_time = []
    fold_id = 0
    for traindata, trainlabel, testdata, testlabel in tenfold_generator(data, label):

        testdata = np.reshape(testdata, [-1, seq_num, feat_num])
        testdata = testdata[:, :act_seq_num, :]

        t1 = time.time()
        saver = tf.train.Saver(max_to_keep=1)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            min_test_mer = float('inf')
            min_test_mse = float('inf')
            es_count = 0
            for epo in range(max_epoch):

                batch_num = 0
                for batch_xs, batch_ys in batch_generator(traindata, trainlabel, batch_size):
                    batch_xs = np.reshape(batch_xs, [-1, seq_num, feat_num])
                    batch_xs = batch_xs[:, :act_seq_num, :]
                    _, mse, mer = sess.run([train_step, model_mse, model_mer], feed_dict={x: batch_xs, y: batch_ys})
                    # _, output, mse = sess.run([train_step, output, loss], feed_dict={x:batch_xs, y:batch_ys})
                    batch_num += 1
                    if batch_num % display_step == 0:
                        print('fold:{} epoch:{} step:{} mse:{} mer:{}'.format(fold_id, epo, batch_num, mse, mer))
                        # print('epoch:{} step:{} pred:{} mse:{}'.format(i + 1, batch_count, output, mse))
                # tenfold_mse += mse * np.shape(batch_xs)[0]


                test_mse, test_mer = sess.run([model_mse, model_mer], feed_dict={x: testdata, y: testlabel})
                print('fold:{} epoch:{} test mse:{} test mer:{}\n'.format(fold_id, epo, test_mse, test_mer))
                if test_mer < min_test_mer:
                # if test_mse < min_test_mse:
                    saver.save(sess, save_path=savedir + 'fold' + str(fold_id) + '/' + 'min.ckpt-' + str(epo))
                    min_test_mer = test_mer
                    min_test_mse = test_mse
                    print('save model at Epoch:{}\n'.format(epo))
                    es_count = 0
                else:
                    es_count += 1

                if es_count == early_stop:
                    print('test_loss have no improvement for {} epochs, early stop.\n\n'.format(early_stop))
                    break

            tenfold_mer.append(min_test_mer)
            tenfold_mse.append(min_test_mse)

        fold_id += 1
        t2 = time.time()
        tenfold_time.append(t2 - t1)
        print('fold time consume:{}'.format(t2 - t1))

    # tenfold_mse = tenfold_mse / (10 * np.shape(traindata)[0])
    # print('fold:{} mse:{}'.format(fold_id, tenfold_mse))
    print(list(zip(tenfold_mse, tenfold_mer)))
    print('tenfold_mse:{} tenfold_mer:{}'.format(np.mean(tenfold_mse), np.mean(tenfold_mer)))
    print('mean_time_cost:{}'.format(np.array(tenfold_time).mean()))
    with open('cnn_res.txt', 'a') as fa:
        print(list(zip(tenfold_mse, tenfold_mer)), file=fa)
        print('tenfold_mse:{} tenfold_mer:{}'.format(np.mean(tenfold_mse), np.mean(tenfold_mer)), file=fa)
        print('mean_time_cost:{}'.format(np.array(tenfold_time).mean()), file=fa)

else:
    tenfold_ratio = []
    tenfold_predtc = []
    last_table = None  # 画预测值实际值对比表

    for fold_id in range(10):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.latest_checkpoint(savedir + 'fold{}/'.format(fold_id)) # 注意
            if ckpt:
                saver = tf.train.Saver()
                saver.restore(sess, ckpt)
                _, _, testdata, testlabel = list(tenfold_generator(data, label))[fold_id] # 注意
                testdata = np.reshape(testdata, [-1, seq_num, feat_num])
                testdata = testdata[:, :act_seq_num, :]

                t1 = time.time()
                pred = sess.run(preds, feed_dict={x: testdata})
                # pred = sess.run(preds, feed_dict={x: testdata[46:47]})
                t2 = time.time()
                pred_num = pred.shape[0]
                c = (t2-t1)/pred_num
                tenfold_predtc.append(c)
                print('prediction time cost:{}'.format(c))

                ### pred1, pred2, pred3, pred4 = list(map(list, zip(*pred)))
                ### label1, label2, label3, label4 = list(map(list, zip(*testlabel)))

                mses = [] # 计算每个输出指标的误差
                mers = [] # 计算每个输出指标的误差
                ol_rates = [] # 计算每个输出指标的误差超限比例
                ni_mers = [] # 计算误差率top5%，top10%的数据的平均误差率多少
                nifi_mers = [] # 计算误差率top5%，top10%的数据的平均误差率多少
                last_twoc = None # 画预测值实际值对比表

                for i, d in enumerate(zip(list(map(list, zip(*pred))), list(map(list, zip(*testlabel))))):
                    pred_lst, label_lst = d # 公用

                    # # 画 预测-标签 图
                    # plt.figure(figsize=(8,4))
                    # plt.plot(range(1, pred_lst.__len__() + 1), pred_lst, linewidth=1, linestyle='-', label='predict')
                    # plt.plot(range(1, pred_lst.__len__() + 1), label_lst, linewidth=1, linestyle='-', label='label')
                    # plt.xlabel('sample number')
                    # plt.ylabel('output value')
                    # # plt.legend(loc='lower right')
                    # plt.legend()
                    # plt.savefig('out_ind{}.png'.format(i+1))
                    # plt.show()

                    ## 公用
                    pred_lst = np.array(pred_lst)
                    label_lst = np.array(label_lst)
                    mer_seq = np.abs((pred_lst - label_lst) / label_lst)
                    mer_lst = np.sort(mer_seq)
                    num_test = mer_lst.shape[0]
                    x_lst = np.arange(1, num_test + 1) / num_test

                    ## 画预测值实际值对比表
                    # tpreds = np.expand_dims(pred_lst, 1) # 注意这里不能与前面变量重名
                    # tlabels = np.expand_dims(label_lst, 1)
                    # twoc = np.c_[tpreds, tlabels]
                    # if last_twoc is not None:
                    #     last_twoc = np.c_[last_twoc, twoc]
                    # else:
                    #     last_twoc = twoc
                    # if i == label_num - 1:
                    #     if last_table is not None:
                    #         last_table = np.r_[last_table, last_twoc]
                    #     else:
                    #         last_table = last_twoc
                    #     last_twoc = None

                    # ## 画 平均误差率统计图
                    # plt.plot(x_lst, mer_lst, linewidth=1, linestyle='-')
                    # ei = np.where(x_lst > 0.8)[0][0]
                    # ni = np.where(x_lst > 0.9)[0][0]
                    # plt.plot(x_lst[ei], mer_lst[ei], color='r', markerfacecolor='red', marker='o')
                    # plt.plot(x_lst[ni], mer_lst[ni], color='r', markerfacecolor='red', marker='o')
                    # plt.text(x_lst[ei], mer_lst[ei], '{:.5f}'.format(mer_lst[ei]), ha='center', va='top', fontsize=8)
                    # plt.text(x_lst[ni], mer_lst[ni], '{:.5f}'.format(mer_lst[ni]), ha='center', va='bottom', fontsize=8)
                    # plt.xlabel('rate')
                    # plt.ylabel('error rate')
                    # # plt.legend()
                    # plt.savefig('er_ind{}.png'.format(i+1))
                    # plt.show()

                    # ## 计算误差率top5%，top10%的数据的平均误差率多少
                    # ni = np.where(x_lst > 0.9)[0][0]
                    # nifi = np.where(x_lst > 0.95)[0][0]
                    # ni_mer = np.mean(mer_lst[ni:])
                    # nifi_mer = np.mean(mer_lst[nifi:])
                    # ni_mers.append(ni_mer)
                    # nifi_mers.append(nifi_mer)

                    ## 计算误差大于0.2的比例，必须放在最后
                    mer_lst[mer_lst < 0.2] = 0
                    overlimit = mer_lst.nonzero()[0].size
                    ol_rate = overlimit / num_test

                    # mses.append(np.mean(np.square(pred_lst - label_lst)))  # 计算每个输出指标的误差
                    # mers.append(np.mean(mer_seq))  # 计算每个输出指标的误差
                    ol_rates.append(ol_rate) # 计算每个输出指标的误差超限比例


                # print(mses) # 计算每个输出指标的误差
                print(mers) # 计算每个输出指标的误差
                print(ol_rates) # 计算每个输出指标的误差超限比例
                tenfold_ratio.append(ol_rates)
                #
                # print(ni_mers) # 计算误差率top5%，top10%的数据的平均误差率多少
                # print(nifi_mers) # 计算误差率top5%，top10%的数据的平均误差率多少

                # print(pred)
                # print(testlabel[46:47])

    # print(np.array(tenfold_predtc).mean())
    print(np.max(np.array(tenfold_ratio), axis=0))
    # np.savetxt('pred_true.csv', last_table, delimiter=',')