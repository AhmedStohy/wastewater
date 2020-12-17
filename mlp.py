import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import csv

import models

from data_utils import process_raw_data, tenfold_generator, batch_generator

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def create_hparams():
    params = tf.contrib.training.HParams(
        raw_data = 'data/20200406data_0.txt', # 原始数据文件，！！注意
        test_rate = 0.2, # 测试数据占总数据比例
        savedir = 'mlp/',  # 一定要这个格式，前面字母，最后'/'

        time_increment = 24, # 注意输入输出时间间隔，单位为分
        time_intv_per_sample = 1, # x min一条数据
        last_equal_intv = 24, # 最后 x min的输入视为一样，0表示无
        feat_num = 8, # 特征个数，注意
        label_num = 3, # 输出个数

        hid_dims=[128], # MLP各隐层维度
        keep_prop = 0.5,
        learning_rate = 0.0003,
        max_epoch = 1000,
        batch_size = 64,
        istrain = True,
        first_random = False,  # processed_data第一次做实验需要乱序，第二次开始False
        use_loss_weight = True,  # 是否启用动态loss weight
        display_step = 5,
        early_stop = 100,
    )
    return params

args = create_hparams()
args.add_hparam('processed_data', args.raw_data[:args.raw_data.rfind('.')] + '_ok.txt') # 处理后文件名
args.add_hparam('last_equal_num', args.last_equal_intv // args.time_intv_per_sample) # 最后 x 个输入一样
args.add_hparam('seq_num', args.time_increment // args.time_intv_per_sample) # 序列长度
args.add_hparam('act_seq_num', args.seq_num // 1)
args.add_hparam('total_feat_num', args.feat_num * args.act_seq_num) # 注意
args.add_hparam('loss_weight', np.ones(args.label_num)) # 初始值

if not os.path.exists(args.savedir):
    os.mkdir(args.savedir)
if args.hid_dims.__len__() == 1:
    args.savedir = 'shlnn/'

data = []
label = []

if not os.path.exists(args.processed_data):
    data, label = process_raw_data(args.raw_data, args.processed_data, args.seq_num, args.label_num, args.time_increment)
else:
    data_label = np.genfromtxt(args.processed_data, np.float32, delimiter='\t')
    data, label = data_label[:, :args.total_feat_num], data_label[:, -args.label_num:]

data = np.array(data, dtype=np.float)
label = np.array(label, dtype=np.float)

## 乱序
if args.istrain and args.first_random:
    m, _ = data.shape
    inds = np.arange(m)
    np.random.shuffle(inds)
    data = data[inds]
    label = label[inds]
    np.savetxt(args.processed_data, np.c_[data, label], '%s', '\t')

## 分离训练集和测试集
test_num = int(args.test_rate * data.__len__())
testdata = data[-test_num:]
testlabel = label[-test_num:]
data = data[:-test_num]
label = label[:-test_num]

model = models.MLP(args)

if args.istrain:

    tenfold_train_mse = []
    tenfold_train_mer = []
    tenfold_train_ratio = []
    tenfold_val_mse = []
    tenfold_val_mer = []
    tenfold_val_ratio = []
    tenfold_time = []
    fold_id = 0

    # for traindata, trainlabel, valdata, vallabel in tenfold_generator(data, label):
    traindata, trainlabel, valdata, vallabel = list(tenfold_generator(data, label))[0]

    train_size = traindata.shape[0]
    val_size = valdata.shape[0]

    t1 = time.time()
    saver = tf.train.Saver(max_to_keep=1)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        min_train_mer = np.ones(args.label_num) * float('inf')
        min_train_mse = np.ones(args.label_num) * float('inf')
        min_train_ratio = np.ones(args.label_num) * float('inf')
        min_val_mer = np.ones(args.label_num) * float('inf')
        min_val_mse = np.ones(args.label_num) * float('inf')
        min_val_ratio = np.ones(args.label_num) * float('inf')
        loss_weights = []
        es_count = 0
        for epo in range(args.max_epoch):

            batch_num = 0
            train_mse, train_mer, train_ratio = np.zeros(args.label_num), np.zeros(args.label_num), np.zeros(args.label_num)
            for batch_xs, batch_ys in batch_generator(traindata, trainlabel, args.batch_size):
                _, mse, mer = sess.run([model.train_step, model.model_mse, model.model_mer],
                                       feed_dict={model.x: batch_xs, model.y: batch_ys, model.lw: args.loss_weight})
                # _, output, mse = sess.run([train_step, output, loss], feed_dict={x:batch_xs, y:batch_ys})

                train_mse += mse.sum(axis=0) / train_size
                train_mer += mer.sum(axis=0) / train_size
                tmp_ratio = np.array(mer)
                tmp_ratio[tmp_ratio <= 0.1] = 0
                tmp_ratio[tmp_ratio > 0.1] = 1
                train_ratio += tmp_ratio.sum(axis=0) / train_size

                batch_num += 1
                if batch_num % args.display_step == 0:
                    print('fold:{} epoch:{} step:{} mse:{} mer:{} ratio:{}'.format(fold_id, epo, batch_num,
                                        mse.mean(), mer.mean(), tmp_ratio.sum(axis=0) / batch_xs.shape[0]))
            # tenfold_mse += mse * np.shape(batch_xs)[0]

            ### val_mse, val_mer: np.array 2D
            val_mse, val_mer = sess.run([model.model_mse, model.model_mer], feed_dict={model.x: valdata, model.y: vallabel})
            val_ratio = np.array(val_mer)
            val_ratio[val_ratio <= 0.1] = 0
            val_ratio[val_ratio > 0.1] = 1
            val_ratio = val_ratio.sum(axis=0) / val_size

            val_mer = val_mer.mean(axis=0)
            if args.use_loss_weight:
                args.loss_weight = val_mer / val_mer.sum()

            val_mse = val_mse.mean(axis=0)
            print(
                'fold:{} epoch:{} val mse:{} val mer:{} val ratio:{}\n'.format(fold_id, epo, val_mse.mean(),
                                                                               val_mer.mean(), val_ratio))
            if val_mer.mean() < min_val_mer.mean():
            # if val_mse.mean() < min_val_mse.mean():
                saver.save(sess, save_path=args.savedir + 'fold' + str(fold_id) + '/' + 'min.ckpt-' + str(epo))
                min_train_mse = train_mse
                min_train_mer = train_mer
                min_train_ratio = train_ratio
                min_val_mse = val_mse
                min_val_mer = val_mer
                min_val_ratio = val_ratio
                es_count = 0
            else:
                es_count += 1

            if es_count == args.early_stop:
                print('val_loss have no improvement for {} epochs, early stop.'.format(args.early_stop))
                break

            print(args.loss_weight)
            loss_weights.append(args.loss_weight)

        tenfold_train_mse.append(min_train_mse)
        tenfold_train_mer.append(min_train_mer)
        tenfold_train_ratio.append(min_train_ratio)
        tenfold_val_mse.append(min_val_mse)
        tenfold_val_mer.append(min_val_mer)
        tenfold_val_ratio.append(min_val_ratio)

        # loss_weights = np.array(loss_weights)
        # for ii in range(label_num):
        #     plt.plot(range(1, loss_weights.shape[0] + 1), loss_weights[:, ii], linewidth=1, linestyle='-',
        #              label='{}'.format(ii))
        # plt.xlabel('Epoch Number')
        # plt.ylabel('Loss Weight Values')
        # plt.legend()
        # plt.savefig('mlp_loss_weights.png')
        # plt.show()

    fold_id += 1
    t2 = time.time()
    tenfold_time.append(t2 - t1)
    print('fold time consume:{}'.format(t2 - t1))

    print('tenfold_train_mse:{} tenfold_train_mer:{} tenfold_train_ratio:{}'.format(
        np.mean(tenfold_train_mse, axis=0),
        np.mean(tenfold_train_mer, axis=0), np.mean(tenfold_train_ratio, axis=0)))
    print('tenfold_val_mse:{} tenfold_val_mer:{} tenfold_val_ratio:{}'.format(np.mean(tenfold_val_mse, axis=0),
                                                                              np.mean(tenfold_val_mer, axis=0),
                                                                              np.mean(tenfold_val_ratio, axis=0)))
    print('mean_time_cost:{}'.format(np.array(tenfold_time).mean()))
    # print(loss_weight)
    if not os.path.exists('res'):
        os.mkdir('res')
    with open('res/mlp_res.txt', 'a') as fa:
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), file=fa)
        print('tenfold_train_mse:{} tenfold_train_mer:{} tenfold_train_ratio:{}'.format(
            np.mean(tenfold_train_mse, axis=0),
            np.mean(tenfold_train_mer, axis=0), np.mean(tenfold_train_ratio, axis=0)), file=fa)
        print('tenfold_val_mse:{} tenfold_val_mer:{} tenfold_val_ratio:{}'.format(np.mean(tenfold_val_mse, axis=0),
                                                                                      np.mean(tenfold_val_mer, axis=0),
                                                                                      np.mean(tenfold_val_ratio,
                                                                                              axis=0)), file=fa)
        print('mean_time_cost:{}'.format(np.array(tenfold_time).mean()), file=fa)


else:
    tenfold_ratio = []
    tenfold_predtc = []
    tenfold_mse = []
    tenfold_mer = []
    last_table = None  # 画预测值实际值对比表

    target_fold_id = 0

    for fold_id in range(10):
        if fold_id != target_fold_id:
            continue

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.latest_checkpoint(args.savedir + 'fold{}/'.format(fold_id))  # 注意
            if ckpt:
                saver = tf.train.Saver()
                saver.restore(sess, ckpt)
                # _, _, testdata, testlabel = list(tenfold_generator(data, label))[fold_id] # 注意
                testdata = testdata[:, :args.total_feat_num]

                t1 = time.time()
                pred = sess.run(model.output, feed_dict={model.x: testdata})
                # pred = sess.run(preds, feed_dict={x: testdata[46:47]})
                t2 = time.time()
                pred_num = pred.shape[0]
                c = (t2 - t1) / pred_num
                tenfold_predtc.append(c)
                print('prediction time cost:{}'.format(c))

                ### 预测值 真实值存入文件
                with open('{}_test_preds.csv'.format(args.savedir[:-1]), 'w') as fw: ### savedir 前面字母最后'/'
                    csv_write = csv.writer(fw)
                    csv_write.writerows(np.c_[pred, testlabel])

                ### pred1, pred2, pred3, pred4 = list(map(list, zip(*pred)))
                ### label1, label2, label3, label4 = list(map(list, zip(*testlabel)))

                mses = []  # 计算每个输出指标的误差
                mers = []  # 计算每个输出指标的误差
                ol_rates = []  # 计算每个输出指标的误差超限比例
                ni_mers = []  # 计算误差率top5%，top10%的数据的平均误差率多少
                nifi_mers = []  # 计算误差率top5%，top10%的数据的平均误差率多少
                last_twoc = None  # 画预测值实际值对比表

                for i, d in enumerate(zip(list(map(list, zip(*pred))), list(map(list, zip(*testlabel))))):
                    pred_lst, label_lst = d  # 公用

                    ### 画 预测-标签 图
                    if fold_id == target_fold_id:
                        plt.figure(figsize=(8,4))
                        plt.plot(range(1, pred_lst.__len__() + 1), pred_lst, linewidth=1, linestyle='-', label='predict')
                        plt.plot(range(1, pred_lst.__len__() + 1), label_lst, linewidth=1, linestyle='-', label='label')
                        plt.xlabel('sample number')
                        plt.ylabel('output value')
                        # plt.legend(loc='lower right')
                        plt.legend()
                        plt.savefig('out_ind{}.png'.format(i+1))
                        plt.show()

                    ## 公用
                    pred_lst = np.array(pred_lst)
                    label_lst = np.array(label_lst)
                    mer_seq = np.abs((pred_lst - label_lst) / label_lst)
                    mse_seq = np.square(pred_lst - label_lst)
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

                    ## 计算误差大于0.1的比例，必须放在最后
                    mer_lst[mer_lst < 0.1] = 0
                    overlimit = mer_lst.nonzero()[0].size
                    ol_rate = overlimit / num_test ### 注
                    # ol_rate = overlimit ### 注
                    # print('test_num:', num_test)

                    mses.append(np.mean(mse_seq))  # 计算每个输出指标的误差
                    mers.append(np.mean(mer_seq))  # 计算每个输出指标的误差
                    ol_rates.append(ol_rate)  # 计算每个输出指标的误差超限比例

                print(mses)  # 计算每个输出指标的误差
                print(mers)  # 计算每个输出指标的误差
                print(ol_rates)  # 计算每个输出指标的误差超限比例
                tenfold_ratio.append(ol_rates)
                tenfold_mer.append(mers)
                tenfold_mse.append(mses)

                # print(ni_mers) # 计算误差率top5%，top10%的数据的平均误差率多少
                # print(nifi_mers) # 计算误差率top5%，top10%的数据的平均误差率多少

                # print(pred)
                # print(testlabel[46:47])

    print()
    # print(np.array(tenfold_predtc).mean())
    print(np.mean(np.array(tenfold_mse), axis=0))
    print(np.mean(np.array(tenfold_mer), axis=0))
    print(np.mean(np.array(tenfold_ratio), axis=0))
    # np.savetxt('pred_true.csv', last_table, delimiter=',')
