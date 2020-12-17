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
        savedir='cnn/',  # 一定要这个格式，前面字母，最后'/'


        time_increment = 40, # 注意输入输出时间间隔，单位为分
        time_intv_per_sample = 1, # x min一条数据
        last_equal_intv = 40, # 最后 x min的输入视为一样，0表示无
        feat_num = 8, # 特征个数，注意
        label_num = 3, # 输出个数

        feature_map_num = [128, 128, 128, 128], # 最后一个为全幅卷积特征映射数
        # feature_map_num = [128, 128, 128],
        kernel_size = [3],
        # kernel_size = [4], # 一厂入水指标预测
        # out_hid_dims = [256],
        out_hid_dims = [128], # 预测MLP各隐层神经元个数，[128,128]表示两个维度128的隐层
        learning_rate = 0.0003,
        max_epoch = 2000,
        batch_size = 64,
        istrain = True,
        first_random = False, # processed_data第一次做实验需要乱序，第二次开始False
        use_loss_weight = True, # 是否启用动态loss weight
        display_step = 5,
        early_stop = 25,
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


data = None
label = None

data_label = None
if not os.path.exists(args.processed_data):
    data, label = process_raw_data(args.raw_data, args.processed_data, args.seq_num, args.label_num, args.time_increment)
else:
    data_label = np.genfromtxt(args.processed_data, np.float32, delimiter='\t')
    # data, label = data_label[:, :total_feat_num], data_label[:, -label_num:]
    data, label = data_label[:, :args.total_feat_num], data_label[:, args.total_feat_num:args.total_feat_num+args.label_num]

data = np.array(data, dtype=np.float32)
label = np.array(label, dtype=np.float32)


### 入水时刻 x min后的入水都视为一样
if args.last_equal_num:
    last_equal_feats_num = args.last_equal_num * args.feat_num
    data[:, -last_equal_feats_num:] = np.tile(data[:, -last_equal_feats_num:
                            -last_equal_feats_num+args.feat_num], args.last_equal_num)


### 乱序
if args.first_random and args.istrain:
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


# ## 应用场景分析，最后n个输入设为一样，不需重新训练，改测试输入
# if not istrain and last_equal_num:
#     last_equal_feats_num = last_equal_num * feat_num
#     testdata[:, -last_equal_feats_num:] = np.tile(testdata[:, -last_equal_feats_num:
#                                 -last_equal_feats_num+feat_num], last_equal_num)


model = models.CNN(args)


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

    valdata = np.reshape(valdata, [-1, args.seq_num, args.feat_num])
    valdata = valdata[:, :args.act_seq_num, :]

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
        es_count = 0
        loss_weights = []
        for epo in range(args.max_epoch):

            batch_num = 0
            train_mse, train_mer, train_ratio = np.zeros(args.label_num), np.zeros(args.label_num), np.zeros(args.label_num)
            for batch_xs, batch_ys in batch_generator(traindata, trainlabel, args.batch_size):
                batch_xs = np.reshape(batch_xs, [-1, args.seq_num, args.feat_num])
                batch_xs = batch_xs[:, :args.act_seq_num, :]

                ### mse, mer: np.array 2D
                _, mse, mer = sess.run([model.train_step, model.model_mse, model.model_mer], feed_dict={model.x: batch_xs,
                                                                        model.y: batch_ys, model.lw: args.loss_weight})
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
            print('fold:{} epoch:{} val mse:{} val mer:{} val ratio:{}\n'.format(fold_id, epo, val_mse.mean(),
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
                print('save model at Epoch:{}\n'.format(epo))
                es_count = 0
            else:
                es_count += 1

            if es_count == args.early_stop:
                print('val_loss have no improvement for {} epochs, early stop.\n'.format(args.early_stop))
                break

            loss_weights.append(args.loss_weight)
            print(args.loss_weight)

    tenfold_train_mse.append(min_train_mse)
    tenfold_train_mer.append(min_train_mer)
    tenfold_train_ratio.append(min_train_ratio)
    tenfold_val_mse.append(min_val_mse)
    tenfold_val_mer.append(min_val_mer)
    tenfold_val_ratio.append(min_val_ratio)

    # loss_weights = np.array(loss_weights)
    # for ii in range(label_num):
    #     plt.plot(range(1, loss_weights.shape[0] + 1), loss_weights[:, ii], linewidth=1, linestyle='-', label='{}'.format(ii))
    # plt.xlabel('Epoch Number')
    # plt.ylabel('Loss Weight Values')
    # plt.legend()
    # plt.savefig('cnn_loss_weights.png')
    # plt.show()

    fold_id += 1
    t2 = time.time()
    tenfold_time.append(t2 - t1)
    print('fold time consume:{}\n'.format(t2 - t1))


    print('tenfold_train_mse:{} tenfold_train_mer:{} tenfold_train_ratio:{}'.format(np.mean(tenfold_train_mse, axis=0),
                                            np.mean(tenfold_train_mer, axis=0), np.mean(tenfold_train_ratio, axis=0)))
    print('tenfold_val_mse:{} tenfold_val_mer:{} tenfold_val_ratio:{}'.format(np.mean(tenfold_val_mse, axis=0),
                                        np.mean(tenfold_val_mer, axis=0), np.mean(tenfold_val_ratio,axis=0)))
    print('mean_time_cost:{}'.format(np.array(tenfold_time).mean()))
    # print(loss_weight)
    if not os.path.exists('res'):
        os.mkdir('res')
    with open('res/cnn_res.txt', 'a') as fa:
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), file=fa)
        print('tenfold_train_mse:{} tenfold_train_mer:{} tenfold_train_ratio:{}'.format(np.mean(tenfold_train_mse, axis=0),
                                    np.mean(tenfold_train_mer, axis=0), np.mean(tenfold_train_ratio, axis=0)), file=fa)
        print('tenfold_val_mse:{} tenfold_val_mer:{} tenfold_val_ratio:{}'.format(np.mean(tenfold_val_mse, axis=0),
                                np.mean(tenfold_val_mer, axis=0), np.mean(tenfold_val_ratio, axis=0)), file=fa)
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
            ckpt = tf.train.latest_checkpoint(args.savedir + 'fold{}/'.format(fold_id)) # 注意
            if ckpt:
                saver = tf.train.Saver()
                saver.restore(sess, ckpt)
                # _, _, testdata, testlabel = list(tenfold_generator(data, label))[fold_id] # 注意
                testdata = np.reshape(testdata, [-1, args.seq_num, args.feat_num])
                testdata = testdata[:, :args.act_seq_num, :]

                t1 = time.time()
                pred = sess.run(model.preds, feed_dict={model.x: testdata})
                # pred = sess.run(model.preds, feed_dict={model.x: testdata[46:47]})
                t2 = time.time()
                pred_num = pred.shape[0]
                c = (t2-t1)/pred_num
                tenfold_predtc.append(c)
                print('prediction time cost:{}'.format(c))

                ### 预测值 真实值存入文件
                if fold_id == target_fold_id:

                    # ### 训练集，验证集预测也存入文件
                    # traindata, trainlabel, valdata, vallabel = list(tenfold_generator(data, label))[fold_id]
                    # traindata = np.reshape(traindata, [-1, seq_num, feat_num])
                    # traindata = traindata[:, :act_seq_num, :]
                    # valdata = np.reshape(valdata, [-1, seq_num, feat_num])
                    # valdata = valdata[:, :act_seq_num, :]
                    # train_preds = sess.run(model.preds, feed_dict={model.x: traindata})
                    # val_preds = sess.run(model.preds, feed_dict={model.x: valdata})
                    # for stage in ['train', 'val']:
                    #     file_name = '{}_{}_preds.csv'.format(savedir[:-1], stage)
                    #     with open(file_name, 'w') as fw:  ### savedir 前面字母最后'/'
                    #         csv_write = csv.writer(fw)
                    #         if stage == 'train':
                    #             csv_write.writerows(np.c_[train_preds, trainlabel])
                    #         else:
                    #             csv_write.writerows(np.c_[val_preds, vallabel])
                    # train_mer = np.abs(train_preds - trainlabel) / trainlabel
                    # print('train_mse:', np.mean(np.square(train_preds - trainlabel), axis=0))
                    # print('train_mer:', np.mean(train_mer, axis=0))
                    # train_mer[train_mer <= 0.1] = 0
                    # train_mer[train_mer > 0.1] = 1
                    # print('train_ratio', np.sum(train_mer, axis=0))
                    # val_mer = np.abs(val_preds - vallabel) / vallabel
                    # print('val_mse:', np.mean(np.square(val_preds - vallabel), axis=0))
                    # print('val_mer:', np.mean(val_mer, axis=0))
                    # val_mer[val_mer <= 0.1] = 0
                    # val_mer[val_mer > 0.1] = 1
                    # print('val_ratio:', np.sum(val_mer, axis=0))
                    # print('train_size:', traindata.shape[0])
                    # print('val_size:', valdata.shape[0])

                    file_name = '{}_test_preds.csv'.format(args.savedir[:-1])
                    if args.last_equal_intv:
                        file_name = '{}_test_preds_{}min.csv'.format(args.savedir[:-1], args.last_equal_intv)
                    with open(file_name, 'w') as fw:  ### savedir 前面字母最后'/'
                        csv_write = csv.writer(fw)
                        csv_write.writerows(np.c_[pred, testlabel])

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

                    ### 画 预测-标签 图
                    if fold_id == target_fold_id:
                        plt.figure(figsize=(8,4))
                        plt.plot(range(1, pred_lst.__len__() + 1), pred_lst, linewidth=1, linestyle='-', label='predict')
                        plt.plot(range(1, pred_lst.__len__() + 1), label_lst, linewidth=1, linestyle='-', label='label')
                        plt.xlabel('sample number')
                        plt.ylabel('output value')
                        # plt.legend(loc='lower right')
                        plt.legend()
                        # plt.savefig('out_ind{}.png'.format(i+1))
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
                    mer_lst[mer_lst <= 0.1] = 0
                    overlimit = mer_lst.nonzero()[0].size
                    ol_rate = overlimit / num_test ### 注
                    # ol_rate = overlimit ### 注
                    # print('test_size:', num_test)

                    mses.append(np.mean(mse_seq))  # 计算每个输出指标的误差
                    mers.append(np.mean(mer_seq))  # 计算每个输出指标的误差
                    ol_rates.append(ol_rate) # 计算每个输出指标的误差超限比例


                print(mses) # 计算每个输出指标的误差
                print(mers) # 计算每个输出指标的误差
                print(ol_rates) # 计算每个输出指标的误差超限比例
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