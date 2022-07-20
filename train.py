import tensorflow as tf
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import time
import csv
import pandas as pd
# import seaborn as sns
import functools as ft

import models
import transformer
import optimization

from data_utils import process_raw_data, tenfold_generator, batch_generator, process_raw_data_variable_len

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def create_hparams():
    params = tf.contrib.training.HParams(
        raw_data='/kaggle/working/wastewater/data/20200406data_0_ok.txt', # 原始数据文件，！！注意
        test_rate=0.2,  # 测试数据占总数据比例
        savedir='./ckpt/trm_200406x0varilen_1enc_64bs_first_nodrop_allInputPad', # 保存实验结果
        model_select='transformer',
        number_of_runs=20, # 重复训练测试若干次
        get_test_res_by = 'max', # 取测试效果平均/最大值 ['mean', 'max']
        record_train_detail=False, # 是否将训练细节输出到文件
        first_random=False, # processed_data第一次做实验需要乱序，第二次开始False
        liuliang_ind=4,  # 在原始文件中的列号，从0开始
        is_variable_len=True, # 输入序列是否变长
        pad_test_only=False, # True：只有测试集填充；False：训练、验证、测试集都是真实/填充数据
        is_multi_var = False, # 是否是多变量进水水质预测
        plot_seq_len=False, # 输入序列变长时是否画出序列长度统计图
        plot_test_res=False, # 是否在测试时画出预测值真实值对比图
        plot_line_or_scatter='line', # ['line', 'scatter'] 测试时画出预测值真实值对比折线图或散点图

        time_increment=30 * 60, # 注意输入输出时间间隔，单位为分
        time_intv_per_sample=5, # x min一条数据
        last_equal_intv=10, # 最后 x min的输入视为一样，0表示无
        truncate_factor=1, # 定长时，最终的输入序列长度=seq_len * truncate_factor
        feat_num=8,  # 特征个数，注意
        label_num=3, # 非自回归预测时模型最终输出的维度；单变量进水水质预测时，也代表需要向后预测的时间步个数
        mer_limit=0.1, # mer > mer_limit的被认为超限
        seq_len_limit=None,

        # MLP参数
        mlp_hid_dims=[256], # MLP各隐层维度

        # LSTM参数
        rnn_layer_num=1, # LSTM层数
        rnn_head_num=1,
        rnn_hid_dim=128,
        rnn_pred_hid_dims=[128],
        rnn_keep_prop=1.0,
        rnn_pool = 'last', # last, mean, max, conv1d
        rnn_concat_c = False, # 是否将c拼接到最后的输出中

        # CNN参数
        feature_map_num=[128, 128, 128, 128], # 最后一个为全幅卷积特征映射数
        conv_kernel_size=[3], # 卷积核大小
        conv_stride=[3], # 卷积计算步长
        # kernel_size = [4], # 一厂入水指标预测
        last_max_pool = True, # 最后的全幅卷积前是否加一层max pool
        cnn_pred_hid_dims=[256],  # 预测MLP各隐层神经元个数，[128,128]表示两个维度128的隐层

        # Transformer参数
        d_model=128,
        trm_head_num=4,
        enc_block_num=1,
        trm_keep_prop=1.0,
        pos_max_length=1024,  # 输入变长序列的最大长度
        SE_select='first',  # 句子表示选择 cls, first, sum, mean, max, conv
        trm_pred_layer_num=2,

        learning_rate=0.0003,
        max_epoch=10000,
        batch_size=64,
        early_stop=100,

        auto_regressive=False,  # 是否自回归地向后预测
        use_loss_weight=True,  # 是否启用动态loss weight
        input_add_differ=False,  # 是否在输入中加入一阶差分信息
        ensemble_naive=False,
        use_last_n_step=0, # 只取最后若干时间步的特征作为输入，只当>0时有效
        loss_function='mer', # ['mer', 'mse']

        istrain=False,
        display_step=5,

        is_pso=False,
    )
    return params


args = create_hparams()
if args.is_variable_len:
    args.add_hparam('processed_data', args.raw_data[:args.raw_data.rfind('.')] + '_varilen_ok.txt')  # 处理后文件名
else:
    args.add_hparam('processed_data', args.raw_data[:args.raw_data.rfind('.')] + '_ok.txt')  # 处理后文件名
args.add_hparam('last_equal_num', args.last_equal_intv // args.time_intv_per_sample)  # 最后 x 个时间步输入一样
args.add_hparam('seq_num', args.time_increment // args.time_intv_per_sample)  # 存储时数据的输入序列长度
args.add_hparam('total_feat_num', args.feat_num * args.seq_num)  # 存储时数据的总特征数
args.add_hparam('act_seq_num', int(args.seq_num * args.truncate_factor)) # 实验时数据的实际输入序列长度
if args.auto_regressive:
    args.add_hparam('loss_weight', np.ones(args.feat_num))  # 初始值
else:
    args.add_hparam('loss_weight', np.ones(args.label_num))  # 初始值

if not os.path.exists('ckpt'):
    os.mkdir('ckpt')

if not os.path.exists('figure/'): # 存储各种效果图
    os.mkdir('figure')

if not os.path.exists(args.savedir):
    os.mkdir(args.savedir)

data = None
label = None

data_label = None
if not os.path.exists(args.processed_data):
    if not args.is_variable_len:
        data, label = process_raw_data(args.raw_data, args.processed_data, args.seq_num, args.label_num,
                                       args.time_increment)
    else:
        data, label = process_raw_data_variable_len(args.raw_data, args.processed_data, args.liuliang_ind + 1,
                                                    args.feat_num, args.label_num, args.seq_len_limit)
else:
    if not args.is_variable_len:
        data_label = np.genfromtxt(args.processed_data, np.float32, delimiter='\t')
        data, label = data_label[:, :args.total_feat_num], \
                      data_label[:, args.total_feat_num :
                                    args.total_feat_num + args.label_num] # total_feat_num不可有误
    else:
        seqlen_data_label = np.genfromtxt(args.processed_data, np.float32, delimiter='\t')
        data, label = seqlen_data_label[:, :-args.label_num], seqlen_data_label[:, -args.label_num:]

data = np.array(data, dtype=np.float32)
label = np.array(label, dtype=np.float32)


if args.is_variable_len:
    max_len = int((data.shape[1] - 1) / args.feat_num)  # -1是因为第一维是序列长度
    args.add_hparam('seq_max_len', max_len)
    if args.plot_seq_len:
        df_seq_len = pd.Series(data[:, 0])
        print(df_seq_len.describe())
        df_seq_len.hist()
        df_seq_len.plot(kind='kde', secondary_y=False)
        plt.xlabel('Input Sequence Length', fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
        plt.savefig('figure/{}_seq_len.png'.format(args.raw_data[args.raw_data.rfind('/')+1:
                                                                 args.raw_data.rfind('.')]))
        plt.show()


### 乱序
if args.istrain and args.first_random:
    m, _ = data.shape
    inds = np.arange(m)
    np.random.shuffle(inds)
    data = data[inds]
    label = label[inds]
    np.savetxt(args.processed_data, np.c_[data, label], '%s', '\t')

# ### 造完美数据集
# data = np.tile(data[:, 0:1], data.shape[1])
# label = np.tile(data[:, 0:1], label.shape[1])


## 分离训练集和测试集
test_num = int(args.test_rate * data.__len__())
testdata = data[-test_num:]
testlabel = label[-test_num:]
data = data[:-test_num]
label = label[:-test_num]


### 输入填充：入水时刻 x min后的入水指标都视为一样
if args.last_equal_intv:
    if args.is_variable_len: # 变长填充，所有时间步入水指标一样，等于第一时间步的
        if not args.pad_test_only:
            data_seq_len = data[:, 0].astype(np.int32)  ### 注
            for i in range(data.shape[0]):
                last_equal_feats_num = args.feat_num * data_seq_len[i]  ### 注
                data[i, 1:last_equal_feats_num+1] = np.tile(data[i, 1:args.feat_num+1], data_seq_len[i]) # 第一维是序列长度
        test_seq_len = testdata[:, 0].astype(np.int32)  ### data_seq_len
        for i in range(testdata.shape[0]):
            last_equal_feats_num = args.feat_num * test_seq_len[i]  ### 注
            testdata[i, 1:last_equal_feats_num+1] = np.tile(testdata[i, 1:args.feat_num + 1], test_seq_len[i])  # 第一维是序列长度
            # pad_size = (args.seq_max_len - seq_len[i]) * args.feat_num
            # if pad_size:
            #     data[i, -(last_equal_feats_num + pad_size): -pad_size] = np.tile(
            #         data[i, -(last_equal_feats_num + pad_size): -(last_equal_feats_num + pad_size) + args.feat_num],
            #         seq_len[i])
            # else:
            #     data[i, -(last_equal_feats_num + pad_size):] = np.tile(
            #         data[i, -(last_equal_feats_num + pad_size): -(last_equal_feats_num + pad_size) + args.feat_num],
            #         seq_len[i])
    else:
        last_equal_feats_num = args.last_equal_num * args.feat_num
        if not args.pad_test_only:
            data[:, -last_equal_feats_num:] = np.tile(
                data[:, -last_equal_feats_num: -last_equal_feats_num + args.feat_num], args.last_equal_num)
        testdata[:, -last_equal_feats_num:] = np.tile(
            testdata[:, -last_equal_feats_num: -last_equal_feats_num + args.feat_num], args.last_equal_num)


if not args.is_variable_len:
    testdata = np.reshape(testdata, [-1, args.seq_num, args.feat_num])
    testdata = testdata[:, :args.act_seq_num, :]
else:
    test_seq_len = np.array(testdata[:, 0], dtype=np.int32)
    testdata = np.reshape(testdata[:, 1:], [-1, args.seq_max_len, args.feat_num])


def add_variation_info(input_data):
    assert len(input_data.shape) == 3
    input_data_copy = input_data.copy()
    input_data_copy[:, 1:, :] = np.abs(input_data_copy[:, 1:, :] - input_data_copy[:, :-1, :])
    input_data_copy[:, 0, :] = np.min(input_data_copy[:, 1:, :], axis=1)
    # res = np.concatenate([input_data, input_data_copy], axis=-1)
    res = input_data + input_data_copy
    return res


if args.input_add_differ:
    testdata = add_variation_info(testdata)

if args.model_select == 'transformer':
    cls_pad = 0.1 * np.ones([testdata.shape[0], 1, testdata.shape[2]])
    testdata = np.concatenate((cls_pad, testdata), axis=1)


def tf_predict(sess, model, test_data):
    cur_preds = sess.run(model.preds, feed_dict={model.x: test_data})
    return cur_preds


def auto_regressive_test(predict_func, pred_step, test_data):
    test_data_backup = test_data.copy()
    test_preds = None
    for tmp_ind in range(int(pred_step)):
        cur_preds = predict_func(test_data)
        # cur_preds = sess.run(model.preds, feed_dict={model.x: test_data})
        # cur_preds = cur_preds[:, np.newaxis, :]
        cur_preds = np.expand_dims(cur_preds, axis=1)
        if test_preds is None:
            test_preds = cur_preds
        else:
            test_preds = np.concatenate([test_preds, cur_preds], axis=1)
            # test_preds = np.c_[test_preds, cur_preds]
        test_data_backup = np.concatenate((test_data_backup, cur_preds), axis=1)[:, 1:]
        if args.input_add_differ:
            test_data = add_variation_info(test_data_backup)
        else:
            test_data = test_data_backup
    test_preds = np.reshape(test_preds, [test_data.shape[0], -1])
    return test_preds


def transform_train_data(traindata, trainlabel):
    # train_all = None
    # for tmp_ind in range(args.label_num):
    #     cur_all = np.c_[traindata[:, tmp_ind:], trainlabel[:, :tmp_ind + 1]]
    #     if train_all is None:
    #         train_all = cur_all
    #     else:
    #         train_all = np.r_[train_all, cur_all]
    # traindata, trainlabel = train_all[:, :-1], train_all[:, -1:]

    trainlabel = trainlabel[:, 0:args.feat_num]
    return traindata, trainlabel


model_dict = {
    'mlp': models.MLP,
    'lstm': models.LSTM,
    'cnn': models.CNN,
    'transformer': transformer.Transformer,
}


def train(round_n): # round_n : 第几次运行
    # model = models.LSTM(args)
    model = model_dict[args.model_select](args)

    tenfold_train_mse = []
    tenfold_train_mer = []
    tenfold_train_ratio = []
    tenfold_val_mse = []
    tenfold_val_mer = []
    tenfold_val_ratio = []
    tenfold_time = []
    # for traindata, trainlabel, valdata, vallabel in tenfold_generator(data, label):
    traindata, trainlabel, valdata, vallabel = list(tenfold_generator(data, label))[0]

    if args.auto_regressive:
        if args.is_variable_len:
            raise NotImplementedError
        else:
            traindata, trainlabel = transform_train_data(traindata, trainlabel)
            valdata, vallabel = transform_train_data(valdata, vallabel)

    train_size = traindata.shape[0]
    val_size = valdata.shape[0]

    if not args.is_variable_len:
        valdata = np.reshape(valdata, [-1, args.seq_num, args.feat_num])
        valdata = valdata[:, :args.act_seq_num, :]
    else:
        val_seq_len = valdata[:, 0]
        valdata = np.reshape(valdata[:, 1:], [-1, args.seq_max_len, args.feat_num])  ### 第1列是seq_len

    if args.input_add_differ:
        valdata = add_variation_info(valdata)

    if args.model_select == 'transformer':
        cls_pad = 0.1 * np.ones([valdata.shape[0], 1, valdata.shape[2]])
        valdata = np.concatenate((cls_pad, valdata), axis=1)

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
        best_loss_weight = np.ones(args.label_num)
        best_ensemble_weight = -1
        loss_weights = []
        es_count = 0
        for epo in range(args.max_epoch):

            batch_num = 0
            if args.auto_regressive:
                tmp_dim = args.feat_num
            else:
                tmp_dim = args.label_num
            train_mse, train_mer, train_ratio = np.zeros(tmp_dim), np.zeros(tmp_dim), np.zeros(tmp_dim)
            for batch_xs, batch_ys in batch_generator(traindata, trainlabel, args.batch_size):
                if not args.is_variable_len:
                    batch_xs = np.reshape(batch_xs, [-1, args.seq_num, args.feat_num])
                    batch_xs = batch_xs[:, :args.act_seq_num, :]
                    if args.input_add_differ:
                        batch_xs = add_variation_info(batch_xs)
                    if args.model_select == 'transformer':
                        cls_pad = 0.1 * np.ones([batch_xs.shape[0], 1, batch_xs.shape[2]])
                        batch_xs = np.concatenate((cls_pad, batch_xs), axis=1)
                    _, mse, mer = sess.run([model.train_step, model.model_mse, model.model_mer],
                                           feed_dict={model.x: batch_xs, model.y: batch_ys,
                                                      model.lw: args.loss_weight})
                else:
                    batch_seq_len = batch_xs[:, 0]
                    batch_xs = np.reshape(batch_xs[:, 1:], [-1, args.seq_max_len, args.feat_num])
                    if args.input_add_differ:
                        batch_xs = add_variation_info(batch_xs)
                    if args.model_select == 'transformer':
                        cls_pad = 0.1 * np.ones([batch_xs.shape[0], 1, batch_xs.shape[2]])
                        batch_xs = np.concatenate((cls_pad, batch_xs), axis=1)
                    _, mse, mer = sess.run([model.train_step, model.model_mse, model.model_mer],
                                           feed_dict={model.x: batch_xs, model.y: batch_ys,
                                                      model.lw: args.loss_weight,
                                                      model.seq_lens: batch_seq_len})
                # _, output, mse = sess.run([train_step, output, loss], feed_dict={x:batch_xs, y:batch_ys})
                train_mse += mse.sum(axis=0) / train_size
                train_mer += mer.sum(axis=0) / train_size
                tmp_ratio = np.array(mer)
                tmp_ratio[tmp_ratio <= args.mer_limit] = 0
                tmp_ratio[tmp_ratio > args.mer_limit] = 1
                train_ratio += tmp_ratio.sum(axis=0) / train_size

                batch_num += 1
                if batch_num % args.display_step == 0:
                    print('round:{} epoch:{} step:{} mse:{} mer:{} ratio:{}'.format(round_n, epo, batch_num,
                                                                                   mse.mean(), mer.mean(),
                                                                                   tmp_ratio.sum(axis=0) /
                                                                                   batch_xs.shape[0]))
                    if args.record_train_detail:
                        print('round:{} epoch:{} step:{} mse:{} mer:{} ratio:{}'.format(round_n, epo, batch_num,
                                                                                       mse.mean(), mer.mean(),
                                                                                       tmp_ratio.sum(axis=0) /
                                                                                       batch_xs.shape[0]), file=fa)

            ### val_mse, val_mer: np.array 2D
            # if args.auto_regressive:
            #     val_preds = auto_regressive_test(sess, model, args.label_num / args.feat_num, valdata)
            #     val_mse = np.square(val_preds - vallabel)
            #     val_mer = np.abs((val_preds - vallabel) / vallabel)
            # else:
            if not args.is_variable_len:
                val_mse, val_mer = sess.run([model.model_mse, model.model_mer],
                                            feed_dict={model.x: valdata, model.y: vallabel})
            else:
                val_mse, val_mer = sess.run([model.model_mse, model.model_mer],
                                            feed_dict={model.x: valdata, model.y: vallabel,
                                                       model.seq_lens: val_seq_len})
            val_ratio = np.array(val_mer)
            val_ratio[val_ratio <= args.mer_limit] = 0
            val_ratio[val_ratio > args.mer_limit] = 1
            val_ratio = val_ratio.sum(axis=0) / val_size

            val_mer = val_mer.mean(axis=0)

            val_mse = val_mse.mean(axis=0)
            print('round:{} epoch:{} val mse:{} val mer:{} val ratio:{}'.format(round_n, epo, val_mse.mean(),
                                                                               val_mer.mean(), val_ratio))
            if args.record_train_detail:
                print('round:{} epoch:{} val mse:{} val mer:{} val ratio:{}'.format(round_n, epo, val_mse.mean(),
                                                                                val_mer.mean(), val_ratio), file=fa)
            if args.loss_function == 'mer':
                if_val_perform_better = val_mer.mean() < min_val_mer.mean()
            elif args.loss_function == 'mse':
                if_val_perform_better = val_mse.mean() < min_val_mse.mean()
            else:
                raise NotImplementedError('args.loss_function invalid.')
            if if_val_perform_better:
                saver.save(sess, save_path=args.savedir + '/min.ckpt')
                min_train_mse = train_mse
                min_train_mer = train_mer
                min_train_ratio = train_ratio
                min_val_mse = val_mse
                min_val_mer = val_mer
                min_val_ratio = val_ratio
                if args.use_loss_weight and (not args.auto_regressive or args.is_multi_var):
                    best_loss_weight = args.loss_weight
                if args.ensemble_naive:
                    best_ensemble_weight = sess.run(model.alpha)
                print('save model at Epoch:{}'.format(epo))
                if args.record_train_detail:
                    print('save model at Epoch:{}'.format(epo), file=fa)
                es_count = 0
            else:
                es_count += 1

            if es_count == args.early_stop:
                print('val_loss have no improvement for {} epochs, early stop.\n'.format(args.early_stop))
                print('train stop at epoch:{}.'.format(epo), file=fa)
                if args.record_train_detail:
                    print('val_loss have no improvement for {} epochs, early stop.\n'.format(args.early_stop), file=fa)
                if args.use_loss_weight and (not args.auto_regressive or args.is_multi_var):
                    print('best loss weight:', best_loss_weight, file=fa)
                if args.ensemble_naive:
                    print('best ensemble weight:', best_ensemble_weight, file=fa)
                break

            if args.use_loss_weight and (not args.auto_regressive or args.is_multi_var):
                if args.auto_regressive:
                    args.loss_weight = val_mer / val_mer.sum() * args.feat_num
                else:
                    args.loss_weight = val_mer / val_mer.sum() * args.label_num
                loss_weights.append(args.loss_weight)
                print('current loss weight:', args.loss_weight)
            if args.ensemble_naive:
                print('current ensemble weight:', sess.run(model.alpha))
            print()
            if args.record_train_detail:
                print('current loss weight:', args.loss_weight, file=fa)
                print(file=fa)
            # break

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
    # plt.savefig('figure/cnn_loss_weights.eps')
    # plt.show()

    t2 = time.time()
    tenfold_time.append(t2 - t1)
    print('train time consume:{}\n'.format(t2 - t1))

    print('train_mse:{} train_mer:{} train_ratio:{}'.format(np.mean(tenfold_train_mse, axis=0),
                                                            np.mean(tenfold_train_mer, axis=0),
                                                            np.mean(tenfold_train_ratio, axis=0)))
    print('val_mse:{} val_mer:{} val_ratio:{}'.format(np.mean(tenfold_val_mse, axis=0),
                                                      np.mean(tenfold_val_mer, axis=0),
                                                      np.mean(tenfold_val_ratio, axis=0)))
    print('mean_time_cost:{}'.format(np.array(tenfold_time).mean()))
    # print(loss_weight)
    if not os.path.exists('res'):
        os.mkdir('res')
    # with open('res/cnn_res.txt', 'a') as fa:
    print('train_mse:{} train_mer:{} train_ratio:{}'.format(np.mean(tenfold_train_mse, axis=0),
                                                            np.mean(tenfold_train_mer, axis=0),
                                                            np.mean(tenfold_train_ratio, axis=0)), file=fa)
    print('val_mse:{} val_mer:{} val_ratio:{}'.format(np.mean(tenfold_val_mse, axis=0),
                                                      np.mean(tenfold_val_mer, axis=0),
                                                      np.mean(tenfold_val_ratio, axis=0)), file=fa)
    print('mean_time_cost:{}'.format(np.array(tenfold_time).mean()), file=fa)
    print(file=fa)


def draw_predict_and_true_value_curve(pred_lst, label_lst, plot_method, save_name=None):
    assert len(pred_lst) == len(label_lst)
    if plot_method == 'line':
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, pred_lst.__len__() + 1), pred_lst, linewidth=1, linestyle='-', label='predict value')
        plt.plot(range(1, label_lst.__len__() + 1), label_lst, linewidth=1, linestyle='-', label='true value')
        plt.xlabel('sample number', fontsize=18)
        plt.ylabel('value', fontsize=18)
    elif plot_method == 'scatter':
        fig = plt.figure(figsize=(12, 12))
        sns.regplot(pred_lst, label_lst)
        fig_ax = fig.gca()
        fig_ax.set_xlim(bottom=0)
        fig_ax.set_ylim(bottom=0)
        plt.xlabel('predict value')
        plt.ylabel('true value')
    # plt.legend(loc='lower right')
    plt.legend()
    if save_name is not None:
        plt.savefig('figure/{}.eps'.format(save_name))
    plt.show()


def plot_predict_and_true_value_curve(filename_prefix, pred_value, true_value, plot_method='line',
                                        save_fig=False, max_pred_step=4):
    for tmp_ind in range(pred_value.shape[1]):
        if tmp_ind >= max_pred_step:
            break
        fig_name = None
        if save_fig:
            fig_name = '{}_{}_predstep{}'.format(args.savedir[args.savedir.rfind('/') + 1:],
                                                 filename_prefix, tmp_ind)
        draw_predict_and_true_value_curve(pred_value[:, tmp_ind], true_value[:, tmp_ind],
                                          plot_method=plot_method, save_name=fig_name)


def save_predict_and_true_value_to_file(filename_prefix, pred_value, true_value):
    assert pred_value.shape == true_value.shape
    file_name = '{}/{}_preds.csv'.format(args.savedir, filename_prefix)
    with open(file_name, 'w') as fw:
        csv_write = csv.writer(fw)
        csv_write.writerows(np.c_[pred_value, true_value])


def print_mse_mer_ratio(preds, labels, name, writer=None):
    assert preds.shape == labels.shape and len(labels.shape) <= 2
    data_size = labels.shape[0]
    print('data size: {}'.format(data_size))
    res_mse = np.mean(np.square(preds - labels), axis=0)

    mer = np.abs(preds - labels) / labels
    res_mer = np.mean(mer, axis=0)

    mer[mer <= args.mer_limit] = 0
    mer[mer > args.mer_limit] = 1
    res_ratio = np.sum(mer, axis=0) / data_size

    print('{} mse per indicator::'.format(name), res_mse)
    print('{} mer per indicator:'.format(name), res_mer)
    print('{} ratio per indicator:'.format(name), res_ratio)
    if writer is not None:
        # with open(args.savedir + '/res.txt', 'a') as fa:
        print('{} mse per indicator::'.format(name), res_mse, file=writer)
        print('{} mer per indicator:'.format(name), res_mer, file=writer)
        print('{} ratio per indicator:'.format(name), res_ratio, file=writer)
        print(file=writer)
    return res_mse, res_mer, res_ratio


def test():
    if args.istrain:
        args.istrain = False
        model = model_dict[args.model_select](args)
        args.istrain = True
    else:
        model = model_dict[args.model_select](args)

    print()
    if args.istrain:
        print(file=fa)

    test_mse, test_mer, test_ratio = None, None, None

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.latest_checkpoint(args.savedir)  # 注意
        if ckpt:
            saver = tf.train.Saver()
            saver.restore(sess, ckpt)

            tf_predict_wrap = ft.partial(tf_predict, sess, model)

            t1 = time.time()
            if args.auto_regressive:
                if args.is_variable_len:
                    raise NotImplementedError
                else:
                    assert args.label_num % args.feat_num == 0
                    test_preds = auto_regressive_test(tf_predict_wrap, args.label_num / args.feat_num, testdata)
            else:
                if not args.is_variable_len:
                    test_preds = sess.run(model.preds, feed_dict={model.x: testdata})
                else:
                    test_preds = sess.run(model.preds, feed_dict={model.x: testdata, model.seq_lens: test_seq_len})
                # pred = sess.run(preds, feed_dict={x: testdata[46:47]})
            t2 = time.time()
            test_pred_num = test_preds.shape[0]
            c = (t2 - t1) / test_pred_num
            print('prediction time cost per instance:{}'.format(c))
            if args.istrain:
                print('prediction time cost per instance:{}'.format(c), file=fa)


            ### 训练集，验证集，测试集预测存入文件
            if not args.istrain:
                traindata, trainlabel, valdata, vallabel = list(tenfold_generator(data, label))[0]
                if not args.is_variable_len:
                    if args.auto_regressive:
                        traindata, trainlabel = transform_train_data(traindata, trainlabel)
                        valdata, vallabel = transform_train_data(valdata, vallabel)
                    traindata = np.reshape(traindata, [-1, args.seq_num, args.feat_num])
                    traindata = traindata[:, :args.act_seq_num, :]  ### 注变长情况要改
                    valdata = np.reshape(valdata, [-1, args.seq_num, args.feat_num])
                    valdata = valdata[:, :args.act_seq_num, :]  ### 注变长情况要改
                    if args.input_add_differ:
                        traindata = add_variation_info(traindata)
                        valdata = add_variation_info(valdata)
                    if args.model_select == 'transformer':
                        cls_pad = 0.1 * np.ones([traindata.shape[0], 1, traindata.shape[2]])
                        traindata = np.concatenate((cls_pad, traindata), axis=1)
                        cls_pad = 0.1 * np.ones([valdata.shape[0], 1, valdata.shape[2]])
                        valdata = np.concatenate((cls_pad, valdata), axis=1)
                    train_preds = sess.run(model.preds, feed_dict={model.x: traindata})
                    # if args.auto_regressive:
                    #     val_preds = auto_regressive_test(sess, model, args.label_num / args.feat_num, valdata)
                    # else:
                    val_preds = sess.run(model.preds, feed_dict={model.x: valdata})
                else:
                    if args.auto_regressive:
                        raise NotImplementedError
                    train_seq_len = traindata[:, 0]
                    traindata = np.reshape(traindata[:, 1:], [-1, args.seq_max_len, args.feat_num])
                    val_seq_len = valdata[:, 0]
                    valdata = np.reshape(valdata[:, 1:], [-1, args.seq_max_len, args.feat_num])
                    if args.input_add_differ:
                        traindata = add_variation_info(traindata)
                        valdata = add_variation_info(valdata)
                    if args.model_select == 'transformer':
                        cls_pad = 0.1 * np.ones([traindata.shape[0], 1, traindata.shape[2]])
                        traindata = np.concatenate((cls_pad, traindata), axis=1)
                        cls_pad = 0.1 * np.ones([valdata.shape[0], 1, valdata.shape[2]])
                        valdata = np.concatenate((cls_pad, valdata), axis=1)
                    train_preds = sess.run(model.preds, feed_dict={model.x: traindata, model.seq_lens: train_seq_len})
                    val_preds = sess.run(model.preds, feed_dict={model.x: valdata, model.seq_lens: val_seq_len})

                save_predict_and_true_value_to_file('train', train_preds, trainlabel)
                save_predict_and_true_value_to_file('val', val_preds, vallabel)
                save_predict_and_true_value_to_file('test', test_preds, testlabel)
                if args.plot_test_res:
                    plot_predict_and_true_value_curve('test', test_preds, testlabel,
                                                      plot_method=args.plot_line_or_scatter, save_fig=True)

                print_mse_mer_ratio(train_preds, trainlabel, 'train')
                print_mse_mer_ratio(val_preds, vallabel, 'val')

            test_mse, test_mer, test_ratio = print_mse_mer_ratio(test_preds, testlabel, 'test')

            if args.istrain:
                print('test mse per indicator:', test_mse, file=fa)
                print('test mer per indicator:', test_mer, file=fa)
                print('test ratio per indicator:', test_ratio, file=fa)

        else:
            raise FileNotFoundError("ckpt '{}' not found.".format(args.savedir))

        if args.is_pso:
            # assert last_equal_num
            dosage_inds = [7] # 冰醋酸
            dosage_bounds = [[5, 30]]
            output_quality_inds = [1] # NH3N, TN, COD
            output_quality_limits = [6.17]
            optimization.pso(model, sess, args, dosage_inds, dosage_bounds, output_quality_inds, output_quality_limits,
                             testdata, testlabel, test_seq_len)

    return test_mse, test_mer, test_ratio


if __name__ == '__main__':

    fa = open('{}/res.txt'.format(args.savedir), 'a')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    if args.istrain:
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), file=fa)
    for k, v in args.values().items():
        print('{} : {}'.format(k, v))
        if args.istrain:
            print('{} : {}'.format(k, v), file=fa)

    if not args.istrain:
        test()
        sys.exit(0)

    t_begin = time.time()

    n_runs_mse = []
    n_runs_mer = []
    n_runs_ratio = []

    for round_ind in range(args.number_of_runs):
        print('\n\n-----------------Round {}-----------------'.format(round_ind))
        print('\n\n-----------------Round {}-----------------'.format(round_ind), file=fa)
        train(round_ind)
        test_mse, test_mer, test_ratio = test()
        n_runs_mse.append(test_mse)
        n_runs_mer.append(test_mer)
        n_runs_ratio.append(test_ratio)

    if args.get_test_res_by == 'mean':
        n_runs_mse = np.mean(np.array(n_runs_mse), axis=0)
        n_runs_mer = np.mean(np.array(n_runs_mer), axis=0)
        n_runs_ratio = np.mean(np.array(n_runs_ratio), axis=0)
    elif args.get_test_res_by == 'max':
        n_runs_mse = np.max(np.array(n_runs_mse), axis=0)
        n_runs_mer = np.max(np.array(n_runs_mer), axis=0)
        n_runs_ratio = np.max(np.array(n_runs_ratio), axis=0)
    print('\n')
    print('\n', file=fa)
    print('{} runs {} test mse per indicator:'.format(args.number_of_runs, args.get_test_res_by), n_runs_mse)
    print('{} runs {} test mse per indicator:'.format(args.number_of_runs, args.get_test_res_by), n_runs_mse, file=fa)
    print('{} runs {} test mer per indicator:'.format(args.number_of_runs, args.get_test_res_by), n_runs_mer)
    print('{} runs {} test mer per indicator:'.format(args.number_of_runs, args.get_test_res_by), n_runs_mer, file=fa)
    print('{} runs {} test ratio per indicator:'.format(args.number_of_runs, args.get_test_res_by), n_runs_ratio)
    print('{} runs {} test ratio per indicator:'.format(args.number_of_runs, args.get_test_res_by), n_runs_ratio, file=fa)

    t_end = time.time()
    print('{} runs total time cost:'.format(args.number_of_runs), t_end - t_begin)
    print('{} runs total time cost:'.format(args.number_of_runs), t_end - t_begin, file=fa)
    print('\n\n\n', file=fa)

    fa.close()
