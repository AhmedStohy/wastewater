import tensorflow as tf
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import time
import csv

import models
import algorithms

from data_utils import process_raw_data, tenfold_generator, batch_generator, process_raw_data_variable_len

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def create_hparams():
    params = tf.contrib.training.HParams(
        raw_data = 'data/20200414dataNH3N_H_24seqlen.txt', # 原始数据文件，！！注意
        test_rate = 0.2, # 测试数据占总数据比例
        savedir = 'ckpt/rnn_200414NH3N_H_24seqlen_lw_ph_last_sigmoid_ln_ar_32hid_3layer', # 保存实验结果
        number_of_runs = 10, # 重复训练测试若干次，取测试效果平均
        record_train_detail = False, # 是否将训练细节输出到文件
        first_random = False, # processed_data第一次做实验需要乱序，第二次开始False
        liuliangInd=4,  # 在原始文件中的列号，从0开始
        isVariableLen=False,

        time_increment = 24 * 60, # 注意输入输出时间间隔，单位为分
        time_intv_per_sample = 1 * 60, # x min一条数据
        last_equal_intv = 0, # 最后 x min的输入视为一样，0表示无
        feat_num = 1, # 特征个数，注意
        label_num = 3, # 需要向后预测的时间步个数
        mer_limit = 0.1, # mer > mer_limit的被认为超限
        seq_len_limit=None,

        layer_num = 3, # LSTM层数
        head_num = 1,
        hid_dims = [32],
        # hid_dims = [128],
        # out_hid_dims = [256],
        out_hid_dims = [32],

        keep_prop = 1.0,
        learning_rate = 0.0003,
        max_epoch = 10000,
        batch_size = 64,
        early_stop = 100,

        auto_regressive = True, # 是否自回归地向后预测
        use_loss_weight = False, # 是否启用动态loss weight
        input_add_differ = False, # 是否在输入中加入一阶差分信息
        ensemble_naive = True,

        istrain = True,
        display_step = 5,

        is_pso=False,
    )
    return params

args = create_hparams()
args.add_hparam('processed_data', args.raw_data[:args.raw_data.rfind('.')] + '_ok.txt') # 处理后文件名
args.add_hparam('last_equal_num', args.last_equal_intv // args.time_intv_per_sample) # 最后 x 个输入一样
args.add_hparam('seq_num', args.time_increment // args.time_intv_per_sample) # 序列长度
args.add_hparam('act_seq_num', args.seq_num // 1)
args.add_hparam('total_feat_num', args.feat_num * args.act_seq_num) # 注意
if args.auto_regressive:
    args.add_hparam('loss_weight', np.ones(1)) # 初始值
else:
    args.add_hparam('loss_weight', np.ones(args.label_num)) # 初始值

if not os.path.exists('ckpt'):
    os.mkdir('ckpt')

if not os.path.exists(args.savedir):
    os.mkdir(args.savedir)


data = None
label = None

data_label = None
if not os.path.exists(args.processed_data):
    if not args.isVariableLen:
        data, label = process_raw_data(args.raw_data, args.processed_data, args.seq_num, args.label_num, args.time_increment)
    else:
        data, label = process_raw_data_variable_len(args.raw_data, args.processed_data, args.liuliangInd+1, args.feat_num, args.label_num, args.seq_len_limit)
else:
    if not args.isVariableLen:
        data_label = np.genfromtxt(args.processed_data, np.float32, delimiter='\t')
        data, label = data_label[:, :args.total_feat_num], data_label[:, args.total_feat_num:args.total_feat_num+args.label_num]
    else:
        seqlen_data_label = np.genfromtxt(args.processed_data, np.float32, delimiter='\t')
        data, label = seqlen_data_label[:,:-args.label_num], seqlen_data_label[:, -args.label_num:]

data = np.array(data, dtype=np.float32)
label = np.array(label, dtype=np.float32)

### 入水时刻 x min后的入水都视为一样
if not args.isVariableLen:
    if args.last_equal_num:
        last_equal_feats_num = args.last_equal_num * args.feat_num
        data[:, -last_equal_feats_num:] = np.tile(data[:, -last_equal_feats_num:
                                -last_equal_feats_num+args.feat_num], args.last_equal_num)
else:
    max_len = int((data.shape[1]-1) / args.feat_num) # -1是因为第一维是序列长度
    args.seq_num = max_len
    if args.last_equal_num: # 变长填充，所有时间步入水指标一样，等于第一时间步的
        seq_len = data[:, 0].astype(np.int32) ### 注
        for i in range(data.shape[0]):
            last_equal_feats_num = args.feat_num * seq_len[i] ### 注
            pad_size = (max_len - seq_len[i]) * args.feat_num
            if pad_size:
                data[i, -(last_equal_feats_num + pad_size) : -pad_size] = np.tile(data[i, -(last_equal_feats_num + pad_size) : -(last_equal_feats_num + pad_size) + args.feat_num], seq_len[i])
            else:
                data[i, -(last_equal_feats_num + pad_size):] = np.tile(data[i, -(last_equal_feats_num + pad_size) : -(last_equal_feats_num + pad_size) + args.feat_num], seq_len[i])


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

if not args.isVariableLen:
    testdata = np.reshape(testdata, [-1, args.seq_num, args.feat_num])
    testdata = testdata[:, :args.act_seq_num, :]
else:
    test_seq_len = np.array(testdata[:, 0], dtype=np.int32)
    testdata = np.reshape(testdata[:, 1:], [-1, args.seq_num, args.feat_num])


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


def auto_regressive_test(sess, model, pred_num, test_data):
    test_data_backup = test_data.copy()
    test_preds = None
    for tmp_ind in range(pred_num):
        cur_preds = sess.run(model.preds, feed_dict={model.x: test_data})
        if test_preds is None:
            test_preds = cur_preds
        else:
            test_preds = np.c_[test_preds, cur_preds]
        cur_preds = cur_preds[:, :, np.newaxis]
        test_data_backup = np.concatenate((test_data_backup, cur_preds), axis=1)[:, 1:, :]
        if args.input_add_differ:
            test_data = add_variation_info(test_data_backup)
        else:
            test_data = test_data_backup
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

    trainlabel = trainlabel[:, 0:1]
    return traindata, trainlabel


def train():
    model = models.LSTM(args)

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

    if args.auto_regressive:
        if args.isVariableLen:
            raise NotImplementedError
        else:
            traindata, trainlabel = transform_train_data(traindata, trainlabel)

    train_size = traindata.shape[0]
    val_size = valdata.shape[0]

    if not args.isVariableLen:
        valdata = np.reshape(valdata, [-1, args.seq_num, args.feat_num])
        valdata = valdata[:, :args.act_seq_num, :]
    else:
        val_seq_len = valdata[:, 0]
        valdata = np.reshape(valdata[:, 1:], [-1, args.seq_num, args.feat_num]) ### 第1列是seq_len

    if args.input_add_differ:
        valdata = add_variation_info(valdata)

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
            train_mse, train_mer, train_ratio = np.zeros(args.label_num), np.zeros(args.label_num), \
                                                np.zeros(args.label_num)
            for batch_xs, batch_ys in batch_generator(traindata, trainlabel, args.batch_size):
                if not args.isVariableLen:
                    batch_xs = np.reshape(batch_xs, [-1, args.seq_num, args.feat_num])
                    batch_xs = batch_xs[:, :args.act_seq_num, :]
                    if args.input_add_differ:
                        batch_xs = add_variation_info(batch_xs)
                    _, mse, mer = sess.run([model.train_step, model.model_mse, model.model_mer],
                                           feed_dict={model.x: batch_xs, model.y: batch_ys,
                                                      model.lw: args.loss_weight})
                else:
                    batch_seq_len = batch_xs[:, 0]
                    batch_xs = np.reshape(batch_xs[:, 1:], [-1, args.seq_num, args.feat_num])
                    if args.input_add_differ:
                        batch_xs = add_variation_info(batch_xs)
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
                    print('fold:{} epoch:{} step:{} mse:{} mer:{} ratio:{}'.format(fold_id, epo, batch_num,
                                            mse.mean(), mer.mean(), tmp_ratio.sum(axis=0) / batch_xs.shape[0]))
                    if args.record_train_detail:
                        print('fold:{} epoch:{} step:{} mse:{} mer:{} ratio:{}'.format(fold_id, epo, batch_num,
                                            mse.mean(), mer.mean(), tmp_ratio.sum(axis=0) / batch_xs.shape[0]), file=fa)


            ### val_mse, val_mer: np.array 2D
            if args.auto_regressive:
                val_preds = auto_regressive_test(sess, model, args.label_num, valdata)
                val_mse = np.square(val_preds - vallabel)
                val_mer = np.abs((val_preds - vallabel) / vallabel)
            else:
                if not args.isVariableLen:
                    val_mse, val_mer = sess.run([model.model_mse, model.model_mer],
                                                feed_dict={model.x: valdata, model.y: vallabel})
                else:
                    val_mse, val_mer = sess.run([model.model_mse, model.model_mer],
                                                feed_dict={model.x: valdata, model.y: vallabel,
                                                           model.seq_lens:val_seq_len})
            val_ratio = np.array(val_mer)
            val_ratio[val_ratio <= args.mer_limit] = 0
            val_ratio[val_ratio > args.mer_limit] = 1
            val_ratio = val_ratio.sum(axis=0) / val_size

            val_mer = val_mer.mean(axis=0)
            if args.use_loss_weight and not args.auto_regressive:
                args.loss_weight = val_mer / val_mer.sum() * args.label_num

            val_mse = val_mse.mean(axis=0)
            print('fold:{} epoch:{} val mse:{} val mer:{} val ratio:{}'.format(fold_id, epo, val_mse.mean(),
                                                                                 val_mer.mean(), val_ratio))
            if args.record_train_detail:
                print('fold:{} epoch:{} val mse:{} val mer:{} val ratio:{}'.format(fold_id, epo, val_mse.mean(),
                                                                                 val_mer.mean(), val_ratio), file=fa)
            if val_mer.mean() < min_val_mer.mean():
            # if val_mse.mean() < min_val_mse.mean():
                saver.save(sess, save_path=args.savedir + '/min.ckpt')
                min_train_mse = train_mse
                min_train_mer = train_mer
                min_train_ratio = train_ratio
                min_val_mse = val_mse
                min_val_mer = val_mer
                min_val_ratio = val_ratio
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
                break

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
    # plt.savefig('cnn_loss_weights.png')
    # plt.show()

    fold_id += 1
    t2 = time.time()
    tenfold_time.append(t2 - t1)
    print('train time consume:{}\n'.format(t2 - t1))


    print('train_mse:{} train_mer:{} train_ratio:{}'.format(np.mean(tenfold_train_mse, axis=0),
                                            np.mean(tenfold_train_mer, axis=0), np.mean(tenfold_train_ratio, axis=0)))
    print('val_mse:{} val_mer:{} val_ratio:{}'.format(np.mean(tenfold_val_mse, axis=0),
                                        np.mean(tenfold_val_mer, axis=0), np.mean(tenfold_val_ratio, axis=0)))
    print('mean_time_cost:{}'.format(np.array(tenfold_time).mean()))
    # print(loss_weight)
    if not os.path.exists('res'):
        os.mkdir('res')
    # with open('res/cnn_res.txt', 'a') as fa:
    print('train_mse:{} train_mer:{} train_ratio:{}'.format(np.mean(tenfold_train_mse, axis=0),
                                np.mean(tenfold_train_mer, axis=0), np.mean(tenfold_train_ratio, axis=0)), file=fa)
    print('val_mse:{} val_mer:{} val_ratio:{}'.format(np.mean(tenfold_val_mse, axis=0),
                            np.mean(tenfold_val_mer, axis=0), np.mean(tenfold_val_ratio, axis=0)), file=fa)
    print('mean_time_cost:{}'.format(np.array(tenfold_time).mean()), file=fa)
    print(file=fa)
    
    
def draw_predict_and_true_value_curve(pred_lst, label_lst):
    assert len(pred_lst) == len(label_lst)
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, pred_lst.__len__() + 1), pred_lst, linewidth=1, linestyle='-', label='predict')
    plt.plot(range(1, label_lst.__len__() + 1), label_lst, linewidth=1, linestyle='-', label='label')
    plt.xlabel('sample number')
    plt.ylabel('value')
    # plt.legend(loc='lower right')
    plt.legend()
    # plt.savefig('out_ind{}.png'.format(i+1))
    plt.show()


def save_predict_and_true_value_to_file(filename_prefix, pred_value, true_value):
    assert pred_value.shape == true_value.shape
    file_name = '{}/{}_preds.csv'.format(args.savedir, filename_prefix)
    with open(file_name, 'w') as fw:
        csv_write = csv.writer(fw)
        csv_write.writerows(np.c_[pred_value, true_value])
        for tmp_ind in range(pred_value.shape[1]):
            draw_predict_and_true_value_curve(pred_value[:, tmp_ind], true_value[:, tmp_ind])


def print_mse_mer_ratio(preds, labels, name):
    assert preds.shape == labels.shape and len(labels.shape) <= 2
    data_size = labels.shape[0]
    print('data size: {}'.format(data_size))
    mer = np.abs(preds - labels) / labels
    print('{}_mse:'.format(name), np.mean(np.square(preds - labels), axis=0))
    print('{}_mer:'.format(name), np.mean(mer, axis=0))
    mer[mer <= args.mer_limit] = 0
    mer[mer > args.mer_limit] = 1
    print('{}_ratio'.format(name), np.sum(mer, axis=0) / data_size)


def test():
    if args.istrain:
        args.istrain = False
        model = models.LSTM(args)
        args.istrain = True
    else:
        model = models.LSTM(args)

    print()
    print(file=fa)
    tenfold_ratio = []
    tenfold_predtc = []
    tenfold_mse = []
    tenfold_mer = []
    last_table = None  # 画预测值实际值对比表

    target_fold_id = 0

    # for fold_id in range(10):
    #     if fold_id != target_fold_id:
    #         continue

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.latest_checkpoint(args.savedir) # 注意
        if ckpt:
            saver = tf.train.Saver()
            saver.restore(sess, ckpt)

            t1 = time.time()
            if args.auto_regressive:
                if args.isVariableLen:
                    raise NotImplementedError
                else:
                    pred = auto_regressive_test(sess, model, args.label_num, testdata)
            else:
                if not args.isVariableLen:
                    pred = sess.run(model.preds, feed_dict={model.x: testdata})
                else:
                    pred = sess.run(model.preds, feed_dict={model.x: testdata, model.seq_lens: test_seq_len})
                # pred = sess.run(preds, feed_dict={x: testdata[46:47]})
            t2 = time.time()
            pred_num = pred.shape[0]
            c = (t2 - t1) / pred_num
            tenfold_predtc.append(c)
            print('prediction time cost per instance:{}'.format(c))
            print('prediction time cost per instance:{}'.format(c), file=fa)

            ### 预测值 真实值存入文件
            # if fold_id == target_fold_id:

            ### 训练集，验证集预测也存入文件
            if not args.istrain:
                traindata, trainlabel, valdata, vallabel = list(tenfold_generator(data, label))[0]
                if not args.isVariableLen:
                    if args.auto_regressive:
                        traindata, trainlabel = transform_train_data(traindata, trainlabel)
                    traindata = np.reshape(traindata, [-1, args.seq_num, args.feat_num])
                    traindata = traindata[:, :args.act_seq_num, :] ### 注变长情况要改
                    valdata = np.reshape(valdata, [-1, args.seq_num, args.feat_num])
                    valdata = valdata[:, :args.act_seq_num, :] ### 注变长情况要改
                    if args.input_add_differ:
                        traindata = add_variation_info(traindata)
                        valdata = add_variation_info(valdata)
                    train_preds = sess.run(model.preds, feed_dict={model.x: traindata})
                    if args.auto_regressive:
                        val_preds = auto_regressive_test(sess, model, args.label_num, valdata)
                    else:
                        val_preds = sess.run(model.preds, feed_dict={model.x: valdata})
                else:
                    if args.auto_regressive:
                        raise NotImplementedError
                    train_seq_len = traindata[:, 0]
                    traindata = np.reshape(traindata[:, 1:], [-1, args.seq_num, args.feat_num])
                    val_seq_len = valdata[:, 0]
                    valdata = np.reshape(valdata[:, 1:], [-1, args.seq_num, args.feat_num])
                    if args.input_add_differ:
                        traindata = add_variation_info(traindata)
                        valdata = add_variation_info(valdata)
                    train_preds = sess.run(model.preds, feed_dict={model.x: traindata, model.seq_lens: train_seq_len})
                    val_preds = sess.run(model.preds, feed_dict={model.x: valdata, model.seq_lens: val_seq_len})

                save_predict_and_true_value_to_file('train', train_preds, trainlabel)
                save_predict_and_true_value_to_file('val', val_preds, vallabel)

                print_mse_mer_ratio(train_preds, trainlabel, 'train')
                print_mse_mer_ratio(val_preds, vallabel, 'val')

            ### 真实值预测值写入文件
            if not args.istrain:
                file_name = '{}/test_preds.csv'.format(args.savedir)
                if args.last_equal_intv:
                    file_name = '{}/test_preds_{}min.csv'.format(args.savedir, args.last_equal_intv)
                with open(file_name, 'w') as fw:
                    csv_write = csv.writer(fw)
                    csv_write.writerows(np.c_[pred, testlabel])

            ### pred1, pred2, pred3, pred4 = list(map(list, zip(*pred)))
            ### label1, label2, label3, label4 = list(map(list, zip(*testlabel)))

            mses = [] # 计算每个输出指标的误差
            mers = [] # 计算每个输出指标的误差
            ol_rates = [] # 计算每个输出指标的误差超限比例
            ni_mers = [] # 计算误差率top5%，top10%的数据的平均误差率多少
            nifi_mers = [] # 计算误差率top5%，top10%的数据的平均误差率多少

            for i, d in enumerate(zip(list(map(list, zip(*pred))), list(map(list, zip(*testlabel))))):
                pred_lst, label_lst = d # 公用

                ### 画 预测-标签 图
                draw_predict_and_true_value_curve(pred_lst, label_lst)

                ## 公用
                pred_lst = np.array(pred_lst)
                label_lst = np.array(label_lst)
                mer_seq = np.abs((pred_lst - label_lst) / label_lst)
                mse_seq = np.square(pred_lst - label_lst)
                mer_lst = np.sort(mer_seq)
                num_test = mer_lst.shape[0]
                x_lst = np.arange(1, num_test + 1) / num_test

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

                ## 计算误差大于args.mer_limit的比例，必须放在最后
                mer_lst[mer_lst <= args.mer_limit] = 0
                overlimit = mer_lst.nonzero()[0].size
                ol_rate = overlimit / num_test ### 注
                # ol_rate = overlimit ### 注
                # print('test_size:', num_test)

                mses.append(np.mean(mse_seq))  # 计算每个输出指标的误差
                mers.append(np.mean(mer_seq))  # 计算每个输出指标的误差
                ol_rates.append(ol_rate) # 计算每个输出指标的误差超限比例


            # print(mses) # 计算每个输出指标的误差
            # print(mers) # 计算每个输出指标的误差
            # print(ol_rates) # 计算每个输出指标的误差超限比例
            tenfold_ratio.append(ol_rates)
            tenfold_mer.append(mers)
            tenfold_mse.append(mses)

            # print(ni_mers) # 计算误差率top5%，top10%的数据的平均误差率多少
            # print(nifi_mers) # 计算误差率top5%，top10%的数据的平均误差率多少

            # print(pred)
            # print(testlabel[46:47])

    # print(np.array(tenfold_predtc).mean())
    test_mse = np.mean(np.array(tenfold_mse), axis=0)
    test_mer = np.mean(np.array(tenfold_mer), axis=0)
    test_ratio = np.mean(np.array(tenfold_ratio), axis=0)
    print('test mse per indicator:', test_mse)
    print('test mse per indicator:', test_mse,file=fa)
    print('test mer per indicator:', test_mer)
    print('test mer per indicator:', test_mer, file=fa)
    print('test ratio per indicator:', test_ratio)
    print('test ratio per indicator:', test_ratio, file=fa)
    # np.savetxt('pred_true.csv', last_table, delimiter=',')

    if args.is_pso:
        # assert last_equal_num
        zd_target = 6
        algorithms.pso(testdata, test_seq_len, model, args, zd_target, args.savedir, testlabel, None)

    return test_mse, test_mer, test_ratio


if __name__ == '__main__':

    fa = open('{}/res.txt'.format(args.savedir), 'a')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), file=fa)
    for k, v in args.values().items():
        print('{} : {}'.format(k, v))
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
        train()
        test_mse, test_mer, test_ratio = test()
        n_runs_mse.append(test_mse)
        n_runs_mer.append(test_mer)
        n_runs_ratio.append(test_ratio)

    n_runs_mse = np.mean(np.array(n_runs_mse), axis=0)
    n_runs_mer = np.mean(np.array(n_runs_mer), axis=0)
    n_runs_ratio = np.mean(np.array(n_runs_ratio), axis=0)
    print('\n')
    print('\n', file=fa)
    print('{} runs test mse per indicator:'.format(args.number_of_runs), n_runs_mse)
    print('{} runs test mse per indicator:'.format(args.number_of_runs), n_runs_mse, file=fa)
    print('{} runs test mer per indicator:'.format(args.number_of_runs), n_runs_mer)
    print('{} runs test mer per indicator:'.format(args.number_of_runs), n_runs_mer, file=fa)
    print('{} runs test ratio per indicator:'.format(args.number_of_runs), n_runs_ratio)
    print('{} runs test ratio per indicator:'.format(args.number_of_runs), n_runs_ratio, file=fa)

    t_end = time.time()
    print('{} runs total time cost:'.format(args.number_of_runs), t_end - t_begin)
    print('{} runs total time cost:'.format(args.number_of_runs), t_end - t_begin, file=fa)
    print('\n\n\n', file=fa)

    fa.close()
