import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import os, sys
import pandas as pd

import train

from sklearn import metrics


def calculate_pearson(data_df):

    values = data_df.values
    mean = data_df.mean().values[np.newaxis:]
    std = data_df.std().values[np.newaxis:]
    values = (values - mean) / std
    data_df = pd.DataFrame(values, columns=data_df.columns)

    corr_df = pd.DataFrame(data_df.corr(method='pearson'), index=data_df.columns,
                           columns=data_df.columns)
    figure, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(corr_df, square=True, annot=True, ax=ax)  # 画出热力图
    plt.show()


def NMI_matrix(df):  # 计算标准化互信息矩阵
    number = df.columns.size  # 获取df的列数
    # Name = list(df.columns)
    Name = ['input NH3N', 'input TN', 'input COD', 'input PH', 'input flow rate', 'Temp', 'ORP', 'GAAD',
            'output NH3N', 'output TN', 'output COD']
    # number = df.columns.size  # 获取df的列数
    List = []
    for i in range(number):
        A = []
        X = df[df.columns[i]]  # df.columns[i]获取对应列的索引，df['索引']获取对应列的数值
        for j in range(number):
            Y = df[df.columns[j]]
            A.append(metrics.normalized_mutual_info_score(X, Y))  # 计算标准化互信息
        List.append(A)  # List是列表格式
    df_NMI = pd.DataFrame(List, index=Name, columns=Name)
    print('NMI(标准化互信息) = \n', df_NMI)  # 将二维列表转为dataframe格式
    figure, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(df_NMI.iloc[:Name.index('output NH3N'), :Name.index('output NH3N')],
                square=True, annot=True, ax=ax, cbar_kws={'shrink': 0.6})  # 画出热力图
    plt.savefig('ch4_NMI.png')
    plt.show()

    df_NMI['input_relate_score'] = np.sum(df_NMI.values[:, :Name.index('output NH3N')], axis=1)
    df_NMI['output_relate_score'] = np.sum(df_NMI.values[:, Name.index('output NH3N'):], axis=1)
    print('input_relate_score')
    print(df_NMI['input_relate_score'])
    print('output_relate_score')
    print(df_NMI['output_relate_score'])
    
    
def pair_plot(data_df):
    sns.pairplot(data_df, kind="reg") # 拟合
    # sns.pairplot(data_df, kind="scatter") # 不拟合
    # sns.pairplot(data_df, kind="scatter", hue="species", markers=["o", "s", "D"], palette="Set2")
    plt.show()
    
    
def partial_dependence_plot():
    def plot_figure(data_df, output_water_quality_names, dosage_name):
        assert data_df.shape[1] == len(output_water_quality_names) + 1
        # data_df.columns = ['Dosage'] + output_water_quality_names
        data_df.columns = ['Temperature'] + output_water_quality_names
        plt.figure(figsize=(6, 6))
        # data_df.set_index('Dosage', inplace=True)
        data_df.set_index('Temperature', inplace=True)
        for out_ind_name in output_water_quality_names:
            # if out_ind_name == 'TN':
                data_df[out_ind_name].plot()
                # plt.xlabel('Dosage', fontsize=16)
                plt.xlabel('Temperature', fontsize=16)
                # plt.ylabel('Output TN', fontsize=16)
                plt.ylabel(out_ind_name, fontsize=16)
                # plt.tick_params(labelsize=20)
                # plt.legend()
                plt.grid(True)
                plt.savefig('figure/PDP_{}_out{}.eps'.format(dosage_name, out_ind_name))
                plt.show()

    model_dict = train.model_dict
    testdata = train.testdata
    test_seq_len = train.test_seq_len
    test_size = testdata.shape[1]
    args = train.args
    if args.istrain:
        args.istrain = False
        model = model_dict[args.model_select](args)
        args.istrain = True
    else:
        model = model_dict[args.model_select](args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.latest_checkpoint(args.savedir)  # 注意
        if ckpt:
            saver = tf.train.Saver()
            saver.restore(sess, ckpt)

            # dosage_names = ['Anhydrous_Acetic_Acid']
            # dosage_inds = [7]  # 冰醋酸
            # dosage_bounds = [[5, 30]]
            dosage_names = ['Temp']
            dosage_inds = [5]  # 温度
            dosage_bounds = [[0, 40]]
            granularity = 1
            output_water_quality_names = ['NH3N', 'TN', 'COD']
            for ind, dosage_ind in enumerate(dosage_inds):
                save_file_name = '{}/input_feat{}_pdp.csv'.format(args.savedir, dosage_ind)
                # if os.path.exists(save_file_name):
                #     data_df = pd.read_csv(save_file_name, header=None)
                #     plot_figure(data_df, output_water_quality_names, dosage_names[ind])
                #     continue
                dosage_vals = np.arange(dosage_bounds[ind][0],
                                        dosage_bounds[ind][1] + granularity,
                                        granularity)
                model_mean_preds = []
                for dosage in dosage_vals:
                    cur_test_preds = []
                    for test_ind in range(test_size):
                        fake_sample = testdata[test_ind].copy()
                        ### 输入填充
                        if args.model_select == 'transformer':
                            fake_sample[1, dosage_ind] = dosage
                            if test_seq_len[test_ind] > 1:
                                fake_sample[2:test_seq_len[test_ind] + 2 - 1] = fake_sample[1:2]
                        else:
                            fake_sample[0, dosage_ind] = dosage
                            if test_seq_len[test_ind] > 1:
                                fake_sample[1:test_seq_len[test_ind] + 1 - 1] = fake_sample[0:1]
                        fake_samples = np.expand_dims(fake_sample, 0)
                        test_pred = sess.run(model.preds, feed_dict={model.x: fake_samples})
                        # print('feat ind:{} dosage:{} test feat:{} test predict:{}'.format(dosage_ind, dosage,
                        #                                                                   testdata[test_ind],
                        #                                                                   test_pred))
                        test_pred = np.squeeze(test_pred)
                        cur_test_preds.append(test_pred)
                    cur_test_preds = np.array(cur_test_preds)
                    cur_mean_preds = np.mean(cur_test_preds, axis=0)
                    model_mean_preds.append(cur_mean_preds)
                    print('feat ind:{} dosage:{} mean predict:{}'.format(dosage_ind, dosage, cur_mean_preds))
                    print()

                model_mean_preds = np.array(model_mean_preds)
                cur_dosage_res_arr = np.c_[np.expand_dims(dosage_vals, 1), model_mean_preds]
                data_df = pd.DataFrame(cur_dosage_res_arr)
                plot_figure(data_df, output_water_quality_names, dosage_names[ind])
                save_file_name = '{}/input_feat{}_pdp.csv'.format(args.savedir, dosage_ind)
                with open(save_file_name, 'w') as fw:  ### savedir 前面字母最后'/'
                    csv_write = csv.writer(fw)
                    assert len(dosage_vals.shape) == 1 and len(model_mean_preds.shape) == 2
                    csv_write.writerows(cur_dosage_res_arr)

        else:
            raise FileNotFoundError("ckpt '{}' not found.".format(args.savedir))



if __name__ == '__main__':

    # datafile = 'data/20200406data_0.txt'
    # data_df = pd.read_csv(datafile, sep='\t', header=0)
    # data_df = data_df.iloc[:, 1:]

    # calculate_pearson(data_df)
    # NMI_matrix(data_df)
    # pair_plot(data_df)

    partial_dependence_plot()