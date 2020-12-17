import csv
import numpy as np
import time
import os, sys
import data_pprc_191219
import data_utils
import data_prep_200402
import data_prep_200406
import matplotlib.pyplot as plt
import pandas as pd

from copy import deepcopy
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler



def transform_tablestruct(filename):
    '''
    :param filename: str: csv file name 行必须按时间顺序递增顺序排列
    :return: None
    '''
    ft_d = {'w01018':2, 'w21011':3, 'w21003':4}
    fts_lst = ['id', 'datetime', 'w01018', 'w21011', 'w21003',]
    # fts_lst = ['id', 'datetime', 'CS_COD_V', 'CS_TP_V', 'CS_NH3N_V']
    with open(filename, 'r') as fr:
        csv_file = csv.reader(fr)
        out = []
        feats = []
        csv_file = list(csv_file)
        out.append(fts_lst)

        ind = 0
        count = 0
        new_line = [''] * fts_lst.__len__()
        last_time = csv_file[1][0]
        for line in csv_file[1:]:
            cur_time = line[0]
            if cur_time != last_time:
                if count < 3:
                    for i in range(2, new_line.__len__()):
                        if new_line[i] == '':
                            new_line[i] = 'NULL'
                else:
                    count = 0
                new_line[0] = ind
                new_line[1] = last_time
                last_time = cur_time
                out.append(new_line)
                new_line = [''] * fts_lst.__len__()
                ind += 1
            new_line[ft_d[line[1]]] = line[2]
            count += 1


    with open('data/wmd_transf_res.csv', 'w') as fw:
        csv_write = csv.writer(fw)
        csv_write.writerows(out)



### 必须确保每个csv文件记录按时间递增顺序排列，csv_lst和feature_lst都按重要性从前往后排，最后的是出水指标
csv_lst = ['control_factor_1_waterin.csv', 'control_factor_1_mbr6.csv', 'control_factor_1_pump.csv',
           'wmd_transf_res.csv'
           ]

feature_lst = ['JS1_COD_V', 'JS1_TP_V', 'JS1_NH3_V', 'JS1_PH_V',
               'DO1_V', 'DO3_V', 'MLSS_V', 'DO2_V',
               # 'YW_V',
               'JS3_F_V',
               # 'C7_TJF_PLFK','C6_TJF_PLFK','C1_TJF_PLFK',
               # 'CS_COD_V', 'CS_TP_V', 'CS_NH3N_V',
               'w01018', 'w21011', 'w21003',
               # 'CS_PH_V'
               ]



feat_name_lst = ['datetime']  # 一开始特征的顺序为csv文件读入的顺序

# label_lst = ['CS_COD_V', 'CS_TP_V', 'CS_NH3N_V']
# label_lst = ['w01018', 'w21011', 'w21003', 'CS_PH_V'] # 最终文件的标签特征顺序为['CS_PH_V', 'w01018', 'w21011', 'w21003']
label_lst = ['w01018', 'w21011', 'w21003'] # 最终文件的标签特征顺序为['w01018', 'w21011', 'w21003']

# # interpolation_lst = ['C7_TJF_PLFK','C6_TJF_PLFK','C1_TJF_PLFK']
# interpolation_lst = []
interpolation_ind = []
# for i, l in enumerate(feat_name_lst + feature_lst): # 'datatime' + feature_lst为最终csv文件特征的顺序 # 注，有问题
#     if l in interpolation_lst:
#         interpolation_ind.append(i)

if __name__ == '__main__':

    # transform_tablestruct('data/water_mon_data.csv')

    # # 把每张表的数据整合到一起，存入table_lst和first_ind_lst
    # table_lst = [] # 三维，table_lst[i]为一张表的所有内容，table_lst[i][j]为一行特征
    # first_ind_lst = []  # 一维，first_ind_lst[i]为一个字典，存这张表每个时间戳第一次出现的行数
    # for filename in csv_lst:
    #     with open('data/' + filename, 'r') as fr:
    #         csv_file = csv.reader(fr)
    #
    #         feat_ind = [1] # id, datetime, ...
    #         csv_file = list(csv_file)
    #         for i, feat_name in enumerate(csv_file[0]):  # 读取表第一行 特征名
    #             if feat_name in feature_lst:
    #                 feat_ind.append(i)
    #                 feat_name_lst.append(feat_name)
    #
    #         data = np.array(csv_file[1:]) # 去首行
    #         data = data[:, feat_ind]
    #         data = data.tolist()
    #         first_ind = {}
    #         for i in range(data.__len__()):
    #             data[i][0], _ = data_utils.round_seconds(data[i][0])  # 将日期转换为绝对时间(s)
    #             if first_ind.get(data[i][0], -1) == -1:
    #                 first_ind[data[i][0]] = i
    #
    #         table_lst.append(data)
    #         first_ind_lst.append(first_ind)

    # table_lst, first_ind_lst, feature_lst, label_lst = data_prep_200402.extract_tables()
    table_lst, first_ind_lst, feature_lst, label_lst, file_ind = data_prep_200406.extract_xlxs()
    feat_name_lst.extend(feature_lst)
    feat_name_lst.extend(label_lst)

    ### 表连接
    all_data = table_lst[0]
    for i in range(1, table_lst.__len__()):
        for j in range(all_data.__len__()):
            index = first_ind_lst[i].get(all_data[j][0], -1)  # 根据时间戳连接表
            if index != -1:
                all_data[j].extend(table_lst[i][index][1:]) # 第一个是时间戳，所以从1开始
            else:
                success = False
                for delta in [-60, 60, -120, 120]:  # 允许的误差在2min/120s之内
                    index = first_ind_lst[i].get(str(int(all_data[j][0]) + delta), -1)  # 根据时间戳连接表
                    if index != -1:
                        all_data[j].extend(table_lst[i][index][1:])
                        success = True
                        break

                if success == False:  # 时间戳匹配失败
                    all_data[j].extend(['NULL'] * (table_lst[i][0].__len__() - 1))

    ### 20191219 批次特殊处理
    # all_data = data_pprc_191219.fix_water_mon_data(all_data, feat_name_lst)
    # all_data = data_pprc_191219.truncate_data(all_data, '2019-11-01 00:00:00', '2019-12-31 23:59:59')
    # all_data = data_pprc_191219.add_feature_col(all_data, ['ORP1_V', 'JLBA_RUN', 'JLBB_RUN', 'T'],
    #                                             feat_name_lst, label_lst)


    ### 20200406 批次
    if file_ind == 2:
        all_data = data_pprc_191219.add_feature_col(all_data, ['T'], feat_name_lst, label_lst)


    ### 去除缺失值，必须在字段顺序调整前
    # tmp3 = all_data
    # all_data = []
    # for row in tmp3:
    #     tmp4 = list(np.delete(np.array(row), interpolation_ind))  # 需要插值的字段保留缺失值
    #     # if ('NULL' in tmp4) or ('#N/A' in tmp4) or ('' in tmp4) or ('0' in tmp4):
    #     if ('NULL' in tmp4) or ('#N/A' in tmp4) or ('' in tmp4): # --------注意是否将0视为缺失值
    #         pass
    #     else:
    #         all_data.append(row)  # 注意应附加上原行

    ### 字段顺序调整，将标签列接在最后
    label_ind = []
    for i, l in enumerate(feat_name_lst):
        if l in label_lst:
            label_ind.append(i)

    tmp0 = np.array(all_data)[:, label_ind]
    tmp1 = np.array(all_data)
    tmp1 = np.delete(tmp1, label_ind, axis=1)  # 删除标签列
    tmp2 = np.c_[tmp1, tmp0]  # 将标签列接在最后
    all_data = tmp2.tolist()

    tmp0 = np.array(feat_name_lst)[label_ind]
    tmp1 = np.array(feat_name_lst)
    tmp1 = np.delete(tmp1, label_ind, axis=0)  # 删除标签列
    tmp2 = np.r_[tmp1, tmp0]  # 将标签列接在最后
    feat_lst = tmp2.tolist()


    # ### 时间戳形式转换，转换成h，方便预处理时输入输出对齐------------------注意
    # for i in range(all_data.__len__()):
    #     # time_abs = int(all_data[i][0])
    #     # arr_time = time.localtime(time_abs)
    #     # stamp_time = time.strftime('%Y-%m-%d %H:%M', arr_time)
    #     # all_data[i][0] = stamp_time
    #     time_abs = float(all_data[i][0])
    #     time_abs = time_abs / (60 * 60 * 24) # 这里会造成精度损失
    #     all_data[i][0] = str(time_abs)

    for j in range(1, feat_lst.__len__()): ### 第一个为时间戳
        if feat_lst[j] == 'CS_NH3H_V':
            for i in range(all_data.__len__()):
                if all_data[i][j] != 'NULL':
                    all_data[i][j] = str(float(all_data[i][j]) + 0.1)
        data_utils.interpolation_bot(j, all_data)

    clf = IsolationForest(random_state=1)
    # preds = clf.fit_predict(np.array(all_data)[:, 1:feature_lst.__len__()+1]) ## -------注 +1 datetime
    # preds = clf.fit_predict(np.array(all_data)[:, -label_lst.__len__():]) ## -------注
    preds = clf.fit_predict(np.array(all_data)[:, 1:]) ## -------注
    abnormal_inds = np.squeeze(np.argwhere(preds == -1))
    for ind in abnormal_inds:
        # all_data[ind][1:feature_lst.__len__()+1] = ['NULL'] * (feature_lst.__len__()) # 注与上面对应
        # all_data[ind][-label_lst.__len__():] = ['NULL'] * (label_lst.__len__()) # 注与上面对应
        all_data[ind][1:] = ['NULL'] * (feat_lst.__len__()-1) # 注与上面对应
    # for i in range(all_data.__len__()): ## 第一个特征有很多0值，注
    #     if all_data[i][1] == '0':
    #         all_data[i][1] = 'NULL'

    for j in range(1, feat_lst.__len__()): ### 第一个为时间戳
        data_utils.interpolation_bot(j, all_data)


    # ### 标准化
    # ss = StandardScaler()
    # all_data_arr = np.array(all_data)
    # std_all_data = ss.fit_transform(all_data_arr[:, 1:])
    # all_data = np.c_[all_data_arr[:, 0], std_all_data].tolist()

    # ### 画箱型图
    # all_data = np.array(all_data, dtype=np.float32)
    # df = pd.DataFrame(all_data[:, 1:], columns=range(1, feat_lst.__len__()))
    # f = df.boxplot(sym='+',  # 异常点形状
    #                vert=True,  # 是否垂直
    #                whis=1.5,  # IQR
    #                patch_artist=True,  # 上下四分位框是否填充
    #                # meanline=False, showmeans=True,  # 是否有均值线及其形状
    #                showbox=True,  # 是否显示箱线
    #                showfliers=True,  # 是否显示异常值
    #                notch=False,  # 中间箱体是否缺口
    #                return_type='dict')  # 返回类型为字典
    # # plt.title('箱线图', fontproperties=myfont)
    # plt.xlabel('Features')
    # plt.ylabel('Normalized Values')
    # plt.savefig('boxplot.png')
    # plt.show()
    # all_data = all_data.tolist()

    ### 存入txt文件
    # with open('data/20200406data_{}.txt', 'w') as fw: # ---------------------注
    with open('data/20200406data_{}.txt'.format(file_ind), 'w') as fw: # ---------------------注
        first_row = '\t'.join(feat_lst) + '\n'
        text = [first_row]
        for data in all_data:
            text.append('\t'.join(data) + '\n')

        fw.writelines(text)

    ### 存入csv文件，要在存入txt文件之后
    with open('data/all_data.csv', 'w') as fw:

        for i in range(all_data.__len__()):  # 转换时间戳形式
            time_abs = int(all_data[i][0])
            # arr_time = time.localtime(time_abs * (60 * 60 * 24))
            arr_time = time.localtime(time_abs)
            stamp_time = time.strftime('%Y-%m-%d %H:%M', arr_time)
            all_data[i][0] = stamp_time

        csv_write = csv.writer(fw)
        csv_write.writerow(feat_lst)
        csv_write.writerows(all_data)

    # print()
