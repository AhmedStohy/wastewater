import csv
import numpy as np
import time
import os, sys
import data_utils

from copy import deepcopy


def connect_interpolate(fix_data, prc_data, fix_ind):
    first_ind = {}
    for k in range(fix_data.__len__()):
        fix_data[k][0], _ = data_utils.round_seconds(fix_data[k][0])  # 将日期转换为绝对时间(s)
        if first_ind.get(fix_data[k][0], -1) == -1:
            first_ind[fix_data[k][0]] = k

    first = True
    last_ind = None
    for j in range(prc_data.shape[0]):
        success = False
        index = first_ind.get(prc_data[j][0], -1)  # 根据时间戳连接表
        if index != -1:
            prc_data[j][fix_ind] = deepcopy(fix_data[index][1:])
            success = True
        else:
            for delta in [-60, 60, -120, 120]:  # 允许的误差在2min/120s之内
                index = first_ind.get(str(int(prc_data[j][0]) + delta), -1)  # 根据时间戳连接表
                if index != -1:
                    prc_data[j][fix_ind] = deepcopy(fix_data[index][1:])
                    success = True
                    break

        ### 预测指标插值，平均插值，头和尾缺失部分都分别等于第一个和最后一个准确值
        if success == True:
            if first == True: # 处理头部
                prc_data[:j, fix_ind] = deepcopy(fix_data[index][1:])
                first = False
            else:
                inter_num = j - last_ind
                incre_val = (prc_data[j][fix_ind].astype(np.float32) -
                             prc_data[last_ind][fix_ind].astype(np.float32)) / inter_num
                for k in range(last_ind + 1, j):
                    prc_data[k][fix_ind] = (prc_data[last_ind][fix_ind].astype(np.float32) +
                                            (k - last_ind) * incre_val).astype(np.str)
            last_ind = j
        if j == prc_data.shape[0] - 1 and last_ind is not None: # 处理尾部
            prc_data[last_ind:, fix_ind] = deepcopy(prc_data[last_ind][fix_ind])


def fix_water_mon_data(all_data, feat_name_lst):
    all_data = np.array(all_data)
    fix_lst = ['w01018', 'w21011', 'w21003']
    fix_ind = []
    for i, l in enumerate(feat_name_lst):
        if l in fix_lst:
            fix_ind.append(i)
    with open('data/拍照数据.csv') as fr:
        csv_file = list(csv.reader(fr))
        fix_data = np.array(csv_file[1:]) # skip_head=1

    connect_interpolate(fix_data, all_data, fix_ind)

    return all_data.tolist()


def truncate_data(all_data, time_from, time_to):
    all_data = np.array(all_data)
    tf = int(data_utils.round_seconds(time_from)[0])
    tt = int(data_utils.round_seconds(time_to)[0])
    tss = deepcopy(all_data[:, 0].astype(np.int64))
    start_ind = (1 - (tss >= tf)).sum()
    end_ind_plus1 = (tss <= tt).sum()
    all_data = all_data[start_ind: end_ind_plus1, :]
    return all_data.tolist()


def add_feature_col(all_data, feat_to_add, feat_name_lst, label_lst):
    all_data = np.array(all_data)
    label_start_ind = -1
    for i, l in enumerate(feat_name_lst):
        if l in label_lst:
            label_start_ind = i
            break
    add_data = np.zeros([all_data.shape[0], feat_to_add.__len__()], dtype=np.float32)
    for i, feat in enumerate(feat_to_add): # 'ORP1_V', 'JLBA_RUN', 'JLBB_RUN', 'T'
        if feat == 'ORP1_V':
            val = -400
            add_data[:, i] = val
        elif feat == 'JLBA_RUN': # 三氯化铁
            val = 92
            add_data[:, i] = val
        elif feat == 'JLBB_RUN': # 冰醋酸
            val = 0
            add_data[:, i] = val
        elif feat == 'T':
            T_files = ['data/temp/wuxi_2019.txt', 'data/temp/wuxi_2020.txt']
            # save_file = temp_file[:temp_file.rfind('.')] + '_pp' + temp_file[temp_file.rfind('.'):]
            T_data = []
            for T_f in T_files:
                with open(T_f, 'r') as fr:
                    for line in fr.readlines():
                        date, minT, maxT = line.strip().split(',')

                        ### 转成round_seconds输入格式
                        date = '{}-{}-{}'.format(date[:4], date[4:6], date[6:])
                        date_minT = date + ' 02:00:00'
                        date_maxT = date + ' 14:00:00'
                        T_data.extend([[date_minT, minT], [date_maxT, maxT]])

            fix_data = T_data # 第一列时间戳(日期格式，待转换成s)，之后为将填补到其他地方的数据
            prc_data = np.c_[all_data[:, 0], add_data] # 加上时间戳(s)，prc_data为填补的目标
            fix_ind = [i + 1] # 填补的目标位置，由于加了一列时间戳，因此 +1

            connect_interpolate(fix_data, prc_data, fix_ind)

            add_data = prc_data[:, 1:] # 去掉第一列时间戳
        feat_name_lst.insert(label_start_ind + i, feat)
    tmp0, tmp1 = np.split(all_data, [label_start_ind], axis=1)
    all_data = np.c_[tmp0, add_data, tmp1]
    return all_data.tolist()