import numpy as np
import xlrd
import time
import csv

import data_utils

from copy import deepcopy


def extract_xlxs():

    xlsxes = ['data/需要连接的数据/二厂数据1.xlsx', 'data/需要连接的数据/二厂数据2.xlsx', 'data/需要连接的数据/四期数据1.xlsx',
              'data/需要连接的数据/四期数据2.xlsx']

    feature_lst = [['CSYBJ.CS_NH3N_V', 'CSYBJ.CS_TN_V', 'CSYBJ.CS_COD_V', 'CSYBJ.CS_PH_V', 'LLJ1', 'WD', 'ORP', 'JYSD1'],
                   ['CSYBJ.CS_NH3N_V', 'CSYBJ.CS_TN_V', 'CSYBJ.CS_COD_V', 'CSYBJ.CS_PH_V', 'LLJ1', 'WD', 'ORP', 'JYSD1'],
                   ['JS1_COD_V', 'JS1_NH3_V', 'JS1_TP_V', 'JS1_PH_V', 'JS3_F_V', 'DO2_V', 'DO1_V', 'DO3_V', 'MLSS_V',
                    'JLBB', 'JLBA'],
                   ['JS1_COD_V', 'JS1_NH3_V', 'JS1_TP_V', 'JS1_PH_V', 'JS3_F_V', 'DO2_V', 'DO1_V', 'DO3_V', 'MLSS_V',
                    'JLBB', 'JLBA']
                   ]

    lable_lst = [['AD', 'ZD', 'COD'], ['AD', 'ZD', 'COD'], ['CS_COD_V', 'CS_NH3H_V', 'CS_TP_V'],
                 ['CS_COD_V', 'CS_NH3H_V', 'CS_TP_V']]

    # file_ind = [2, 3] ### ------------注
    file_ind = [0, 1] ### ------------注

    table_datas = []
    datetime2inds = []

    pad_count = 0

    for fi, f_ind in enumerate(file_ind):
        data = xlrd.open_workbook(xlsxes[f_ind])

        for ind, sname in enumerate(data.sheet_names()): ## 需要各文件各表前后顺序一致
            table = data.sheet_by_name(sname)

            if fi == 0:
                table_data = []
                datetime2ind = {}
                datetime_ind = 0
            else:
                table_data = table_datas[ind]
                datetime2ind = datetime2inds[ind]
                datetime_ind = datetime2ind.__len__()

            last_dt = -1
            for row_ind in range(2, table.nrows): ### 前两行是字段名
                cur_row = table.row_values(row_ind)

                ### 时间戳格式转换
                tmp_dt = xlrd.xldate_as_tuple(cur_row[0], 0)
                dt = '{}-{}-{} {}:{}:{}'.format(tmp_dt[0], tmp_dt[1], tmp_dt[2], tmp_dt[3], tmp_dt[4], tmp_dt[5])
                dt = int(data_utils.round_seconds(dt)[0]) # 注

                ### 特殊处理
                if dt == last_dt:
                    if row_ind == table.nrows - 1:
                        dt += 60
                    else:
                        tmp_dt = xlrd.xldate_as_tuple(table.row_values(row_ind+1)[0], 0)
                        next_dt = '{}-{}-{} {}:{}:{}'.format(tmp_dt[0], tmp_dt[1], tmp_dt[2], tmp_dt[3], tmp_dt[4], tmp_dt[5])
                        next_dt = int(data_utils.round_seconds(next_dt)[0])  # 注
                        if next_dt - dt > 60: ### 注意可能不止 120 !!!
                            dt += 60

                ### 注意是对四期数据中的入水数据的特殊处理，后来觉得二期数据也要如此处理
                # if (f_ind == 2 or f_ind == 3) and ind == 0:
                if ind == 0:
                    if last_dt != -1 and dt - last_dt > 60 and dt - last_dt <= 60 * 60 * 4: ### 允许的最大缺失时间长度为4h
                        for pad_dt in range(last_dt+60, dt, 60):
                            if f_ind == 3 and pad_dt % 300 != 0:
                                continue

                            pad_row = ['NULL'] * cur_row.__len__()
                            pad_row[0] = str(pad_dt)
                            datetime2ind[str(pad_dt)] = datetime_ind  ### 注---这里假设了不包括重复的时刻，否则应等于row_ind左右
                            datetime_ind += 1
                            table_data.append(pad_row)
                            pad_count += 1

                last_dt = dt

                ### assert 数据的时间戳都为整5分钟

                ### 注意是对四期数据的特殊处理
                if f_ind == 3:
                    if dt % 300 != 0:
                        continue

                dt = str(dt) ## 注
                datetime2ind[dt] = datetime_ind ### 注---这里假设了不包括重复的时刻，否则应等于row_ind左右
                datetime_ind += 1
                cur_row[0] = dt

                for ii in range(cur_row.__len__()):
                    if cur_row[ii] == 42:  ### 注意在excel中不能用 #N/A，xlrd 读出来是 42
                        cur_row[ii] = 'NULL'

                table_data.append(cur_row)

            if fi == 0:
                table_datas.append(deepcopy(table_data))
                datetime2inds.append(datetime2ind)

            if fi == file_ind.__len__()-1:
                table_data = deepcopy(table_data)
                with open('data/200406_{}_{}.csv'.format(file_ind[0], ind), 'w') as fw:

                    for i in range(table_data.__len__()):  # 转换时间戳形式
                        time_abs = float(table_data[i][0])
                        arr_time = time.localtime(time_abs)
                        stamp_time = time.strftime('%Y-%m-%d %H:%M', arr_time)
                        table_data[i][0] = stamp_time

                    csv_write = csv.writer(fw)
                    if ind == 0:
                        csv_write.writerow(['datetime'] + feature_lst[file_ind[0]])
                    else:
                        csv_write.writerow(['datetime'] + lable_lst[file_ind[0]])
                    csv_write.writerows(table_data)

    print('pad_count:', pad_count)
    return table_datas, datetime2inds, feature_lst[file_ind[0]], lable_lst[file_ind[0]], file_ind[0]


if __name__ == '__main__':

    extract_xlxs()
    print()