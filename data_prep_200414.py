import numpy as np
import re
import xlrd
import time
import csv

import data_utils
import data_prep_191219

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from copy import deepcopy

def extract_tables():

    feature_lst = ['Elevation_Pump.JS1_COD_V', 'Elevation_Pump.JS1_NH3_V',
                   'Elevation_Pump.JS1_TP_V', 'Elevation_Pump.JS1_PH_V']
    label_lst = deepcopy(feature_lst)

    need_inds = {8: 'Elevation_Pump.JS1_COD_V', 4: 'Elevation_Pump.JS1_NH3_V',
                 1: 'Elevation_Pump.JS1_TP_V', 2: 'Elevation_Pump.JS1_PH_V'}

    field2ind = {'datetime': 0, 'Elevation_Pump.JS1_COD_V': 1, 'Elevation_Pump.JS1_NH3_V': 2,
                 'Elevation_Pump.JS1_TP_V': 3, 'Elevation_Pump.JS1_PH_V': 4}

    # start_date = '2016-12-7 6:0:0' ### 本来应是 2016-12-7 8:43:42
    # end_date = '2020-4-14 0:0:0'

    start_date = '2018-01-01 00:00:00'
    end_date = '2019-01-01 00:00:00'

    interval = 1 ### 间隔为h

    # table_data = []
    # date2ind = {}
    # date_ind = 0

    start = int(data_utils.round_seconds(start_date)[0]) ## 单位为s
    end = int(data_utils.round_seconds(end_date)[0])

    oriCount = -1
    curCount = -1
    maxCount = -1
    maxIncre = -1
    # for ind, incre in enumerate(range(0, 60 * 60 * interval, 60)):
    incre = 600 # 时间戳整体向后偏移600s
    # incre = 0
    cur = start + incre

    table_data = []
    date2ind = {}
    date_ind = 0

    while cur < end:
        tmpRow = ['NULL'] * (1 + feature_lst.__len__())
        tmpRow[0] = str(cur)
        date2ind[cur] = date_ind # data2ind[d]为日期d在table_data中的行号
        date_ind += 1
        table_data.append(tmpRow)

        cur += int((60 * 60) * interval)

    oriCount = table_data.__len__()
    curCount = oriCount

    visited = np.zeros(table_data.__len__())

    for file_ind in range(1, 7):
        # if file_ind > 1:
        #     break
        xls = 'data/进水历史数据/jinshuishuju{}.xls'.format(file_ind)
        data = xlrd.open_workbook(xls)

        datetime_pat = re.compile('Sample_TDate')
        value_pat = re.compile('Sample_Value')

        for sname in data.sheet_names():
            table = data.sheet_by_name(sname)
            if table.nrows < 100: ##### 小于100的一般不存数据
                continue

            first_row = table.row_values(0)
            time_value = []
            row_value_inds = []
            for i, field in enumerate(first_row):
                if datetime_pat.match(field) is not None:
                    time_value.append(i)
                elif value_pat.match(field) is not None:
                    time_value.append(i)
                    row_value_inds.append(deepcopy(time_value))
                    time_value.clear()

            for row_ind in range(1, table.nrows):
                cur_row = table.row_values(row_ind)
                col = -1
                if need_inds.get(cur_row[0], None) is not None:
                    if field2ind.get(need_inds[cur_row[0]], None) is not None:
                        col = field2ind[need_inds[cur_row[0]]]
                else:
                    continue

                for time_value in row_value_inds:
                    if not cur_row[time_value[0]] or not cur_row[time_value[1]]:
                        continue
                    tmp_dt = xlrd.xldate_as_tuple(cur_row[time_value[0]], 0)
                    dt = '{}-{}-{} {}:{}:{}'.format(tmp_dt[0], tmp_dt[1], tmp_dt[2], tmp_dt[3], tmp_dt[4], tmp_dt[5])
                    dt = int(data_utils.round_seconds(dt)[0])
                    if start - dt > 60 * 60 or dt - end > 60 * 60:
                        continue
                    row = -1
                    if date2ind.get(dt, None) is not None:
                        row = date2ind[dt]
                        table_data[row][col] = str(cur_row[time_value[1]]) ### 注意这里可能会产生不必要的空格
                    else:
                        for delta in range(60, 60 * 61, 60): ### 60min容忍度的模糊匹配
                            # dt = int(dt)
                            if date2ind.get(dt + delta, None) is not None:
                                row = date2ind[dt + delta]
                            elif date2ind.get(dt - delta, None) is not None:
                                row = date2ind[dt - delta]
                            if(table_data[row][col] == 'NULL'):
                                table_data[row][col] = str(cur_row[time_value[1]])

                    if row != -1 and not visited[row]:
                        curCount -= 1
                        visited[row] = 1

    # if oriCount - curCount > maxCount:
    #     maxCount = oriCount - curCount
    #     maxIncre = incre
    print('缺失值比例: {}/{}'.format(curCount, oriCount), '时间戳偏移:', incre)
    #
    # print(maxCount, maxIncre)

    if table_data.__len__() == 0:
        print('no data extracted')
        exit(1)
    # offset = 0
    # curInd = 0
    # while curInd < table_data.__len__():
    #     isNull = True
    #     for col in range(1, table_data[0].__len__()):
    #         if table_data[curInd][col] != 'NULL':
    #             isNull = False
    #             break
    #     if isNull:
    #         date2ind.pop(int(table_data[curInd][0]))
    #         offset += 1
    #     else:
    #         table_data = table_data[curInd:]
    #         break
    #     curInd += 1
    #
    # curInd = table_data.__len__()-1
    # while curInd >= 0:
    #     isNull = True
    #     for col in range(1, table_data[0].__len__()):
    #         if table_data[curInd][col] != 'NULL':
    #             isNull = False
    #             break
    #     if isNull:
    #         date2ind.pop(int(table_data[curInd][0]))
    #     else:
    #         table_data = table_data[:curInd+1]
    #         break
    #     curInd -= 1
    #
    # for row in range(table_data.__len__()):
    #     label_date = int(table_data[row][0]) + (60 * 60) * 24
    #     if date2ind.get(label_date + 60 * 60 * 4, None) is not None:
    #         if date2ind[label_date + 60 * 60 * 4] - offset >= table_data.__len__():
    #             break
    #         for label_incre in range(0, 60 * 60 * 6, 60 * 60 * 2):
    #             # try:
    #             table_data[row].extend(deepcopy(table_data[date2ind[label_date+label_incre]-offset][1:])) ### 注意第一个是datetime
    #             # except:
    #             #     print(row, table_data.__len__(), label_date, label_incre, date2ind[label_date+label_incre])
    #     else:
    #         break


    # table_data = data_pprc_191219.truncate_data(table_data, '2018-03-01 00:00:00', '2018-09-01 00:00:00')

    ### 平均插值
    for ind in range(1, feature_lst.__len__()+1):
        data_utils.interpolation_bot(ind, table_data)

    table_data_arr = np.array(table_data)
    has_outlier = np.array([False] * (feature_lst.__len__() + 1), np.bool) # 某维是否有异常值的标志；第一维datetime

    ### 去掉先验异常值
    indicator = 1 ### COD
    target_data = np.array(table_data_arr[:, indicator], dtype=np.float32)
    outlier_inds = np.squeeze(np.argwhere(target_data < 10)) ### 去掉小于10的
    table_data_arr[outlier_inds, indicator] = 'NULL'
    has_outlier[indicator] = True
    indicator = 3  ### TP
    target_data = np.array(table_data_arr[:, indicator], dtype=np.float32)
    outlier_inds = np.squeeze(np.argwhere(target_data < 0.1))  ### 去掉小于0.1的
    table_data_arr[outlier_inds, indicator] = 'NULL'
    has_outlier[indicator] = True

    # ### 1.5σ 检测各维度异常值，每维度单独检测
    # # for ind in range(1, feature_lst.__len__() + 1):
    # for ind in [1]:
    #     target_data = np.array(table_data_arr[:, ind], dtype=np.float32) ## 注意类型转换
    #     miu, sigma = np.mean(target_data), np.std(target_data, ddof=1)
    #     print(miu, sigma)
    #     outlier_inds = np.r_[np.argwhere(target_data < miu - 1.5 * sigma), np.argwhere(target_data > miu + 1.5 * sigma)]
    #     outlier_inds = np.reshape(outlier_inds, [-1])
    #     table_data_arr[outlier_inds, ind] = 'NULL'
    #     has_outlier[ind] = True
    #
    # # ### 每指标单独做孤立树
    # # for ind in range(1, feature_lst.__len__() + 1):
    # #     clf = IsolationForest(random_state=1, contamination=0.02)
    # #     # preds = clf.fit_predict(np.array(all_data)[:, 1:1+feature_lst.__len__()]) ## -------注 +1 datetime
    # #     preds = clf.fit_predict(table_data_arr[:, ind:ind+1])  ## -------注
    # #     outlier_inds = np.squeeze(np.argwhere(preds == -1))
    # #     table_data_arr[outlier_inds, ind] = 'NULL'
    # #     has_outlier[ind] = True

    table_data = table_data_arr.tolist()
    # for ind in range(1, feature_lst.__len__() + 1): ### 注
    for ind in range(1, 1 + feature_lst.__len__()):
        if has_outlier[ind]:
            data_utils.interpolation_bot(ind, table_data)
    table_data_arr = np.array(table_data)

    ### 标准化
    do_standardize = False
    if do_standardize:
        # table_data_arr = np.array(table_data_arr, np.float32)
        ss = StandardScaler()
        # ss.scale_ = np.std(table_data_arr[:, 1:], ddof=1)
        std_all_data = ss.fit_transform(np.array(table_data_arr[:, 1:], np.float32))
        table_data_arr[:, 1:] = np.array(std_all_data, np.str)
        # table_data_arr = np.array(table_data_arr, np.str)
        table_data = table_data_arr.tolist()

    input_step = 24
    pred_step = 96
    is_single_var = False # True: univar   False: multivar
    indicators = [1, 2, 3] # COD NH3N TP
    # indicators = [2] # COD NH3N TP
    ind_names = {1:'COD', 2:'NH3N', 3:'TP'}
    for target_ind in indicators:
        if is_single_var:
            indicator = [target_ind] #待遇测指标
        else:
            indicator = indicators

        all_data = []
        for row in range(table_data_arr.shape[0]-input_step-pred_step+1):
            tmpdata = np.reshape(table_data_arr[row : row+input_step, indicator], [-1])
            tmplabel = np.reshape(table_data_arr[row+input_step : row+input_step+pred_step, indicator], [-1])
            tmpdata = np.r_[tmpdata, tmplabel]
            # for ind in indicator:
            #     tmpdata = np.r_[tmpdata, table_data_arr[row+input_step : row+input_step+pred_step, ind]]
            all_data.append(tmpdata)

        all_data = np.array(all_data)
        if is_single_var:
            save_file = 'data/20200414data{}_H_{}seqlen_{}predlen_ok.txt'.format(ind_names[target_ind],
                                                                                 input_step, pred_step)
        else:
            save_file = 'data/20200414data_H_{}seqlen_{}predlen_ok.txt'.format(input_step, pred_step)
        if do_standardize:
            save_file = save_file[:save_file.rfind('_')] + '_std_ok.txt'
        np.savetxt(save_file, all_data, fmt='%s', delimiter='\t')

        if not is_single_var:
            break


    # ### 存入txt文件
    # with open('data/20200414data_H.txt', 'w') as fw:  # ---------------------注
    #     first_r = ['datetime'] + feature_lst
    #     first_r = '\t'.join(first_r) + '\n'
    #     text = [first_r]
    #     for data in table_data:
    #         text.append('\t'.join(data) + '\n')
    #
    #     fw.writelines(text)
    #
    # save_file = 'data/进水历史数据/jinshuishuju_H_平均插值_去除异常COD&TP.csv'
    #
    # for i in range(table_data.__len__()):
    #     time_abs = int(table_data[i][0])
    #     # arr_time = time.localtime(time_abs * (60 * 60 * 24))
    #     arr_time = time.localtime(time_abs)
    #     stamp_time = time.strftime('%Y-%m-%d %H:%M', arr_time)
    #     table_data[i][0] = stamp_time
    #
    # with open(save_file, 'w') as fw:
    #     csv_write = csv.writer(fw)
    #     first_r = ['datetime'] + feature_lst
    #     csv_write.writerow(first_r)
    #     csv_write.writerows(table_data)


if __name__ == '__main__':
    extract_tables()

