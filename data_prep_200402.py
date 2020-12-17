import numpy as np
import re
import xlrd
import time
import csv

import data_utils

from copy import deepcopy

def extract_tables():

    feature_lst = ['Elevation_Pump.JS1_TP_V', 'Elevation_Pump.JS1_PH_V', 'Elevation_Pump.JS1_NH3_V',
                   'Elevation_Pump.JS1_COD_V', 'Elevation_Pump.JS3_F_V',
                   'MSBR6.DO1_V', 'MSBR6.DO2_V', 'MSBR6.DO3_V', 'MSBR6.MLSS_V',
                   'MSBR6.ORP1_V', 'MSBR6.ORP2_V',
                   ]

    label_lst = ['CSYBJ.CS_PH_V', 'CSYBJ.CS_NH3N_V', 'CSYBJ.CS_COD_V',
                 'CSYBJ.CS_TN_V']

    table_features = [['Elevation_Pump.JS1_TP_V', 'Elevation_Pump.JS1_PH_V', 'Elevation_Pump.JS1_NH3_V',
                       'Elevation_Pump.JS1_COD_V', 'Elevation_Pump.JS3_F_V'],
                      ['MSBR6.DO1_V', 'MSBR6.DO2_V', 'MSBR6.DO3_V', 'MSBR6.MLSS_V',
                       'MSBR6.ORP1_V', 'MSBR6.ORP2_V'],
                      ['CSYBJ.CS_PH_V', 'CSYBJ.CS_NH3N_V', 'CSYBJ.CS_COD_V'],
                      ['CSYBJ.CS_TN_V']]

    xlxss = ['data/四期进水.xlsx', 'data/四期状态2.xlsx', 'data/二厂进水COD氨氮pH.xlsx', 'data/二厂进水总氮.xlsx']
    need_inds = [{1 : 'Elevation_Pump.JS1_TP_V', 2 : 'Elevation_Pump.JS1_PH_V', 4 : 'Elevation_Pump.JS1_NH3_V',
                  8: 'Elevation_Pump.JS1_COD_V', 9 : 'Elevation_Pump.JS3_F_V'},
                 {15 : 'MSBR6.DO1_V', 16 : 'MSBR6.DO2_V', 17 : 'MSBR6.DO3_V', 18 : 'MSBR6.MLSS_V',
                  19 : 'MSBR6.ORP1_V', 20 : 'MSBR6.ORP2_V'},
                 {3 : 'CSYBJ.CS_PH_V', 4 : 'CSYBJ.CS_NH3N_V', 6 : 'CSYBJ.CS_COD_V'},
                 {6: 'CSYBJ.CS_TN_V'}]


    table_datas = []
    datetime2inds = []
    for ind in range(xlxss.__len__()):
        data = xlrd.open_workbook(xlxss[ind])
        # tables = ['TOTAL_DATA_5', 'TOTAL_DATA_5', 'TOTAL_DATA_12']
        need_ind = need_inds[ind]
        field_ind = 1
        field2ind = {'datetime' : 0}

        datetime2ind = {}
        datetime_ind = 0
        datetime_pat = re.compile('Sample_TDate')
        value_pat = re.compile('Sample_Value')

        table_data = []
        # for t in tables:
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
                if need_ind.get(cur_row[0], None) is not None:
                    if field2ind.get(need_ind[cur_row[0]], None) is not None:
                        col = field2ind[need_ind[cur_row[0]]]
                    else:
                        field2ind[need_ind[cur_row[0]]] = field_ind
                        col = field_ind
                        field_ind += 1
                else:
                    continue

                for time_value in row_value_inds:
                    if not cur_row[time_value[0]] or not cur_row[time_value[1]]:
                        continue
                    tmp_dt = xlrd.xldate_as_tuple(cur_row[time_value[0]], 0)
                    dt = '{}-{}-{} {}:{}:{}'.format(tmp_dt[0], tmp_dt[1], tmp_dt[2], tmp_dt[3], tmp_dt[4], tmp_dt[5])
                    dt = data_utils.round_seconds(dt)[0]
                    if datetime2ind.get(dt, None) is None:
                        datetime2ind[dt] = datetime_ind
                        datetime_ind += 1  ### 注---这里假设了不包括重复的时刻，否则应等于row_ind左右
                        tmp_row = ['NULL'] * (need_ind.__len__() + 1) # +1 因为 datetime
                        tmp_row[0] = dt
                        table_data.append(tmp_row)

                    row = datetime2ind[dt]
                    table_data[row][col] = '{:.9f}'.format(cur_row[time_value[1]]) ### 注意这里会产生不必要的空格

        table_datas.append(deepcopy(table_data))
        datetime2inds.append(datetime2ind)

        with open('data/200402_{}.csv'.format(ind), 'w') as fw:

            for i in range(table_data.__len__()):  # 转换时间戳形式
                time_abs = float(table_data[i][0])
                arr_time = time.localtime(time_abs)
                stamp_time = time.strftime('%Y-%m-%d %H:%M', arr_time)
                table_data[i][0] = stamp_time

            csv_write = csv.writer(fw)
            csv_write.writerow(['datetime'] + table_features[ind])
            csv_write.writerows(table_data)


    return table_datas, datetime2inds, feature_lst, label_lst


if __name__ == '__main__':
    extract_tables()
    print()