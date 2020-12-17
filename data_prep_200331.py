import numpy as np
import csv
import sys

import data_utils

feature_lst = ['datetime',
               'Elevation Pump.JS1_COD_V', 'Elevation Pump.JS1_TP_V', 'Elevation Pump.JS1_NH3_V', 'Elevation Pump.JS1_PH_V',
               'MSBR6.DO1_V', 'MSBR6.DO3_V', 'MSBR6.MLSS_V', 'MSBR6.DO2_V',
               'MSBR6.ORP2_V', 'MSBR6.ORP3_V',
               # 'YW_V',
               'Elevation Pump.JS3_F_V',
               # 'C7_TJF_PLFK','C6_TJF_PLFK','C1_TJF_PLFK',
               'CSYBJ.CS_COD_V',
               # 'CSYBJ.CS_TP_V',
               'CSYBJ.CS_NH3N_V',
               # 'w01018', 'w21011', 'w21003',
               'CSYBJ.CS_PH_V'
               ]


def generateTable(filename):

    csv_file = None
    with open(filename, 'r') as fr:
        csv_file = csv.reader(fr)
        csv_file = list(csv_file)

    datetime_ind = 5
    tag_id_ind = 2
    tag_value_ind = 3

    datetime2ind = {}
    all_data = []
    # last_datetime = '-1'
    row_id = 0

    for ind, line in enumerate(csv_file[1:]):
        _, cur_datetime = data_utils.round_seconds(line[datetime_ind])
        if datetime2ind.get(cur_datetime, None) is None:
            datetime2ind[cur_datetime] = row_id
            row_id += 1
            # last_datetime = cur_datetime
            tmpRow = ['NULL'] * feature_lst.__len__()
            tmpRow[0] = cur_datetime
            all_data.append(tmpRow)

        sys.stdout.write('\r{} / {}'.format(ind, csv_file.__len__()))
        sys.stdout.flush()

    feature2ind = {}
    ind = 0
    for feat in feature_lst:
        feature2ind[feat] = ind
        ind += 1

    for int, line in enumerate(csv_file[1:]):
        if(feature2ind.get(line[tag_id_ind], None)):
            dt, tag_id, tag_value = data_utils.round_seconds(line[datetime_ind])[1], line[tag_id_ind], line[tag_value_ind]
            all_data[datetime2ind[dt]][feature2ind[tag_id]] = tag_value

        sys.stdout.write('\r{} / {}'.format(ind, csv_file.__len__()))
        sys.stdout.flush()

    all_data = [feature_lst] + all_data
    return all_data


if __name__ == '__main__':
    # all_data = generateTable('data/opchisdata_0331_small.csv')
    all_data = generateTable('data/opchisdata_0331.csv')

    # with open('data/tmp.csv', 'w') as fw:
    with open('data/opchisdata_0331_res.csv', 'w') as fw:
        csv_write = csv.writer(fw)
        csv_write.writerows(all_data)