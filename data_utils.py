import numpy as np
import time
from copy import deepcopy

# from table_connect import interpolation_ind, feature_lst, label_lst


def round_seconds(timestamp):
    '''
    :param timestamp: str
    :return: str  time_abs(seconds)
    '''
    time_arr = ''

    try:
        time_arr = time.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    except:
        # print(timestamp)
        try:
            time_arr = time.strptime(timestamp, '%Y-%m-%d %H:%M')
        except:
            time_arr = time.strptime(timestamp, '%Y/%m/%d %H:%M:%S')
    time_abs = time.mktime(time_arr)
    time_abs = round(time_abs / 60) * 60
    arr_time = time.localtime(time_abs)
    stamp_time = time.strftime('%Y-%m-%d %H:%M', arr_time)
    return (str(time_abs), stamp_time)


def binary_search(target, datalst):
    '''
    :param target: number int
    :param datalst: sorted number list int
    :return: equal or right larger than target or right smaller than target
    '''
    if not datalst:
        return -1
    begin = 0
    end = datalst.__len__()-1
    # mid = 0
    while begin <= end:
        mid = (begin + end) // 2
        if datalst[mid] == target:
            return mid
        elif datalst[mid] < target:
            begin = mid + 1
        else:
            end = mid - 1
    return min(begin, datalst.__len__()-1)


def find_label_1n(times, time_incre):
    '''
    :param times :dict，单位为秒-------------注意
    :return: dict of key int, value int
    '''
    time_increment = time_incre * 60  # 单位为s，timeincre 为min  --------------注意
    data2label = {}
    labels = list(map(int, list(times.keys()))) # 转换为int，避免str下相同值判断为不同，要拿字典键值做比较时，键最好是数值类型
    labels = sorted(labels) # list
    leng = labels.__len__()
    # dist = np.ones([leng, leng]) * float('inf')
    for i in range(leng):
        out_time = (labels[i] + time_increment)
        # labels_h = deepcopy(list(np.array(labels) * 24)) # element-wised, 单位都转换为h，避免除的运算，方便比较
        res = binary_search(out_time, labels)
        # res = times.get(out_time, -1)
        if res != -1:
            if abs(labels[res] - out_time) < 3 * 60: # 误差不超过3min，*60使得单位为分钟
                data2label[labels[i]] = labels[res]

    return data2label


def interpolation_bot(feat_id, dataset):
    '''
    :param feat_id:
    :param dataset: first data must not NULL
    :return: no return 平均插值
    '''

    former_id = 0
    former = -1.0
    count = 1
    leng = dataset.__len__()

    # if dataset[0][feat_id] == '#N/A' or dataset[0][feat_id] == 'NULL' or dataset[0][feat_id] == '':
    success = False
    for i in range(1, leng):
        if dataset[i][feat_id] != '#N/A' and dataset[i][feat_id] != 'NULL' and dataset[0][feat_id] != '':
            dataset[0][feat_id] = dataset[i][feat_id]
            success =True
            break
    if success == False:
        raise ValueError('A column cannot be none')


    for i in range(leng):
        # try:
        if dataset[i][feat_id] != '#N/A' and dataset[i][feat_id] != 'NULL' and dataset[i][feat_id] != '':
            if count > 1:
                increment = (float(dataset[i][feat_id]) - former) / float(count)
                for j in range(1, count):
                    dataset[former_id + j][feat_id] = str(former + increment * j)
                former_id = i
                former = float(dataset[i][feat_id])
                count = 1
            else:
                former_id = i
                former = float(dataset[i][feat_id])
        else:
            count += 1
        # except:
        #     print(i, feat_id, np.array(dataset).shape)

    if count > 1:
        for j in range(1, count):
            dataset[former_id + j][feat_id] = str(former)


def process_raw_data(filename_in, filename_out, seq_len, label_num, time_incre):
    '''
    :param filename_in: str
    :param filename_out: str
    :param seq_len: int
    :return: list
    '''
    train_tmp = []
    label_tmp = []
    train = []
    label = []
    times = []
    time_series = {} # 用字典可以去重，python3.6之后字典可记录插入顺序，相当于OrderedDict
    add24h_map = {}
    first_index = {}

    # with open('20190408data.txt', 'r') as fr:
    with open(filename_in, 'r') as fr:
        fr.readline() # 第一行为字段名称
        for i, line in enumerate(fr.readlines()):
            line = line.strip()
            splt = line.split('\t')
            if not splt or splt[0] == '':
                break
            if filename_in[filename_in.rfind('/') + 1:] == '20190408data.txt':
                times.append(splt[2])  # 每个样本的时间戳
                time_series[splt[2]] = time_series.get(splt[2], 0) + 1 # 不需要字典，用列表存keys就可以了
                first_index[int(splt[2])] = i # 实际上应该是last_index，但其他表没有时间重复的记录，因此last_index = first_index
                train_tmp.append(splt[3:-4]) # 4为样本标签个数
                label_tmp.append(splt[-4:])
            else:
                times.append(splt[0])  # 每个样本的时间戳
                time_series[splt[0]] = time_series.get(splt[0], 0) + 1 # 不需要字典，用列表存keys就可以了
                first_index[int(splt[0])] = i # 实际上应该是last_index，但其他表没有时间重复的记录，因此last_index = first_index
                train_tmp.append(splt[1:-label_num]) # 第一个为时间戳
                label_tmp.append(splt[-label_num:])

    add24h_map = find_label_1n(time_series, time_incre) # time_series 以秒为单位

    # ## contatenate sequence and match label
    # has_label = list(map(int, list(add24h_map.keys()))) # 转换为int，避免str下相同值判断为不同
    # has_label = sorted(has_label)

    assert train_tmp.__len__() == label_tmp.__len__()
    for i in range(train_tmp.__len__()):
        # if binary_search(int(times[i]), has_label) != -1: # != -1 代表这条输入有对应的输出标签
        if add24h_map.get(int(times[i]), -1) != -1:
            # tmp = list(map(str, train_tmp[i]))
            tmp = train_tmp[i] # 本来就位str
            success = True
            for j in range(1, seq_len):
                if i + j < train_tmp.__len__() and i + j < first_index[add24h_map[int(times[i])] ]:
                    # tmp += list(map(str, train_tmp[i + j]))
                    tmp += train_tmp[i + j]
                else:
                    # raise ValueError('row{} next{}row is the last'.format(i, j))
                    success = False
            if success:
                train.append(tmp)
                label.append(label_tmp[first_index[add24h_map[int(times[i])]] ])

    save_train = np.array(train)
    save_label = np.array(label)
    save = np.c_[save_train, save_label]
    np.savetxt(filename_out, save, fmt='%s', delimiter='\t')
    # with open(filename_out, 'w') as fw:
    #     all = []
    #     for i in range(train.__len__()):
    #         # all.append('\t'.join(train[i]) + '\t' + '\t'.join(label[i]) + '\n')
    #         all.append(' '.join(train[i]) + ' ' + ' '.join(label[i]) + '\n') # 注意
    #     fw.writelines(all)

    return train, label


def process_raw_data_variable_len(filename_in, filename_out, liuliangInd, feat_num, label_num, len_limit=None):

    seq_len = []
    max_len = 0
    raw_data = []
    data = []
    label = []
    volumn = 280

    # with open('20190408data.txt', 'r') as fr:
    with open(filename_in, 'r') as fr:
        fr.readline() # 第一行为字段名称
        for i, line in enumerate(fr.readlines()):
            line = line.strip()
            splt = line.split('\t')
            if not splt or splt[0] == '':
                break
            raw_data.append(splt)

    for i in range(raw_data.__len__()):
        cur_vol = 0
        cur_data = []
        for j in range(i, raw_data.__len__()):
            cur_vol += float(raw_data[j][liuliangInd]) / 60
            if cur_vol > 280:
                data.append(cur_data)
                seq_len.append(j-i)
                label.append(raw_data[j][-label_num:])
                max_len = max(max_len, j-i)
                break
            cur_data.extend(raw_data[j][1:1+feat_num])

    data_arr = np.array(np.zeros([data.__len__(), feat_num * max_len]), dtype=np.str)
    for i in range(data.__len__()):
        data_arr[i, :feat_num * seq_len[i]] = data[i]

    seq_len = np.array(seq_len)[:, np.newaxis]
    if len_limit and len_limit < max_len:
        data_arr = data_arr[:, :len_limit * feat_num]
        seq_len[seq_len > len_limit] = len_limit

    data_arr = np.c_[seq_len, data_arr]
    save_data = np.c_[data_arr, np.array(label)]
    np.savetxt(filename_out, save_data, fmt='%s', delimiter='\t')

    return data_arr.tolist(), label


def tenfold_generator(data, label):
    '''
    :param data: np.array
    :param label: np.array
    :return: np.array
    '''
    m, n = np.shape(data)
    nlabel = np.shape(label)[1]
    increment = int(np.round(m / 10.0))

    for i in range(10):

        if i != 9:
            traindata = np.zeros([m - increment, n])
            traindata[:increment * i, :] = deepcopy(data[:increment * i, :])
            traindata[increment * i:, :] = deepcopy(data[increment * (i + 1):, :])

            trainlabel = np.zeros([m - increment, nlabel])
            trainlabel[:increment * i, :] = deepcopy(label[:increment * i, :])
            trainlabel[increment * i:, :] = deepcopy(label[increment * (i + 1):, :])
            testdata = deepcopy(data[increment * i: increment * (i + 1), :])
            testlabel = deepcopy(label[increment * i: increment * (i + 1), :])
            yield traindata, trainlabel, testdata, testlabel
        else:
            traindata = deepcopy(data[:increment * i, :])
            trainlabel = deepcopy(label[:increment * i, :])
            testdata = deepcopy(data[increment * i:, :])
            testlabel = deepcopy(label[increment * i:, :])
            yield (traindata, trainlabel, testdata, testlabel)


# def batch_generator(data, label, batch_size):
#     '''
#     :param data: np.array
#     :param label: np.array
#     :param batch_size: int
#     :return: np.array
#     '''
#     mdata, ndata = np.shape(data)
#     mlabel, nlabel = np.shape(label)
#     order = np.arange(mdata)
#     np.random.shuffle(order)
#     batch_num = int(np.ceil(mdata / batch_size))
#     for i in range(batch_num):
#         if i != batch_num - 1:
#             yield data[order[i * batch_size:(i + 1) * batch_size]], label[order[i * batch_size:(i + 1) * batch_size]]
#         else:
#             yield data[order[i * batch_size:]], label[order[i * batch_size:]]


def batch_generator(data, label, batch_size):
    '''
    :param data: np.array
    :param label: np.array
    :param batch_size: int
    :return: np.array, fixed batch_size
    '''
    mdata, _ = np.shape(data)
    # mlabel, nlabel = np.shape(label)
    order = np.arange(mdata)
    np.random.shuffle(order)
    batch_num = int(np.ceil(mdata / batch_size))
    for i in range(batch_num):
        if i != batch_num - 1:
            inds = order[i * batch_size:(i + 1) * batch_size]
            yield data[inds], label[inds]
        else:
            inds = order[i * batch_size:]
            count = inds.size
            if count != batch_size:
                past = deepcopy(order[:i * batch_size])
                np.random.shuffle(past)
                sample_from_past = past[:batch_size - count]
                inds = np.concatenate((inds, sample_from_past))
            yield data[inds], label[inds]


def datasmpet_split(data, labels, pos_neg=None, train_percent=0.8, val_percent=0.1, test_percent=0.1):
    num_samples = data.shape[0]
    ## 乱序
    inds = np.arange(num_samples)
    np.random.shuffle(inds)
    data = data[inds]
    labels = labels[inds]
    if pos_neg is not None:
        pos_neg = pos_neg[inds]

    train_val = int(num_samples * train_percent)
    val_test = train_val + int(num_samples * val_percent)

    if pos_neg is None:
        return data[:train_val], labels[:train_val], \
               data[train_val:val_test], labels[train_val:val_test], \
               data[val_test:], labels[val_test:]

    else:
        return data[:train_val], labels[:train_val], pos_neg[:train_val], \
               data[train_val:val_test], labels[train_val:val_test], pos_neg[train_val:val_test], \
               data[val_test:], labels[val_test:], pos_neg[val_test:]


if __name__ == '__main__':
    raw_data = 'data/20190419data.txt'
    preprocessed_data = raw_data[:raw_data.rfind('.')] + '_ok.txt'
    seq_num = int(60 / 6 * 22)
    process_raw_data(raw_data, preprocessed_data, seq_num)
