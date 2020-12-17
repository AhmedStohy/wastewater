import tensorflow as tf
import numpy as np
import os

from data_utils import process_raw_data, tenfold_generator


def stand_regression(datamat, labelmat):  # 每行代表一个样本
    tmp = datamat.T * datamat
    if np.linalg.det(tmp) == 0:
        print('This data matrix is singular, cannot do inverse')
        return None
    else:
        ws = tmp.I * (datamat.T * labelmat)
        return ws


def testfunction(testmat, labelmat, weights):
    m, n = np.shape(testmat)
    mean_abs_error = np.sum(abs(testmat * weights - labelmat)) / m
    mean_square_error = np.sum(np.square(testmat * weights - labelmat)) / m
    return mean_abs_error, mean_square_error


raw_data = '20190408data.txt'
preprocessed_data = raw_data[:raw_data.rfind('.')] + '_ok.txt'

seq_num = int(60 / 3 * 22)
feat_num = 13 * seq_num
label_num = 4
data = []
label = []

if not os.path.exists(preprocessed_data):
    train, label = process_raw_data(raw_data, preprocessed_data, seq_num)
else:
    with open(preprocessed_data) as fr:
        for line in fr.readlines():
            line = line.strip()
            splt = line.split('\t')
            data.append(splt[:feat_num])
            label.append(splt[-label_num:])

# datamat = np.mat(data, dtype=np.float)
# labelmat = np.mat(label, dtype=np.float)
# minvec = np.min(np.array(datamat), axis=0)
# rangevec = np.max(np.array(datamat), axis=0)
# datamat = np.mat((np.array(datamat)-minvec) / rangevec)

total_abs_error = 0.0
total_square_error = 0.0

data = np.mat(data, dtype=np.float)
label = np.mat(label, dtype=np.float)

for traindata, trainlabel, testdata, testlabel in tenfold_generator(data, label):

    # # 归一化
    # datamin = np.min(np.array(traindata), axis=0)
    # datarange = np.max(np.array(traindata), axis=0) - datamin
    # traindata = np.mat((np.array(traindata) - datamin) / datarange)
    # labelmin = np.min(np.array(trainlabel), axis=0)
    # labelrange = np.max(np.array(trainlabel), axis=0) - labelmin
    # trainlabel = np.mat((np.array(trainlabel) - labelmin) / labelrange)
    #
    #
    # # 归一化
    # testdata = np.mat((np.array(testdata) - datamin) / datarange)
    # testlabel = np.mat((np.array(testlabel) - labelmin) / labelrange)


    # traindata = np.mat([np.array(row) for row in traindata]).T
    # trainlabel = np.mat([np.mat(row) for row in trainlabel])
    # testdata = np.mat([np.mat(row) for row in testdata]).T
    # testlabel = np.mat([np.mat(row) for row in testlabel])
    for j in range(label_num):
        # print('round label', j, ':')
        ws = stand_regression(traindata, trainlabel[:, j])
        mean_abs_error, mean_square_error = testfunction(testdata, testlabel[:, j], ws)

        # locally weighted linear regression is too costly
        # mean_abs_error, mean_square_error = lwlr(testdata, testlabel[:, j], traindata, trainlabel[:, j])
        total_abs_error[j] += mean_abs_error / 10
        total_square_error[j] += mean_square_error / 10

print(total_square_error)
print(total_abs_error)
