import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import numpy as np
import itertools
import statsmodels.api as sma
import warnings
import os, sys
import functools as ft

import train

from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima_model import ARIMA
from sklearn.svm import SVR

def draw_data_profile(data_df):
    assert 'datetime' in data_df.columns
    # ts_df = ts_df.iloc[:50, :]
    data_df['datetime'] = data_df['datetime'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M', time.localtime(x)))
    data_df['datetime'] = pd.to_datetime(data_df['datetime'])
    data_df.sort_values('datetime', inplace=True)
    data_df.set_index('datetime', inplace=True)
    # ts_df.plot(y=['COD', 'NH3N', 'TP', 'pH'])
    # print(ts_df.info())
    # plt.figure(figsize=(24, 12))
    # plt.subplot(1, 1, 1)
    for ind, ind_name in enumerate(data_df.columns):
        plt.figure(figsize=(24, 12))
        # if ind_name != 'LL':
        #     continue
        if ind_name == 'datetime':
            continue

        # data_df[ind_name] = data_df[ind_name].apply(lambda x: np.log(1 + x))

        # data_df.iloc[1:, ind] = data_df[ind_name].values[1:] - data_df[ind_name].values[:-1]
        # data_df.iloc[0, ind] = 0

        if ind_name == 'LL':
            data_df[ind_name].plot(label='water flow rate')
        else:
            data_df[ind_name].plot()
        plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y/%m/%d %H:%M'))
        if ind_name == 'LL':
            plt.xlim('2020/01/21 12:00:00', '2020/01/26 00:00:00')
        # plt.xticks(fontsize=14)
        # plt.yticks(fontsize=14)
        plt.xlabel('Date Time', fontsize=40)
        plt.ylabel('Value', fontsize=40)
        plt.tick_params(labelsize=20)
        plt.legend(fontsize=20)
        plt.grid(True)
        # plt.title('Input {} Series Profile'.format(ind_name.upper()), fontsize=24)
        plt.savefig('figure/in_{}_series.eps'.format(ind_name))
        plt.show()


def time_series_decompose(data_df):
    data_df['datetime'] = data_df['datetime'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M', time.localtime(x)))
    data_df['datetime'] = pd.to_datetime(data_df['datetime'])
    data_df.sort_values('datetime', inplace=True)
    data_df.set_index('datetime', inplace=True)
    for ind_name in ts_df.columns:
        if ind_name == 'datetime':
            continue
        result_add = seasonal_decompose(data_df[ind_name], period=12, model='additive')
        result_add.plot()
        plt.gcf().autofmt_xdate()
        date_format = mdate.DateFormatter('%Y-%m-%d')
        plt.gca().xaxis.set_major_formatter(date_format)

        result_mul = seasonal_decompose(data_df[ind_name], period=12, model='multiplicative')
        result_mul.plot()
        plt.gcf().autofmt_xdate()
        date_format = mdate.DateFormatter('%Y-%m-%d')
        plt.gca().xaxis.set_major_formatter(date_format)
        plt.show()


def Holt_Winters_method():
    test_data, test_label = train.testdata, train.testlabel
    args = train.args
    fa = open('{}/res.txt'.format(args.savedir), 'a')

    test_size = test_data.shape[0]
    test_data = np.reshape(test_data, [test_size, -1])
    test_pred_add = []
    test_pred_mul = []
    for test_ind in range(test_size):
        model_add = ExponentialSmoothing(test_data[test_ind], trend='additive', seasonal=None, damped=True).fit()
        model_add_pred = model_add.forecast(args.label_num)
        test_pred_add.append(model_add_pred)

        # model_mul = ExponentialSmoothing(test_data[test_ind], trend='multiplicative', seasonal=None, damped=True).fit()
        # model_mul_pred = model_mul.forecast(args.label_num)
        # test_pred_mul.append(model_mul_pred)

    test_pred_add, test_pred_mul = np.array(test_pred_add), np.array(test_pred_mul)
    train.print_mse_mer_ratio(test_pred_add, test_label, 'HoltWintersAdd', writer=fa)
    train.save_predict_and_true_value_to_file('HoltWintersAdd_test', test_pred_add, test_label)
    # for ind in range(test_label.shape[1]):
    #     train.draw_predict_and_true_value_curve(test_pred_add[:, ind], test_label[:, ind], plot_method='line')
    #     # train.draw_predict_and_true_value_curve(test_pred_add[:, ind], test_label[:, ind], plot_method='scatter')

    # train.print_mse_mer_ratio(test_pred_mul, test_label, 'HoltWintersMul')
    # for ind in range(test_label.shape[1]):
    #     train.draw_predict_and_true_value_curve(test_pred_add[:, ind], test_label[:, ind], plot_method='line')
    #     # train.draw_predict_and_true_value_curve(test_pred_add[:, ind], test_label[:, ind], plot_method='scatter')

    fa.close()


def ARIMA_method():
    test_data, test_label = train.testdata, train.testlabel
    args = train.args
    fa = open('{}/res.txt'.format(args.savedir), 'a')

    test_size = test_data.shape[0]
    test_preds = []
    for test_ind in range(test_size):
        q = d = range(0, 2)
        p = range(0, 4)
        pdq = list(itertools.product(p, d, q))
        best_param = None
        best_model = None
        best_res = None
        min_AIC = float('inf')
        for para_ind, param in enumerate(pdq):
            try:
                # cur_model = sma.tsa.statespace.SARIMAX(test_data[test_ind], order=param,
                #                                        seasonal_order=param_seasonal,
                #                                        enforce_stationarity=False,
                #                                        enforce_invertibility=False)
                cur_model = ARIMA(test_data[test_ind], order=param)

                cur_res = cur_model.fit()

                print('ARIMA{} - AIC:{}'.format(param, cur_res.aic))

                if cur_res.aic < min_AIC:
                    best_param = param
                    best_model = cur_model
                    best_res = cur_res
                    min_AIC = cur_res.aic

            except:
                print('Sample {} Error: {}'.format(test_ind, param))
                continue
        print('Sample {}: The smallest AIC is {} for model ARIMA{}'.format(test_ind, min_AIC, best_param))
        cur_pred = best_res.forecast(args.label_num)[0]
        test_preds.append(cur_pred)

    test_preds = np.array(test_preds)
    train.print_mse_mer_ratio(test_preds, test_label, 'ARIMA', writer=fa)
    train.save_predict_and_true_value_to_file('ARIMA', test_preds, test_label)
    # for ind in range(test_label.shape[1]):
    #     train.draw_predict_and_true_value_curve(test_preds[:, ind], test_label[:, ind], plot_method='line')

    fa.close()


def svr_method():
    train_data, train_label = train.data, train.label
    test_data, test_label = train.testdata, train.testlabel
    args = train.args
    fa = open('{}/res.txt'.format(args.savedir), 'a')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), file=fa)

    ### preprocess
    # if args.auto_regressive:
    train_data, train_label = train.transform_train_data(train_data, train_label)
    train_data, train_label = np.squeeze(train_data), np.squeeze(train_label)
    test_data = np.squeeze(test_data)

    def svr_predict(svr_model, test_data):
        cur_preds = svr_model.predict(test_data)
        # cur_preds = cur_preds[:, np.newaxis]
        return cur_preds

    kernel_options = ['rbf', 'linear', 'sigmoid', 'poly']
    min_test_mer = np.ones(args.label_num) * float('inf')
    min_test_mse = np.ones(args.label_num) * float('inf')
    min_test_ratio = np.ones(args.label_num) * float('inf')
    for kernel in kernel_options:
        try:
            model_svr = SVR(kernel=kernel)
            model_svr.fit(train_data, train_label)
            cur_predict_func = ft.partial(svr_predict, model_svr)
            test_preds = train.auto_regressive_test(cur_predict_func, args.label_num / args.feat_num, test_data)
            print('kernel {} res:'.format(kernel))
            print('kernel {} res:'.format(kernel), file=fa)
            test_mse, test_mer, test_ratio = train.print_mse_mer_ratio(test_preds, test_label, 'test', writer=fa)
            train.plot_predict_and_true_value_curve('test', test_preds, test_label)
            print()
            print(file=fa)

            if args.loss_function == 'mer':
                if_val_perform_better = test_mer.mean() < min_test_mer.mean()
            elif args.loss_function == 'mse':
                if_val_perform_better = test_mse.mean() < min_test_mse.mean()
            else:
                raise NotImplementedError('args.loss_function invalid.')
            if if_val_perform_better:
                min_test_mer = test_mer
                min_test_mse = test_mse
                min_test_ratio = test_ratio
        except:
            print('kernel {} fail'.format(kernel))
            print('kernel {} fail'.format(kernel), file=fa)
            print()
            print(file=fa)

    print('min test mse per indicator:', min_test_mse)
    print('min test mer per indicator:', min_test_mer)
    print('min test ratio per indicator:', min_test_ratio)
    print('min test mse per indicator:', min_test_mse, file=fa)
    print('min test mer per indicator:', min_test_mer, file=fa)
    print('min test ratio per indicator:', min_test_ratio, file=fa)
    print('\n\n', file=fa)

    fa.close()



if __name__ == '__main__':
    # warnings.filterwarnings("ignore")
    if not os.path.exists('figure/'): # 存储各种效果图
        os.mkdir('figure')

    ts_file = 'data/20200414data_H.txt'
    # ts_file = 'data/20200406data_0.txt'
    ts_df = pd.read_csv(ts_file, sep='\t', skiprows=1)
    ts_df.columns = ['datetime', 'COD', 'NH3N', 'TP', 'pH']
    # ts_df.columns = ['datetime', 'NH3N', 'TN', 'COD', 'pH', 'LL', 'T', 'ORP', 'GAAD', 'AD', 'ZD', 'COD']

    ### data overview
    # draw_data_profile(ts_df)

    ### evaluate Naive forecast
    # test_data, test_label = train.testdata, train.testlabel
    # args = train.args
    # naive_preds = np.tile(np.squeeze(test_data[:, -1:], axis=-1), (1, args.label_num))
    # train.print_mse_mer_ratio(naive_preds, test_label, 'naive', print_to_file=True)
    # train.save_predict_and_true_value_to_file('naive_test', naive_preds, test_label)

    ### draw autocorrelation_plot, acf, pacf plot
    # for ind_name in ts_df.columns:
    #     if ind_name == 'datetime':
    #         continue
    #     autocorrelation_plot(ts_df[ind_name])
    #     plt.title('Input {} Series Auto Correlation Plot'.format(ind_name.upper()), fontsize=16)
    #     plt.xlabel('Lag', fontsize=14)
    #     plt.ylabel('Auto Correlation', fontsize=14)
    #     plt.ylim(-0.3, 0.85)
    #     plt.savefig('figure/AutoCorr_{}.eps'.format(ind_name))
    #     plt.show()
    # for ind_name in ts_df.columns:
    #     if ind_name == 'datetime':
    #         continue
    #     plot_acf(ts_df[ind_name], lags=40, use_vlines=True)
    #     plt.title('Input {} Series ACF Plot'.format(ind_name.upper()), fontsize=16)
    #     plt.xlabel('Lag', fontsize=14)
    #     plt.ylabel('Auto Correlation', fontsize=14)
    #     # plt.ylim(-0.3, 0.85)
    #     plt.savefig('figure/ACF_{}.png'.format(ind_name))
    #     plt.show()
    for ind_name in ts_df.columns:
        if ind_name == 'datetime':
            continue
        plot_pacf(ts_df[ind_name], lags=40, use_vlines=True, title=None)
        # plt.title('Input {} Series PACF Plot'.format(ind_name.upper()), fontsize=16)
        plt.xlabel('Lag', fontsize=14)
        plt.ylabel('Partial Auto Correlation', fontsize=14)
        # plt.ylim(-0.3, 0.85)
        plt.savefig('figure/PACF_{}.png'.format(ind_name))
        plt.show()

    ### TS decompose
    # time_series_decompose(ts_df)

    ### evaluate holt winters
    # Holt_Winters_method()

    ### evaluate arima
    # ARIMA_method()

    ### evaluate svr
    # svr_method()


    print()


