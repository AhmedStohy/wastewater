import numpy as np
import tensorflow as tf
import sys, os
import time
import csv

from copy import deepcopy


def check_model_pred(X, V, cur_sample, cur_seq_len, cur_sample_label, model, sess, args, opt_inds, opt_bounds, count, state):
    N, D = X.shape

    fake_samples = np.tile(np.expand_dims(cur_sample, 0), [N, 1, 1])
    if args.model_select == 'transformer':
        fake_samples[:, 1, opt_inds] = X.copy()
        if cur_seq_len > 1:
            fake_samples[:, 2:cur_seq_len + 2 - 1] = fake_samples[:, 1:2]
    else:
        fake_samples[:, 0, opt_inds] = X.copy()
        if cur_seq_len > 1:
            fake_samples[:, 1:cur_seq_len + 1 - 1] = fake_samples[:, 0:1]

    fake_samples_preds = sess.run(model.preds, feed_dict={model.x: fake_samples})
    for tmp_row_ind in range(N):
        while np.sign(fake_samples_preds[tmp_row_ind]).sum() < D or \
                (fake_samples_preds[tmp_row_ind] > cur_sample_label * 10).any():  # 预测结果异常，<=0均认为异常
            count += 1  # 模型异常预测次数
            if state == 'initial':
                for tmp_col_ind in range(D):
                    X[tmp_row_ind] = np.random.uniform(opt_bounds[tmp_col_ind][0], opt_bounds[tmp_col_ind][1], 1) # 重新初始化
            elif state == 'optimizing':
                X[tmp_row_ind] = X[tmp_row_ind] + np.random.uniform(-0.5, 0.5, 1) * V[tmp_row_ind] # 基于当前速度随机移动粒子
            else:
                raise NotImplementedError('state undefined.')

            if args.model_select == 'transformer':
                fake_samples[tmp_row_ind, 1, opt_inds] = X[tmp_row_ind].copy()
                if cur_seq_len > 1:
                    fake_samples[tmp_row_ind, 2:cur_seq_len + 2 - 1] = fake_samples[tmp_row_ind, 1:2]
            else:
                fake_samples[tmp_row_ind, 0, opt_inds] = X[tmp_row_ind].copy()
                if cur_seq_len > 1:
                    fake_samples[tmp_row_ind, 1:cur_seq_len + 1 - 1] = fake_samples[tmp_row_ind, 0:1]

            fake_samples_preds[tmp_row_ind] = sess.run(model.preds,
                                                       feed_dict={model.x: fake_samples[tmp_row_ind:tmp_row_ind + 1]})

    return X, fake_samples_preds, count


def pso(pred_model, session, args, opt_inds, opt_bounds, cond_inds, cond_vals, testdata, testlabel, test_seq_len):
    assert len(opt_inds) == len(opt_bounds) and len(cond_inds) == len(cond_vals) and np.array(opt_bounds).shape[1] == 2
    print()
    print('------------------------Begin PSO------------------------')
    t_begin = time.time()
    cond_vals = np.array(cond_vals, dtype=np.float32)

    D = len(opt_inds)
    N = 10
    test_size = testdata.shape[0]

    opt_res = []
    preds_after_opt = []
    model_abnomal_count = 0
    liuliang_under_limit_count = 0
    for test_ind, test_sample in enumerate(testdata):
        if args.model_select == 'transformer':
            cur_liuliang = test_sample[1][args.liuliang_ind]
        else:
            cur_liuliang = test_sample[0][args.liuliang_ind]
        if cur_liuliang < 5:
            liuliang_under_limit_count += 1
            print('sample:{} water flow rate:{}'.format(test_ind, cur_liuliang))
            opt_res.append(np.zeros(D, np.float32))
            preds_after_opt.append(np.zeros(args.label_num, np.float32))
            continue

        cp = 6
        cg = 6
        w_init = 0.9
        w_end = 0.2
        X = np.zeros([N, D], dtype=np.float32)  # 粒子位置
        V = np.ones([N, D], np.float32) * -1  # 粒子初始速度
        opt_direction = np.ones(D, dtype=np.float32) * -1.
        # 初始化粒子位置
        for tmp_col_ind in range(D):
            tmp_low, tmp_high = opt_bounds[tmp_col_ind]
            # X[:, tmp_col_ind] = np.random.uniform(tmp_low, tmp_high, N)
            X[:, tmp_col_ind] = np.arange(tmp_low, tmp_high, (tmp_high - tmp_low) / N, dtype=np.float32)

        X, cur_model_preds, model_abnomal_count = check_model_pred(X, V, test_sample, test_seq_len[test_ind],
                                                                   testlabel[test_ind],
                                                                   pred_model, session, args, opt_inds, opt_bounds,
                                                                   model_abnomal_count, 'initial')

        pbest = X.copy() # 注copy
        # pbest_min_ind = np.argmin(np.sum(pbest, axis=-1))  # 评判粒子位置是否更优的标准是总和更小
        gbest = np.array(opt_bounds)[:, 1]
        gbest, model_pred_of_gbest, model_abnomal_count = check_model_pred(np.expand_dims(gbest, 0),
                                                                           np.expand_dims(np.ones(D) * -0.1, 0),
                                                                           test_sample, test_seq_len[test_ind],
                                                                           testlabel[test_ind],
                                                                           pred_model, session, args, opt_inds,
                                                                           opt_bounds, model_abnomal_count,
                                                                           'optimizing')

        max_iter = 200
        early_stop = 20
        es_count = 0
        display_step = 5
        for iter_ind in range(max_iter):

            w = w_init + iter_ind * (w_end - w_init) / max_iter
            rp, rg = np.random.uniform(0, 1, 2)
            for tmp_row_ind in range(N):
                if (cur_model_preds[tmp_row_ind][cond_inds] < cond_vals).all():  # 出水水质达标
                    # 更新pbest和gbest
                    if X[tmp_row_ind].sum() < pbest[tmp_row_ind].sum():
                        pbest[tmp_row_ind] = X[tmp_row_ind].copy()
                        if X[tmp_row_ind].sum() < gbest.sum():
                            gbest = X[tmp_row_ind].copy()
                            model_pred_of_gbest = cur_model_preds[tmp_row_ind].copy()
                            es_count = 0

                    V[tmp_row_ind] = w * V[tmp_row_ind] + (cp * rp * (pbest[tmp_row_ind] - X[tmp_row_ind]) +
                                                                     cg * rg * (gbest - X[tmp_row_ind]))
                else:
                    # V[tmp_row_ind] = -0.1 * opt_direction
                    V[tmp_row_ind] = -0.5 * V[tmp_row_ind]

                X[tmp_row_ind] = X[tmp_row_ind] + V[tmp_row_ind]

            for tmp_col_ind in range(D):
                X[:, tmp_col_ind] = np.clip(X[:, tmp_col_ind], opt_bounds[tmp_col_ind][0], opt_bounds[tmp_col_ind][1])

            X, cur_model_preds, model_abnomal_count = check_model_pred(X, V, test_sample, test_seq_len[test_ind],
                                                                       testlabel[test_ind],
                                                                       pred_model, session, args, opt_inds, opt_bounds,
                                                                       model_abnomal_count, 'optimizing')


            if iter_ind % display_step == 0:
                print('sample:{} iter:{} gbest:{} gbest model pred:{}'.format(test_ind, iter_ind, gbest, model_pred_of_gbest))

            if es_count == early_stop:
                print('gbest has not been improved for {} iters, early stop.'.format(early_stop))
                print('current particles: {}'.format(X))
                print()
                break

            es_count += 1

        opt_res.append(gbest)
        preds_after_opt.append(model_pred_of_gbest)

    t_end = time.time()
    print('pso total time cost:{}'.format(t_end - t_begin))
    print('test size:{} model pred abnormal count:{} low water flow rate count:{}'.format(test_size,
                                                                                          model_abnomal_count,
                                                                                          liuliang_under_limit_count))

    opt_res = np.array(opt_res)
    preds_after_opt = np.array(preds_after_opt)
    print(testdata.shape, testlabel.shape, opt_res.shape, preds_after_opt.shape)
    if args.model_select == 'transformer':
        dosage_origin_mean = np.mean(testdata[:, 1, opt_inds], axis=0)
    else:
        dosage_origin_mean = np.mean(testdata[:, 0, opt_inds], axis=0)
    print('dosage origin mean:{} output water quality origin mean:{}'.format(dosage_origin_mean,
                                                                             np.mean(testlabel, axis=0)))

    print('dosage optimized mean:{} output water quality after optimization:{}'.format(np.mean(opt_res, axis=0),
                                                                                  np.mean(preds_after_opt, axis=0)))
    print()
    file_name = '{}/pso_res_{}TN.csv'.format(args.savedir, str(cond_vals[0]).replace('.', ''))
    with open(file_name, 'w') as fw:  ### savedir 前面字母最后'/'
        csv_write = csv.writer(fw)
        if args.model_select == 'transformer':
            csv_write.writerows(np.c_[testdata[:, 1, opt_inds], testlabel, opt_res, preds_after_opt])
        else:
            csv_write.writerows(np.c_[testdata[:, 0, opt_inds], testlabel, opt_res, preds_after_opt])

    # return opt_res, preds_after_opt
