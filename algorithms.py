import numpy as np
import tensorflow as tf
import sys, os
import time
import csv

from copy import deepcopy

def pso(testdata, test_seq_len, model, arg, zongdan_target, model_path=None, testlabel=None, session=None):

    opt_result = []
    opt_output = []
    output_delay = []

    if session is None:
        if model_path is None:
            print("If param 'sess' is None, then 'model_path' should not be None.")
            sys.exit(1)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.latest_checkpoint(model_path)  # 注意
        if ckpt:
            saver = tf.train.Saver()
            saver.restore(sess, ckpt)
        else:
            print('No model finded in model_path')
            sess.close()
            sys.exit(1)
    else:
        sess = session

    t1 = time.time()
    for i in range(testdata.__len__()):

        print('progress: {} / {}'.format(i, testdata.__len__()))
        particle_num = 10
        bingcusuan_ind = 7
        opt_ind = [bingcusuan_ind]  ### 需要优化的变量在输入指标中的列号，从0开始
        opt_direction = np.array([-1], dtype=np.float32) ### -opt_direction即为解被拒绝时的速度
        assert opt_ind.__len__() == opt_direction.__len__()

        judge_ind = [1] ### 哪些输出指标必须达标，在testlabel中的列号，从0开始
        judge_val = [zongdan_target] ### 达标值，注意，假设都需要小于达标值
        assert judge_ind.__len__() == judge_val.__len__()

        solu_space_dim = opt_ind.__len__()
        solu_low = 5 ### 每一个待寻优的变量的下界，对应opt_ind列表
        solu_high = 30 ### 上界

        if testdata[i, 0, arg.liuliangInd] < 5:
            opt_result.append(np.zeros([solu_space_dim]))
            opt_output.append(np.ones([arg.label_num]) * -1)
            output_delay.append([-1])
            continue

        particles = np.random.uniform(solu_low, solu_high, [particle_num, solu_space_dim])
        velocities = np.ones([particle_num, solu_space_dim], np.float32) * -1

        ### 注意，假设解都越小越优
        pbest = np.ones([particle_num, solu_space_dim], dtype=np.float32) * solu_high
        gbest = np.ones([solu_space_dim], dtype=np.float32) * solu_high
        gbest_output = None

        c1 = 2
        c2 = 2
        w_init = 0.9
        w_end = 0.4

        max_iter = 1000
        early_stop = 100
        es_count = 0
        update = False
        for j in range(max_iter):
            es_count += 1
            # last_gbest = gbest

            cur_inputs = np.repeat(testdata[i][np.newaxis, :, :], particle_num, axis=0)

            for k in range(particle_num):
                cur_inputs[k, 0, opt_ind] = particles[k]
                if test_seq_len is None:
                    print("Not support error: param 'test_seq_len' cannot be None.")
                    sys.exit(1)
                else:
                    cur_inputs[k, 1:test_seq_len[i], :] = cur_inputs[k, 0, :] ### 注indices类型一定为int

            if test_seq_len is None:
                pred = sess.run(model.preds, feed_dict={model.x: cur_inputs})
            else:
                cur_seq_lens = np.repeat(test_seq_len[i], particle_num) ### test_seq_len[i] is a scalar
                pred = sess.run(model.preds, feed_dict={model.x: cur_inputs, model.seq_lens: cur_seq_lens})

            total_valid = []
            for kk in range(judge_ind.__len__()):
                one_valid = pred[:, judge_ind[kk]] < judge_val[kk]
                total_valid.append(one_valid)

            total_valid = np.array(total_valid)
            total_valid = total_valid.sum(axis=0)
            total_valid = total_valid == judge_ind.__len__()

            w = w_init - j * (w_init - w_end) / max_iter
            for k in range(particle_num):
                r1, r2 = np.random.uniform(0, 1, 2)

                if total_valid[k]:
                    pbest[k] = np.min(np.r_[pbest[k][np.newaxis, :], particles[k][np.newaxis, :]], axis=0)
                    is_min = pbest[k] < gbest
                    if is_min.sum() == solu_space_dim:
                        update = True
                        gbest = pbest[k]
                        gbest_output = pred[k]
                        es_count = 0

                if total_valid[k]:
                    velocities[k] = w * velocities[k] + c1 * r1 * (pbest[k] - particles[k]) +\
                                    c2 * r2 * (gbest - particles[k])
                else:
                    velocities[k] = -w * opt_direction

                particles[k] += velocities[k]
                np.clip(particles[k], solu_low, solu_high, out=particles[k])

            sys.stdout.write('\roptimizing: {} / {}'.format(j, max_iter))
            sys.stdout.flush()

            if es_count == early_stop:
                print()
                print('gbest has not been improved for {} iters, early stop.'.format(early_stop))
                break

        opt_result.append(gbest)
        if not update:
            cur_inputs = deepcopy(testdata[i])[np.newaxis, :, :]
            cur_inputs[0, 0, opt_ind] = gbest
            # cur_inputs[0, 1:test_seq_len[i], :] = cur_inputs[0, 0, :]  ### 注indices类型一定为int
            if not arg.isVariableLen:
                gbest_output = sess.run(model.preds, feed_dict={model.x: cur_inputs})[0]
            else:
                cur_seq_lens = test_seq_len[i][np.newaxis] ### test_seq_len[i] is a scalar
                gbest_output = sess.run(model.preds, feed_dict={model.x: cur_inputs, model.seq_lens: cur_seq_lens})[0]
        opt_output.append(gbest_output)
        output_delay.append([test_seq_len[i]])

    t2 = time.time()
    opt_result = np.array(opt_result)
    opt_output = np.array(opt_output)
    output_delay = np.array(output_delay)
    ori = testdata[:, 0, :]
    if session is None:
        if testlabel is None:
            print("If param 'sess' is None, then 'testlabel' should not be None.")
            sys.exit(1)
        print(ori.shape, opt_result.shape, opt_output.shape, output_delay.shape, testlabel.shape)
        print()
        file_name = 'data/res/{}_opt_res.csv'.format(model_path[:model_path.find('/')])
        with open(file_name, 'w') as fw:  ### savedir 前面字母最后'/'
            csv_write = csv.writer(fw)
            csv_write.writerows(np.c_[ori, opt_result, output_delay, opt_output, testlabel])
        sess.close()

    return np.c_[opt_result, output_delay, opt_output]