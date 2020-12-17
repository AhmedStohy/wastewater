import tensorflow as tf
import os, sys
import numpy as np
import math
import logging
import traceback

import models
import algorithms

from flask import Flask
from flask import request

# Flask初始化参数尽量使用你的包名，这个初始化方式是官方推荐的，官方解释：http://flask.pocoo.org/docs/0.12/api/#flask.Flask
app = Flask(__name__)

logging.basicConfig(filename='log.txt')

class Config(object):

    time_increment = 24  ### 注意时间间隔，单位为分
    time_intv_per_sample = 1  ### x min一条数据
    last_equal_intv = 0  ### 最后 x min的输入视为一样
    last_equal_num = int(last_equal_intv / time_intv_per_sample)  ### 最后 x 个输入一样
    # seq_num = int(60 / 5 * time_increment)
    # seq_num = int(60 / 1 * time_increment)
    seq_num = int(time_increment / time_intv_per_sample)  ### 序列长度
    # seq_num = 12
    act_seq_num = int(seq_num / 1)  # 注意
    feat_num = 8  # 注意
    # feat_num = 4 # 注意
    total_feat_num = feat_num * act_seq_num  # 注意
    label_num = 3

    layer_num = 1
    hid_dims = [128]
    # hid_dims = [128]
    # out_hid_dims = [256]
    out_hid_dims = [128]
    keep_prop = 1.0
    learning_rate = 0.0003
    max_epoch = 1000
    batch_size = 64
    istrain = False
    use_loss_weight = False
    liuliangInd = 4  ### 在原始文件中的列号，从0开始
    isVariableLen = True
    seq_len_limit = None
    is_pso = True

def load_model():
    model_path = 'bi_rnn/fold0'
    model = models.LSTM(arg)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.latest_checkpoint(model_path)  # 注意
    if ckpt:
        saver = tf.train.Saver()
        saver.restore(sess, ckpt)
    else:
        print('No TensorFlow model found.')
        sys.exit(1)
    return sess, model


@app.route('/')
def hello_world():
    return "Hello World!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        inputs = request.json['Inputs']
        zd_target = inputs[-1]
        if inputs[arg.liuliangInd] < 5:
            return dict({"state": "NORMAL", "predict": [0., -1., -1., -1., -1.], "info": "LiuLiang==0, no predicts."})
        else:
            seq_len = math.ceil(280 / (inputs[arg.liuliangInd] / 60))
            model_input = np.repeat(np.array([inputs], dtype=np.float32), seq_len, axis=0)[np.newaxis, :, :]
            result = algorithms.pso(model_input, np.array([seq_len]), model, arg, zd_target, session=sess)
            return dict({"state": "SUCCESS", "predict": list(map(float, result[0].tolist())), "info": "Good Job."})
    except:
        error = traceback.format_exc()
        logging.error(error)
        return dict({"state": "FAIL", "predict": [], "info": "Oops, some errors occur, see './log.txt'"})


if __name__ == "__main__":
    arg = Config()
    sess, model = load_model()
    app.run(host='0.0.0.0', port=12000)