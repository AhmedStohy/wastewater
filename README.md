# Introduction
This repo contains the source code for paper [Zhang L, Ma X, Shi P, et al. RegCNN: A Deep Multi-output Regression Method for Wastewater Treatment[C]//2019 IEEE 31st International Conference on Tools with Artificial Intelligence (ICTAI). IEEE, 2019: 816-823.](https://ieeexplore.ieee.org/abstract/document/8995367)


# Contents

Python files with keyword 'prep' are used to preprocess raw data.\
[table_connect.py](table_connect.py) joins the data from different tables according to their time stamps.\
[cnn.py](cnn.py) is the training code for RegCNN.\
Due to confidentiality issues, only processed data [20200406data_0_ok.txt](data/20200406data_0_ok.txt) is provided here.

# Requirements
Recommended tensorflow version is 1.12.


# Usage
## Train
Take cnn.py as an example, \
set line 37 'istrain = True', then
```sh
python cnn.py
```
The training results will be recorded at 'wastewater/res/cnn_res.txt'.

## Test
Take cnn.py as an example,\
modify line 37: 'istrain = False', then
```sh
python cnn.py
```