import os
os.environ["PYTHONHASHSEED"] = str(1)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

from DB_Handler import DBHandler
from fit_test import *
import tensorflow as tf
import numpy as np
import random as python_random

python_random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
# strategy = tf.distribute.MirroredStrategy()

if __name__ == '__main__':
    batch_size = 64 # fix(ibk : 64, taiwan : 128, lending : 128)
    unit = 64 # fix(ibk : 64, taiwan : 128, lending : 128)
    all_iter = 50 # fix
    sub_iter = 50 # fix
    margin = 1
    # fix(ibk - triplet : 5.047, tuplet : 3.168, siamese : 9.022, k-means : 1.92, GMM : 1), (taiwan : 1), (lending - 1.15)
    epoch = 300 # ibk - 250, taiwan - 500, lending - 1500
    cnn_kernel_size = 2
    # fix(ibk - triplet : 2, tuplet : 1, siamese : 2, k-means : 2, GMM : 7), (taiwan : 7), (lending - 7)
    cnn_filter = 1
    # fix(ibk - triplet : 30, tuplet : 3, siamese : 17, k-means : 9, GMM : 1), (taiwan : 1), (lending - 1)
    bayes = False
    db_handler = DBHandler()
    model_name = "DMETRIC_SN"
    num_neg_sample = 1
    voting = 1
    sharpness_param = 0.7
    distance = False
    oversampling = None
    bigbang_train = True
    dataset = "T"

    if model_name == "DMETRIC_TN":
        test = test_triplenet(model_name, all_iter, batch_size, sub_iter, unit, epoch, bayes, db_handler, margin,
                              cnn_kernel_size=cnn_kernel_size, cnn_filter=cnn_filter, voting=voting, distance=distance)
        test.triple_net_test()

    elif model_name == "DMETRIC_TL":
        test = test_tuplet(model_name, all_iter, batch_size, sub_iter, unit, epoch, bayes, db_handler,
                              cnn_kernel_size=cnn_kernel_size, cnn_filter=cnn_filter, voting=voting, distance=distance, num_neg_sample=num_neg_sample)
        test._net_test()

    elif model_name == "DMETRIC_LFE":
        test = test_lfe(model_name, all_iter, batch_size, sub_iter, unit, epoch, bayes, db_handler, margin, bigbang_train,
                           dataset, cnn_kernel_size=cnn_kernel_size, cnn_filter=cnn_filter, voting=voting, distance=distance)
        test._net_test()

    elif model_name == "DMETRIC_SN":
        test = test_siamese(model_name, all_iter, batch_size, sub_iter, unit, epoch, bayes, db_handler, margin,
                        cnn_kernel_size=cnn_kernel_size, cnn_filter=cnn_filter, voting=voting, distance=distance, sharpness_param=sharpness_param)
        test._net_test()

    elif model_name == "ML-XGB":
        test = test_xgboost(model_name, all_iter, sub_iter, db_handler, oversampling, bigbang_train, dataset, bayes=False)
        test._net_test()

    else:
        test = test_randomforest(model_name, all_iter, sub_iter, db_handler, oversampling, bigbang_train, dataset, bayes=False)
        test._net_test()