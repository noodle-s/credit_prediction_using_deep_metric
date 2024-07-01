import os
os.environ["PYTHONHASHSEED"] = str(1)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

from functools import partial
from keras import optimizers
from preprocessing import *
from model import *
from bayes_opt import BayesianOptimization
import xgboost as xgb
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random as python_random
tf.random.set_seed(1)
python_random.seed(1)
np.random.seed(1)

def _bayes_opti(model, random_state=1):
    pbounds = {'margin': (0, 10), 'cnn_filter' : (1, 10)}

    optimizer = BayesianOptimization(
        f=model,
        pbounds=pbounds,
        verbose=1,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=random_state,
    )

    optimizer.maximize(
        init_points=5,
        n_iter=30
    )

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

class test_triplenet:
    def __init__(self, model_name, all_iter, batch_size, sub_iter, unit, epoch, bayes=False, db_handler=None,
                 margin=0, cnn_kernel_size=1, cnn_filter=1, voting=1, distance=False):
        super(test_triplenet, self).__init__()
        self.model_name = model_name
        self.all_iter = all_iter
        self.batch_size = batch_size
        self.sub_iter = sub_iter
        self.unit = unit
        self.epoch = epoch
        self.bayes = bayes
        self.db_handler = db_handler
        self.margin = margin
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter = cnn_filter
        self.voting = voting
        self.distance = distance

    def _fit_with(self, train, test, margin, cnn_kernel_size, cnn_filter):
        cnn_kernel_size = int(cnn_kernel_size)
        cnn_filter = int(cnn_filter)
        network = self._get_network(train)

        y_true = test['label'][0, :]
        y_pred = []
        ori_cat = test['origin_cat'][0, :, :]
        ori_num = test['origin_num'][0, :, :]
        pos_cat = test['pos_cat'][0, :, :]
        pos_num = test['pos_num'][0, :, :]
        neg_cat = test['neg_cat'][0, :, :]
        neg_num = test['neg_num'][0, :, :]
        cal = test['cal'][0, :, :]

        conf_matrix = {
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0}

        pred = network.predict([ori_cat, ori_num, pos_cat, pos_num, neg_cat, neg_num], batch_size=1,
                               use_multiprocessing=True)

        for idx in range(0, ori_cat.shape[0], self.voting):
            cnt = 0
            distance = {1: 0, 0: 0}

            if self.voting == 1:
                if pred[1][idx] < pred[0][idx]:
                    y_pred.append(1)

                    if y_true[int(idx / self.voting)] == 0:
                        conf_matrix["fp"] += 1

                    else:
                        conf_matrix["tp"] += 1

                else:
                    y_pred.append(0)

                    if y_true[int(idx / self.voting)] == 1:
                        conf_matrix["fn"] += 1

                    else:
                        conf_matrix["tn"] += 1

            else:
                for v in range(self.voting):
                    if self.distance:
                        distance[1] += pred[1][v + idx]
                        distance[0] += pred[0][v + idx]

                    else:
                        if pred[0][v + idx] < pred[1][v + idx]:
                            cnt += 1

                if self.distance:
                    if distance[1] < distance[0]:
                        y_pred.append(1)

                        if y_true[int(idx / self.voting)] == 0:
                            conf_matrix["fp"] += 1

                        else:
                            conf_matrix["tp"] += 1

                    else:
                        y_pred.append(0)

                        if y_true[int(idx / self.voting)] == 1:
                            conf_matrix["fn"] += 1

                        else:
                            conf_matrix["tn"] += 1

                else:
                    if cnt > int(self.voting / 2):
                        y_pred.append(0)

                        if y_true[int(idx / self.voting)] == 0:
                            conf_matrix["tn"] += 1

                        else:
                            conf_matrix["fn"] += 1

                    else:
                        y_pred.append(1)

                        if y_true[int(idx / self.voting)] == 1:
                            conf_matrix["tp"] += 1

                        else:
                            conf_matrix["fp"] += 1

        FN = conf_matrix['fn']
        TN = conf_matrix['tn']
        TP = conf_matrix['tp']
        FP = conf_matrix['fp']

        not_nan = 0.00000000000000001
        auc = round((TN / (TN + FP + not_nan) + TP / (TP + FN + not_nan)) * 0.5 * 100, 2)

        return auc

    def _get_network(self, train):
        network = TripletsNetwork([train['origin_cat'].shape[-1], train['origin_num'].shape[-1]], self.batch_size,
                                  margin=self.margin, unit=self.unit, cnn_kernel_size=self.cnn_kernel_size, cnn_filter=self.cnn_filter)
        network.compile(optimizer=optimizers.Adam(0.001))
        network.fit([train['origin_cat'], train['origin_num'],
                     train['pos_cat'], train['pos_num'],
                     train['neg_cat'], train['neg_num']], batch_size=self.batch_size, epochs=self.epoch,
                    verbose=1, use_multiprocessing=True)

        return network

    def triple_net_test(self):
        batch_id = 996
        sampling = ""
        model_full_name = f"{self.model_name}"
        model_no = self.db_handler.select_model_no(self.model_name)

        res_auc = []
        res_negative_recall = []
        res_positive_recall = []
        res_positive_rate = []
        res_negative_rate = []

        for iter in range(self.all_iter):
            iter = iter
            # start = time.time()
            gen = TripletGenerator(self.batch_size, self.sub_iter, self.voting, iter)
            train,test = gen.get_triplets_batch()

            if self.bayes:
                network = partial(self._fit_with, train, test)
                _bayes_opti(network)

                return

            else:
                network = self._get_network(train)

            tmp_auc = []
            tmp_negative_recall = []
            tmp_positive_recall = []
            tmp_positive_rate = []
            tmp_negative_rate = []

            for sub_iter in range(self.sub_iter):
                y_true = test['label'][sub_iter, :]
                y_pred = []
                ori_cat = test['origin_cat'][sub_iter, :, :]
                ori_num = test['origin_num'][sub_iter, :, :]
                pos_cat = test['pos_cat'][sub_iter, :, :]
                pos_num = test['pos_num'][sub_iter, :, :]
                neg_cat = test['neg_cat'][sub_iter, :, :]
                neg_num = test['neg_num'][sub_iter, :, :]

                conf_matrix = {
                    "tn" : 0,
                    "fp" : 0,
                    "fn" : 0,
                    "tp" : 0 }

                pred = network.predict([ori_cat,ori_num,pos_cat,pos_num,neg_cat,neg_num], batch_size=1, use_multiprocessing=True)

                for idx in range(0, ori_cat.shape[0], self.voting):
                    # cnt = 0
                    #
                    # if self.voting == 1:
                    if pred[1][idx] < pred[0][idx]:
                        y_pred.append(1)

                        if y_true[int(idx / self.voting)] == 0:
                            conf_matrix["fp"] += 1

                        else:
                            conf_matrix["tp"] += 1

                    else:
                        y_pred.append(0)

                        if y_true[int(idx / self.voting)] == 1:
                            conf_matrix["fn"] += 1

                        else:
                            conf_matrix["tn"] += 1

                    # else:
                    #     for v in range(self.voting):
                    #         if pred[0][v + idx] < pred[1][v + idx]:
                    #             cnt += 1
                    #
                    #     if cnt > int(self.voting / 2):
                    #         y_pred.append(0)
                    #
                    #         if y_true[int(idx / self.voting)] == 0:
                    #             conf_matrix["tn"] += 1
                    #
                    #         else:
                    #             conf_matrix["fn"] += 1
                    #
                    #     else:
                    #         y_pred.append(1)
                    #
                    #         if y_true[int(idx / self.voting)] == 1:
                    #             conf_matrix["tp"] += 1
                    #
                    #         else:
                    #             conf_matrix["fp"] += 1

                not_nan = 0.00000000000000000001
                auc = round((conf_matrix['tn'] / (conf_matrix['tn'] + conf_matrix['fp'] + not_nan)
                             + conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fn'] + not_nan)) * 0.5 * 100, 2),
                tmp_auc.append(auc)
                tmp_negative_recall.append(
                    round((conf_matrix['tn'] / (conf_matrix['tn'] + conf_matrix['fp'] + not_nan)) * 100, 2))
                tmp_positive_recall.append(
                    round((conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fn'] + not_nan)) * 100, 2))
                tmp_positive_rate.append(
                    round(conf_matrix['fn'] / (conf_matrix['tn'] + conf_matrix['fn'] + not_nan) * 100, 2))
                tmp_negative_rate.append(round((conf_matrix['tn'] + conf_matrix['fn']) / y_true.shape[0] * 100, 2))

                print(
                    f"iter = {iter + 1} / {self.all_iter}, sub_iter = {sub_iter + 1} / {self.sub_iter}, auc = {auc}")

            res_auc.append(round(np.mean(tmp_auc), 2))
            res_negative_recall.append(round(np.mean(tmp_negative_recall), 2))
            res_positive_recall.append(round(np.mean(tmp_positive_recall), 2))
            res_negative_rate.append(round(np.mean(tmp_negative_rate), 2))
            res_positive_rate.append(round(np.mean(tmp_positive_rate), 2))

            print(f"lfe result_auc : {round((auc[0]), 2)}")
            print(f"lfe res_negative_recall : {round(np.mean(tmp_negative_recall), 2)}")
            print(f"lfe res_positive_recall : {round(np.mean(tmp_positive_recall), 2)}")
            print(f"lfe res_negative_rate : {round(np.mean(tmp_negative_rate), 2)}")
            print(f"lfe res_positive_rate : {round(np.mean(tmp_positive_rate), 2)}")

            with open(os.getcwd() + "/experiment/taiwan/triplet_train_0.7_test_0.5_auc.txt", 'w') as fp:
                for item in res_auc:
                    # write each item on a new line
                    fp.write(f"{item}\n")

                fp.close()

            with open(os.getcwd() + "/experiment/taiwan/triplet_train_0.7_test_0.5_res_negative_recall.txt",
                      'w') as fp:
                for item in res_negative_recall:
                    # write each item on a new line
                    fp.write(f"{item}\n")

                fp.close()

            with open(os.getcwd() + "/experiment/taiwan/triplet_train_0.7_test_0.5_res_positive_recall.txt",
                      'w') as fp:
                for item in res_positive_recall:
                    # write each item on a new line
                    fp.write(f"{item}\n")

                fp.close()

            with open(os.getcwd() + "/experiment/taiwan/triplet_train_0.7_test_0.5_res_negative_rate.txt", 'w') as fp:
                for item in res_negative_rate:
                    # write each item on a new line
                    fp.write(f"{item}\n")

                fp.close()

            with open(os.getcwd() + "/experiment/taiwan/triplet_train_0.7_test_0.5_res_positive_rate.txt", 'w') as fp:
                for item in res_positive_rate:
                    # write each item on a new line
                    fp.write(f"{item}\n")

                fp.close()

            print('Done')

                # print("good")
                # self.db_handler.learning_test_rebuilding(y_true, y_pred, iter, sub_iter, conf_matrix,
                #                          y_true.shape[0], cal, batch_id, model_no, model_full_name)
            # # #
            # self.db_handler.update_for_t3_detail(batch_id, model_no, iter, 'T')
            # self.db_handler.update_for_t3_detail(batch_id, model_no, iter, 'B')
            # end = time.time()
            # print(f"{end - start}")


class test_tuplet:
    def __init__(self, model_name, all_iter, batch_size, sub_iter, unit, epoch, bayes=False, db_handler=None,
                 cnn_kernel_size=1, cnn_filter=1, voting=1, distance=False, num_neg_sample=1):
        super(test_tuplet, self).__init__()
        self.model_name = model_name
        self.all_iter = all_iter
        self.batch_size = batch_size
        self.sub_iter = sub_iter
        self.unit = unit
        self.epoch = epoch
        self.bayes = bayes
        self.db_handler = db_handler
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter = cnn_filter
        self.voting = voting
        self.distance = distance
        self.num_neg_sample = num_neg_sample

    def _fit_with(self, train, test, margin, cnn_kernel_size, cnn_filter):
        cnn_kernel_size = int(cnn_kernel_size)
        cnn_filter = int(cnn_filter)
        network = self._get_network(train)

        y_true = test['label'][0, :]
        y_pred = []
        ori_cat = test['origin_cat'][0, :, :]
        ori_num = test['origin_num'][0, :, :]
        pos_cat = test['pos_cat'][0, :, :]
        pos_num = test['pos_num'][0, :, :]
        neg_cat = np.split(test['neg_cat'][0, :, :], self.num_neg_sample, axis=1)
        neg_cat = list(map(lambda e: np.squeeze(e, axis=1), neg_cat))
        neg_num = np.split(test['neg_num'][0, :, :], self.num_neg_sample, axis=1)
        neg_num = list(map(lambda e: np.squeeze(e, axis=1), neg_num))

        conf_matrix = {
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0}

        pred = network.predict([ori_cat, ori_num, pos_cat, pos_num, [neg_cat, neg_num]], batch_size=1,
                               use_multiprocessing=True)

        for idx in range(0, ori_cat.shape[0], self.voting):
            if self.voting == 1:
                if pred[1][idx] > pred[2][idx]:  # cosine simliarity
                    y_pred.append(1)

                    if y_true[int(idx / self.voting)] == 0:
                        conf_matrix["fp"] += 1

                    else:
                        conf_matrix["tp"] += 1

                else:
                    y_pred.append(0)

                    if y_true[int(idx / self.voting)] == 1:
                        conf_matrix["fn"] += 1

                    else:
                        conf_matrix["tn"] += 1

            else:
                cnt = 0
                for v in range(self.voting):
                    if pred[1][v + idx] > pred[2][v + idx]:  # cosine simliarity
                        cnt += 1

                if cnt > int(self.voting / 2):
                    y_pred.append(1)

                    if y_true[int(idx / self.voting)] == 0:
                        conf_matrix["fp"] += 1

                    else:
                        conf_matrix["tp"] += 1

                else:
                    y_pred.append(0)

                    if y_true[int(idx / self.voting)] == 1:
                        conf_matrix["fn"] += 1

                    else:
                        conf_matrix["tn"] += 1

        FN = conf_matrix['fn']
        TN = conf_matrix['tn']
        TP = conf_matrix['tp']
        FP = conf_matrix['fp']

        not_nan = 0.00000000000000001
        auc = round((TN / (TN + FP + not_nan) + TP / (TP + FN + not_nan)) * 0.5 * 100, 2)

        print(f"auc = {auc}")

        return auc

    def _get_network(self, train):
        neg_sample_cat = np.split(train['neg_cat'], self.num_neg_sample, axis=0)
        # neg_sample_cat = list(map(lambda e: np.squeeze(e, axis=0), neg_sample_cat))

        neg_sample_num = np.split(train['neg_num'], self.num_neg_sample, axis=0)
        # neg_sample_num = list(map(lambda e: np.squeeze(e, axis=0), neg_sample_num))

        network = TupletLoss([train['origin_cat'].shape[-1], train['origin_num'].shape[-1]],
                                  unit=self.unit, cnn_kernel_size=self.cnn_kernel_size, cnn_filter=self.cnn_filter,
                             num_neg_sample=self.num_neg_sample, batch_size=self.batch_size)
        network.compile(optimizer=optimizers.Adam(0.005))
        network.fit([train['origin_cat'], train['origin_num'],
                     train['pos_cat'], train['pos_num'],
                     [neg_sample_cat, neg_sample_num]], batch_size=self.batch_size, epochs=self.epoch,
                    verbose=1, use_multiprocessing=True)

        return network

    def _net_test(self):
        batch_id = 995
        sampling = ""
        model_full_name = f"{self.model_name}"
        model_no = self.db_handler.select_model_no(model_full_name)

        for iter in range(self.all_iter):
            # iter = 45
            # start = time.time()
            gen = TupletGenerator(self.batch_size, self.sub_iter, self.voting, self.num_neg_sample, iter)
            train,test = gen.get_tuplet_batch()

            if self.bayes:
                network = partial(self._fit_with, train, test)
                _bayes_opti(network)

                return

            else:
                network = self._get_network(train)

            for sub_iter in range(self.sub_iter):
                y_true = test['label'][sub_iter, :]
                y_pred = []
                ori_cat = test['origin_cat'][sub_iter, :, :]
                ori_num = test['origin_num'][sub_iter, :, :]
                pos_cat = test['pos_cat'][sub_iter, :, :]
                pos_num = test['pos_num'][sub_iter, :, :]
                neg_cat = np.split(test['neg_cat'][sub_iter, :, :], self.num_neg_sample, axis=1)
                neg_cat = list(map(lambda e: np.squeeze(e, axis=1), neg_cat))
                neg_num = np.split(test['neg_num'][sub_iter, :, :], self.num_neg_sample, axis=1)
                neg_num = list(map(lambda e: np.squeeze(e, axis=1), neg_num))
                cal = test['cal'][sub_iter, :, :]

                conf_matrix = {
                    "tn" : 0,
                    "fp" : 0,
                    "fn" : 0,
                    "tp" : 0 }

                pred = network.predict([ori_cat,ori_num,pos_cat,pos_num,[neg_cat,neg_num]], batch_size=1, use_multiprocessing=True)

                for idx in range(0, ori_cat.shape[0], self.voting):
                    if self.voting == 1:
                        if pred[1][idx] > pred[2][idx]: # cosine simliarity
                            y_pred.append(1)

                            if y_true[int(idx / self.voting)] == 0:
                                conf_matrix["fp"] += 1

                            else:
                                conf_matrix["tp"] += 1

                        else:
                            y_pred.append(0)

                            if y_true[int(idx / self.voting)] == 1:
                                conf_matrix["fn"] += 1

                            else:
                                conf_matrix["tn"] += 1

                    else:
                        cnt = 0
                        for v in range(self.voting):
                            if pred[1][v + idx] > pred[2][v + idx]: # cosine simliarity
                                cnt += 1

                        if cnt > int(self.voting / 2):
                            y_pred.append(1)

                            if y_true[int(idx / self.voting)] == 0:
                                conf_matrix["fp"] += 1

                            else:
                                conf_matrix["tp"] += 1

                        else:
                            y_pred.append(0)

                            if y_true[int(idx / self.voting)] == 1:
                                conf_matrix["fn"] += 1

                            else:
                                conf_matrix["tn"] += 1

                self.db_handler.learning_test_rebuilding(y_true, y_pred, iter, sub_iter, conf_matrix,
                                         y_true.shape[0], cal, batch_id, model_no, model_full_name)
            # #
            self.db_handler.update_for_t3_detail(batch_id, model_no, iter, 'T')
            self.db_handler.update_for_t3_detail(batch_id, model_no, iter, 'B')
            # print("good")
            # end = time.time()
            # print(f"{end - start}")


class test_lfe:
    def __init__(self, model_name, all_iter, batch_size, sub_iter, unit, epoch, bayes=False, db_handler=None, margin=0,
                 bigbang_train = False, dataset=None, cnn_kernel_size=1, cnn_filter=1, voting=1, distance=False):
        super(test_lfe, self).__init__()
        self.model_name = model_name
        self.all_iter = all_iter
        self.batch_size = batch_size
        self.sub_iter = sub_iter
        self.unit = unit
        self.epoch = epoch
        self.bayes = bayes
        self.bigbang_train = bigbang_train
        self.db_handler = db_handler
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter = cnn_filter
        self.voting = voting
        self.distance = distance
        self.margin = margin
        self.bigbang_train = bigbang_train
        self.dataset = dataset

    def _fit_with(self, train, test, margin, cnn_filter):
        cnn_filter = int(self.cnn_filter)
        network, cluster = self._get_network(train)

        y_true = test['label'][0, :]
        y_pred = []
        cat = test['cat'][0, :]
        num = test['num'][0, :]

        conf_matrix = {
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0}

        pred = network.predict([cat, num], batch_size=1, use_multiprocessing=True)
        res = cluster.predict(pred)

        for idx in range(0, cat.shape[0], self.voting):
            if res[idx] == 0:
                y_pred.append(0)

                if y_true[int(idx / self.voting)] == 0:
                    conf_matrix["tn"] += 1

                else:
                    conf_matrix["fn"] += 1

            else:
                y_pred.append(1)

                if y_true[int(idx / self.voting)] == 1:
                    conf_matrix["tp"] += 1

                else:
                    conf_matrix["fp"] += 1


        FN = conf_matrix['fn']
        TN = conf_matrix['tn']
        TP = conf_matrix['tp']
        FP = conf_matrix['fp']

        not_nan = 0.00000000000000001
        auc = round((TN / (TN + FP + not_nan) + TP / (TP + FN + not_nan)) * 0.5 * 100, 2)

        print(f"auc = {auc}")

        return auc

    def _get_network(self, train):
        os.environ['PYTHONHASHSEED'] = str(1)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        tf.random.set_seed(1)
        np.random.seed(1)
        python_random.seed(1)
        network = LiftedStructLoss([train['cat'].shape[-1], train['num'].shape[-1]],
                                  unit=self.unit, cnn_kernel_size=self.cnn_kernel_size, cnn_filter=self.cnn_filter,
                                   margin=self.margin, batch_size=self.batch_size, dataset = self.dataset)
        network.compile(optimizer=optimizers.Adam(0.0005)) # ibk : 0.0001, taiwan : 0.0001, leding_club : 0.0005
        network.fit([train['cat'], train['num']],train['label'], batch_size=self.batch_size,
                    epochs=self.epoch, verbose=1)
        pred = network.predict([train['cat'], train['num']])
        # self._tsne(pred, train['label'], True)
        # cluster = KMeans(2, n_init=200, random_state=0)
        cluster = GaussianMixture(2, covariance_type='diag', random_state=1, max_iter=300, n_init=30, tol=1e-10) # ibk : diag, taiwan : spherical, lending_club : tied
        # cluster = BayesianGaussianMixture(n_components=2, random_state=42, covariance_type='full', max_iter=300, n_init=50)
        cluster.fit(pred, train['label']) # diag(0.0005) diag(0.0003)

        return network, cluster

    def _tsne(self, data, label=None, deep_metric=False):
        train_emb = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=0, perplexity=50).fit_transform(data)
        neg_index = np.where(label == 0)
        pos_index = np.where(label == 1)

        x_neg = train_emb[neg_index, 0]
        y_neg = train_emb[neg_index, 1]
        # z_neg = train_emb[neg_index, 2]
        #
        x_pos = train_emb[pos_index, 0]
        y_pos = train_emb[pos_index, 1]
        # z_pos = train_emb[pos_index, 2]

        fig = plt.figure(figsize=(9, 5))
        fig.set_size_inches(10, 10)
        plt.scatter(x_neg, y_neg, c='r', label='negative class(0)')
        plt.scatter(x_pos, y_pos, c='b', label='positive class(1)')
        # ax = fig.add_subplot(projection='3d')

        # ax.scatter(x_neg, y_neg, z_neg, c= 'r', s = 5, marker='+')
        # ax.scatter(x_pos, y_pos, z_pos, c= 'b', s = 5, marker='.')
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # ax.view_init(0, 270)

        if deep_metric:
            title = "using deep metric"

        else:
            title = "origin data"

        # plt.legend()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.getcwd() + f"/experiment/{title}")
        plt.show()
        plt.close()

    def _calculate(self, cal, predict, label):
        temp = cal
        temp['pred'] = predict
        temp['y2'] = np.logical_not(label).astype(int)
        temp['pred2'] = np.logical_not(predict).astype(int)

        temp['[기존]이자수익'] = temp['대출실행금액'] * temp['금리'] * 0.01 * temp['y2']
        temp['[기존]원금손실'] = temp['대출실행금액'] * label

        temp['[예상]대출금액'] = temp['대출실행금액'] * temp['pred2']
        temp['[예상]이자수익'] = temp['대출실행금액'] * temp['금리'] * 0.01 * temp['pred2'] * temp['y2']
        temp['[예상]원금손실'] = temp['대출실행금액'] * temp['pred2'] * label

        return temp

    def _net_test(self):
        batch_id = 983
        model_full_name = f"{self.model_name}"
        model_no = self.db_handler.select_model_no(model_full_name)

        res_auc = []
        res_negative_recall = []
        res_positive_recall = []
        res_positive_rate = []
        res_negative_rate = []
        res_profit = []
        res_profit_diff = []

        for iter in range(self.all_iter):
            # iter = iter + 12
            # start = time.time()
            gen = LFEGenerator(self.batch_size, self.sub_iter, iter, self.bigbang_train, self.dataset)
            train, test = gen.get_lfe_batch()
            # data = np.concatenate((train['cat'],train['num']), axis=1)
            # self._tsne(data, train['label'])

            if self.bayes:
                model = partial(self._fit_with, train, test)
                _bayes_opti(model)

                return

            else:
                model, cluster = self._get_network(train)

            if self.bigbang_train:
                tmp_auc = []
                tmp_negative_recall = []
                tmp_positive_recall = []
                tmp_positive_rate = []
                tmp_negative_rate = []
                tmp_profit = []
                tmp_profit_diff = []

                for sub_iter in range(self.sub_iter):
                    cal = pd.DataFrame(test['cal'][sub_iter, :, :], columns=['대출실행금액', '금리'])
                    y_true = test['label'][sub_iter, :]
                    y_pred = []
                    cat = test['cat'][sub_iter, :]
                    num = test['num'][sub_iter, :]

                    conf_matrix = {
                        "tn": 0,
                        "fp": 0,
                        "fn": 0,
                        "tp": 0}

                    pred = model.predict([cat, num])
                    res = cluster.predict(pred)
                    neg = 0

                    if np.count_nonzero(res == 0) < np.count_nonzero(res == 1):
                        neg = 1

                    for idx in range(y_true.shape[0]):
                        if res[idx] == neg:
                            y_pred.append(0)

                            if y_true[idx] == 0:
                                conf_matrix["tn"] += 1

                            else:
                                conf_matrix["fn"] += 1

                        else:
                            y_pred.append(1)

                            if y_true[idx] == 1:
                                conf_matrix["tp"] += 1

                            else:
                                conf_matrix["fp"] += 1

                    # print(conf_matrix, y_true)
                    not_nan = 0.00000000000000000001
                    auc = round((conf_matrix['tn'] / (conf_matrix['tn'] + conf_matrix['fp'] + not_nan)
                                 + conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fn'] + not_nan)) * 0.5 * 100,
                                2)

                    result = self._calculate(cal, np.array(y_pred), y_true)
                    total_loan = result['대출실행금액'].sum()
                    before_inter = result['[기존]이자수익'].sum()
                    after_inter = result['[예상]이자수익'].sum()

                    after_total_loan = result['[예상]대출금액'].sum()
                    before_loss = result['[기존]원금손실'].sum()
                    after_loss = result['[예상]원금손실'].sum()
                    gross_profit_rate_before = round((before_inter - before_loss) / total_loan * 100, 2)
                    gross_profit_rate = round((after_inter - after_loss) / (after_total_loan + not_nan) * 100, 2)

                    tmp_auc.append(auc)
                    tmp_negative_recall.append(
                        round((conf_matrix['tn'] / (conf_matrix['tn'] + conf_matrix['fp'] + not_nan)) * 100, 2))
                    tmp_positive_recall.append(
                        round((conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fn'] + not_nan)) * 100, 2))
                    tmp_positive_rate.append(
                        round(conf_matrix['fn'] / (conf_matrix['tn'] + conf_matrix['fn'] + not_nan) * 100, 2))
                    tmp_negative_rate.append(round((conf_matrix['tn'] + conf_matrix['fn']) / y_true.shape[0] * 100, 2))
                    tmp_profit.append(gross_profit_rate)
                    tmp_profit_diff.append(gross_profit_rate - gross_profit_rate_before)

                    print(
                        f"iter = {iter + 1} / {self.all_iter}, sub_iter = {sub_iter + 1} / {self.sub_iter}, auc = {auc}")

                res_auc.append(round(np.mean(tmp_auc), 2))
                res_negative_recall.append(round(np.mean(tmp_negative_recall), 2))
                res_positive_recall.append(round(np.mean(tmp_positive_recall), 2))
                res_negative_rate.append(round(np.mean(tmp_negative_rate), 2))
                res_positive_rate.append(round(np.mean(tmp_positive_rate), 2))
                res_profit.append(round(np.mean(tmp_profit), 2))
                res_profit_diff.append(round(np.mean(tmp_profit_diff), 2))

                print(f"lfe result_auc : {round((res_auc[-1]), 2)}")
                print(f"lfe res_negative_recall : {round(np.mean(tmp_negative_recall), 2)}")
                print(f"lfe res_positive_recall : {round(np.mean(tmp_positive_recall), 2)}")
                print(f"lfe res_negative_rate : {round(np.mean(tmp_negative_rate), 2)}")
                print(f"lfe res_positive_rate : {round(np.mean(tmp_positive_rate), 2)}")
                print(f"lfe res_profit : {res_profit[-1]}")
                print(f"lfe res_profit_diff : {res_profit_diff[-1]}")

            else:
                y_true = np.array(test['label'])
                y_pred = []
                data = test['data']

                conf_matrix = {
                    "tn": 0,
                    "fp": 0,
                    "fn": 0,
                    "tp": 0}

                res = model.predict(data)

                for idx in range(y_true.shape[0]):
                    if res[idx] == 0:
                        y_pred.append(0)

                        if y_true[idx] == 0:
                            conf_matrix["tn"] += 1

                        else:
                            conf_matrix["fn"] += 1

                    else:
                        y_pred.append(1)

                        if y_true[idx] == 1:
                            conf_matrix["tp"] += 1

                        else:
                            conf_matrix["fp"] += 1

                not_nan = 0.00000000000000000001
                auc = round((conf_matrix['tn'] / (conf_matrix['tn'] + conf_matrix['fp'] + not_nan)
                             + conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fn'] + not_nan)) * 0.5 * 100, 2)
                print(f"iter = {iter + 1} / {self.all_iter}, auc = {auc}")

                res_auc.append(round((auc[0]), 2))
                res_negative_recall.append(
                    round((conf_matrix['tn'] / (conf_matrix['tn'] + conf_matrix['fp'] + not_nan)) * 100, 2))
                res_positive_recall.append(
                    round((conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fn'] + not_nan)) * 100, 2))
                res_negative_rate.append(
                    round(conf_matrix['fn'] / (conf_matrix['tn'] + conf_matrix['fn'] + not_nan) * 100, 2))
                res_positive_rate.append(round((conf_matrix['tn'] + conf_matrix['fn']) / y_true.shape[0] * 100, 2))

                print(f"lfe result_auc : {round((auc[0]), 2)}")
                print(
                    f"lfe res_negative_recall : {round(conf_matrix['tn'] / (conf_matrix['tn'] + conf_matrix['fp'] + not_nan), 2)}")
                print(
                    f"lfe res_positive_recall : {round(conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fn'] + not_nan), 2)}")

        with open(os.getcwd() + "/experiment/ibk/lfe_train_0.7_test_0.5_auc.txt", 'w') as fp:
            for item in res_auc:
                # write each item on a new line
                fp.write(f"{item}\n")

            fp.close()

        with open(os.getcwd() + "/experiment/ibk/lfe_train_0.7_test_0.5_res_negative_recall.txt",
                  'w') as fp:
            for item in res_negative_recall:
                # write each item on a new line
                fp.write(f"{item}\n")

            fp.close()

        with open(os.getcwd() + "/experiment/ibk/lfe_train_0.7_test_0.5_res_positive_recall.txt",
                  'w') as fp:
            for item in res_positive_recall:
                # write each item on a new line
                fp.write(f"{item}\n")

            fp.close()

        with open(os.getcwd() + "/experiment/ibk/lfe_train_0.7_test_0.5_res_negative_rate.txt", 'w') as fp:
            for item in res_negative_rate:
                # write each item on a new line
                fp.write(f"{item}\n")

            fp.close()

        with open(os.getcwd() + "/experiment/ibk/lfe_train_0.7_test_0.5_res_positive_rate.txt", 'w') as fp:
            for item in res_positive_rate:
                # write each item on a new line
                fp.write(f"{item}\n")

            fp.close()

        with open(os.getcwd() + "/experiment/ibk/lfe_train_0.7_test_0.5_res_profit.txt", 'w') as fp:
            for item in res_profit:
                # write each item on a new line
                fp.write(f"{item}\n")

            fp.close()

        with open(os.getcwd() + "/experiment/ibk/lfe_train_0.7_test_0.5_res_profit_diff.txt", 'w') as fp:
            for item in res_profit_diff:
                # write each item on a new line
                fp.write(f"{item}\n")

            fp.close()

        print('Done')

            #     self.db_handler.learning_test_rebuilding(y_true, y_pred, iter, sub_iter, conf_matrix,
            #                              y_true.shape[0], cal, batch_id, model_no, model_full_name)
            # # #
            # self.db_handler.update_for_t3_detail(batch_id, model_no, iter, 'T')
            # self.db_handler.update_for_t3_detail(batch_id, model_no, iter, 'B')
            # print("good")
            # end = time.time()
            # print(f"{end - start}")


class test_siamese:
    def __init__(self, model_name, all_iter, batch_size, sub_iter, unit, epoch, bayes=False, db_handler=None,
                 margin=0, cnn_kernel_size=1, cnn_filter=1, voting=1, distance=False, sharpness_param=0):
        super(test_siamese, self).__init__()
        self.model_name = model_name
        self.all_iter = all_iter
        self.batch_size = batch_size
        self.sub_iter = sub_iter
        self.unit = unit
        self.epoch = epoch
        self.bayes = bayes
        self.db_handler = db_handler
        self.margin = margin
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter = cnn_filter
        self.voting = voting
        self.distance = distance
        self.sharpness_param = sharpness_param

    def _fit_with(self, train, test, margin, cnn_kernel_size, cnn_filter, sharpness_param):
        cnn_kernel_size = int(self.cnn_kernel_size)
        cnn_filter = int(self.cnn_filter)
        network = self._get_siamese_network(train)

        y_true = test['label'][0, :]
        y_pred = []
        ori_cat = test['origin_cat'][0, :, :]
        ori_num = test['origin_num'][0, :, :]
        disc_cat = test['disc_cat'][0, :, :]
        disc_num = test['disc_num'][0, :, :]
        cal = test['cal'][0, :, :]

        conf_matrix = {
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0}

        pred = network.predict([ori_cat, ori_num, disc_cat, disc_num], batch_size=1, use_multiprocessing=True)

        for idx in range(0, ori_cat.shape[0], self.voting):
            if self.voting == 1:
                if pred[idx] < self.margin:
                    y_pred.append(int(y_true[int(idx / self.voting)]))

                    if y_true[int(idx / self.voting)] == 0:
                        conf_matrix["tn"] += 1

                    else:
                        conf_matrix["tp"] += 1

                else:
                    y_pred.append(int(not (y_true[int(idx / self.voting)])))

                    if y_true[int(idx / self.voting)] == 0:
                        conf_matrix["fp"] += 1

                    else:
                        conf_matrix["fn"] += 1

            else:
                cnt = 0
                for v in range(self.voting):
                    if pred[1][v + idx] > pred[2][v + idx]:  # cosine simliarity
                        cnt += 1

                if cnt > int(self.voting / 2):
                    y_pred.append(1)

                    if y_true[int(idx / self.voting)] == 0:
                        conf_matrix["fp"] += 1

                    else:
                        conf_matrix["tp"] += 1

                else:
                    y_pred.append(0)

                    if y_true[int(idx / self.voting)] == 1:
                        conf_matrix["fn"] += 1

                    else:
                        conf_matrix["tn"] += 1

        FN = conf_matrix['fn']
        TN = conf_matrix['tn']
        TP = conf_matrix['tp']
        FP = conf_matrix['fp']

        not_nan = 0.00000000000000001
        auc = round((TN / (TN + FP + not_nan) + TP / (TP + FN + not_nan)) * 0.5 * 100, 2)

        return auc

    def _get_siamese_network(self, train):
        subject_param = (train['label'] == train['disc_label'])
        subject_param_sim = subject_param.astype(int)
        subject_param_dis = np.logical_not(subject_param).astype(int)

        network = SiameseNetwork([train['origin_cat'].shape[-1], train['origin_num'].shape[-1]], margin=self.margin,
                                  unit=self.unit, cnn_kernel_size=self.cnn_kernel_size, cnn_filter=self.cnn_filter, batch_size=self.batch_size,
                                 sharpness_param = self.sharpness_param)
        network.compile(optimizer=optimizers.Adam(0.0005))
        network.fit([train['origin_cat'], train['origin_num'],
                     train['disc_cat'], train['disc_num']], [subject_param_sim, subject_param_dis],
                    batch_size=self.batch_size, epochs=self.epoch,
                    verbose=1, use_multiprocessing=True)

        return network

    def _net_test(self):
        batch_id = 984
        sampling = ""
        model_full_name = f"{self.model_name}"
        model_no = self.db_handler.select_model_no(self.model_name)

        res_auc = []

        for iter in range(self.all_iter):
            # iter = iter + 75
            # start = time.time()
            gen = SiameseGenerator(self.batch_size, self.sub_iter, self.voting, iter)
            train,test = gen.get_siamese_batch()

            if self.bayes:
                network = partial(self._fit_with, train, test)
                _bayes_opti(network)

                return

            else:
                network = self._get_siamese_network(train)

            tmp_auc = []

            for sub_iter in range(self.sub_iter):
                y_true = test['label'][sub_iter, :]
                y_pred = []
                ori_cat = test['origin_cat'][sub_iter, :, :]
                ori_num = test['origin_num'][sub_iter, :, :]
                disc_cat = test['disc_cat'][sub_iter, :, :]
                disc_num = test['disc_num'][sub_iter, :, :]
                # cal = test['cal'][sub_iter, :, :]

                conf_matrix = {
                    "tn" : 0,
                    "fp" : 0,
                    "fn" : 0,
                    "tp" : 0 }

                pred = network.predict([ori_cat,ori_num,disc_cat,disc_num], batch_size=1, use_multiprocessing=True)

                for idx in range(ori_cat.shape[0]):
                    if pred[idx] < self.margin:
                        y_pred.append(int(y_true[idx]))

                        if y_true[idx] == 0:
                            conf_matrix["tn"] += 1

                        else:
                            conf_matrix["tp"] += 1

                    else:
                        y_pred.append(int(not(y_true[idx])))

                        if y_true[idx] == 0:
                            conf_matrix["fp"] += 1

                        else:
                            conf_matrix["fn"] += 1

                not_nan = 0.00000000000000000001
                auc = round((conf_matrix['tn'] / (conf_matrix['tn'] + conf_matrix['fp'] + not_nan)
                             + conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fn'] + not_nan)) * 0.5 * 100,
                            2)
                print(
                    f"iter = {iter + 1} / {self.all_iter}, sub_iter = {sub_iter + 1} / {self.sub_iter}, auc = {auc}")

                tmp_auc.append(auc)


            res_auc.append(round(np.mean(tmp_auc), 2))
            print(f"siamese result_auc : {round((res_auc[-1]), 2)}")
            with open(os.getcwd() + "/experiment/taiwan/siamese_train_0.7_test_0.5_auc.txt", 'w') as fp:
                for item in res_auc:
                    # write each item on a new line
                    fp.write(f"{item}\n")

                fp.close()

                # print("good")
            #     self.db_handler.learning_test_rebuilding(y_true, y_pred, iter, sub_iter, conf_matrix,
            #                              y_true.shape[0], cal, batch_id, model_no, model_full_name)
            # # #
            # self.db_handler.update_for_t3_detail(batch_id, model_no, iter, 'T')
            # self.db_handler.update_for_t3_detail(batch_id, model_no, iter, 'B')
            # # end = time.time()
            # print(f"{end - start}")


class test_xgboost:
    def __init__(self, model_name, all_iter, sub_iter, db_handler,oversampling, bia_bang_train, dataset, bayes=False):
        super(test_xgboost, self).__init__()
        self.model_name = model_name
        self.all_iter = all_iter
        self.sub_iter = sub_iter
        self.bayes = bayes
        self.db_handler = db_handler
        self.oversampling = oversampling
        self.bigbang_train = bia_bang_train
        self.dataset = dataset

    def _get_model(self, train):
        # model = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0, max_depth=2, scale_pos_weight=15)
        model = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0)
        data = train['data']
        label = train['label']
        xgb_model = model.fit(data, label)

        return xgb_model

    def _calculate(self, cal, predict, label):
        temp = cal
        temp['pred'] = predict
        temp['y2'] = np.logical_not(label).astype(int)
        temp['pred2'] = np.logical_not(predict).astype(int)

        temp['[기존]이자수익'] = temp['loan_amnt'] * temp['int_rate'] * 0.01 * temp['y2']
        temp['[기존]원금손실'] = temp['loan_amnt'] * label

        temp['[예상]대출금액'] = temp['loan_amnt'] * temp['pred2']
        temp['[예상]이자수익'] = temp['loan_amnt'] * temp['int_rate'] * 0.01 * temp['pred2'] * temp['y2']
        temp['[예상]원금손실'] = temp['loan_amnt'] * temp['pred2'] * label

        return temp

    def _net_test(self):
        batch_id = 983

        if self.oversampling is not None:
            model_full_name = f"{self.model_name}-{self.oversampling}"

        else:
            model_full_name = f"{self.model_name}"

        model_no = self.db_handler.select_model_no(model_full_name)

        res_auc = []
        res_negative_recall = []
        res_positive_recall = []
        res_positive_rate = []
        res_negative_rate = []
        res_profit = []
        res_profit_diff = []

        for iter in range(self.all_iter):
            # iter = iter + 75
            # start = time.time()
            gen = CompareGenerator(self.sub_iter, iter, self.oversampling, self.bigbang_train, self.dataset)
            train,test = gen.get_compare_batch()
            model = self._get_model(train)

            if self.bigbang_train:
                tmp_auc = []
                tmp_negative_recall = []
                tmp_positive_recall = []
                tmp_positive_rate = []
                tmp_negative_rate = []
                tmp_profit = []
                tmp_profit_diff = []

                for sub_iter in range(self.sub_iter):
                    cal = pd.DataFrame(test['cal'][sub_iter, :, :], columns=['loan_amnt', 'int_rate'])
                    y_true = test['label'][sub_iter, :]
                    y_pred = []
                    data = test['data'][sub_iter, :]

                    conf_matrix = {
                        "tn" : 0,
                        "fp" : 0,
                        "fn" : 0,
                        "tp" : 0 }

                    res = model.predict(data)

                    for idx in range(y_true.shape[0]):
                        if res[idx] == 0:
                            y_pred.append(0)

                            if y_true[idx] == 0:
                                conf_matrix["tn"] += 1

                            else:
                                conf_matrix["fn"] += 1

                        else:
                            y_pred.append(1)

                            if y_true[idx] == 1:
                                conf_matrix["tp"] += 1

                            else:
                                conf_matrix["fp"] += 1

                    not_nan = 0.00000000000000000001
                    auc = round((conf_matrix['tn'] / (conf_matrix['tn'] + conf_matrix['fp'] + not_nan)
                                 + conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fn'] + not_nan)) * 0.5 * 100, 2),
                    result = self._calculate(cal, np.array(y_pred), y_true)
                    total_loan = result['loan_amnt'].sum()
                    before_inter = result['[기존]이자수익'].sum()
                    after_inter = result['[예상]이자수익'].sum()

                    after_total_loan = result['[예상]대출금액'].sum()
                    before_loss = result['[기존]원금손실'].sum()
                    after_loss = result['[예상]원금손실'].sum()
                    gross_profit_rate_before = round((before_inter - before_loss) / total_loan * 100, 2)
                    gross_profit_rate=round((after_inter - after_loss) / (after_total_loan + not_nan) * 100, 2)

                    tmp_auc.append(auc)
                    tmp_negative_recall.append(round((conf_matrix['tn'] / (conf_matrix['tn'] + conf_matrix['fp'] + not_nan)) * 100, 2))
                    tmp_positive_recall.append(round((conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fn'] + not_nan)) * 100, 2))
                    tmp_positive_rate.append(round(conf_matrix['fn'] / (conf_matrix['tn'] + conf_matrix['fn'] + not_nan) * 100, 2))
                    tmp_negative_rate.append(round((conf_matrix['tn'] + conf_matrix['fn']) / y_true.shape[0] * 100, 2))
                    tmp_profit.append(gross_profit_rate)
                    tmp_profit_diff.append(gross_profit_rate - gross_profit_rate_before)

                    print(f"iter = {iter + 1} / {self.all_iter}, sub_iter = {sub_iter + 1} / {self.sub_iter}, auc = {auc}")
                    print(conf_matrix)

                res_auc.append(round(np.mean(tmp_auc), 2))
                res_negative_recall.append(round(np.mean(tmp_negative_recall), 2))
                res_positive_recall.append(round(np.mean(tmp_positive_recall), 2))
                res_negative_rate.append(round(np.mean(tmp_negative_rate), 2))
                res_positive_rate.append(round(np.mean(tmp_positive_rate), 2))
                res_profit.append(round(np.mean(tmp_profit), 2))
                res_profit_diff.append(round(np.mean(tmp_profit_diff), 2))

                print(f"xgboost result_auc : {(res_auc[-1])}")
                print(f"xgboost res_negative_recall : {res_negative_recall[-1]}")
                print(f"xgboost res_positive_recall : {res_positive_recall[-1]}")
                print(f"xgboost res_negative_rate : {res_negative_rate[-1]}")
                print(f"xgboost res_positive_rate : {res_positive_rate[-1]}")
                print(f"xgboost res_profit : {res_profit[-1]}")
                print(f"xgboost res_profit_diff : {res_profit_diff[-1]}")

            else:
                y_true = np.array(test['label'])
                y_pred = []
                data = test['data']

                conf_matrix = {
                    "tn": 0,
                    "fp": 0,
                    "fn": 0,
                    "tp": 0}

                res = model.predict(data)

                for idx in range(y_true.shape[0]):
                    if res[idx] == 0:
                        y_pred.append(0)

                        if y_true[idx] == 0:
                            conf_matrix["tn"] += 1

                        else:
                            conf_matrix["fn"] += 1

                    else:
                        y_pred.append(1)

                        if y_true[idx] == 1:
                            conf_matrix["tp"] += 1

                        else:
                            conf_matrix["fp"] += 1

                not_nan = 0.00000000000000000001
                auc = round((conf_matrix['tn'] / (conf_matrix['tn'] + conf_matrix['fp'] + not_nan)
                             + conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fn'] + not_nan)) * 0.5 * 100, 2),
                print(f"iter = {iter + 1} / {self.all_iter}, auc = {auc}")

                res_auc.append(round((auc[0]), 2))
                res_negative_recall.append(round((conf_matrix['tn'] / (conf_matrix['tn'] + conf_matrix['fp'] + not_nan)) * 100, 2))
                res_positive_recall.append(round((conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fn'] + not_nan)) * 100, 2))
                res_negative_rate.append(round(conf_matrix['fn'] / (conf_matrix['tn'] + conf_matrix['fn'] + not_nan) * 100, 2))
                res_positive_rate.append(round((conf_matrix['tn'] + conf_matrix['fn']) / y_true.shape[0] * 100, 2))

                print(f"xgboost result_auc : {round((auc[0]), 2)}")
                print(f"xgboost res_negative_recall : {round(conf_matrix['tn'] / (conf_matrix['tn'] + conf_matrix['fp'] + not_nan), 2)}")
                print(f"xgboost res_positive_recall : {round(conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fn'] + not_nan), 2)}")

        with open(os.getcwd() + "/experiment/lending_club/xgboost_train_0.7_test_0.5_auc.txt", 'w') as fp:
            for item in res_auc:
                # write each item on a new line
                fp.write(f"{item}\n")

            fp.close()

        with open(os.getcwd() + "/experiment/lending_club/xgboost_train_0.7_test_0.5_res_negative_recall.txt", 'w') as fp:
            for item in res_negative_recall:
                # write each item on a new line
                fp.write(f"{item}\n")

            fp.close()

        with open(os.getcwd() + "/experiment/lending_club/xgboost_train_0.7_test_0.5_res_positive_recall.txt", 'w') as fp:
            for item in res_positive_recall:
                # write each item on a new line
                fp.write(f"{item}\n")

            fp.close()

        with open(os.getcwd() + "/experiment/lending_club/xgboost_train_0.7_test_0.5_res_negative_rate.txt", 'w') as fp:
            for item in res_negative_rate:
                # write each item on a new line
                fp.write(f"{item}\n")

            fp.close()

        with open(os.getcwd() + "/experiment/lending_club/xgboost_train_0.7_test_0.5_res_positive_rate.txt", 'w') as fp:
            for item in res_positive_rate:
                # write each item on a new line
                fp.write(f"{item}\n")

            fp.close()

        with open(os.getcwd() + "/experiment/lending_club/xgboost_train_0.7_test_0.5_res_profit.txt", 'w') as fp:
            for item in res_profit:
                # write each item on a new line
                fp.write(f"{item}\n")

            fp.close()

        with open(os.getcwd() + "/experiment/lending_club/xgboost_train_0.7_test_0.5_res_profit_diff.txt", 'w') as fp:
            for item in res_profit_diff:
                # write each item on a new line
                fp.write(f"{item}\n")

            fp.close()

        print('Done')

                # self.db_handler.learning_test_rebuilding(y_true, y_pred, iter, sub_iter, conf_matrix,
                #                          y_true.shape[0], cal, batch_id, model_no, model_full_name)

            # self.db_handler.update_for_t3_detail(batch_id, model_no, iter, 'T')
            # self.db_handler.update_for_t3_detail(batch_id, model_no, iter, 'B')
            # print("good")
            # end = time.time()
            # print(f"{end - start}")


class test_randomforest:
    def __init__(self, model_name, all_iter, sub_iter, db_handler,oversampling, bia_bang_train, dataset, bayes=False):
        super(test_randomforest, self).__init__()
        self.model_name = model_name
        self.all_iter = all_iter
        self.sub_iter = sub_iter
        self.bayes = bayes
        self.db_handler = db_handler
        self.oversampling = oversampling
        self.bigbang_train = bia_bang_train
        self.dataset = dataset

    def _get_model(self, train):
        model = RandomForestClassifier(random_state=0)
        data = train['data']
        label = train['label']
        rf_model = model.fit(data, label)

        return rf_model

    def _calculate(self, cal, predict, label):
        temp = cal
        temp['pred'] = predict
        temp['y2'] = np.logical_not(label).astype(int)
        temp['pred2'] = np.logical_not(predict).astype(int)

        temp['[기존]이자수익'] = temp['loan_amnt'] * temp['int_rate'] * 0.01 * temp['y2']
        temp['[기존]원금손실'] = temp['loan_amnt'] * label

        temp['[예상]대출금액'] = temp['loan_amnt'] * temp['pred2']
        temp['[예상]이자수익'] = temp['loan_amnt'] * temp['int_rate'] * 0.01 * temp['pred2'] * temp['y2']
        temp['[예상]원금손실'] = temp['loan_amnt'] * temp['pred2'] * label

        return temp

    def _net_test(self):
        res_auc = []
        res_negative_recall = []
        res_positive_recall = []
        res_positive_rate = []
        res_negative_rate = []
        res_profit = []
        res_profit_diff = []

        for iter in range(self.all_iter):
            # iter = iter + 75
            # start = time.time()
            gen = CompareGenerator(self.sub_iter, iter, self.oversampling, self.bigbang_train, self.dataset)
            train, test = gen.get_compare_batch()
            model = self._get_model(train)

            if self.bigbang_train:
                tmp_auc = []
                tmp_negative_recall = []
                tmp_positive_recall = []
                tmp_positive_rate = []
                tmp_negative_rate = []
                tmp_profit = []
                tmp_profit_diff = []

                for sub_iter in range(self.sub_iter):
                    cal = pd.DataFrame(test['cal'][sub_iter, :, :], columns=['loan_amnt', 'int_rate'])
                    y_true = test['label'][sub_iter, :]
                    y_pred = []
                    data = test['data'][sub_iter, :]

                    conf_matrix = {
                        "tn": 0,
                        "fp": 0,
                        "fn": 0,
                        "tp": 0}

                    res = model.predict(data)

                    for idx in range(y_true.shape[0]):
                        if res[idx] == 0:
                            y_pred.append(0)

                            if y_true[idx] == 0:
                                conf_matrix["tn"] += 1

                            else:
                                conf_matrix["fn"] += 1

                        else:
                            y_pred.append(1)

                            if y_true[idx] == 1:
                                conf_matrix["tp"] += 1

                            else:
                                conf_matrix["fp"] += 1

                    not_nan = 0.00000000000000000001
                    auc = round((conf_matrix['tn'] / (conf_matrix['tn'] + conf_matrix['fp'] + not_nan)
                                 + conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fn'] + not_nan)) * 0.5 * 100,
                                2),
                    result = self._calculate(cal, np.array(y_pred), y_true)
                    total_loan = result['loan_amnt'].sum()
                    before_inter = result['[기존]이자수익'].sum()
                    after_inter = result['[예상]이자수익'].sum()

                    after_total_loan = result['[예상]대출금액'].sum()
                    before_loss = result['[기존]원금손실'].sum()
                    after_loss = result['[예상]원금손실'].sum()
                    gross_profit_rate_before = round((before_inter - before_loss) / total_loan * 100, 2)
                    gross_profit_rate = round((after_inter - after_loss) / (after_total_loan + not_nan) * 100, 2)

                    tmp_auc.append(auc)
                    tmp_negative_recall.append(
                        round((conf_matrix['tn'] / (conf_matrix['tn'] + conf_matrix['fp'] + not_nan)) * 100, 2))
                    tmp_positive_recall.append(
                        round((conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fn'] + not_nan)) * 100, 2))
                    tmp_positive_rate.append(
                        round(conf_matrix['fn'] / (conf_matrix['tn'] + conf_matrix['fn'] + not_nan) * 100, 2))
                    tmp_negative_rate.append(round((conf_matrix['tn'] + conf_matrix['fn']) / y_true.shape[0] * 100, 2))
                    tmp_profit.append(gross_profit_rate)
                    tmp_profit_diff.append(gross_profit_rate - gross_profit_rate_before)

                    print(
                        f"iter = {iter + 1} / {self.all_iter}, sub_iter = {sub_iter + 1} / {self.sub_iter}, auc = {auc}")

                res_auc.append(round(np.mean(tmp_auc), 2))
                res_negative_recall.append(round(np.mean(tmp_negative_recall), 2))
                res_positive_recall.append(round(np.mean(tmp_positive_recall), 2))
                res_negative_rate.append(round(np.mean(tmp_negative_rate), 2))
                res_positive_rate.append(round(np.mean(tmp_positive_rate), 2))
                res_profit.append(round(np.mean(tmp_profit), 2))
                res_profit_diff.append(round(np.mean(tmp_profit_diff), 2))

                print(f"RF result_auc : {(res_auc[-1])}")
                print(f"RF res_negative_recall : {res_negative_recall[-1]}")
                print(f"RF res_positive_recall : {res_positive_recall[-1]}")
                print(f"RF res_negative_rate : {res_negative_rate[-1]}")
                print(f"RF res_positive_rate : {res_positive_rate[-1]}")
                print(f"RF res_profit : {res_profit[-1]}")
                print(f"RF res_profit_diff : {res_profit_diff[-1]}")

        with open(os.getcwd() + "/experiment/lending_club/RF_smote_train_0.7_test_0.5_auc.txt", 'w') as fp:
            for item in res_auc:
                # write each item on a new line
                fp.write(f"{item}\n")

            fp.close()

        with open(os.getcwd() + "/experiment/lending_club/RF_smote_train_0.7_test_0.5_res_negative_recall.txt", 'w') as fp:
            for item in res_negative_recall:
                # write each item on a new line
                fp.write(f"{item}\n")

            fp.close()

        with open(os.getcwd() + "/experiment/lending_club/RF_smote_train_0.7_test_0.5_res_positive_recall.txt", 'w') as fp:
            for item in res_positive_recall:
                # write each item on a new line
                fp.write(f"{item}\n")

            fp.close()

        with open(os.getcwd() + "/experiment/lending_club/RF_smote_train_0.7_test_0.5_res_negative_rate.txt", 'w') as fp:
            for item in res_negative_rate:
                # write each item on a new line
                fp.write(f"{item}\n")

            fp.close()

        with open(os.getcwd() + "/experiment/lending_club/RF_smote_train_0.7_test_0.5_res_positive_rate.txt", 'w') as fp:
            for item in res_positive_rate:
                # write each item on a new line
                fp.write(f"{item}\n")

            fp.close()

        with open(os.getcwd() + "/experiment/lending_club/RF_smote_train_0.7_test_0.5_res_profit.txt", 'w') as fp:
            for item in res_profit:
                # write each item on a new line
                fp.write(f"{item}\n")

            fp.close()

        with open(os.getcwd() + "/experiment/lending_club/RF_smote_train_0.7_test_0.5_res_profit_diff.txt", 'w') as fp:
            for item in res_profit_diff:
                # write each item on a new line
                fp.write(f"{item}\n")

            fp.close()

        print('Done')

        # self.db_handler.learning_test_rebuilding(y_true, y_pred, iter, sub_iter, conf_matrix,
        #                          y_true.shape[0], cal, batch_id, model_no, model_full_name)
        #
        # self.db_handler.update_for_t3_detail(batch_id, model_no, iter, 'T')
        # self.db_handler.update_for_t3_detail(batch_id, model_no, iter, 'B')
        # print("good")
        # end = time.time()
        # print(f"{end - start}")