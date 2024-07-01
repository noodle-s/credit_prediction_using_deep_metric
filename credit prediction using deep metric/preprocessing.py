import os
os.environ["PYTHONHASHSEED"] = str(1)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import numpy as np
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler, RobustScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import *
from sklearn.manifold import TSNE

class TripletGenerator:
    def __init__(self, batch_size, sub_iter, voting, all_iter):
        self.gen = None
        self.batch_size = batch_size
        self.sub_iter = sub_iter
        self.label = None
        self.voting = voting
        self.all_iter = all_iter

    def _encoder(self, slice_data):
        le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=999)
        slice_data = slice_data.copy()
        slice_data = le.fit_transform(slice_data)

        return slice_data

    def normalize(self, x):
        min_val = np.min(x)
        max_val = np.max(x)
        x = (x - min_val) / (max_val - min_val)
        return x

    def _scaling(self, data):
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)

        return data, scaler

    def split_same_ratio(self, data, label, test_size, random_state):

        label = pd.DataFrame(label)
        label_0 = label[label['Y'] == 0]
        label_1 = label[label['Y'] == 1]
        # print("0의 갯수 : ", label_0.shape, "1의 갯수 : ", label_1.shape)
        data_0 = data.loc[label_0.index]
        data_1 = data.loc[label_1.index]
        # print("0의 갯수 : ", data_0.shape, "1의 갯수 : ", data_1.shape)

        x_train_0, x_test_0, y_train_0, y_test_0 = train_test_split(data_0, label_0, test_size=test_size,
                                                                    random_state=random_state)
        x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(data_1, label_1, test_size=test_size,
                                                                    random_state=random_state)

        x_train = pd.concat([x_train_0, x_train_1], axis=0)
        y_train = pd.concat([y_train_0, y_train_1], axis=0)
        x_test = pd.concat([x_test_0, x_test_1], axis=0)
        y_test = pd.concat([y_test_0, y_test_1], axis=0)

        return x_train, x_test, y_train, y_test

    def _taiwan_preprocessing(self):
        data = pd.read_excel('./taiwan_default of credit card clients Data Set.xls')
        label = data['Y']
        # print(list(label).count(0) / len(label))
        # print(list(label).count(1) / len(label))

        cat_col = ['X2', 'X3', 'X4', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11']

        data[cat_col] = self._encoder(data[cat_col].astype(str))
        data = data.drop(columns='Y')

        num_col = list(set(data.columns) - set(cat_col))

        X_train, X_test, y_train, y_test = self.split_same_ratio(data, label, test_size=0.4, random_state=0)
        X_train[num_col], scaler = self._scaling(X_train[num_col])
        X_test[num_col] = scaler.transform(X_test[num_col])

        X_train, _, y_train, _ = self.split_same_ratio(X_train, y_train, test_size=0.3, random_state=self.all_iter)
        X_train = X_train.reset_index(drop=True)
        train = {'cat': np.array(X_train[cat_col]), 'num': np.array(X_train[num_col])}

        self.label = {'train': y_train, 'test': y_test}

        return train, X_test, np.array(y_train).reshape(-1), y_test, cat_col, num_col

    def _preprocessing(self):
        data = pd.read_excel('./taiwan_default of credit card clients Data Set.xls')
        label = data['Y']
        # print(list(label).count(0) / len(label))
        # print(list(label).count(1) / len(label))

        cat_col = ['X2', 'X3', 'X4', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11']

        data[cat_col] = self._encoder(data[cat_col].astype(str))

        data = data.drop(columns='Y')

        X_train, X_test, y_train, y_test = self.split_same_ratio(data, label, test_size = 0.4, random_state=0)
        num_col = list(set(X_train.columns) - set(cat_col))
        X_train[num_col], scaler = self._scaling(X_train[num_col])
        X_test[num_col] = scaler.transform(X_test[num_col])

        X_train, _, y_train, _ = self.split_same_ratio(X_train, y_train, test_size=0.3, random_state=self.all_iter)
        X_train = X_train.reset_index(drop=True)
        train = {'cat': np.array(X_train[cat_col]), 'num': np.array(X_train[num_col])}

        self.label = {'train' : y_train, 'test' : y_test}

        return train, X_test, np.array(y_train).reshape(-1), y_test, cat_col, num_col

    def _flow(self, test=False, label_idx=None, length=1):
        if test:
            # np.random.seed(0)
            positive_data = np.random.choice(label_idx[1], length, replace=True)
            negative_data = np.random.choice(label_idx[0], length, replace=True)

            return positive_data, negative_data

        else:
            label = self.label['train']

            classes = np.unique(label)
            indices = {c: np.where(label == c)[0] for c in classes}

            label = int(np.random.choice(2,1,replace=False))
            orig_data = np.random.choice(indices[label], 1, replace=False)[0]
            positive_data = np.random.choice(indices[label], 1, replace=False)[0]
            negative_data = np.random.choice(indices[label ^ 1], 1, replace=False)[0]

            if label == 0:
                label_idx[0].append(orig_data)

            else:
                label_idx[1].append(orig_data)

            return int(orig_data), int(positive_data), int(negative_data)

    def get_triplets_batch(self):
        x_train, x_test, y_train, y_test, cat_col, num_col = self._taiwan_preprocessing()
        train_idxs = {'origin': [], 'pos': [], 'neg': []}
        label_idx = {1 : [], 0: []}

        for i in range(30000):
            o, p, n = self._flow(label_idx = label_idx)
            train_idxs['origin'].append(o)
            train_idxs['pos'].append(p)
            train_idxs['neg'].append(n)

        train = {'origin_cat': x_train['cat'][train_idxs['origin'], :],
                 'origin_num': x_train['num'][train_idxs['origin'], :],
                 'pos_cat': x_train['cat'][train_idxs['pos'], :],
                 'pos_num': x_train['num'][train_idxs['pos'], :],
                 'neg_cat': x_train['cat'][train_idxs['neg'], :],
                 'neg_num': x_train['num'][train_idxs['neg'], :],
                 'label': y_train[train_idxs['origin']]}

        test = {'origin_cat': np.zeros((self.sub_iter, int(y_test.shape[0] * 0.5) * self.voting, len(cat_col))),
                'origin_num': np.zeros((self.sub_iter, int(y_test.shape[0] * 0.5) * self.voting, len(num_col))),
                'pos_cat': np.zeros((self.sub_iter, int(y_test.shape[0] * 0.5) * self.voting, len(cat_col))),
                'pos_num': np.zeros((self.sub_iter, int(y_test.shape[0] * 0.5) * self.voting, len(num_col))),
                'neg_cat': np.zeros((self.sub_iter, int(y_test.shape[0] * 0.5) * self.voting, len(cat_col))),
                'neg_num': np.zeros((self.sub_iter, int(y_test.shape[0] * 0.5) * self.voting, len(num_col))),
                'label': np.zeros((self.sub_iter, int(y_test.shape[0] * 0.5)))}

        for cnt in range(self.sub_iter):
            tmp_x_test, _, tmp_y_test, _ = self.split_same_ratio(x_test, y_test, test_size=0.5, random_state=cnt)
            tmp_test = {'cat': np.array(tmp_x_test.loc[:,cat_col]), 'num': np.array(tmp_x_test.loc[:,num_col])}
            p, n = self._flow(True, label_idx=label_idx, length=tmp_y_test.shape[0] * self.voting)

            test['origin_cat'][cnt, :, :] = np.repeat(tmp_test['cat'], self.voting, axis=0)
            test['pos_cat'][cnt, :, :] = x_train['cat'][p, :]
            test['neg_cat'][cnt, :, :] = x_train['cat'][n, :]
            test['origin_num'][cnt, :, :] = np.repeat(tmp_test['num'], self.voting, axis=0)
            test['pos_num'][cnt, :, :] = x_train['num'][p, :]
            test['neg_num'][cnt, :, :] = x_train['num'][n, :]
            test['label'][cnt, :] = np.array(tmp_y_test).reshape(-1)

        # print(list(train['label']).count(0) / len(train['label']))
        # print(list(train['label']).count(1) / len(train['label']))
        return train, test

class TupletGenerator:
    def __init__(self, batch_size, sub_iter, voting, num_neg_sample, all_iter):
        self.gen = None
        self.batch_size = batch_size
        self.sub_iter = sub_iter
        self.label = None
        self.voting = voting
        self.num_neg_sample = num_neg_sample
        self.all_iter = all_iter

    def _encoder(self, slice_data):
        le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=999)
        slice_data = slice_data.copy()
        slice_data = le.fit_transform(slice_data)

        return slice_data

    def normalize(self, x):
        min_val = np.min(x)
        max_val = np.max(x)
        x = (x - min_val) / (max_val - min_val)
        return x

    def _scaling(self, data):
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)

        return data, scaler

    def split_same_ratio(self, data, label, test_size, random_state):

        label = pd.DataFrame(label)
        label_0 = label[label['Y'] == 0]
        label_1 = label[label['Y'] == 1]
        # print("0의 갯수 : ", label_0.shape, "1의 갯수 : ", label_1.shape)
        data_0 = data.loc[label_0.index]
        data_1 = data.loc[label_1.index]
        # print("0의 갯수 : ", data_0.shape, "1의 갯수 : ", data_1.shape)

        x_train_0, x_test_0, y_train_0, y_test_0 = train_test_split(data_0, label_0, test_size=test_size,
                                                                    random_state=random_state)
        x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(data_1, label_1, test_size=test_size,
                                                                    random_state=random_state)

        x_train = pd.concat([x_train_0, x_train_1], axis=0)
        y_train = pd.concat([y_train_0, y_train_1], axis=0)
        x_test = pd.concat([x_test_0, x_test_1], axis=0)
        y_test = pd.concat([y_test_0, y_test_1], axis=0)

        return x_train, x_test, y_train, y_test

    def _preprocessing(self):
        data = pd.read_excel('./taiwan_default of credit card clients Data Set.xls')
        label = data['Y']
        # print(list(label).count(0) / len(label))
        # print(list(label).count(1) / len(label))

        cat_col = ['X2', 'X3', 'X4', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11']

        data[cat_col] = self._encoder(data[cat_col].astype(str))

        data = data.drop(columns='Y')

        X_train, X_test, y_train, y_test = self.split_same_ratio(data, label, test_size=0.4, random_state=0)
        num_col = list(set(X_train.columns) - set(cat_col))
        X_train[num_col], scaler = self._scaling(X_train[num_col])
        X_test[num_col] = scaler.transform(X_test[num_col])

        X_train, _, y_train, _ = self.split_same_ratio(X_train, y_train, test_size=0.3, random_state=self.all_iter)
        X_train = X_train.reset_index(drop=True)
        train = {'cat': np.array(X_train[cat_col]), 'num': np.array(X_train[num_col])}

        self.label = {'train': y_train, 'test': y_test}

        return train, X_test, np.array(y_train).reshape(-1), y_test, cat_col, num_col

    def _flow(self, test=False, label_idx=None, length=1):
        if test:
            # np.random.seed(0)
            positive_data = np.random.choice(label_idx[1], length, replace=True)
            negative_data = np.random.choice(label_idx[0], length * self.num_neg_sample, replace=True)

            return positive_data, negative_data

        else:
            label = self.label['train']

            classes = np.unique(label)
            indices = {c: np.where(label == c)[0] for c in classes}

            label = int(np.random.choice(2,1, replace=False))
            orig_data = np.random.choice(indices[label], 1, replace=False)[0]
            positive_data = np.random.choice(indices[label], 1, replace=False)[0]
            negative_data = np.random.choice(indices[label ^ 1], self.num_neg_sample, replace=False)

            if label == 0:
                label_idx[0].append(orig_data)

            else:
                label_idx[1].append(orig_data)

            return int(orig_data), int(positive_data), negative_data

    def get_tuplet_batch(self):
        x_train, x_test, y_train, y_test, cat_col, num_col = self._preprocessing()
        train_idxs = {'origin': [], 'pos': [], 'neg': []}
        label_idx = {1 : [], 0: []}

        for i in range(5000):
            o, p, n = self._flow(label_idx = label_idx)
            train_idxs['origin'].append(o)
            train_idxs['pos'].append(p)
            train_idxs['neg'] += n.tolist()

        train = {'origin_cat': x_train['cat'][train_idxs['origin'], :],
                 'origin_num': x_train['num'][train_idxs['origin'], :],
                 'pos_cat': x_train['cat'][train_idxs['pos'], :],
                 'pos_num': x_train['num'][train_idxs['pos'], :],
                 'neg_cat': x_train['cat'][train_idxs['neg'], :],
                 'neg_num': x_train['num'][train_idxs['neg'], :],
                 'label': y_train[train_idxs['origin']]}

        test = {'origin_cat': np.zeros((self.sub_iter, int(y_test.shape[0] * 0.5) * self.voting, len(cat_col))),
                'origin_num': np.zeros((self.sub_iter, int(y_test.shape[0] * 0.5) * self.voting, len(num_col))),
                'pos_cat': np.zeros((self.sub_iter, int(y_test.shape[0] * 0.5) * self.voting, len(cat_col))),
                'pos_num': np.zeros((self.sub_iter, int(y_test.shape[0] * 0.5) * self.voting, len(num_col))),
                'neg_cat': np.zeros((self.sub_iter, int(y_test.shape[0] * 0.5) * self.voting, self.num_neg_sample, len(cat_col))),
                'neg_num': np.zeros((self.sub_iter, int(y_test.shape[0] * 0.5) * self.voting, self.num_neg_sample, len(num_col))),
                'label': np.zeros((self.sub_iter, int(y_test.shape[0] * 0.5)))}

        for cnt in range(self.sub_iter):
            tmp_x_test, _, tmp_y_test, _ = self.split_same_ratio(x_test, y_test, test_size=0.5, random_state=cnt)
            tmp_test = {'cat': np.array(tmp_x_test.loc[:,cat_col]), 'num': np.array(tmp_x_test.loc[:,num_col])}
            p, n = self._flow(True, label_idx=label_idx, length=tmp_y_test.shape[0] * self.voting)
            n_sample = {'cat' : x_train['cat'][n, :].reshape(tmp_y_test.shape[0] * self.voting, self.num_neg_sample, -1),
                        'num' : x_train['num'][n, :].reshape(tmp_y_test.shape[0] * self.voting, self.num_neg_sample, -1)}

            test['origin_cat'][cnt, :, :] = np.repeat(tmp_test['cat'], self.voting, axis=0)
            test['pos_cat'][cnt, :, :] = x_train['cat'][p, :]
            test['neg_cat'][cnt, :, :, :] = n_sample['cat']
            test['origin_num'][cnt, :, :] = np.repeat(tmp_test['num'], self.voting, axis=0)
            test['pos_num'][cnt, :, :] = x_train['num'][p, :]
            test['neg_num'][cnt, :, :, :] = n_sample['num']
            test['label'][cnt, :] = np.array(tmp_y_test).reshape(-1)

        # print(list(train['label']).count(0) / train_split)
        # print(list(train['label']).count(1) / train_split)
        return train, test

class LFEGenerator:
    def __init__(self, batch_size, sub_iter, all_iter, bigbang_train, dataset):
        self.gen = None
        self.batch_size = batch_size
        self.sub_iter = sub_iter
        self.label = None
        self.all_iter = all_iter
        self.bigbang_train = bigbang_train
        self.dataset = dataset

    def _encoder(self, slice_data):
        le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=999)
        slice_data = slice_data.copy()
        slice_data = le.fit_transform(slice_data)

        return slice_data

    def _scaling(self, data):
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data).astype('float32')

        return data, scaler

    def split_same_ratio(self, data, label, test_size, random_state):
        if self.dataset == 'I':
            target = 'y1'

        elif self.dataset == 'T':
            target = 'Y'

        else:
            target = 'loan_status'

        label = pd.DataFrame(label)
        label_0 = label[label[target] == 0]
        label_1 = label[label[target] == 1]
        # print("0의 갯수 : ", label_0.shape, "1의 갯수 : ", label_1.shape)
        data_0 = data.loc[label_0.index]
        data_1 = data.loc[label_1.index]
        # print("0의 갯수 : ", data_0.shape, "1의 갯수 : ", data_1.shape)

        x_train_0, x_test_0, y_train_0, y_test_0 = train_test_split(data_0, label_0, test_size=test_size,
                                                                    random_state=random_state)
        x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(data_1, label_1, test_size=test_size,
                                                                    random_state=random_state)

        x_train = pd.concat([x_train_0, x_train_1], axis=0)
        y_train = pd.concat([y_train_0, y_train_1], axis=0)
        x_test = pd.concat([x_test_0, x_test_1], axis=0)
        y_test = pd.concat([y_test_0, y_test_1], axis=0)

        return x_train, x_test, y_train, y_test

    def _taiwan_preprocessing(self):
        data = pd.read_excel('./taiwan_default of credit card clients Data Set.xls')
        label = data['Y']
        # print(list(label).count(0) / len(label))
        # print(list(label).count(1) / len(label))

        cat_col = ['X2', 'X3', 'X4', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11']

        data[cat_col] = self._encoder(data[cat_col].astype(str))
        data = data.drop(columns='Y')

        num_col = list(set(data.columns) - set(cat_col))

        X_train, X_test, y_train, y_test = self.split_same_ratio(data, label, test_size=0.4, random_state=0)
        X_train[num_col], scaler = self._scaling(X_train[num_col])
        X_test[num_col] = scaler.transform(X_test[num_col])

        if self.bigbang_train:
            X_train, _, y_train, _ = self.split_same_ratio(X_train, y_train, test_size=0.3, random_state=self.all_iter)

        X_train = X_train.reset_index(drop=True)
        train = {'cat': np.array(X_train[cat_col]), 'num': np.array(X_train[num_col])}

        self.label = {'train': y_train, 'test': y_test}

        return train, X_test, np.array(y_train).reshape(-1), y_test, cat_col, num_col

    def pub_rec(self, number):
        if number == 0.0:
            return 0
        else:
            return 1

    def mort_acc(self, number):
        if number == 0.0:
            return 0
        elif number >= 1.0:
            return 1
        else:
            return number

    def pub_rec_bankruptcies(self, number):
        if number == 0.0:
            return 0
        elif number >= 1.0:
            return 1
        else:
            return number

    def fill_mort_acc(self,total_acc, mort_acc, total_acc_avg):
        if np.isnan(mort_acc):
            return total_acc_avg.loc[total_acc].round()
        else:
            return mort_acc

    def _lending_club_preprocessing(self):
        data = pd.read_csv('./lending_club_loan_two.csv')[:50000]
        label_values = {'Fully Paid': 0, 'Charged Off': 1}
        data['loan_status'] = data.loan_status.map(label_values)

        data['pub_rec'] = data.pub_rec.apply(self.pub_rec)
        data['mort_acc'] = data.mort_acc.apply(self.mort_acc)
        data['pub_rec_bankruptcies'] = data.pub_rec_bankruptcies.apply(self.pub_rec_bankruptcies)

        data.drop('emp_title', axis=1, inplace=True)
        data.drop('emp_length', axis=1, inplace=True)
        data.drop('title', axis=1, inplace=True)
        total_acc_avg = data.groupby(by='total_acc').mean().mort_acc
        data['mort_acc'] = data.apply(lambda x: self.fill_mort_acc(x['total_acc'], x['mort_acc'], total_acc_avg),
                                      axis=1)

        # for column in data.columns:
        #     if data[column].isna().sum() != 0:
        #         missing = data[column].isna().sum()
        #         portion = (missing / data.shape[0]) * 100
        #         print(f"'{column}': number of missing values '{missing}' ==> '{portion:.3f}%'")

        data.dropna(inplace=True)
        label = data['loan_status']
        # print(list(label).count(0) / len(label))
        # print(list(label).count(1) / len(label))
        data.drop('loan_status', axis=1, inplace=True)
        # print(data.shape)

        term_values = {' 36 months': 36, ' 60 months': 60}
        data['term'] = data.term.map(term_values)
        data.drop('grade', axis=1, inplace=True)

        dummies = ['sub_grade', 'verification_status', 'purpose', 'initial_list_status',
                   'application_type', 'home_ownership']
        # data = pd.get_dummies(data, columns=dummies, drop_first=True)

        data['zip_code'] = data.address.apply(lambda x: x[-5:])

        # data = pd.get_dummies(data, columns=['zip_code'], drop_first=True)
        data.drop('address', axis=1, inplace=True)
        data.drop('issue_d', axis=1, inplace=True)
        data['earliest_cr_line'] = pd.to_datetime(data['earliest_cr_line'])
        data['earliest_cr_line'] = data.earliest_cr_line.dt.year

        cat_col = dummies + ['zip_code', 'earliest_cr_line']
        data[cat_col] = self._encoder(data[cat_col].astype(str))

        num_col = list(set(data.columns) - set(cat_col))

        X_train, X_test, y_train, y_test = self.split_same_ratio(data, label, test_size=0.4, random_state=0)
        X_train[num_col], scaler = self._scaling(X_train[num_col])
        cal = X_test[['loan_amnt', 'int_rate']]
        X_test[num_col] = scaler.transform(X_test[num_col])

        if self.bigbang_train:
            X_train, _, y_train, _ = self.split_same_ratio(X_train, y_train, test_size=0.3, random_state=self.all_iter)

        X_train = X_train.reset_index(drop=True)
        train = {'cat': np.array(X_train[cat_col]), 'num': np.array(X_train[num_col])}

        self.label = {'train': y_train, 'test': y_test}

        return train, X_test, np.array(y_train).reshape(-1), y_test, cal, cat_col, num_col

    def _ibk_preprocessing(self):
        data = pd.read_csv('./ibksb_cham.csv', encoding='CP949')

        label = data['y1']

        data = data.drop(columns=['y1', '계좌번호', 'CL0631905', 'CL0631906', 'LS0001197', 'LS0001196'])

        data_type = pd.read_csv('./ibk_datatype.csv', encoding='CP949')
        cat_col = data_type[data_type['t'] == 'categorical']['구분'].values

        data[cat_col] = self._encoder(data[cat_col].astype(str))

        X_train, X_test, y_train, y_test = self.split_same_ratio(data, label, test_size=0.4, random_state=0)
        num_col = list(set(X_train.columns) - set(cat_col))
        X_train[num_col], scaler = self._scaling(X_train[num_col])
        # cal = X_test[['loan_amnt', 'int_rate']]
        X_test[num_col] = scaler.transform(X_test[num_col])

        cal = X_test[['대출실행금액', '금리']]

        if self.bigbang_train:
            X_train, _, y_train, _ = self.split_same_ratio(X_train, y_train, test_size=0.3, random_state=self.all_iter)

        X_train = X_train.reset_index(drop=True)

        train = {'cat': np.array(X_train[cat_col]), 'num': np.array(X_train[num_col])}

        self.label = {'train': y_train, 'test': y_test}
        # max = np.max(X_train)

        return train, X_test, np.array(y_train).reshape(-1), y_test, cal, cat_col, num_col

    def _flow(self):
        label = self.label['train']

        classes = np.unique(label)
        indices = {c: np.where(label == c)[0] for c in classes}

        data = [indices[0].tolist() + indices[1].tolist()]

        return data

    def get_lfe_batch(self):
        if self.dataset == 'T':
            x_train, x_test, y_train, y_test, cat_col, num_col = self._taiwan_preprocessing()

        elif self.dataset == 'I':
            x_train, x_test, y_train, y_test, cal, cat_col, num_col = self._ibk_preprocessing()

        else:
            x_train, x_test, y_train, y_test, cal, cat_col, num_col = self._lending_club_preprocessing()

        train_idxs = {'data': []}

        train_idxs['data'] += self._flow()

        train = {'cat': np.squeeze(x_train['cat'][train_idxs['data'], :], axis=0),
                 'num': np.squeeze(x_train['num'][train_idxs['data'], :], axis=0),
                 'label': y_train[tuple(train_idxs['data'])]}

        test = {'cat': np.zeros((self.sub_iter, int(y_test.shape[0] * 0.5), len(cat_col))),
                'num': np.zeros((self.sub_iter, int(y_test.shape[0] * 0.5), len(num_col))),
                'cal': np.zeros((self.sub_iter, int(y_test.shape[0] * 0.5), 2)),
                'label': np.zeros((self.sub_iter, int(y_test.shape[0] * 0.5)))}

        if self.bigbang_train:
            for cnt in range(self.sub_iter):
                tmp_x_test, _, tmp_y_test, _ = self.split_same_ratio(x_test, y_test, test_size=0.5, random_state=cnt)

                test['cat'][cnt, :, :] = tmp_x_test.loc[:,cat_col]
                test['num'][cnt, :, :] =  tmp_x_test.loc[:,num_col]
                test['cal'][cnt, :, :] = cal.loc[tmp_x_test.index, ['대출실행금액', '금리']]
                test['label'][cnt, :] = np.array(tmp_y_test).reshape(-1)

        else:
            test = {'cat': x_test.loc[:,cat_col],
                    'num' : x_test.loc[:,num_col],
                    'label': y_test}

        # print(list(train['label']).count(0) / train_split)
        # print(list(train['label']).count(1) / train_split)
        return train, test

class SiameseGenerator:
    def __init__(self, batch_size, sub_iter, voting, all_iter):
        self.gen = None
        self.batch_size = batch_size
        self.sub_iter = sub_iter
        self.label = None
        self.voting = voting
        self.all_iter = all_iter

    def _encoder(self, slice_data):
        le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=999)
        slice_data = slice_data.copy()
        slice_data = le.fit_transform(slice_data)

        return slice_data

    def normalize(self, x):
        min_val = np.min(x)
        max_val = np.max(x)
        x = (x - min_val) / (max_val - min_val)
        return x

    def _scaling(self, data):
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)

        return data, scaler

    def split_same_ratio(self, data, label, test_size, random_state):

        label = pd.DataFrame(label)
        label_0 = label[label['y1'] == 0]
        label_1 = label[label['y1'] == 1]
        # print("0의 갯수 : ", label_0.shape, "1의 갯수 : ", label_1.shape)
        data_0 = data.loc[label_0.index]
        data_1 = data.loc[label_1.index]
        # print("0의 갯수 : ", data_0.shape, "1의 갯수 : ", data_1.shape)

        x_train_0, x_test_0, y_train_0, y_test_0 = train_test_split(data_0, label_0, test_size=test_size,
                                                                    random_state=random_state)
        x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(data_1, label_1, test_size=test_size,
                                                                    random_state=random_state)

        x_train = pd.concat([x_train_0, x_train_1], axis=0)
        y_train = pd.concat([y_train_0, y_train_1], axis=0)
        x_test = pd.concat([x_test_0, x_test_1], axis=0)
        y_test = pd.concat([y_test_0, y_test_1], axis=0)

        return x_train, x_test, y_train, y_test

    def _preprocessing(self):
        data = pd.read_csv('./ibksb_cham.csv', encoding='CP949')
        data = data.fillna(0)

        label = data['y1']

        data = data.drop(columns='y1')
        data = data.drop(columns='계좌번호')
        data = data.drop(columns='CL0631905')
        data = data.drop(columns='CL0631906')
        data = data.drop(columns='LS0001197')
        data = data.drop(columns='LS0001196')

        data_type = pd.read_csv('./ibk_datatype.csv', encoding='CP949')
        cat_col = data_type[data_type['t'] == 'categorical']['구분'].values

        data[cat_col] = self._encoder(data[cat_col].astype(str))

        X_train, X_test, y_train, y_test = self.split_same_ratio(data, label, test_size = 0.4, random_state=0)
        num_col = list(set(X_train.columns) - set(cat_col))
        X_train[num_col], scaler = self._scaling(X_train[num_col])
        cal = X_test[['대출실행금액', '금리']]
        X_test[num_col] = scaler.transform(X_test[num_col])

        X_train, _, y_train, _ = self.split_same_ratio(X_train, y_train, test_size=0.3, random_state=self.all_iter)
        X_train = X_train.reset_index(drop=True)
        train = {'cat': np.array(X_train[cat_col]), 'num': np.array(X_train[num_col])}

        self.label = {'train' : y_train, 'test' : y_test}

        return train, X_test, np.array(y_train).reshape(-1), y_test, cal, cat_col, num_col

    def _flow(self, train=True):
        label = self.label['train']

        classes = np.unique(label)
        indices = {c: np.where(label == c)[0] for c in classes}

        label = int(np.random.choice(2, 1, replace=False))
        orig_data = np.random.choice(indices[label], 1, replace=False)[0]

        disc_label = int(np.random.choice([label, label ^ 1], 1, replace=False))
        disc_data = np.random.choice(indices[disc_label], 1, replace=False)[0]

        if train:
            return orig_data, disc_data

        else:
            return disc_data

    def get_siamese_batch(self):
        x_train, x_test, y_train, y_test, cal, cat_col, num_col = self._preprocessing()
        train_idxs = {'origin': [], 'disc': []}

        for i in range(1000):
            o, d = self._flow()
            train_idxs['origin'].append(o)
            train_idxs['disc'].append(d)

        train = {'origin_cat': x_train['cat'][train_idxs['origin'], :],
                 'origin_num': x_train['num'][train_idxs['origin'], :],
                 'disc_cat': x_train['cat'][train_idxs['disc'], :],
                 'disc_num': x_train['num'][train_idxs['disc'], :],
                 'label': y_train[train_idxs['origin']],
                 'disc_label' : y_train[train_idxs['disc']]}

        test = {'origin_cat': np.zeros((self.sub_iter, int(y_test.shape[0] * 0.5) * self.voting, len(cat_col))),
                'origin_num': np.zeros((self.sub_iter, int(y_test.shape[0] * 0.5) * self.voting, len(num_col))),
                'disc_cat': np.zeros((self.sub_iter, int(y_test.shape[0] * 0.5) * self.voting, len(cat_col))),
                'disc_num': np.zeros((self.sub_iter, int(y_test.shape[0] * 0.5) * self.voting, len(num_col))),
                'label': np.zeros((self.sub_iter, int(y_test.shape[0] * 0.5))),
                'disc_label': np.zeros((self.sub_iter, int(y_test.shape[0] * 0.5)))}

        for cnt in range(self.sub_iter):
            tmp_x_test, _, tmp_y_test, _ = self.split_same_ratio(x_test, y_test, test_size=0.5, random_state=cnt)
            tmp_test = {'cat': np.array(tmp_x_test.loc[:,cat_col]), 'num': np.array(tmp_x_test.loc[:,num_col])}
            d = self._flow(False)

            test['origin_cat'][cnt, :, :] = np.repeat(tmp_test['cat'], self.voting, axis=0)
            test['disc_cat'][cnt, :, :] = x_train['cat'][d, :]
            test['origin_num'][cnt, :, :] = np.repeat(tmp_test['num'], self.voting, axis=0)
            test['disc_num'][cnt, :, :] = x_train['num'][d, :]
            test['label'][cnt, :] = np.array(tmp_y_test).reshape(-1)
            test['disc_label'][cnt, :] = y_train[d]
            # test['cal'][cnt, :, :] = cal.loc[tmp_x_test.index, ['대출실행금액', '금리']]

        # print(list(train['label']).count(0) / train_split)
        # print(list(train['label']).count(1) / train_split)
        return train, test

class CompareGenerator:
    def __init__(self, sub_iter, all_iter, oversampling, bia_bang_train, dataset):
        self.gen = None
        self.sub_iter = sub_iter
        self.label = None
        self.all_iter = all_iter
        self.oversampling = oversampling
        self.bia_bang_train = bia_bang_train
        self.dataset = dataset

    def _encoder(self, slice_data):
        le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=999)
        slice_data = slice_data.copy()
        slice_data = le.fit_transform(slice_data)

        return slice_data

    def normalize(self, x):
        min_val = np.min(x)
        max_val = np.max(x)
        x = (x - min_val) / (max_val - min_val)
        return x

    def _scaling(self, data):
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)

        return data, scaler

    def split_same_ratio(self, data, label, test_size, random_state):
        if self.dataset == 'I':
            target = 'y1'

        elif self.dataset == 'T':
            target = 'Y'

        else:
            target = 'loan_status'

        label = pd.DataFrame(label)
        label_0 = label[label[target] == 0]
        label_1 = label[label[target] == 1]
        # print("0의 갯수 : ", label_0.shape, "1의 갯수 : ", label_1.shape)
        data_0 = data.loc[label_0.index]
        data_1 = data.loc[label_1.index]
        # print("0의 갯수 : ", data_0.shape, "1의 갯수 : ", data_1.shape)

        x_train_0, x_test_0, y_train_0, y_test_0 = train_test_split(data_0, label_0, test_size=test_size,
                                                                    random_state=random_state)
        x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(data_1, label_1, test_size=test_size,
                                                                    random_state=random_state)

        x_train = pd.concat([x_train_0, x_train_1], axis=0)
        y_train = pd.concat([y_train_0, y_train_1], axis=0)
        x_test = pd.concat([x_test_0, x_test_1], axis=0)
        y_test = pd.concat([y_test_0, y_test_1], axis=0)

        return x_train, x_test, y_train, y_test

    def Adasyn(self, x_data, y_data):
        adasyn = ADASYN(random_state=0)
        x_resampled, y_resampled = adasyn.fit_resample(x_data, y_data)

        return x_resampled, y_resampled

    def Smote(self, x_data, y_data):
        sm = SMOTE(random_state=0)
        x_resampled, y_resampled = sm.fit_resample(x_data, y_data)

        return x_resampled, y_resampled

    def pub_rec(self, number):
        if number == 0.0:
            return 0
        else:
            return 1

    def mort_acc(self, number):
        if number == 0.0:
            return 0
        elif number >= 1.0:
            return 1
        else:
            return number

    def pub_rec_bankruptcies(self, number):
        if number == 0.0:
            return 0
        elif number >= 1.0:
            return 1
        else:
            return number

    def fill_mort_acc(self,total_acc, mort_acc, total_acc_avg):
        if np.isnan(mort_acc):
            return total_acc_avg.loc[total_acc].round()
        else:
            return mort_acc

    def _taiwan_preprocessing(self):
        data = pd.read_excel('./taiwan_default of credit card clients Data Set.xls')
        label = data['Y']
        # print(list(label).count(0) / len(label))
        # print(list(label).count(1) / len(label))

        cat_col = ['X2', 'X3', 'X4', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11']

        data[cat_col] = self._encoder(data[cat_col].astype(str))

        data = data.drop(columns='Y')

        X_train, X_test, y_train, y_test = self.split_same_ratio(data, label, test_size=0.4, random_state=0)
        num_col = list(set(X_train.columns) - set(cat_col))
        X_train[num_col], scaler = self._scaling(X_train[num_col])
        X_test[num_col] = scaler.transform(X_test[num_col])

        if self.bia_bang_train:
            X_train, _, y_train, _ = self.split_same_ratio(X_train, y_train, test_size=0.3, random_state=self.all_iter)

        X_train = X_train.reset_index(drop=True)

        self.label = {'train': y_train, 'test': y_test}

        if self.oversampling is not None:
            if self.oversampling == "ADASYN":
                X_train, y_train = self.Adasyn(X_train, y_train)

            else:
                X_train, y_train = self.Smote(X_train, y_train)

        return X_train, X_test, np.array(y_train).reshape(-1), y_test, cat_col, num_col

    def _ibk_preprocessing(self):
        data = pd.read_csv('./ibksb_cham.csv', encoding='CP949')
        data = data.fillna(0)

        label = data['y1']

        data = data.drop(columns='y1')
        data = data.drop(columns='계좌번호')
        data = data.drop(columns='CL0631905')
        data = data.drop(columns='CL0631906')
        data = data.drop(columns='LS0001197')
        data = data.drop(columns='LS0001196')

        data_type = pd.read_csv('./ibk_datatype.csv', encoding='CP949')
        cat_col = data_type[data_type['t'] == 'categorical']['구분'].values

        data[cat_col] = self._encoder(data[cat_col].astype(str))

        X_train, X_test, y_train, y_test = self.split_same_ratio(data, label, test_size=0.4, random_state=0)
        num_col = list(set(X_train.columns) - set(cat_col))
        X_train[num_col], scaler = self._scaling(X_train[num_col])
        cal = X_test[['대출실행금액', '금리']]
        X_test[num_col] = scaler.transform(X_test[num_col])

        if self.bia_bang_train:
            X_train, _, y_train, _ = self.split_same_ratio(X_train, y_train, test_size=0.7, random_state=self.all_iter)
        X_train = X_train.reset_index(drop=True)
        # train = {'cat': np.array(X_train[cat_col]), 'num': np.array(X_train[num_col])}

        self.label = {'train': y_train, 'test': y_test}

        if self.oversampling is not None:
            if self.oversampling == "ADASYN":
                X_train, y_train = self.Adasyn(X_train, y_train)

            else:
                X_train, y_train = self.Smote(X_train, y_train)

        return X_train, X_test, np.array(y_train).reshape(-1), y_test, cal, cat_col, num_col

    def _lending_club_preprocessing(self):
        data = pd.read_csv('./lending_club_loan_two.csv')[:50000]
        label_values = {'Fully Paid': 0, 'Charged Off': 1}
        data['loan_status'] = data.loan_status.map(label_values)

        data['pub_rec'] = data.pub_rec.apply(self.pub_rec)
        data['mort_acc'] = data.mort_acc.apply(self.mort_acc)
        data['pub_rec_bankruptcies'] = data.pub_rec_bankruptcies.apply(self.pub_rec_bankruptcies)

        data.drop('emp_title', axis=1, inplace=True)
        data.drop('emp_length', axis=1, inplace=True)
        data.drop('title', axis=1, inplace=True)
        total_acc_avg = data.groupby(by='total_acc').mean().mort_acc
        data['mort_acc'] = data.apply(lambda x: self.fill_mort_acc(x['total_acc'], x['mort_acc'], total_acc_avg), axis=1)

        for column in data.columns:
            if data[column].isna().sum() != 0:
                missing = data[column].isna().sum()
                portion = (missing / data.shape[0]) * 100
                print(f"'{column}': number of missing values '{missing}' ==> '{portion:.3f}%'")

        data.dropna(inplace=True)
        label = data['loan_status']
        print(list(label).count(0) / len(label))
        print(list(label).count(1) / len(label))
        data.drop('loan_status', axis=1, inplace=True)
        print(data.shape)

        term_values = {' 36 months': 36, ' 60 months': 60}
        data['term'] = data.term.map(term_values)
        data.drop('grade', axis=1, inplace=True)

        dummies = ['sub_grade', 'verification_status', 'purpose', 'initial_list_status',
                   'application_type', 'home_ownership']
        # data = pd.get_dummies(data, columns=dummies, drop_first=True)

        data['zip_code'] = data.address.apply(lambda x: x[-5:])

        # data = pd.get_dummies(data, columns=['zip_code'], drop_first=True)
        data.drop('address', axis=1, inplace=True)
        data.drop('issue_d', axis=1, inplace=True)
        data['earliest_cr_line'] = pd.to_datetime(data['earliest_cr_line'])
        data['earliest_cr_line'] = data.earliest_cr_line.dt.year

        cat_col = dummies + ['zip_code', 'earliest_cr_line']
        data[cat_col] = self._encoder(data[cat_col].astype(str))

        num_col = list(set(data.columns) - set(cat_col))

        X_train, X_test, y_train, y_test = self.split_same_ratio(data, label, test_size=0.4, random_state=0)
        X_train[num_col], scaler = self._scaling(X_train[num_col])
        cal = X_test[['loan_amnt', 'int_rate']]
        X_test[num_col] = scaler.transform(X_test[num_col])

        if self.bia_bang_train:
            X_train, _, y_train, _ = self.split_same_ratio(X_train, y_train, test_size=0.3, random_state=self.all_iter)

        X_train = X_train.reset_index(drop=True)
        # train = {'cat': np.array(X_train[cat_col]), 'num': np.array(X_train[num_col])}

        self.label = {'train': y_train, 'test': y_test}

        if self.oversampling is not None:
            if self.oversampling == "ADASYN":
                X_train, y_train = self.Adasyn(X_train, y_train)

            else:
                X_train, y_train = self.Smote(X_train, y_train)

        return X_train, X_test, np.array(y_train).reshape(-1), y_test, cal, cat_col, num_col

    def get_compare_batch(self):
        if self.dataset == 'I':
            x_train, x_test, y_train, y_test, cal, cat_col, num_col = self._ibk_preprocessing()

        elif self.dataset == 'L':
            x_train, x_test, y_train, y_test, cal, cat_col, num_col = self._lending_club_preprocessing()

        else:
            x_train, x_test, y_train, y_test, cat_col, num_col = self._taiwan_preprocessing()

        train = {'data' : x_train, 'label' : y_train}

        if self.bia_bang_train:
            test = {'data': np.zeros((self.sub_iter, int(len(x_test) * 0.5), x_test.shape[-1])),
                    'label': np.zeros((self.sub_iter, int(len(x_test) * 0.5))),
                    'cal' : np.zeros((self.sub_iter, int(y_test.shape[0] * 0.5), 2))}

            for cnt in range(self.sub_iter):
                tmp_x_test, _, tmp_y_test, _ = self.split_same_ratio(x_test, y_test, test_size=0.5, random_state=cnt)

                test['data'][cnt, :, :] = tmp_x_test
                test['label'][cnt, :] = np.array(tmp_y_test).reshape(-1)
                test['cal'][cnt, :, :] = cal.loc[tmp_x_test.index, ['loan_amnt', 'int_rate']]

        else:
            test = {'data': x_test,
                    'label': y_test}
        # print(list(train['label']).count(0) / train_split)
        # print(list(train['label']).count(1) / train_split)
        return train, test