#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 17:35:55 2019

@author: Yan
"""
import numpy as np

import pandas as pd
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import GridSearchCV
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")
# parameters
data_path = './spectrum_data/Nothing_Mine_CEandRock.xlsx'
max_score = 2/15
min_score = -1/5
num_experiments = 100
# prepare train/test data

# read data
spectrum_data = pd.read_excel(data_path,  sep = ',', header = 0)
# remove freqs prior to 129
spectrum_data_new = spectrum_data.iloc[:, 400:1539]


# df.insert(loc=idx, column='A', value=new_col)
spectrum_data_new.insert(loc = 0, column = 'Signal Path', value = spectrum_data.iloc[:, 0])
# prepare data: for each label, split the data into 25% test and 75 train
raw = spectrum_data_new.values

# utility functions
def normalize_score(scores, max_score, min_score):
    np_scores = np.array(scores)
    return (np_scores - min_score)/(max_score - min_score)

def get_score(cf_matrix):
    pseudo_acc_case_rate = cf_matrix[1, 1] / np.sum(cf_matrix)
    bad_case_rate = (cf_matrix[1, 0] + cf_matrix[1, 2]) / np.sum(cf_matrix)
    undesired_case_rate = (cf_matrix[0, 1] + cf_matrix[2, 1]) / np.sum(cf_matrix)

    # give accuracy more weight
    score = (0.4 * pseudo_acc_case_rate - 0.4 * bad_case_rate - 0.2 * undesired_case_rate) # the max score is 1 and the min score is 0.
    return  score


def get_mormalized_score(cf_matrix):
    return normalize_score(get_score(cf_matrix), max_score, min_score)

def my_score(y_true, y_pred):
    cf = confusion_matrix(y_true, y_pred)
    return get_mormalized_score(cf)

score = make_scorer(my_score)

def get_model_data():
    # prepare data
    X_0 = raw[0:100, 0:1139]
    y_0 = raw[0:100, 1139]

    X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X_0, y_0, test_size=0.25, random_state=random.randint(0,5000))

    X_1 = raw[100:200, 0:1139]
    y_1 = raw[100:200, 1139]

    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.25, random_state=random.randint(0,5000))

    X_2 = raw[200:300, 0:1139]
    y_2 = raw[200:300, 1139]

    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.25, random_state=random.randint(0,5000))

    y_test = np.concatenate((y_test_0, y_test_1, y_test_2), axis=0)
    y_train = np.concatenate((y_train_0, y_train_1, y_train_2), axis=0)
    X_test = np.concatenate((X_test_0, X_test_1, X_test_2), axis=0)
    X_train = np.concatenate((X_train_0, X_train_1, X_train_2), axis=0)

    # Feature Scaling
    end = 1139
    scaler = StandardScaler()
    X_train[:, 1:end] = scaler.fit_transform(X_train[:, 1:end])   # Fit to data, then transform it. Fit means Compute the mean and std to be used for later scaling.
    X_test[:, 1:end] = scaler.transform(X_test[:, 1:end]) # Perform standardization by centering and scaling
    
    
    # grid search a best model
    grid = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
    logreg = LogisticRegression()
    logreg_cv = GridSearchCV(logreg, grid, cv = 5, scoring = score)
    logreg_cv.fit(X_train, y_train)
    
    # compute scores on test data
    return score(logreg_cv, X_test, y_test), confusion_matrix(y_test, logreg_cv.predict(X_test))

def write_list_to_csv(guest_list, filename):
    with open(filename, 'w') as outfile:
        for entries in guest_list:
            outfile.write(entries)
            outfile.write(",")

score_list = []
cf_matrix_list = []
for i in range(num_experiments):
    score_svm, cf_matrix = get_model_data()
    score_list.append(str(score_svm))
    cf_matrix_list.append(cf_matrix)

write_list_to_csv(score_list, "LR_scores.csv")
