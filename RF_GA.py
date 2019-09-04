#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:26:25 2019

@author: Yan Zhang

This file provides all the different versions of RF + GA algorithm for 
generating different models.

The current version returns scores and confusion matrices of each model
"""

# import libraries
# library for GA
import random
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
# library for RF
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings; warnings.simplefilter('ignore')
import scipy.stats

# algorithm parameters
print("Set up algorithm parameters")
num_trees = 100 # Number of trees in RF
IND_SIZE =  10  # Number of trees selected by GA 
POP_SIZE =  30  # Population size used in the GA algorithm
experiment_times = 2  #
input_data_file_path = './spectrum_data/Nothing_Mine_CEandRock.xlsx'
max_score = 2/15  # what is the max_score under our fitness function, used in normalization of the score
min_score = -1/5  # what is the min_score under our fitness fucntion
num_actual_landmine_in_text_data = 25


# read data
print("Read data")
spectrum_data = pd.read_excel(input_data_file_path,  sep = ',', header = 0)
# remove freqs prior to 129
spectrum_data_new = spectrum_data.iloc[:, 400:1539]


# df.insert(loc=idx, column='A', value=new_col)
spectrum_data_new.insert(loc = 0, column = 'Signal Path', value = spectrum_data.iloc[:, 0])


def run_GA(NUM_TREES, IND_SIZE, POP_SIZE, CX_RATE = 0.8):
    """
    NUM_TREES is the number of trees the random forest model pro
    """    
    MUTATE_RATE = 1.0/IND_SIZE 
    
    
    
    # prepare data: for each label, split the data into 25% test and 75 train
    raw = spectrum_data_new.values

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
    
    model = RandomForestClassifier(n_estimators= NUM_TREES) # create a random forest with NUM_TREES = 20 
    model.fit(X_train, y_train) # train the model
    estimators = model.estimators_ # get all the trees
    
    # implement individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    # implement functions for initialize population and create individual
    toolbox = base.Toolbox() # create a toolbox of operators for our GA algorithm
    toolbox.register("indices", random.sample, range(NUM_TREES), NUM_TREES) # this is a helper function for creating
                                                                            # each individual
    toolbox.register("individual", tools.initIterate, creator.Individual,   # this is the function for creating an 
                     toolbox.indices)                                       # individual


    #toolbox.individual()  # a test 

    # implement function for creating a population
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n = POP_SIZE)

    #toolbox.population()  # a test
    
    def sub_rf_predict(sub_rf, X_test):
        """
        return the predict result using the sub_rf and X_test data;
        the rule is that predict result(labels) with the maximum number of votes wins.
        """
        predict_results = []
        for tree in sub_rf:
            prediction = tree.predict(X_test)
            # record prediction result for a tree
            predict_results.append(prediction)

        # compute the vote_result, i.e. the final result
        y_predict = [0]*len(X_test)
        for idx in range(len(X_test)):
            # for each test data
            # create a vote result
            v_result = vote_result()
            for predict_tree in predict_results:
                v_result[predict_tree[idx]] += 1

            # final result
            y_predict[idx] = keywithmaxval(v_result)

        return  np.array(y_predict, dtype = float)

    # helper function
    y_set = set(y_test).union(y_train)
    def vote_result():
        result = {}
        for k in y_set:
            result[k] = 0
        return result

    def keywithmaxval(d):
        """ a) create a list of the dict's keys and values; 
        b) return the key with the max value"""  
        v=list(d.values())
        k=list(d.keys())
        return k[v.index(max(v))]
    
    
    def evaluate(individual):
        # return the accuracy on the test data
        sub_random_forest = []
        for tree_idx in individual[0: IND_SIZE]:
            sub_random_forest.append(estimators[tree_idx])

        predict_sub_trees = sub_rf_predict(sub_random_forest, X_test)
        # print(predict_sub_trees.__repr__())
        # score = precision_score(y_test, predict_sub_trees, average = 'macro')
        cf_matrix = evaluate_confusion_matrix(individual)
        
        pseudo_acc_case_rate = cf_matrix[1, 1] / np.sum(cf_matrix)
        bad_case_rate = (cf_matrix[1, 0] + cf_matrix[1, 2]) / np.sum(cf_matrix)
        undesired_case_rate = (cf_matrix[0, 1] + cf_matrix[2, 1]) / np.sum(cf_matrix)
        
        # give accuracy more weight
        score = (0.4 * pseudo_acc_case_rate - 0.4 * bad_case_rate - 0.2 * undesired_case_rate) # the max score is 1 and the min score is 0.
        return  score,  # must return an tuple!!!!
    
    def evaluate_confusion_matrix(individual):
        # return the confusion matrix of a model
        sub_random_forest = []
        for tree_idx in individual[0: IND_SIZE]:
            sub_random_forest.append(estimators[tree_idx])  

        predict_sub_trees = sub_rf_predict(sub_random_forest, X_test)
        cf_matrix = confusion_matrix(y_test, predict_sub_trees)
        return  cf_matrix 

    
    
    # implement mutation operator
    mutation_op = tools.mutShuffleIndexes
    
    
    # implement crossover
    def crossover_op(ind1, ind2):
        # only cross over the first IND_SIZE elements in the individual in place
        crossover_idx = random.randint(0, IND_SIZE - 2)
        # print(crossover_idx)
        temp = toolbox.clone(ind1[crossover_idx + 1: IND_SIZE])
        ind1[crossover_idx + 1: IND_SIZE] = ind2[crossover_idx + 1: IND_SIZE]
        ind2[crossover_idx + 1: IND_SIZE] = temp
        return (ind1, ind2)
    
    # implement selection operator
    selection_op = tools.selTournament
    
    
    # register everything in our toolbox
    toolbox.register("mate", crossover_op)
    toolbox.register("mutate", mutation_op, indpb = MUTATE_RATE)
    toolbox.register("select", selection_op, tournsize=3)
    toolbox.register("evaluate", evaluate)
    
    
    h_fame = tools.HallOfFame(100) # keep track of the first 100 best individuals and store them in h_fame

    pop = toolbox.population()
    final_pop = algorithms.eaSimple(pop, toolbox, cxpb = CX_RATE, mutpb=MUTATE_RATE, ngen=1000, 
                                    stats = None, halloffame = h_fame, verbose = False)
    
    # accuracy_of_the_best_individual = evaluate(h_fame[0])
    # accuracy_of_the_whole_trees_model = accuracy_score(y_test, model.predict(X_test))
    cf_matrix_RF_model = confusion_matrix(y_test, model.predict(X_test))
    cf_matrix_GA_RF_model = evaluate_confusion_matrix(h_fame[0])
    
    return cf_matrix_GA_RF_model, cf_matrix_RF_model, evaluate_confusion_matrix, h_fame, estimators


def measure_model_performance(C):
    """
    C is a confusion matrix
    accuracy, sensitivity, specificity: the higher the better
    FP_rate: the lower the better
    
    Now, only compute accuracy
    """
    accuracy = (C[0,0] + C[1, 1] + C[2, 2]) / np.sum(C)
    sensitivity = 0
    specificity = 0 
    FP_rate = 0
    
    cf_matrix = C
    pseudo_acc_case_rate = cf_matrix[1, 1] / np.sum(cf_matrix)
    bad_case_rate = (cf_matrix[1, 0] + cf_matrix[1, 2]) / np.sum(cf_matrix)
    undesired_case_rate = (cf_matrix[0, 1] + cf_matrix[2, 1]) / np.sum(cf_matrix)
    
    left_diag_case_rate = (cf_matrix[0, 0] + cf_matrix[2, 2]) / np.sum(cf_matrix)
    right_diag_case_rate = (cf_matrix[0, 2] + cf_matrix[2, 0]) / np.sum(cf_matrix)
        # give accuracy more weight
    score = (0.4 * pseudo_acc_case_rate - 0.4 * bad_case_rate - 0.2 * undesired_case_rate)
    
    return accuracy, sensitivity, specificity, FP_rate, score

def show_result():
    print("mean accuracy = ",  np.mean(GA_accuracy_result))
    print("std accuracy = ", np.std(GA_accuracy_result))
    print("Max accuracy = ", np.max(GA_accuracy_result))
    print("confusion Matrix for model with max accuracy is \n", confusion_matrices_list_GA_RF[np.argmax(GA_accuracy_result)])
    print("Max score = ", np.max(GA_score_list))
    print("confusion Matrix for model with max score is \n", confusion_matrices_list_GA_RF[np.argmax(GA_score_list)])


def normalize_score(scores, max_score, min_score):
    np_scores = np.array(scores)
    return (np_scores - min_score)/(max_score - min_score)

# main
GA_accuracy_result = []
RF_accuracy_result = []

GA_sensitivity_result = []
RF_sensitivity_result = []

GA_specificity_result = []
RF_specificity_result = []

GA_FP_rate_result = []
RF_FP_rate_result = []

confusion_matrices_list_GA_RF = []
h_fame_list = []
GA_RF_model_list = []

GA_score_list = []
RF_score_list = []


for i in range(experiment_times):  
    print("Working hard to generate the ", i, "th RF, RF_GA model")
    cf_matrix_GA_RF_model, cf_matrix_RF_model, evaluate_confusion_matrix, h_fame, estimators = run_GA(num_trees, IND_SIZE, POP_SIZE)
    
    GA_accuracy, GA_sensitivity, GA_specificity, GA_FP_rate, GA_score = measure_model_performance(cf_matrix_GA_RF_model)
    RF_accuracy, RF_sensitivity, RF_specificity, RF_FP_rate, RF_score = measure_model_performance(cf_matrix_RF_model)

    GA_accuracy_result.append(GA_accuracy)
    RF_accuracy_result.append(RF_accuracy)

    GA_sensitivity_result.append(GA_sensitivity)
    RF_sensitivity_result.append(RF_sensitivity)

    GA_specificity_result.append(GA_specificity)
    RF_specificity_result.append(RF_specificity)

    GA_FP_rate_result.append(GA_FP_rate)
    RF_FP_rate_result.append(RF_FP_rate)
    
    confusion_matrices_list_GA_RF.append(cf_matrix_GA_RF_model)
    GA_RF_model_list.append(estimators)
    
    GA_score_list.append(GA_score)
    RF_score_list.append(RF_score)

# compute normalized score
GA_norm_score = normalize_score(GA_score_list, max_score, min_score)
RF_norm_score = normalize_score(RF_score_list, max_score, min_score)
show_result()
print("GA_Normed_scores are ", GA_norm_score)
print("RF_Normed_scores are ", RF_norm_score)

t, t_p_value = scipy.stats.ttest_ind(GA_norm_score, RF_norm_score, equal_var = False)
u, u_p_value = scipy.stats.mannwhitneyu(GA_norm_score, RF_norm_score)
