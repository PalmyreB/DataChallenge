#! python3.6
# -*- coding: cp1252 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn import linear_model, decomposition, metrics, model_selection, preprocessing
from numpy import linalg
import pprint

import warnings
warnings.filterwarnings("ignore")

print("----------------------------------------Import matrix--------------------------------------------------------------")
#matrix_test = np.array([[[('stub.exe_9f01000', 'sample.exe_1356d000')], ['sample.exe_1356d000,1fa617,api_1487\n', 'sample.exe_1356d000,1fa626,api_0496\n', 'sample.exe_1356d000,1fa62f,api_0338\n']], [[('stub.exe_9f01000', 'sample.exe_1356d000')], ['sample.exe_1356d000,1fa617,api_1487\n', 'sample.exe_1356d000,1fa626,api_0496\n', 'sample.exe_1356d000,1fa62f,api_0338\n']], [[('stub.exe_1319c000', 'sample.exe_11ad0000'), ('stub.exe_1319c000', 'sample.exe_11ad0000')], ['sample.exe_1356d000,1fa617,api_1487\n', 'sample.exe_1356d000,1fa626,api_0496\n', 'sample.exe_1356d000,1fa62f,api_0338\n']]])
matrix_test_total = np.load("../palmyre/matrix.raw")
matrix_test = matrix_test_total[0:50, :]
N = len(matrix_test)

def clean(line):
    for i in range(0, len(line[1])):
        string = re.sub("\n", "", line[1][i])
        line[1][i] = np.array(string.split(","))
        
print("----------------------------------------Cleaning--------------------------------------------------------------")
np.apply_along_axis(lambda line: clean(line), 1, matrix_test)

print("----------------------------------------Printing------------------------------------------------------")
pprint.pprint(matrix_test)

p = 11 #Nb de features

X = []

print("----------------------------------------Feature Engineering------------------------------------------------------")
count_line_tab = []

def count(line):
    unique, counts = np.unique(line, return_counts=True)
    return dict(zip(unique, counts))

#Count
np.apply_along_axis(lambda line: count_line_tab.append(count(line[1])), 1, matrix_test)

npcount_line_tab = np.array(count_line_tab)
print("----------------------------------------Printing------------------------------------------------------")
#pprint.pprint(npcount_line_tab)

def count_max_same_api_call(dict_line):
    maxi = 0
    for key in dict_line:
        if str(key)[0:4] == "api_":
            if dict_line[key] > maxi:
                maxi = dict_line[key]
    return maxi

def count_min_same_api_call(dict_line):
    mini = 1e20
    for key in dict_line:
        if str(key)[0:4] == "api_":
            if dict_line[key] < mini:
                mini = dict_line[key]
    return mini

def count_moy_api_call(dict_line):
    summ = 0.0
    counter = 0
    for key in dict_line:
        if str(key)[0:4] == "api_":
            summ += dict_line[key]
            counter += 1
    if counter != 0:
        return summ/counter
    else:
        return 0.0

def count_max_same_rsi_call(dict_line):
    maxi = 0
    for key in dict_line:
        if str(key)[0:4] != "api_" and "." not in str(key):
            if dict_line[key] > maxi:
                maxi = dict_line[key]
    return maxi

def count_min_same_rsi_call(dict_line):
    mini = 1e20
    for key in dict_line:
        if str(key)[0:4] != "api_" and "." not in str(key):
            if dict_line[key] < mini:
                mini = dict_line[key]
    return mini

def count_moy_rsi_call(dict_line):
    summ = 0.0
    counter = 0
    for key in dict_line:
        if str(key)[0:4] != "api_" and "." not in str(key):
            summ += dict_line[key]
            counter += 1
    if counter != 0:
        return summ/counter
    else:
        return 0.0

def count_max_api_call_par_process(dict_line):
    maxi = 0
    for key in dict_line:
        if "." in str(key):
            if dict_line[key] > maxi:
                maxi = dict_line[key]
    return maxi

def count_min_api_call_par_process(dict_line):
    mini = 1e20
    for key in dict_line:
        if "." in str(key):
            if dict_line[key] < mini:
                mini = dict_line[key]
    return mini

def count_moy_api_call_par_process(dict_line):
    summ = 0.0
    counter = 0
    for key in dict_line:
        if "." in str(key):
            summ += dict_line[key]
            counter += 1
    if counter != 0:
        return summ/counter
    else:
        return 0.0

#Total Nb process generation
np.apply_along_axis(lambda line: X.append(len(line[0])), 1, matrix_test)
#Total Nb api/rsi calls
np.apply_along_axis(lambda line: X.append(len(line[1])), 1, matrix_test)

#Nb min d'api call par process
for i in range(0, len(npcount_line_tab)):
    X.append(count_min_api_call_par_process(npcount_line_tab[i]))
#Nb max d'api call par process
for i in range(0, len(npcount_line_tab)):
    X.append(count_max_api_call_par_process(npcount_line_tab[i]))
#Nb moy d'api call par process
for i in range(0, len(npcount_line_tab)):
    X.append(count_moy_api_call_par_process(npcount_line_tab[i]))

#Nb min same api call
for i in range(0, len(npcount_line_tab)):
    X.append(count_min_same_api_call(npcount_line_tab[i]))
#Nb max same api call
for i in range(0, len(npcount_line_tab)):
    X.append(count_max_same_api_call(npcount_line_tab[i]))
#Nb moy same api call
for i in range(0, len(npcount_line_tab)):
    X.append(count_moy_api_call(npcount_line_tab[i]))

#Nb min same rsi call
for i in range(0, len(npcount_line_tab)):
    X.append(count_min_same_rsi_call(npcount_line_tab[i]))
#Nb max same rsi call
for i in range(0, len(npcount_line_tab)):
    X.append(count_max_same_rsi_call(npcount_line_tab[i]))
#Nb moy same rsi call
for i in range(0, len(npcount_line_tab)):
    X.append(count_moy_rsi_call(npcount_line_tab[i]))

npX = np.reshape(X, (p, N))

print("----------------------------------------Printing------------------------------------------------------")
pprint.pprint(npX)

#Min process generation
#Max process generation
#Threshold process generation -> np.where(y_tr >= 1800, 1, 0)

#sequence of behaviors
#nb/min/max of subprocess id
#api is greater than 1
#api is 0

#behavior graph has boucle
#nombre d'api différentes
#nombre d'adresses différentes


# Set up a cross-validation with sklearn
#skf = model_selection.StratifiedKFold()
#sk_folds = skf.split(X_clf, y_clf)

