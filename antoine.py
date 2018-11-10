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

p = 3 #Nb de features

X = []

print("----------------------------------------Feature Engineering------------------------------------------------------")
def count(line):
    unique, counts = np.unique(line, return_counts=True)
    return dict(zip(unique, counts))

def count_api_by_process(line):
    dict_count = count(line)
    summ = 0.0
    counter = 0
    for key in dict_count:
        try:
            int(key, 16)
        except: #not RSI
            if str(key)[0:4] != "api_":
                summ += dict_count[key]
                counter += 1
    if counter != 0:
        return summ/counter
    else:
        return 0.0
                
                

#Total Nb process generation
np.apply_along_axis(lambda line: X.append(len(line[0])), 1, matrix_test)
#Total Nb api calls
np.apply_along_axis(lambda line: X.append(len(line[1])), 1, matrix_test)
#Nb moyen d'api calls par process
np.apply_along_axis(lambda line: X.append(count_api_by_process(line[1])), 1, matrix_test)


npX = np.reshape(X, (p, N))

print("----------------------------------------Printing------------------------------------------------------")
pprint.pprint(npX)

#Min process generation
#Max process generation
#Threshold process generation -> np.where(y_tr >= 1800, 1, 0)
#the total number of process generation

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

