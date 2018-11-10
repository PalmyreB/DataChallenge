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

matrix_test = np.array([[[('stub.exe_9f01000', 'sample.exe_1356d000')], ['sample.exe_1356d000,1fa617,api_1487\n', 'sample.exe_1356d000,1fa626,api_0496\n', 'sample.exe_1356d000,1fa62f,api_0338\n']], [[('stub.exe_9f01000', 'sample.exe_1356d000')], ['sample.exe_1356d000,1fa617,api_1487\n', 'sample.exe_1356d000,1fa626,api_0496\n', 'sample.exe_1356d000,1fa62f,api_0338\n']], [[('stub.exe_1319c000', 'sample.exe_11ad0000'), ('stub.exe_1319c000', 'sample.exe_11ad0000')], ['sample.exe_1356d000,1fa617,api_1487\n', 'sample.exe_1356d000,1fa626,api_0496\n', 'sample.exe_1356d000,1fa62f,api_0338\n']]])

matrix_test_clean = []

def clean(line):
    matrix_test_clean.append([line[0], []])
    for i in range(0, len(line[1])):
        string = re.sub("\n", "", line[1][i])
        matrix_test_clean[-1][1].append(string.split(","))
        
## Cleaning
np.apply_along_axis(lambda x: clean(x), 1, matrix_test)

print("------------------------------------------------------------------------------------------------------")

pprint.pprint(matrix_test_clean)

#Nb process generation
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

#the total number of api_call


# Set up a cross-validation with sklearn
#skf = model_selection.StratifiedKFold()
#sk_folds = skf.split(X_clf, y_clf)

