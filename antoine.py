#! python3.6
# -*- coding: cp1252 -*-
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import savefig
import pandas as pd
import re
from sklearn import linear_model, decomposition, metrics, model_selection, preprocessing, svm
import seaborn as sns
from numpy import linalg
import pprint
import time

import warnings
warnings.filterwarnings("ignore")

start_time = time.time()
print("----------------------------------------Import matrix--------------------------------------------------------------")
#matrix_test = np.array([[[('stub.exe_9f01000', 'sample.exe_1356d000')], ['sample.exe_1356d000,1fa617,api_1487\n', 'sample.exe_1356d000,1fa626,api_0496\n', 'sample.exe_1356d000,1fa62f,api_0338\n']], [[('stub.exe_9f01000', 'sample.exe_1356d000')], ['sample.exe_1356d000,1fa617,api_1487\n', 'sample.exe_1356d000,1fa626,api_0496\n', 'sample.exe_1356d000,1fa62f,api_0338\n']], [[('stub.exe_1319c000', 'sample.exe_11ad0000'), ('stub.exe_1319c000', 'sample.exe_11ad0000')], ['sample.exe_1356d000,1fa617,api_1487\n', 'sample.exe_1356d000,1fa626,api_0496\n', 'sample.exe_1356d000,1fa62f,api_0338\n']]])
matrix_test_total = np.load("../palmyre/matrix.raw")
matrix_validation_total = np.load("../palmyre/matrix_validation.npy")
matrix_test = matrix_test_total[0:100, :]
matrix_validation = matrix_test_total[1:, :]
N = len(matrix_test)

end_time = time.time()
print("----------------------------------------Done in "+str(end_time-start_time)+" s.------------------------------------------------------")
start_time = time.time()

start_time = time.time()
print("----------------------------------------Import label--------------------------------------------------------------")
with open('../true_labels_training.txt') as f:
    lines = f.readlines()
labels=np.array([int(lines[0][i]) for i in range(0,len(lines[0]))])

pprint.pprint(labels)

end_time = time.time()
print("----------------------------------------Done in "+str(end_time-start_time)+" s.------------------------------------------------------")
start_time = time.time()

def clean(line):
    for i in range(0, len(line[1])):
        string = re.sub("\n", "", line[1][i])
        line[1][i] = np.array(string.split(","))
        
print("----------------------------------------Cleaning--------------------------------------------------------------")
np.apply_along_axis(lambda line: clean(line), 1, matrix_test)

end_time = time.time()
print("----------------------------------------Done in "+str(end_time-start_time)+" s.------------------------------------------------------")
start_time = time.time()

p = 13 #Nb de features
pourcenttrain = 0.9
X = []

print("----------------------------------------Feature Engineering------------------------------------------------------")
count_line_tab = []

def count(line):
    unique, counts = np.unique(line, return_counts=True)
    return dict(zip(unique, counts))

#Count
np.apply_along_axis(lambda line: count_line_tab.append(count(line[1])), 1, matrix_test)

npcount_line_tab = np.array(count_line_tab)


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

def count_nb_api(dict_line):
    counter = 0
    for key in dict_line:
        if str(key)[0:4] == "api_":
            counter += 1
    return counter

def count_nb_rsi(dict_line):
    counter = 0
    for key in dict_line:
        if str(key)[0:4] != "api_" and "." not in str(key):
            counter += 1
    return counter

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

#Nb api differentes
for i in range(0, len(npcount_line_tab)):
    X.append(count_nb_api(npcount_line_tab[i]))
#Nb rsi differentes
for i in range(0, len(npcount_line_tab)):
    X.append(count_nb_rsi(npcount_line_tab[i]))

npX = np.reshape(X, (p, N))

end_time = time.time()
print("----------------------------------------Done in "+str(end_time-start_time)+" s.------------------------------------------------------")
start_time = time.time()

print("----------------------------------------Save in file------------------------------------------------------")
np.savetxt("X.csv", npX, delimiter=",")

end_time = time.time()
print("----------------------------------------Done in "+str(end_time-start_time)+" s.------------------------------------------------------")
start_time = time.time()

print("----------------------------------------Printing------------------------------------------------------")
#pprint.pprint(matrix_test)
#pprint.pprint(npcount_line_tab)
#pprint.pprint(npX)

end_time = time.time()
print("----------------------------------------Done in "+str(end_time-start_time)+" s.------------------------------------------------------")
start_time = time.time()

print("----------------------------------------Plotting------------------------------------------------------")
for i in range(0, p):
    plt.clf()
    df = pd.DataFrame(data=npX[:, i], index=np.arange(0,p), columns=[0])
    svm = sns.distplot(df)
    figure = svm.get_figure() 
    figure.savefig("figures/output_feature_"+str(i)+".png")

end_time = time.time()
print("----------------------------------------Done in "+str(end_time-start_time)+" s.------------------------------------------------------")
start_time = time.time()

print("----------------------------------------Scaling------------------------------------------------------")
X_train = pd.DataFrame(data=npX, index=np.arange(0,p), columns=np.arange(0,N))
scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_validation = pd.DataFrame(scaler.transform(X_validation), columns=X_validation.columns)

end_time = time.time()
print("----------------------------------------Done in "+str(end_time-start_time)+" s.------------------------------------------------------")
start_time = time.time()

print("----------------------------------------Training------------------------------------------------------")
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC(gamma="scale")
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(X_train, label)
label_result = clf.predict(X_validation)

end_time = time.time()
print("----------------------------------------Done in "+str(end_time-start_time)+" s.------------------------------------------------------")
start_time = time.time()

print("----------------------------------------Save in file------------------------------------------------------")
file_name='answer.txt'
np.savetxt(file_name, label_result.values, fmt='%d')

end_time = time.time()
print("----------------------------------------Done in "+str(end_time-start_time)+" s.------------------------------------------------------")
start_time = time.time()
