import numpy as np
import matplotlib.pyplot as plt

matrix_test = np.array([[[('stub.exe_9f01000', 'sample.exe_1356d000')], ['sample.exe_1356d000,1fa617,api_1487\n', 'sample.exe_1356d000,1fa626,api_0496\n', 'sample.exe_1356d000,1fa62f,api_0338\n']], [[('stub.exe_9f01000', 'sample.exe_1356d000')], ['sample.exe_1356d000,1fa617,api_1487\n', 'sample.exe_1356d000,1fa626,api_0496\n', 'sample.exe_1356d000,1fa62f,api_0338\n']], [[('stub.exe_1319c000', 'sample.exe_11ad0000'), ('stub.exe_1319c000', 'sample.exe_11ad0200')], ['sample.exe_1356d000,1fa617,api_1487\n', 'sample.exe_1356d000,1fa626,api_0496\n', 'sample.exe_1356d000,1fa62f,api_0338\n']]])


def bar_char(Y,name):
    index = np.arange(len(Y))
    plt.bar(index,Y)
    plt.title(name)
    plt.savefig(name+'.png')

#nombre d'api
def nb_api(X):
    Y=np.zeros(len(X))
    for i in range(0,len(X)):
        Y[i]=len(X[i][1])
    print(Y)
    bar_char(Y,"nombre d'api")   


#nombre de process
def nb_process(X):
    Y=np.zeros(len(X))
    for i in range(0,len(X)):
        Y[i]=len(X[i][0])
    bar_char(Y,"nombre de process") 
    
    
#nombre de branches max d'un noeud
def nb_enfant_max(X):
    Y=np.zeros(len(X))
    for i in range(0,len(X)):
        max=0
        for j in range(0,len(X[i][0])):
            parent,enfant = X[i][0][j]
            noeud = parent
            max_prov=0
            for k in range(0,len(X[i][0])):
                parent2,enfant2 = X[i][0][k]
                if parent==parent2:
                    max_prov +=1 
            if max_prov > max : 
                max=max_prov
        Y[i] = max
    bar_char(Y,"nombre maximal d'enfants")
    
def nb_parent_max(X):
    Y=np.zeros(len(X))
    for i in range(0,len(X)):
        max=0
        for j in range(0,len(X[i][0])):
            parent,enfant = X[i][0][j]
            noeud = enfant
            max_prov=0
            for k in range(0,len(X[i][0])):
                parent2,enfant2 = X[i][0][k]
                if enfant==enfant2:
                    max_prov +=1 
            if max_prov > max : 
                max=max_prov
        Y[i] = max
    bar_char(Y,"nombre maximal de parents")

def nb_branche_max(X):
    Y=np.zeros(len(X))
    for i in range(0,len(X)):
        max=0
        for j in range(0,len(X[i][0])):
            parent,enfant = X[i][0][j]
            noeud = parent
            max_prov=0
            for k in range(0,len(X[i][0])):
                parent2,enfant2 = X[i][0][k]
                if parent==enfant2 or parent==parent2:
                    max_prov +=1 
            if max_prov > max : 
                max=max_prov
        Y[i] = max
    bar_char(Y,"nombre maximal de branches")

#nombre max d'occurence d'une API
def nb_occurence_api_max(X):
    Y=np.zeros(len(X))
    for i in range(0,len(X)):
        max=0
        for j in range(0,len(X[i][1])):
            api= X[i][1][j][2]
            max_prov=0
            for k in range(0,len(X[i][1])):
                if api==X[i][1][k][2]:
                    max_prov +=1 
            if max_prov > max : 
                max=max_prov
        Y[i] = max
    bar_char(Y,"nombre maximal d'occurences d'une API")

nb_occurence_api_max(matrix_test)



    
  



