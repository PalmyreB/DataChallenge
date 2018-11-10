###I- Importing useful libraries

import numpy as np
import math
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics: import auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import shutil
from sklearn.cross_validation import train_test_split

###II- Loading data:
X=np.load("antoine/main.py") #1000 raws
y= np.load("true_labels_training.txt")
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2)
#X_val=np.load(



### III- feature engineering


###IV- Algorithms ### (use gridsearch)

#1. Naïve Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print("Accuracy score (training): {0:.4f}".format(gnb.score(X_train, y_train)))
print("Accuracy score (testing): {0:.4f}".format(gnb.score(X_test, y_test)))

y_scores_gnb = gnb.decision_function(X_test)
fpr_gnb, tpr_gnb, threshold = roc_curve(list(y_test), y_scores_gnb)
roc_auc_gnb = auc(fpr_gnb, tpr_gnb)

print("Area under ROC curve = {:0.2f}".format(roc_auc_gnb))
#plt.plot(fpr_gnb, tpr_gnb)
#plt.xlabel('False positive rate')
#plt.ylabel('True positive rate')
#plt.title('ROC curve')
#plt.legend(loc='Gaussian Naive Bayes')
#plt.show()    
#2. SVM 
C= list(range(100)) #Tune on this param or min_samples_split or max_features
param_grid={'C':C}
svm = SVC(kernel='rbf)
best_svm=GridSearchCV(svm, param_grid, cv=20)
best_svm.fit(X_train, y_train)
print("Learning rate: ", best_svm.best_params_)
print("Accuracy score (training): {0:.4f}".format(best_svm.score(X_train, y_train)))
print("Accuracy score (testing): {0:.4f}".format(best_svm.score(X_test, y_test)))

y_scores_svm = svm.decision_function(X_test)
fpr_svm, tpr_svm, threshold = roc_curve(list(y_test), y_scores_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

print("Area under ROC curve = {:0.2f}".format(roc_auc_svm))
#plt.plot(fpr_svm, tpr_svm)
#plt.xlabel('False positive rate')
#plt.ylabel('True positive rate')
#plt.title('ROC curve')
#plt.legend(loc='best for SVM')
#plt.show()    

#3. Decision Trees
max_depth= list(range(20)) #Tune on this param or min_samples_split or max_features
param_grid={'max_depth':max_depth}
dt = DecisionTreeClassifier()
best_dt=GridSearchCV(dt, param_grid, cv=20)
best_dt.fit(X_train, y_train)
#print("Learning rate: ", best_dt.best_params_)
#print("Accuracy score (training): {0:.4f}".format(best_dt.score(X_train, y_train)))
#print("Accuracy score (testing): {0:.4f}".format(best_dt.score(X_test, y_test)))

y_scores_dt = dt.decision_function(X_test)
fpr_dt, tpr_dt, threshold = roc_curve(list(y_test), y_scores_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)

#print("Area under ROC curve = {:0.2f}".format(roc_auc_dt))
#plt.plot(fpr_dt, tpr_dt)
#plt.xlabel('False positive rate')
#plt.ylabel('True positive rate')
#plt.title('ROC curve')
#plt.legend(loc='best for Decision Trees')
#plt.show()    



#4. RandomForest
estimators = [10,15] #Tune on this param or max_depth, or max_features
param_grid={'n_estimators':estimators}
rf = RandomForestClassifier()
best_rf=GridSearchCV(rf, param_grid, cv=20)
best_rf.fit(X_train, y_train)
print("Learning rate: ", best_rf.best_params_)
print("Accuracy score (training): {0:.4f}".format(best_rf.score(X_train, y_train)))
print("Accuracy score (testing): {0:.4f}".format(best_rf.score(X_test, y_test)))

y_scores_rf = rf.decision_function(X_test)
fpr_rf, tpr_rf, threshold = roc_curve(list(y_test), y_scores_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

print("Area under ROC curve = {:0.2f}".format(roc_auc_rf))
#plt.plot(fpr_rf, tpr_rf)
#plt.xlabel('False positive rate')
#plt.ylabel('True positive rate')
#plt.title('ROC curve')
#plt.legend(loc='best for RandomForest')
#plt.show()    




#5. Boosting
learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1] #Tune on this param or n_estimators or max_features
param_grid={'learning_rate':learning_rates}
gb = GradientBoostingClassifier(n_estimators=20, max_features=2, max_depth = 2, random_state = 0)
best_gb=GridSearchCV(gb, param_grid, cv=20)
best_gb.fit(X_train, y_train)
print("Learning rate: ", best_gb.best_params_)
print("Accuracy score (training): {0:.4f}".format(best_gb.score(X_train, y_train)))
print("Accuracy score (testing): {0:.4f}".format(best_gb.score(X_test, y_test)))

y_scores_gb = gb.decision_function(X_test)
fpr_gb, tpr_gb, threshold = roc_curve(list(y_test), y_scores_gb)
roc_auc_gb = auc(fpr_gb, tpr_gb)

print("Area under ROC curve = {:0.2f}".format(roc_auc_gb))
#plt.plot(fpr_gb, tpr_gb)
#plt.xlabel('False positive rate')
#plt.ylabel('True positive rate')
#plt.title('ROC curve')
#plt.legend(loc='best for XGBoost')
#plt.show()    

          
          
#### Best classifier
          
Best_classifier=best_gb
best_classifier.fit(X_train,y_train)
y_pred=pd.DataFrame(best_classifier.predict(X_val))
file_path=''
file_name='answer.txt'
zip_file_directory=''
np.savetxt(file_path+file_name, y_pred.values, fmt='%d')
#shutil.make_archive(file_path+file_name, 'zip',zip_file_directory)
          
