###I- Importing useful libraries

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.ensemble import RandomForestClassifier


###II- Loading data:

X_train=....... 
y_train=.......
X_test=........
y_test=........


###IV- Algorithms ### (use gridsearch)

#1. Naïve Bayes



#2. Decision Trees
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
plt.plot(fpr_rf, tpr_rf)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best for RandomForest')
plt.show()    



#3. RandomForest
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
plt.plot(fpr_rf, tpr_rf)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best for RandomForest')
plt.show()    




#Boosting
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
plt.plot(fpr_gb, tpr_gb)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best for XGBoost')
plt.show()    
