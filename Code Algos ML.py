# Importing useful libraries

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

#Loading data:

X_train=....... 
y_train=.......
X_test=........
y_test=........


#Feature Engineering 

WIP...

### Algorithms ### (use gridsearch)

#Naïve Bayes



#Decision Trees




#RandomForest





#Boosting
learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
param_grid={'learning_rate':learning_rates}
gb = GradientBoostingClassifier(n_estimators=20, max_features=2, max_depth = 2, random_state = 0)
best_gb=GridSearchCV(estimator, param_grid, cv=20)
best_gb.fit(X_train, y_train)
print("Learning rate: ", best_gb.best_params_)
print("Accuracy score (training): {0:.4f}".format(best_gb.score(X_train, y_train)))
print("Accuracy score (testing): {0:.4f}".format(best_gb.score(X_test, y_test)))


