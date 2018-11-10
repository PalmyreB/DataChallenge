import numpy as np
import math
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import shutil


X=pd.read_csv("antoine/X.csv").T #1000 raws
X_val=np.load("palmyre/matrix_validation.npy")
print("Shape X_val",np.shape(X_val))
print("Shape_X_train", np.shape(X))
with open('true_labels_training.txt') as f:
    lines = f.readlines()
y=[i for i in lines[0]]
yy=y[:1000]
print(np.shape(y))
X_train, X_test, y_train, y_test= train_test_split(X,yy,test_size=0.2)

#5. Boosting
learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1] #Tune on this param or n_estim$
param_grid={'learning_rate':learning_rates}
gb = GradientBoostingClassifier(n_estimators=20, max_features=2, max_depth = 2,$
best_gb=GridSearchCV(gb, param_grid, cv=20)
best_gb.fit(X_train, y_train)
print("Learning rate: ", best_gb.best_params_)
print("Accuracy score Boosting (training): {0:.4f}".format(best_gb.score(X_trai$
print("Accuracy score (testing): {0:.4f}".format(best_gb.score(X_test, y_test)))

y_pred=np.array(best_gb.predict(X_val))
file_path=''
file_name='answer.txt'
zip_file_directory=''
np.savetxt(file_path+file_name, y_pred.values, fmt='%d')
shutil.make_archive(file_path+file_name, 'zip',zip_file_directory)
