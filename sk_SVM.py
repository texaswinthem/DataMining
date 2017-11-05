
import numpy as np
from estimatorz import SVCCV, SGDCCV, SVC
from Preprocessingz import SC, RobSc, PolFeat, MAS, PCA
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import sys


print('Reading data:')
lines_train = np.loadtxt("data/handout_train.txt")
lines_test = np.loadtxt("data/handout_test.txt")
Xtrain = lines_train[:, 1:]
Ytrain = lines_train[:, 0]
Xtest = lines_test[:, 1:]
Ytest = lines_test[:, 0]







MethList = [

    SVC]




for Meth in MethList:
# Format: pipes is a list of:
# [list_of_consecutive_preprocessings, estimator_to_use, whether_to_adjust_parameters_with_CV]
########################################################################
# If the preprocessing is finished, the preprocessed data will be saved for later use,
# but only after all consecutive preprocessings of an element of pipes.
# A possible submision and the score file will be produced for
# every entry of the pipes list (if n_test == 138)
    pipes = [

        [PCA, Meth],
        [RobSc, Meth],
        [MAS, Meth],
        [PolFeat, Meth],
        [SC, Meth],

    ]


search_parm={'kernel': ['linear','rbf','poly','sigmoid'],
             'degree': np.linspace(2,5,4),
            'C': np.geomspace(1e-6,100,num=9)
             }

grid_search = GridSearchCV(SVC(C=1),search_parm, cv=5)


print("Performing grid search...")

print("parameters:")
print(search_parm)

grid_search.fit(Xtrain, Ytrain)

print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(search_parm.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
Ytest_predict=grid_search.best_estimator_.predict(Xtest)
print("Train Accuracy Score: ",grid_search.best_score_)
print("Test Accuracy-Score: ", accuracy_score(Ytest,Ytest_predict))