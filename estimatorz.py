from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
import numpy as np
SGDCCV = [SGDClassifier(learning_rate="optimal", shuffle=True), 'SGDCCV', True, {'loss': ('hinge', 'log'),
                                                                                 'penalty': ('l2', 'l1'),
                                                                                 'alpha': np.logspace(1e-6, 100,
                                                                                                      num=6)}]
SVCCV = [SVC(), 'SVCCV', True, [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-4],
                                 'C': [1, 10, 100, 1000]},
                                {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]]