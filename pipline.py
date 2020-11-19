#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from reduce_dimensions import Xtrain_std, Xtest_std, ytrain, ytest, np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(solver='lbfgs'))
pipe_lr.fit(Xtrain_std, ytrain)
y_pred = pipe_lr.predict(Xtest_std)
print('Test Accuracy: %.3f' % pipe_lr.score(Xtest_std, ytest))

scores = cross_val_score(estimator=pipe_lr, X=Xtrain_std, y=ytrain,
                         cv=10,n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


pipe_svc = make_pipeline(StandardScaler(), SVC())
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'svc__C': param_range, 
               'svc__kernel': ['sigmoid']},
              {'svc__C': param_range, 
               'svc__gamma': param_range, 
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, 
                  scoring='accuracy', refit=True, cv=10, n_jobs=-1)
gs = gs.fit(Xtrain_std, ytrain)
print(gs.best_score_)
print(gs.best_params_)
clf = gs.best_estimator_
print('Test accuracy: %.3f' % clf.score(Xtest_std, ytest))



