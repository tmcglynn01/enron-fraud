#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" a basic script for importing student's POI identifier,
    and checking the results that they get from it 
 
    requires that the algorithm, dataset, and features list
    be written to my_classifier.pkl, my_dataset.pkl, and
    my_feature_list.pkl, respectively
    that process should happen at the end of poi_id.py
"""
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from sys import version_info

# PLEASE USE PYTHON 
assert version_info >= (3, 0)


def test_classifier(clf, dataset, feature_list): 
    # Split the data
    df = dataset.copy()
    X, y = df.iloc[:,1:], df.iloc[:,0]
    sss = StratifiedShuffleSplit(test_size=0.4, random_state=42, n_splits=10)
    for trainidx, testidx in sss.split(X, y):
        Xtrain, Xtest = X.iloc[trainidx], X.iloc[testidx]
        ytrain, ytest = y.iloc[trainidx], y.iloc[testidx]
    
    
    clf.fit(Xtrain, ytrain)
    predictions = clf.predict(Xtest)
    print(classification_report(ytest, predictions))


def main():
    clf = pd.read_pickle('data/my_classifier.pkl')
    pickledf = pd.read_pickle('data/my_dataset.pkl')
    dataset = pd.DataFrame.from_dict(pickledf).T

    feature_list = pd.read_pickle('data/features_list.pkl')
    ### load up student's classifier, dataset, and feature_list
    ### Run testing script
    test_classifier(clf, dataset, feature_list)

if __name__ == '__main__':
    main()