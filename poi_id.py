import pandas as pd
from sys import version_info

# PLEASE USE PYTHON 
assert version_info >= (3, 0)


### LOAD DATA
# Import the data set, clean up columns and organize
pickle = pd.read_pickle('data/final_project_dataset.pkl')
df = pd.DataFrame.from_dict(pickle).T

poicol = ['poi']
payfeatures = ['salary', 'bonus', 'long_term_incentive', 'deferred_income', 
               'deferral_payments', 'loan_advances', 'other', 'expenses',
               'director_fees', 'total_payments']
stockfeatures = ['exercised_stock_options', 'restricted_stock', 
                 'restricted_stock_deferred', 'total_stock_value']
emailfeatures = ['to_messages', 'from_poi_to_this_person', 'from_messages', 
                 'from_this_person_to_poi', 'shared_receipt_with_poi']

# Reindex columns per PDF
df = df.reindex(columns=['poi', *payfeatures, *stockfeatures, *emailfeatures])


### OUTLIER REMOVAL
# Drop aggregate vals, reindex improperly entered vals, fill NA
df.drop('TOTAL', inplace=True) # Aggregate
df.drop('THE TRAVEL AGENCY IN THE PARK', inplace=True) # Non-person
df.drop('LOCKHART EUGENE E', inplace=True) # outlier (all null vals)
df.loc['BELFER ROBERT'] = [
    False,0,0,0,-102500,0,0,0,3285,102500,3285,0,44093,-44093,0,
    *df[emailfeatures].loc['BELFER ROBERT']
    ]
df.loc['BHATNAGAR SANJAY'] = [
    False,0,0,0,0,0,0,0,137864,0,137864,15456290,2604490,-2604490,15456290,
    *df[emailfeatures].loc['BHATNAGAR SANJAY']
    ]
df = df.astype(float).fillna(0).astype(int)
dropcols = ['to_messages', 'from_messages', 'restricted_stock_deferred']
df.drop(columns=dropcols, inplace=True) # noisy data
stockfeatures.remove(dropcols[-1])
[emailfeatures.remove(x) for x in dropcols[:2]]


# Apply base 10 log to better normalize income data, sqrt for expenses
for x in [*payfeatures, *stockfeatures, *emailfeatures]:
    df[x] = np.log10(1+abs(df[x].fillna(0)))

# Try a variety of classifiers
pipe = make_pipeline(
    Imputer(axis=0, copy=True, missing_values='NaN',
            strategy='median', verbose=0),
    PCA(copy=True, n_components=12, whiten=True),
    LogisticRegression(C=1, dual=False, fit_intercept=True,
                       intercept_scaling=0.6, max_iter=100,
                       multi_class='ovr', n_jobs=-1, penalty='l2',
                       random_state=42, solver='liblinear',
                       tol=0.0001, verbose=0, warm_start=False))

# Task 4: Try a varity of classifiers
# Task 5: Tune your classifier to achieve better than .3 precision and recall

# Based on my assessment observed in ./output/result_all.txt,
# the best five models are chosen to be tested by tester.py.
# Please look at ./output/result_final.txt for the test result of these five.
# I chose the fourth from the top as the my final model because it has the
# best f1 score.

#pipe = make_pipeline(
#          Imputer(axis=0, copy=True, missing_values='NaN',
#                  strategy='median', verbose=0),
#          PCA(copy=True, n_components=12, whiten=True),
#          LogisticRegression(C=1, class_weight='balanced', dual=False,
#                             fit_intercept=True, intercept_scaling=0.6,
#                             max_iter=100, multi_class='ovr', n_jobs=-1,
#                             penalty='l2', random_state=None,
#                             solver='liblinear', tol=0.0001, verbose=0,
#                             warm_start=False))
#pipe = make_pipeline(
#          Imputer(axis=0, copy=True, missing_values='NaN',
#                  strategy='median', verbose=0),
#          SVC(C=1, cache_size=200, class_weight='balanced', coef0=0.0,
#              decision_function_shape='ovo', degree=3, gamma='auto',
#              kernel='linear', max_iter=-1, probability=False,
#              random_state=20160308, shrinking=False, tol=0.001,
#              verbose=False))
#pipe = make_pipeline(
#          Imputer(axis=0, copy=True, missing_values='NaN',
#                  strategy='median', verbose=0),
#          PCA(copy=True, n_components=18, whiten=True),
#          SVC(C=1, cache_size=200, class_weight='balanced', coef0=0.0,
#              decision_function_shape='ovo', degree=3, gamma='auto',
#              kernel='linear', max_iter=-1, probability=False,
#              random_state=20160308, shrinking=False, tol=0.001,
#              verbose=False))
pipe = make_pipeline(
          Imputer(axis=0, copy=True, missing_values='NaN',
                  strategy='median', verbose=0),
          ExtraTreesClassifier(bootstrap=False, class_weight='balanced',
                               criterion='gini', max_depth=None,
                               max_features='sqrt', max_leaf_nodes=None,
                               min_samples_leaf=3, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=30,
                               n_jobs=-1, oob_score=False,
                               random_state=20160308, verbose=0,
                               warm_start=False))
#pipe = make_pipeline(
#          Imputer(axis=0, copy=True, missing_values='NaN',
#                  strategy='median', verbose=0),
#          SelectFpr(alpha=0.05, score_func=f_classif),
#          ExtraTreesClassifier(bootstrap=False, class_weight='balanced',
#                               criterion='gini', max_depth=None,
#                               max_features='sqrt', max_leaf_nodes=None,
#                               min_samples_leaf=3, min_samples_split=2,
#                               min_weight_fraction_leaf=0.0, n_estimators=30,
#                               n_jobs=-1, oob_score=False,
#                               random_state=20160308, verbose=0,
#                               warm_start=False))

# Task 6: Dump your classifier, dataset, and features_list
dump_classifier_and_data(pipe, df.to_dict(orient='index'), ['poi'] + F_ALL_NEW)