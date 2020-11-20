import warnings
import pandas as pd
from sys import version_info

# sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler

# PLEASE USE PYTHON 
assert version_info >= (3, 0)
warnings.filterwarnings("ignore", category=RuntimeWarning) 


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features
print('Feature selection by Kbest')

### Load the dictionary containing the dataset
pickle = pd.read_pickle('data/final_project_dataset.pkl')
df = pd.DataFrame.from_dict(pickle).T
payfeatures = ['salary', 'bonus', 'long_term_incentive', 'deferred_income', 
               'deferral_payments', 'loan_advances', 'other', 'expenses',
               'director_fees', 'total_payments']
stockfeatures = ['exercised_stock_options', 'restricted_stock', 
                 'restricted_stock_deferred', 'total_stock_value']
emailfeatures = ['to_messages', 'from_poi_to_this_person', 'from_messages', 
                 'from_this_person_to_poi', 'shared_receipt_with_poi']
df = df.reindex(columns=['poi', *payfeatures, *stockfeatures, *emailfeatures])

### Task 2: Remove outliers
df.drop('TOTAL', inplace=True) # Aggregate
df.drop('THE TRAVEL AGENCY IN THE PARK', inplace=True) # Non-person
df.loc['BELFER ROBERT'] = [
    False,0,0,0,-102500,0,0,0,3285,102500,3285,0,44093,-44093,0,
    *df[emailfeatures].loc['BELFER ROBERT']
    ]
df.loc['BHATNAGAR SANJAY'] = [
    False,0,0,0,0,0,0,0,137864,0,137864,15456290,2604490,-2604490,15456290,
    *df[emailfeatures].loc['BHATNAGAR SANJAY']
    ]
df = df.astype(float).fillna(0).astype(int)

### Task 3: Create new feature(s)
print('New feature created: options_to_stock\n')
df['options_to_stock'] = (df.exercised_stock_options/df.total_stock_value)
df['options_to_stock'].fillna(0, inplace=True)
df.drop(columns='total_stock_value', inplace=True)

### Extract features and labels from dataset for local testing
X = df.copy()
X, y = X.iloc[:,1:], X.iloc[:,0]
sss = StratifiedShuffleSplit(test_size=0.4, random_state=42)
for trainidx, testidx in sss.split(X, y):
    Xtrain, Xtest = X.iloc[trainidx], X.iloc[testidx]
    ytrain, ytest = y.iloc[trainidx], y.iloc[testidx]


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#model.get_params()['estimator__selector'].get_support(indices=True)
robust = MinMaxScaler()
pipeline = Pipeline([
    ('robust_scale', robust),
    ('selector', SelectKBest(f_regression)),
    ('model', RandomForestClassifier())])
search = GridSearchCV(
    estimator = pipeline,
    param_grid = {'selector__k':[2,3,4], 
                  'model__n_estimators': range(70, 91, 5)},
    n_jobs=-1, scoring='f1', cv=10)
clf = search.fit(Xtrain,ytrain)

print('Best params: ', search.best_params_)
print('Best score: ', search.best_score_)
idx = search.best_estimator_['selector'].get_support(indices=True)
predictions = clf.predict(Xtest)
print('*'*60)
print(classification_report(ytest, predictions))
print('*'*60, '\n')
features = Xtrain.iloc[:,idx].columns.values
print('Features used: ', features)

# pipe = make_pipeline(MinMaxScaler(), 
#                       SelectKBest(k=3, score_func=f_regression),
#                       RandomForestClassifier(n_estimators=70))
# clf = pipe.fit(Xtrain, ytrain)
# predictions = clf.predict(Xtest)
# print('*'*60)
# print(classification_report(ytest, predictions))
# print('*'*60, '\n')


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
print('For tuning, please see pipeline.py')


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

pd.to_pickle(clf, 'data/my_classifier.pkl')
pd.to_pickle(df.T.to_dict(), 'data/my_dataset.pkl')
pd.to_pickle(features, 'data/features_list.pkl')
