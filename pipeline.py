from clean_data import Xtrain, Xtest, ytrain, ytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# std = StandardScaler()
# pca = PCA()
# lr = LogisticRegression()
# pipeline = Pipeline(steps=[
#     ('std_scale', std),
#     ('pca', pca), 
#     ('logistic_reg', lr)])
# n_components = list(range(1,Xtrain.shape[1]+1,1))
# C = [0.001, 0.01, 0.1, 1]
# penalty = ['l2']
# parameters = dict(
#     pca__n_components=n_components,
#     logistic_reg__C=C,
#     logistic_reg__penalty=penalty)

# clf = GridSearchCV(pipeline, parameters, scoring='f1', cv=5)
# clf = clf.fit(Xtrain, ytrain)
# print(clf.best_estimator_.get_params()['logistic_reg'])
# predictions = clf.predict(Xtest)
# print(classification_report(ytest, predictions))


robust = MinMaxScaler()
pipeline = Pipeline([
    ('robust_scale', robust),
    ('selector', SelectKBest(f_regression)),
    ('model', RandomForestClassifier())])

search = GridSearchCV(
    estimator = pipeline,
    param_grid = {'selector__k':[2,3,4], 
                  'model__n_estimators': [60, 100, 1000]},
    n_jobs=-1,
    scoring='f1',
    cv=10, verbose=3)
model = search.fit(Xtrain,ytrain)
print(search.best_params_)
print(search.best_score_)
predictions = model.predict(Xtest)
print(classification_report(ytest, predictions))

