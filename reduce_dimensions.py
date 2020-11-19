from clean_data import df

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Split the data set using SSS
X, y = df.iloc[:,1:], df.iloc[:,0]
sss = StratifiedShuffleSplit(train_size=0.66, random_state=42)
for trainidx, testidx in sss.split(X, y):
    Xtrain, Xtest = X.iloc[trainidx], X.iloc[testidx]
    ytrain, ytest = y.iloc[trainidx], y.iloc[testidx]
    
# Apply scaling to the data sets
sc = StandardScaler()
Xtrain_std = sc.fit_transform(Xtrain)
Xtest_std = sc.transform(Xtest)

# Principle component analysis
pca = PCA()
Xtrain_pca = pca.fit_transform(Xtrain_std)
plt.bar(range(1, 19), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(1, 19), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.savefig('plots/explained_variance.png', dpi=300)
plt.show()


pca = PCA(n_components=2)
Xtrain_pca = pca.fit_transform(Xtrain_std)
Xtest_pca = pca.transform(Xtest_std)
plt.scatter(Xtrain_pca[:, 0], Xtrain_pca[:, 1])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()


