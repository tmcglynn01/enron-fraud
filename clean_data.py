from load_data import df, emailfeatures
from sklearn.model_selection import StratifiedShuffleSplit



### OUTLIER REMOVAL
# Drop aggregate vals, reindex improperly entered vals, fill NA
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

# Fill NAs
df = df.astype(float).fillna(0).astype(int)

# How many zero values?
#print((df < 0).astype(int).sum(axis=0).sort_values())

# Subtract resticted stock and deferred income from their aggregates
# Probably will be easier than dealing with them otherwise, nor do they seem
# to correlate much
#df['total_payments'] = df.total_payments - df.deferred_income
#df.drop(columns='deferred_income', inplace=True)
#df['total_stock_value'] = df.total_stock_value - df.restricted_stock_deferred
#df.drop(columns='restricted_stock_deferred', inplace=True)

# Drop noisy/less correlated data
dropcols = ['to_messages', 'from_messages']
df.drop(columns=dropcols, inplace=True)

# Try taking abs for negative deferred_income
df['deferred_income'] = abs(df.deferred_income)
df['deferred_income'] = abs(df.deferred_income)

# Add some features
#suspay = ['other', 'expenses', 'bonus']
#df['suspay'] = df[suspay].sum(1)
# Apply base 10 log to better normalize income data
#for x in df.iloc[:, 1:]:
#    df[x] = np.log10(abs(1+df[x]))

X = df.copy()
X, y = X.iloc[:,1:], X.iloc[:,0]

# Split the data set using SSS
sss = StratifiedShuffleSplit(test_size=0.4, random_state=42)
for trainidx, testidx in sss.split(X, y):
    Xtrain, Xtest = X.iloc[trainidx], X.iloc[testidx]
    ytrain, ytest = y.iloc[trainidx], y.iloc[testidx]