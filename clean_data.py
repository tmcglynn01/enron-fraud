from load_data import df, emailfeatures, stockfeatures
import numpy as np


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

# Fill NAs
df = df.astype(float).fillna(0).astype(int)

# Drop noisy/less correlated data
dropcols = ['to_messages', 'from_messages', 'restricted_stock_deferred']
df.drop(columns=dropcols, inplace=True)
stockfeatures.remove(dropcols[-1])
[emailfeatures.remove(x) for x in dropcols[:2]]

# Add some features
df['total_comp'] = df.total_payments + df.total_stock_value
suspay = ['other', 'expenses', 'bonus', 'expenses', 'salary']
df['suspay'] = df[suspay].sum(1)

# Apply base 10 log to better normalize income data, sqrt for expenses
for x in df.iloc[:, 1:]:
    df[x] = np.log10(1+abs(df[x].fillna(0)))