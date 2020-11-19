#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:11:00 2020

@author: trevor
"""

import pandas as pd
import numpy as np
from sys import version_info

# PLEASE USE PYTHON 
assert version_info >= (3, 0)

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