import pandas as pd
from sys import version_info

# PLEASE USE PYTHON 
assert version_info >= (3, 0)

# Import the data set, clean up columns and organize
pickle = pd.read_pickle('data/final_project_dataset.pkl')
df = pd.DataFrame.from_dict(pickle).T
payfeatures = ['salary', 'bonus', 'long_term_incentive', 'deferred_income', 
               'deferral_payments', 'loan_advances', 'other', 'expenses',
               'director_fees', 'total_payments']
stockfeatures = ['exercised_stock_options', 'restricted_stock', 
                 'restricted_stock_deferred', 'total_stock_value']
emailfeatures = ['to_messages', 'from_poi_to_this_person', 'from_messages', 
                 'from_this_person_to_poi', 'shared_receipt_with_poi']

# Reindex columns per PDF
df = df.reindex(columns=['poi', *payfeatures, *stockfeatures, *emailfeatures])