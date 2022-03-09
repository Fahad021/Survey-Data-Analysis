# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 10:55:20 2022

@author: 54651
"""

import pandas as pd
#from numpy import asarray
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('binned_data_28feb.csv')

# define data
X = df[['Tech_binned', 'Evnum_binned', 'gender_binned', 'raceeth_binned',
       'edu_binned', 'income_binned', 'age_binned']]
Y = df[['Install_binned']]

Install_method_map = { 'PV':0,
                      'PV_BESS':1}

data_X = pd.get_dummies(X, columns = X.columns)
Y = Y['Install_binned'].map(Install_method_map)

final = pd.concat([data_X, Y], axis =1)

final.to_csv('encoded_data_20220303.csv')
df = pd.read_csv('encoded_data_20220303.csv', index_col = [0])
X = df.iloc[:,0:(df.shape[1]-1)].values
Y = df.iloc[:,df.shape[1]-1].values

clf = LogisticRegression().fit(X,Y)

