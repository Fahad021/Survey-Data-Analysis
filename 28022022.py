# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 10:08:33 2022

@author: 54651
"""


# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 22:52:57 2022

@author: 54651
"""


import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# https://towardsdatascience.com/feature-engineering-examples-binning-categorical-features-9f8d582455da

Tech_method_map = {'I am usually the first among my friends to buy a new technology': 'Early',
                   'I often buy a new technology after a few of my friends have purchased it': 'Early',
                   'I typically wait until most of my friends have bought it': 'late',
                   'I typically wait until most of my friends have bought it':'late',
                   'I am generally the last of my friends to buy a new technology': 'late',
                   '\xa0I generally wait until at least half of my friends have bought it': 'late'}

EV_method_map =   {'0':'No',
                   '1':'Yes',
                   '2':'Yes',
                   'More than 2':'Yes'}


gender_method_map = {'Female':'Female',
                     'Male':'Male',
                     'Prefer not to say': 'prefer_not_to_say',
                     'Non-binary': 'others',
                     'Do not identity as Female, Male, Non-Binary, or Transgender':'others'}


race_method_map = { 'Non-Hispanic White':'white',
                   'Prefer not to say': 'prefer_not_to_say',
                   'Hispanic': 'non-white',
                   'Non-Hispanic Asian': 'non-white',
                   'Non-Hispanic Black': 'non-white',
                   'Other':'non-white'}



edu_method_map = {'Four-year college graduate':'college_degree',
                  'Graduate degree':'college_degree',
                  'Some graduate school':'college_degree',
                  '2-year technical/Associates degree':'did_not_finish_college',
                  'Some college' :'did_not_finish_college',
                  'Prefer not to say': 'prefer_not_to_say',
                  'High school graduate or GED': 'high_school_grad_or_GED',
                  'Less than finishing high school': 'did_not_finish_high_school'}

income_method_map = {'$250,000 or more': '100K_and_more',
                     '$150,000 to less than $200,000': '100K_and_more',
                     '$200,000 to less than $250,000': '100K_and_more',
                     '$100,000 to less than $150,000': '100K_and_more',
                     '$80,000 to less than $100,000': 'less_than_100K' ,
                     '$60,000 to less than $80,000': 'less_than_100K',
                     '$40,000to less than $60,000':  'less_than_100K',
                     'Less than $40,000':  'less_than_100K',
                     'Prefer not to say': 'prefer_not_to_say'}

#Install_method_map = { 'Yes, a solar system only':'PV',
#                      'Yes, both a solar system and battery storage system':'PV_BESS'}

Install_method_map = { 'Yes, a solar system only':0,
                      'Yes, both a solar system and battery storage system':1}

ls_dict = [Tech_method_map, EV_method_map,
           gender_method_map,
           race_method_map,
           edu_method_map, income_method_map,
           Install_method_map]

columns = ['Tech', 'Evnum', 'age', 'gender', 'raceeth', 'edu', 'income','Install']

df = pd.read_csv('dte_bess_screener-v3.csv', usecols = columns)

df = df[df.Install != 'Yes, a battery storage system only (I previously had a solar-only system)']

df = df.dropna() #new_df.shape


new_df = pd.DataFrame()

new_df[columns[0]+'_binned'] = df[columns[0]].map(ls_dict[0])
new_df[columns[1]+'_binned'] = df[columns[1]].map(ls_dict[1])
new_df[columns[3]+'_binned'] = df[columns[3]].map(ls_dict[2])
new_df[columns[4]+'_binned'] = df[columns[4]].map(ls_dict[3])
new_df[columns[5]+'_binned'] = df[columns[5]].map(ls_dict[4])
new_df[columns[6]+'_binned'] = df[columns[6]].map(ls_dict[5])
new_df[columns[7]+'_binned'] = df[columns[7]].map(ls_dict[6])
new_df['age_binned'] = df['age'].values

#new_df.to_csv('binned_data_28feb.csv')

new_df.to_csv('binned_data_03march_v2.csv')

X = ['Tech_binned', 'Evnum_binned',  'age_binned', 'gender_binned', 'raceeth_binned',
       'edu_binned', 'income_binned' ]

chi_value =[]
p_value = []
dof_value = []
expected_value = []

for i in X:
    contigency= pd.crosstab(new_df['Install_binned'], new_df[i])
    plt.figure(figsize=(12,8))
    sns_plot = sns.heatmap(contigency, annot=True, cmap="YlGnBu", fmt='g')
    fig = sns_plot.get_figure()
    # Chi-square test of independence.
    c, p, dof, expected = chi2_contingency(contigency)
    chi_value.append(c)
    p_value.append(p)
    dof_value.append(dof)
    expected_value.append(expected)

print(p_value)

all = [X,chi_value, p_value]
main = pd.DataFrame(all)
main = main.T
main.columns = ['factor','test_stat','p_value']
#final = pd.DataFrame(main, columns = ['factor','test_stat','p_value'])

x = main['factor']
y = main['p_value']

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.hlines(y=0.05, xmin=0, xmax=7, linewidth=2, color='r')
plt.ylabel('p_value')
plt.xlabel('independent varible')
plt.xticks(rotation=270)
#rotate x-axis labels by 45 degrees

plt.yticks(rotation=90)
plt.figure(figsize = (12,12))
#plt.savefig('p_val.png')
#figure(figsize=(8, 6), dpi=80)

plt.show()


