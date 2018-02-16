import pandas as pd
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("result.csv")
del df['Unnamed: 0']

df = df[
    ['Levels', 'T1#', 'Age in Years', 'Gender', 'Trauma Type', 'Report of physical abuse?', 'MV Speed', 'Fall Height',
     'Transport Mode', 'Field SBP', 'Field HR', 'Field Shock Index', 'Field RR', 'Resp Assistance', 'RTS', 'Field GCS',
     'Arrived From']]

dummy = pd.get_dummies(df['Arrived From'])
df = pd.concat([df, dummy], axis=1)
del df['Arrived From']

dummy1 = pd.get_dummies(df['Transport Mode'])
df = pd.concat([df, dummy1], axis=1)
del df['Transport Mode']

dummy2 = pd.get_dummies(df['Report of physical abuse?'])
df = pd.concat([df, dummy2], axis=1)
del df['Report of physical abuse?']

dummy3 = pd.get_dummies(df['Trauma Type'])
df = pd.concat([df, dummy3], axis=1)
del df['Trauma Type']

dummy4 = pd.get_dummies(df['Gender'])
df = pd.concat([df, dummy4], axis=1)
del df['Gender']

df[['Fall Height', 'MV Speed']] = df[['Fall Height', 'MV Speed']].replace(['*NA', '*ND', '*BL'], [0, 0, 0])
df[['Levels']] = df[['Levels']].replace(['N'], [4])
df[['Field Shock Index']] = df[['Field Shock Index']].replace(['#VALUE!'], [0])

X = df.values[:, 1:len(df)]
Y = df.values[:, 0:1]
Y = Y.astype('int')

param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "presort": [True, False],
              "criterion": ["gini", "entropy"]}

y = Y.reshape(-1, )

clf = DecisionTreeClassifier()
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search)
random_search.fit(X, y)
print random_search.best_params_
print random_search.best_score_
