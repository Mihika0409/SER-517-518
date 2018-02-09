import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)

print "Accuracy is ", accuracy_score(y_test, y_pred) * 100
