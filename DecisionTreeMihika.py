import pandas as pd
from scipy.stats import randint as sp_randint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("FullData.csv")

df.columns = ['T1',    'ED/Hosp Arrival Date',    'Age in Years',    'Gender',  'Levels',  'ICD-10 E-code','ICD-9 E-code',  'Trauma Type', 'Report of physical abuse',    'Injury Comments', 'Airbag Deployment',   'Patient Position in Vehicle',
              'Safet Equipment Issues',    'Child Restraint', 'MV Speed',    'Fall Height', 'Transport Type',  'Transport Mode',  'Field SBP',   'Field HR', 'Field RR',    'Resp Assistance',
              'RTS',   'Field GCS',   'Arrived From',    'ED LOS (mins)',   'Dispositon from  ED', 'ED SBP',  'ED HR',   'ED RR',
              'ED GCS',    'Total Vent Days', 'Total Days in ICU',   'Admission Hosp LOS (days)',   'Total LOS (ED+Admit)',    'Received blood within 4 hrs', 'Severe Brain Injury', 'Time to 1st OR Visit (mins.)',
              'Final Outcome-Dead or Alive',   'Discharge Disposition',   'Injury Severity Score', 'ICD-9','ICD-10', 'AIS 2005']

df = df[['Levels', 'T1', 'Age in Years', 'Gender', 'Trauma Type',
         'Report of physical abuse', 'Fall Height', 'Transport Mode', 'Field SBP', 'Field HR', 'Field RR', 'RTS',
         'Field GCS', 'Arrived From', 'ED LOS (mins)', 'ED SBP', 'ED HR', 'ED RR', 'ED GCS', 'Total Days in ICU',
         'Admission Hosp LOS (days)', 'Total LOS (ED+Admit)', 'Received blood within 4 hrs', 'Severe Brain Injury',
         'Final Outcome-Dead or Alive', 'Injury Severity Score']]

columns = ['Levels', 'T1', 'Age in Years', 'Gender', 'Trauma Type',
           'Report of physical abuse', 'Fall Height', 'Transport Mode', 'Field SBP', 'Field HR', 'Field RR', 'RTS',
           'Field GCS', 'Arrived From', 'ED LOS (mins)', 'ED SBP', 'ED HR', 'ED RR', 'ED GCS', 'Total Days in ICU',
           'Admission Hosp LOS (days)', 'Total LOS (ED+Admit)', 'Received blood within 4 hrs', 'Severe Brain Injury',
           'Final Outcome-Dead or Alive', 'Injury Severity Score']

for x in columns:
    df = df[pd.notnull(df[x])]

df = df.loc[df['Levels'].isin(['1','2'])]
df[['Levels']] = df[['Levels']].replace([2], [0])
df[['Report of physical abuse']] = df[['Report of physical abuse']].replace(['*BL'], ['N'])
df['Report of physical abuse'] = df['Report of physical abuse'].replace(['N', 'Y'], value=['0', '1'])
df['Gender'] = df['Gender'].replace(['M', 'F'], value=['1', '2'])
df['Final Outcome-Dead or Alive'] = df['Final Outcome-Dead or Alive'].replace(['L', 'D'], value=['1', '0'])
df['Fall Height'] = df['Fall Height'].replace(['*NA', '*ND', '*BL'], value=['5', '5', '5'])
df['Field SBP'] = df['Field SBP'].replace(['*NA', '*ND', '*BL'], value=['76', '76', '76'])
df['Field HR'] = df['Field HR'].replace(['*NA', '*ND', '*BL'], value=['83', '83', '83'])
df['Field RR'] = df['Field RR'].replace(['*NA', '*ND', '*BL'], value=['17', '17', '17'])
df['ED SBP'] = df['ED SBP'].replace(['*NA', '*ND', '*BL'], value=['76', '76', '76'])
df['ED HR'] = df['ED HR'].replace(['*NA', '*ND', '*BL'], value=['83', '83', '83'])
df['ED RR'] = df['ED RR'].replace(['*NA', '*ND', '*BL'], value=['17', '17', '17'])
df['RTS'] = df['RTS'].replace(['*NA', '*ND', '*BL'], value=['3.39613496933', '3.39613496933', '3.39613496933'])
df['Field GCS'] = df['Field GCS'].replace(['*NA', '*ND', '*BL'], value=['9', '9', '9'])
df['ED GCS'] = df['ED GCS'].replace(['*NA', '*ND', '*BL'], value=['9', '9', '9'])
df['ED LOS (mins)'] = df['ED LOS (mins)'].replace(['*NA', '*ND', '*BL'], value=['0', '0', '0'])
df['Admission Hosp LOS (days)'] = df['Admission Hosp LOS (days)'].replace(['*NA', '*ND', '*BL'], value=['0', '0', '0'])
df['Received blood within 4 hrs'] = df['Received blood within 4 hrs'].replace(['*NA', '*ND', '*BL'],
                                                                              value=['N', 'N', 'N'])
df['Severe Brain Injury'] = df['Severe Brain Injury'].replace(['*NA', '*ND', '*BL'], value=['N', 'N', 'N'])
df['Received blood within 4 hrs'] = df['Received blood within 4 hrs'].replace(['N', 'Y'], value=['0', '1'])
df['Severe Brain Injury'] = df['Severe Brain Injury'].replace(['N', 'Y'], value=['0', '1'])
df['Total Days in ICU'] = df['Total Days in ICU'].replace(['*NA', '*ND', '*BL'], value=['0', '0', '0'])
df['Injury Severity Score'] = df['Injury Severity Score'].replace(['*NA', '*ND', '*BL'], value=['5', '5', '5'])
df['T1'] = df['Injury Severity Score'].replace(['*BL'], value=['6160'])

dummy = pd.get_dummies(df['Arrived From'])
df = pd.concat([df, dummy], axis=1)
del df['Arrived From']

dummy1 = pd.get_dummies(df['Transport Mode'])
df = pd.concat([df, dummy1], axis=1)
del df['Transport Mode']

dummy3 = pd.get_dummies(df['Trauma Type'])
df = pd.concat([df, dummy3], axis=1)
del df['Trauma Type']

df.to_csv("sample.csv")

X = df.values[:, 1:len(df)]
Y = df.values[:, 0:1]

param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "presort": [True, False],
              "criterion": ["gini", "entropy"]}

clf = DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)

print "Accuracy is ", accuracy_score(y_test, y_pred) * 100

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
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, return_train_score=True, refit = True, cv = 10)
random_search.fit(X, y)


print "Random search predict: "

print random_search.predict(X)


print "Random search predict probabilities: "

print random_search.predict_proba(X)


print "Random search log probabilities: "

print random_search.predict_log_proba(X)


print "Random search score: "

print random_search.score(X, y)

print "Best parameters found with the randomized search: "
print random_search.best_params_


print "Accuracy of the model: "
print random_search.best_score_


print "Estimators used in this model: "

print random_search.best_estimator_
