# OPtimization of the decision tree after the first implementation

import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
df = pd.read_csv('/Users/gowtham/Downloads/newdata_trauma.csv', header = None, error_bad_lines=False)

df.columns = ['T1',    'ED/Hosp Arrival Date',    'Age in Years',    'Gender',  'Levels',  'ICD-10 E-code',   'Trauma Type', 'Report of physical abuse',    'Injury Comments', 'Airbag Deployment',   'Patient Position in Vehicle',
              'Safet Equipment Issues',    'Child Restraint', 'MV Speed',    'Fall Height', 'Transport Type',  'Transport Mode',  'Field SBP',   'Field HR',    'Field Shock Index',   'Field RR',    'Resp Assistance',
              'RTS',   'Field GCS',   'Arrived From',    'ED LOS (mins)',   'Dispositon from  ED', 'ED SBP',  'ED HR',   'ED RR',
              'ED GCS',    'Total Vent Days', 'Total Days in ICU',   'Admission Hosp LOS (days)',   'Total LOS (ED+Admit)',    'Received blood within 4 hrs', 'Severe Brain Injury', 'Time to 1st OR Visit (mins.)',
              'Final Outcome-Dead or Alive',   'Discharge Disposition',   'Injury Severity Score',   'AIS 2005']

#print df.head()

df = df[[ 'Age in Years', 'Gender','Field SBP', 'Field HR', 'Field Shock Index', 'Field RR', 'RTS', 'Field GCS','Levels']]
#print df.head()
#Dropping all the rows with null values
features = [ 'Age in Years', 'Gender','Field SBP', 'Field HR', 'Field Shock Index', 'Field RR', 'RTS', 'Field GCS','Levels']
for x in features:
    df = df[pd.notnull(df[x])]
df = df.loc[df['Levels'].isin(['1', '2'])]

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'].values)

print df.head()



X= df.iloc[:,:-1].values
y= df['Levels'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

dt = tree.DecisionTreeClassifier()
dt = dt.fit(X_train,y_train)

#y_train_pred = dt.predict(X_train)
#y_test_pred = dt.predict(X_test)

#print('MSE train: %.3f, test: %.3f' % ( mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))


#decision trees with the gini index
#clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)
y_pred_train = clf_gini.predict(X_train)

print y_pred

print "test Accuracy is ", accuracy_score(y_test,y_pred)*100
print "train Accuracy is ", accuracy_score(y_train,y_pred_train)*100

print "testing error is ", (1-accuracy_score(y_test,y_pred))*100
print "training error is ", (1-accuracy_score(y_train,y_pred_train))*100

# decision tree with the information gain
#clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=3, min_samples_leaf=5)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100)
clf_entropy.fit(X_train, y_train)

y_pred_en = clf_entropy.predict(X_test)
y_pred_train_en = clf_gini.predict(X_train)

print y_pred_en

#print "Accuracy is ", accuracy_score(y_test,y_pred_en)*100

print "test Accuracy is ", accuracy_score(y_test,y_pred_en)*100
print "train Accuracy is ", accuracy_score(y_train,y_pred_train_en)*100

print "testing error is ", (1-accuracy_score(y_test,y_pred_en))*100
print "training error is ", (1-accuracy_score(y_train,y_pred_train_en))*100
