import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("fall_analysing_data.csv")

df = pd.read_csv('refined_data_todd.csv', header = None, error_bad_lines=False)
df.columns = ['T1',	'ED/Hosp Arrival Date',	'Age in Years',	'Gender',	'Levels',	'ICD-10 E-code',	'Trauma Type',	'Report of physical abuse',	'Injury Comments',	'Airbag Deployment',	'Patient Position in Vehicle',
              'Safet Equipment Issues',	'Child Restraint',	'MV Speed',	'Fall Height',	'Transport Type',	'Transport Mode',	'Field SBP',	'Field HR',	'Field Shock Index',	'Field RR',	'Resp Assistance',
              'RTS',	'Field GCS',	'Arrived From',	'ED LOS (mins)',	'Dispositon from  ED',	'ED SBP',	'ED HR',	'ED RR',
              'ED GCS',	'Total Vent Days',	'Total Days in ICU',	'Admission Hosp LOS (days)',	'Total LOS (ED+Admit)',	'Received blood within 4 hrs',	'Severe Brain Injury',	'Time to 1st OR Visit (mins.)',

              'Final Outcome-Dead or Alive',	'Discharge Disposition',	'Injury Severity Score',	'AIS 2005']
#print df.head()

df.drop(['T1','ED/Hosp Arrival Date','ICD-10 E-code','Injury Comments', 'Patient Position in Vehicle','Safet Equipment Issues','Child Restraint','Transport Type',	'Transport Mode','Resp Assistance','Arrived From','Dispositon from  ED','Received blood within 4 hrs','Severe Brain Injury',	'Time to 1st OR Visit (mins.)','Discharge Disposition'], axis=1, inplace = True)


dummy_row1 = pd.get_dummies(df['Arrived From'])
df = pd.concat([df, dummy_row1], axis=1)


dummy_row2 = pd.get_dummies(df['Transport Mode'])
df = pd.concat([df, dummy_row2], axis=1)

dummy_row3 = pd.get_dummies(df['Report of physical abuse?'])
df = pd.concat([df, dummy_row3], axis=1)


dummy_row4 = pd.get_dummies(df['Trauma Type'])
df = pd.concat([df, dummy_row4], axis=1)
del df['Trauma Type']

dummy_row5 = pd.get_dummies(df['Gender'])
df = pd.concat([df, dummy_row5], axis=1)


dummy_row6 = pd.get_dummies(df['Child Restraint'])
df = pd.concat([df, dummy_row6], axis=1)

dummy_row7 = pd.get_dummies(df['Final Outcome-Dead or Alive'])
df = pd.concat([df, dummy_row7], axis=1)



df.drop(['Gender','Arrived From','Transport Mode','Report of Physical Abuse?','Child Restraint','Final Outcome-Dead or Alive'])


X = df.values[:, 1:len(df)]
Y = df.values[:, 0:1]
Y = Y.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=50)

Classifier_Criterion = DecisionTreeClassifier(criterion="gini", random_state=50)
Classifier_Criterion.fit(X_train, y_train)

y_pred = Classifier_Criterion.predict(X_test)

print "Accuracy is ", accuracy_score(y_test, y_pred) * 100