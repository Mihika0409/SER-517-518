import pandas as pd
import sklearn.preprocessing as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/vc/Downloads/Trauma_dataset.csv')
df.columns = ['T1',    'ED/Hosp Arrival Date',    'Age in Years',    'Gender',  'Levels',  'ICD-10 E-code',   'Trauma Type',
              'Report of physical abuse',    'Injury Comments', 'Airbag Deployment',   'Patient Position in Vehicle',  'Safet Equipment Issues',
              'Child Restraint', 'MV Speed',    'Fall Height', 'Transport Type',  'Transport Mode',  'Field SBP',   'Field HR',
              'Field Shock Index',   'Field RR',    'Resp Assistance', 'RTS',   'Field GCS',   'Arrived From', 'ED LOS (mins)',
              'Dispositon from  ED', 'ED SBP',  'ED HR',   'ED RR', 'ED GCS',    'Total Vent Days', 'Total Days in ICU',
              'Admission Hosp LOS (days)',   'Total LOS (ED+Admit)',    'Received blood within 4 hrs', 'Severe Brain Injury',
              'Time to 1st OR Visit (mins.)', 'Final Outcome-Dead or Alive',   'Discharge Disposition',   'Injury Severity Score', 'AIS 2005']

#print (df.head())
#The dataframe only contains the Level 1 and Level 2 rows
df = df.loc[df['Levels'].isin(['1', '2'])]

#The dataframe df now contains the rows with Field GCS less than 8
df_GCS = df[df['Field GCS'] <= 8]
print(len(df_GCS))

#The dataframe df_GCS1 contains the level1 rows with GCS less than 8
#The dataframe df_GCS1 contains the level1 rows with GCS less than 8
df_GCS1 = df_GCS.loc[df_GCS['Levels'].isin(['1'])]
df_GCS2 = df_GCS.loc[df_GCS['Levels'].isin(['2'])]

#replace the null values in ED GCS with a default value that is 15
df['ED GCS'] = df['ED GCS'].replace(['*NA', '*ND', '*BL'], value = ['15', '15', '15'])

df[['Field GCS', 'ED GCS']] = df[['Field GCS', 'ED GCS']].apply(pd.to_numeric)

print(df['Field GCS'])
print (df['ED GCS'])

#The df_GCS_diff contains the rows with difference of Field GCS and ED GCS greater than or equal to 2
df_GCS_diff = df[(df['Field GCS'] - df['ED GCS']) >= 2]
print ("The rows length with diff between Field and ED GCS >=2 are: " + str(len(df_GCS_diff)))

#The rows in df after considering the age specific hypotension criteria
df_age = df[(((df['Age in Years'] >= 4) & (df['Age in Years'] <= 6)) & (df['Field SBP'] < 90)) |
            (((df['Age in Years'] >= 7) & (df['Age in Years'] <= 16)) & (df['Field SBP'] < 100)) ]
print ("The rows length with SBP less than normal:" + str(len(df_age)))

#The df_injuryComments contains rows with injury comments criteria
df_injuryComments = df[df['Injury Comments'].str.contains("gun", na = False) &
                        (df['Injury Comments'].str.contains("abdomen", na = False) |
                         df['Injury Comments'].str.contains("chest", na = False) | df['Injury Comments'].str.contains("head", na = False))]
print ("The length of Injury comments dataframe is:" + str(len(df_injuryComments)))

frames = [df_injuryComments, df_GCS, df_GCS_diff, df_age]
result = pd.concat(frames)
print(len(result))

#Considering only the rows with level 1 and level 2 trauma levels
result = result.loc[result['Levels'].isin(['1', '2'])]
lcount = 0

print(len(result))
for i in result['Levels']:
    if i == '1':
        lcount = lcount + 1

#print("The count of level 1's are: " + lcount)

df = pd.read_csv('/Users/vc/Downloads/Trauma_dataset.csv')
df.columns = ['T1',    'ED/Hosp Arrival Date',    'Age in Years',    'Gender',  'Levels',  'ICD-10 E-code',   'Trauma Type',
              'Report of physical abuse',    'Injury Comments', 'Airbag Deployment',   'Patient Position in Vehicle',  'Safet Equipment Issues',
              'Child Restraint', 'MV Speed',    'Fall Height', 'Transport Type',  'Transport Mode',  'Field SBP',   'Field HR',
              'Field Shock Index',   'Field RR',    'Resp Assistance', 'RTS',   'Field GCS',   'Arrived From', 'ED LOS (mins)',
              'Dispositon from  ED', 'ED SBP',  'ED HR',   'ED RR', 'ED GCS',    'Total Vent Days', 'Total Days in ICU',
              'Admission Hosp LOS (days)',   'Total LOS (ED+Admit)',    'Received blood within 4 hrs', 'Severe Brain Injury',
              'Time to 1st OR Visit (mins.)', 'Final Outcome-Dead or Alive',   'Discharge Disposition',   'Injury Severity Score', 'AIS 2005']

df = df[['Age in Years', 'Gender', 'Field SBP', 'Field HR', 'Field Shock Index', 'Field RR', 'RTS', 'Field GCS', 'Levels']]
result = result[['Age in Years', 'Gender', 'Field SBP', 'Field HR', 'Field Shock Index', 'Field RR', 'RTS', 'Field GCS', 'Levels']]

df = df.loc[df['Levels'].isin(['1', '2'])]

# Dropping null rows
list = ['Age in Years', 'Gender', 'Field SBP', 'Field HR', 'Field Shock Index', 'Field RR', 'RTS', 'Field GCS', 'Levels']
for i in list:
    df = df[pd.notnull(df[i])]


for j in list:
    result = result[pd.notnull(result[j])]


df['Levels'] = df['Levels'].replace(['1', '2'], value = [0 , 1])
result['Levels'] = result['Levels'].replace(['1', '2'], value = [0 , 1])

df['Gender'] = df['Gender'].replace(['M', 'F'], value = ['1', '2'])
result['Gender'] = result['Gender'].replace(['M', 'F'], value = ['1', '2'])
#Input variables
X_train = df.drop('Levels', 1)
X_test = result.drop('Levels', 1)

#Target variables
Y_train = df['Levels']
Y_test = result['Levels']
print("Train_x Shape :: ", X_train.shape)
print("Train_y Shape :: ", Y_train.shape)
print("Test_x Shape :: ", X_test.shape)
print("Test_y Shape :: ", Y_test.shape)


def random_forest_classifier(X_train, Y_train):
    clf = RandomForestClassifier()
    clf.fit(X_train, Y_train)
    return clf

trained_model = random_forest_classifier(X_train, Y_train)
print ("Trained model :: ", trained_model)
predictions = trained_model.predict(X_test)

#for i in xrange(0,5):
 #   print ("Actual outcome :: {} and Predicted outcome :: {}".format(list(Y_test)[i], predictions[i]))

print ("Train Accuracy :: ", accuracy_score(Y_train, trained_model.predict(X_train)))
print ("Test Accuracy :: ", accuracy_score(Y_test, predictions))
print ("Confusion matrix ", confusion_matrix(Y_test, predictions))