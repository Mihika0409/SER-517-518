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