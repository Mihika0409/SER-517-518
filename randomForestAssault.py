import pandas as pd
import sklearn.preprocessing as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/vc/Documents/Susmitha Documents/Software Factory/TraumaDataSixYears.csv', header = None, error_bad_lines=False)
df.columns = ['T1#', 'ED/Hosp Arrival Date', 'Age in Years', 'Gender', 'Levels', 'ICD-10 E-code (after 1/2016)', 'ICD-9 E-code (before 2016)',
               'Trauma Type', 'Report of abuse? (started 2014)', 'Injury Comments', 'Airbag Deployment', 'Patient Position in Vehicle',
               'Safety Equipment Issues', 'Child Restraint', 'MV Speed', 'Fall Height', 'Transport Type', 'Transport Mode', 'Field SBP',
               'Field HR', 'Field RR', 'Resp Assistance', 'RTS', 'Field GCS', 'Arrived From', 'ED LOS (mins)', 'ED Disposition', 'ED SBP',
               'ED HR','ED RR', 'ED GCS', 'Total Vent Days', 'Total Days in ICU', 'Admission Hosp LOS (days)', 'Total LOS (ED+Admit)',
               'Early transfusion? (started 2016)', 'Severe TBI? (started 2016)', 'Time to 1st OR Visit (mins.)', 'Final Outcome-Dead or Alive', 'Hospital Disposition',
               'Injury Severity Score', 'ICD 9 Dx (before 2016)', 'ICD 10 Dx (after 1/2016)', 'AIS 2005']

#print(df.head())

df = df.loc[df['Levels'].isin(['1', '2'])]

# Taking the rows with injury comments which contain the ACS gun shot critertia
df_assault_icd9 = df[df['ICD-9 E-code (before 2016)'].str.contains("ASSAULT", na=False)]
print ("Number of rows:")
print (len(df_assault_icd9))

# Taking the rows with injury comments which contain the ACS gun shot critertia
df_assault_icd10 = df[df['ICD-10 E-code (after 1/2016)'].str.contains("ASSAULT", na=False)]
print("Number of rows:")
print(len(df_assault_icd10))

print("The dataframe is: ")

#Both ICD 9 and 10 codes data are combined below
frames = [df_assault_icd9, df_assault_icd10]
result = pd.concat(frames)
print ("The number of rows in the entire dataframe:")
print (len(result))
print (result)

# Considering columns that we require
result = result[['Levels', 'Age in Years', 'Gender', 'Field SBP', 'Field HR', 'Field RR', 'RTS', 'Field GCS']]

# Only the level 1 and 2 trauma levels are considered
result = result.loc[result['Levels'].isin(['1', '2'])]

# Male and female are replaced with int values 1 and 2
result['Gender'] = result['Gender'].replace(['M', 'F'], value=['1', '2'])

#Replace the missing values with mean values
result['Field SBP']=result['Field SBP'].replace(['*NA','*ND','*BL',''],'119')
result['Field HR']=result['Field HR'].replace(['*NA','*ND','*BL',''],'110')
result['Field RR']=result['Field RR'].replace(['*NA','*ND','*BL',''],'21')
result['RTS'] = result['RTS'].replace(['*NA','*ND','*BL',''],'7.65')
result['Field GCS']=result['Field GCS'].replace(['*NA','*ND','*BL',''],'14.54')

#Target variable Y
Y = result['Levels']
#print (Y)

#Input variable
X = result.drop('Levels', 1)
print (X)
