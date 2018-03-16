import pandas as pd
import numpy as np
import sklearn.preprocessing as sk
import unicodecsv as unicodecsv
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from xlrd import open_workbook

# Converts the xls file to csv
from xlrd.timemachine import xrange

'''def xls2csv (xls_filename, csv_filename):
    wb = open_workbook(xls_filename)
    sh = wb.sheet_by_index(0)
    fh = open(csv_filename,"wb")
    csv_out = unicodecsv.writer(fh, encoding='utf-8')
    for row_number in xrange (sh.nrows):
        csv_out.writerow(sh.row_values(row_number))
    fh.close()
    return csv_filename
csv_file = xls2csv('Copy of Trauma Data Sample From Jan 2016 to Jan 2017.xlsx','csv.csv')'''

df = pd.read_csv('/Users/vc/Documents/Susmitha Documents/Software Factory/TraumaDataSixYears.csv')
df.columns = ['T1',    'ED/Hosp Arrival Date',    'Age in Years',    'Gender',  'Levels',  'ICD-10 E-code', 'ICD-9 E-code',  'Trauma Type',
              'Report of physical abuse',    'Injury Comments', 'Airbag Deployment',   'Patient Position in Vehicle',  'Safet Equipment Issues',
              'Child Restraint', 'MV Speed',    'Fall Height', 'Transport Type',  'Transport Mode',  'Field SBP',   'Field HR',
              'Field RR',    'Resp Assistance', 'RTS',   'Field GCS',   'Arrived From', 'ED LOS (mins)',
              'Dispositon from  ED', 'ED SBP',  'ED HR',   'ED RR', 'ED GCS',    'Total Vent Days', 'Total Days in ICU',
              'Admission Hosp LOS (days)',   'Total LOS (ED+Admit)',    'Received blood within 4 hrs', 'Severe Brain Injury',
              'Time to 1st OR Visit (mins.)', 'Final Outcome-Dead or Alive',   'Discharge Disposition',   'Injury Severity Score', 'ICD 9 Dx', 'ICD 10 Dx', 'AIS 2005']

#print (df.head())
#The dataframe only contains the Level 1 and Level 2 rows
df = df.loc[df['Levels'].isin(['1', '2'])]
df['Levels'] = df['Levels'].replace(['1', '2'], value = [0 , 1])
levels = ["Levels"]

#Convert the Gender attribute to numeric
df['Gender'] = df['Gender'].replace(['M', 'F'], value = ['0', '1'])
gender = ["gender"]

#replace the null values in Field GCS with a default value that is 15
df['Field GCS'] = df['Field GCS'].replace(['*NA', '*ND', '*BL'], value = ['15', '15', '15'])

#df['Field SBP'] = df['Field SBP'].replace(['*NA', '*ND', '*BL'], value = ['15', '15', '15'])
df['Field SBP'] = np.where((int(df['Age in Years'])<= 5 &
                            (df['Field SBP'] == '*NA' | df['Field SBP'] == '*ND' |
                             df['Field SBP'] == '*BL')),'90',df['Field SBP'])

df['Field SBP'] = np.where((int(df['Age in Years'])> 5 & int(df['Age in Years'])<= 13
                            (df['Field SBP'] == '*NA' | df['Field SBP'] == '*ND' |
                             df['Field SBP'] == '*BL')),'105',df['Field SBP'])

df['Field SBP'] = np.where((int(df['Age in Years'])> 13 & int(df['Age in Years'])<= 19
                            (df['Field SBP'] == '*NA' | df['Field SBP'] == '*ND' |
                             df['Field SBP'] == '*BL')),'117',df['Field SBP'])

#df['Field HR'] = df['Field HR'].replace(['*NA', '*ND', '*BL'], value = ['15', '15', '15'])
df['Field HR'] = np.where((int(df['Age in Years'])<= 5 &
                            (df['Field HR'] == '*NA' | df['Field HR'] == '*ND' |
                             df['Field HR'] == '*BL')),'100',df['Field HR'])

df['Field HR'] = np.where((int(df['Age in Years'])> 5 & int(df['Age in Years'])<= 13
                            (df['Field HR'] == '*NA' | df['Field HR'] == '*ND' |
                             df['Field HR'] == '*BL')),'90',df['Field HR'])

df['Field HR'] = np.where((int(df['Age in Years'])> 13 & int(df['Age in Years'])<= 19
                            (df['Field HR'] == '*NA' | df['Field HR'] == '*ND' |
                             df['Field HR'] == '*BL')),'80',df['Field HR'])

df['Field RR'] = df['Field RR'].replace(['*NA', '*ND', '*BL'], value = ['20', '20', '20'])

#replace the null values in ED GCS with a default value that is 15
df['ED GCS'] = df['ED GCS'].replace(['*NA', '*ND', '*BL'], value = ['15', '15', '15'])

#the null values in fall height are considered as 0
df['Fall Height'] = df['Fall Height'].replace(['*NA', '*ND', '*BL'], value = ['0', '0', '0'])

# creates a new dataframe with just comments
'''dfn = pd.read_csv('csv.csv')
df1 = dfn.ix[:, 19:20].dropna(axis=0, how='any')
df1 = df1.rename(columns={'Unnamed: 19': 'Comments'})

for index, row in df1.iterrows():
    dataC = df1['Comments']
    print dataC'''

df['MV Speed'] = df['MV Speed'].replace(['*NA', '*ND', '*BL'], value = ['0', '0', '0'])

df['Injury Severity Score'] = df['Injury Severity Score'].replace(['*NA', '*ND', '*BL'], value = ['0', '0', '0'])

fieldgcs = ["Field GCS"]
df.to_csv('output1.csv', columns = fieldgcs)