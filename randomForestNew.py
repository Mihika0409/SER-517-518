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

#Training the random forest classifier with the scikit learn
def random_forest_classifier(features, target):
    clf = RandomForestClassifier()
    clf.fit(features, target)
    from sklearn.cross_validation import cross_val_score
    cross_val = cross_val_score(clf, features, target, cv = 10)
    return clf, cross_val

def feature_importances(clf, data):
    feats = {}  # a dict to hold feature_name: feature_importance
    for feature, importance in zip(data.columns, clf.feature_importances_):
        feats[feature] = importance  # add the name/value pair

    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
    importances.sort_values(by='Gini-importance').plot(kind='bar', rot=45)

    plt.show()

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

print(df.shape)
df = df[['Age in Years', 'Gender', 'Field SBP', 'Field HR', 'Field RR', 'RTS', 'Field GCS', 'Levels']]
cols = df.columns

for col in cols:
    df = df[pd.notnull(df[col])]


#replace the char values of gender with int
df['Gender'] = df['Gender'].replace(['M', 'F'], value = ['1', '0'])

#Data frame to replace the missing values with mean values
#fields = ['Age in Years', 'Gender', 'Field SBP', 'Field HR', 'Field RR', 'RTS', 'Field GCS']

#Replace the missing values with mean values
df['Field SBP']=df['Field SBP'].replace(['*NA','*ND','*BL',''],'119')
df['Field HR']=df['Field HR'].replace(['*NA','*ND','*BL',''],'110')
df['Field RR']=df['Field RR'].replace(['*NA','*ND','*BL',''],'21')
df['RTS'] = df['RTS'].replace(['*NA','*ND','*BL',''],'7.65')
df['Field GCS']=df['Field GCS'].replace(['*NA','*ND','*BL',''],'14.54')

#The missing values imputed into the dataframe based on gender
df['RTS'] = np.where(((df['RTS'] == '*NA') | (df['RTS'] == '*ND') | (df['RTS'] == '*BL') | (df['RTS'] == '')) & (df['Gender'] == 1),'6.15',df['RTS'])
df['RTS'] = np.where(((df['RTS'] == '*NA') | (df['RTS'] == '*ND') | (df['RTS'] == '*BL') | (df['RTS'] == '')) & (df['Gender'] == 0),'7.75',df['RTS'])
df['Field GCS'] = np.where(((df['Field GCS'] == '*NA') | (df['Field GCS'] == '*ND') | (df['Field GCS'] == '*BL') | (df['Field GCS'] == '')) & (df['Gender'] == 1),'10.24',df['Field GCS'])
df['Field GCS'] = np.where(((df['Field GCS'] == '*NA') | (df['Field GCS'] == '*ND') | (df['Field GCS'] == '*BL') | (df['Field GCS'] == '')) & (df['Gender'] == 0),'14.90',df['Field GCS'])
df['Field SBP'] = np.where(((df['Field SBP'] == '*NA') | (df['Field SBP'] == '*ND') | (df['Field SBP'] == '*BL') | (df['Field SBP'] == '')) & (df['Gender'] == 1),'111.50',df['Field SBP'])
df['Field SBP'] = np.where(((df['Field SBP'] == '*NA') | (df['Field SBP'] == '*ND') | (df['Field SBP'] == '*BL') | (df['Field SBP'] == '')) & (df['Gender'] == 0),'120.25',df['Field SBP'])
df['Field HR'] = np.where(((df['Field HR'] == '*NA') | (df['Field HR'] == '*ND') | (df['Field HR'] == '*BL') | (df['Field HR'] == '')) & (df['Gender'] == 1),'106',df['Field HR'])
df['Field HR'] = np.where(((df['Field HR'] == '*NA') | (df['Field HR'] == '*ND') | (df['Field HR'] == '*BL') | (df['Field HR'] == '')) & (df['Gender'] == 0),'110',df['Field HR'])

df['Levels'] = df['Levels'].replace(['1', '2'], value = [0 , 1])
Y = df["Levels"]
df = df.drop('Levels', axis = 1)

#print("Shape: " + str(df.shape[0]) + " " + str(Y.shape[0]))
print(Y.value_counts())

X, Y = make_classification(n_classes=2, class_sep=2,
                           weights=[0.2, 0.8], n_informative=3, n_redundant=1, flip_y=0,
                           n_features=8, n_clusters_per_class=1, n_samples=4943, random_state=10)
print('Original dataset shape {}'.format(Counter(Y)))
sm = SMOTE(random_state=42)
#print(df.shape)
#print(Y.shape)

X_res, Y_res = sm.fit_sample(df, Y)
#print(Y_res[:10], Y[:10])

clf, cv_scores = random_forest_classifier(X_res, Y_res)
print(sum(cv_scores)/ len(cv_scores))
feature_importances(clf, df)
import sys
sys.exit()
print('Resampled dataset shape {}'.format(Counter(Y_res)))