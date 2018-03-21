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

df = pd.read_csv('/Users/vc/Downloads/Trauma_dataset.csv')
df.columns = ['T1#', 'ED/Hosp Arrival Date', 'Age in Years', 'Gender', 'Levels', 'ICD-10 E-code (after 1/2016)', 'ICD-9 E-code (before 2016)',
               'Trauma Type', 'Report of abuse? (started 2014)', 'Injury Comments', 'Airbag Deployment', 'Patient Position in Vehicle',
               'Safety Equipment Issues', 'Child Restraint', 'MV Speed', 'Fall Height', 'Transport Type', 'Transport Mode', 'Field SBP',
               'Field HR', 'Field RR', 'Resp Assistance', 'RTS', 'Field GCS', 'Arrived From', 'ED LOS (mins)', 'ED Disposition', 'ED SBP',
               'ED HR','ED RR', 'ED GCS', 'Total Vent Days', 'Total Days in ICU', 'Admission Hosp LOS (days)', 'Total LOS (ED+Admit)',
               'Early transfusion? (started 2016)', 'Severe TBI? (started 2016)', 'Time to 1st OR Visit (mins.)', 'Final Outcome-Dead or Alive', 'Hospital Disposition',
               'Injury Severity Score', 'ICD 9 Dx (before 2016)', 'ICD 10 Dx (after 1/2016)', 'AIS 2005']
#print (df.head())

df = df.loc[df['Levels'].isin(['1', '2'])]
df = df[['Age in Years', 'Gender', 'Field SBP', 'Field HR', 'Field Shock Index', 'Field RR', 'RTS', 'Field GCS', 'Levels']]
cols = df.columns

for col in cols:
    df = df[pd.notnull(df[col])]

df['Levels'] = df['Levels'].replace(['1', '2'], value = [0 , 1])
Y = df["Levels"]
df = df.drop('Levels', axis = 1)

print("Shape: " + str(df.shape[0]) + " " + str(Y.shape[0]))
print(Y.value_counts())
le = sk.LabelEncoder()
#gender_tranform = le.fit_transform(df['Gender'])
df['Gender'] = df['Gender'].replace(['M', 'F'], value = ['1', '2'])
#df['Levels'] = df['Levels'].replace(['N'])

# Taking only the rows with levels 1 and 2 trauma levels


X, Y = make_classification(n_classes=2, class_sep=2,
                           weights=[0.2, 0.8], n_informative=3, n_redundant=1, flip_y=0,
                           n_features=8, n_clusters_per_class=1, n_samples=669, random_state=10)
print('Original dataset shape {}'.format(Counter(Y)))
sm = SMOTE(random_state=42)

X_res, Y_res = sm.fit_sample(df, Y)
#print(Y_res[:10], Y[:10])
def random_forest_classifier(features, target):
    clf = RandomForestClassifier()
    clf.fit(features, target)
    from sklearn.cross_validation import cross_val_score
    cross_val = cross_val_score(clf, features, target, cv = 10)
    return clf, cross_val

def feature_imporotances(clf, data):
    feats = {}  # a dict to hold feature_name: feature_importance
    for feature, importance in zip(data.columns, clf.feature_importances_):
        feats[feature] = importance  # add the name/value pair
    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
    importances.sort_values(by='Gini-importance').plot(kind='bar', rot=45)
    plt.show()

clf, cv_scores = random_forest_classifier(X_res, Y_res)
print(sum(cv_scores)/ len(cv_scores))
feature_imporotances(clf, df)
import sys
sys.exit()
print('Resampled dataset shape {}'.format(Counter(Y_res)))