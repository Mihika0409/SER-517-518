import pandas as pd
import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

df = pd.read_excel('NewFallDataset.xlsx', header=None)

df.columns = ['T1', 'ED/Hosp Arrival Date', 'Age in Years', 'Gender', 'Levels', 'ICD-10 E-code', 'ICD-9 E-code',
              'Trauma Type', 'Report of physical abuse',
              'Injury Comments', 'Airbag Deployment',
              'Patient Position in Vehicle',
              'Safet Equipment Issues', 'Child Restraint', 'MV Speed', 'Fall Height', 'Transport Type',
              'Transport Mode', 'Field SBP',
              'Field HR', 'Field RR', 'Resp Assistance',
              'RTS', 'Field GCS', 'Arrived From', 'ED LOS (mins)', 'Dispositon from  ED', 'ED SBP', 'ED HR', 'ED RR',
              'ED GCS', 'Total Vent Days', 'Total Days in ICU', 'Admission Hosp LOS (days)', 'Total LOS (ED+Admit)',
              'Early transfusion? (started 2016)', 'Severe TBI? (started 2016)', 'Time to 1st OR Visit (mins.)',
              'Final Outcome-Dead or Alive', 'Hospital Disposition', 'Injury Severity Score', 'ICD 9 Dx (before 2016)',
              'ICD 10 Dx (after 1/2016)', 'AIS 2005']

FallICD09 = df[df['ICD-9 E-code'].str.contains("FALL", na=False)]
FallICD10 = df[df['ICD-10 E-code'].str.contains("FALL", na=False)]

NewDataFrame = [FallICD09, FallICD10]
fallDF = pd.concat(NewDataFrame)

columns = ['T1', 'ED/Hosp Arrival Date', 'Age in Years', 'Gender', 'Levels', 'ICD-10 E-code', 'ICD-9 E-code',
           'Trauma Type', 'Report of physical abuse',
           'Injury Comments', 'Airbag Deployment',
           'Patient Position in Vehicle',
           'Safet Equipment Issues', 'Child Restraint', 'MV Speed', 'Fall Height', 'Transport Type', 'Transport Mode',
           'Field SBP',
           'Field HR', 'Field RR', 'Resp Assistance',
           'RTS', 'Field GCS', 'Arrived From', 'ED LOS (mins)', 'Dispositon from  ED', 'ED SBP', 'ED HR', 'ED RR',
           'ED GCS', 'Total Vent Days', 'Total Days in ICU', 'Admission Hosp LOS (days)', 'Total LOS (ED+Admit)',
           'Early transfusion? (started 2016)', 'Severe TBI? (started 2016)', 'Time to 1st OR Visit (mins.)',
           'Final Outcome-Dead or Alive', 'Hospital Disposition', 'Injury Severity Score', 'ICD 9 Dx (before 2016)',
           'ICD 10 Dx (after 1/2016)', 'AIS 2005']

for i in columns:
    fallDF = fallDF[pd.notnull(fallDF[i])]

fallDF = fallDF.loc[fallDF['Levels'].isin(['1', '2'])]


fallDF = fallDF[
    ['Levels', 'Fall Height', 'Age in Years', 'Gender', 'Field SBP', 'Field HR', 'Field RR', 'RTS', 'Field GCS', 'Airbag Deployment']]

fallDF['Gender'] = fallDF['Gender'].replace(['M', 'F'], value=['1', '2'])

fallDF['Fall Height'] = fallDF['Fall Height'].replace(['*NA', '*ND', '*BL'], value=['5','5','5'])

fallDF['Field SBP'] = fallDF['Field SBP'].replace(['*NA', '*ND', '*BL'], value=['76','76','76'])

fallDF['Field HR'] = fallDF['Field HR'].replace(['*NA', '*ND', '*BL'], value=['83','83','83'])

fallDF['Field RR'] = fallDF['Field RR'].replace(['*NA', '*ND', '*BL'], value=['17','17','17'])

fallDF['RTS'] = fallDF['RTS'].replace(['*NA', '*ND', '*BL'], value=['3.39613496933','3.39613496933','3.39613496933'])

fallDF['Field GCS'] = fallDF['Field GCS'].replace(['*NA', '*ND', '*BL'], value=['9','9','9'])

fallDF['Age in Years'] = fallDF['Age in Years'].replace(['*NA', '*ND', '*BL'], value=['6','6','6'])

fallDF['Airbag Deployment'] = fallDF['Airbag Deployment'].replace(['*NA'], value=['0'])


outputlist = fallDF['Fall Height']


y = fallDF['Levels']

X = fallDF.drop('Levels', 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1)

classification_pipeline = Pipeline([('pca', PCA(n_components=7)), ('classifier',LogisticRegression(random_state=1))])

classification_pipeline.fit(X_train,Y_train)


accuracy = classification_pipeline.score(X_test, Y_test)
print('Test Accuracy: %.3f' % accuracy)


accuracy_training = classification_pipeline.score(X_train, Y_train)
print('Training Accuracy: %.3f' % accuracy_training)

y_pred = classification_pipeline.predict(X_test)

tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
print "The metrics True Negatives, False Positive, False Negatives, True Positive in the order are: "
print (tn, fp, fn, tp)



print "The precision, recall and f-score are:"
print (precision_recall_fscore_support(Y_test, y_pred, average='macro'))

over_triage_count = 0;

y_test = Y_test.tolist()
Y_pred = y_pred.tolist()

for x in range(0, len(y_pred)):
    if Y_pred[x] == '1' and y_test[x] == '2':
        over_triage_count = over_triage_count + 1

print "The over triage percentage is:"
print float(over_triage_count)/float(len(Y_pred))


