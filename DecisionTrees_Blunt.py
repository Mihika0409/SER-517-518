import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

df = pd.read_excel('NewBluntDataset.xlsx', header=None)

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


fallDF = df

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

fallDF = fallDF[
    ['Levels', 'Fall Height', 'Gender', 'Field SBP', 'Field HR', 'Field RR', 'RTS', 'Field GCS', 'Report of physical abuse','Trauma Type','MV Speed']]

fallDF['Gender'] = fallDF['Gender'].replace(['M', 'F'], value=['1', '2'])


fallDF['Fall Height'] = fallDF['Fall Height'].replace(['*NA', '*ND', '*BL'], value=['5','5','5'])

fallDF['Field SBP'] = fallDF['Field SBP'].replace(['*NA', '*ND', '*BL'], value=['76','76','76'])

fallDF['Field HR'] = fallDF['Field HR'].replace(['*NA', '*ND', '*BL'], value=['83','83','83'])

fallDF['Field RR'] = fallDF['Field RR'].replace(['*NA', '*ND', '*BL'], value=['17','17','17'])

fallDF['RTS'] = fallDF['RTS'].replace(['*NA', '*ND', '*BL'], value=['3.39613496933','3.39613496933','3.39613496933'])

fallDF['Field GCS'] = fallDF['Field GCS'].replace(['*NA', '*ND', '*BL'], value=['9','9','9'])

fallDF['Report of physical abuse'] = fallDF['Report of physical abuse'].replace(['*BL'], value=['N'])

fallDF['Report of physical abuse'] = fallDF['Report of physical abuse'].replace(['N','Y'], value=['0','1'])

fallDF['Trauma Type'] = fallDF['Trauma Type'].replace(['Blunt','Penetrating'], value = ['1','0'])

fallDF['MV Speed'] = fallDF['MV Speed'].replace(['*NA', '*ND', '*BL'], value=['21','21','21'])




y = fallDF['Levels']

X = fallDF.drop('Levels', 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1)

classification_pipeline = Pipeline([('pca', PCA(n_components=7)), ('classifier',DecisionTreeClassifier(random_state=1))])

classification_pipeline.fit(X_train,Y_train)


accuracy = classification_pipeline.score(X_test, Y_test)
accuracy = accuracy * 100
print('Test Accuracy:' , accuracy)


accuracy2 = classification_pipeline.score(X_train, Y_train)
accuracy2 = accuracy2 * 100

print('Training Accuracy: %.3f' % accuracy2)

y_pred = classification_pipeline.predict(X_test)

tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
print "The metrics True Negatives, False Positive, False Negatives, True Positive in the order are: "
print (tn, fp, fn, tp)



print "The precision, recall and f-score are:"
print (precision_recall_fscore_support(Y_test, y_pred, average='macro'))

triage_count = 0

y_test = Y_test.tolist()
Y_pred = y_pred.tolist()

for x in range(0, len(y_pred)):
    if Y_pred[x] == '1' and y_test[x] == '2':
        triage_count = triage_count + 1

print "over triage percentage ="
print float(triage_count)/float(len(Y_pred))


print "UnderTriage Percentage ="
print (tn + fp)/tp


