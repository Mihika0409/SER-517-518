import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

df = pd.read_csv('/Users/Pramod/Desktop/SER517/Original datasets/FullData.csv', header = None, error_bad_lines=False)
df.columns = ['T1#', 'ED/Hosp Arrival Date', 'Age in Years', 'Gender', 'Levels', 'ICD-10 E-code (after 1/2016)', 'ICD-9 E-code (before 2016)',
              'Trauma Type', 'Report of abuse? (started 2014)', 'Injury Comments', 'Airbag Deployment', 'Patient Position in Vehicle',
              'Safety Equipment Issues', 'Child Restraint', 'MV Speed', 'Fall Height', 'Transport Type', 'Transport Mode', 'Field SBP',
              'Field HR', 'Field RR', 'Resp Assistance', 'RTS', 'Field GCS', 'Arrived From', 'ED LOS (mins)', 'ED Disposition', 'ED SBP',
              'ED HR','ED RR', 'ED GCS', 'Total Vent Days', 'Total Days in ICU', 'Admission Hosp LOS (days)', 'Total LOS (ED+Admit)',
              'Early transfusion? (started 2016)', 'Severe TBI? (started 2016)', 'Time to 1st OR Visit (mins.)', 'Final Outcome-Dead or Alive', 'Hospital Disposition',
              'Injury Severity Score', 'ICD 9 Dx (before 2016)', 'ICD 10 Dx (after 1/2016)', 'AIS 2005']

#********************************************************************************
#Taking the rows with injury comments which contain the ACS gun shot critertia
df_assault_icd9_fall = df[df['ICD-9 E-code (before 2016)'].str.contains("FALL", na = False) ]
print "Number of rows:"
print len(df_assault_icd9_fall)
print ""

#Taking the rows with injury comments which contain the ACS gun shot critertia
df_assault_icd10_fall = df[df['ICD-10 E-code (after 1/2016)'].str.contains("FALL", na = False) ]
print "Number of rows:"
print len(df_assault_icd10_fall)
print ""
#********************************************************************************

#Taking the rows with injury comments which contain the ACS gun shot critertia
df_assault_icd9 = df[df['ICD-9 E-code (before 2016)'].str.contains("ASSAULT", na = False) ]
print "Number of rows:"
print len(df_assault_icd9)
print ""

#Taking the rows with injury comments which contain the ACS gun shot critertia
df_assault_icd10 = df[df['ICD-10 E-code (after 1/2016)'].str.contains("ASSAULT", na = False) ]
print "Number of rows:"
print len(df_assault_icd10)
print ""

print "The dataframe is: "
#print df_assault_icd10

#Combining data of both icd 9 and icd 10 codes
frames = [df_assault_icd9, df_assault_icd10]
result = pd.concat(frames)
print "Number of rows in complete dataframe:"
print len(result)
#Printing out the resultant dataframe
print result

#Removing the null values in all the columns
list1 = ['T1#', 'ED/Hosp Arrival Date', 'Age in Years', 'Gender', 'Levels', 'ICD-10 E-code (after 1/2016)', 'ICD-9 E-code (before 2016)',
              'Trauma Type', 'Report of abuse? (started 2014)', 'Injury Comments', 'Airbag Deployment', 'Patient Position in Vehicle',
              'Safety Equipment Issues', 'Child Restraint', 'MV Speed', 'Fall Height', 'Transport Type', 'Transport Mode', 'Field SBP',
              'Field HR', 'Field RR', 'Resp Assistance', 'RTS', 'Field GCS', 'Arrived From', 'ED LOS (mins)', 'ED Disposition', 'ED SBP',
              'ED HR','ED RR', 'ED GCS', 'Total Vent Days', 'Total Days in ICU', 'Admission Hosp LOS (days)', 'Total LOS (ED+Admit)',
              'Early transfusion? (started 2016)', 'Severe TBI? (started 2016)', 'Time to 1st OR Visit (mins.)', 'Final Outcome-Dead or Alive', 'Hospital Disposition',
              'Injury Severity Score', 'ICD 9 Dx (before 2016)', 'ICD 10 Dx (after 1/2016)', 'AIS 2005']

for x in list1:
    result = result[pd.notnull(result[x])]

#Printing the length of the resultant dataframe after removing all the null values
print len(result)
# print result

#Only considering the columns we will require
result = result[['Levels', 'Age in Years', 'Gender','Field SBP', 'Field HR', 'Field RR', 'RTS', 'Field GCS']]

# Taking only the rows with levels 1 and 2 trauma levels
result = result.loc[result['Levels'].isin(['1', '2'])]

#Replacing Male with 1 and female with 2
result['Gender'] = result['Gender'].replace(['M', 'F'], value = ['1', '2'])

#Replacing the NA, ND, and Bl rows with a very small value
list2 = ['Levels', 'Age in Years', 'Gender','Field SBP', 'Field HR', 'Field RR', 'RTS', 'Field GCS']
for x in list2:
    result[x] = result[x].replace(['*NA', '*ND', '*BL'], value=['0.0000001', '0.000001', '0.000001'])

print result

#Taking the target variable
y = result['Levels']
#print y
#Taking the input variable
X = result.drop('Levels', 1)
print X

# Applying Logistic regression on the pre factors
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1)

classification_pipeline = Pipeline([('StandardScalar', StandardScaler()), ('pca', PCA(n_components=4)), ('classifier',DecisionTreeClassifier(random_state=1))])
#
classification_pipeline.fit(X_train,Y_train)

#compute the accuracy of the model on test data
accuracy = classification_pipeline.score(X_test, Y_test)
print('Test Accuracy: %.3f' % accuracy)

accuracy_training = classification_pipeline.score(X_train, Y_train)
print('Training Accuracy: %.3f' % accuracy_training)


y_pred = classification_pipeline.predict(X_test)

tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
print "The metrics True Negatives, False Positive, False Negatives, True Positive in the order are: "
print (tn, fp, fn, tp)
print ""
# confmat = confusion_matrix(y_true=Y_test,y_pred=y_pred)
# print(confmat)

# Printing out the different metrics
print "The precision, recall and f-score are:"
print (precision_recall_fscore_support(Y_test, y_pred, average='macro'))

over_triage_count = 0;
total_ones = 0

y_test = Y_test.tolist()
Y_pred = y_pred.tolist()

for x in range(0, len(y_pred)):
    if Y_pred[x] == '1' and y_test[x] == '2':
        over_triage_count = over_triage_count + 1

for x in range(0, len(y_pred)):
    if Y_pred[x] == '1':
        total_ones = total_ones + 1

print over_triage_count

print "The over triage percentage is:"
print float(over_triage_count)/float(total_ones)
