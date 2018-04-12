import pandas as pd
import sklearn.preprocessing as sk
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
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

#Taking the rows which do not contain the words FALL, MOTOR or CAR (na = False means that it igoners the rows with NaN values)
df_assault_icd9 = df[df['ICD-9 E-code (before 2016)'].str.contains(r'^(?:(?!FALL|MOTOR|CAR).)*$', na = False) ]

#df_assault_icd9 =  df['A'].str.contains(r'^(?:(?!ASSAULT|World).)*$')
print "Number of rows:"
print len(df_assault_icd9)
print ""

#Taking the rows which do not contain the words FALL, MOTOR or CAR (na = False means that it igoners the rows with NaN values)
df_assault_icd10 = df[df['ICD-10 E-code (after 1/2016)'].str.contains(r'^(?:(?!FALL|MOTOR|CAR).)*$', na = False) ]
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
#print result


#Only considering the columns we will require
result = result[['Levels', 'Age in Years', 'Gender','Field SBP', 'Field HR', 'Field RR', 'RTS', 'Field GCS']]

# Taking only the rows with levels 1 and 2 trauma levels
result = result.loc[result['Levels'].isin(['1', '2'])]

#Replacing Male with 1 and female with 2
result['Gender'] = result['Gender'].replace(['M', 'F'], value = ['1', '2'])

#*************************************************************************
list1 = [result['Field SBP'], result['Field HR'], result['Field RR'], result['RTS'], result['Field GCS']]

def create_list(x):
    return x.tolist()

# list_dummy = create_list(result['Field SBP'])

def average(list):
    avgList = []
    count = 0;
    for i in list:
        if i != '*NA' and i != '*ND' and i != '*BL':
            avgList.append(float(i))
            count = count + 1
    return (sum(avgList)/count)

# print "the average of dummy list is"
# print average(list_dummy)

list_names = ['Field SBP', 'Field HR', 'Field RR', 'RTS', 'Field GCS']

list_avg_values = []

for i in range(0, len(list1)):
    print list_names[i]
    x = (average(create_list(list1[i])))
    print x
    list_avg_values.append(x)

#*************************************************************************

#Replacing the NA, ND, and Bl rows with a very small value
list2 = ['Field SBP', 'Field HR', 'Field RR', 'RTS', 'Field GCS']
for x in range(0, len(list2)):
    result[list_names[x]] = result[list_names[x]].replace(['*NA', '*ND', '*BL'], value=[list_avg_values[x], list_avg_values[x], list_avg_values[x]])

print result
#*************************************************************************
# Logistic Regression

#Taking the target variable
y = result['Levels']
#print y
#Taking the input variable
X = result.drop('Levels', 1)
print X

# Applying Logistic regression on the pre factors
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1)

classification_pipeline = Pipeline([('pca', PCA(n_components=7)), ('classifier',LogisticRegression(random_state=1))])
#
classification_pipeline.fit(X_train,Y_train)

#compute the accuracy of the model on test data
accuracy = classification_pipeline.score(X_test, Y_test)
print('Test Accuracy: %.3f' % accuracy)

accuracy_training = classification_pipeline.score(X_train, Y_train)
print('Training Accuracy: %.3f' % accuracy_training)


y_pred = classification_pipeline.predict(X_test)

print(confusion_matrix(Y_test, y_pred))
# print "The metrics True Negatives, False Positive, False Negatives, True Positive in the order are: "
# print (tn, fp, fn, tp)
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

print "The over triage percentage for Logistic Regression is:"
print float(over_triage_count)/float(total_ones)

#*************************************************************************
# Decision Tree

classification_pipeline = Pipeline([('StandardScalar', StandardScaler()), ('pca', PCA(n_components=7)), ('classifier',DecisionTreeClassifier(random_state=1))])
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

print "The over triage percentage for Decision Tree is:"
print float(over_triage_count)/float(total_ones)

#**********************************************************

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier

#**********************************************************

bg = BaggingClassifier(LogisticRegression(), max_samples=1.0, n_estimators = 10)
bg.fit(X_train, Y_train)

print "The bagging score for training data is: "
print bg.score(X_train, Y_train)

print "The bagging score for test data is: "
print bg.score(X_test, Y_test)

y_pred_bagging = bg.predict(X_test)
y_pred_bagging_list = y_pred_bagging.tolist()

print "The predicted values without bagging are :"
print Y_pred

print "The predicted values of bagging are :"
print y_pred_bagging_list

print "The actual values are: "
print y_test

print "The precision, recall and f-score of bagging are:"
print (precision_recall_fscore_support(y_test, y_pred_bagging_list, average='macro'))

#**********************************************************

adb = AdaBoostClassifier(LogisticRegression(), n_estimators = 5, learning_rate = 0.5)
adb.fit(X_train, Y_train)

print "The boosting score for training data is: "
print adb.score(X_train, Y_train)

print "The boosting score for test data is: "
print adb.score(X_test, Y_test)

#**********************************************************

lr = LogisticRegression()
dt = DecisionTreeClassifier()
svm = SVC(kernel = 'poly', degree = 2 )

vc = VotingClassifier( estimators= [('lr',lr),('dt',dt),('svm',svm)], voting = 'hard')
vc.fit_transform(X_train, Y_train)

print "The voting classifier score is: "
print vc.score(X_test, Y_test)

# Printing out the different metrics
print "The precision, recall and f-score are:"
print (precision_recall_fscore_support(Y_test, vc.predict(X_test), average='macro'))
