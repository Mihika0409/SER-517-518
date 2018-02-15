import pandas as pd
import sklearn.preprocessing as sk
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


df = pd.read_csv('/Users/Pramod/Desktop/SER517/Trauma2.csv', header = None, error_bad_lines=False)
df.columns = ['T1',    'ED/Hosp Arrival Date',    'Age in Years',    'Gender',  'Levels',  'ICD-10 E-code',   'Trauma Type', 'Report of physical abuse',    'Injury Comments', 'Airbag Deployment',   'Patient Position in Vehicle',
              'Safet Equipment Issues',    'Child Restraint', 'MV Speed',    'Fall Height', 'Transport Type',  'Transport Mode',  'Field SBP',   'Field HR',    'Field Shock Index',   'Field RR',    'Resp Assistance',
              'RTS',   'Field GCS',   'Arrived From',    'ED LOS (mins)',   'Dispositon from  ED', 'ED SBP',  'ED HR',   'ED RR',
              'ED GCS',    'Total Vent Days', 'Total Days in ICU',   'Admission Hosp LOS (days)',   'Total LOS (ED+Admit)',    'Received blood within 4 hrs', 'Severe Brain Injury', 'Time to 1st OR Visit (mins.)',
              'Final Outcome-Dead or Alive',   'Discharge Disposition',   'Injury Severity Score',   'AIS 2005']

#print df.head()
df = df[['Levels', 'Age in Years', 'Gender','Field SBP', 'Field HR', 'Field Shock Index', 'Field RR', 'RTS', 'Field GCS']]
#print df.head()

# Taking only the rows with levels 1 and 2 trauma levels
df = df.loc[df['Levels'].isin(['1', '2'])]

#Dropping all the rows with null values
list1 = ['Levels', 'Age in Years', 'Gender','Field SBP', 'Field HR', 'Field Shock Index', 'Field RR', 'RTS', 'Field GCS']

for x in list1:
    df = df[pd.notnull(df[x])]

#applying label encoder to the gender column to convert it into  numerical values
gender_original = df['Gender']
#print gender_original

le = sk.LabelEncoder()
gender_tranform = le.fit_transform(df['Gender'])

df['Gender'] = df['Gender'].replace(['M', 'F'], value = ['1', '2'])

#Taking the target variable
y = df['Levels']
#print y
#Taking the input variable
X = df.drop('Levels', 1)
print X

# Applying Logistic regression on the pre factors
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1)

classification_pipeline = Pipeline([('pca', PCA(n_components=2)), ('classifier',LogisticRegression(random_state=1))])
#
classification_pipeline.fit(X_train,Y_train)

#compute the accuracy of the model on test data
accuracy = classification_pipeline.score(X_test, Y_test)
print('Test Accuracy: %.3f' % accuracy)

accuracy_training = classification_pipeline.score(X_train, Y_train)
print('Training Accuracy: %.3f' % accuracy_training)

# Confusion matrix
y_pred = classification_pipeline.predict(X_test)

print "The actual values are :"

Y_test_manual = Y_test.tolist()
print type(Y_test_manual)
print Y_test_manual

y_pred_manual = y_pred.tolist()
print "The predicted values :"
#print y_pred
print y_pred_manual

count = 0 # variable to keep count of the matched values between the the predicted values(y_pred) and the actual values(Y_test)

# Calculating the accuracy of the model manually
for x in range(0, len(y_pred_manual)):
    if(Y_test_manual[x] == y_pred_manual[x]):
        count = count + 1

print "The count is"
print count
print len(y_pred_manual)
accuracy_logistic_manual = (float(count)/float(len(y_pred_manual))) * 100

print "The accuracy manually calculated is: "
print accuracy_logistic_manual

tn, fp, fn, tp = confusion_matrix(Y_test_manual, y_pred_manual).ravel()
print "The metrics True Negatives, False Positive, False Negatives, True Positive in the order are: "
print (tn, fp, fn, tp)
print ""
# confmat = confusion_matrix(y_true=Y_test,y_pred=y_pred)
# print(confmat)

# Printing out the different metrics
print "The precision, recall and f-score are:"
print (precision_recall_fscore_support(Y_test_manual, y_pred_manual, average='macro'))


#**********************************************************

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

#**********************************************************

bg = BaggingClassifier(LogisticRegression(), max_samples=1.0, n_estimators = 10)
bg.fit(X_train, Y_train)

print "The bagging score for training data is: "
print bg.score(X_train, Y_train)

print "The bagging score for test data is: "
print bg.score(X_test, Y_test)

y_pred_bagging = bg.predict(X_test)
y_pred_bagging_list = y_pred_bagging.tolist()

# print "The predicted values without bagging are :"
# print y_pred_manual
#
# print "The predicted values of bagging are :"
# print y_pred_bagging_list
#
# print "The actual values are: "
# print Y_test_manual

#**********************************************************

adb = AdaBoostClassifier(LogisticRegression(), n_estimators = 5, learning_rate = 0.5)
adb.fit(X_train, Y_train)

print "The boosting score for training data is: "
print adb.score(X_train, Y_train)

print "The boosting score for test data is: "
print adb.score(X_test, Y_test)