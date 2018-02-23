import pandas as pd
import sklearn.preprocessing as sk
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

df = pd.read_csv('/Users/Pramod/Desktop/SER517/Trauma2.csv', header = None, error_bad_lines=False)
df.columns = ['T1',    'ED/Hosp Arrival Date',    'Age in Years',    'Gender',  'Levels',  'ICD',   'Trauma Type', 'Report of physical abuse',    'Injury Comments', 'Airbag Deployment',   'Patient Position in Vehicle',
              'Safet Equipment Issues',    'Child Restraint', 'MV Speed',    'Fall Height', 'Transport Type',  'Transport Mode',  'Field SBP',   'Field HR',    'Field Shock Index',   'Field RR',    'Resp Assistance',
              'RTS',   'Field GCS',   'Arrived From',    'ED LOS (mins)',   'Dispositon from  ED', 'ED SBP',  'ED HR',   'ED RR',
              'ED GCS',    'Total Vent Days', 'Total Days in ICU',   'Admission Hosp LOS (days)',   'Total LOS (ED+Admit)',    'Received blood within 4 hrs', 'Severe Brain Injury', 'Time to 1st OR Visit (mins.)',
              'Final Outcome-Dead or Alive',   'Discharge Disposition',   'Injury Severity Score',   'AIS 2005']


df = df[df['ICD'].str.contains("FALL", na = False)]

#finding out the size of the dataframe with only Fall values
print df.size

df = df[['Levels', 'Age in Years', 'Fall Height', 'Gender', 'Field SBP', 'Field HR', 'Field Shock Index', 'Field RR', 'RTS', 'Field GCS']]

# Taking only the rows with levels 1 and 2 trauma levels
df = df.loc[df['Levels'].isin(['1', '2'])]

#Dropping all the rows with null values
list1 = ['Levels', 'Age in Years', 'Fall Height', 'Gender','Field SBP', 'Field HR', 'Field Shock Index', 'Field RR', 'RTS', 'Field GCS']

for x in list1:
    df = df[pd.notnull(df[x])]

#applying label encoder to the gender column to convert it into  numerical values
gender_original = df['Gender']
#print gender_original

# le = sk.LabelEncoder()
# gender_tranform = le.fit_transform(df['Gender'])

df['Gender'] = df['Gender'].replace(['M', 'F'], value = ['1', '2'])
df['Fall Height'] = df['Fall Height'].replace(['*NA', '*ND', '*BL'], value = ['0', '0', '0'])

#Taking the target variable
y = df['Levels']
#print y
#Taking the input variable
X = df.drop('Levels', 1)
print X
# print y

#Applying Decision Tree Classifier on the pre factors
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1)

classification_pipeline = Pipeline([('StandardScalar', StandardScaler()), ('classifier',DecisionTreeClassifier(random_state=1))])

classification_pipeline.fit(X_train,Y_train)

#compute the accuracy of the model on test data
accuracy = classification_pipeline.score(X_test, Y_test)
print('Test Accuracy: %.3f' % accuracy)

accuracy_training = classification_pipeline.score(X_train, Y_train)
print('Training Accuracy: %.3f' % accuracy_training)

tn, fp, fn, tp = confusion_matrix(Y_test, classification_pipeline.predict(X_test)).ravel()
print "The metrics True Negatives, False Positive, False Negatives, True Positive in the order are: "
print (tn, fp, fn, tp)
print ""

print "The precision, recall and f-score are:"
print (precision_recall_fscore_support(Y_test, classification_pipeline.predict(X_test), average='macro'))
