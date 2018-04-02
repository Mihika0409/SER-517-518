import pandas as pd
import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

df = pd.read_excel('NewFallDataset.xlsx', header = None)

df.columns = ['T1',    'ED/Hosp Arrival Date',    'Age in Years',    'Gender',  'Levels',  'ICD-10 E-code','ICD-9 E-code',  'Trauma Type', 'Report of physical abuse',
              'Injury Comments', 'Airbag Deployment',
              'Patient Position in Vehicle',
              'Safet Equipment Issues',    'Child Restraint', 'MV Speed',    'Fall Height', 'Transport Type',  'Transport Mode',  'Field SBP',
              'Field HR', 'Field RR',    'Resp Assistance',
              'RTS',   'Field GCS',   'Arrived From',    'ED LOS (mins)',   'Dispositon from  ED', 'ED SBP',  'ED HR',   'ED RR',
              'ED GCS',    'Total Vent Days', 'Total Days in ICU',   'Admission Hosp LOS (days)',   'Total LOS (ED+Admit)',
              'Early transfusion? (started 2016)','Severe TBI? (started 2016)', 'Time to 1st OR Visit (mins.)',
              'Final Outcome-Dead or Alive',   'Hospital Disposition',   'Injury Severity Score', 'ICD 9 Dx (before 2016)','ICD 10 Dx (after 1/2016)','AIS 2005']



list = ['Levels', 'Age in Years', 'Fall Height', 'Gender','Field SBP', 'Field HR', 'Field RR', 'RTS', 'Field GCS']

for x in list:
    df = df[pd.notnull(df[x])]

def findAverage(list):
    sum = 0
    for i in range(0,len(list)):
        if list[i] == "*NA" or list[i] == "*ND" or list[i] == "*BL":
            list[i] = sys.maxint

    newlist = []
    for i in list:
        if i == sys.maxint:
            continue
        else:
            newlist.append(i)
    newlist.pop(0)

    for i in newlist:
        sum = sum + i
    average = sum/len(list)
    return average



FallHeightList = df['Fall Height']
FieldSBPList = df['Field SBP']
FieldHR = df['Field HR']
FieldRR = df['Field RR']
RTS = df['RTS']
FieldGCS = df['Field GCS']
AgeInYears = df['Age in Years']


averageFallHeight = findAverage(FallHeightList)

avergaeFieldSBP = findAverage(FieldSBPList)

avgFieldHR = findAverage(FieldHR)

avgFieldRR = findAverage(FieldRR)

avgFieldRTS = findAverage(RTS)

avgFieldGCS = findAverage(FieldGCS)

avgAge = findAverage(AgeInYears)


df['Fall Height'] = df['Fall Height'].replace([sys.maxint], value = [averageFallHeight])

df['Field SBP'] = df['Field SBP'].replace([sys.maxint], value = [avergaeFieldSBP])

df['Field HR'] = df['Field HR'].replace([sys.maxint], value = [avgFieldHR])

df['Field RR'] = df['Field RR'].replace([sys.maxint], value = [avgFieldRR])

df['RTS'] = df['RTS'].replace([sys.maxint], value = [avgFieldRTS])

df['Field GCS'] = df['Field GCS'].replace([sys.maxint], value = [avgFieldGCS])

df['Age in Years'] = df['Age in Years'].replace([sys.maxint], value = [avgAge])


gender_original = df['Gender']

df['Gender'] = df['Gender'].replace(['M', 'F'], value = ['1', '2'])

df[['Report of physical abuse']] = df[['Report of physical abuse']].replace(['*BL'], ['N'])
df[['Severe Brain Injury']] = df[['Severe Brain Injury']].replace(['*BL'], ['N'])
df[['Time to 1st OR Visit (mins.)']] = df[['Time to 1st OR Visit (mins.)']].replace(['*BL'], ['0'])

df[['Levels']] = df[['Levels']].replace([2], [0])

dummy = pd.get_dummies(df['Arrived From'])
df = pd.concat([df, dummy], axis=1)
del df['Arrived From']

dummy1 = pd.get_dummies(df['Transport Mode'])
df = pd.concat([df, dummy1], axis=1)
del df['Transport Mode']

dummy3 = pd.get_dummies(df['Trauma Type'])
df = pd.concat([df, dummy3], axis=1)
del df['Trauma Type']

dummy4 = pd.get_dummies(df['Gender'])
df = pd.concat([df, dummy4], axis=1)
del df['Gender']


y = df['Levels']

X = df.drop('Levels', 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1)

classification_pipeline = Pipeline([('StandardScalar', StandardScaler()), ('classifier',DecisionTreeClassifier(random_state=1))])

classification_pipeline.fit(X_train,Y_train)


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
