use Trauma;
 
 
 create table Trauma_NewData ( tid Bigint(10) , hosp_date date, age int, gender char(1), 
                           levels varchar(3), icd_code varchar (200),icd_code_9 varchar (200),trauma_type varchar(10),physical_abuse varchar (5),
                           airbag_deploy varchar(15),patient_pos varchar(15),
                           safety_equip_issues varchar(30), child_restraint varchar(10),mv_speed  varchar(10), 
                           fall_height varchar(10), transport_type varchar(100),transport_mode  varchar(50),feild_SBP int ,
                           feild_HR int, feild_RR int, resp_assis varchar(50),RTS int ,
                           feild_GCS int,arrived_from varchar(20),ED_LOS int,disposition varchar(10),ED_SBP int, 
                           ED_HR int, ED_RR int, ED_GCS int, total_vent_days varchar(5), days_in_icu varchar(5),  
                           hosp_LOS varchar(5),total_LOS varchar(5),received_blood varchar (5), brain_injury varchar(5),
                           time_to_first_OR varchar(15) , death varchar (5),discharge_dispo varchar (30),AIS varchar(10),AIS_2005 bigint);
                           
                           
                           
 LOAD DATA LOCAL INFILE '/Users/gowtham/Downloads/sixyears_newdata.csv' 
 INTO TABLE  Trauma.Trauma_NewData FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n';
 
 
 SELECT * 
FROM Trauma.Trauma_NewData
where tid <>0 and icd_code like '% fall%' 
or icd_code_9 like '% fall%' 
and transport_mode <> 'Private/Public Vehicle/Walk-in'
limit  50000
                           

 import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
df = pd.read_csv('/Users/gowtham/Desktop/SER-517&518/Fall_trauma_newdata.csv', header = None, error_bad_lines=False)

df.columns = ['T1',    'ED/Hosp Arrival Date',    'Age in Years',    'Gender',  'Levels',  'ICD-10 E-code','ICD-9 E-code',  'Trauma Type', 'Report of physical abuse', 'Airbag Deployment',   'Patient Position in Vehicle',
              'Safet Equipment Issues',    'Child Restraint', 'MV Speed',    'Fall Height', 'Transport Type',  'Transport Mode',  'Field SBP',   'Field HR', 'Field RR',    'Resp Assistance',
              'RTS',   'Field GCS',   'Arrived From',    'ED LOS (mins)',   'Dispositon from  ED', 'ED SBP',  'ED HR',   'ED RR',
              'ED GCS',    'Total Vent Days', 'Total Days in ICU',   'Admission Hosp LOS (days)',   'Total LOS (ED+Admit)',    'Received blood within 4 hrs', 'Severe Brain Injury', 'Time to 1st OR Visit (mins.)',
              'Final Outcome-Dead or Alive',   'Discharge Disposition',   'Injury Severity Score', 'AIS 2005']

#print df.head()

df = df[[ 'Age in Years', 'Gender','Fall Height','Field SBP', 'Field HR', 'Field RR', 'RTS', 'Field GCS','Levels']]
#print df.head()
#Dropping all the rows with null values
features = [ 'Age in Years', 'Gender','Fall Height','Field SBP', 'Field HR', 'Field RR', 'RTS', 'Field GCS','Levels']
for x in features:
    df = df[pd.notnull(df[x])]
    df = df[pd.notnull(df[x])].replace(['*NA', '*ND', '*BL'], value=['0', '0', '0'])
df = df.loc[df['Levels'].isin(['1', '2'])]

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'].values)

print df.head()


X= df.iloc[:,:-1].values
y= df['Levels'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

dt = tree.DecisionTreeClassifier()
dt = dt.fit(X_train,y_train)

#y_train_pred = dt.predict(X_train)
#y_test_pred = dt.predict(X_test)

#print('MSE train: %.3f, test: %.3f' % ( mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))


#decision trees with the gini index
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
#clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)
y_pred_train = clf_gini.predict(X_train)

print y_pred

print "test Accuracy is ", accuracy_score(y_test,y_pred)*100
print "train Accuracy is ", accuracy_score(y_train,y_pred_train)*100

print "testing error is ", (1-accuracy_score(y_test,y_pred))*100
print "training error is ", (1-accuracy_score(y_train,y_pred_train))*100

# Code to append a new intubated column to data.
with open('/Users/gowtham/Desktop/SER-517&518/Fall_trauma_newdata.csv', 'rb') as inp, open(
        '/Users/gowtham/Desktop/SER-517&518/Fall_trauma_newdata1.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if "intubated" in row[9].lower():
            row.append('1')
        else:
            row.append('0')
        writer.writerow(row)
        
  # Modified the RespAssistance Column.
with open('/Users/gowtham/Desktop/SER-517&518/Fall_trauma_newdata1.csv', 'rb') as inp, open(
        '/Users/gowtham/Desktop/SER-517&518/Fall_trauma_newdata2.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[21] == "Assisted Respiratory Rate":
            row[21] = '1'
        else:
            row[21] = '0'
        writer.writerow(row)

# decision tree with the information gain
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=3, min_samples_leaf=5)

#clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100)
clf_entropy.fit(X_train, y_train)

y_pred_en = clf_entropy.predict(X_test)
y_pred_train_en = clf_entropy.predict(X_train)

print y_pred_en

print "test Accuracy is ", accuracy_score(y_test,y_pred)*100
print "train Accuracy is ", accuracy_score(y_train,y_pred_train)*100

print "testing error is ", (1-accuracy_score(y_test,y_pred))*100
print "training error is ", (1-accuracy_score(y_train,y_pred_train))*100

# decision tree with the information gain
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=3, min_samples_leaf=5)

#clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100)
clf_entropy.fit(X_train, y_train)

y_pred_en = clf_entropy.predict(X_test)
y_pred_train_en = clf_entropy.predict(X_train)

print y_pred_en

#print "Accuracy is ", accuracy_score(y_test,y_pred_en)*100

print "test Accuracy is ", accuracy_score(y_test,y_pred_en)*100
print "train Accuracy is ", accuracy_score(y_train,y_pred_train_en)*100

print "testing error is ", (1-accuracy_score(y_test,y_pred_en))*100
print "training error is ", (1-accuracy_score(y_train,y_pred_train_en))*100


scores = cross_val_score(estimator=clf_gini,     # Model to test
                X= X_train,
                y = y_train,      # Target variable
                scoring = "accuracy",               # Scoring metric
                cv=10)                              # Cross validation folds

scores=scores*100
print("Accuracy per fold: ")
print(scores)
print("Average accuracy for gini using training data: ", scores.mean())

#.......cross validation for gini index using test data ......

scores_test = cross_val_score(estimator=clf_gini,     # Model to test
                X= X_test,
                y = y_test,      # Target variable
                scoring = "accuracy",               # Scoring metric
                cv=10)                              # Cross validation folds

scores_test=scores_test*100
print("Accuracy per fold: ")
print(scores_test)
print("Average accuracy for gini using test data: ", scores_test.mean())

#.......cross validation for entropy  using training data ......
en_scores = cross_val_score(estimator=clf_entropy,     # Model to test
                X= X_train,
                y = y_train,      # Target variable
                scoring = "accuracy",               # Scoring metric
                cv=10)                              # Cross validation folds

en_scores=en_scores*100
print("Accuracy per fold: ")
print(en_scores)
print("Average accuracy for entrpoy training: ", en_scores.mean())

#.......cross validation for entropy using test data ......

en_scores_test = cross_val_score(estimator=clf_entropy,     # Model to test
                X= X_test,
                y = y_test,                                 # Target variable
                scoring = "accuracy",                       # Scoring metric
                cv=10)                                      # Cross validation folds

en_scores_test=en_scores_test*100
print("Accuracy per fold: ")
print(en_scores_test)
print("Average accuracy for entropy testing: ", en_scores_test.mean())

# Confusion matrix  matrix with the Gini index
conf_matrix = metrics.confusion_matrix(y_test, y_pred_en)
print conf_matrix

# confusion matrix with the entropy
conf_matrix2 = metrics.confusion_matrix(y_test, y_pred)
print conf_matrix2


