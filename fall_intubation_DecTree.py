
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
        
        
        
  
        
        
  
