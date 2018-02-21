import pandas as pd
import sklearn.preprocessing as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('/Users/vc/Downloads/Trauma_dataset.csv')
df.columns = ['T1',    'ED/Hosp Arrival Date',    'Age in Years',    'Gender',  'Levels',  'ICD-10 E-code',   'Trauma Type',
              'Report of physical abuse',    'Injury Comments', 'Airbag Deployment',   'Patient Position in Vehicle',  'Safet Equipment Issues',
              'Child Restraint', 'MV Speed',    'Fall Height', 'Transport Type',  'Transport Mode',  'Field SBP',   'Field HR',
              'Field Shock Index',   'Field RR',    'Resp Assistance', 'RTS',   'Field GCS',   'Arrived From', 'ED LOS (mins)',
              'Dispositon from  ED', 'ED SBP',  'ED HR',   'ED RR', 'ED GCS',    'Total Vent Days', 'Total Days in ICU',
              'Admission Hosp LOS (days)',   'Total LOS (ED+Admit)',    'Received blood within 4 hrs', 'Severe Brain Injury',
              'Time to 1st OR Visit (mins.)', 'Final Outcome-Dead or Alive',   'Discharge Disposition',   'Injury Severity Score', 'AIS 2005']

#print (df.head())
df = df[['Age in Years', 'Gender','Field SBP', 'Field HR', 'Field Shock Index', 'Field RR', 'RTS', 'Field GCS','Levels']]

#Dropping all the rows with null values
list1 = ['Age in Years', 'Gender','Field SBP', 'Field HR', 'Field Shock Index', 'Field RR', 'RTS', 'Field GCS','Levels']
for x in list1:
    df = df[pd.notnull(df[x])]

#applying label encoder to the gender column to convert it into  numerical values
gender_original = df['Gender']

le = sk.LabelEncoder()
gender_tranform = le.fit_transform(df['Gender'])
df['Gender'] = df['Gender'].replace(['M', 'F'], value = ['1', '2'])
df['Levels'] = df['Levels'].replace(['N'])

# Taking only the rows with levels 1 and 2 trauma levels
df = df.loc[df['Levels'].isin(['1', '2'])]
#y = df.values()
print(df.groupby('Levels').count())
#import sys
#sys.exit()
print (df.head())

#Split data into train and test datasets
def split_dataset(df, train_percentage, feature_headers, target_header):
    train_x, test_x, train_y, test_y = train_test_split(df[feature_headers], df[target_header],train_size=train_percentage)
    return train_x,test_x,train_y,test_y

def handle_missing_values(df,missing_values_header,missing_label):
    return df[df[missing_values_header] != missing_label]

#Training the random forest classifier with the scikit learn
def random_forest_classifier(features, target):
    clf = RandomForestClassifier(n_estimators=100, max_depth=5)
    clf.fit(features, target)
    return clf

def main():
    #df = pd.read_csv('/Users/vc/Downloads/Trauma_dataset.csv')
    headers = ['Age in Years', 'Gender','Field SBP', 'Field HR', 'Field Shock Index', 'Field RR', 'RTS', 'Field GCS','Levels']
    print(headers)
    # df = handle_missing_values(df, headers[7], None)
    train_x, test_x, train_y, test_y = split_dataset(df, 0.7, headers[0:-1], headers[-1])

    print("Train_x Shape :: ", train_x.shape)
    print("Train_y Shape :: ", train_y.shape)
    print("Test_x Shape :: ", test_x.shape)
    print("Test_y Shape :: ", test_y.shape)

    #Performing predictions
    trained_model = random_forest_classifier(train_x, train_y)
    print("Trained model :: ", trained_model)
    predictions = trained_model.predict(test_x)

    for i in range(0, 5):
        print("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]))

    #Calculating Train and Test accuracy
    print("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
    print("Test Accuracy  :: ", accuracy_score(test_y, predictions))
    print(" Confusion matrix ", confusion_matrix(test_y, predictions))

if __name__ == "__main__":
    main()