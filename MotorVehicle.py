import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from IntegratedCode import incoming_df

incoming_df = incoming_df[
    ['Age in Years', 'Gender', 'MV Speed', 'Field SBP', 'Field HR', 'Field RR', 'RTS', 'Field GCS',
     'Resp Assistance', 'Intubated']]

le = LabelEncoder()
incoming_df['Gender'] = le.fit_transform(incoming_df['Gender'].values)
incoming_df['RTS'] = incoming_df['RTS'].replace(['*NA', '*ND', '*BL', ''], '7.65')
incoming_df['Field GCS'] = incoming_df['Field GCS'].replace(['*NA', '*ND', '*BL', ''], '14.54')
incoming_df['Field SBP'] = incoming_df['Field SBP'].replace(['*NA', '*ND', '*BL', ''], '119')
incoming_df['Field HR'] = incoming_df['Field HR'].replace(['*NA', '*ND', '*BL', ''], '110')
incoming_df['Field RR'] = incoming_df['Field RR'].replace(['*NA', '*ND', '*BL', ''], '21')
incoming_df['MV Speed'] = incoming_df['MV Speed'].replace(['*NA', '*ND', '*BL', ''], '0')

fields = ['Age in Years', 'Gender', 'MV Speed', 'RTS', 'Field GCS', 'Field SBP', 'Field HR', 'Field RR', 'Intubated',
          'Resp Assistance']
fields1 = ['Levels']

df = pd.read_csv(filepath_or_buffer='MotorVehicle.csv', usecols=fields)
Y = pd.read_csv(filepath_or_buffer='MotorVehicle.csv', usecols=fields1)

df['RTS'] = df['RTS'].replace(['*NA', '*ND', '*BL', ''], '7.65')
df['Field GCS'] = df['Field GCS'].replace(['*NA', '*ND', '*BL', ''], '14.54')
df['Field SBP'] = df['Field SBP'].replace(['*NA', '*ND', '*BL', ''], '119')
df['Field HR'] = df['Field HR'].replace(['*NA', '*ND', '*BL', ''], '110')
df['Field RR'] = df['Field RR'].replace(['*NA', '*ND', '*BL', ''], '21')
df['MV Speed'] = df['MV Speed'].replace(['*NA', '*ND', '*BL', ''], '0')


def logistic(x, y):
    x_train, y_train, X_test = x, y, incoming_df
    logreg = LogisticRegression(penalty='l2', class_weight='balanced')
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(X_test)
    print "The predicted trauma level is: " + str(y_pred)


logistic(df, Y)
