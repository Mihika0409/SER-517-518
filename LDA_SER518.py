import pandas as pd
from sklearn.preprocessing import LabelEncoder
import math
import matplotlib.pyplot as plot
import numpy as np

import pandas as pd

from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

from sklearn.linear_model import LogisticRegression as logisticregression

from sklearn.metrics import confusion_matrix


df = pd.read_csv('refined_data_todd.csv', header = None, error_bad_lines=False)
df.columns = ['T1',	'ED/Hosp Arrival Date',	'Age in Years',	'Gender',	'Levels',	'ICD-10 E-code',	'Trauma Type',	'Report of physical abuse',	'Injury Comments',	'Airbag Deployment',	'Patient Position in Vehicle',
              'Safet Equipment Issues',	'Child Restraint',	'MV Speed',	'Fall Height',	'Transport Type',	'Transport Mode',	'Field SBP',	'Field HR',	'Field Shock Index',	'Field RR',	'Resp Assistance',
              'RTS',	'Field GCS',	'Arrived From',	'ED LOS (mins)',	'Dispositon from  ED',	'ED SBP',	'ED HR',	'ED RR',
              'ED GCS',	'Total Vent Days',	'Total Days in ICU',	'Admission Hosp LOS (days)',	'Total LOS (ED+Admit)',	'Received blood within 4 hrs',	'Severe Brain Injury',	'Time to 1st OR Visit (mins.)',

              'Final Outcome-Dead or Alive',	'Discharge Disposition',	'Injury Severity Score',	'AIS 2005']
#print df.head()

df.drop(['T1','ED/Hosp Arrival Date','ICD-10 E-code','Injury Comments', 'Patient Position in Vehicle','Safet Equipment Issues','Child Restraint','Transport Type',	'Transport Mode','Resp Assistance','Arrived From','Dispositon from  ED','Received blood within 4 hrs','Severe Brain Injury',	'Time to 1st OR Visit (mins.)','Discharge Disposition'], axis=1, inplace = True)

def column_NullHandler(columnlist):
    newlist = []
    for i in columnlist:
        x = float(i)
        if not math.isnan(x):
            newlist.append(x)
        else:
            newlist.append(0.0000000000000000001)
    return newlist


#LABEL ENCODING
gender_column = df.iloc[:,1].values

le = LabelEncoder()

gender_column = le.fit_transform(gender_column)

genderlist = []
for i in gender_column:
    genderlist.append(i)

df['Gender_Numerical'] = genderlist

df.drop(['Gender'], axis = 1, inplace= True)



#LABEL ENCODING
AgeInYears = df.iloc[:,0].values

AgeInYears_list = column_NullHandler(AgeInYears)
df['Age_Numerical'] = AgeInYears_list

df.drop(['Age in Years'], axis = 1, inplace= True)

#print df.head()




#LABEL ENCODING
Trauma_Type = df.iloc[:,1].values

Trauma_Type = le.fit_transform(Trauma_Type)
trauma_list = []
for i in Trauma_Type:
    trauma_list.append(i)

df['Trauma_Type_Numerical'] = trauma_list

df.drop(['Trauma Type'], axis = 1, inplace= True)

print df.head()



X = df.iloc[:,0:41].values

Y = df.iloc[:,4].values

X_training, X_testing , Y_training, Y_testing =  train_test_split(X,Y, test_size=0.2, random_state=0)



standardscaler = StandardScaler()

X_training = standardscaler.fit_transform(X_training)

X_testing = standardscaler.fit_transform(X_testing)

LinearDiscriminantAnalysis = lda(n_components=8)

X_training = LinearDiscriminantAnalysis.fit_transform(X_training, Y_training)

X_testing = LinearDiscriminantAnalysis.fit_transform(X_testing)

classifier =  logisticregression(random_state=0)
classifier.fit(X_training, Y_training)


Y_Predicted = classifier.predict(X_testing)


confusionMatrix = confusion_matrix(Y_testing, Y_Predicted)





X_set, Y_set = X_training, Y_training

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plot.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('red', 'blue', 'green')))

plot.xlim(X1.min(), X1.max())

plot.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plot.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plot.title('Logistic Regression (Training set)')

plot.xlabel('LD1')

plot.ylabel('LD2')

plot.legend()

plot.show()










