import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from IntegratedCode import incoming_df

incoming_df = incoming_df[['Age in Years', 'Gender', 'Field SBP', 'Field HR', 'Field RR', 'RTS', 'Field GCS',
                           'Resp Assistance']]

le = LabelEncoder()
incoming_df['Gender'] = le.fit_transform(incoming_df['Gender'].values)

incoming_df['RTS'] = incoming_df['RTS'].replace(['*NA', '*ND', '*BL', ''], '7.65')
incoming_df['Field GCS'] = incoming_df['Field GCS'].replace(['*NA', '*ND', '*BL', ''], '14.54')
incoming_df['Field SBP'] = incoming_df['Field SBP'].replace(['*NA', '*ND', '*BL', ''], '119')
incoming_df['Field HR'] = incoming_df['Field HR'].replace(['*NA', '*ND', '*BL', ''], '110 ')
incoming_df['Field RR'] = incoming_df['Field RR'].replace(['*NA', '*ND', '*BL', ''], '21')

result = pd.read_csv('Others.csv')

y = result['Levels']
X = result.drop('Levels', 1)

X_train = X
Y_train = y
X_test = incoming_df

classification_pipeline = Pipeline([('StandardScalar', StandardScaler()), ('pca', PCA(n_components=7)),
                                    ('classifier', DecisionTreeClassifier(random_state=1))])

classification_pipeline.fit(X_train, Y_train)

y_pred = classification_pipeline.predict(X_test)
print "The predicted trauma level is: " + str(y_pred)
