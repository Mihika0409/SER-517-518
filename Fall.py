import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from IntegratedCode import incoming_df

incoming_df = incoming_df[
    ['Age in Years', 'Gender', 'Fall Height', 'Field SBP', 'Field HR', 'Field RR', 'RTS', 'Field GCS']]
le = LabelEncoder()
incoming_df['Gender'] = le.fit_transform(incoming_df['Gender'].values)

incoming_df['RTS'] = incoming_df['RTS'].replace(['*NA', '*ND', '*BL', ''], '7.65')
incoming_df['Field GCS'] = incoming_df['Field GCS'].replace(['*NA', '*ND', '*BL', ''], '14.54')
incoming_df['Field SBP'] = incoming_df['Field SBP'].replace(['*NA', '*ND', '*BL', ''], '119')
incoming_df['Field HR'] = incoming_df['Field HR'].replace(['*NA', '*ND', '*BL', ''], '110 ')
incoming_df['Field RR'] = incoming_df['Field RR'].replace(['*NA', '*ND', '*BL', ''], '21')
incoming_df['Fall Height'] = incoming_df['Fall Height'].replace(['*NA', '*ND', '*BL', ''], '0')

df = pd.read_csv("Fall.csv")
df = df[['Age in Years', 'Gender', 'Fall Height', 'Field SBP', 'Field HR', 'Field RR', 'RTS', 'Field GCS', 'Levels']]

features = ['Age in Years', 'Gender', 'Fall Height', 'Field SBP', 'Field HR', 'Field RR', 'RTS', 'Field GCS', 'Levels']
for x in features:
    df = df[pd.notnull(df[x])]
    df = df[pd.notnull(df[x])].replace(['*NA', '*ND', '*BL'], value=['0', '0', '0'])
df = df.loc[df['Levels'].isin(['1', '2'])]

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'].values)

df['RTS'] = df['RTS'].replace(['*NA', '*ND', '*BL', ''], '7.65')
df['Field GCS'] = df['Field GCS'].replace(['*NA', '*ND', '*BL', ''], '14.54')
df['Field SBP'] = df['Field SBP'].replace(['*NA', '*ND', '*BL', ''], '119')
df['Field HR'] = df['Field HR'].replace(['*NA', '*ND', '*BL', ''], '110 ')
df['Field RR'] = df['Field RR'].replace(['*NA', '*ND', '*BL', ''], '21')
df['Fall Height'] = df['Fall Height'].replace(['*NA', '*ND', '*BL', ''], '0')

X = df.iloc[:, :-1].values
y = df['Levels'].values

X_train, X_test, y_train = X, incoming_df, y

dt = tree.DecisionTreeClassifier()
dt = dt.fit(X_train, y_train)

# decision trees with the gini index
clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)
y_pred_train = clf_gini.predict(X_train)

clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5,
                                     class_weight="balanced")
clf_entropy.fit(X_train, y_train)

y_pred_en = clf_entropy.predict(X_test)
y_pred_train_en = clf_entropy.predict(X_train)

print "The predicted trauma level through gini is: " + str(y_pred)
print "The predicted trauma level through entropy is: " + str(y_pred_en)
